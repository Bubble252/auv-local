"""Diagnose the composition of PC BC path-tracking error without changing metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.geometry import ground_velocity_body_for_contract
from auv_lowlevel.batch.pc import PCTerminalKind
from auv_lowlevel.batch.pc_bc import PCBCPolicy
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config

TRACE_FIELDS = (
    "position_north_m",
    "position_east_m",
    "position_down_m",
    "hint_progress_m",
    "hint_segment_index",
    "hint_horizontal_error_m",
    "hint_vertical_error_m",
    "hint_longitudinal_error_m",
    "hint_error_3d_m",
    "global_progress_m",
    "global_segment_index",
    "global_horizontal_error_m",
    "global_vertical_error_m",
    "global_longitudinal_error_m",
    "global_error_3d_m",
    "hint_minus_global_error_m",
    "hint_minus_global_progress_m",
    "segment_disagreement",
    "completion_ratio",
    "goal_distance_m",
    "goal_tangent_offset_m",
    "ground_surge_mps",
    "commanded_speed_mps",
    "speed_error_mps",
    "heading_error_rad",
    "pitch_error_rad",
    "roll_abs_rad",
    "rudder_rad",
    "stern_plane_rad",
    "propeller_rpm",
    "rate_clipped",
    "material_saturated",
    "hold_fraction",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit path-error decomposition for a PC BC checkpoint"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--path-dim", type=int, choices=(0, 8), default=8)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection_components(projection: Any, position: torch.Tensor) -> dict[str, torch.Tensor]:
    offset = position - projection.point_ned_m
    longitudinal = torch.sum(offset * projection.tangent_ned, dim=1)
    return {
        "horizontal_error_m": projection.horizontal_error_m,
        "vertical_error_m": projection.vertical_error_m,
        "longitudinal_error_m": longitudinal,
        "error_3d_m": projection.error_3d_m,
    }


def _run_policy(
    *,
    config_path: Path,
    checkpoint_path: Path,
    seeds: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    path_dim: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if path_dim <= 0:
        raise ValueError("the diagnostic requires a path-conditioned policy")
    config = load_config(config_path)
    if config.arm != "PC":
        raise ValueError("evaluation requires a PC config")
    if str(config.pc_stage) not in {"C2", "C3", "C4", "C5", "C6"}:
        raise ValueError("the diagnostic requires a path-enabled PC stage")

    env = BatchedPCLowLevelEnv(
        config, num_envs=len(seeds), device=device, dtype=dtype, path_dim=path_dim
    )
    env.reset(seeds)
    if env.eta is None:
        raise RuntimeError("environment reset did not produce eta")
    hint = env.path.project(env.eta[:, :3]).segment_index
    student = PCBCPolicy(checkpoint_path, device=device)
    if student.model.path_dim != path_dim:
        raise ValueError(
            "checkpoint path dimension does not match requested path_dim: "
            f"checkpoint={student.model.path_dim}, requested={path_dim}"
        )

    max_steps = int(config.plant.max_episode_steps)
    traces = {
        name: np.full((len(seeds), max_steps), np.nan, dtype=np.float64) for name in TRACE_FIELDS
    }
    active_steps = np.zeros(len(seeds), dtype=np.int64)
    done = np.zeros(len(seeds), dtype=bool)
    success = np.zeros(len(seeds), dtype=bool)
    timeout = np.zeros(len(seeds), dtype=bool)
    safety = np.zeros(len(seeds), dtype=bool)
    terminal_reason = np.full(len(seeds), "incomplete", dtype=object)

    path = env.path
    total_length = path.total_length_m_for(len(seeds))
    goal = path.goal_ned_m_for(len(seeds))
    before_goal = path.point_at_s(torch.clamp(total_length - 0.01, min=0.0))
    end_direction = goal - before_goal
    end_tangent = end_direction / torch.linalg.norm(end_direction, dim=1, keepdim=True)

    for step_index in range(max_steps):
        active = ~done
        if not bool(active.any()):
            break
        with torch.inference_mode():
            observation = env.observation
            command = torch.stack(
                [
                    observation.command.heading_rad,
                    observation.command.pitch_rad,
                    observation.command.speed_mps,
                ],
                dim=1,
            )
            actor_command = command if student.model.actor_command_dim else None
            actor_path = observation.actor_path
            if actor_path is None:
                raise RuntimeError("path-conditioned actor received no path observation")
            actor_current = (
                observation.actor_current_body
                if student.model.actor_current_encoder is not None
                else None
            )
            action = student.act_inference(
                observation.actor_state,
                actor_command,
                actor_path,
                actor_current,
            )
        result = env.step(action, auto_reset=False)
        active = ~done

        eta = env.eta
        nu = env.nu
        if eta is None or nu is None or env.current_ned_mps is None:
            raise RuntimeError("environment state is unavailable after step")
        position = eta[:, :3]
        hint_projection = path.project(
            position,
            segment_hint=hint,
            search_radius=env.reset_projection_search_radius,
        )
        hint = hint_projection.segment_index
        global_projection = path.project(position)
        hint_components = _projection_components(hint_projection, position)
        global_components = _projection_components(global_projection, position)

        command_physical = result.info["command_physical"]
        ground_velocity = ground_velocity_body_for_contract(
            eta,
            nu,
            env.current_ned_mps,
            config.plant.velocity_contract,
        )
        heading_error = command_physical[:, 0] - eta[:, 5]
        heading_error = (heading_error + math.pi) % (2.0 * math.pi) - math.pi
        pitch_error = command_physical[:, 1] - eta[:, 4]
        speed_error = command_physical[:, 2] - ground_velocity[:, 0]
        goal_distance = torch.linalg.norm(position - goal, dim=1)
        goal_tangent_offset = torch.sum((position - goal) * end_tangent, dim=1)
        actual_actuator = result.info["actual_actuator"]

        values = {
            "position_north_m": position[:, 0],
            "position_east_m": position[:, 1],
            "position_down_m": position[:, 2],
            "hint_progress_m": hint_projection.progress_m,
            "hint_segment_index": hint_projection.segment_index,
            "hint_horizontal_error_m": hint_components["horizontal_error_m"],
            "hint_vertical_error_m": hint_components["vertical_error_m"],
            "hint_longitudinal_error_m": hint_components["longitudinal_error_m"],
            "hint_error_3d_m": hint_components["error_3d_m"],
            "global_progress_m": global_projection.progress_m,
            "global_segment_index": global_projection.segment_index,
            "global_horizontal_error_m": global_components["horizontal_error_m"],
            "global_vertical_error_m": global_components["vertical_error_m"],
            "global_longitudinal_error_m": global_components["longitudinal_error_m"],
            "global_error_3d_m": global_components["error_3d_m"],
            "hint_minus_global_error_m": (
                hint_components["error_3d_m"] - global_components["error_3d_m"]
            ),
            "hint_minus_global_progress_m": (
                hint_projection.progress_m - global_projection.progress_m
            ),
            "segment_disagreement": torch.abs(
                hint_projection.segment_index.to(torch.float64)
                - global_projection.segment_index.to(torch.float64)
            ),
            "completion_ratio": hint_projection.progress_m / path.total_length_m_tensor,
            "goal_distance_m": goal_distance,
            "goal_tangent_offset_m": goal_tangent_offset,
            "ground_surge_mps": ground_velocity[:, 0],
            "commanded_speed_mps": command_physical[:, 2],
            "speed_error_mps": speed_error,
            "heading_error_rad": heading_error,
            "pitch_error_rad": pitch_error,
            "roll_abs_rad": torch.abs(eta[:, 3]),
            "rudder_rad": actual_actuator[:, 0],
            "stern_plane_rad": actual_actuator[:, 1],
            "propeller_rpm": actual_actuator[:, 2],
            "rate_clipped": result.info["rate_clipped"].any(dim=1).to(torch.float64),
            "material_saturated": result.info["material_saturated"].to(torch.float64),
            "hold_fraction": result.info["hold_fraction"],
        }
        for name, value in values.items():
            traces[name][active, step_index] = (
                value.detach().cpu().numpy()[active].astype(np.float64, copy=False)
            )
        active_steps[active] += 1

        terminal = result.info["terminal_kind"].detach().cpu().numpy()
        done_now = result.done.detach().cpu().numpy() & active
        goal_now = done_now & (terminal == int(PCTerminalKind.GOAL))
        timeout_now = done_now & (terminal == int(PCTerminalKind.TIME_LIMIT))
        safety_now = done_now & ~goal_now & ~timeout_now
        success |= goal_now
        timeout |= timeout_now
        safety |= safety_now
        for index in np.flatnonzero(goal_now):
            terminal_reason[index] = "goal"
        for index in np.flatnonzero(timeout_now):
            terminal_reason[index] = "timeout"
        for index in np.flatnonzero(safety_now):
            terminal_reason[index] = "safety"
        done |= done_now

    metadata = {
        "seeds": list(seeds),
        "path_length_m": float(path.total_length_m),
        "max_episode_steps": max_steps,
        "active_steps": active_steps.tolist(),
        "success": success.tolist(),
        "timeout": timeout.tolist(),
        "safety": safety.tolist(),
        "terminal_reason": terminal_reason.tolist(),
        "hint_search_radius": int(env.reset_projection_search_radius),
    }
    return traces, metadata


def _mse_rmse(values: np.ndarray, steps: int) -> tuple[float, float]:
    if steps <= 0:
        return float("nan"), float("nan")
    active = values[:steps]
    finite = np.isfinite(active)
    if not finite.any():
        return float("nan"), float("nan")
    mse = float(np.mean(np.square(active[finite])))
    return mse, math.sqrt(mse)


def _per_seed_records(
    traces: dict[str, np.ndarray], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(metadata["seeds"]):
        steps = int(metadata["active_steps"][index])
        if steps <= 0:
            continue
        record: dict[str, Any] = {
            "seed": int(seed),
            "steps": steps,
            "success": bool(metadata["success"][index]),
            "timeout": bool(metadata["timeout"][index]),
            "safety": bool(metadata["safety"][index]),
            "terminal_reason": metadata["terminal_reason"][index],
        }
        for prefix in ("hint", "global"):
            mse_e, rmse_e = _mse_rmse(traces[f"{prefix}_error_3d_m"][index], steps)
            _, rmse_h = _mse_rmse(traces[f"{prefix}_horizontal_error_m"][index], steps)
            _, rmse_v = _mse_rmse(traces[f"{prefix}_vertical_error_m"][index], steps)
            _, rmse_l = _mse_rmse(traces[f"{prefix}_longitudinal_error_m"][index], steps)
            record[f"{prefix}_mse_3d_m2"] = mse_e
            record[f"{prefix}_rmse_3d_m"] = rmse_e
            record[f"{prefix}_rmse_horizontal_m"] = rmse_h
            record[f"{prefix}_rmse_vertical_m"] = rmse_v
            record[f"{prefix}_rmse_longitudinal_m"] = rmse_l

        for name in (
            "hint_horizontal_error_m",
            "hint_vertical_error_m",
            "hint_longitudinal_error_m",
            "goal_distance_m",
            "goal_tangent_offset_m",
            "speed_error_mps",
            "heading_error_rad",
            "pitch_error_rad",
            "roll_abs_rad",
            "rate_clipped",
            "material_saturated",
        ):
            values = traces[name][index, :steps]
            finite = values[np.isfinite(values)]
            if finite.size:
                record[f"{name}_mean"] = float(np.mean(finite))
                record[f"{name}_p95_abs"] = float(np.percentile(np.abs(finite), 95))
                record[f"{name}_max_abs"] = float(np.max(np.abs(finite)))

        hint_long = np.abs(traces["hint_longitudinal_error_m"][index, :steps])
        segment_disagreement = traces["segment_disagreement"][index, :steps]
        progress_disagreement = np.abs(traces["hint_minus_global_progress_m"][index, :steps])
        global_better = (
            traces["global_error_3d_m"][index, :steps]
            < traces["hint_error_3d_m"][index, :steps] - 1e-9
        )
        record.update(
            {
                "hint_endpoint_longitudinal_fraction": float(np.mean(hint_long > 0.05)),
                "segment_disagreement_rate": float(np.mean(segment_disagreement > 0)),
                "progress_disagreement_mean_m": float(np.mean(progress_disagreement)),
                "progress_disagreement_p95_m": float(np.percentile(progress_disagreement, 95)),
                "global_better_step_fraction": float(np.mean(global_better)),
                "final_completion_ratio": float(traces["completion_ratio"][index, steps - 1]),
                "final_goal_distance_m": float(traces["goal_distance_m"][index, steps - 1]),
                "final_goal_tangent_offset_m": float(
                    traces["goal_tangent_offset_m"][index, steps - 1]
                ),
            }
        )
        hint_mse = record["hint_mse_3d_m2"]
        if hint_mse > 0.0:
            record["hint_mse_share_horizontal"] = record["hint_rmse_horizontal_m"] ** 2 / hint_mse
            record["hint_mse_share_vertical"] = record["hint_rmse_vertical_m"] ** 2 / hint_mse
            record["hint_mse_share_longitudinal"] = (
                record["hint_rmse_longitudinal_m"] ** 2 / hint_mse
            )
        records.append(record)
    return records


def _mean_field(records: list[dict[str, Any]], name: str) -> float | None:
    values = [float(record[name]) for record in records if name in record]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else None


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    hint_mse = [float(record["hint_mse_3d_m2"]) for record in records]
    global_mse = [float(record["global_mse_3d_m2"]) for record in records]
    aggregate: dict[str, Any] = {
        "episode_count": len(records),
        "success_count": sum(bool(record["success"]) for record in records),
        "timeout_count": sum(bool(record["timeout"]) for record in records),
        "safety_count": sum(bool(record["safety"]) for record in records),
        "hint_rmse_3d_m": math.sqrt(float(np.mean(hint_mse))),
        "hint_rmse_horizontal_m": math.sqrt(
            float(np.mean([record["hint_rmse_horizontal_m"] ** 2 for record in records]))
        ),
        "hint_rmse_vertical_m": math.sqrt(
            float(np.mean([record["hint_rmse_vertical_m"] ** 2 for record in records]))
        ),
        "hint_rmse_longitudinal_m": math.sqrt(
            float(np.mean([record["hint_rmse_longitudinal_m"] ** 2 for record in records]))
        ),
        "global_rmse_3d_m": math.sqrt(float(np.mean(global_mse))),
        "global_rmse_horizontal_m": math.sqrt(
            float(np.mean([record["global_rmse_horizontal_m"] ** 2 for record in records]))
        ),
        "global_rmse_vertical_m": math.sqrt(
            float(np.mean([record["global_rmse_vertical_m"] ** 2 for record in records]))
        ),
        "global_rmse_longitudinal_m": math.sqrt(
            float(np.mean([record["global_rmse_longitudinal_m"] ** 2 for record in records]))
        ),
    }
    hint_total_mse = float(np.mean(hint_mse))
    if hint_total_mse > 0.0:
        aggregate["hint_mse_share_horizontal"] = (
            aggregate["hint_rmse_horizontal_m"] ** 2 / hint_total_mse
        )
        aggregate["hint_mse_share_vertical"] = (
            aggregate["hint_rmse_vertical_m"] ** 2 / hint_total_mse
        )
        aggregate["hint_mse_share_longitudinal"] = (
            aggregate["hint_rmse_longitudinal_m"] ** 2 / hint_total_mse
        )
    aggregate["global_rmse_delta_vs_hint_m"] = (
        aggregate["global_rmse_3d_m"] - aggregate["hint_rmse_3d_m"]
    )
    for name in (
        "segment_disagreement_rate",
        "progress_disagreement_mean_m",
        "progress_disagreement_p95_m",
        "global_better_step_fraction",
        "hint_endpoint_longitudinal_fraction",
        "goal_distance_m_mean",
        "goal_tangent_offset_m_mean",
        "speed_error_mps_mean",
        "heading_error_rad_mean",
        "pitch_error_rad_mean",
        "material_saturated_mean",
        "rate_clipped_mean",
    ):
        aggregate[name] = _mean_field(records, name)
    return aggregate


def _group_aggregate(records: list[dict[str, Any]], *, success: bool) -> dict[str, Any]:
    selected = [record for record in records if bool(record["success"]) is success]
    if not selected:
        return {"episode_count": 0}
    result = _aggregate(selected)
    result["episode_count"] = len(selected)
    return result


def _write_trace_csv(path: Path, traces: dict[str, np.ndarray], index: int, steps: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", *TRACE_FIELDS))
        for step in range(steps):
            writer.writerow((step, *(float(traces[name][index, step]) for name in TRACE_FIELDS)))


def _selected_trace_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for terminal in ("goal", "timeout"):
        group = [record for record in records if record["terminal_reason"] == terminal]
        group.sort(key=lambda record: float(record["hint_rmse_3d_m"]))
        if not group:
            continue
        indices = {0, len(group) // 2, len(group) - 1}
        for index in sorted(indices):
            record = group[index]
            selected.append(
                {
                    "seed": int(record["seed"]),
                    "terminal_reason": terminal,
                    "hint_rmse_3d_m": float(record["hint_rmse_3d_m"]),
                    "rank_in_terminal_group": index,
                }
            )
    return selected


def _summary_markdown(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"]
    success = summary["success_group"]
    timeout = summary["timeout_group"]
    lines = [
        "# PC BC path-error audit",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- config: `{summary['config_path']}`",
        f"- seeds: `{summary['seeds']}`",
        "",
        "## Hinted projection",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| 3D RMSE | {aggregate['hint_rmse_3d_m']:.6f} m |",
        f"| horizontal RMSE | {aggregate['hint_rmse_horizontal_m']:.6f} m |",
        f"| normal/vertical RMSE | {aggregate['hint_rmse_vertical_m']:.6f} m |",
        f"| longitudinal RMSE | {aggregate['hint_rmse_longitudinal_m']:.6f} m |",
        f"| horizontal MSE share | {aggregate['hint_mse_share_horizontal']:.4f} |",
        f"| normal MSE share | {aggregate['hint_mse_share_vertical']:.4f} |",
        f"| longitudinal MSE share | {aggregate['hint_mse_share_longitudinal']:.4f} |",
        f"| endpoint longitudinal fraction | {aggregate['hint_endpoint_longitudinal_fraction']:.4f} |",
        "",
        "## Projection comparison",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| global 3D RMSE | {aggregate['global_rmse_3d_m']:.6f} m |",
        f"| global minus hint 3D RMSE | {aggregate['global_rmse_delta_vs_hint_m']:.6f} m |",
        f"| segment disagreement rate | {aggregate['segment_disagreement_rate']:.4f} |",
        f"| progress disagreement mean | {aggregate['progress_disagreement_mean_m']:.6f} m |",
        f"| global-better step fraction | {aggregate['global_better_step_fraction']:.4f} |",
        "",
        "## Terminal groups",
        "",
        "| group | episodes | 3D RMSE | horizontal | normal | longitudinal | speed error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, group in (("success", success), ("timeout", timeout)):
        if group.get("episode_count", 0) == 0:
            lines.append(f"| {label} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {label} | {group['episode_count']} | "
            f"{group['hint_rmse_3d_m']:.6f} | "
            f"{group['hint_rmse_horizontal_m']:.6f} | "
            f"{group['hint_rmse_vertical_m']:.6f} | "
            f"{group['hint_rmse_longitudinal_m']:.6f} | "
            f"{(group.get('speed_error_mps_mean') or float('nan')):.6f} |"
        )
    lines.extend(
        [
            "",
            "Selected traces are stored under `selected_traces/`.",
            "The historical `path_*` metrics are left unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    traces, metadata = _run_policy(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        seeds=seeds,
        device=args.device,
        dtype=dtype,
        path_dim=args.path_dim,
    )
    records = _per_seed_records(traces, metadata)
    aggregate = _aggregate(records)
    success_group = _group_aggregate(records, success=True)
    timeout_group = _group_aggregate(records, success=False)
    summary = {
        "schema_version": "auv_pc_bc_path_error_audit.v1",
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "path_dim": args.path_dim,
        "seeds": list(seeds),
        "metadata": metadata,
        "aggregate": aggregate,
        "success_group": success_group,
        "timeout_group": timeout_group,
        "per_seed": records,
        "metric_definitions": {
            "hint_*": "same rolling-hint projection used by evaluate_pc_bc.py",
            "global_*": "diagnostic global-nearest projection over all path segments",
            "longitudinal_*": "position residual projected on the selected path tangent",
            "normal/vertical": "projection.vertical_error_m, normal to tangent and lateral axis",
        },
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    np.savez_compressed(output / "traces.npz", **traces)

    selected = _selected_trace_records(records)
    trace_dir = output / "selected_traces"
    trace_dir.mkdir(exist_ok=True)
    seed_to_index = {int(seed): index for index, seed in enumerate(seeds)}
    for record in selected:
        seed = int(record["seed"])
        index = seed_to_index[seed]
        steps = int(metadata["active_steps"][index])
        _write_trace_csv(trace_dir / f"seed_{seed}.csv", traces, index, steps)
    (trace_dir / "index.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": str(output / "summary.json"), "selected": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
