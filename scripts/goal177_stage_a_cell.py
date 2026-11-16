#!/usr/bin/env python3
"""Run one Goal177 Stage A evaluation cell and audit its action interface.

The worker uses the same evaluator as Goal175, but wraps the built policy so the
observations actually sent to the actor can be probed after the closed-loop
episode.  It writes:

* ``summary.json``: evaluator metrics and per-seed records;
* ``trace.npz``: closed-loop trace;
* ``interface.json``: raw/applied/total residual Jacobians and applied shares;
* ``cell.json``: audit record used by the Stage A launcher/auditor.

No training, reward, termination, success, radius or held-out access is done.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import pc_path_feature_scales_for_dim
from auv_lowlevel.cli.evaluate_pc_ppo import runner as eval_runner

SCHEMA_VERSION = "goal177_stage_a_cell.v1"
PATH_CROSS = 0
PATH_VERTICAL = 1
PATH_HEADING = 2
PATH_PITCH = 3
PATH_PROGRESS = 4
PATH_REMAINING = 5
PATH_ENDPOINT_PROXIMITY = 8
STATE_Q = 7
STATE_W = 11
STATE_U = 9
COMMAND_U = 2
DIRECTIONS = {
    "cross": ("path_cross", "path_heading"),
    "vertical": ("path_vertical", "path_pitch", "state_q", "state_w"),
    "forward": (
        "path_progress",
        "path_remaining",
        "path_endpoint_proximity",
        "command_u",
        "state_u",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--stage", default="A")
    parser.add_argument("--level", required=True)
    parser.add_argument("--radius", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--action-mode", choices=("policy", "frozen-base", "alos"), required=True)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--actuator-residual-scale", type=float, default=None)
    parser.add_argument("--actuator-residual-channel-gains", default=None)
    parser.add_argument("--capture-stride", type=int, default=10)
    parser.add_argument("--max-jacobian-samples", type=int, default=128)
    parser.add_argument("--allow-sensor-override", action="store_true")
    parser.add_argument("--skip-jacobian", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization(metadata: dict[str, Any]) -> PCNormalization:
    raw = metadata.get("normalization", {})
    if not isinstance(raw, dict):
        raise ValueError("checkpoint metadata has no normalization object")
    return PCNormalization(
        position_scale_m=float(raw["position_scale_m"]),
        attitude_scale_rad=float(raw["attitude_scale_rad"]),
        angular_rate_scale_rad_s=float(raw["angular_rate_scale_rad_s"]),
        surge_speed_scale_mps=float(raw["surge_speed_scale_mps"]),
        sway_heave_speed_scale_mps=float(raw["sway_heave_speed_scale_mps"]),
        current_scale_mps=float(raw["current_scale_mps"]),
    )


def _policy_outputs(policy: Any, observation: Tensor) -> dict[str, Tensor]:
    return {
        "raw_residual": policy._actor_residual_mean(observation),
        "applied_residual": policy.residual_action(observation),
        "total_action": policy.act_inference(observation),
    }


def _sample_rows(rows: np.ndarray, *, max_samples: int) -> np.ndarray:
    if rows.shape[0] <= max_samples:
        return rows
    indices = np.linspace(0, rows.shape[0] - 1, max_samples, dtype=np.int64)
    return rows[indices]


def _interface(
    *,
    policy: Any,
    observations: np.ndarray,
    metadata: dict[str, Any],
    max_samples: int,
) -> dict[str, Any]:
    if observations.shape[0] == 0:
        raise ValueError("no observations were captured for the interface audit")
    sampled = _sample_rows(observations, max_samples=max_samples)
    policy_dtype = next(policy.parameters()).dtype
    obs = torch.as_tensor(sampled, device=next(policy.parameters()).device, dtype=policy_dtype)
    state_dim = policy.actor_layout.state_dim
    path_dim = policy.actor_layout.path_dim
    if path_dim < 12:
        raise ValueError("Stage A requires the 12-D path block")
    normalization = _normalization(metadata)
    path_scales = pc_path_feature_scales_for_dim(path_dim)
    command_start = state_dim + path_dim + policy.actor_layout.current_dim

    perturbations: dict[str, tuple[int, float, float]] = {
        "path_cross": (state_dim + PATH_CROSS, 0.10 / path_scales[PATH_CROSS], 0.10),
        "path_vertical": (
            state_dim + PATH_VERTICAL,
            0.10 / path_scales[PATH_VERTICAL],
            0.10,
        ),
        "path_heading": (
            state_dim + PATH_HEADING,
            0.05 / path_scales[PATH_HEADING],
            0.05,
        ),
        "path_pitch": (state_dim + PATH_PITCH, 0.05 / path_scales[PATH_PITCH], 0.05),
        "path_progress": (
            state_dim + PATH_PROGRESS,
            0.02 / path_scales[PATH_PROGRESS],
            0.02,
        ),
        "path_remaining": (
            state_dim + PATH_REMAINING,
            0.02 / path_scales[PATH_REMAINING],
            0.02,
        ),
        "path_endpoint_proximity": (
            state_dim + PATH_ENDPOINT_PROXIMITY,
            0.10 / path_scales[PATH_ENDPOINT_PROXIMITY],
            0.10,
        ),
        "command_u": (
            command_start + COMMAND_U,
            0.05 / normalization.command_scales[COMMAND_U],
            0.05,
        ),
        "state_q": (STATE_Q, 0.05 / normalization.angular_rate_scale_rad_s, 0.05),
        "state_w": (STATE_W, 0.05 / normalization.sway_heave_speed_scale_mps, 0.05),
        "state_u": (STATE_U, 0.05 / normalization.surge_speed_scale_mps, 0.05),
    }
    action_full_scale = tuple(float(value) for value in metadata["action_full_scale"])
    result: dict[str, Any] = {
        "sample_count": int(obs.shape[0]),
        "observed_rows": int(observations.shape[0]),
        "inputs": {},
        "directions": {},
        "action_full_scale": list(action_full_scale),
    }
    with torch.inference_mode():
        zero = _policy_outputs(policy, obs)
        for name, (column, normalized_delta, physical_delta) in perturbations.items():
            plus = obs.clone()
            minus = obs.clone()
            plus[:, column] += normalized_delta
            minus[:, column] -= normalized_delta
            plus_outputs = _policy_outputs(policy, plus)
            minus_outputs = _policy_outputs(policy, minus)
            entry: dict[str, Any] = {
                "column": int(column),
                "normalized_delta": float(normalized_delta),
                "physical_delta": float(physical_delta),
                "channels": {},
            }
            for output_name in ("raw_residual", "applied_residual", "total_action"):
                derivative = (plus_outputs[output_name] - minus_outputs[output_name]) / (
                    2.0 * physical_delta
                )
                derivative_np = derivative.detach().cpu().numpy()
                zero_np = zero[output_name].detach().cpu().numpy()
                channel_rows: dict[str, Any] = {}
                for channel, channel_name in enumerate(("rudder", "stern", "rpm")):
                    values = derivative_np[:, channel]
                    positive = float(np.mean(values > 0.0))
                    negative = float(np.mean(values < 0.0))
                    channel_rows[channel_name] = {
                        "median_abs_derivative": float(np.median(np.abs(values))),
                        "median_signed_derivative": float(np.median(values)),
                        "sign_consistency": float(max(positive, negative)),
                        "effect_at_perturbation_normalized": float(
                            np.median(np.abs(values)) * physical_delta
                        ),
                        "effect_at_perturbation_physical": float(
                            np.median(np.abs(values)) * physical_delta * action_full_scale[channel]
                        ),
                        "mean_abs_zero_action": float(np.mean(np.abs(zero_np[:, channel]))),
                    }
                entry["channels"][output_name] = channel_rows
            for output_name in ("raw_residual", "applied_residual"):
                for channel_name in ("rudder", "stern", "rpm"):
                    numerator = entry["channels"][output_name][channel_name][
                        "median_abs_derivative"
                    ]
                    denominator = entry["channels"]["total_action"][channel_name][
                        "median_abs_derivative"
                    ]
                    entry["channels"][output_name][channel_name]["share_of_total_median_abs"] = (
                        None if denominator <= 1e-15 else numerator / denominator
                    )
            result["inputs"][name] = entry

    for direction, input_names in DIRECTIONS.items():
        channel_summary: dict[str, Any] = {}
        for channel_name in ("rudder", "stern", "rpm"):
            candidates = []
            for input_name in input_names:
                applied = result["inputs"][input_name]["channels"]["applied_residual"][
                    channel_name
                ]
                candidates.append(
                    {
                        "input": input_name,
                        "applied_share": applied["share_of_total_median_abs"],
                        "sign_consistency": applied["sign_consistency"],
                        "applied_median_abs_derivative": applied["median_abs_derivative"],
                    }
                )
            finite = [row for row in candidates if row["applied_share"] is not None]
            best = max(finite, key=lambda row: float(row["applied_share"])) if finite else None
            channel_summary[channel_name] = {
                "inputs": candidates,
                "best_input": None if best is None else best["input"],
                "applied_share": None if best is None else best["applied_share"],
                "sign_consistency_at_best": None if best is None else best["sign_consistency"],
            }
        best_channel = max(
            channel_summary,
            key=lambda name: (
                -1.0
                if channel_summary[name]["applied_share"] is None
                else float(channel_summary[name]["applied_share"])
            ),
        )
        result["directions"][direction] = {
            "channels": channel_summary,
            "best_channel": best_channel,
            "applied_share": channel_summary[best_channel]["applied_share"],
            "sign_consistency": channel_summary[best_channel]["sign_consistency_at_best"],
        }

    if hasattr(policy, "std"):
        action_std = float(policy.std.detach().mean().cpu().item())
    elif hasattr(policy, "log_std"):
        action_std = float(torch.exp(policy.log_std.detach()).mean().cpu().item())
    else:
        action_std = math.nan
    result["action_std_mean"] = action_std
    return result


def _route_subset(route: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "route_e3d_t_rmse_m",
        "route_cross_abs_p90_m",
        "route_vert_abs_p90_m",
        "route_preterminal_e3d_rmse_m",
        "route_preterminal_cross_abs_p90_m",
        "route_preterminal_vert_abs_p90_m",
        "keypoint_mean_e3d_m",
        "keypoint_e3d_p90_m",
        "completion_fraction",
        "progress_efficiency",
        "final_progress_m",
        "speed_mean_mps",
        "speed_p95_mps",
        "speed_max_mps",
        "success_count",
        "timeout_count",
        "safety_count",
        "overspeed_steps",
    )
    return {key: route.get(key) for key in keys}


def _forward_subset(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "path_completion_ratio_mean",
        "final_path_progress_m_mean",
        "pre_goal_3d_rmse_m",
        "miss_distance_at_plane_m",
        "terminal_longitudinal_rmse_m",
        "post_goal_overshoot_m",
        "ground_route_speed_ratio_mean",
        "success_rate",
        "timeout_rate",
        "safety_termination_rate",
        "goal_plane_miss_rate",
    )
    return {key: summary.get(key) for key in keys}


def _stable_stability(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "residual_action_abs_mean",
        "residual_action_abs_p95",
        "residual_to_base_l2_ratio_mean",
        "actuator_residual_abs_mean",
        "actuator_residual_abs_p95",
        "actuator_residual_scale_clip_rate",
        "actuator_residual_magnitude_clip_rate",
        "actuator_residual_rate_clip_rate",
        "ground_speed_max_mps",
        "ground_speed_above_max_rate",
        "safety_termination_rate",
        "timeout_rate",
    )
    return {key: summary.get(key) for key in keys}


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in ("summary.json", "interface.json", "cell.json", "trace.npz"):
            (output / name).unlink(missing_ok=True)
    elif any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    repo_root = args.repo_root.resolve()
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from analyze_goal170_route_transfer import analyze_run  # noqa: PLC0415

    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    if not seeds:
        raise ValueError("seed count must be positive")
    if any(seed >= 100 for seed in seeds):
        raise ValueError(f"held-out seeds are forbidden in Stage A: {seeds}")
    gains = None
    if args.actuator_residual_channel_gains:
        values = [float(value) for value in args.actuator_residual_channel_gains.split(",")]
        if len(values) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("channel gains must be three positive finite values")
        gains = tuple(values)
    trace_path = output / "trace.npz"
    captured: list[np.ndarray] = []
    state: dict[str, Any] = {"policy": None, "metadata": None, "original_inference": None}
    original_builder = eval_runner._build_policy
    capture_count = 0
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)

    def wrapped_builder(*builder_args: Any, **builder_kwargs: Any) -> Any:
        nonlocal capture_count
        policy, metadata = original_builder(*builder_args, **builder_kwargs)
        state["policy"] = policy
        state["metadata"] = metadata
        if args.action_mode == "policy" and not args.skip_jacobian:
            original_inference = policy.act_inference

            def hooked_inference(observation: Tensor) -> Tensor:
                nonlocal capture_count
                if capture_count % max(args.capture_stride, 1) == 0:
                    captured.append(observation.detach().cpu().numpy().copy())
                capture_count += 1
                return original_inference(observation)

            policy.act_inference = hooked_inference
            state["original_inference"] = original_inference
        return policy, metadata

    eval_runner._build_policy = wrapped_builder
    try:
        summary, metadata, episodes, contract = eval_runner._run(
            training_config_path=args.training_config,
            checkpoint_path=args.checkpoint,
            environment_config_path=args.environment_config,
            seeds=seeds,
            device=device,
            dtype=dtype,
            action_mode=args.action_mode,
            allow_sensor_override=bool(args.allow_sensor_override),
            allow_alos_current_mode_override=True,
            disable_curriculum_direct_command=True,
            success_frame="both",
            trace_output=trace_path,
            actuator_residual_scale_override=args.actuator_residual_scale,
            actuator_residual_channel_gains=gains,
        )
    finally:
        if state.get("policy") is not None and state.get("original_inference") is not None:
            state["policy"].act_inference = state["original_inference"]
        eval_runner._build_policy = original_builder

    summary_payload = {
        "schema_version": "goal177_stage_a_summary.v1",
        "goal_id": "Goal177",
        "stage": args.stage,
        "arm": args.arm,
        "level": args.level,
        "radius": args.radius,
        "scenario": args.scenario,
        "seeds": list(seeds),
        "metrics": summary,
        "episodes": episodes,
        "contract": contract,
    }
    (output / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    route = analyze_run(output, default_length_m=35.0)

    interface: dict[str, Any] | None = None
    interface_error: str | None = None
    if args.action_mode == "policy" and not args.skip_jacobian:
        try:
            observations = np.concatenate(captured, axis=0) if captured else np.zeros((0, 1))
            interface = _interface(
                policy=state["policy"],
                observations=observations,
                metadata=metadata,
                max_samples=args.max_jacobian_samples,
            )
            residual_abs = summary.get("residual_action_abs_mean")
            if residual_abs is not None and interface.get("action_std_mean"):
                interface["residual_to_action_std"] = float(residual_abs) / float(
                    interface["action_std_mean"]
                )
            else:
                interface["residual_to_action_std"] = None
            (output / "interface.json").write_text(
                json.dumps(interface, indent=2), encoding="utf-8"
            )
        except Exception as error:  # noqa: BLE001
            interface_error = f"{type(error).__name__}: {error}"
            interface = {"status": "error", "error": interface_error}

    cell = {
        "schema_version": SCHEMA_VERSION,
        "goal_id": "Goal177",
        "stage": args.stage,
        "arm": args.arm,
        "level": args.level,
        "radius": args.radius,
        "scenario": args.scenario,
        "seeds": list(seeds),
        "action_mode": args.action_mode,
        "actuator_residual_scale": args.actuator_residual_scale,
        "actuator_residual_channel_gains": None if gains is None else list(gains),
        "training_config": str(args.training_config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "environment_config": str(args.environment_config.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "environment_sha256": _sha256(args.environment_config.resolve()),
        "trace": str(trace_path),
        "summary": str(output / "summary.json"),
        "route": _route_subset(route),
        "forward": _forward_subset(summary),
        "stability": _stable_stability(summary),
        "interface": interface,
        "interface_error": interface_error,
        "status": "ok" if interface_error is None else "interface_error",
    }
    (output / "cell.json").write_text(json.dumps(cell, indent=2), encoding="utf-8")
    print(json.dumps({"arm": args.arm, "scenario": args.scenario, "status": cell["status"]}))
    return 0 if cell["status"] == "ok" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
