from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.checkpoint import validate_checkpoint_metadata
from auv_lowlevel.batch.env import BatchedLowLevelEnv
from auv_lowlevel.batch.observation import (
    OBSERVATION_SCHEMA_V1,
    OBSERVATION_SCHEMA_V2,
    ObservationSchema,
)
from auv_lowlevel.batch.reward import TerminalKind, terminal_kind_name
from auv_lowlevel.config import LowLevelConfig, load_config
from auv_lowlevel.ppo_config import load_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Policy(Protocol):
    def __call__(self, observation: torch.Tensor) -> torch.Tensor: ...


class _DeterministicActorPolicy:
    def __init__(
        self,
        checkpoint_path: Path,
        training_config_path: Path,
        *,
        observation_dim: int,
        device: torch.device,
    ) -> None:
        from rsl_rl.modules import ActorCritic

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint_metadata(
            checkpoint,
            expected_observation_schema=("v1" if observation_dim == 27 else "v2"),
            expected_observation_dim=observation_dim,
        )
        training = load_training_config(training_config_path)
        self.model = ActorCritic(
            observation_dim,
            observation_dim,
            3,
            **training.policy,
        ).to(device)
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint does not contain model_state_dict")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.device = device

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            policy_observation = observation.to(
                device=self.device,
                dtype=torch.float32,
            )
            return self.model.act_inference(policy_observation)


def _resolve(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _parse_seeds(text: str | None, default_start: int, default_count: int) -> tuple[int, ...]:
    if text is None:
        return tuple(range(default_start, default_start + default_count))
    seeds: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("seed range end must not precede start")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be non-empty, unique, and non-negative")
    return tuple(seeds)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    success_values = [bool(item["success"]) for item in episodes]
    successful_arrivals = [
        float(item["arrival_time_s"]) for item in episodes if item["arrival_time_s"] is not None
    ]
    terminations = sorted({str(item["termination"]) for item in episodes})

    def stats(key: str) -> dict[str, float]:
        values = [float(item[key]) for item in episodes]
        return {
            f"{key}_mean": float(np.mean(values)),
            f"{key}_std": float(np.std(values)),
            f"{key}_median": float(np.median(values)),
            f"{key}_p95": float(np.percentile(values, 95)),
        }

    result: dict[str, Any] = {
        "episode_count": len(episodes),
        "success_count": sum(success_values),
        "success_rate": float(np.mean(success_values)),
        "termination_counts": {
            kind: sum(item["termination"] == kind for item in episodes) for kind in terminations
        },
        "arrival_time_s_mean": _mean(successful_arrivals),
        "arrival_time_s_median": _percentile(successful_arrivals, 50.0),
    }
    for key in (
        "steps",
        "duration_s",
        "return_sum",
        "ppo_return_sum",
        "horizontal_rmse_m",
        "vertical_rmse_m",
        "three_d_rmse_m",
        "yaw_rmse_rad",
        "speed_error_rmse_mps",
        "saturation_rate",
        "material_saturation_rate",
        "constraint_violations",
        "command_smoothness_mean_sq",
    ):
        result.update(stats(key))
    return result


def _run_batch(
    config: LowLevelConfig,
    seeds: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    projection_search_radius: int,
    schema: ObservationSchema,
    policy: _Policy | None,
) -> list[dict[str, Any]]:
    env = BatchedLowLevelEnv(
        config,
        num_envs=len(seeds),
        device=device,
        dtype=dtype,
        schema=schema,
        validate_shadow=True,
        projection_search_radius=projection_search_radius,
    )
    observation = env.reset(list(seeds))
    count = len(seeds)
    active = torch.ones(count, dtype=torch.bool, device=device)
    steps = torch.zeros(count, dtype=torch.long, device=device)
    squared_horizontal = torch.zeros(count, dtype=torch.float64, device=device)
    squared_vertical = torch.zeros(count, dtype=torch.float64, device=device)
    squared_three_d = torch.zeros(count, dtype=torch.float64, device=device)
    squared_yaw = torch.zeros(count, dtype=torch.float64, device=device)
    squared_speed = torch.zeros(count, dtype=torch.float64, device=device)
    return_sum = torch.zeros(count, dtype=torch.float64, device=device)
    ppo_return_sum = torch.zeros(count, dtype=torch.float64, device=device)
    saturation_count = torch.zeros(count, dtype=torch.long, device=device)
    material_saturation_count = torch.zeros(count, dtype=torch.long, device=device)
    constraint_violations = torch.zeros(count, dtype=torch.long, device=device)
    command_smoothness = torch.zeros(count, dtype=torch.float64, device=device)
    terminal = torch.full((count,), int(TerminalKind.NONE), dtype=torch.long, device=device)
    final_progress = torch.zeros(count, dtype=torch.float64, device=device)
    final_position = torch.full((count, 3), torch.nan, dtype=torch.float64, device=device)
    previous_command = torch.zeros((count, 3), dtype=torch.float64, device=device)

    for _ in range(config.plant.max_episode_steps):
        if not bool(active.any()):
            break
        residual = (
            torch.zeros((count, 3), dtype=dtype, device=device)
            if policy is None
            else policy(observation).to(device=device, dtype=dtype)
        )
        result = env.step(residual, auto_reset=False)
        observation = result.observation
        projection = result.info["projection_after"]
        step_mask = active
        steps = steps + step_mask.to(torch.long)
        squared_horizontal = torch.where(
            step_mask,
            squared_horizontal + projection.horizontal_error_m.square(),
            squared_horizontal,
        )
        squared_vertical = torch.where(
            step_mask,
            squared_vertical + projection.vertical_error_m.square(),
            squared_vertical,
        )
        squared_three_d = torch.where(
            step_mask,
            squared_three_d + projection.error_3d_m.square(),
            squared_three_d,
        )
        yaw_error = result.info["yaw_error_rad"]
        speed_error = result.info["water_surge_mps"] - result.info["reference_speed_mps"]
        squared_yaw = torch.where(
            step_mask,
            squared_yaw + yaw_error.square(),
            squared_yaw,
        )
        squared_speed = torch.where(
            step_mask,
            squared_speed + speed_error.square(),
            squared_speed,
        )
        return_sum = torch.where(step_mask, return_sum + result.reward.total, return_sum)
        ppo_return_sum = torch.where(
            step_mask, ppo_return_sum + result.reward.ppo_scaled, ppo_return_sum
        )
        saturated = result.info["saturated"]
        material_saturated = result.info["material_saturated"]
        saturation_count = saturation_count + (step_mask & saturated).to(torch.long)
        material_saturation_count = material_saturation_count + (step_mask & material_saturated).to(
            torch.long
        )
        constraint_violations = constraint_violations + (
            step_mask.to(torch.long) * result.info["constraint_violation_count"]
        )
        limited_total = result.info["limited_total"]
        command_delta = limited_total - previous_command
        command_smoothness = torch.where(
            step_mask,
            command_smoothness + command_delta.square().sum(dim=1),
            command_smoothness,
        )
        previous_command = limited_total
        done = result.done & step_mask
        terminal = torch.where(done, result.info["terminal_kind"], terminal)
        final_progress = torch.where(done, projection.progress_m, final_progress)
        final_position = torch.where(done[:, None], result.info["position_ned_m"], final_position)
        active = active & ~done

    if bool(active.any()):
        raise RuntimeError("some environments did not terminate within max_episode_steps")
    goal = env.path.goal_ned_m.to(dtype=torch.float64)
    final_goal_distance = torch.linalg.norm(final_position - goal[None, :], dim=1)
    total_length = env.path.total_length_m
    goal_radius = config.plant.goal_radius_m

    episodes: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        step_count = int(steps[index].item())
        terminal_kind = int(terminal[index].item())
        success = terminal_kind == int(TerminalKind.GOAL)
        episodes.append(
            {
                "seed": seed,
                "steps": step_count,
                "duration_s": step_count * env.dt_s,
                "termination": terminal_kind_name(terminal_kind),
                "success": success,
                "arrival_time_s": step_count * env.dt_s if success else None,
                "return_sum": float(return_sum[index].item()),
                "ppo_return_sum": float(ppo_return_sum[index].item()),
                "horizontal_rmse_m": float(
                    torch.sqrt(squared_horizontal[index] / step_count).item()
                ),
                "vertical_rmse_m": float(torch.sqrt(squared_vertical[index] / step_count).item()),
                "three_d_rmse_m": float(torch.sqrt(squared_three_d[index] / step_count).item()),
                "yaw_rmse_rad": float(torch.sqrt(squared_yaw[index] / step_count).item()),
                "speed_error_rmse_mps": float(torch.sqrt(squared_speed[index] / step_count).item()),
                "saturation_rate": float(saturation_count[index].item() / step_count),
                "material_saturation_rate": float(
                    material_saturation_count[index].item() / step_count
                ),
                "constraint_violations": int(constraint_violations[index].item()),
                "command_smoothness_mean_sq": float(command_smoothness[index].item() / step_count),
                "final_progress_m": float(final_progress[index].item()),
                "path_completion_ratio": float(
                    np.clip(final_progress[index].item() / total_length, 0.0, 1.0)
                ),
                "final_goal_distance_m": float(final_goal_distance[index].item()),
                "goal_radius_m": goal_radius,
            }
        )
    return episodes


def _compare(
    aggregate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    reference_aggregate = reference["aggregate"]
    success_drop = float(reference_aggregate["success_rate"]) - float(aggregate["success_rate"])
    comparisons = {}
    for key in (
        "three_d_rmse_m_mean",
        "horizontal_rmse_m_mean",
        "vertical_rmse_m_mean",
        "saturation_rate_mean",
        "constraint_violations_mean",
    ):
        if reference_aggregate.get(key) is not None and aggregate.get(key) is not None:
            comparisons[key] = float(aggregate[key]) - float(reference_aggregate[key])
    return {
        "success_rate_drop": success_drop,
        "metric_gaps": comparisons,
        "passes_temporary_gate": success_drop <= 0.02
        and comparisons.get("three_d_rmse_m_mean", 0.0) <= 0.02,
    }


def _write_markdown(
    output: Path,
    aggregate: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> None:
    lines = [
        "# Batched low-level B0 evaluation",
        "",
        f"- success rate: `{aggregate['success_rate']:.4f}`",
        f"- 3D RMSE mean: `{aggregate['three_d_rmse_m_mean']:.6f} m`",
        f"- horizontal RMSE mean: `{aggregate['horizontal_rmse_m_mean']:.6f} m`",
        f"- vertical RMSE mean: `{aggregate['vertical_rmse_m_mean']:.6f} m`",
        f"- material saturation mean: `{aggregate['material_saturation_rate_mean']:.6f}`",
        f"- terminations: `{aggregate['termination_counts']}`",
    ]
    if comparison is not None:
        lines.extend(
            [
                "",
                "## PVS comparison",
                "",
                f"- success drop: `{comparison['success_rate_drop']:.6f}`",
                f"- metric gaps: `{comparison['metric_gaps']}`",
                f"- temporary gate: `{comparison['passes_temporary_gate']}`",
            ]
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the batched low-level B0 controller on paired seeds."
    )
    parser.add_argument("--config", default="configs/lowlevel/odarrl_control_base.json")
    parser.add_argument("--training-config", default=None)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--observation-schema", choices=("v1", "v2"), default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output", default="artifacts/lowlevel/batch4096/p1_b0_50seed")
    parser.add_argument(
        "--compare-summary",
        default="artifacts/lowlevel/server_b0_50seed_r2/summary.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--projection-search-radius", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.projection_search_radius < 1:
        parser.error("--projection-search-radius must be positive")

    torch.set_num_threads(args.threads)
    config = load_config(_resolve(args.config))
    policy_checkpoint = _resolve(args.policy_checkpoint) if args.policy_checkpoint else None
    training_config = _resolve(args.training_config) if args.training_config else None
    if policy_checkpoint is not None and training_config is None:
        parser.error("--training-config is required with --policy-checkpoint")
    if policy_checkpoint is None and training_config is not None:
        parser.error("--training-config is only valid with --policy-checkpoint")

    checkpoint_metadata: dict[str, Any] | None = None
    if policy_checkpoint is not None:
        checkpoint = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_metadata = validate_checkpoint_metadata(checkpoint)
    if args.observation_schema is None:
        schema_version = (
            str(checkpoint_metadata["observation_schema"])
            if checkpoint_metadata is not None
            else "v1"
        )
    else:
        schema_version = args.observation_schema
    schema = OBSERVATION_SCHEMA_V1 if schema_version == "v1" else OBSERVATION_SCHEMA_V2
    if checkpoint_metadata is not None:
        validate_checkpoint_metadata(
            checkpoint,
            expected_observation_schema=schema.version,
            expected_observation_dim=schema.dim,
        )
    seeds = _parse_seeds(
        args.seeds,
        config.evaluation.seed_start,
        config.evaluation.seed_count,
    )
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    policy = (
        _DeterministicActorPolicy(
            policy_checkpoint,
            training_config,
            observation_dim=schema.dim,
            device=device,
        )
        if policy_checkpoint is not None and training_config is not None
        else None
    )
    episodes = _run_batch(
        config,
        seeds,
        device=device,
        dtype=dtype,
        projection_search_radius=args.projection_search_radius,
        schema=schema,
        policy=policy,
    )
    aggregate = _aggregate(episodes)
    comparison = None
    if args.compare_summary and Path(args.compare_summary).exists():
        reference = json.loads(Path(args.compare_summary).read_text(encoding="utf-8"))
        comparison = _compare(aggregate, reference)
    payload = {
        "config_path": args.config,
        "config_sha256": config.config_sha256,
        "backend": "batch_torch",
        "schema_version": schema.version,
        "observation_dim": schema.dim,
        "policy_checkpoint": str(policy_checkpoint) if policy_checkpoint else None,
        "policy_checkpoint_sha256": (sha256_file(policy_checkpoint) if policy_checkpoint else None),
        "checkpoint_metadata": checkpoint_metadata,
        "seeds": list(seeds),
        "aggregate": aggregate,
        "comparison": comparison,
        "episodes": episodes,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(output, aggregate, comparison)
    print(json.dumps({"aggregate": aggregate, "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
