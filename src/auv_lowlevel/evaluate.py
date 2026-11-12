from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from auv_core.contracts import FloatArray

from .config import LowLevelConfig
from .env import BaseCommandFunction, LowLevelEnv

ResidualPolicy = Callable[[FloatArray, int], FloatArray]


@dataclass(frozen=True)
class Rollout:
    seed: int
    config_sha256: str
    termination: str | None
    success: bool
    t_s: FloatArray
    position_ned_m: FloatArray
    reference_point_ned_m: FloatArray
    horizontal_error_m: FloatArray
    vertical_error_m: FloatArray
    error_3d_m: FloatArray
    progress_m: FloatArray
    yaw_error_rad: FloatArray
    water_surge_mps: FloatArray
    current_ned_mps: FloatArray
    reference_speed_mps: FloatArray
    path_length_m: float
    goal_radius_m: float
    goal_ned_m: FloatArray
    observation: FloatArray
    final_observation: FloatArray
    base_command: FloatArray
    residual_normalized: FloatArray
    delta_command: FloatArray
    requested_total: FloatArray
    limited_total: FloatArray
    actual_actuator: FloatArray
    reward_total: FloatArray
    reward_ppo_scaled: FloatArray
    terminated: np.ndarray
    truncated: np.ndarray
    saturated: np.ndarray
    material_saturated: np.ndarray
    residual_clipped: np.ndarray
    constraint_violation_count: np.ndarray
    pvs_step_latency_s: FloatArray
    control_loop_latency_s: FloatArray
    shadow_teacher_margin_rpm: FloatArray

    @property
    def steps(self) -> int:
        return int(self.t_s.shape[0])

    def summary(self) -> dict[str, Any]:
        if self.steps == 0:
            raise ValueError("cannot summarize an empty rollout")
        h_rmse = _rmse(self.horizontal_error_m)
        v_rmse = _rmse(self.vertical_error_m)
        error_rmse = _rmse(self.error_3d_m)
        yaw_rmse = _rmse(self.yaw_error_rad)
        speed_rmse = _rmse(self.water_surge_mps - self.reference_speed_mps)
        command_deltas = np.diff(self.limited_total, axis=0)
        smoothness = (
            float(np.mean(np.sum(np.square(command_deltas), axis=1)))
            if command_deltas.size
            else 0.0
        )
        margins = self.shadow_teacher_margin_rpm[
            np.isfinite(self.shadow_teacher_margin_rpm)
        ]
        goal_distances = np.linalg.norm(
            self.position_ned_m - self.goal_ned_m[np.newaxis, :],
            axis=1,
        )
        minimum_goal_distance = float(np.min(goal_distances))
        residual_abs = np.abs(self.residual_normalized)
        return {
            "seed": self.seed,
            "config_sha256": self.config_sha256,
            "steps": self.steps,
            "duration_s": float(self.t_s[-1]),
            "termination": self.termination,
            "success": self.success,
            "return_sum": float(np.sum(self.reward_total)),
            "ppo_return_sum": float(np.sum(self.reward_ppo_scaled)),
            "horizontal_rmse_m": h_rmse,
            "vertical_rmse_m": v_rmse,
            "three_d_rmse_m": error_rmse,
            "horizontal_mean_abs_error_m": float(np.mean(np.abs(self.horizontal_error_m))),
            "vertical_mean_abs_error_m": float(np.mean(np.abs(self.vertical_error_m))),
            "three_d_mean_error_m": float(np.mean(self.error_3d_m)),
            "yaw_rmse_rad": yaw_rmse,
            "water_surge_mean_mps": float(np.mean(self.water_surge_mps)),
            "water_surge_std_mps": float(np.std(self.water_surge_mps)),
            "speed_error_rmse_mps": speed_rmse,
            "final_progress_m": float(self.progress_m[-1]),
            "path_completion_ratio": float(
                np.clip(self.progress_m[-1] / self.path_length_m, 0.0, 1.0)
            ),
            "goal_radius_m": self.goal_radius_m,
            "final_goal_distance_m": float(goal_distances[-1]),
            "minimum_goal_distance_m": minimum_goal_distance,
            "arrival_time_s": float(self.t_s[-1]) if self.success else None,
            "saturation_rate": float(np.mean(self.saturated)),
            "material_saturation_rate": float(np.mean(self.material_saturated)),
            "residual_clip_rate": float(np.mean(self.residual_clipped)),
            "residual_abs_mean": float(np.mean(residual_abs)),
            "residual_abs_max": float(np.max(residual_abs)),
            "constraint_violations": int(np.sum(self.constraint_violation_count)),
            "command_smoothness_mean_sq": smoothness,
            "pvs_step_latency_p50_ms": float(np.percentile(self.pvs_step_latency_s, 50) * 1e3),
            "pvs_step_latency_p95_ms": float(np.percentile(self.pvs_step_latency_s, 95) * 1e3),
            "control_loop_latency_p50_ms": float(
                np.percentile(self.control_loop_latency_s, 50) * 1e3
            ),
            "control_loop_latency_p95_ms": float(
                np.percentile(self.control_loop_latency_s, 95) * 1e3
            ),
            "control_loop_p95_rate_hz": float(
                1.0 / np.percentile(self.control_loop_latency_s, 95)
            ),
            "pvs_step_rate_hz": float(1.0 / np.mean(self.pvs_step_latency_s)),
            "shadow_teacher_max_abs_margin_rpm": (
                float(np.max(np.abs(margins))) if margins.size else None
            ),
        }


def run_rollout(
    config: LowLevelConfig,
    seed: int,
    *,
    residual_policy: ResidualPolicy | None = None,
    base_mode: Literal["teacher", "frozen"] = "teacher",
    base_command_fn: BaseCommandFunction | None = None,
    expose_current: bool | None = None,
) -> Rollout:
    env = LowLevelEnv(
        config,
        base_mode=base_mode,
        base_command_fn=base_command_fn,
        expose_current=expose_current,
    )
    if residual_policy is not None:
        bind_environment = getattr(residual_policy, "bind_environment", None)
        if callable(bind_environment):
            bind_environment(env)
    observation = env.reset(seed)
    if residual_policy is not None:
        reset_policy = getattr(residual_policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
    records: dict[str, list[np.ndarray | float | int | bool | str | None]] = {
        "t_s": [],
        "position_ned_m": [],
        "reference_point_ned_m": [],
        "horizontal_error_m": [],
        "vertical_error_m": [],
        "error_3d_m": [],
        "progress_m": [],
        "yaw_error_rad": [],
        "water_surge_mps": [],
        "current_ned_mps": [],
        "reference_speed_mps": [],
        "observation": [],
        "base_command": [],
        "residual_normalized": [],
        "delta_command": [],
        "requested_total": [],
        "limited_total": [],
        "actual_actuator": [],
        "reward_total": [],
        "reward_ppo_scaled": [],
        "terminated": [],
        "truncated": [],
        "saturated": [],
        "material_saturated": [],
        "residual_clipped": [],
        "constraint_violation_count": [],
        "pvs_step_latency_s": [],
        "control_loop_latency_s": [],
        "shadow_teacher_margin_rpm": [],
    }
    termination: str | None = None
    while True:
        residual = (
            np.zeros(3, dtype=np.float64)
            if residual_policy is None
            else np.asarray(residual_policy(observation.copy(), env.step_count), dtype=np.float64)
        )
        result = env.step(residual)
        info = result.info
        projection = info["projection_after"]
        context = env.last_context
        records["t_s"].append(context.state.t_s)
        records["position_ned_m"].append(context.state.position_ned_m.copy())
        records["reference_point_ned_m"].append(projection.point_ned_m.copy())
        records["horizontal_error_m"].append(projection.horizontal_error_m)
        records["vertical_error_m"].append(projection.vertical_error_m)
        records["error_3d_m"].append(projection.error_3d_m)
        records["progress_m"].append(projection.progress_m)
        records["yaw_error_rad"].append(context.heading_error_rad)
        records["water_surge_mps"].append(context.water_surge_mps)
        records["current_ned_mps"].append(context.current_ned_mps.copy())
        records["reference_speed_mps"].append(
            float(context.reference.linear_velocity_body_mps[0])
        )
        records["observation"].append(info["observation_before"].copy())
        records["base_command"].append(info["base_command"].copy())
        records["residual_normalized"].append(info["residual_normalized"].copy())
        records["delta_command"].append(info["delta_command"].copy())
        records["requested_total"].append(info["requested_total"].copy())
        records["limited_total"].append(info["limited_total"].copy())
        records["actual_actuator"].append(info["actual_actuator"].copy())
        records["reward_total"].append(result.reward.total)
        records["reward_ppo_scaled"].append(result.reward.ppo_scaled)
        records["terminated"].append(result.terminated)
        records["truncated"].append(result.truncated)
        records["saturated"].append(result.info["saturated"])
        records["material_saturated"].append(result.info["material_saturated"])
        records["residual_clipped"].append(result.info["residual_clipped"])
        records["constraint_violation_count"].append(info["constraint_violation_count"])
        records["pvs_step_latency_s"].append(info["pvs_step_latency_s"])
        records["control_loop_latency_s"].append(info["control_loop_latency_s"])
        records["shadow_teacher_margin_rpm"].append(
            np.nan if info["shadow_teacher_margin_rpm"] is None else info["shadow_teacher_margin_rpm"]
        )
        observation = result.observation
        if result.terminated or result.truncated:
            termination = str(info["terminal_kind"])
            break

    return Rollout(
        seed=seed,
        config_sha256=config.config_sha256,
        termination=termination,
        success=termination == "goal",
        t_s=np.asarray(records["t_s"], dtype=np.float64),
        position_ned_m=np.asarray(records["position_ned_m"], dtype=np.float64),
        reference_point_ned_m=np.asarray(records["reference_point_ned_m"], dtype=np.float64),
        horizontal_error_m=np.asarray(records["horizontal_error_m"], dtype=np.float64),
        vertical_error_m=np.asarray(records["vertical_error_m"], dtype=np.float64),
        error_3d_m=np.asarray(records["error_3d_m"], dtype=np.float64),
        progress_m=np.asarray(records["progress_m"], dtype=np.float64),
        yaw_error_rad=np.asarray(records["yaw_error_rad"], dtype=np.float64),
        water_surge_mps=np.asarray(records["water_surge_mps"], dtype=np.float64),
        current_ned_mps=np.asarray(records["current_ned_mps"], dtype=np.float64),
        reference_speed_mps=np.asarray(records["reference_speed_mps"], dtype=np.float64),
        path_length_m=env.path.total_length_m,
        goal_radius_m=config.plant.goal_radius_m,
        goal_ned_m=env.path.goal_ned_m.copy(),
        observation=np.asarray(records["observation"], dtype=np.float64),
        final_observation=observation.copy(),
        base_command=np.asarray(records["base_command"], dtype=np.float64),
        residual_normalized=np.asarray(records["residual_normalized"], dtype=np.float64),
        delta_command=np.asarray(records["delta_command"], dtype=np.float64),
        requested_total=np.asarray(records["requested_total"], dtype=np.float64),
        limited_total=np.asarray(records["limited_total"], dtype=np.float64),
        actual_actuator=np.asarray(records["actual_actuator"], dtype=np.float64),
        reward_total=np.asarray(records["reward_total"], dtype=np.float64),
        reward_ppo_scaled=np.asarray(records["reward_ppo_scaled"], dtype=np.float64),
        terminated=np.asarray(records["terminated"], dtype=bool),
        truncated=np.asarray(records["truncated"], dtype=bool),
        saturated=np.asarray(records["saturated"], dtype=bool),
        material_saturated=np.asarray(records["material_saturated"], dtype=bool),
        residual_clipped=np.asarray(records["residual_clipped"], dtype=bool),
        constraint_violation_count=np.asarray(
            records["constraint_violation_count"], dtype=np.int64
        ),
        pvs_step_latency_s=np.asarray(records["pvs_step_latency_s"], dtype=np.float64),
        control_loop_latency_s=np.asarray(
            records["control_loop_latency_s"], dtype=np.float64
        ),
        shadow_teacher_margin_rpm=np.asarray(
            records["shadow_teacher_margin_rpm"], dtype=np.float64
        ),
    )


def summarize_rollouts(rollouts: Iterable[Rollout]) -> dict[str, Any]:
    items = list(rollouts)
    if not items:
        raise ValueError("at least one rollout is required")
    summaries = [rollout.summary() for rollout in items]
    scalar_keys = [
        "steps",
        "duration_s",
        "return_sum",
        "ppo_return_sum",
        "horizontal_rmse_m",
        "vertical_rmse_m",
        "three_d_rmse_m",
        "horizontal_mean_abs_error_m",
        "vertical_mean_abs_error_m",
        "three_d_mean_error_m",
        "yaw_rmse_rad",
        "water_surge_mean_mps",
        "water_surge_std_mps",
        "speed_error_rmse_mps",
        "goal_radius_m",
        "saturation_rate",
        "material_saturation_rate",
        "residual_clip_rate",
        "residual_abs_mean",
        "residual_abs_max",
        "constraint_violations",
        "command_smoothness_mean_sq",
        "pvs_step_latency_p50_ms",
        "pvs_step_latency_p95_ms",
        "control_loop_latency_p50_ms",
        "control_loop_latency_p95_ms",
        "control_loop_p95_rate_hz",
        "pvs_step_rate_hz",
    ]
    aggregate: dict[str, Any] = {
        "episode_count": len(items),
        "success_count": sum(bool(item["success"]) for item in summaries),
        "success_rate": float(np.mean([bool(item["success"]) for item in summaries])),
        "termination_counts": {
            str(kind): sum(item["termination"] == kind for item in summaries)
            for kind in sorted({str(item["termination"]) for item in summaries})
        },
    }
    for key in scalar_keys:
        values = np.asarray([item[key] for item in summaries], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
        aggregate[f"{key}_median"] = float(np.median(values))
        aggregate[f"{key}_p95"] = float(np.percentile(values, 95))
    arrivals = [item["arrival_time_s"] for item in summaries if item["arrival_time_s"] is not None]
    aggregate["arrival_time_s_mean"] = float(np.mean(arrivals)) if arrivals else None
    aggregate["arrival_time_s_median"] = float(np.median(arrivals)) if arrivals else None
    for key in (
        "final_goal_distance_m",
        "minimum_goal_distance_m",
    ):
        values = np.asarray([item[key] for item in summaries], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_median"] = float(np.median(values))
        aggregate[f"{key}_p95"] = float(np.percentile(values, 95))
    return {"aggregate": aggregate, "episodes": summaries}


def _rmse(values: FloatArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


__all__ = ["ResidualPolicy", "Rollout", "run_rollout", "summarize_rollouts"]
