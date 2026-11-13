"""Per-episode record construction for PC PPO evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from auv_lowlevel.outcome_contract import DEFAULT_SUCCESS_FRAME, score_success_contract

from .common import TURN_CURVATURE_THRESHOLD
from .metrics import _episode_series


def _episode_records(
    metrics: dict[str, Any],
    seeds: Sequence[int],
    *,
    path_enabled: bool,
    path_length_m: float,
    action_full_scale: Sequence[float],
    control_dt_s: float = 0.02,
    endpoint_stop_enabled: bool = False,
    goal_radius_m: float = 2.0,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
    plant_scale_factors: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return per-seed closed-loop metrics for paired ABC analysis."""

    active_steps = np.maximum(metrics["active_steps"].astype(np.int64), 1)
    path_steps = np.maximum(metrics["path_steps"].astype(np.int64), 1)
    scale = np.asarray(action_full_scale, dtype=np.float64)
    if scale.shape != (3,):
        raise ValueError("action_full_scale must contain exactly three channels")
    count = len(seeds)
    arrays = (
        "goal_terminations",
        "timeout_terminations",
        "safety_terminations",
        "material_saturation_steps",
    )
    if any(len(metrics[name]) != count for name in arrays):
        raise ValueError("episode metrics do not align with the requested seeds")
    if len(metrics["active_steps"]) != count:
        raise ValueError("active step metrics do not align with the requested seeds")
    if plant_scale_factors is not None and plant_scale_factors.shape != (count, 5):
        raise ValueError("plant_scale_factors must have shape (episodes, 5)")

    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        steps = int(active_steps[index])
        row: dict[str, Any] = {
            "seed": int(seed),
            "plant_scale_factors": (
                None
                if plant_scale_factors is None
                else {
                    "drag_scale": float(plant_scale_factors[index, 0]),
                    "thrust_scale": float(plant_scale_factors[index, 1]),
                    "fin_scale": float(plant_scale_factors[index, 2]),
                    "added_mass_scale": float(plant_scale_factors[index, 3]),
                    "actuator_time_scale": float(plant_scale_factors[index, 4]),
                }
            ),
            "steps": steps,
            "success": bool(metrics["goal_terminations"][index] > 0),
            "goal_plane_miss": bool(metrics["goal_plane_miss_terminations"][index] > 0),
            "timeout": bool(metrics["timeout_terminations"][index] > 0),
            "safety": bool(metrics["safety_terminations"][index] > 0),
            "goal_capture_pnav": bool(metrics["pnav_goal_capture_seen"][index]),
            "goal_capture_eta": bool(metrics["eta_goal_capture_seen"][index]),
            "goal_capture_eta_at_done": bool(metrics["eta_goal_capture_at_done"][index]),
            "endpoint_entry_reached": bool(metrics["endpoint_entry_reached"][index]),
            "mean_reward": float(metrics["reward_sum"][index] / steps),
            "material_saturation_rate": float(metrics["material_saturation_steps"][index] / steps),
            "command_residual_abs_mean": (
                None
                if metrics["command_residual_steps"][index] <= 0
                else float(
                    np.mean(
                        np.abs(
                            _episode_series(
                                metrics,
                                "command_residual_applied_values",
                                "command_residual_applied_indices",
                                index,
                            )
                        )
                    )
                )
            ),
            "command_residual_scale_clip_rate": float(
                metrics["command_residual_scale_clip_steps"][index]
                / max(int(metrics["command_residual_steps"][index]), 1)
            ),
            "command_residual_rate_clip_rate": float(
                metrics["command_residual_rate_clip_steps"][index]
                / max(int(metrics["command_residual_steps"][index]), 1)
            ),
            "actuator_residual_abs_mean": (
                None
                if metrics["actuator_residual_steps"][index] <= 0
                else float(
                    np.mean(
                        np.abs(
                            _episode_series(
                                metrics,
                                "actuator_residual_applied_values",
                                "actuator_residual_applied_indices",
                                index,
                            )
                        )
                    )
                )
            ),
            "actuator_residual_scale_clip_rate": float(
                metrics["actuator_residual_scale_clip_steps"][index]
                / max(int(metrics["actuator_residual_steps"][index]), 1)
            ),
            "actuator_residual_magnitude_clip_rate": float(
                metrics["actuator_residual_magnitude_clip_steps"][index]
                / max(int(metrics["actuator_residual_steps"][index]), 1)
            ),
            "actuator_residual_rate_clip_rate": float(
                metrics["actuator_residual_rate_clip_steps"][index]
                / max(int(metrics["actuator_residual_steps"][index]), 1)
            ),
            "actuator_residual_gate_mean": (
                None
                if not metrics["actuator_residual_gate_indices"]
                or not any(index in values for values in metrics["actuator_residual_gate_indices"])
                else float(
                    np.mean(
                        _episode_series(
                            metrics,
                            "actuator_residual_gate_values",
                            "actuator_residual_gate_indices",
                            index,
                        )
                    )
                )
            ),
            "mean_abs_yaw_error_rad": float(metrics["abs_yaw_error_sum"][index] / steps),
            "mean_abs_pitch_error_rad": float(metrics["abs_pitch_error_sum"][index] / steps),
            "mean_abs_speed_error_mps": float(metrics["abs_speed_error_sum"][index] / steps),
            "yaw_rmse_rad": float(np.sqrt(metrics["yaw_error_sq_sum"][index] / steps)),
            "pitch_rmse_rad": float(np.sqrt(metrics["pitch_error_sq_sum"][index] / steps)),
            "speed_rmse_mps": float(np.sqrt(metrics["speed_error_sq_sum"][index] / steps)),
            "roll_abs_max_rad": float(metrics["abs_roll_max"][index]),
            "final_hold_fraction": float(metrics["final_hold_fraction"][index]),
        }
        row.update(
            score_success_contract(
                p_nav_tight_success=row["success"],
                eta_tight_ever=row["goal_capture_eta"],
                eta_tight_at_done=row["goal_capture_eta_at_done"],
                success_frame=success_frame,
                eta_done_distance_m=row.get("final_goal_distance_m"),
                p_nav_done_distance_m=row.get("p_nav_done_distance_m"),
                goal_radius_m=float(goal_radius_m),
            )
        )
        if path_enabled:
            path_count = int(path_steps[index])
            pre_goal_count = int(metrics["pre_goal_path_steps"][index])
            miss_distance = float(metrics["miss_distance_at_plane_m"][index])
            overshoot = float(metrics["post_goal_overshoot_m"][index])
            terminal_steps = int(metrics["terminal_longitudinal_steps"][index])
            row.update(
                {
                    "path_horizontal_rmse_m": float(
                        np.sqrt(metrics["horizontal_error_sq_sum"][index] / path_count)
                    ),
                    "path_vertical_rmse_m": float(
                        np.sqrt(metrics["vertical_error_sq_sum"][index] / path_count)
                    ),
                    "path_three_d_rmse_m": float(
                        np.sqrt(metrics["three_d_error_sq_sum"][index] / path_count)
                    ),
                    "pre_goal_3d_rmse_m": (
                        None
                        if pre_goal_count <= 0
                        else float(
                            np.sqrt(
                                metrics["pre_goal_three_d_error_sq_sum"][index] / pre_goal_count
                            )
                        )
                    ),
                    "final_path_progress_m": float(metrics["final_progress_m"][index]),
                    "path_completion_ratio": float(
                        np.clip(
                            metrics["final_progress_m"][index] / max(path_length_m, 1e-12),
                            0.0,
                            1.0,
                        )
                    ),
                    "miss_distance_at_plane_m": (
                        None if not np.isfinite(miss_distance) else miss_distance
                    ),
                    "post_goal_overshoot_m": (
                        None if endpoint_stop_enabled or not np.isfinite(overshoot) else overshoot
                    ),
                    "terminal_longitudinal_rmse_m": (
                        None
                        if endpoint_stop_enabled or terminal_steps <= 0
                        else float(
                            np.sqrt(metrics["terminal_longitudinal_sq_sum"][index] / terminal_steps)
                        )
                    ),
                }
            )
        path_action_count = int(metrics["path_action_steps"][index])
        if path_action_count > 0:
            path_action_abs = metrics["path_action_delta_abs_sum"][index]
            row.update(
                {
                    "path_action_delta_l2_mean": float(
                        metrics["path_action_delta_l2_sum"][index] / path_action_count
                    ),
                    "path_action_delta_l2_max": float(metrics["path_action_delta_l2_max"][index]),
                    "path_action_delta_abs_mean": float(
                        np.sum(path_action_abs) / (path_action_count * 3)
                    ),
                    "path_action_delta_rudder_abs_mean_rad": float(
                        path_action_abs[0] / path_action_count * scale[0]
                    ),
                    "path_action_delta_stern_abs_mean_rad": float(
                        path_action_abs[1] / path_action_count * scale[1]
                    ),
                    "path_action_delta_rpm_abs_mean_rpm": float(
                        path_action_abs[2] / path_action_count * scale[2]
                    ),
                }
            )
        history_action_count = int(metrics["history_action_steps"][index])
        if history_action_count > 0:
            history_action_abs = metrics["history_action_delta_abs_sum"][index]
            row.update(
                {
                    "history_action_delta_l2_mean": float(
                        metrics["history_action_delta_l2_sum"][index] / history_action_count
                    ),
                    "history_action_delta_l2_max": float(
                        metrics["history_action_delta_l2_max"][index]
                    ),
                    "history_action_delta_abs_mean": float(
                        np.sum(history_action_abs) / (history_action_count * 3)
                    ),
                    "history_action_delta_rudder_abs_mean_rad": float(
                        history_action_abs[0] / history_action_count * scale[0]
                    ),
                    "history_action_delta_stern_abs_mean_rad": float(
                        history_action_abs[1] / history_action_count * scale[1]
                    ),
                    "history_action_delta_rpm_abs_mean_rpm": float(
                        history_action_abs[2] / history_action_count * scale[2]
                    ),
                }
            )
        residual_ratio_values = _episode_series(
            metrics,
            "residual_to_base_ratio_values",
            "residual_to_base_ratio_indices",
            index,
        )
        if residual_ratio_values.size:
            row["residual_to_base_l2_ratio_mean"] = float(np.mean(residual_ratio_values))
            row["residual_to_base_l2_ratio_p95"] = float(np.percentile(residual_ratio_values, 95))
        surge_values = _episode_series(
            metrics, "ground_surge_values", "ground_surge_indices", index
        )
        if surge_values.size:
            row["ground_surge_mean_mps"] = float(np.mean(surge_values))
            row["ground_surge_std_mps"] = float(np.std(surge_values))
        surge_delta_values = _episode_series(
            metrics,
            "ground_surge_delta_values",
            "ground_surge_delta_indices",
            index,
        )
        if surge_delta_values.size:
            row["ground_surge_delta_abs_p95_mps"] = float(
                np.percentile(np.abs(surge_delta_values), 95)
            )
        speed_values = _episode_series(
            metrics, "ground_speed_values", "ground_speed_indices", index
        )
        if speed_values.size:
            row["ground_speed_mean_mps"] = float(np.mean(speed_values))
            row["ground_speed_std_mps"] = float(np.std(speed_values))
            row["ground_speed_min_mps"] = float(np.min(speed_values))
            row["ground_speed_max_mps"] = float(np.max(speed_values))
        envelope_steps = int(metrics["ground_speed_envelope_steps"][index])
        if envelope_steps > 0:
            row["ground_speed_below_min_rate"] = float(
                metrics["ground_speed_below_min_steps"][index] / envelope_steps
            )
            row["ground_speed_above_max_rate"] = float(
                metrics["ground_speed_above_max_steps"][index] / envelope_steps
            )
        row.update(
            {
                "initial_speed_mps": float(metrics["initial_speed_mps"][index]),
                "initial_propeller_rpm": float(metrics["initial_propeller_rpm"][index]),
                "initial_speed_ratio": float(metrics["initial_speed_ratio"][index]),
                "initial_propeller_rpm_ratio": float(metrics["initial_propeller_rpm_ratio"][index]),
            }
        )
        rpm_values = _episode_series(metrics, "rpm_values", "rpm_indices", index)
        if rpm_values.size:
            row["rpm_sq_mean_rpm2"] = float(np.mean(np.square(rpm_values)))
            row["rpm_sq_integral_proxy_rpm2_s"] = float(
                np.sum(np.square(rpm_values)) * control_dt_s
            )
        curvature_values = _episode_series(
            metrics,
            "path_curvature_values",
            "path_curvature_indices",
            index,
        )
        if curvature_values.size:
            row["abs_curvature_mean_1_per_m"] = float(np.mean(np.abs(curvature_values)))
            row["turn_step_rate"] = float(
                np.mean(np.abs(curvature_values) >= TURN_CURVATURE_THRESHOLD)
            )
        action_count = int(metrics["action_delta_steps"][index])
        if action_count > 0:
            action_abs = metrics["action_delta_abs_sum"][index]
            row.update(
                {
                    "action_delta_abs_mean": float(
                        np.sum(action_abs) / (action_count * scale.size)
                    ),
                    "action_delta_l2_mean": float(
                        metrics["action_delta_l2_sum"][index] / action_count
                    ),
                    "action_delta_rudder_abs_mean_rad": float(
                        action_abs[0] / action_count * scale[0]
                    ),
                    "action_delta_stern_abs_mean_rad": float(
                        action_abs[1] / action_count * scale[1]
                    ),
                    "action_delta_rpm_abs_mean_rpm": float(action_abs[2] / action_count * scale[2]),
                }
            )
        turn_steps = int(metrics["turn_steps"][index])
        straight_steps = int(metrics["straight_steps"][index])
        if turn_steps > 0:
            row["turn_speed_mean_mps"] = float(metrics["turn_speed_sum"][index] / turn_steps)
        if straight_steps > 0:
            row["straight_speed_mean_mps"] = float(
                metrics["straight_speed_sum"][index] / straight_steps
            )
        if turn_steps > 0 and straight_steps > 0:
            row["turn_straight_speed_ratio"] = float(
                row["turn_speed_mean_mps"] / max(row["straight_speed_mean_mps"], 1e-12)
            )
        row["episode_time_s"] = float(steps * control_dt_s)
        row["time_to_goal_s"] = float(steps * control_dt_s) if row["success"] else None
        row["path_overshoot_m"] = float(metrics["path_overshoot_m"][index])
        if bool(metrics["final_goal_distance_set"][index]):
            row["final_goal_distance_m"] = float(metrics["final_goal_distance_m"][index])
            row["p_nav_done_distance_m"] = float(metrics["pnav_final_goal_distance_m"][index])
        rows.append(row)
    return rows
