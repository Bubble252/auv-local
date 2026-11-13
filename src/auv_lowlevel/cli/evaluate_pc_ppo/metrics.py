"""Episode metric buffers and compact summary helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _new_episode_metrics(count: int) -> dict[str, Any]:
    return {
        "active_steps": np.zeros(count, dtype=np.int64),
        "reward_sum": np.zeros(count, dtype=np.float64),
        "abs_yaw_error_sum": np.zeros(count, dtype=np.float64),
        "abs_pitch_error_sum": np.zeros(count, dtype=np.float64),
        "abs_speed_error_sum": np.zeros(count, dtype=np.float64),
        "yaw_error_sq_sum": np.zeros(count, dtype=np.float64),
        "pitch_error_sq_sum": np.zeros(count, dtype=np.float64),
        "speed_error_sq_sum": np.zeros(count, dtype=np.float64),
        "abs_roll_max": np.zeros(count, dtype=np.float64),
        "abs_roll_sum": np.zeros(count, dtype=np.float64),
        "roll_abs_values": [],
        "roll_rate_abs_values": [],
        "path_steps": np.zeros(count, dtype=np.int64),
        "horizontal_error_sq_sum": np.zeros(count, dtype=np.float64),
        "vertical_error_sq_sum": np.zeros(count, dtype=np.float64),
        "three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "pre_goal_path_steps": np.zeros(count, dtype=np.int64),
        "pre_goal_three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "final_progress_m": np.zeros(count, dtype=np.float64),
        "alos_horizontal_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_vertical_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_final_progress_m": np.zeros(count, dtype=np.float64),
        "route_ground_speed_ratio_min": np.full(count, np.inf, dtype=np.float64),
        "route_ground_speed_ratio_min_time_s": np.full(count, np.inf, dtype=np.float64),
        "route_ground_speed_ratio_sum": np.zeros(count, dtype=np.float64),
        "route_ground_speed_steps": np.zeros(count, dtype=np.int64),
        "ground_speed_ratio_min": np.full(count, np.inf, dtype=np.float64),
        "ground_speed_ratio_min_time_s": np.full(count, np.inf, dtype=np.float64),
        "ground_speed_ratio_sum": np.zeros(count, dtype=np.float64),
        "ground_speed_steps": np.zeros(count, dtype=np.int64),
        "material_saturation_steps": np.zeros(count, dtype=np.int64),
        "safety_terminations": np.zeros(count, dtype=np.int64),
        "goal_terminations": np.zeros(count, dtype=np.int64),
        "goal_plane_miss_terminations": np.zeros(count, dtype=np.int64),
        "timeout_terminations": np.zeros(count, dtype=np.int64),
        "pnav_goal_capture_seen": np.zeros(count, dtype=bool),
        "eta_goal_capture_seen": np.zeros(count, dtype=bool),
        "eta_goal_capture_at_done": np.zeros(count, dtype=bool),
        "miss_distance_at_plane_m": np.full(count, np.nan, dtype=np.float64),
        "endpoint_entry_reached": np.zeros(count, dtype=bool),
        "post_goal_overshoot_m": np.full(count, np.nan, dtype=np.float64),
        "terminal_longitudinal_steps": np.zeros(count, dtype=np.int64),
        "terminal_longitudinal_sq_sum": np.zeros(count, dtype=np.float64),
        "max_hold_fraction": np.zeros(count, dtype=np.float64),
        "final_hold_fraction": np.zeros(count, dtype=np.float64),
        "sensor_p_nav_error_m": [],
        "sensor_dvl_bottom_valid": [],
        "sensor_current_estimate_valid": [],
        "sensor_usbl_valid": [],
        "sensor_usbl_outlier": [],
        "sensor_nav_age_s": [],
        "sensor_dvl_bottom_age_s": [],
        "sensor_current_estimate_age_s": [],
        "guidance_current_valid": [],
        "guidance_current_age_s": [],
        "guidance_crab_angle_deg": [],
        "guidance_speed_scale": [],
        "guidance_terminal_phase": [],
        "guidance_terminal_speed_ratio": [],
        "guidance_terminal_speed_ratio_values": [],
        "guidance_saturation_ema": [],
        "guidance_saturation_emergency": [],
        "guidance_saturation_emergency_steps": np.zeros(count, dtype=np.int64),
        "guidance_route_floor_active": [],
        "guidance_route_floor_launch_active": [],
        "guidance_route_floor_valid": [],
        "guidance_route_floor_boost_mps": [],
        "guidance_route_floor_ratio": [],
        "residual_action_values": [],
        "base_action_l2_values": [],
        "residual_action_l2_values": [],
        "residual_to_base_ratio_values": [],
        "residual_to_base_ratio_indices": [],
        "command_residual_applied_values": [],
        "command_residual_applied_indices": [],
        "command_residual_scale_clip_steps": np.zeros(count, dtype=np.int64),
        "command_residual_rate_clip_steps": np.zeros(count, dtype=np.int64),
        "command_residual_steps": np.zeros(count, dtype=np.int64),
        "actuator_residual_applied_values": [],
        "actuator_residual_applied_indices": [],
        "actuator_residual_gate_values": [],
        "actuator_residual_gate_indices": [],
        "actuator_residual_scale_clip_steps": np.zeros(count, dtype=np.int64),
        "actuator_residual_magnitude_clip_steps": np.zeros(count, dtype=np.int64),
        "actuator_residual_rate_clip_steps": np.zeros(count, dtype=np.int64),
        "actuator_residual_steps": np.zeros(count, dtype=np.int64),
        "frozen_residual_gate_values": [],
        "frozen_residual_gate_indices": [],
        "frozen_residual_clip_steps": np.zeros(count, dtype=np.int64),
        "frozen_residual_steps": np.zeros(count, dtype=np.int64),
        "ppo_residual_attenuation_values": [],
        "ppo_residual_attenuation_distance_values": [],
        "ppo_residual_attenuation_indices": [],
        "path_action_delta_l2_sum": np.zeros(count, dtype=np.float64),
        "path_action_delta_l2_max": np.zeros(count, dtype=np.float64),
        "path_action_delta_abs_sum": np.zeros((count, 3), dtype=np.float64),
        "path_action_steps": np.zeros(count, dtype=np.int64),
        "history_action_delta_l2_sum": np.zeros(count, dtype=np.float64),
        "history_action_delta_l2_max": np.zeros(count, dtype=np.float64),
        "history_action_delta_abs_sum": np.zeros((count, 3), dtype=np.float64),
        "history_action_steps": np.zeros(count, dtype=np.int64),
        "ground_surge_sum": np.zeros(count, dtype=np.float64),
        "ground_surge_sq_sum": np.zeros(count, dtype=np.float64),
        "ground_speed_sum": np.zeros(count, dtype=np.float64),
        "ground_speed_sq_sum": np.zeros(count, dtype=np.float64),
        "rpm_sum": np.zeros(count, dtype=np.float64),
        "rpm_sq_sum": np.zeros(count, dtype=np.float64),
        "action_delta_abs_sum": np.zeros((count, 3), dtype=np.float64),
        "action_delta_l2_sum": np.zeros(count, dtype=np.float64),
        "action_delta_steps": np.zeros(count, dtype=np.int64),
        "turn_speed_sum": np.zeros(count, dtype=np.float64),
        "turn_steps": np.zeros(count, dtype=np.int64),
        "straight_speed_sum": np.zeros(count, dtype=np.float64),
        "straight_steps": np.zeros(count, dtype=np.int64),
        "final_goal_distance_m": np.zeros(count, dtype=np.float64),
        "final_goal_distance_set": np.zeros(count, dtype=bool),
        "pnav_final_goal_distance_m": np.zeros(count, dtype=np.float64),
        "path_overshoot_m": np.zeros(count, dtype=np.float64),
        "previous_ground_surge_for_delta": np.full(count, np.nan, dtype=np.float64),
        "previous_action_for_delta": np.zeros((count, 3), dtype=np.float64),
        "has_previous_action": np.zeros(count, dtype=bool),
        "ground_surge_values": [],
        "ground_surge_indices": [],
        "ground_surge_delta_values": [],
        "ground_surge_delta_indices": [],
        "ground_speed_values": [],
        "ground_speed_indices": [],
        "ground_speed_below_min_steps": np.zeros(count, dtype=np.int64),
        "ground_speed_above_max_steps": np.zeros(count, dtype=np.int64),
        "ground_speed_envelope_steps": np.zeros(count, dtype=np.int64),
        "initial_speed_mps": np.zeros(count, dtype=np.float64),
        "initial_propeller_rpm": np.zeros(count, dtype=np.float64),
        "initial_speed_ratio": np.zeros(count, dtype=np.float64),
        "initial_propeller_rpm_ratio": np.zeros(count, dtype=np.float64),
        "rpm_values": [],
        "rpm_indices": [],
        "path_curvature_values": [],
        "path_curvature_indices": [],
        "done": np.zeros(count, dtype=bool),
    }


def _concat_series(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)


def _episode_series(
    metrics: dict[str, Any],
    values_key: str,
    indices_key: str,
    index: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for values, indices in zip(
        metrics.get(values_key, []), metrics.get(indices_key, []), strict=False
    ):
        mask = indices == index
        if bool(mask.any()):
            chunks.append(values[mask])
    return _concat_series(chunks)


def _residual_action_summary(
    metrics: dict[str, Any],
    action_full_scale: Sequence[float],
) -> dict[str, float]:
    values = metrics.get("residual_action_values", [])
    if not values:
        return {}
    residual_abs = np.concatenate(values, axis=0)
    base_l2 = np.concatenate(metrics["base_action_l2_values"], axis=0)
    residual_l2 = np.concatenate(metrics["residual_action_l2_values"], axis=0)
    scale = np.asarray(action_full_scale, dtype=np.float64)
    if residual_abs.ndim != 2 or residual_abs.shape[1] != scale.size:
        raise ValueError("residual action shape does not match action_full_scale")
    physical_abs = residual_abs * scale[None, :]
    ratios = residual_l2 / np.maximum(base_l2, 1e-12)
    return {
        "residual_action_abs_mean": float(np.mean(residual_abs)),
        "residual_action_abs_p95": float(np.percentile(residual_abs, 95)),
        "residual_action_abs_max": float(np.max(residual_abs)),
        "residual_action_l2_mean": float(np.mean(residual_l2)),
        "base_action_l2_mean": float(np.mean(base_l2)),
        "residual_to_base_l2_ratio_mean": float(np.mean(ratios)),
        "residual_to_base_l2_ratio_p95": float(np.percentile(ratios, 95)),
        "residual_rudder_abs_mean_rad": float(np.mean(physical_abs[:, 0])),
        "residual_stern_abs_mean_rad": float(np.mean(physical_abs[:, 1])),
        "residual_rpm_abs_mean_rpm": float(np.mean(physical_abs[:, 2])),
    }


def _command_residual_summary(metrics: dict[str, Any]) -> dict[str, float]:
    values = metrics.get("command_residual_applied_values", [])
    if not values:
        return {}
    applied = np.concatenate(values, axis=0)
    if applied.ndim != 2 or applied.shape[1] != 3:
        raise ValueError("command residual values must have shape (N, 3)")
    steps = max(int(np.sum(metrics.get("command_residual_steps", []))), 1)
    absolute = np.abs(applied)
    result = {
        "command_residual_abs_mean": float(np.mean(absolute)),
        "command_residual_abs_p95": float(np.percentile(absolute, 95)),
        "command_residual_abs_max": float(np.max(absolute)),
        "command_residual_heading_abs_mean_rad": float(np.mean(absolute[:, 0])),
        "command_residual_pitch_abs_mean_rad": float(np.mean(absolute[:, 1])),
        "command_residual_speed_abs_mean_mps": float(np.mean(absolute[:, 2])),
        "command_residual_scale_clip_rate": float(
            np.sum(metrics["command_residual_scale_clip_steps"]) / steps
        ),
        "command_residual_rate_clip_rate": float(
            np.sum(metrics["command_residual_rate_clip_steps"]) / steps
        ),
    }
    return result


def _actuator_residual_summary(metrics: dict[str, Any]) -> dict[str, float]:
    values = metrics.get("actuator_residual_applied_values", [])
    if not values:
        return {}
    applied = np.concatenate(values, axis=0)
    if applied.ndim != 2 or applied.shape[1] != 3:
        raise ValueError("actuator residual values must have shape (N, 3)")
    steps = max(int(np.sum(metrics.get("actuator_residual_steps", []))), 1)
    absolute = np.abs(applied)
    result = {
        "actuator_residual_abs_mean": float(np.mean(absolute)),
        "actuator_residual_abs_p95": float(np.percentile(absolute, 95)),
        "actuator_residual_abs_max": float(np.max(absolute)),
        "actuator_residual_rudder_abs_mean_rad": float(np.mean(absolute[:, 0])),
        "actuator_residual_stern_abs_mean_rad": float(np.mean(absolute[:, 1])),
        "actuator_residual_rpm_abs_mean_rpm": float(np.mean(absolute[:, 2])),
        "actuator_residual_scale_clip_rate": float(
            np.sum(metrics["actuator_residual_scale_clip_steps"]) / steps
        ),
        "actuator_residual_magnitude_clip_rate": float(
            np.sum(metrics["actuator_residual_magnitude_clip_steps"]) / steps
        ),
        "actuator_residual_rate_clip_rate": float(
            np.sum(metrics["actuator_residual_rate_clip_steps"]) / steps
        ),
    }
    gate_values = metrics.get("actuator_residual_gate_values", [])
    if gate_values:
        gate = np.concatenate(gate_values, axis=0)
        result.update(
            {
                "actuator_residual_gate_mean": float(np.mean(gate)),
                "actuator_residual_gate_zero_fraction": float(np.mean(gate <= 1e-9)),
                "actuator_residual_gate_full_fraction": float(np.mean(gate >= 1.0 - 1e-9)),
            }
        )
    return result


def _frozen_residual_summary(metrics: dict[str, Any]) -> dict[str, float]:
    values = metrics.get("frozen_residual_gate_values", [])
    if not values:
        return {}
    gate = np.concatenate(values, axis=0)
    steps = max(int(np.sum(metrics.get("frozen_residual_steps", []))), 1)
    return {
        "frozen_residual_gate_mean": float(np.mean(gate)),
        "frozen_residual_gate_zero_fraction": float(np.mean(gate <= 1e-9)),
        "frozen_residual_gate_full_fraction": float(np.mean(gate >= 1.0 - 1e-9)),
        "frozen_residual_clip_rate": float(np.sum(metrics["frozen_residual_clip_steps"]) / steps),
    }


def _ppo_residual_attenuation_summary(metrics: dict[str, Any]) -> dict[str, float]:
    values = metrics.get("ppo_residual_attenuation_values", [])
    if not values:
        return {}
    attenuation = np.concatenate(values, axis=0)
    return {
        "ppo_residual_attenuation_mean": float(np.mean(attenuation)),
        "ppo_residual_attenuation_zero_fraction": float(np.mean(attenuation <= 1e-9)),
        "ppo_residual_attenuation_full_fraction": float(np.mean(attenuation >= 1.0 - 1e-9)),
        "ppo_residual_attenuation_steps": int(attenuation.size),
    }
