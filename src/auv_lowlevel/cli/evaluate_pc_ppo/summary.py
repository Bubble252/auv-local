"""Aggregate summary construction for PC PPO evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from auv_lowlevel.outcome_contract import (
    DEFAULT_SUCCESS_FRAME,
    score_success_contract,
    summarize_success_contract,
)

from .common import TURN_CURVATURE_THRESHOLD
from .metrics import (
    _actuator_residual_summary,
    _command_residual_summary,
    _concat_series,
    _frozen_residual_summary,
    _ppo_residual_attenuation_summary,
    _residual_action_summary,
)


def _summary(
    metrics: dict[str, Any],
    *,
    path_enabled: bool,
    path_length_m: float,
    action_full_scale: Sequence[float],
    speed_envelope_min_mps: float,
    speed_envelope_max_mps: float,
    control_dt_s: float = 0.02,
    endpoint_stop_enabled: bool = False,
    goal_radius_m: float = 2.0,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
) -> dict[str, float]:
    steps = np.maximum(metrics["active_steps"], 1)
    roll_abs = np.concatenate(metrics["roll_abs_values"])
    roll_rate_abs = np.concatenate(metrics["roll_rate_abs_values"])
    p_nav_error = np.concatenate(metrics["sensor_p_nav_error_m"])
    nav_age = np.concatenate(metrics["sensor_nav_age_s"])
    dvl_age = np.concatenate(metrics["sensor_dvl_bottom_age_s"])
    current_age = np.concatenate(metrics["sensor_current_estimate_age_s"])
    guidance_current_valid = np.concatenate(metrics["guidance_current_valid"])
    guidance_current_age = np.concatenate(metrics["guidance_current_age_s"])
    guidance_crab_deg = np.concatenate(metrics["guidance_crab_angle_deg"])
    guidance_speed_scale = np.concatenate(metrics["guidance_speed_scale"])
    guidance_terminal_phase = np.concatenate(metrics["guidance_terminal_phase"])
    guidance_terminal_ratios = np.concatenate(metrics["guidance_terminal_speed_ratio"])
    guidance_terminal_ratio_values = metrics["guidance_terminal_speed_ratio_values"]
    if guidance_terminal_ratio_values:
        guidance_terminal_ratio_values = np.concatenate(guidance_terminal_ratio_values)
    else:
        guidance_terminal_ratio_values = np.zeros(0, dtype=np.float64)
    guidance_saturation_ema = np.concatenate(metrics["guidance_saturation_ema"])
    guidance_saturation_emergency = np.concatenate(metrics["guidance_saturation_emergency"])
    summary = {
        "success_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "command_hold_success_rate": float(np.mean(metrics["max_hold_fraction"] >= 1.0 - 1e-9)),
        "mean_abs_yaw_error_rad": float(np.mean(metrics["abs_yaw_error_sum"] / steps)),
        "mean_abs_pitch_error_rad": float(np.mean(metrics["abs_pitch_error_sum"] / steps)),
        "mean_abs_speed_error_mps": float(np.mean(metrics["abs_speed_error_sum"] / steps)),
        "yaw_rmse_rad": float(np.sqrt(np.mean(metrics["yaw_error_sq_sum"] / steps))),
        "pitch_rmse_rad": float(np.sqrt(np.mean(metrics["pitch_error_sq_sum"] / steps))),
        "speed_rmse_mps": float(np.sqrt(np.mean(metrics["speed_error_sq_sum"] / steps))),
        "roll_abs_mean_rad": float(np.mean(metrics["abs_roll_sum"] / steps)),
        "roll_abs_p95_rad": float(np.percentile(roll_abs, 95)),
        "roll_abs_max_rad": float(np.max(metrics["abs_roll_max"])),
        "roll_rate_abs_p95_rad_s": float(np.percentile(roll_rate_abs, 95)),
        "material_saturation_rate": float(
            np.sum(metrics["material_saturation_steps"]) / np.sum(steps)
        ),
        "safety_termination_rate": float(np.mean(metrics["safety_terminations"] > 0)),
        "goal_termination_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "goal_plane_miss_rate": float(np.mean(metrics["goal_plane_miss_terminations"] > 0)),
        "timeout_rate": float(np.mean(metrics["timeout_terminations"] > 0)),
        "goal_capture_pnav_rate": float(np.mean(metrics["pnav_goal_capture_seen"])),
        "goal_capture_eta_rate": float(np.mean(metrics["eta_goal_capture_seen"])),
        "mean_reward": float(np.mean(metrics["reward_sum"] / steps)),
        "mean_episode_reward": float(np.mean(metrics["reward_sum"])),
        "final_hold_fraction_mean": float(np.mean(metrics["final_hold_fraction"])),
        "nav_position_error_mean_m": float(np.mean(p_nav_error)),
        "nav_position_error_rmse_m": float(np.sqrt(np.mean(np.square(p_nav_error)))),
        "nav_position_error_p95_m": float(np.percentile(p_nav_error, 95)),
        "dvl_bottom_valid_rate": float(np.mean(np.concatenate(metrics["sensor_dvl_bottom_valid"]))),
        "dvl_bottom_outage_rate": float(
            1.0 - np.mean(np.concatenate(metrics["sensor_dvl_bottom_valid"]))
        ),
        "current_estimate_valid_rate": float(
            np.mean(np.concatenate(metrics["sensor_current_estimate_valid"]))
        ),
        "usbl_valid_rate": float(np.mean(np.concatenate(metrics["sensor_usbl_valid"]))),
        "usbl_outlier_rate": float(np.mean(np.concatenate(metrics["sensor_usbl_outlier"]))),
        "nav_age_s_mean": float(np.mean(nav_age)),
        "nav_age_s_p95": float(np.percentile(nav_age, 95)),
        "dvl_bottom_age_s_mean": float(np.mean(dvl_age)),
        "dvl_bottom_age_s_p95": float(np.percentile(dvl_age, 95)),
        "current_estimate_age_s_mean": float(np.mean(current_age)),
        "current_estimate_age_s_p95": float(np.percentile(current_age, 95)),
        "guidance_current_valid_rate": float(np.mean(guidance_current_valid)),
        "guidance_current_age_s_mean": float(np.mean(guidance_current_age)),
        "guidance_crab_deg_mean": float(np.mean(guidance_crab_deg)),
        "guidance_crab_deg_abs_mean": float(np.mean(np.abs(guidance_crab_deg))),
        "guidance_speed_scale_mean": float(np.mean(guidance_speed_scale)),
        "guidance_terminal_phase_rate": float(np.mean(guidance_terminal_phase)),
        "terminal_speed_ratio_mean": float(
            np.mean(guidance_terminal_ratio_values) if guidance_terminal_ratio_values.size else 0.0
        ),
        "guidance_terminal_speed_ratio_mean": float(np.mean(guidance_terminal_ratios)),
        "guidance_speed_ratio_mean": float(np.mean(guidance_terminal_ratios)),
        "guidance_route_speed_ratio_min": float(np.min(guidance_terminal_ratios)),
        "ground_route_speed_ratio_min": float(
            np.min(metrics["route_ground_speed_ratio_min"][metrics["route_ground_speed_steps"] > 0])
            if np.any(metrics["route_ground_speed_steps"] > 0)
            else np.nan
        ),
        "ground_route_speed_ratio_episode_min_mean": float(
            np.mean(
                metrics["route_ground_speed_ratio_min"][metrics["route_ground_speed_steps"] > 0]
            )
            if np.any(metrics["route_ground_speed_steps"] > 0)
            else np.nan
        ),
        "ground_route_speed_ratio_mean": float(
            np.mean(
                metrics["route_ground_speed_ratio_sum"]
                / np.maximum(metrics["route_ground_speed_steps"], 1)
            )
        ),
        "ground_route_speed_ratio_min_time_s": float(
            np.mean(metrics["route_ground_speed_ratio_min_time_s"])
        ),
        "ground_speed_ratio_min": float(
            np.min(metrics["ground_speed_ratio_min"][metrics["ground_speed_steps"] > 0])
            if np.any(metrics["ground_speed_steps"] > 0)
            else np.nan
        ),
        "ground_speed_ratio_episode_min_mean": float(
            np.mean(metrics["ground_speed_ratio_min"][metrics["ground_speed_steps"] > 0])
            if np.any(metrics["ground_speed_steps"] > 0)
            else np.nan
        ),
        "ground_speed_ratio_mean": float(
            np.mean(
                metrics["ground_speed_ratio_sum"] / np.maximum(metrics["ground_speed_steps"], 1)
            )
        ),
        "ground_speed_ratio_min_time_s": float(np.mean(metrics["ground_speed_ratio_min_time_s"])),
        "guidance_saturation_ema_mean": float(np.mean(guidance_saturation_ema)),
        "guidance_saturation_emergency_rate": float(np.mean(guidance_saturation_emergency)),
        "guidance_saturation_emergency_episode_rate": float(
            np.mean(metrics["guidance_saturation_emergency_steps"] > 0)
        ),
        "guidance_route_floor_active_rate": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_active"]))
        ),
        "guidance_route_floor_launch_active_rate": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_launch_active"]))
        ),
        "guidance_route_floor_valid_rate": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_valid"]))
        ),
        "guidance_route_floor_boost_mps_mean": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_boost_mps"]))
        ),
        "guidance_route_floor_ratio_min": float(
            np.min(np.concatenate(metrics["guidance_route_floor_ratio"]))
        ),
        "guidance_route_floor_ratio_mean": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_ratio"]))
        ),
    }
    summary.update(_residual_action_summary(metrics, action_full_scale))
    summary.update(_command_residual_summary(metrics))
    summary.update(_actuator_residual_summary(metrics))
    summary.update(_frozen_residual_summary(metrics))
    summary.update(_ppo_residual_attenuation_summary(metrics))
    if path_enabled:
        path_steps = np.maximum(metrics["path_steps"], 1)
        pre_goal_steps_total = int(np.sum(metrics["pre_goal_path_steps"]))
        pre_goal_rmse = (
            None
            if pre_goal_steps_total <= 0
            else float(
                np.sqrt(np.sum(metrics["pre_goal_three_d_error_sq_sum"]) / pre_goal_steps_total)
            )
        )
        miss_values = metrics["miss_distance_at_plane_m"][
            np.isfinite(metrics["miss_distance_at_plane_m"])
        ]
        entry_mask = metrics["endpoint_entry_reached"]
        overshoot_values = metrics["post_goal_overshoot_m"][
            entry_mask & np.isfinite(metrics["post_goal_overshoot_m"])
        ]
        terminal_steps = int(np.sum(metrics["terminal_longitudinal_steps"]))
        terminal_longitudinal_rmse = (
            None
            if endpoint_stop_enabled or terminal_steps <= 0
            else float(np.sqrt(np.sum(metrics["terminal_longitudinal_sq_sum"]) / terminal_steps))
        )
        summary.update(
            {
                "path_horizontal_rmse_m": float(
                    np.sqrt(np.mean(metrics["horizontal_error_sq_sum"] / path_steps))
                ),
                "path_vertical_rmse_m": float(
                    np.sqrt(np.mean(metrics["vertical_error_sq_sum"] / path_steps))
                ),
                "path_three_d_rmse_m": float(
                    np.sqrt(np.mean(metrics["three_d_error_sq_sum"] / path_steps))
                ),
                "pre_goal_3d_rmse_m": pre_goal_rmse,
                "final_path_progress_m_mean": float(np.mean(metrics["final_progress_m"])),
                "miss_distance_at_plane_m": (
                    None if miss_values.size == 0 else float(np.mean(miss_values))
                ),
                "miss_distance_at_plane_max_m": (
                    None if miss_values.size == 0 else float(np.max(miss_values))
                ),
                "post_goal_overshoot_m": (
                    None
                    if endpoint_stop_enabled or overshoot_values.size == 0
                    else float(np.mean(overshoot_values))
                ),
                "terminal_longitudinal_rmse_m": terminal_longitudinal_rmse,
                "path_completion_ratio_mean": float(
                    np.mean(
                        np.clip(
                            metrics["final_progress_m"] / max(path_length_m, 1e-12),
                            0.0,
                            1.0,
                        )
                    )
                ),
                "alos_projection_horizontal_rmse_m": float(
                    np.sqrt(np.mean(metrics["alos_horizontal_error_sq_sum"] / path_steps))
                ),
                "alos_projection_vertical_rmse_m": float(
                    np.sqrt(np.mean(metrics["alos_vertical_error_sq_sum"] / path_steps))
                ),
                "alos_projection_three_d_rmse_m": float(
                    np.sqrt(np.mean(metrics["alos_three_d_error_sq_sum"] / path_steps))
                ),
                "alos_projection_final_progress_m_mean": float(
                    np.mean(metrics["alos_final_progress_m"])
                ),
            }
        )
    path_action_steps = int(np.sum(metrics["path_action_steps"]))
    if path_action_steps > 0:
        scale = np.asarray(action_full_scale, dtype=np.float64)
        action_abs_sum = np.sum(metrics["path_action_delta_abs_sum"], axis=0)
        summary.update(
            {
                "path_action_delta_l2_mean": float(
                    np.sum(metrics["path_action_delta_l2_sum"]) / path_action_steps
                ),
                "path_action_delta_l2_max": float(np.max(metrics["path_action_delta_l2_max"])),
                "path_action_delta_abs_mean": float(
                    np.sum(action_abs_sum) / (path_action_steps * scale.size)
                ),
                "path_action_delta_rudder_abs_mean_rad": float(
                    action_abs_sum[0] / path_action_steps * scale[0]
                ),
                "path_action_delta_stern_abs_mean_rad": float(
                    action_abs_sum[1] / path_action_steps * scale[1]
                ),
                "path_action_delta_rpm_abs_mean_rpm": float(
                    action_abs_sum[2] / path_action_steps * scale[2]
                ),
            }
        )
    history_action_steps = int(np.sum(metrics["history_action_steps"]))
    if history_action_steps > 0:
        scale = np.asarray(action_full_scale, dtype=np.float64)
        action_abs_sum = np.sum(metrics["history_action_delta_abs_sum"], axis=0)
        summary.update(
            {
                "history_action_delta_l2_mean": float(
                    np.sum(metrics["history_action_delta_l2_sum"]) / history_action_steps
                ),
                "history_action_delta_l2_max": float(
                    np.max(metrics["history_action_delta_l2_max"])
                ),
                "history_action_delta_abs_mean": float(
                    np.sum(action_abs_sum) / (history_action_steps * scale.size)
                ),
                "history_action_delta_rudder_abs_mean_rad": float(
                    action_abs_sum[0] / history_action_steps * scale[0]
                ),
                "history_action_delta_stern_abs_mean_rad": float(
                    action_abs_sum[1] / history_action_steps * scale[1]
                ),
                "history_action_delta_rpm_abs_mean_rpm": float(
                    action_abs_sum[2] / history_action_steps * scale[2]
                ),
            }
        )
    ground_surge_values = _concat_series(metrics.get("ground_surge_values", []))
    ground_speed_values = _concat_series(metrics.get("ground_speed_values", []))
    ground_surge_delta_values = _concat_series(metrics.get("ground_surge_delta_values", []))
    rpm_values = _concat_series(metrics.get("rpm_values", []))
    curvature_values = _concat_series(metrics.get("path_curvature_values", []))
    if ground_surge_values.size:
        summary["ground_surge_mean_mps"] = float(np.mean(ground_surge_values))
        summary["ground_surge_std_mps"] = float(np.std(ground_surge_values))
    if ground_speed_values.size:
        summary["ground_speed_mean_mps"] = float(np.mean(ground_speed_values))
        summary["ground_speed_std_mps"] = float(np.std(ground_speed_values))
        summary["ground_speed_min_mps"] = float(np.min(ground_speed_values))
        summary["ground_speed_max_mps"] = float(np.max(ground_speed_values))
    envelope_steps = int(np.sum(metrics["ground_speed_envelope_steps"]))
    if envelope_steps > 0:
        summary.update(
            {
                "speed_envelope_min_mps": float(speed_envelope_min_mps),
                "speed_envelope_max_mps": float(speed_envelope_max_mps),
                "ground_speed_below_min_rate": float(
                    np.sum(metrics["ground_speed_below_min_steps"]) / envelope_steps
                ),
                "ground_speed_above_max_rate": float(
                    np.sum(metrics["ground_speed_above_max_steps"]) / envelope_steps
                ),
            }
        )
    summary.update(
        {
            "initial_speed_mean_mps": float(np.mean(metrics["initial_speed_mps"])),
            "initial_speed_min_mps": float(np.min(metrics["initial_speed_mps"])),
            "initial_speed_max_mps": float(np.max(metrics["initial_speed_mps"])),
            "initial_speed_nonzero_fraction": float(np.mean(metrics["initial_speed_mps"] > 1e-9)),
            "initial_propeller_rpm_mean": float(np.mean(metrics["initial_propeller_rpm"])),
            "initial_propeller_rpm_min": float(np.min(metrics["initial_propeller_rpm"])),
            "initial_propeller_rpm_max": float(np.max(metrics["initial_propeller_rpm"])),
            "initial_propeller_rpm_nonzero_fraction": float(
                np.mean(metrics["initial_propeller_rpm"] > 1e-9)
            ),
        }
    )
    if ground_surge_delta_values.size:
        summary["ground_surge_delta_abs_p95_mps"] = float(
            np.percentile(np.abs(ground_surge_delta_values), 95)
        )
    if rpm_values.size:
        summary["rpm_sq_mean_rpm2"] = float(np.mean(np.square(rpm_values)))
        summary["rpm_sq_integral_proxy_rpm2_s"] = float(
            np.sum(np.square(rpm_values)) * control_dt_s
        )
    if curvature_values.size:
        summary["abs_curvature_mean_1_per_m"] = float(np.mean(np.abs(curvature_values)))
        summary["turn_step_rate"] = float(
            np.mean(np.abs(curvature_values) >= TURN_CURVATURE_THRESHOLD)
        )
    action_delta_steps = int(np.sum(metrics["action_delta_steps"]))
    if action_delta_steps > 0:
        scale = np.asarray(action_full_scale, dtype=np.float64)
        action_abs_sum = np.sum(metrics["action_delta_abs_sum"], axis=0)
        summary["action_delta_abs_mean"] = float(
            np.sum(action_abs_sum) / (action_delta_steps * scale.size)
        )
        summary["action_delta_l2_mean"] = float(
            np.sum(metrics["action_delta_l2_sum"]) / action_delta_steps
        )
        summary["action_delta_rudder_abs_mean_rad"] = float(
            action_abs_sum[0] / action_delta_steps * scale[0]
        )
        summary["action_delta_stern_abs_mean_rad"] = float(
            action_abs_sum[1] / action_delta_steps * scale[1]
        )
        summary["action_delta_rpm_abs_mean_rpm"] = float(
            action_abs_sum[2] / action_delta_steps * scale[2]
        )
    turn_steps = int(np.sum(metrics["turn_steps"]))
    straight_steps = int(np.sum(metrics["straight_steps"]))
    if turn_steps > 0:
        summary["turn_speed_mean_mps"] = float(np.sum(metrics["turn_speed_sum"]) / turn_steps)
    if straight_steps > 0:
        summary["straight_speed_mean_mps"] = float(
            np.sum(metrics["straight_speed_sum"]) / straight_steps
        )
    if turn_steps > 0 and straight_steps > 0:
        summary["turn_straight_speed_ratio"] = float(
            summary["turn_speed_mean_mps"] / max(summary["straight_speed_mean_mps"], 1e-12)
        )
    goal_mask = metrics["goal_terminations"] > 0
    if bool(goal_mask.any()):
        summary["time_to_goal_s_mean"] = float(
            np.mean(metrics["active_steps"][goal_mask]) * control_dt_s
        )
    summary["episode_time_s_mean"] = float(np.mean(steps) * control_dt_s)
    if bool(metrics["final_goal_distance_set"].any()):
        done_set = metrics["final_goal_distance_set"]
        summary["eta_done_distance_m_mean"] = float(
            np.mean(metrics["final_goal_distance_m"][done_set])
        )
        summary["p_nav_done_distance_m_mean"] = float(
            np.mean(metrics["pnav_final_goal_distance_m"][done_set])
        )
        summary["success_margin_m_mean"] = float(
            np.mean(goal_radius_m - metrics["final_goal_distance_m"][done_set])
        )
    summary["path_overshoot_m_mean"] = float(np.mean(metrics["path_overshoot_m"]))
    success_rows = [
        score_success_contract(
            p_nav_tight_success=bool(metrics["goal_terminations"][index] > 0),
            eta_tight_ever=bool(metrics["eta_goal_capture_seen"][index]),
            eta_tight_at_done=bool(metrics["eta_goal_capture_at_done"][index]),
            success_frame=success_frame,
            eta_done_distance_m=(
                float(metrics["final_goal_distance_m"][index])
                if metrics["final_goal_distance_set"][index]
                else None
            ),
            p_nav_done_distance_m=(
                float(metrics["pnav_final_goal_distance_m"][index])
                if metrics["final_goal_distance_set"][index]
                else None
            ),
            goal_radius_m=float(goal_radius_m),
        )
        for index in range(len(metrics["goal_terminations"]))
    ]
    summary.update(summarize_success_contract(success_rows, success_frame=success_frame))
    return summary
