"""CLI entry point for PC PPO evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

from .args import _parse_args
from .common import _resolve
from .output import _format_summary_value
from .runner import _run
from .validation import (
    _validate_trace_estimator_state,
    _validate_trace_reward_breakdown,
    _validate_trace_sensor_ages,
)


def main() -> int:
    args = _parse_args()
    if args.seed_start < 0 or args.seed_count <= 0:
        raise ValueError("seed start must be non-negative and seed count positive")
    _validate_trace_sensor_ages(args.trace_output, args.trace_sensor_ages)
    _validate_trace_estimator_state(args.trace_output, args.trace_estimator_state)
    _validate_trace_reward_breakdown(args.trace_output, args.trace_reward_breakdown)
    training_config_path = _resolve(args.training_config)
    checkpoint_path = _resolve(args.checkpoint)
    training_config = load_training_config(training_config_path)
    environment_config_path = (
        None if args.environment_config is None else _resolve(args.environment_config)
    )
    guidance_config_path = None if args.guidance_config is None else _resolve(args.guidance_config)
    controller_config_path = (
        None if args.controller_config is None else _resolve(args.controller_config)
    )
    selected_environment_config = (
        _resolve(Path(training_config.environment_config))
        if environment_config_path is None
        else environment_config_path
    )
    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.curriculum_current_speed_range_mps is not None:
        if args.curriculum_stage is None:
            raise ValueError("--curriculum-current-speed-range-mps requires --curriculum-stage")
        speed_min, speed_max = (float(value) for value in args.curriculum_current_speed_range_mps)
        if not (np.isfinite(speed_min) and np.isfinite(speed_max)):
            raise ValueError("curriculum current speed range must be finite")
        if speed_min < 0.0 or speed_max < speed_min:
            raise ValueError("curriculum current speed range requires 0 <= MIN <= MAX")
    if args.curriculum_shear_scale is not None:
        if args.curriculum_stage is None:
            raise ValueError("--curriculum-shear-scale requires --curriculum-stage")
        if not np.isfinite(args.curriculum_shear_scale) or args.curriculum_shear_scale < 0.0:
            raise ValueError("curriculum shear scale must be finite and non-negative")
    summary, metadata, episodes, contract = _run(
        training_config_path=training_config_path,
        checkpoint_path=checkpoint_path,
        environment_config_path=environment_config_path,
        seeds=seeds,
        device=torch.device(args.device),
        dtype=dtype,
        history_ablation=args.history_ablation,
        path_ablation=args.path_ablation,
        action_mode=args.action_mode,
        injected_residual_profile=args.injected_residual_profile,
        allow_stage_override=args.allow_stage_override,
        allow_sensor_override=args.allow_sensor_override,
        allow_actor_observation_override=args.allow_actor_observation_override,
        allow_guidance_rate_limit_override=args.allow_guidance_rate_limit_override,
        allow_diversity_environment_override=(args.allow_diversity_environment_override),
        allow_start_progress_training_override=(args.allow_start_progress_training_override),
        curriculum_stage=args.curriculum_stage,
        disable_curriculum_direct_command=args.disable_curriculum_direct_command,
        curriculum_current_speed_range_mps=(
            None
            if args.curriculum_current_speed_range_mps is None
            else tuple(float(value) for value in args.curriculum_current_speed_range_mps)
        ),
        curriculum_shear_scale=args.curriculum_shear_scale,
        guidance_config_path=guidance_config_path,
        controller_config_path=controller_config_path,
        trace_output=None if args.trace_output is None else _resolve(args.trace_output),
        trace_sensor_ages=args.trace_sensor_ages,
        trace_estimator_state=args.trace_estimator_state,
        trace_reward_breakdown=args.trace_reward_breakdown,
        success_frame=args.success_frame,
        actuator_residual_scale_override=args.actuator_residual_scale_override,
        actuator_residual_channel_gains=args.actuator_residual_channel_gains,
        actuator_residual_scale_near=args.actuator_residual_scale_near,
        actuator_residual_scale_far=args.actuator_residual_scale_far,
        actuator_residual_scale_switch_distance_m=(args.actuator_residual_scale_switch_distance_m),
        actuator_residual_near_channel_gains=args.actuator_residual_near_channel_gains,
        actuator_residual_channel_gain_near_distance_m=(
            args.actuator_residual_channel_gain_near_distance_m
        ),
        actuator_residual_channel_gain_far_distance_m=(
            args.actuator_residual_channel_gain_far_distance_m
        ),
        terminal_gate_enter_m=args.terminal_gate_enter_m,
        terminal_gate_full_m=args.terminal_gate_full_m,
        ppo_residual_attenuation_full_m=args.ppo_residual_attenuation_full_m,
        ppo_residual_attenuation_enter_m=args.ppo_residual_attenuation_enter_m,
    )
    selected_config = load_config(selected_environment_config)
    selected_stage = str(selected_config.pc_stage)
    effective_guidance = selected_config.guidance
    if guidance_config_path is not None:
        effective_guidance = load_config(guidance_config_path).guidance
    effective_controller = selected_config.controller
    if controller_config_path is not None:
        effective_controller = load_config(controller_config_path).controller
    report = {
        "schema_version": "auv_pc_ppo_evaluation.v1",
        "training_config": str(training_config_path),
        "environment_config": str(selected_environment_config),
        "curriculum_stage": args.curriculum_stage,
        "curriculum_current_speed_range_override_mps": (
            None
            if args.curriculum_current_speed_range_mps is None
            else [float(value) for value in args.curriculum_current_speed_range_mps]
        ),
        "curriculum_shear_scale_override": args.curriculum_shear_scale,
        "curriculum_direct_commands_enabled": (not args.disable_curriculum_direct_command),
        "environment_config_override": (
            None if environment_config_path is None else str(environment_config_path)
        ),
        "guidance_config": (None if guidance_config_path is None else str(guidance_config_path)),
        "controller_config": (
            None if controller_config_path is None else str(controller_config_path)
        ),
        "guidance_rate_limit_override": args.allow_guidance_rate_limit_override,
        "diversity_environment_override": (args.allow_diversity_environment_override),
        "start_progress_training_override": (args.allow_start_progress_training_override),
        "guidance_command_rate_limit_enabled": (effective_guidance.command_rate_limit_enabled),
        "guidance_pitch_rate_limit_rad_s": (effective_guidance.pitch_rate_limit_rad_s),
        "guidance_heading_rate_limit_rad_s": (effective_guidance.heading_rate_limit_rad_s),
        "guidance_speed_rate_limit_mps2": (effective_guidance.speed_rate_limit_mps2),
        "guidance_terminal_braking_enabled": (effective_guidance.terminal_braking_enabled),
        "guidance_terminal_speed_ratio": effective_guidance.terminal_speed_ratio,
        "guidance_vertical_gain": effective_guidance.vertical_gain,
        "guidance_vertical_lookahead_m": effective_guidance.vertical_lookahead_m,
        "controller_pitch_gain": effective_controller.pitch_gain,
        "controller_pitch_rate_gain": effective_controller.pitch_rate_gain,
        "trace_output": (None if args.trace_output is None else str(_resolve(args.trace_output))),
        "trace_reward_breakdown": args.trace_reward_breakdown,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": metadata,
        "plant_randomization": selected_config.plant_randomization.as_dict(),
        "seeds": list(seeds),
        "actor_decision_stride": training_config.actor_decision_stride,
        "history_ablation": args.history_ablation,
        "path_ablation": args.path_ablation,
        "action_mode": args.action_mode,
        "injected_residual_profile": {
            "name": args.injected_residual_profile[0],
            "const": (
                list(args.injected_residual_profile[1])
                if args.injected_residual_profile[1] is not None
                else None
            ),
        },
        "actuator_residual_scale_override": args.actuator_residual_scale_override,
        "actuator_residual_channel_gains": (
            list(args.actuator_residual_channel_gains)
            if args.actuator_residual_channel_gains is not None
            else [1.0, 1.0, 1.0]
        ),
        "actuator_residual_scale_near": args.actuator_residual_scale_near,
        "actuator_residual_scale_far": args.actuator_residual_scale_far,
        "actuator_residual_scale_switch_distance_m": (
            args.actuator_residual_scale_switch_distance_m
        ),
        "actuator_residual_scale_schedule_enabled": (args.actuator_residual_scale_near is not None),
        "actuator_residual_channel_gains_near": (
            list(args.actuator_residual_near_channel_gains)
            if args.actuator_residual_near_channel_gains is not None
            else None
        ),
        "actuator_residual_channel_gain_near_distance_m": (
            args.actuator_residual_channel_gain_near_distance_m
        ),
        "actuator_residual_channel_gain_far_distance_m": (
            args.actuator_residual_channel_gain_far_distance_m
        ),
        "actuator_residual_channel_gain_schedule_enabled": (
            args.actuator_residual_near_channel_gains is not None
        ),
        "terminal_gate_enter_m": (
            args.terminal_gate_enter_m if args.action_mode == "gated-residual" else None
        ),
        "terminal_gate_full_m": (
            args.terminal_gate_full_m if args.action_mode == "gated-residual" else None
        ),
        "ppo_residual_attenuation_enabled": (args.action_mode == "attenuated-residual"),
        "ppo_residual_attenuation_full_m": (
            args.ppo_residual_attenuation_full_m
            if args.action_mode == "attenuated-residual"
            else None
        ),
        "ppo_residual_attenuation_enter_m": (
            args.ppo_residual_attenuation_enter_m
            if args.action_mode == "attenuated-residual"
            else None
        ),
        "pc_action_interface": metadata.get("pc", {}).get("action_interface"),
        "success_frame": args.success_frame,
        "stage_override": metadata["pc_stage"] != selected_stage,
        "sensor_override": contract["sensor_override"],
        "actor_observation_source": contract["actor_observation_source"],
        "actor_observation_override": contract["actor_observation_override"],
        "endpoint_stop_enabled": contract["endpoint_stop_enabled"],
        "base_sensor": contract["base_sensor"],
        "override_sensor": contract["override_sensor"],
        "metric_definitions": {
            "path_*": "projection of true eta position onto the reference path",
            "alos_projection_*": "ALOS projection of p_nav; diagnostic only",
            "nav_position_error_*": "p_nav minus true NED position",
            "dvl_bottom_outage_rate": "fraction of control steps with bottom-lock invalid",
            "current_estimate_valid_rate": "fraction of control steps with a valid current estimate",
            "guidance_current_valid_rate": "fraction of steps whose deployment current estimate passed the crab gate",
            "terminal_speed_ratio_mean": "mean command speed divided by episode cruise speed on terminal-capture steps",
            "guidance_route_speed_ratio_min": "minimum command speed divided by episode cruise speed over all active control steps",
            "ground_speed_ratio_min": "minimum measured ground-speed magnitude divided by episode cruise speed over all active control steps",
            "ground_speed_ratio_mean": "mean measured ground-speed magnitude divided by episode cruise speed over all active control steps",
            "ground_route_speed_ratio_min": "minimum measured route-progress speed divided by episode cruise speed over all active control steps",
            "ground_route_speed_ratio_episode_min_mean": "mean over episodes of each episode's minimum measured route-progress speed divided by its cruise speed",
            "ground_route_speed_ratio_mean": "mean measured route-progress speed divided by episode cruise speed over all active control steps",
            "guidance_crab_deg_mean": "signed crab heading offset requested by ALOS-R",
            "guidance_saturation_ema_mean": "EMA of material actuator saturation used by the speed governor",
            "guidance_saturation_emergency_rate": "fraction of steps with sustained saturation protection active",
            "usbl_outlier_rate": "fraction of control steps flagged as a raw USBL outlier",
            "pre_goal_3d_rmse_m": "true-position 3D path RMSE before the goal radius or terminal tangent plane is reached",
            "goal_plane_miss_rate": "fraction of episodes terminated by missing the terminal goal plane",
            "miss_distance_at_plane_m": "true or estimated goal distance at a goal-plane-miss termination under the selected terminal protocol",
            "post_goal_overshoot_m": "mean positive terminal tangent offset after endpoint entry; legacy termination only",
            "terminal_longitudinal_rmse_m": "RMSE of signed terminal tangent offset after endpoint entry; legacy termination only",
            "success_pnav_tight": "deployment success: p_nav tight-radius GOAL termination",
            "success_eta_tight_ever": "research primary: true eta ever entered tight radius",
            "success_eta_tight_at_done": "secondary: true eta at done remained inside tight radius",
        },
        "metrics": summary,
        "episodes": episodes,
    }
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PC20 PPO closed-loop evaluation",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- environment config: `{report['environment_config']}`",
        f"- seeds: `{len(seeds)}`",
        "- `path_*`: true `eta` position; `alos_projection_*`: navigation-position diagnostic",
    ]
    lines.extend(f"- `{name}`: `{_format_summary_value(value)}`" for name, value in summary.items())
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
