"""Closed-loop evaluation orchestration for PC PPO."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.geometry import (
    body_to_world,
    ground_velocity_body_for_contract,
    quaternion_from_euler,
)
from auv_lowlevel.batch.pc import PCTerminalKind
from auv_lowlevel.batch.pc_actuator_residual import BatchedPCActuatorResidualEnv
from auv_lowlevel.batch.pc_command_residual import (
    BatchedPCCommandResidualEnv,
    smooth_terminal_gate,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_frozen_residual_vec_env import BatchTorchPCFrozenBCResidualVecEnv
from auv_lowlevel.batch.pc_history_ablation import ablate_pc_history, ablate_pc_path
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import PC_PATH_DIM
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import curriculum_episode_metadata
from auv_lowlevel.outcome_contract import DEFAULT_SUCCESS_FRAME, validate_success_frame
from auv_lowlevel.ppo_config import load_training_config, resolve_pc_action_interface

from .common import TURN_CURVATURE_THRESHOLD, _resolve
from .episodes import _episode_records
from .metrics import _new_episode_metrics
from .policy import _build_curriculum_sampler, _build_policy
from .summary import _summary
from .validation import (
    _final_path_blocks,
    _ppo_residual_attenuation,
    _record_trace_estimator_state,
    _record_trace_sensor_ages,
    _reward_breakdown_trace_fields,
    _stage_override_allowed,
    _uses_frozen_residual_wrapper,
    _validate_actuator_channel_gain_schedule,
    _validate_actuator_residual_scale_schedule,
    _validate_controller_pitch_override,
    _validate_environment_override,
)


def _run(
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    environment_config_path: Path | None,
    seeds: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    history_ablation: str = "none",
    path_ablation: str = "none",
    action_mode: str = "policy",
    injected_residual_profile: tuple[str, tuple[float, float, float] | None] = (
        "zero",
        None,
    ),
    allow_stage_override: bool = False,
    allow_sensor_override: bool = False,
    allow_actor_observation_override: bool = False,
    allow_guidance_rate_limit_override: bool = False,
    allow_diversity_environment_override: bool = False,
    allow_start_progress_training_override: bool = False,
    curriculum_stage: str | None = None,
    disable_curriculum_direct_command: bool = False,
    curriculum_current_speed_range_mps: tuple[float, float] | None = None,
    curriculum_shear_scale: float | None = None,
    guidance_config_path: Path | None = None,
    controller_config_path: Path | None = None,
    trace_output: Path | None = None,
    trace_sensor_ages: bool = False,
    trace_estimator_state: bool = False,
    trace_reward_breakdown: bool = False,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
    actuator_residual_scale_override: float | None = None,
    actuator_residual_channel_gains: tuple[float, float, float] | None = None,
    actuator_residual_scale_near: float | None = None,
    actuator_residual_scale_far: float | None = None,
    actuator_residual_scale_switch_distance_m: float | None = None,
    actuator_residual_near_channel_gains: tuple[float, float, float] | None = None,
    actuator_residual_channel_gain_near_distance_m: float | None = None,
    actuator_residual_channel_gain_far_distance_m: float | None = None,
    terminal_gate_enter_m: float = 20.0,
    terminal_gate_full_m: float = 5.0,
    ppo_residual_attenuation_full_m: float = 1.25,
    ppo_residual_attenuation_enter_m: float = 3.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    success_frame = validate_success_frame(success_frame)
    _validate_actuator_residual_scale_schedule(
        near=actuator_residual_scale_near,
        far=actuator_residual_scale_far,
        switch_distance_m=actuator_residual_scale_switch_distance_m,
        constant_override=actuator_residual_scale_override,
    )
    _validate_actuator_channel_gain_schedule(
        near_gains=actuator_residual_near_channel_gains,
        near_distance_m=actuator_residual_channel_gain_near_distance_m,
        far_distance_m=actuator_residual_channel_gain_far_distance_m,
    )
    if action_mode not in {
        "policy",
        "frozen-base",
        "zero-command-residual",
        "zero-actuator-residual",
        "zero-residual",
        "gated-residual",
        "attenuated-residual",
        "injected-residual",
        "alos",
    }:
        raise ValueError(f"unsupported action mode: {action_mode}")
    if action_mode == "attenuated-residual":
        if (
            not np.isfinite(ppo_residual_attenuation_full_m)
            or ppo_residual_attenuation_full_m <= 0.0
            or not np.isfinite(ppo_residual_attenuation_enter_m)
            or ppo_residual_attenuation_enter_m <= ppo_residual_attenuation_full_m
        ):
            raise ValueError("PPO attenuation requires finite positive full_m < enter_m")
    if (
        not np.isfinite(terminal_gate_enter_m)
        or terminal_gate_enter_m <= 0.0
        or not np.isfinite(terminal_gate_full_m)
        or terminal_gate_full_m <= 0.0
        or terminal_gate_full_m >= terminal_gate_enter_m
    ):
        raise ValueError(
            "terminal gate requires finite positive enter_m/full_m with full_m < enter_m"
        )
    training = load_training_config(training_config_path)
    if training.pc is None or training.history is None:
        raise ValueError("evaluation requires a PC training config with history")
    resolved_environment_config = (
        environment_config_path
        if environment_config_path is not None
        else _resolve(Path(training.environment_config))
    )
    config = load_config(resolved_environment_config)
    base_config = None
    sensor_override = False
    if environment_config_path is not None:
        base_config = load_config(_resolve(Path(training.environment_config)))
        sensor_override = bool(allow_sensor_override and config.sensors != base_config.sensors)
        _validate_environment_override(
            base_config,
            config,
            allow_sensor_override=allow_sensor_override,
            allow_actor_observation_override=allow_actor_observation_override,
            allow_guidance_rate_limit_override=allow_guidance_rate_limit_override,
            allow_diversity_environment_override=(allow_diversity_environment_override),
            allow_start_progress_training_override=(allow_start_progress_training_override),
        )
    if config.pc_control_mode.startswith("path_") and training.pc.path_dim != PC_PATH_DIM:
        raise ValueError("ALOS-free path_* PC control requires pc.path_dim=PC_PATH_DIM (8)")
    if guidance_config_path is not None:
        config = replace(config, guidance=load_config(guidance_config_path).guidance)
    if controller_config_path is not None:
        controller_override = load_config(controller_config_path).controller
        _validate_controller_pitch_override(config.controller, controller_override)
        config = replace(config, controller=controller_override)
    if config.arm != "PC" or config.policy_mode != "pure_controller":
        raise ValueError("environment config is not a PC pure-controller config")
    effective_current_override = curriculum_current_speed_range_mps
    if effective_current_override is None:
        effective_current_override = config.curriculum.current_speed_range_override_mps
    effective_shear_override = curriculum_shear_scale
    if effective_shear_override is None:
        effective_shear_override = config.curriculum.shear_scale_override
    curriculum_sampler = _build_curriculum_sampler(
        curriculum_stage,
        current_speed_range_mps=effective_current_override,
        shear_scale=effective_shear_override,
        current_variation_speed_amplitude_mps=(
            0.0
            if config.current.time_variation is None
            else config.current.time_variation.speed_amplitude_mps
        ),
    )
    inner = BatchedPCLowLevelEnv(
        config,
        num_envs=len(seeds),
        device=device.type,
        dtype=dtype,
        validate_finite=True,
        path_dim=training.pc.path_dim,
        curriculum_sampler=curriculum_sampler,
        curriculum_direct_commands_enabled=not disable_curriculum_direct_command,
    )
    resolved_action_interface = resolve_pc_action_interface(
        training.pc.action_interface,
        command_residual_enabled=config.command_residual.enabled,
    )
    if resolved_action_interface == "command":
        inner = BatchedPCCommandResidualEnv(
            inner,
            scale_heading_rad=config.command_residual.scale_heading_rad,
            scale_pitch_rad=config.command_residual.scale_pitch_rad,
            scale_speed_mps=config.command_residual.scale_speed_mps,
            rate_heading_rad_s=config.command_residual.rate_heading_rad_s,
            rate_pitch_rad_s=config.command_residual.rate_pitch_rad_s,
            rate_speed_mps2=config.command_residual.rate_speed_mps2,
            controller_state=config.command_residual.controller_state,
            terminal_gate_enabled=config.command_residual.terminal_gate_enabled,
            terminal_gate_enter_m=config.command_residual.terminal_gate_enter_m,
            terminal_gate_full_m=config.command_residual.terminal_gate_full_m,
        )
    elif resolved_action_interface == "actuator":
        if actuator_residual_scale_near is not None and action_mode not in {
            "policy",
            "gated-residual",
        }:
            raise ValueError(
                "actuator residual scale schedule requires policy or gated-residual action mode"
            )
        if action_mode == "gated-residual":
            if training.pc.actuator_terminal_gate_enabled:
                raise ValueError(
                    "gated-residual eval requires training.pc.actuator_terminal_gate_enabled=false"
                )
        residual_scale = training.pc.actuator_residual_scale
        channel_gains = (
            tuple(training.pc.actuator_residual_channel_gains)
            if actuator_residual_channel_gains is None
            else tuple(float(value) for value in actuator_residual_channel_gains)
        )
        if len(channel_gains) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in channel_gains
        ):
            raise ValueError(
                "actuator residual channel gains must contain three finite positive values"
            )
        if actuator_residual_scale_override is not None:
            if (
                not np.isfinite(actuator_residual_scale_override)
                or actuator_residual_scale_override <= 0.0
            ):
                raise ValueError("actuator residual scale override must be finite and positive")
            residual_scale = float(actuator_residual_scale_override)
        inner = BatchedPCActuatorResidualEnv(
            inner,
            residual_scale=residual_scale,
            residual_channel_gains=channel_gains,
            max_action=training.pc.actuator_max_action,
            controller_state="actor",
            terminal_gate_enabled=training.pc.actuator_terminal_gate_enabled,
            terminal_gate_enter_m=training.pc.actuator_terminal_gate_enter_m,
            terminal_gate_full_m=training.pc.actuator_terminal_gate_full_m,
            residual_scale_near=actuator_residual_scale_near,
            residual_scale_far=actuator_residual_scale_far,
            residual_scale_switch_distance_m=(actuator_residual_scale_switch_distance_m),
            residual_channel_gains_near=actuator_residual_near_channel_gains,
            residual_channel_gain_near_distance_m=(actuator_residual_channel_gain_near_distance_m),
            residual_channel_gain_far_distance_m=(actuator_residual_channel_gain_far_distance_m),
        )
    elif resolved_action_interface != "absolute":
        raise ValueError("unsupported resolved PC action interface")
    if actuator_residual_scale_near is not None and resolved_action_interface != "actuator":
        raise ValueError("actuator residual scale schedule requires the actuator action interface")
    if actuator_residual_channel_gains is not None and resolved_action_interface != "actuator":
        raise ValueError("actuator residual channel gains require the actuator action interface")
    if actuator_residual_near_channel_gains is not None and resolved_action_interface != "actuator":
        raise ValueError(
            "actuator residual channel gain schedule requires the actuator action interface"
        )
    if action_mode == "zero-command-residual" and resolved_action_interface != "command":
        raise ValueError(
            "zero-command-residual requires the checkpoint's command residual interface"
        )
    if action_mode == "zero-actuator-residual" and resolved_action_interface != "actuator":
        raise ValueError(
            "zero-actuator-residual requires the checkpoint's actuator residual interface"
        )
    frozen_residual_output = (
        training.base_mode == "frozen_pc_bc_residual" and training.base_output_mode == "residual"
    )
    if action_mode == "zero-residual" and not frozen_residual_output:
        raise ValueError("zero-residual requires base_policy.output_mode='residual'")
    if action_mode == "attenuated-residual" and not frozen_residual_output:
        raise ValueError("attenuated-residual requires frozen BC residual output")
    if action_mode == "injected-residual" and not frozen_residual_output:
        raise ValueError("injected-residual requires frozen BC residual output")
    path_blocks = _final_path_blocks(training)
    path_enabled = str(config.pc_stage) in {"C2", "C3", "C4", "C5", "C6"}
    endpoint_stop_enabled = bool(config.termination.pc_goal_plane_miss_termination_enabled)
    env = BatchTorchPCVecEnv(
        inner,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        short_stride=training.history.short_stride,
        long_stride=training.history.long_stride,
        actor_command_dim=training.pc.actor_command_dim,
        critic_command_dim=training.pc.critic_command_dim,
        history_path_dim=training.pc.history_path_dim,
        actor_current_encoder_enabled=training.pc.actor_current_encoder_enabled,
        normalization=PCNormalization(
            position_scale_m=config.normalization.position_scale_m,
            attitude_scale_rad=config.normalization.attitude_scale_rad,
            angular_rate_scale_rad_s=config.normalization.angular_rate_scale_rad_s,
            surge_speed_scale_mps=config.normalization.surge_speed_scale_mps,
            sway_heave_speed_scale_mps=config.normalization.sway_heave_speed_scale_mps,
            current_scale_mps=config.normalization.current_scale_mps,
        ),
        current_frame_dropout_prob=0.0,
        path_blocks=path_blocks,
    )
    policy, metadata = _build_policy(
        training_config_path,
        checkpoint_path,
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        device=device,
    )
    checkpoint_action_interface = metadata.get("pc", {}).get("action_interface")
    if (
        checkpoint_action_interface is not None
        and str(checkpoint_action_interface) != resolved_action_interface
    ):
        raise ValueError(
            "checkpoint and training config disagree on the PC action interface: "
            f"{checkpoint_action_interface!r} != {resolved_action_interface!r}"
        )
    checkpoint_bias = metadata.get("pc", {}).get("base_residual_bias")
    if checkpoint_bias is not None and not np.allclose(
        np.asarray(checkpoint_bias, dtype=np.float64),
        np.asarray(training.pc.base_residual_bias, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "checkpoint and training config disagree on pc.base_residual_bias: "
            f"{checkpoint_bias!r} != {list(training.pc.base_residual_bias)!r}"
        )
    frozen_residual_env: BatchTorchPCFrozenBCResidualVecEnv | None = None
    if _uses_frozen_residual_wrapper(
        base_mode=training.base_mode,
        base_output_mode=training.base_output_mode,
        action_mode=action_mode,
    ):
        if resolved_action_interface not in {"absolute", "actuator"}:
            raise ValueError(
                "frozen BC residual output requires the resolved absolute or "
                f"actuator action interface, got {resolved_action_interface!r}"
            )
        assert training.pc is not None
        frozen_residual_env = BatchTorchPCFrozenBCResidualVecEnv(
            env,
            base_action_fn=policy.base_action,
            terminal_gate_enabled=training.pc.residual_terminal_gate_enabled,
            terminal_gate_enter_m=training.pc.residual_terminal_gate_enter_m,
            terminal_gate_full_m=training.pc.residual_terminal_gate_full_m,
            rate_aware_residual_enabled=training.pc.rate_aware_residual_enabled,
            ppo_rate_fraction=training.pc.ppo_rate_fraction,
            ppo_fin_gain=training.pc.ppo_fin_gain,
            ppo_rpm_gain=training.pc.ppo_rpm_gain,
            base_residual_bias=training.pc.base_residual_bias,
        )
        if action_mode == "gated-residual":
            frozen_residual_env.set_terminal_gate(
                enabled=True,
                enter_m=terminal_gate_enter_m,
                full_m=terminal_gate_full_m,
            )
        env = frozen_residual_env
    actor_decision_stride = training.actor_decision_stride
    if actor_decision_stride <= 0:
        raise ValueError("actor_decision_stride must be positive")
    expected_normalization = PCNormalization(
        position_scale_m=config.normalization.position_scale_m,
        attitude_scale_rad=config.normalization.attitude_scale_rad,
        angular_rate_scale_rad_s=config.normalization.angular_rate_scale_rad_s,
        surge_speed_scale_mps=config.normalization.surge_speed_scale_mps,
        sway_heave_speed_scale_mps=config.normalization.sway_heave_speed_scale_mps,
        current_scale_mps=config.normalization.current_scale_mps,
    ).as_dict()
    if metadata["normalization"] != expected_normalization:
        raise ValueError("environment override changes PC normalization scales")
    stage_override = metadata["pc_stage"] != str(config.pc_stage)
    if stage_override and not _stage_override_allowed(
        metadata["pc_stage"],
        str(config.pc_stage),
        allow_stage_override=allow_stage_override,
    ):
        raise ValueError("environment override changes the PC stage")
    if metadata["actuator_envelope"] != config.actuator.envelope:
        raise ValueError("environment override changes the actuator envelope")
    expected_action_full_scale = (
        config.actuator.policy_max_rudder_angle_rad,
        config.actuator.policy_max_stern_plane_angle_rad,
        config.actuator.policy_max_propeller_rpm,
    )
    if any(
        not np.isclose(actual, expected, rtol=0.0, atol=1e-12)
        for actual, expected in zip(
            metadata["action_full_scale"], expected_action_full_scale, strict=True
        )
    ):
        raise ValueError("environment override changes the policy action scale")
    observations, _ = env.reset(seeds)
    assert inner.eta is not None
    assert inner.nu is not None
    assert inner.actuator is not None
    initial_heading_rad = inner.eta[:, 5].detach().cpu().numpy().copy()
    true_projection_hint = inner.path.project(inner.eta[:, :3]).segment_index
    metrics = _new_episode_metrics(len(seeds))
    metrics["initial_speed_mps"] = inner.nu[:, 0].detach().cpu().numpy().copy()
    metrics["initial_propeller_rpm"] = inner.actuator[:, 2].detach().cpu().numpy().copy()
    metrics["initial_speed_ratio"] = (
        inner._episode_initial_speed_ratio.detach().cpu().numpy().copy()
    )
    metrics["initial_propeller_rpm_ratio"] = (
        inner._episode_initial_propeller_rpm_ratio.detach().cpu().numpy().copy()
    )
    speed_envelope_min_mps = float(inner.vehicle.min_speed_mps)
    speed_envelope_max_mps = float(inner.vehicle.max_speed_mps)
    trace: dict[str, list[np.ndarray]] | None = {} if trace_output is not None else None

    def record_trace(name: str, value: torch.Tensor | np.ndarray) -> None:
        if trace is None:
            return
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        trace.setdefault(name, []).append(array[active_numpy])

    held_action = torch.zeros((len(seeds), 3), dtype=torch.float32, device=device)
    ppo_residual_attenuation = None
    ppo_residual_attenuation_distance = None
    for step_index in range(config.plant.max_episode_steps):
        active = ~metrics["done"]
        if not bool(active.any()):
            break
        active_tensor = torch.as_tensor(active, device=device, dtype=torch.bool)
        decision_step = step_index % actor_decision_stride == 0
        ppo_residual_attenuation = None
        ppo_residual_attenuation_distance = None
        base_action = None
        residual_action = None
        clean_action = None
        history_clean_action = None
        if decision_step:
            with torch.inference_mode():
                history_observations = ablate_pc_history(
                    observations, policy.actor_layout, history_ablation
                )
                policy_observations = ablate_pc_path(
                    history_observations, policy.actor_layout, path_ablation
                )
                base_action = policy.base_action(policy_observations)
                if action_mode == "alos":
                    held_action = torch.zeros_like(held_action)
                    residual_action = held_action
                    clean_action = held_action
                    history_clean_action = held_action
                elif frozen_residual_output:
                    if action_mode in {"zero-residual", "frozen-base"}:
                        held_action = torch.zeros_like(held_action)
                        residual_action = held_action
                        clean_action = held_action
                        history_clean_action = held_action
                    elif action_mode == "gated-residual":
                        residual_action = policy.residual_action(policy_observations)
                        held_action = residual_action
                        clean_action = held_action
                        history_clean_action = held_action
                    elif action_mode == "policy":
                        residual_action = policy.residual_action(policy_observations)
                        held_action = policy.act_inference(policy_observations)
                        clean_action = (
                            policy.act_inference(history_observations)
                            if path_ablation != "none"
                            else held_action
                        )
                        history_clean_action = (
                            policy.act_inference(
                                ablate_pc_path(observations, policy.actor_layout, path_ablation)
                            )
                            if history_ablation != "none"
                            else held_action
                        )
                    elif action_mode == "injected-residual":
                        profile_name, profile_const = injected_residual_profile
                        if profile_name == "zero":
                            residual_action = torch.zeros_like(held_action)
                        elif profile_name == "const":
                            assert profile_const is not None
                            residual_action = (
                                torch.as_tensor(
                                    profile_const,
                                    device=device,
                                    dtype=held_action.dtype,
                                )
                                .expand(held_action.shape[0], 3)
                                .clone()
                            )
                        else:
                            ppo_residual = policy.residual_action(policy_observations)
                            if profile_name == "ppo":
                                residual_action = ppo_residual
                            elif profile_name == "neg-ppo":
                                residual_action = -ppo_residual
                            elif profile_name == "ppo-gain":
                                assert profile_const is not None
                                residual_action = profile_const[0] * ppo_residual
                            else:
                                raise ValueError(
                                    f"unsupported injected residual profile {profile_name!r}"
                                )
                        held_action = residual_action
                        clean_action = held_action
                        history_clean_action = held_action
                    elif action_mode == "attenuated-residual":
                        residual_action = policy.residual_action(policy_observations)
                        gate_observation = inner.observation
                        position = gate_observation.actor_state[:, :3].to(
                            device=device, dtype=residual_action.dtype
                        )
                        goal = inner.path.goal_ned_m_for(inner.num_envs).to(
                            device=device, dtype=residual_action.dtype
                        )
                        attenuation_distance = torch.linalg.norm(goal - position, dim=1)
                        attenuation = _ppo_residual_attenuation(
                            attenuation_distance,
                            full_m=ppo_residual_attenuation_full_m,
                            enter_m=ppo_residual_attenuation_enter_m,
                        )
                        held_action = residual_action * attenuation[:, None]
                        ppo_residual_attenuation = attenuation
                        ppo_residual_attenuation_distance = attenuation_distance
                        clean_action = held_action
                        history_clean_action = held_action
                    else:
                        raise ValueError(
                            f"action mode {action_mode!r} is not valid for "
                            "frozen BC residual output"
                        )
                elif action_mode in {
                    "zero-command-residual",
                    "zero-actuator-residual",
                }:
                    held_action = torch.zeros_like(held_action)
                    residual_action = held_action
                    clean_action = held_action
                    history_clean_action = held_action
                elif action_mode == "frozen-base":
                    if resolved_action_interface not in {"absolute", "actuator"}:
                        raise ValueError(
                            "frozen-base evaluation requires the checkpoint's "
                            "absolute or actuator interface"
                        )
                    if base_action is None:
                        raise ValueError(
                            "frozen-base evaluation requires an enabled frozen BC actor"
                        )
                    held_action = base_action
                    residual_action = torch.zeros_like(base_action)
                    clean_action = (
                        policy.base_action(history_observations)
                        if path_ablation != "none"
                        else held_action
                    )
                    history_clean_action = (
                        policy.base_action(
                            ablate_pc_path(observations, policy.actor_layout, path_ablation)
                        )
                        if history_ablation != "none"
                        else held_action
                    )
                elif action_mode == "gated-residual":
                    if base_action is None:
                        raise ValueError(
                            "gated-residual evaluation requires an enabled frozen BC actor"
                        )
                    residual_action = policy.residual_action(policy_observations)
                    gate_observation = inner.observation
                    position = gate_observation.actor_state[:, :3].to(
                        device=device, dtype=held_action.dtype
                    )
                    goal = inner.path.goal_ned_m_for(inner.num_envs).to(
                        device=device, dtype=held_action.dtype
                    )
                    distance = torch.linalg.norm(goal - position, dim=1)
                    gate = smooth_terminal_gate(
                        distance,
                        enter_m=terminal_gate_enter_m,
                        full_m=terminal_gate_full_m,
                    )
                    held_action = base_action + residual_action * gate[:, None]
                    clean_action = held_action
                    history_clean_action = held_action
                else:
                    residual_action = policy.residual_action(policy_observations)
                    held_action = policy.act_inference(policy_observations)
                    clean_action = (
                        policy.act_inference(history_observations)
                        if path_ablation != "none"
                        else held_action
                    )
                    history_clean_action = (
                        policy.act_inference(
                            ablate_pc_path(observations, policy.actor_layout, path_ablation)
                        )
                        if history_ablation != "none"
                        else held_action
                    )
        action = torch.where(active_tensor[:, None], held_action, torch.zeros_like(held_action))
        with torch.inference_mode():
            observations, reward, done, infos = env.step(action, auto_reset=False)

        assert inner.eta is not None and inner.nu is not None
        command = inner.observation.command
        yaw_error = (
            torch.remainder(command.heading_rad - inner.eta[:, 5] + torch.pi, 2.0 * torch.pi)
            - torch.pi
        )
        pitch_error = command.pitch_rad - inner.eta[:, 4]
        assert inner.current_ned_mps is not None
        ground_velocity = ground_velocity_body_for_contract(
            inner.eta,
            inner.nu,
            inner.current_ned_mps,
            config.plant.velocity_contract,
        )
        speed_error = command.speed_mps - ground_velocity[:, 0]
        roll = inner.eta[:, 3].abs()
        roll_rate = inner.nu[:, 3].abs()
        projection = infos["projection"]
        assert inner.sensor_output is not None
        sensor = inner.sensor_output
        active_numpy = active
        if ppo_residual_attenuation is not None:
            attenuation_indices = np.flatnonzero(active_numpy)
            attenuation_values = ppo_residual_attenuation.detach().cpu().numpy()
            attenuation_distance_values = ppo_residual_attenuation_distance.detach().cpu().numpy()
            metrics["ppo_residual_attenuation_values"].append(attenuation_values[active_numpy])
            metrics["ppo_residual_attenuation_distance_values"].append(
                attenuation_distance_values[active_numpy]
            )
            metrics["ppo_residual_attenuation_indices"].append(attenuation_indices)
        if infos.get("command_residual_applied") is not None:
            command_delta = infos["command_residual_applied"].detach().cpu().numpy()
            active_indices = np.flatnonzero(active_numpy)
            metrics["command_residual_applied_values"].append(command_delta[active_numpy])
            metrics["command_residual_applied_indices"].append(active_indices)
            metrics["command_residual_scale_clip_steps"][active_numpy] += (
                infos["command_residual_scale_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["command_residual_rate_clip_steps"][active_numpy] += (
                infos["command_residual_rate_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["command_residual_steps"][active_numpy] += 1
        if infos.get("actuator_residual_applied") is not None:
            actuator_delta = infos["actuator_residual_applied"].detach().cpu().numpy()
            active_indices = np.flatnonzero(active_numpy)
            metrics["actuator_residual_applied_values"].append(actuator_delta[active_numpy])
            metrics["actuator_residual_applied_indices"].append(active_indices)
            if infos.get("actuator_residual_gate") is not None:
                actuator_gate = infos["actuator_residual_gate"].detach().cpu().numpy()
                metrics["actuator_residual_gate_values"].append(actuator_gate[active_numpy])
                metrics["actuator_residual_gate_indices"].append(active_indices)
            metrics["actuator_residual_scale_clip_steps"][active_numpy] += (
                infos["actuator_residual_scale_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["actuator_residual_magnitude_clip_steps"][active_numpy] += (
                infos["actuator_residual_magnitude_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["actuator_residual_rate_clip_steps"][active_numpy] += (
                infos["actuator_residual_rate_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["actuator_residual_steps"][active_numpy] += 1
        if infos.get("frozen_residual_gate") is not None:
            frozen_indices = np.flatnonzero(active_numpy)
            frozen_gate = infos["frozen_residual_gate"].detach().cpu().numpy()
            metrics["frozen_residual_gate_values"].append(frozen_gate[active_numpy])
            metrics["frozen_residual_gate_indices"].append(frozen_indices)
            metrics["frozen_residual_clip_steps"][active_numpy] += (
                infos["frozen_residual_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["frozen_residual_steps"][active_numpy] += 1
        goal_ned_m = inner.path.goal_ned_m_for(len(seeds))
        goal_tangent_ned = inner.path.goal_tangent_ned_for(len(seeds))
        position_source = config.termination.position_source
        if position_source == "eta":
            termination_position_ned_m = inner.eta[:, :3]
        elif position_source == "p_nav":
            termination_position_ned_m = sensor.p_nav
        else:
            termination_position_ned_m = sensor.p_nav if endpoint_stop_enabled else inner.eta[:, :3]
        endpoint_goal_distance_m = (
            torch.linalg.norm(termination_position_ned_m - goal_ned_m, dim=1).cpu().numpy()
        )
        signed_goal_tangent_offset_m = (
            ((termination_position_ned_m - goal_ned_m) * goal_tangent_ned).sum(dim=1).cpu().numpy()
        )
        signed_goal_tangent_offset_eta_m = (
            ((inner.eta[:, :3] - goal_ned_m) * goal_tangent_ned).sum(dim=1).cpu().numpy()
        )
        eta_goal_distance_m = torch.linalg.norm(inner.eta[:, :3] - goal_ned_m, dim=1).cpu().numpy()
        pnav_goal_distance_m = torch.linalg.norm(sensor.p_nav - goal_ned_m, dim=1).cpu().numpy()
        endpoint_now = (endpoint_goal_distance_m <= config.plant.goal_radius_m + 1e-12) | (
            signed_goal_tangent_offset_m >= 0.0
        )
        metrics["pnav_goal_capture_seen"][
            active_numpy & (pnav_goal_distance_m <= config.plant.goal_radius_m + 1e-12)
        ] = True
        metrics["eta_goal_capture_seen"][
            active_numpy & (eta_goal_distance_m <= config.plant.goal_radius_m + 1e-12)
        ] = True
        metrics["endpoint_entry_reached"][active_numpy & endpoint_now] = True
        cruise_speed_mps = infos["episode_command_speed_mps"].cpu().numpy()
        ground_speed_mps = torch.linalg.norm(ground_velocity, dim=1).cpu().numpy()
        ground_speed_ratio = ground_speed_mps / np.maximum(cruise_speed_mps, 1e-9)
        previous_speed_min = metrics["ground_speed_ratio_min"][active_numpy]
        current_speed_ratio = ground_speed_ratio[active_numpy]
        speed_lower = current_speed_ratio < previous_speed_min
        active_indices = np.flatnonzero(active_numpy)
        metrics["ground_speed_ratio_min"][active_numpy] = np.minimum(
            previous_speed_min, current_speed_ratio
        )
        metrics["ground_speed_ratio_min_time_s"][active_indices[speed_lower]] = (
            step_index * config.plant.dt_s
        )
        metrics["ground_speed_ratio_sum"][active_numpy] += current_speed_ratio
        metrics["ground_speed_steps"][active_numpy] += 1
        assert inner.actuator is not None
        surge_numpy = ground_velocity[:, 0].detach().cpu().numpy()
        rpm_numpy = inner.actuator[:, 2].detach().cpu().numpy()
        curvature_numpy = projection.curvature_1_per_m.detach().cpu().numpy()
        metrics["ground_surge_sum"][active_numpy] += surge_numpy[active_numpy]
        metrics["ground_surge_sq_sum"][active_numpy] += np.square(surge_numpy[active_numpy])
        metrics["ground_speed_sum"][active_numpy] += ground_speed_mps[active_numpy]
        metrics["ground_speed_sq_sum"][active_numpy] += np.square(ground_speed_mps[active_numpy])
        metrics["rpm_sum"][active_numpy] += rpm_numpy[active_numpy]
        metrics["rpm_sq_sum"][active_numpy] += np.square(rpm_numpy[active_numpy])
        metrics["ground_surge_values"].append(surge_numpy[active_numpy])
        metrics["ground_surge_indices"].append(active_indices)
        metrics["ground_speed_values"].append(ground_speed_mps[active_numpy])
        metrics["ground_speed_indices"].append(active_indices)
        metrics["ground_speed_below_min_steps"][active_numpy] += (
            ground_speed_mps[active_numpy] < speed_envelope_min_mps - 1e-9
        )
        metrics["ground_speed_above_max_steps"][active_numpy] += (
            ground_speed_mps[active_numpy] > speed_envelope_max_mps + 1e-9
        )
        metrics["ground_speed_envelope_steps"][active_numpy] += 1
        metrics["rpm_values"].append(rpm_numpy[active_numpy])
        metrics["rpm_indices"].append(active_indices)
        metrics["path_curvature_values"].append(curvature_numpy[active_numpy])
        metrics["path_curvature_indices"].append(active_indices)
        previous_surge = metrics["previous_ground_surge_for_delta"]
        delta_valid = active_numpy & np.isfinite(previous_surge)
        if bool(delta_valid.any()):
            delta_indices = np.flatnonzero(delta_valid)
            metrics["ground_surge_delta_values"].append(
                np.abs(surge_numpy - previous_surge)[delta_valid]
            )
            metrics["ground_surge_delta_indices"].append(delta_indices)
        previous_surge[active_numpy] = surge_numpy[active_numpy]
        turn_active = active_numpy & (np.abs(curvature_numpy) >= TURN_CURVATURE_THRESHOLD)
        straight_active = active_numpy & ~turn_active
        metrics["turn_speed_sum"][turn_active] += ground_speed_mps[turn_active]
        metrics["turn_steps"][turn_active] += 1
        metrics["straight_speed_sum"][straight_active] += ground_speed_mps[straight_active]
        metrics["straight_steps"][straight_active] += 1
        if decision_step:
            action_numpy = action.detach().cpu().numpy()
            previous_action = metrics["previous_action_for_delta"]
            action_valid = active_numpy & metrics["has_previous_action"]
            if bool(action_valid.any()):
                action_delta = np.abs(action_numpy - previous_action)
                metrics["action_delta_abs_sum"][action_valid] += action_delta[action_valid]
                metrics["action_delta_l2_sum"][action_valid] += np.linalg.norm(
                    action_delta[action_valid], axis=1
                )
                metrics["action_delta_steps"][action_valid] += 1
            previous_action[active_numpy] = action_numpy[active_numpy]
            metrics["has_previous_action"][active_numpy] = True
        if decision_step and path_ablation != "none":
            action_delta = (action - clean_action).detach().abs()
            metrics["path_action_delta_l2_sum"][active_numpy] += (
                torch.linalg.norm(action_delta, dim=1).cpu().numpy()[active_numpy]
            )
            metrics["path_action_delta_l2_max"][active_numpy] = np.maximum(
                metrics["path_action_delta_l2_max"][active_numpy],
                torch.linalg.norm(action_delta, dim=1).cpu().numpy()[active_numpy],
            )
            metrics["path_action_delta_abs_sum"][active_numpy] += action_delta.cpu().numpy()[
                active_numpy
            ]
            metrics["path_action_steps"][active_numpy] += 1
        if decision_step and history_ablation != "none":
            action_delta = (action - history_clean_action).detach().abs()
            metrics["history_action_delta_l2_sum"][active_numpy] += (
                torch.linalg.norm(action_delta, dim=1).cpu().numpy()[active_numpy]
            )
            metrics["history_action_delta_l2_max"][active_numpy] = np.maximum(
                metrics["history_action_delta_l2_max"][active_numpy],
                torch.linalg.norm(action_delta, dim=1).cpu().numpy()[active_numpy],
            )
            metrics["history_action_delta_abs_sum"][active_numpy] += action_delta.cpu().numpy()[
                active_numpy
            ]
            metrics["history_action_steps"][active_numpy] += 1
        if base_action is not None:
            residual_abs = residual_action.detach().abs().cpu().numpy()[active_numpy]
            base_l2 = torch.linalg.norm(base_action.detach(), dim=1).cpu().numpy()[active_numpy]
            residual_l2 = (
                torch.linalg.norm(residual_action.detach(), dim=1).cpu().numpy()[active_numpy]
            )
            metrics["residual_action_values"].append(residual_abs)
            metrics["base_action_l2_values"].append(base_l2)
            metrics["residual_action_l2_values"].append(residual_l2)
            metrics["residual_to_base_ratio_values"].append(
                residual_l2 / np.maximum(base_l2, 1e-12)
            )
            metrics["residual_to_base_ratio_indices"].append(np.flatnonzero(active_numpy))
        metrics["active_steps"][active_numpy] += 1
        metrics["reward_sum"][active_numpy] += reward.cpu().numpy()[active_numpy]
        metrics["abs_yaw_error_sum"][active_numpy] += yaw_error.abs().cpu().numpy()[active_numpy]
        metrics["abs_pitch_error_sum"][active_numpy] += (
            pitch_error.abs().cpu().numpy()[active_numpy]
        )
        metrics["abs_speed_error_sum"][active_numpy] += (
            speed_error.abs().cpu().numpy()[active_numpy]
        )
        metrics["yaw_error_sq_sum"][active_numpy] += yaw_error.square().cpu().numpy()[active_numpy]
        metrics["pitch_error_sq_sum"][active_numpy] += (
            pitch_error.square().cpu().numpy()[active_numpy]
        )
        metrics["speed_error_sq_sum"][active_numpy] += (
            speed_error.square().cpu().numpy()[active_numpy]
        )
        metrics["abs_roll_sum"][active_numpy] += roll.cpu().numpy()[active_numpy]
        metrics["abs_roll_max"][active_numpy] = np.maximum(
            metrics["abs_roll_max"][active_numpy],
            roll.cpu().numpy()[active_numpy],
        )
        metrics["roll_abs_values"].append(roll.cpu().numpy()[active_numpy])
        metrics["roll_rate_abs_values"].append(roll_rate.cpu().numpy()[active_numpy])
        if path_enabled:
            if projection is None:
                raise RuntimeError("C2-C6 evaluation requires ALOS path projection")
            true_projection = inner.path.project(
                inner.eta[:, :3],
                segment_hint=true_projection_hint,
                search_radius=inner.reset_projection_search_radius,
            )
            true_projection_hint = true_projection.segment_index
            true_attitude_quat = quaternion_from_euler(
                inner.eta[:, 3], inner.eta[:, 4], inner.eta[:, 5]
            )
            ground_velocity_ned = body_to_world(ground_velocity, true_attitude_quat)
            route_speed_mps = (
                torch.clamp(
                    (ground_velocity_ned * true_projection.tangent_ned).sum(dim=1),
                    min=0.0,
                )
                .cpu()
                .numpy()
            )
            route_speed_ratio = route_speed_mps / np.maximum(cruise_speed_mps, 1e-9)
            previous_min = metrics["route_ground_speed_ratio_min"][active_numpy]
            current_ratio = route_speed_ratio[active_numpy]
            lower = current_ratio < previous_min
            active_indices = np.flatnonzero(active_numpy)
            metrics["route_ground_speed_ratio_min"][active_numpy] = np.minimum(
                previous_min,
                current_ratio,
            )
            metrics["route_ground_speed_ratio_min_time_s"][active_indices[lower]] = (
                step_index * config.plant.dt_s
            )
            metrics["route_ground_speed_ratio_sum"][active_numpy] += route_speed_ratio[active_numpy]
            metrics["route_ground_speed_steps"][active_numpy] += 1
            metrics["path_steps"][active_numpy] += 1
            metrics["horizontal_error_sq_sum"][active_numpy] += (
                true_projection.horizontal_error_m.square().cpu().numpy()[active_numpy]
            )
            metrics["vertical_error_sq_sum"][active_numpy] += (
                true_projection.vertical_error_m.square().cpu().numpy()[active_numpy]
            )
            metrics["three_d_error_sq_sum"][active_numpy] += (
                true_projection.error_3d_m.square().cpu().numpy()[active_numpy]
            )
            pre_goal_mask = active_numpy & ~metrics["endpoint_entry_reached"] & ~endpoint_now
            metrics["pre_goal_path_steps"][pre_goal_mask] += 1
            metrics["pre_goal_three_d_error_sq_sum"][pre_goal_mask] += (
                true_projection.error_3d_m.square().cpu().numpy()[pre_goal_mask]
            )
            metrics["final_progress_m"][active_numpy] = true_projection.progress_m.cpu().numpy()[
                active_numpy
            ]
            metrics["alos_horizontal_error_sq_sum"][active_numpy] += (
                projection.horizontal_error_m.square().cpu().numpy()[active_numpy]
            )
            metrics["alos_vertical_error_sq_sum"][active_numpy] += (
                projection.vertical_error_m.square().cpu().numpy()[active_numpy]
            )
            metrics["alos_three_d_error_sq_sum"][active_numpy] += (
                projection.error_3d_m.square().cpu().numpy()[active_numpy]
            )
            metrics["alos_final_progress_m"][active_numpy] = projection.progress_m.cpu().numpy()[
                active_numpy
            ]
            if trace is not None:
                record_trace("route_speed_mps", route_speed_mps)
                record_trace("route_tangent_ned", true_projection.tangent_ned)
                record_trace("path_progress_m", true_projection.progress_m)
                # Goal170 eval-only instrumentation: retain per-step route error
                # under the same names used by the SAC evaluator so both policy
                # families share one route-health analyzer.
                record_trace("path_progress", true_projection.progress_m)
                record_trace("path_error_3d", true_projection.error_3d_m)
                record_trace("path_horizontal_error", true_projection.horizontal_error_m)
                record_trace("path_vertical_error", true_projection.vertical_error_m)
        metrics["material_saturation_steps"][active_numpy] += (
            infos["material_saturated"].cpu().numpy()[active_numpy]
        )
        hold = infos["hold_fraction"].cpu().numpy()
        metrics["max_hold_fraction"][active_numpy] = np.maximum(
            metrics["max_hold_fraction"][active_numpy], hold[active_numpy]
        )
        metrics["final_hold_fraction"][active_numpy] = hold[active_numpy]
        metrics["sensor_p_nav_error_m"].append(
            torch.linalg.norm(sensor.p_nav - inner.eta[:, :3], dim=1).cpu().numpy()[active_numpy]
        )
        metrics["sensor_dvl_bottom_valid"].append(
            sensor.bottom_lock_valid.cpu().numpy()[active_numpy]
        )
        metrics["sensor_current_estimate_valid"].append(
            sensor.current_estimate_valid.cpu().numpy()[active_numpy]
        )
        metrics["sensor_usbl_valid"].append(sensor.usbl_valid.cpu().numpy()[active_numpy])
        metrics["sensor_usbl_outlier"].append(sensor.usbl_outlier.cpu().numpy()[active_numpy])
        metrics["sensor_nav_age_s"].append(sensor.nav_age_s.cpu().numpy()[active_numpy])
        metrics["sensor_dvl_bottom_age_s"].append(
            sensor.dvl_bottom_age_s.cpu().numpy()[active_numpy]
        )
        metrics["sensor_current_estimate_age_s"].append(
            sensor.current_estimate_age_s.cpu().numpy()[active_numpy]
        )
        guidance_current_valid = infos["guidance_current_valid"].cpu().numpy()
        guidance_current_age = infos["guidance_current_age_s"].cpu().numpy()
        guidance_crab_deg = np.degrees(infos["guidance_crab_angle_rad"].cpu().numpy())
        guidance_speed_scale = infos["guidance_speed_scale"].cpu().numpy()
        guidance_terminal_phase = infos["guidance_terminal_phase"].cpu().numpy() > 0.5
        guidance_terminal_ratio = infos["guidance_terminal_speed_ratio"].cpu().numpy()
        guidance_saturation_ema = infos["guidance_saturation_ema"].cpu().numpy()
        guidance_saturation_emergency = infos["guidance_saturation_emergency"].cpu().numpy() > 0.5
        metrics["guidance_current_valid"].append(guidance_current_valid[active_numpy])
        metrics["guidance_current_age_s"].append(guidance_current_age[active_numpy])
        metrics["guidance_crab_angle_deg"].append(guidance_crab_deg[active_numpy])
        metrics["guidance_speed_scale"].append(guidance_speed_scale[active_numpy])
        metrics["guidance_terminal_phase"].append(guidance_terminal_phase[active_numpy])
        metrics["guidance_terminal_speed_ratio"].append(guidance_terminal_ratio[active_numpy])
        terminal_active = active_numpy & guidance_terminal_phase
        if bool(terminal_active.any()):
            metrics["guidance_terminal_speed_ratio_values"].append(
                guidance_terminal_ratio[terminal_active]
            )
        metrics["guidance_saturation_ema"].append(guidance_saturation_ema[active_numpy])
        metrics["guidance_saturation_emergency"].append(guidance_saturation_emergency[active_numpy])
        metrics["guidance_saturation_emergency_steps"][
            active_numpy & guidance_saturation_emergency
        ] += 1
        metrics["guidance_route_floor_active"].append(
            (infos["guidance_route_speed_floor_active"].cpu().numpy() > 0.5)[active_numpy]
        )
        metrics["guidance_route_floor_launch_active"].append(
            (infos["guidance_route_speed_floor_launch_active"].cpu().numpy() > 0.5)[active_numpy]
        )
        metrics["guidance_route_floor_valid"].append(
            (infos["guidance_route_speed_floor_valid"].cpu().numpy() > 0.5)[active_numpy]
        )
        metrics["guidance_route_floor_boost_mps"].append(
            infos["guidance_route_speed_floor_boost_mps"].cpu().numpy()[active_numpy]
        )
        metrics["guidance_route_floor_ratio"].append(
            infos["guidance_route_speed_floor_ratio"].cpu().numpy()[active_numpy]
        )

        if trace is not None:
            goal_distance = torch.linalg.norm(
                inner.eta[:, :3] - inner.path.goal_ned_m_for(len(seeds)), dim=1
            )
            record_trace("step", np.full(len(seeds), step_index, dtype=np.int64))
            record_trace("time_s", np.full(len(seeds), step_index * config.plant.dt_s))
            assert inner.episode_seed is not None
            record_trace("episode_seed", inner.episode_seed)
            record_trace("seed", inner.episode_seed)
            record_trace("eta", inner.eta)
            record_trace("nu", inner.nu)
            record_trace("current_ned_mps", inner.current_ned_mps)
            record_trace(
                "command",
                torch.stack(
                    [command.heading_rad, command.pitch_rad, command.speed_mps],
                    dim=1,
                ),
            )
            if infos.get("command_residual_applied") is not None:
                record_trace("command_residual_applied", infos["command_residual_applied"])
                record_trace("command_base_physical", infos["command_base_physical"])
                record_trace("command_corrected_physical", infos["command_corrected_physical"])
            if infos.get("actuator_residual_applied") is not None:
                record_trace(
                    "actuator_residual_applied",
                    infos["actuator_residual_applied"],
                )
                record_trace(
                    "actuator_residual_effective",
                    infos["actuator_residual_effective"],
                )
                record_trace("actuator_base_physical", infos["actuator_base_physical"])
                record_trace("actuator_total_physical", infos["actuator_total_physical"])
                if infos.get("actuator_residual_scale") is not None:
                    record_trace(
                        "actuator_residual_scale",
                        infos["actuator_residual_scale"],
                    )
                if infos.get("actuator_residual_schedule_distance_m") is not None:
                    record_trace(
                        "actuator_residual_schedule_distance_m",
                        infos["actuator_residual_schedule_distance_m"],
                    )
                if infos.get("actuator_residual_channel_gains") is not None:
                    record_trace(
                        "actuator_residual_effective_channel_gains",
                        infos["actuator_residual_channel_gains"],
                    )
                if infos.get("actuator_residual_channel_gain_near_gate") is not None:
                    record_trace(
                        "actuator_residual_channel_gain_near_gate",
                        infos["actuator_residual_channel_gain_near_gate"],
                    )
                if infos.get("actuator_residual_ppo_applied") is not None:
                    record_trace(
                        "actuator_residual_bc_applied",
                        infos["actuator_residual_bc_applied"],
                    )
                    record_trace(
                        "actuator_residual_bc_effective",
                        infos["actuator_residual_bc_effective"],
                    )
                    record_trace(
                        "actuator_residual_ppo_applied",
                        infos["actuator_residual_ppo_applied"],
                    )
                    record_trace(
                        "actuator_residual_ppo_effective",
                        infos["actuator_residual_ppo_effective"],
                    )
            if ppo_residual_attenuation is not None:
                record_trace(
                    "ppo_residual_attenuation",
                    ppo_residual_attenuation,
                )
                record_trace(
                    "ppo_residual_attenuation_distance_m",
                    ppo_residual_attenuation_distance,
                )
            if infos.get("frozen_residual_gate") is not None:
                record_trace(
                    "frozen_residual_base_action",
                    infos["frozen_residual_base_action"],
                )
                record_trace("frozen_residual_action", infos["frozen_residual_action"])
                record_trace("frozen_residual_gate", infos["frozen_residual_gate"])
                record_trace(
                    "frozen_residual_executed_action",
                    infos["frozen_residual_executed_action"],
                )
            record_trace("ground_velocity_body_mps", ground_velocity)
            record_trace("ground_speed_mps", torch.linalg.norm(ground_velocity, dim=1))
            record_trace("goal_distance_m", goal_distance)
            record_trace("p_nav", sensor.p_nav)
            record_trace("goal_ned_m", goal_ned_m)
            record_trace("goal_tangent_ned", goal_tangent_ned)
            record_trace("goal_distance_pnav_m", endpoint_goal_distance_m)
            record_trace("goal_distance_eta_m", eta_goal_distance_m)
            record_trace("signed_goal_tangent_offset_pnav_m", signed_goal_tangent_offset_m)
            record_trace(
                "signed_goal_tangent_offset_eta_m",
                signed_goal_tangent_offset_eta_m,
            )
            record_trace("endpoint_entry_reached", endpoint_now)
            record_trace("terminal_kind", infos["terminal_kind"])
            record_trace("cruise_speed_mps", infos["episode_command_speed_mps"])
            record_trace("heading_error_rad", yaw_error)
            record_trace("pitch_error_rad", pitch_error)
            record_trace("speed_error_mps", speed_error)
            record_trace("requested_action", action)
            if base_action is not None:
                record_trace("base_action", base_action)
            if residual_action is not None:
                record_trace("residual_action", residual_action)
            if inner.actuator is not None:
                record_trace("actual_actuator", inner.actuator)
            record_trace("material_saturated", infos["material_saturated"])
            record_trace(
                "guidance_speed_target_mps",
                infos["guidance_route_speed_floor_target_mps"],
            )
            record_trace(
                "guidance_route_floor_filtered_mps",
                infos["guidance_route_speed_floor_filtered_mps"],
            )
            record_trace(
                "guidance_route_floor_boost_mps",
                infos["guidance_route_speed_floor_boost_mps"],
            )
            record_trace(
                "guidance_route_floor_active",
                infos["guidance_route_speed_floor_active"],
            )
            record_trace(
                "guidance_route_floor_valid",
                infos["guidance_route_speed_floor_valid"],
            )
            record_trace("guidance_terminal_phase", infos["guidance_terminal_phase"])
            record_trace(
                "guidance_terminal_speed_ratio",
                infos["guidance_terminal_speed_ratio"],
            )
            record_trace(
                "guidance_target_heading_rad",
                infos["guidance_target_heading_rad"],
            )
            record_trace(
                "guidance_target_pitch_rad",
                infos["guidance_target_pitch_rad"],
            )
            record_trace(
                "guidance_target_speed_mps",
                infos["guidance_target_speed_mps"],
            )
            record_trace(
                "guidance_limited_heading_rad",
                infos["guidance_limited_heading_rad"],
            )
            record_trace(
                "guidance_limited_pitch_rad",
                infos["guidance_limited_pitch_rad"],
            )
            record_trace(
                "guidance_limited_speed_mps",
                infos["guidance_limited_speed_mps"],
            )
            record_trace(
                "guidance_heading_rate_clipped",
                infos["guidance_heading_rate_clipped"],
            )
            record_trace(
                "guidance_pitch_rate_clipped",
                infos["guidance_pitch_rate_clipped"],
            )
            record_trace(
                "guidance_speed_rate_clipped",
                infos["guidance_speed_rate_clipped"],
            )
            record_trace("dvl_ground_body", sensor.dvl_ground_body)
            record_trace("dvl_bottom_valid", sensor.bottom_lock_valid)
            record_trace("current_estimate_body", sensor.current_estimate_body)
            record_trace("current_estimate_valid", sensor.current_estimate_valid)
            record_trace("current_estimate_age_s", sensor.current_estimate_age_s)
            _record_trace_sensor_ages(record_trace, sensor, enabled=trace_sensor_ages)
            _record_trace_estimator_state(record_trace, sensor, enabled=trace_estimator_state)
            if trace_reward_breakdown:
                for name, value in _reward_breakdown_trace_fields(infos).items():
                    record_trace(name, value)

        terminal = infos["terminal_kind"].cpu().numpy()
        finished = active_numpy & (done.cpu().numpy() > 0.5)
        goal = finished & (terminal == int(PCTerminalKind.GOAL))
        plane_miss = finished & (terminal == int(PCTerminalKind.GOAL_PLANE_MISS))
        timeout = finished & (terminal == int(PCTerminalKind.TIME_LIMIT))
        safety = finished & ~goal & ~plane_miss & ~timeout
        metrics["goal_terminations"][goal] += 1
        metrics["goal_plane_miss_terminations"][plane_miss] += 1
        metrics["timeout_terminations"][timeout] += 1
        metrics["safety_terminations"][safety] += 1
        metrics["miss_distance_at_plane_m"][plane_miss] = endpoint_goal_distance_m[plane_miss]
        if not endpoint_stop_enabled:
            terminal_mask = active_numpy & metrics["endpoint_entry_reached"]
            metrics["terminal_longitudinal_steps"][terminal_mask] += 1
            metrics["terminal_longitudinal_sq_sum"][terminal_mask] += np.square(
                signed_goal_tangent_offset_m[terminal_mask]
            )
            overshoot = np.maximum(0.0, signed_goal_tangent_offset_m[terminal_mask])
            previous = metrics["post_goal_overshoot_m"][terminal_mask]
            metrics["post_goal_overshoot_m"][terminal_mask] = np.where(
                np.isfinite(previous),
                np.maximum(previous, overshoot),
                overshoot,
            )
        if bool(finished.any()):
            goal_distance = (
                torch.linalg.norm(inner.eta[:, :3] - inner.path.goal_ned_m_for(len(seeds)), dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            path_length = inner.path.total_length_m_for(len(seeds)).detach().cpu().numpy()
            metrics["final_goal_distance_m"][finished] = goal_distance[finished]
            metrics["final_goal_distance_set"][finished] = True
            metrics["pnav_final_goal_distance_m"][finished] = pnav_goal_distance_m[finished]
            metrics["eta_goal_capture_at_done"][finished] = (
                goal_distance[finished] <= config.plant.goal_radius_m + 1e-12
            )
            metrics["path_overshoot_m"][finished] = np.maximum(
                metrics["final_progress_m"][finished] - path_length[finished], 0.0
            )
        metrics["done"][finished] = True

    if not bool(metrics["done"].all()):
        raise RuntimeError("some PC episodes did not terminate")
    if trace is not None:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_output,
            **{
                name: np.concatenate(values, axis=0) if values else np.empty(0)
                for name, values in trace.items()
            },
        )
    path_length_m = inner.path.total_length_m
    summary = _summary(
        metrics,
        path_enabled=path_enabled,
        path_length_m=path_length_m,
        action_full_scale=metadata["action_full_scale"],
        speed_envelope_min_mps=speed_envelope_min_mps,
        speed_envelope_max_mps=speed_envelope_max_mps,
        control_dt_s=config.plant.dt_s,
        endpoint_stop_enabled=endpoint_stop_enabled,
        goal_radius_m=config.plant.goal_radius_m,
        success_frame=success_frame,
    )
    episodes = _episode_records(
        metrics,
        seeds,
        path_enabled=path_enabled,
        path_length_m=path_length_m,
        action_full_scale=metadata["action_full_scale"],
        control_dt_s=config.plant.dt_s,
        endpoint_stop_enabled=endpoint_stop_enabled,
        goal_radius_m=config.plant.goal_radius_m,
        success_frame=success_frame,
        plant_scale_factors=(
            None if inner.plant_scales is None else inner.plant_scales.detach().cpu().numpy()
        ),
    )
    if inner.curriculum_specs is not None:
        if len(inner.curriculum_specs) != len(episodes):
            raise ValueError("curriculum specs do not align with evaluation episodes")
        for index, (row, spec) in enumerate(zip(episodes, inner.curriculum_specs, strict=True)):
            row["curriculum"] = curriculum_episode_metadata(
                spec,
                initial_heading_rad=float(initial_heading_rad[index]),
            )
    contract = {
        "endpoint_stop_enabled": endpoint_stop_enabled,
        "sensor_override": sensor_override,
        "actor_observation_source": (
            "ideal" if config.observation.ideal_actor_observations else "nominal"
        ),
        "actor_observation_override": bool(
            base_config is not None
            and config.observation.ideal_actor_observations
            != base_config.observation.ideal_actor_observations
        ),
        "base_sensor": (
            None
            if base_config is None or base_config.sensors is None
            else {
                "profile": base_config.sensors.profile,
                "seed": base_config.sensors.seed,
            }
        ),
        "override_sensor": (
            None
            if config.sensors is None
            else {
                "profile": config.sensors.profile,
                "seed": config.sensors.seed,
            }
        ),
    }
    return summary, metadata, episodes, contract
