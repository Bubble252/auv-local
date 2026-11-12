#!/usr/bin/env python3
"""Read-only reward/credit decomposition for Goal27.

This script never trains and never writes checkpoints. It rolls out a frozen PC
policy, records every reward term returned by the environment, and aligns those
terms with path error, command error, actuator saturation, and the residual
between the learned actuator action and a deployable PD/PI controller driven by
the ALOS command.

The checkpoint used for Goal27 Step 1 (D3) contains an actor only. Therefore a
true GAE advantage cannot be computed here. We report discounted returns,
return shares, and per-term variance shares, plus a critic-free centered
discounted return; these are credit-allocation proxies, not PPO advantages.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.controllers import BatchedControlContext, BatchedRemus100Teacher
from auv_lowlevel.batch.geometry import (
    ground_velocity_body_for_contract,
    quaternion_from_direction,
    wrap_angle,
)
from auv_lowlevel.batch.pc import (
    PCTerminalKind,
    PCRewardBreakdown,
    pc_terminal_kind_name,
)
from auv_lowlevel.batch.pc_bc import PCBCPolicy, pc_action_from_actuator
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config

REWARD_TERMS = tuple(field.name for field in fields(PCRewardBreakdown))
RAW_TERM_FIELDS = frozenset(
    {"path_cross", "path_vertical", "path_progress", "path_heading", "speed_smooth", "energy"}
)
DERIVED_TERMS = frozenset({"total", "ppo_scaled"})
COMPONENT_TERMS = tuple(
    name for name in REWARD_TERMS if name not in RAW_TERM_FIELDS and name not in DERIVED_TERMS
)
PATH_CONTRIBUTION_TERMS = (
    "path_3d_penalty",
    "path_yaw_penalty",
    "path_progress_reward",
    "path_speed_smooth_penalty",
    "path_energy_penalty",
)
BUCKET_LABELS = ("0-25%", "25-50%", "50-75%", "75-100%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--path-dim", type=int, choices=(0, 8, 12), default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--termination-mode",
        choices=("legacy", "endpoint_stop"),
        default="endpoint_stop",
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses the config limit")
    return parser.parse_args()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float64).numpy()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = 0.5 * (start + index - 1)
            start = index
    return ranks


def _corr(x: np.ndarray, y: np.ndarray, *, spearman: bool = False) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 8:
        return None
    x_value = x[mask]
    y_value = y[mask]
    if np.std(x_value) < 1e-12 or np.std(y_value) < 1e-12:
        return None
    if spearman:
        x_value = _rankdata(x_value)
        y_value = _rankdata(y_value)
    return float(np.corrcoef(x_value, y_value)[0, 1])


def _bucket_indices(progress_ratio: np.ndarray) -> np.ndarray:
    clipped = np.clip(progress_ratio, 0.0, 1.0)
    return np.minimum((clipped * len(BUCKET_LABELS)).astype(np.int64), len(BUCKET_LABELS) - 1)


def _build_base_controller(env: BatchedPCLowLevelEnv) -> BatchedRemus100Teacher:
    controller = BatchedRemus100Teacher(
        env.config.controller,
        env.limits,
        device=env.device,
        dtype=env.dtype,
    )
    controller.reset(count=env.num_envs)
    return controller


def _base_action_from_command(
    controller: BatchedRemus100Teacher,
    actor_state: torch.Tensor,
    command: torch.Tensor,
    *,
    dt_s: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """PD/PI actuator command from actor-visible state and the ALOS command."""

    heading = command[:, 0]
    pitch = command[:, 1]
    direction = torch.stack(
        (
            torch.cos(pitch) * torch.cos(heading),
            torch.cos(pitch) * torch.sin(heading),
            -torch.sin(pitch),
        ),
        dim=1,
    )
    context = BatchedControlContext(
        eta=actor_state[:, :6],
        nu=actor_state[:, 6:12],
        quat_xyzw=quaternion_from_direction(direction),
        current_ned_mps=torch.zeros_like(actor_state[:, :3]),
        reference_quat_xyzw=quaternion_from_direction(direction),
        command_speed_mps=command[:, 2],
        observation=actor_state,
        dt_s=float(dt_s),
    )
    actuator = controller.control(context)
    return pc_action_from_actuator(actuator.to(device=device, dtype=dtype))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in (0, 1]")

    config = load_config(args.config)
    config = replace(
        config,
        termination=replace(
            config.termination,
            pc_goal_plane_miss_termination_enabled=args.termination_mode == "endpoint_stop",
        ),
    )
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    max_steps = args.max_steps or int(config.plant.max_episode_steps)

    env = BatchedPCLowLevelEnv(
        config,
        num_envs=len(seeds),
        device=args.device,
        dtype=dtype,
        path_dim=args.path_dim,
    )
    env.reset(seeds)
    policy = PCBCPolicy(args.checkpoint, device=args.device)
    if policy.model.path_dim != args.path_dim:
        raise ValueError(
            f"checkpoint path_dim={policy.model.path_dim} does not match --path-dim={args.path_dim}"
        )
    controller = _build_base_controller(env)

    num_envs = len(seeds)
    term_values: dict[str, list[np.ndarray]] = {name: [] for name in REWARD_TERMS}
    active_rows: list[np.ndarray] = []
    diagnostics: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "command_heading_rad",
            "command_pitch_rad",
            "command_speed_mps",
            "command_heading_rate_rad_s",
            "command_pitch_rate_rad_s",
            "command_speed_rate_mps2",
            "actor_speed_mps",
            "speed_error_mps",
            "ground_surge_mps",
            "path_cross_m",
            "path_vertical_m",
            "path_error_3d_m",
            "alos_projection_error_3d_m",
            "path_progress_ratio",
            "goal_distance_m",
            "action_norm",
            "residual_norm",
            "residual_rudder",
            "residual_stern",
            "residual_rpm",
            "rudder_fraction",
            "stern_fraction",
            "rpm_fraction",
            "material_saturated",
            "rate_clipped",
            "material_rate_clipped",
            "magnitude_clipped",
            "crab_angle_rad",
            "current_estimate_valid",
            "sensor_nav_age_s",
            "hold_fraction",
        )
    }
    termination_kind = np.zeros(num_envs, dtype=np.int64)
    termination_step = np.zeros(num_envs, dtype=np.int64)
    done = np.zeros(num_envs, dtype=bool)
    previous_command: torch.Tensor | None = None
    true_hint = env.path.project(env.eta[:, :3]).segment_index

    for step_index in range(max_steps):
        active = ~done
        if not bool(active.any()):
            break
        active_np = active.astype(bool).copy()
        observation = env.observation
        with torch.inference_mode():
            actor_path = observation.actor_path if policy.model.path_dim else None
            actor_current = (
                observation.actor_current_body
                if policy.model.actor_current_encoder is not None
                else None
            )
            bc_action = policy.act_inference(
                observation.actor_state,
                None,
                actor_path,
                actor_current,
            )
            command = torch.stack(
                (
                    observation.command.heading_rad,
                    observation.command.pitch_rad,
                    observation.command.speed_mps,
                ),
                dim=1,
            )
            base_action = _base_action_from_command(
                controller,
                observation.actor_state,
                command,
                dt_s=env.dt_s,
                device=env.device,
                dtype=env.dtype,
            )
        result = env.step(bc_action, auto_reset=False, advance_mask=active)
        reward = result.reward
        for name in REWARD_TERMS:
            value = getattr(reward, name)
            if value is None:
                value = torch.zeros(env.num_envs, device=env.device, dtype=env.dtype)
            term_values[name].append(_to_numpy(value))
        active_rows.append(active_np)

        eta = env.eta
        nu = env.nu
        assert eta is not None and nu is not None
        true_projection = env.path.project(
            eta[:, :3],
            segment_hint=true_hint,
            search_radius=env.reset_projection_search_radius,
        )
        true_hint = true_projection.segment_index
        alos_projection = result.info["projection"]
        command = result.info["command_physical"]
        if previous_command is None:
            previous_command = command.clone()
        dt_s = float(env.dt_s)
        heading_rate = wrap_angle(command[:, 0] - previous_command[:, 0]) / dt_s
        pitch_rate = (command[:, 1] - previous_command[:, 1]) / dt_s
        speed_rate = (command[:, 2] - previous_command[:, 2]) / dt_s
        previous_command = command.clone()

        ground_velocity = ground_velocity_body_for_contract(
            eta,
            nu,
            env.current_ned_mps,
            config.plant.velocity_contract,
        )
        speed_error = command[:, 2] - ground_velocity[:, 0]
        actuator = env.actuator
        assert actuator is not None
        residual = bc_action.to(device=env.device, dtype=env.dtype) - base_action
        rudder_limit = env.limits.max_rudder_angle_rad
        stern_limit = env.limits.max_stern_plane_angle_rad
        rpm_limit = env.limits.max_propeller_rpm
        path_length = max(float(env.path.total_length_m), 1e-12)

        values = {
            "command_heading_rad": command[:, 0],
            "command_pitch_rad": command[:, 1],
            "command_speed_mps": command[:, 2],
            "command_heading_rate_rad_s": heading_rate,
            "command_pitch_rate_rad_s": pitch_rate,
            "command_speed_rate_mps2": speed_rate,
            "actor_speed_mps": observation.actor_state[:, 9],
            "speed_error_mps": speed_error,
            "ground_surge_mps": ground_velocity[:, 0],
            "path_cross_m": true_projection.horizontal_error_m,
            "path_vertical_m": true_projection.vertical_error_m,
            "path_error_3d_m": true_projection.error_3d_m,
            "alos_projection_error_3d_m": alos_projection.error_3d_m
            if alos_projection is not None
            else torch.zeros_like(true_projection.error_3d_m),
            "path_progress_ratio": true_projection.progress_m / path_length,
            "goal_distance_m": result.info["goal_distance_m"],
            "action_norm": torch.linalg.norm(bc_action.to(env.device, env.dtype), dim=1),
            "residual_norm": torch.linalg.norm(residual, dim=1),
            "residual_rudder": residual[:, 0],
            "residual_stern": residual[:, 1],
            "residual_rpm": residual[:, 2],
            "rudder_fraction": torch.abs(actuator[:, 0]) / rudder_limit,
            "stern_fraction": torch.abs(actuator[:, 1]) / stern_limit,
            "rpm_fraction": actuator[:, 2] / rpm_limit,
            "material_saturated": result.info["material_saturated"].to(env.dtype),
            "rate_clipped": torch.any(result.info["rate_clipped"], dim=1).to(env.dtype),
            "material_rate_clipped": torch.any(
                result.info["material_rate_clipped"], dim=1
            ).to(env.dtype),
            "magnitude_clipped": torch.any(
                result.info["magnitude_clipped"], dim=1
            ).to(env.dtype),
            "crab_angle_rad": result.info["guidance_crab_angle_rad"],
            "current_estimate_valid": result.info["current_estimate_valid"].to(env.dtype),
            "sensor_nav_age_s": result.info["nav_age_s"],
            "hold_fraction": result.info["hold_fraction"],
        }
        for name, value in values.items():
            diagnostics[name].append(_to_numpy(value))

        terminated = result.terminated.detach().cpu().numpy()
        truncated = result.truncated.detach().cpu().numpy()
        newly_done = (terminated | truncated) & active_np
        if bool(newly_done.any()):
            kinds = result.info["terminal_kind"].detach().cpu().numpy()
            termination_kind[newly_done] = kinds[newly_done]
            termination_step[newly_done] = step_index + 1
            done |= newly_done
    term_arrays = {name: np.stack(rows, axis=0) for name, rows in term_values.items()}
    w_path_3d = (
        config.reward.w_path_3d
        if env.pc_control_mode == "alos"
        else config.reward.w_3d
    )
    contribution_arrays = {
        "path_3d_penalty": -w_path_3d
        * np.sqrt(term_arrays["path_cross"] ** 2 + term_arrays["path_vertical"] ** 2)
        / 5.0,
        "path_yaw_penalty": -config.reward.w_yaw
        * np.abs(term_arrays["path_heading"])
        / math.pi,
        "path_progress_reward": config.reward.w_prog * term_arrays["path_progress"],
        "path_speed_smooth_penalty": -config.reward.w_path_speed_smooth
        * term_arrays["speed_smooth"],
        "path_energy_penalty": -config.reward.w_path_energy * term_arrays["energy"],
    }
    term_arrays.update(contribution_arrays)
    all_component_terms = COMPONENT_TERMS + PATH_CONTRIBUTION_TERMS
    reconstructed_total = sum(contribution_arrays[name] for name in PATH_CONTRIBUTION_TERMS)
    for name in COMPONENT_TERMS:
        reconstructed_total = reconstructed_total + term_arrays[name]
    reconstruction_error = float(
        np.max(np.abs(reconstructed_total - term_arrays["total"]))
    )
    active_array = np.stack(active_rows, axis=0)
    diagnostic_arrays = {
        name: np.stack(rows, axis=0) for name, rows in diagnostics.items()
    }
    steps_recorded = active_array.shape[0]
    bad_shapes = {
        name: tuple(array.shape)
        for name, array in diagnostic_arrays.items()
        if array.shape[0] != steps_recorded or array.ndim != 2
    }
    if bad_shapes:
        raise RuntimeError(f"diagnostic shape mismatch: {bad_shapes}")

    episodes: list[dict[str, Any]] = []
    for row, seed in enumerate(seeds):
        active_column = active_array[:, row]
        active_indices = np.flatnonzero(active_column)
        if active_indices.size == 0:
            continue
        length = int(active_indices.size)
        discount = args.gamma ** np.arange(length, dtype=np.float64)
        term_returns: dict[str, float] = {}
        term_discounted: dict[str, float] = {}
        term_variance: dict[str, float] = {}
        term_abs_sum: dict[str, float] = {}
        for name in (*all_component_terms, "total", "ppo_scaled"):
            array = term_arrays[name]
            series = array[active_indices, row]
            term_returns[name] = float(np.sum(series))
            term_discounted[name] = float(np.sum(discount * series))
            term_variance[name] = float(np.var(series))
            term_abs_sum[name] = float(np.sum(np.abs(series)))
        abs_total = sum(term_abs_sum[name] for name in all_component_terms)
        variance_total = sum(term_variance[name] for name in all_component_terms)
        kind = int(termination_kind[row])
        kind_name = pc_terminal_kind_name(kind) if kind != int(PCTerminalKind.NONE) else None
        progress = diagnostic_arrays["path_progress_ratio"][active_indices, row]
        error_3d = diagnostic_arrays["path_error_3d_m"][active_indices, row]
        bucket = _bucket_indices(progress)
        bucket_rmse = {}
        bucket_residual = {}
        for bucket_index, label in enumerate(BUCKET_LABELS):
            mask = bucket == bucket_index
            bucket_rmse[label] = (
                float(np.sqrt(np.mean(error_3d[mask] ** 2))) if bool(mask.any()) else None
            )
            bucket_residual[label] = (
                float(np.mean(diagnostic_arrays["residual_norm"][active_indices, row][mask]))
                if bool(mask.any())
                else None
            )
        episodes.append(
            {
                "seed": int(seed),
                "steps": length,
                "termination_kind": kind_name,
                "success": kind == int(PCTerminalKind.GOAL),
                "return_total": term_returns["total"],
                "ppo_scaled_return": term_returns["ppo_scaled"],
                "term_returns": term_returns,
                "term_discounted_returns": term_discounted,
                "term_return_abs_share": {
                    name: (term_abs_sum[name] / abs_total if abs_total > 1e-12 else 0.0)
                    for name in all_component_terms
                },
                "term_variance_share": {
                    name: (term_variance[name] / variance_total if variance_total > 1e-12 else 0.0)
                    for name in all_component_terms
                },
                "path_progress_3d_rmse_m": bucket_rmse,
                "path_progress_residual_norm": bucket_residual,
                "mean_residual_norm": float(
                    np.mean(diagnostic_arrays["residual_norm"][active_indices, row])
                ),
                "mean_path_error_3d_m": float(np.mean(error_3d)),
                "mechanical_saturation_rate": float(
                    np.mean(diagnostic_arrays["material_saturated"][active_indices, row])
                ),
                "rate_clipped_rate": float(
                    np.mean(diagnostic_arrays["rate_clipped"][active_indices, row])
                ),
                "material_rate_clipped_rate": float(
                    np.mean(diagnostic_arrays["material_rate_clipped"][active_indices, row])
                ),
            }
        )

    pooled_mask = active_array.reshape(-1) > 0
    pooled = {
        name: array.reshape(-1)[pooled_mask] for name, array in diagnostic_arrays.items()
    }
    correlations: dict[str, dict[str, float | None]] = {}
    residual_targets = (
        "abs_speed_error",
        "path_cross_abs",
        "path_vertical_abs",
        "path_error_3d",
        "command_heading_rate_abs",
        "command_speed_rate_abs",
        "rudder_fraction",
        "material_saturated",
        "goal_distance",
        "path_progress_ratio",
    )
    derived = {
        "abs_speed_error": np.abs(pooled["speed_error_mps"]),
        "path_cross_abs": np.abs(pooled["path_cross_m"]),
        "path_vertical_abs": np.abs(pooled["path_vertical_m"]),
        "path_error_3d": pooled["path_error_3d_m"],
        "command_heading_rate_abs": np.abs(pooled["command_heading_rate_rad_s"]),
        "command_speed_rate_abs": np.abs(pooled["command_speed_rate_mps2"]),
        "rudder_fraction": pooled["rudder_fraction"],
        "material_saturated": pooled["material_saturated"],
        "goal_distance": pooled["goal_distance_m"],
        "path_progress_ratio": pooled["path_progress_ratio"],
    }
    for target_name in residual_targets:
        correlations[f"residual_norm__{target_name}"] = _corr(
            pooled["residual_norm"], derived[target_name], spearman=True
        )
    for target_name in residual_targets:
        correlations[f"abs_residual_rudder__{target_name}"] = _corr(
            np.abs(pooled["residual_rudder"]), derived[target_name], spearman=True
        )
    correlations["path_error_3d__goal_distance"] = _corr(
        pooled["path_error_3d_m"], pooled["goal_distance_m"], spearman=True
    )
    correlations["path_error_3d__residual_norm"] = _corr(
        pooled["path_error_3d_m"], pooled["residual_norm"], spearman=True
    )

    aggregate_terms: dict[str, dict[str, float | None]] = {}
    for name in all_component_terms:
        returns = np.array([episode["term_returns"][name] for episode in episodes])
        discounted = np.array(
            [episode["term_discounted_returns"][name] for episode in episodes]
        )
        abs_shares = np.array(
            [episode["term_return_abs_share"][name] for episode in episodes]
        )
        variance_shares = np.array(
            [episode["term_variance_share"][name] for episode in episodes]
        )
        series = term_arrays[name].reshape(-1)[pooled_mask]
        aggregate_terms[name] = {
            "mean_return": float(np.mean(returns)) if returns.size else None,
            "std_return": float(np.std(returns)) if returns.size else None,
            "mean_discounted_return": float(np.mean(discounted)) if discounted.size else None,
            "mean_abs_share": float(np.mean(abs_shares)) if abs_shares.size else None,
            "mean_variance_share": float(np.mean(variance_shares))
            if variance_shares.size
            else None,
            "mean_step_value": float(np.mean(series)) if series.size else None,
            "std_step_value": float(np.std(series)) if series.size else None,
            "mean_abs_step_value": float(np.mean(np.abs(series))) if series.size else None,
        }

    guidance = config.guidance
    heading_limit = float(guidance.heading_rate_limit_rad_s)
    pitch_limit = float(guidance.pitch_rate_limit_rad_s)
    speed_limit = float(guidance.speed_rate_limit_mps2)
    summary = {
        "schema_version": "auv_pc_goal27_offline_decomposition.v1",
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "steps_recorded": int(steps_recorded),
        "active_step_count": int(pooled_mask.sum()),
        "episode_count": len(episodes),
        "termination_mode": args.termination_mode,
        "termination_counts": {
            name: int(np.sum(np.array([episode["termination_kind"] == name for episode in episodes])))
            for name in sorted(
                {episode["termination_kind"] for episode in episodes if episode["termination_kind"]}
            )
        },
        "success_rate": float(np.mean([episode["success"] for episode in episodes]))
        if episodes
        else None,
        "mean_steps": float(np.mean([episode["steps"] for episode in episodes]))
        if episodes
        else None,
        "reward_terms": aggregate_terms,
        "reward_reconstruction_max_abs_error": reconstruction_error,
        "reward_weights": {
            "w_path_3d": w_path_3d,
            "w_path_speed_smooth": config.reward.w_path_speed_smooth,
            "w_path_energy": config.reward.w_path_energy,
            "w_cmd_yaw": config.reward.w_cmd_yaw,
            "w_cmd_pitch": config.reward.w_cmd_pitch,
            "w_cmd_speed": config.reward.w_cmd_speed,
        },
        "residuals": {
            "mean_norm": float(np.mean(pooled["residual_norm"])),
            "p50_norm": float(np.percentile(pooled["residual_norm"], 50)),
            "p95_norm": float(np.percentile(pooled["residual_norm"], 95)),
            "mean_abs_rudder": float(np.mean(np.abs(pooled["residual_rudder"]))),
            "mean_abs_stern": float(np.mean(np.abs(pooled["residual_stern"]))),
            "mean_abs_rpm": float(np.mean(np.abs(pooled["residual_rpm"]))),
        },
        "errors": {
            "path_3d_rmse_m": float(
                np.sqrt(np.mean(pooled["path_error_3d_m"] ** 2))
            ),
            "path_cross_rmse_m": float(
                np.sqrt(np.mean(pooled["path_cross_m"] ** 2))
            ),
            "path_vertical_rmse_m": float(
                np.sqrt(np.mean(pooled["path_vertical_m"] ** 2))
            ),
            "alos_projection_3d_rmse_m": float(
                np.sqrt(np.mean(pooled["alos_projection_error_3d_m"] ** 2))
            ),
            "speed_error_rmse_mps": float(
                np.sqrt(np.mean(pooled["speed_error_mps"] ** 2))
            ),
        },
        "saturation": {
            "material_saturated_rate": float(np.mean(pooled["material_saturated"])),
            "rate_clipped_rate": float(np.mean(pooled["rate_clipped"])),
            "material_rate_clipped_rate": float(
                np.mean(pooled["material_rate_clipped"])
            ),
            "magnitude_clipped_rate": float(np.mean(pooled["magnitude_clipped"])),
            "rudder_at_limit_rate": float(np.mean(pooled["rudder_fraction"] >= 0.99)),
            "stern_at_limit_rate": float(np.mean(pooled["stern_fraction"] >= 0.99)),
            "rpm_at_limit_rate": float(np.mean(pooled["rpm_fraction"] >= 0.99)),
        },
        "guidance_command_rates": {
            "heading_rate_abs_mean_rad_s": float(
                np.mean(np.abs(pooled["command_heading_rate_rad_s"]))
            ),
            "heading_rate_limit_hit_rate": float(
                np.mean(
                    np.abs(pooled["command_heading_rate_rad_s"])
                    >= 0.99 * heading_limit
                )
            )
            if heading_limit > 0
            else None,
            "pitch_rate_limit_hit_rate": float(
                np.mean(np.abs(pooled["command_pitch_rate_rad_s"]) >= 0.99 * pitch_limit)
            )
            if pitch_limit > 0
            else None,
            "speed_rate_limit_hit_rate": float(
                np.mean(np.abs(pooled["command_speed_rate_mps2"]) >= 0.99 * speed_limit)
            )
            if speed_limit > 0
            else None,
            "command_rate_limit_enabled": bool(
                guidance.command_rate_limit_enabled
                if guidance.command_rate_limit_enabled is not None
                else guidance.saturation_governor_enabled
            ),
        },
        "correlations_spearman": correlations,
        "notes": [
            "Actor-only checkpoint: no critic, so GAE advantage is not computable.",
            "term_variance_share is a critic-free credit proxy, not an advantage.",
            "residual_norm is ||BC normalized actuator action - PD/PI action from the ALOS command||.",
            "The PD/PI action uses actor-visible 12D state only; no true current is fed.",
            "No training, no checkpoint writes, no held-out seeds.",
        ],
    }

    return {
        "summary": summary,
        "episodes": episodes,
    }


def main() -> int:
    args = _parse_args()
    report = run(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "decomposition.json").write_text(
        json.dumps(report["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "per_seed.json").write_text(
        json.dumps(report["episodes"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    lines = [
        "# Goal27 Step 1 offline decomposition",
        "",
        f"- config: `{args.config}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- seeds: `{summary['seeds']}`",
        f"- steps: `{summary['steps_recorded']}`, episodes: `{summary['episode_count']}`",
        f"- success: `{summary['success_rate']}`",
        "",
        "## Reward terms",
        "",
        "| term | mean return | mean abs share | variance share | mean step |",
        "|---|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        summary["reward_terms"].items(),
        key=lambda item: abs(item[1]["mean_abs_share"] or 0.0),
        reverse=True,
    )
    for name, values in ranked:
        lines.append(
            f"| `{name}` | {values['mean_return']:.6g} | "
            f"{values['mean_abs_share']:.4f} | {values['mean_variance_share']:.4f} | "
            f"{values['mean_step_value']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- path 3-D RMSE: `{summary['errors']['path_3d_rmse_m']:.4f} m`",
            f"- residual norm mean/p95: `{summary['residuals']['mean_norm']:.4f}` / "
            f"`{summary['residuals']['p95_norm']:.4f}`",
            f"- material saturation rate: `{summary['saturation']['material_saturated_rate']:.4f}`",
            f"- actuator rate-clipped rate: `{summary['saturation']['rate_clipped_rate']:.4f}`",
            f"- rudder at limit: `{summary['saturation']['rudder_at_limit_rate']:.4f}`",
            "",
            "## Spearman correlations with residual norm",
            "",
            "| target | rho |",
            "|---|---:|",
        ]
    )
    for name, value in summary["correlations_spearman"].items():
        if name.startswith("residual_norm__") and value is not None:
            lines.append(f"| `{name.removeprefix('residual_norm__')}` | {value:.4f} |")
    lines.extend(
        [
            "",
            "Actor-only checkpoint: advantages are not computable; variance share is a",
            "critic-free credit proxy. See `decomposition.json` for the full record.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
