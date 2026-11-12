from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.geometry import body_to_world, quaternion_from_euler
from auv_lowlevel.batch.pc import PC_SAFETY_TERMINAL_KINDS, PCTerminalKind
from auv_lowlevel.config import load_config
from auv_lowlevel.outcome_contract import (
    DEFAULT_SUCCESS_FRAME,
    SUCCESS_FRAMES,
    score_success_contract,
    summarize_success_contract,
    validate_success_frame,
)
from auv_rl.ddpg.agent import DDPGAgent
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv
from auv_rl.training import load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Goal58 off-policy PC command-residual checkpoint"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--action-mode",
        choices=("policy", "zero-residual"),
        default="policy",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Diagnostic multiplier applied to normalized policy residuals before the wrapper",
    )
    parser.add_argument("--path-zero", action="store_true")
    parser.add_argument(
        "--eval-goal-radius-m",
        type=float,
        default=None,
        help="Eval-only GOAL termination radius; reward and config remain unchanged",
    )
    parser.add_argument(
        "--eval-goal-plane-miss-guard-radius-m",
        type=float,
        default=None,
        help="Eval-only plane-miss guard radius; defaults to the protocol radius",
    )
    parser.add_argument(
        "--eval-goal-plane-miss-dwell-steps",
        type=int,
        default=0,
        help="Eval-only consecutive steps required before plane-miss termination",
    )
    parser.add_argument(
        "--success-frame",
        choices=SUCCESS_FRAMES,
        default=DEFAULT_SUCCESS_FRAME,
        help=(
            "Eval-only success reporting frame; eta uses tight-radius ever-capture "
            "and never changes rollout, reward, or termination"
        ),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Optional directory for a read-only per-step trace",
    )
    parser.add_argument(
        "--rich-trace",
        action="store_true",
        help="Record read-only terminal state/command/actuator diagnostics in the trace",
    )
    return parser.parse_args()


def _validate_seeds(seed_start: int, seed_count: int) -> np.ndarray:
    seeds = np.arange(seed_start, seed_start + seed_count, dtype=np.int64)
    heldout = (seeds >= 100) & (seeds <= 149)
    if heldout.any():
        raise ValueError("held-out seeds 100-149 are closed for Goal58")
    return seeds


def _scalar(value: Any, index: int) -> float:
    if torch.is_tensor(value):
        return float(value.reshape(-1)[index].item())
    return float(np.asarray(value).reshape(-1)[index])


def _trace_put(
    trace: dict[str, list[np.ndarray]],
    name: str,
    rows: np.ndarray,
    values: Any,
) -> None:
    array = np.asarray(values)
    if array.ndim == 0:
        trace.setdefault(name, []).append(
            np.full(rows.size, array.item(), dtype=array.dtype)
        )
        return
    array = array[rows]
    trace.setdefault(name, []).append(array)


def _summary(
    rows: list[dict[str, Any]],
    *,
    action_mode: str,
    path_zero: bool,
    seed_start: int,
    seed_count: int,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
) -> dict[str, Any]:
    steps = np.asarray([row["steps"] for row in rows], dtype=np.float64)
    pre_goal_sq = np.asarray(
        [row["pre_goal_three_d_error_sq_sum"] for row in rows], dtype=np.float64
    )
    pre_goal_steps = np.asarray([row["pre_goal_steps"] for row in rows], dtype=np.float64)
    path_rmse = np.asarray([row["path_three_d_rmse_m"] for row in rows], dtype=np.float64)
    pre_goal_total = float(np.sum(pre_goal_steps))
    pre_goal_rmse = (
        None
        if pre_goal_total <= 0.0
        else float(np.sqrt(float(np.sum(pre_goal_sq)) / pre_goal_total))
    )
    residual_abs = np.asarray([row["residual_abs_mean"] for row in rows], dtype=np.float64)
    residual_ratio = np.asarray(
        [row["residual_l2_ratio"] for row in rows], dtype=np.float64
    )
    terminal_eta_distance = np.asarray(
        [row["terminal_eta_distance_m"] for row in rows], dtype=np.float64
    )
    terminal_p_nav_distance = np.asarray(
        [row["terminal_p_nav_distance_m"] for row in rows], dtype=np.float64
    )
    terminal_signed_offset = np.asarray(
        [row["terminal_signed_goal_offset_m"] for row in rows], dtype=np.float64
    )
    terminal_vertical_offset = np.asarray(
        [row["terminal_vertical_offset_m"] for row in rows], dtype=np.float64
    )
    terminal_horizontal_offset = np.asarray(
        [row["terminal_horizontal_offset_m"] for row in rows], dtype=np.float64
    )
    summary = {
        "episodes": len(rows),
        "seed_start": seed_start,
        "seed_count": seed_count,
        "action_mode": action_mode,
        "path_zero": bool(path_zero),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "goal_plane_miss_rate": float(np.mean([row["goal_plane_miss"] for row in rows])),
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
        "safety_termination_rate": float(np.mean([row["safety"] for row in rows])),
        "pre_goal_3d_rmse_m": pre_goal_rmse,
        "path_three_d_rmse_m": float(np.mean(path_rmse)),
        "mean_reward": float(np.mean([row["mean_reward"] for row in rows])),
        "material_saturation_rate": float(
            np.sum([row["material_saturation_steps"] for row in rows]) / np.sum(steps)
        ),
        "residual_abs_mean": float(np.mean(residual_abs)),
        "residual_l2_ratio_mean": float(np.mean(residual_ratio)),
        "residual_scale_clip_rate": float(
            np.sum([row["residual_scale_clip_steps"] for row in rows]) / np.sum(steps)
        ),
        "residual_rate_clip_rate": float(
            np.sum([row["residual_rate_clip_steps"] for row in rows]) / np.sum(steps)
        ),
        "terminal_eta_capture_rate": float(
            np.mean([row["terminal_eta_capture"] for row in rows])
        ),
        "terminal_eta_distance_mean_m": float(np.mean(terminal_eta_distance)),
        "terminal_eta_distance_median_m": float(np.median(terminal_eta_distance)),
        "terminal_p_nav_distance_mean_m": float(np.mean(terminal_p_nav_distance)),
        "terminal_p_nav_distance_median_m": float(np.median(terminal_p_nav_distance)),
        "terminal_signed_goal_offset_mean_m": float(np.mean(terminal_signed_offset)),
        "terminal_vertical_offset_mean_m": float(np.mean(terminal_vertical_offset)),
        "terminal_horizontal_offset_mean_m": float(np.mean(terminal_horizontal_offset)),
        "action_abs_mean": float(np.mean([row["action_abs_mean"] for row in rows])),
        "action_saturation_fraction": float(
            np.mean([row["action_saturation_fraction"] for row in rows])
        ),
        "residual_gate_mean": float(np.mean([row["residual_gate_mean"] for row in rows])),
        "gate_active_rate": float(np.mean([row["gate_active_rate"] for row in rows])),
        "residual_l2_ratio_gate_active": float(
            np.mean([row["residual_l2_ratio_gate_active"] for row in rows])
        ),
        "finite": bool(
            np.all(np.isfinite(steps))
            and np.all(np.isfinite(pre_goal_sq))
            and np.all(np.isfinite(path_rmse))
            and np.all(np.isfinite(terminal_eta_distance))
            and np.all(np.isfinite(terminal_p_nav_distance))
        ),
    }
    summary.update(
        summarize_success_contract(rows, success_frame=success_frame)
    )
    return summary


def main() -> int:
    args = _parse_args()
    success_frame = validate_success_frame(args.success_frame)
    if not np.isfinite(args.residual_scale) or args.residual_scale <= 0.0:
        raise ValueError("--residual-scale must be finite and positive")
    payload = load_training_payload(args.config)
    observation = dict(payload.get("observation", {}))
    training = dict(payload.get("training", {}))
    evaluation = dict(payload.get("evaluation", {}))
    seeds = _validate_seeds(args.seed_start, args.seed_count)
    environment_config = load_config(args.environment_config)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    env = PCCommandResidualOffPolicyEnv(
        environment_config,
        num_envs=len(seeds),
        device=args.device,
        dtype=dtype,
        path_dim=int(observation.get("path_dim", 8)),
        path_zero=bool(args.path_zero),
        validate_finite=False,
        eval_goal_radius_m=args.eval_goal_radius_m,
        eval_goal_plane_miss_guard_radius_m=(
            args.eval_goal_plane_miss_guard_radius_m
        ),
        eval_goal_plane_miss_dwell_steps=args.eval_goal_plane_miss_dwell_steps,
    )
    env.set_action_mode(args.action_mode)
    agent: DDPGAgent | None = None
    checkpoint_sha256: str | None = None
    if args.action_mode == "policy":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for policy evaluation")
        agent = DDPGAgent.load(
            args.checkpoint,
            device=args.device,
            expected_observation_version=env.observation_version,
        )
        import hashlib

        checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    observations, _ = env.reset(seeds.tolist())
    if agent is not None:
        for env_id in range(env.num_envs):
            agent.begin_episode_for_env(env_id, int(env.episode_ids[env_id]))

    active = np.ones(env.num_envs, dtype=bool)
    steps = np.zeros(env.num_envs, dtype=np.int64)
    reward_sum = np.zeros(env.num_envs, dtype=np.float64)
    pre_goal_sq_sum = np.zeros(env.num_envs, dtype=np.float64)
    pre_goal_steps = np.zeros(env.num_envs, dtype=np.int64)
    path_sq_sum = np.zeros(env.num_envs, dtype=np.float64)
    path_steps = np.zeros(env.num_envs, dtype=np.int64)
    nav_error_sum_ned = np.zeros((env.num_envs, 3), dtype=np.float64)
    nav_error_steps = np.zeros(env.num_envs, dtype=np.int64)
    residual_abs_sum = np.zeros(env.num_envs, dtype=np.float64)
    residual_l2_sum = np.zeros(env.num_envs, dtype=np.float64)
    base_l2_sum = np.zeros(env.num_envs, dtype=np.float64)
    gate_active_residual_l2_sum = np.zeros(env.num_envs, dtype=np.float64)
    gate_active_base_l2_sum = np.zeros(env.num_envs, dtype=np.float64)
    gate_sum = np.zeros(env.num_envs, dtype=np.float64)
    gate_count = np.zeros(env.num_envs, dtype=np.int64)
    gate_active_steps = np.zeros(env.num_envs, dtype=np.int64)
    action_abs_sum = np.zeros(env.num_envs, dtype=np.float64)
    action_values = np.zeros(env.num_envs, dtype=np.int64)
    action_saturation_values = np.zeros(env.num_envs, dtype=np.int64)
    residual_count = np.zeros(env.num_envs, dtype=np.int64)
    residual_scale_clip_steps = np.zeros(env.num_envs, dtype=np.int64)
    residual_rate_clip_steps = np.zeros(env.num_envs, dtype=np.int64)
    material_saturation_steps = np.zeros(env.num_envs, dtype=np.int64)
    success = np.zeros(env.num_envs, dtype=bool)
    goal_plane_miss = np.zeros(env.num_envs, dtype=bool)
    timeout = np.zeros(env.num_envs, dtype=bool)
    safety = np.zeros(env.num_envs, dtype=bool)
    terminal_kind = np.zeros(env.num_envs, dtype=np.int64)
    endpoint_entry_reached = np.zeros(env.num_envs, dtype=bool)
    terminal_eta_distance = np.full(env.num_envs, np.nan, dtype=np.float64)
    terminal_p_nav_distance = np.full(env.num_envs, np.nan, dtype=np.float64)
    terminal_signed_offset = np.full(env.num_envs, np.nan, dtype=np.float64)
    terminal_vertical_offset = np.full(env.num_envs, np.nan, dtype=np.float64)
    terminal_horizontal_offset = np.full(env.num_envs, np.nan, dtype=np.float64)
    terminal_eta_capture = np.zeros(env.num_envs, dtype=bool)
    pnav_goal_capture_seen = np.zeros(env.num_envs, dtype=bool)
    eta_goal_capture_seen = np.zeros(env.num_envs, dtype=bool)
    eta_goal_capture_at_done = np.zeros(env.num_envs, dtype=bool)
    tight_goal_radius_m = (
        float(args.eval_goal_radius_m)
        if args.eval_goal_radius_m is not None
        else float(environment_config.plant.goal_radius_m)
    )
    actor_output_scale = float(training.get("actor_output_scale", 1.0))
    action_saturation_threshold = 0.9 * actor_output_scale
    segment_hint = env.inner.path.project(env.inner.eta[:, :3]).segment_index
    max_steps = int(env.max_episode_steps)
    endpoint_stop = bool(
        environment_config.termination.pc_goal_plane_miss_termination_enabled
    )
    trace: dict[str, list[np.ndarray]] = {}
    if args.trace_output is not None:
        args.trace_output.mkdir(parents=True, exist_ok=True)

    for step_index in range(max_steps):
        active_before = active.copy()
        active_ids = np.flatnonzero(active_before).tolist()
        if not active_ids:
            break
        actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
        if agent is not None:
            actions[active_before] = agent.act_batch(
                active_ids,
                observations[active_before],
                evaluate=True,
            )
            if args.residual_scale != 1.0:
                actions *= args.residual_scale
        with torch.inference_mode():
            next_observations, rewards, terminated, truncated, info = env.step(
                actions,
                auto_reset=False,
                advance_mask=active_before,
            )

        inner = env.inner
        assert inner.eta is not None and inner.sensor_output is not None
        projection = inner.path.project(
            inner.eta[:, :3],
            segment_hint=segment_hint,
            search_radius=inner.reset_projection_search_radius,
        )
        segment_hint = projection.segment_index
        goal_ned_m = inner.path.goal_ned_m_for(env.num_envs)
        goal_tangent_ned = inner.path.goal_tangent_ned_for(env.num_envs)
        pnav_goal_distance_m = (
            torch.linalg.norm(inner.sensor_output.p_nav - goal_ned_m, dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        eta_goal_distance_m = (
            torch.linalg.norm(inner.eta[:, :3] - goal_ned_m, dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        nav_error_ned = (
            (inner.sensor_output.p_nav - inner.eta[:, :3])
            .detach()
            .cpu()
            .numpy()
        )
        nav_error_sum_ned[active_before] += nav_error_ned[active_before]
        nav_error_steps[active_before] += 1
        pnav_goal_capture_seen[
            active_before & (pnav_goal_distance_m <= tight_goal_radius_m + 1e-12)
        ] = True
        eta_goal_capture_seen[
            active_before & (eta_goal_distance_m <= tight_goal_radius_m + 1e-12)
        ] = True
        position_source = environment_config.termination.position_source
        if position_source == "eta":
            termination_position = inner.eta[:, :3]
        elif position_source == "p_nav":
            termination_position = inner.sensor_output.p_nav
        else:
            termination_position = (
                inner.sensor_output.p_nav if endpoint_stop else inner.eta[:, :3]
            )
        goal_distance = torch.linalg.norm(
            termination_position - goal_ned_m, dim=1
        ).detach().cpu().numpy()
        signed_offset = (
            ((termination_position - goal_ned_m) * goal_tangent_ned).sum(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        endpoint_now = (goal_distance <= float(environment_config.plant.goal_radius_m) + 1e-12) | (
            signed_offset >= 0.0
        )
        endpoint_entry_reached[active_before & endpoint_now] = True
        pre_goal_mask = active_before & ~endpoint_entry_reached & ~endpoint_now
        error_sq = projection.error_3d_m.square().detach().cpu().numpy()
        path_sq_sum[active_before] += error_sq[active_before]
        path_steps[active_before] += 1
        pre_goal_sq_sum[pre_goal_mask] += error_sq[pre_goal_mask]
        pre_goal_steps[pre_goal_mask] += 1

        applied = env.last_applied_residual
        base = env.last_base_command
        gate = env.last_residual_gate
        residual_abs_sum[active_before] += np.sum(np.abs(applied[active_before]), axis=1)
        residual_l2_sum[active_before] += np.linalg.norm(applied[active_before], axis=1)
        base_l2_sum[active_before] += np.linalg.norm(base[active_before], axis=1)
        gate_sum[active_before] += gate[active_before]
        gate_count[active_before] += 1
        gate_active = active_before & (gate > 0.5)
        gate_active_steps[gate_active] += 1
        gate_active_residual_l2_sum[gate_active] += np.linalg.norm(
            applied[gate_active], axis=1
        )
        gate_active_base_l2_sum[gate_active] += np.linalg.norm(base[gate_active], axis=1)
        action_abs_sum[active_before] += np.sum(np.abs(actions[active_before]), axis=1)
        action_values[active_before] += actions.shape[1]
        action_saturation_values[active_before] += np.sum(
            np.abs(actions[active_before]) >= action_saturation_threshold, axis=1
        )
        residual_count[active_before] += 1
        residual_scale_clip_steps[active_before] += np.any(
            env.last_scale_clipped[active_before], axis=1
        )
        residual_rate_clip_steps[active_before] += np.any(
            env.last_rate_clipped[active_before], axis=1
        )
        material_saturation = info["material_saturated"].detach().cpu().numpy()
        material_saturation_steps[active_before] += material_saturation[active_before]
        steps[active_before] += 1
        reward_sum[active_before] += rewards[active_before]

        done_now = active_before & (terminated | truncated)
        if bool(done_now.any()):
            kinds = info["terminal_kind"].detach().cpu().numpy()
            assert inner.eta is not None
            for index in np.flatnonzero(done_now).tolist():
                kind = int(kinds[index])
                terminal_kind[index] = kind
                success[index] = kind == int(PCTerminalKind.GOAL)
                goal_plane_miss[index] = kind == int(PCTerminalKind.GOAL_PLANE_MISS)
                timeout[index] = kind == int(PCTerminalKind.TIME_LIMIT)
                safety[index] = PCTerminalKind(kind) in PC_SAFETY_TERMINAL_KINDS
                delta = inner.eta[index, :3] - goal_ned_m[index]
                signed = float(torch.dot(delta, goal_tangent_ned[index]).detach().cpu())
                distance = float(torch.linalg.norm(delta).detach().cpu())
                terminal_eta_distance[index] = distance
                assert inner.sensor_output is not None
                p_nav_delta = inner.sensor_output.p_nav[index, :3] - goal_ned_m[index]
                terminal_p_nav_distance[index] = float(
                    torch.linalg.norm(p_nav_delta).detach().cpu()
                )
                terminal_signed_offset[index] = signed
                terminal_vertical_offset[index] = float(delta[2].detach().cpu())
                terminal_horizontal_offset[index] = float(
                    torch.linalg.norm(delta[:2]).detach().cpu()
                )
                terminal_eta_capture[index] = distance <= float(
                    environment_config.plant.goal_radius_m
                ) + 1e-12
                eta_goal_capture_at_done[index] = distance <= (
                    tight_goal_radius_m + 1e-12
                )
        if args.trace_output is not None:
            rows = np.flatnonzero(active_before)
            if rows.size:
                _trace_put(trace, "step", rows, step_index)
                _trace_put(trace, "env_index", rows, np.arange(env.num_envs, dtype=np.int64))
                _trace_put(trace, "episode_id", rows, env.episode_ids)
                _trace_put(trace, "seed", rows, seeds)
                _trace_put(trace, "active", rows, active_before)
                _trace_put(trace, "action", rows, actions)
                _trace_put(trace, "residual", rows, applied)
                _trace_put(trace, "base_command", rows, base)
                _trace_put(trace, "gate", rows, gate)
                _trace_put(trace, "scale_clipped", rows, env.last_scale_clipped)
                _trace_put(trace, "rate_clipped", rows, env.last_rate_clipped)
                _trace_put(trace, "reward", rows, rewards)
                _trace_put(trace, "goal_distance", rows, goal_distance)
                _trace_put(trace, "signed_offset", rows, signed_offset)
                _trace_put(trace, "endpoint_now", rows, endpoint_now)
                _trace_put(trace, "material_saturated", rows, material_saturation)
                _trace_put(trace, "path_error_3d", rows, projection.error_3d_m.detach().cpu().numpy())
                _trace_put(
                    trace,
                    "path_horizontal_error",
                    rows,
                    projection.horizontal_error_m.detach().cpu().numpy(),
                )
                _trace_put(
                    trace,
                    "path_vertical_error",
                    rows,
                    projection.vertical_error_m.detach().cpu().numpy(),
                )
                _trace_put(
                    trace,
                    "path_progress",
                    rows,
                    projection.progress_m.detach().cpu().numpy(),
                )
                _trace_put(trace, "eta_ned", rows, inner.eta[:, :3].detach().cpu().numpy())
                _trace_put(
                    trace,
                    "p_nav_ned",
                    rows,
                    inner.sensor_output.p_nav.detach().cpu().numpy(),
                )
                _trace_put(trace, "goal_ned", rows, goal_ned_m.detach().cpu().numpy())
                _trace_put(
                    trace,
                    "goal_tangent_ned",
                    rows,
                    goal_tangent_ned.detach().cpu().numpy(),
                )
                _trace_put(
                    trace,
                    "terminal_kind",
                    rows,
                    info["terminal_kind"].detach().cpu().numpy(),
                )
                _trace_put(trace, "done", rows, done_now)
                if args.rich_trace:
                    assert inner.t_s is not None and inner.nu is not None
                    attitude_quat = quaternion_from_euler(
                        inner.eta[:, 3], inner.eta[:, 4], inner.eta[:, 5]
                    )
                    ground_velocity_ned = body_to_world(
                        info["ground_velocity_body_mps"], attitude_quat
                    )
                    _trace_put(trace, "time_s", rows, inner.t_s.detach().cpu().numpy())
                    _trace_put(trace, "eta_rpy", rows, inner.eta[:, 3:6].detach().cpu().numpy())
                    _trace_put(trace, "nu", rows, inner.nu.detach().cpu().numpy())
                    _trace_put(
                        trace,
                        "ground_velocity_body_mps",
                        rows,
                        info["ground_velocity_body_mps"].detach().cpu().numpy(),
                    )
                    _trace_put(
                        trace,
                        "ground_velocity_ned_mps",
                        rows,
                        ground_velocity_ned.detach().cpu().numpy(),
                    )
                    _trace_put(
                        trace,
                        "actual_actuator",
                        rows,
                        info["actual_actuator"].detach().cpu().numpy(),
                    )
                    _trace_put(
                        trace,
                        "command_physical",
                        rows,
                        info["command_physical"].detach().cpu().numpy(),
                    )
                    _trace_put(
                        trace,
                        "command_water_relative_speed_mps",
                        rows,
                        info["command_water_relative_speed_mps"].detach().cpu().numpy(),
                    )
                    _trace_put(
                        trace,
                        "command_reachability_feasible",
                        rows,
                        info["command_reachability_feasible"].detach().cpu().numpy(),
                    )
                    _trace_put(
                        trace,
                        "guidance_terminal_phase",
                        rows,
                        info["guidance_terminal_phase"].detach().cpu().numpy(),
                    )
                    _trace_put(
                        trace,
                        "guidance_crab_angle_rad",
                        rows,
                        info["guidance_crab_angle_rad"].detach().cpu().numpy(),
                    )
                    _trace_put(
                        trace,
                        "guidance_speed_scale",
                        rows,
                        info["guidance_speed_scale"].detach().cpu().numpy(),
                    )
                for component, component_values in env.last_reward_breakdown.items():
                    if component_values is not None:
                        _trace_put(trace, f"reward_{component}", rows, component_values)
        active[done_now] = False
        observations = next_observations

    if np.any(np.isnan(terminal_eta_distance)):
        assert inner.eta is not None
        for index in np.flatnonzero(np.isnan(terminal_eta_distance)).tolist():
            delta = inner.eta[index, :3] - goal_ned_m[index]
            signed = float(torch.dot(delta, goal_tangent_ned[index]).detach().cpu())
            distance = float(torch.linalg.norm(delta).detach().cpu())
            terminal_eta_distance[index] = distance
            assert inner.sensor_output is not None
            p_nav_delta = inner.sensor_output.p_nav[index, :3] - goal_ned_m[index]
            terminal_p_nav_distance[index] = float(
                torch.linalg.norm(p_nav_delta).detach().cpu()
            )
            terminal_signed_offset[index] = signed
            terminal_vertical_offset[index] = float(delta[2].detach().cpu())
            terminal_horizontal_offset[index] = float(
                torch.linalg.norm(delta[:2]).detach().cpu()
            )
            terminal_eta_capture[index] = distance <= float(
                environment_config.plant.goal_radius_m
            ) + 1e-12
            eta_goal_capture_at_done[index] = distance <= (
                tight_goal_radius_m + 1e-12
            )

    rows: list[dict[str, Any]] = []
    assert inner.plant_scales is not None
    plant_scale_values = inner.plant_scales.detach().cpu().numpy()
    for index, seed in enumerate(seeds.tolist()):
        count = int(residual_count[index])
        row = {
            "seed": int(seed),
            "plant_scale_factors": {
                "drag_scale": float(plant_scale_values[index, 0]),
                "thrust_scale": float(plant_scale_values[index, 1]),
                "fin_scale": float(plant_scale_values[index, 2]),
                "added_mass_scale": float(plant_scale_values[index, 3]),
                "actuator_time_scale": float(plant_scale_values[index, 4]),
            },
            "steps": int(steps[index]),
            "success": bool(success[index]),
            "goal_plane_miss": bool(goal_plane_miss[index]),
            "timeout": bool(timeout[index]),
            "safety": bool(safety[index]),
            "terminal_kind": int(terminal_kind[index]),
            "mean_reward": float(reward_sum[index] / max(int(steps[index]), 1)),
            "pre_goal_three_d_error_sq_sum": float(pre_goal_sq_sum[index]),
            "pre_goal_steps": int(pre_goal_steps[index]),
            "pre_goal_3d_rmse_m": (
                None
                if pre_goal_steps[index] <= 0
                else float(np.sqrt(pre_goal_sq_sum[index] / pre_goal_steps[index]))
            ),
            "path_three_d_rmse_m": float(
                np.sqrt(max(path_sq_sum[index], 0.0) / max(int(path_steps[index]), 1))
            ),
            "nav_error_mean_ned_m": [
                float(value)
                for value in nav_error_sum_ned[index]
                / max(int(nav_error_steps[index]), 1)
            ],
            "material_saturation_steps": int(material_saturation_steps[index]),
            "residual_abs_mean": float(residual_abs_sum[index] / max(count, 1)),
            "residual_l2_ratio": float(
                residual_l2_sum[index] / max(float(base_l2_sum[index]), 1e-12)
            ),
            "residual_scale_clip_steps": int(residual_scale_clip_steps[index]),
            "residual_rate_clip_steps": int(residual_rate_clip_steps[index]),
            "terminal_eta_capture": bool(terminal_eta_capture[index]),
            "goal_capture_pnav": bool(pnav_goal_capture_seen[index]),
            "goal_capture_eta": bool(eta_goal_capture_seen[index]),
            "goal_capture_eta_at_done": bool(eta_goal_capture_at_done[index]),
            "terminal_eta_distance_m": float(terminal_eta_distance[index]),
            "terminal_p_nav_distance_m": float(terminal_p_nav_distance[index]),
            "terminal_signed_goal_offset_m": float(terminal_signed_offset[index]),
            "terminal_vertical_offset_m": float(terminal_vertical_offset[index]),
            "terminal_horizontal_offset_m": float(terminal_horizontal_offset[index]),
            "action_abs_mean": float(
                action_abs_sum[index] / max(int(action_values[index]), 1)
            ),
            "action_saturation_fraction": float(
                action_saturation_values[index] / max(int(action_values[index]), 1)
            ),
            "residual_gate_mean": float(
                gate_sum[index] / max(int(gate_count[index]), 1)
            ),
            "gate_active_rate": float(
                gate_active_steps[index] / max(int(residual_count[index]), 1)
            ),
            "residual_l2_ratio_gate_active": float(
                gate_active_residual_l2_sum[index]
                / max(float(gate_active_base_l2_sum[index]), 1e-12)
            ),
        }
        row.update(
            score_success_contract(
                p_nav_tight_success=bool(success[index]),
                eta_tight_ever=bool(eta_goal_capture_seen[index]),
                eta_tight_at_done=bool(eta_goal_capture_at_done[index]),
                success_frame=success_frame,
            )
        )
        rows.append(row)

    summary = _summary(
        rows,
        action_mode=args.action_mode,
        path_zero=bool(args.path_zero),
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        success_frame=success_frame,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "episodes.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    payload_out = {
        "schema": "auv_pc_command_residual_offpolicy_eval.v1",
        "training_config": str(args.config),
        "environment_config": str(args.environment_config),
        "environment_config_sha256": environment_config.config_sha256,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "device": args.device,
        "dtype": args.dtype,
        "termination_mode": evaluation.get("termination_mode", "endpoint_stop"),
        "termination_position_source": environment_config.termination.position_source,
        "sensor_profile": (
            None
            if environment_config.sensors is None
            else environment_config.sensors.profile
        ),
        "ideal_actor_observations": bool(
            environment_config.observation.ideal_actor_observations
        ),
        "plant_randomization": environment_config.plant_randomization.as_dict(),
        "eval_goal_radius_m": (
            float(args.eval_goal_radius_m)
            if args.eval_goal_radius_m is not None
            else float(environment_config.plant.goal_radius_m)
        ),
        "eval_goal_plane_miss_guard_radius_m": (
            float(args.eval_goal_plane_miss_guard_radius_m)
            if args.eval_goal_plane_miss_guard_radius_m is not None
            else float(environment_config.plant.goal_radius_m)
        ),
        "eval_goal_plane_miss_dwell_steps": int(
            args.eval_goal_plane_miss_dwell_steps
        ),
        "path_zero": bool(args.path_zero),
        "residual_scale_multiplier": float(args.residual_scale),
        "success_frame": success_frame,
        "outcome_contract": {
            "deployment_primary": "p_nav_tight_success",
            "research_primary": "eta_tight_ever",
            "secondary": "eta_tight_at_done",
            "protocol_radius_label": "terminal_eta_capture",
        },
        "summary": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.trace_output is not None:
        arrays = {
            name: np.concatenate(parts, axis=0)
            for name, parts in trace.items()
            if parts
        }
        np.savez_compressed(args.trace_output / "trace.npz", **arrays)
        finite = all(
            bool(np.all(np.isfinite(values)))
            for values in arrays.values()
            if np.issubdtype(values.dtype, np.floating)
        )
        trace_meta = {
            "schema": "auv_pc_command_residual_offpolicy_trace.v1",
            "training_config": str(args.config),
            "environment_config": str(args.environment_config),
            "environment_config_sha256": environment_config.config_sha256,
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "device": args.device,
            "dtype": args.dtype,
            "action_mode": args.action_mode,
            "termination_position_source": environment_config.termination.position_source,
            "sensor_profile": (
                None
                if environment_config.sensors is None
                else environment_config.sensors.profile
            ),
            "ideal_actor_observations": bool(
                environment_config.observation.ideal_actor_observations
            ),
            "plant_randomization": environment_config.plant_randomization.as_dict(),
            "rich_trace": bool(args.rich_trace),
            "path_zero": bool(args.path_zero),
            "residual_scale_multiplier": float(args.residual_scale),
            "eval_goal_radius_m": (
                float(args.eval_goal_radius_m)
                if args.eval_goal_radius_m is not None
                else float(environment_config.plant.goal_radius_m)
            ),
            "eval_goal_plane_miss_guard_radius_m": (
                float(args.eval_goal_plane_miss_guard_radius_m)
                if args.eval_goal_plane_miss_guard_radius_m is not None
                else float(environment_config.plant.goal_radius_m)
            ),
            "eval_goal_plane_miss_dwell_steps": int(
                args.eval_goal_plane_miss_dwell_steps
            ),
            "seed_start": int(args.seed_start),
            "seed_count": int(args.seed_count),
            "episodes": int(len(seeds)),
            "trace_fields": sorted(arrays),
            "finite": finite,
        }
        (args.trace_output / "trace_meta.json").write_text(
            json.dumps(trace_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(payload_out, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
