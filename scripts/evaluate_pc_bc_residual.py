#!/usr/bin/env python3
"""Deterministic closed-loop evaluation for a PA residual-BC actor.

The BC checkpoint is applied as an actuator residual around the ALOS base
controller.  This is a diagnostic for whether the BC prior itself carries a
useful signal; it is not an RL training result and does not open held-out seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.pc import PC_SAFETY_TERMINAL_KINDS, PCTerminalKind
from auv_lowlevel.batch.pc_bc import PCBCPolicy
from auv_lowlevel.config import load_config
from auv_lowlevel.outcome_contract import (
    DEFAULT_SUCCESS_FRAME,
    SUCCESS_FRAMES,
    score_success_contract,
    summarize_success_contract,
    validate_success_frame,
)
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv
from auv_rl.training import load_training_payload

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--action-mode", choices=("policy", "zero-residual"), default="policy")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--actuator-residual-scale", type=float, default=None)
    parser.add_argument("--path-zero", action="store_true")
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--disable-curriculum-direct-command",
        action="store_true",
        help="disable the C0/C1 direct-command scaffold for path-following evaluation",
    )
    parser.add_argument("--eval-goal-radius-m", type=float, default=None)
    parser.add_argument("--eval-goal-plane-miss-guard-radius-m", type=float, default=None)
    parser.add_argument("--eval-goal-plane-miss-dwell-steps", type=int, default=0)
    parser.add_argument("--success-frame", choices=SUCCESS_FRAMES, default=DEFAULT_SUCCESS_FRAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_eval_seeds(seed_start: int, seed_count: int) -> np.ndarray:
    if seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    seeds = np.arange(seed_start, seed_start + seed_count, dtype=np.int64)
    if ((seeds >= 100) & (seeds <= 149)).any():
        raise ValueError("held-out seeds 100-149 are closed")
    return seeds


def _stack_history(history: np.ndarray) -> np.ndarray:
    return history.reshape(history.shape[0], -1)


def main() -> int:
    args = _parse_args()
    success_frame = validate_success_frame(args.success_frame)
    if not np.isfinite(args.residual_scale) or args.residual_scale <= 0.0:
        raise ValueError("--residual-scale must be finite and positive")
    payload = load_training_payload(args.config)
    observation = dict(payload.get("observation", {}))
    training = dict(payload.get("training", {}))
    action_interface = str(payload.get("action_interface", "command"))
    if action_interface not in {"command", "actuator"}:
        raise ValueError("config.action_interface must be 'command' or 'actuator'")
    actuator_residual_scale = float(
        args.actuator_residual_scale
        if args.actuator_residual_scale is not None
        else payload.get("actuator_residual_scale", 0.25)
    )
    actuator_max_action = float(payload.get("actuator_max_action", 1.0))
    environment_value = payload.get("environment_config")
    if not isinstance(environment_value, str) or not environment_value:
        raise ValueError("config.environment_config must be a non-empty string")
    environment_path = (
        _resolve(args.environment_config)
        if args.environment_config is not None
        else _resolve(Path(environment_value))
    )
    environment_config = load_config(environment_path)
    history_length = int(
        args.history_length
        if args.history_length is not None
        else training.get("history_length", 10)
    )
    if history_length <= 0:
        raise ValueError("--history-length must be positive")
    seeds = _validate_eval_seeds(args.seed_start, args.seed_count)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    env = PCCommandResidualOffPolicyEnv(
        environment_config,
        num_envs=len(seeds),
        device=args.device,
        dtype=dtype,
        path_dim=int(observation.get("path_dim", 8)),
        path_zero=bool(args.path_zero),
        action_interface=action_interface,
        actuator_residual_scale=actuator_residual_scale,
        actuator_max_action=actuator_max_action,
        validate_finite=False,
        eval_goal_radius_m=args.eval_goal_radius_m,
        eval_goal_plane_miss_guard_radius_m=args.eval_goal_plane_miss_guard_radius_m,
        eval_goal_plane_miss_dwell_steps=args.eval_goal_plane_miss_dwell_steps,
        curriculum_direct_commands_enabled=not args.disable_curriculum_direct_command,
    )
    env.set_action_mode(args.action_mode)
    flat_dim = int(env.observation_dim)

    checkpoint = _resolve(args.checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)
    policy = PCBCPolicy(checkpoint, device=args.device)
    if policy.model.path_dim not in (0, 8):
        raise ValueError(
            "PA residual BC evaluator requires a path-free or path8 checkpoint; "
            f"got path_dim={policy.model.path_dim}"
        )
    if policy.model.actor_command_dim != 0:
        raise ValueError("PA residual BC evaluator requires actor_command_dim=0")
    if args.action_mode != "policy":
        raise ValueError("PA residual BC evaluator requires --action-mode policy")

    observations, _ = env.reset(seeds.tolist())
    active = np.ones(env.num_envs, dtype=bool)
    steps = np.zeros(env.num_envs, dtype=np.int64)
    reward_sum = np.zeros(env.num_envs, dtype=np.float64)
    path_sq_sum = np.zeros(env.num_envs, dtype=np.float64)
    path_steps = np.zeros(env.num_envs, dtype=np.int64)
    residual_l2_sum = np.zeros(env.num_envs, dtype=np.float64)
    base_l2_sum = np.zeros(env.num_envs, dtype=np.float64)
    gate_active_residual_l2_sum = np.zeros(env.num_envs, dtype=np.float64)
    gate_active_base_l2_sum = np.zeros(env.num_envs, dtype=np.float64)
    action_abs_max = np.zeros(env.num_envs, dtype=np.float64)
    residual_abs_max = np.zeros(env.num_envs, dtype=np.float64)
    terminal_kind = np.full(env.num_envs, -1, dtype=np.int64)
    terminal_eta_distance = np.full(env.num_envs, np.nan, dtype=np.float64)
    terminal_p_nav_distance = np.full(env.num_envs, np.nan, dtype=np.float64)
    eta_capture_seen = np.zeros(env.num_envs, dtype=bool)
    p_nav_capture_seen = np.zeros(env.num_envs, dtype=bool)
    eta_capture_at_done = np.zeros(env.num_envs, dtype=bool)
    tight_radius = (
        float(args.eval_goal_radius_m)
        if args.eval_goal_radius_m is not None
        else float(environment_config.plant.goal_radius_m)
    )
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_episode_steps)
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    for _ in range(max_steps):
        active_before = active.copy()
        if not bool(active_before.any()):
            break
        actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
        if bool(active_before.any()):
            with torch.inference_mode():
                observation_batch = env.inner.observation
                actor_path = observation_batch.actor_path
                if actor_path is not None:
                    actor_path = actor_path[:, : policy.model.path_dim]
                bc_actions = policy.act_inference(
                    observation_batch.actor_state[active_before],
                    None,
                    actor_path[active_before] if actor_path is not None else None,
                    None,
                )
            actions[active_before] = bc_actions.detach().cpu().numpy()
            actions *= args.residual_scale
        with torch.inference_mode():
            next_observations, rewards, terminated, truncated, info = env.step(
                actions,
                auto_reset=False,
                advance_mask=active_before,
            )

        inner = env.inner
        assert inner.eta is not None and inner.sensor_output is not None
        goal_ned = inner.path.goal_ned_m_for(env.num_envs)
        eta_distance = torch.linalg.norm(inner.eta[:, :3] - goal_ned, dim=1).detach().cpu().numpy()
        p_nav_distance = (
            torch.linalg.norm(inner.sensor_output.p_nav - goal_ned, dim=1).detach().cpu().numpy()
        )
        projection = inner.path.project(inner.eta[:, :3])
        error_sq = projection.error_3d_m.square().detach().cpu().numpy()
        path_sq_sum[active_before] += error_sq[active_before]
        path_steps[active_before] += 1
        eta_capture_seen[active_before & (eta_distance <= tight_radius + 1e-12)] = True
        p_nav_capture_seen[active_before & (p_nav_distance <= tight_radius + 1e-12)] = True

        applied = env.last_applied_residual
        base = env.last_base_command
        gate = env.last_residual_gate
        applied_norm = np.linalg.norm(applied, axis=1)
        base_norm = np.linalg.norm(base, axis=1)
        residual_l2_sum[active_before] += applied_norm[active_before]
        base_l2_sum[active_before] += base_norm[active_before]
        gate_active = active_before & (gate > 0.5)
        gate_active_residual_l2_sum[gate_active] += applied_norm[gate_active]
        gate_active_base_l2_sum[gate_active] += base_norm[gate_active]
        action_abs_max[active_before] = np.maximum(
            action_abs_max[active_before],
            np.max(np.abs(actions[active_before]), axis=1),
        )
        residual_abs_max[active_before] = np.maximum(
            residual_abs_max[active_before],
            np.max(np.abs(applied[active_before]), axis=1),
        )
        steps[active_before] += 1
        reward_sum[active_before] += rewards[active_before]

        done_now = active_before & (terminated | truncated)
        if bool(done_now.any()):
            kinds = info["terminal_kind"].detach().cpu().numpy()
            for index in np.flatnonzero(done_now).tolist():
                terminal_kind[index] = int(kinds[index])
                terminal_eta_distance[index] = float(eta_distance[index])
                terminal_p_nav_distance[index] = float(p_nav_distance[index])
                eta_capture_at_done[index] = bool(eta_distance[index] <= tight_radius + 1e-12)

        active[done_now] = False

    if bool(active.any()):
        raise RuntimeError("evaluation ended before every environment terminated")

    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds.tolist()):
        kind = int(terminal_kind[index])
        p_nav_success = kind == int(PCTerminalKind.GOAL)
        row = {
            "seed": int(seed),
            "steps": int(steps[index]),
            "reward_total": float(reward_sum[index]),
            "terminal_kind": kind,
            "success": bool(p_nav_success),
            "goal_plane_miss": kind == int(PCTerminalKind.GOAL_PLANE_MISS),
            "timeout": kind == int(PCTerminalKind.TIME_LIMIT),
            "safety": PCTerminalKind(kind) in PC_SAFETY_TERMINAL_KINDS,
            "path_three_d_rmse_m": float(
                np.sqrt(path_sq_sum[index] / max(int(path_steps[index]), 1))
            ),
            "residual_l2_ratio": float(
                residual_l2_sum[index] / max(float(base_l2_sum[index]), 1e-12)
            ),
            "residual_l2_ratio_gate_active": float(
                gate_active_residual_l2_sum[index]
                / max(float(gate_active_base_l2_sum[index]), 1e-12)
            ),
            "residual_abs_max": float(residual_abs_max[index]),
            "action_abs_max": float(action_abs_max[index]),
            "goal_capture_eta": bool(eta_capture_seen[index]),
            "goal_capture_pnav": bool(p_nav_capture_seen[index]),
            "terminal_eta_capture": bool(eta_capture_at_done[index]),
            "terminal_eta_distance_m": float(terminal_eta_distance[index]),
            "terminal_p_nav_distance_m": float(terminal_p_nav_distance[index]),
        }
        row.update(
            score_success_contract(
                p_nav_tight_success=p_nav_success,
                eta_tight_ever=bool(eta_capture_seen[index]),
                eta_tight_at_done=bool(eta_capture_at_done[index]),
                success_frame=success_frame,
                eta_done_distance_m=float(terminal_eta_distance[index]),
                p_nav_done_distance_m=float(terminal_p_nav_distance[index]),
                goal_radius_m=tight_radius,
            )
        )
        rows.append(row)

    summary = {
        "episode_count": len(rows),
        "steps_mean": float(np.mean([row["steps"] for row in rows])),
        "reward_mean": float(np.mean([row["reward_total"] for row in rows])),
        "path_three_d_rmse_m_mean": float(np.mean([row["path_three_d_rmse_m"] for row in rows])),
        "residual_l2_ratio_mean": float(np.mean([row["residual_l2_ratio"] for row in rows])),
        "residual_l2_ratio_gate_active_mean": float(
            np.mean([row["residual_l2_ratio_gate_active"] for row in rows])
        ),
        "residual_abs_max_mean": float(np.mean([row["residual_abs_max"] for row in rows])),
        "action_abs_max_mean": float(np.mean([row["action_abs_max"] for row in rows])),
        "finite": bool(
            all(
                np.isfinite(row[name])
                for row in rows
                for name in (
                    "reward_total",
                    "path_three_d_rmse_m",
                    "residual_l2_ratio",
                    "terminal_eta_distance_m",
                    "terminal_p_nav_distance_m",
                )
            )
        ),
    }
    summary.update(summarize_success_contract(rows, success_frame=success_frame))

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "episodes.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = {
        "schema": "auv_pc_bc_residual_eval.v1",
        "training_config": str(args.config),
        "environment_config": str(environment_path),
        "environment_config_sha256": environment_config.config_sha256,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "algorithm": "BC_residual",
        "action_mode": args.action_mode,
        "residual_scale": float(args.residual_scale),
        "actuator_residual_scale": float(actuator_residual_scale),
        "path_zero": bool(args.path_zero),
        "history_length": history_length,
        "bc_path_dim": int(policy.model.path_dim),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "device": args.device,
        "dtype": args.dtype,
        "success_frame": success_frame,
        "eval_goal_radius_m": tight_radius,
        "termination_position_source": environment_config.termination.position_source,
        "summary": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
