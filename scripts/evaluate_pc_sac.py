#!/usr/bin/env python3
"""Deterministic evaluation for SB3 SAC PC command-residual checkpoints.

The rollout uses the same actor observation, terminal contract and episode
metadata as the Goal58/59 evaluator.  It deliberately keeps only the metrics
needed to audit the SAC adapter and zero-residual parity; full formal analysis
still uses the frozen Goal86 evaluation protocol.
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
from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import (
    CurriculumStageSampler,
    curriculum_episode_metadata,
    curriculum_stages_with_overrides,
)
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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument(
        "--curriculum-stage",
        choices=("C0", "C1", "C2", "C3", "C4", "C5", "C6"),
        default=None,
        help="evaluate a fixed curriculum stage by sampling its episode parameters",
    )
    parser.add_argument(
        "--disable-curriculum-direct-command",
        action="store_true",
        help=(
            "diagnostic-only: keep the curriculum episode parameters but use ALOS "
            "instead of C0/C1 random direct commands"
        ),
    )
    parser.add_argument(
        "--curriculum-current-speed-range-mps",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help=(
            "eval-only override for the sampled curriculum current speed; requires "
            "--curriculum-stage and keeps depth/path/noise/speed unchanged"
        ),
    )
    parser.add_argument(
        "--curriculum-shear-scale",
        type=float,
        default=None,
        help=(
            "eval-only override for the sampled curriculum shear scale; requires "
            "--curriculum-stage and keeps depth/path/noise/speed unchanged"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--action-mode", choices=("policy", "zero-residual"), default="policy")
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help=(
            "Additional multiplier applied to deterministic policy output before "
            "env.step(). The environment already applies config.actuator_residual_scale; "
            "keep this at 1.0 for parity evaluation."
        ),
    )
    parser.add_argument(
        "--learner-terminal-gate-mode",
        choices=("none", "shell"),
        default="none",
        help=(
            "Inference-only learner authority mode. 'shell' keeps the learner active "
            "only in the 3-10 m approach shell and suppresses it inside the capture radius."
        ),
    )
    parser.add_argument(
        "--learner-norm-cap-ratio",
        type=float,
        default=0.0,
        help="Cap learner residual norm at this fraction of the same-step frozen BC norm.",
    )
    parser.add_argument(
        "--learner-channel-cap-ratios",
        type=str,
        default=None,
        help=(
            "Comma-separated per-channel caps for [rudder, stern_plane, propeller_rpm]. "
            "Values <=0 disable that channel."
        ),
    )
    parser.add_argument(
        "--learner-command-channel-mask",
        type=str,
        default=None,
        help=(
            "Comma-separated multiplicative mask for command residual channels "
            "[heading, pitch, speed]. Values must be in [0, 1]; 0 disables a channel."
        ),
    )
    parser.add_argument(
        "--actuator-residual-channel-gains",
        type=str,
        default=None,
        help=(
            "Comma-separated learner-only gains for actuator residuals "
            "[rudder, stern_plane, rpm]. Values must be finite and non-negative."
        ),
    )
    parser.add_argument(
        "--learner-direction-gate-cos-min",
        type=float,
        default=None,
        help="Zero the learner when cos(learner, frozen BC) is below this threshold.",
    )
    parser.add_argument("--geometry-vertical-gain", type=float, default=0.0)
    parser.add_argument("--geometry-rpm-gain", type=float, default=0.0)
    parser.add_argument("--geometry-max-bias", type=float, default=0.15)
    parser.add_argument("--geometry-gate-enter-m", type=float, default=10.0)
    parser.add_argument("--geometry-gate-full-m", type=float, default=2.0)
    parser.add_argument("--path-zero", action="store_true")
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-goal-radius-m", type=float, default=None)
    parser.add_argument("--eval-goal-plane-miss-guard-radius-m", type=float, default=None)
    parser.add_argument("--eval-goal-plane-miss-dwell-steps", type=int, default=0)
    parser.add_argument("--success-frame", choices=SUCCESS_FRAMES, default=DEFAULT_SUCCESS_FRAME)
    parser.add_argument("--trace-output", type=Path, default=None)
    parser.add_argument("--trace-reward-breakdown", action="store_true")
    parser.add_argument("--rich-trace", action="store_true")
    parser.add_argument("--sb3-root", type=Path, default=None)
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


def _discover_sb3_root() -> Path:
    ancestor = REPO_ROOT.parent
    for _ in range(5):
        for candidate in (
            ancestor / "third_party" / "stable-baselines3",
            ancestor / "stable-baselines3",
            ancestor / "src" / "stable-baselines3",
        ):
            if (candidate / "stable_baselines3" / "sac" / "sac.py").is_file():
                return candidate
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    raise FileNotFoundError("could not find stable-baselines3; pass --sb3-root explicitly")


def _validate_eval_seeds(seed_start: int, seed_count: int) -> np.ndarray:
    if seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    seeds = np.arange(seed_start, seed_start + seed_count, dtype=np.int64)
    if ((seeds >= 100) & (seeds <= 149)).any():
        raise ValueError("held-out seeds 100-149 are closed")
    return seeds


def _stack_history(history: np.ndarray) -> np.ndarray:
    return history.reshape(history.shape[0], -1)


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


def _trace_optional(
    trace: dict[str, list[np.ndarray]],
    name: str,
    rows: np.ndarray,
    values: Any,
) -> None:
    if values is not None:
        _trace_put(trace, name, rows, values)


def main() -> int:
    args = _parse_args()
    if args.trace_reward_breakdown and args.trace_output is None:
        raise ValueError("--trace-reward-breakdown requires --trace-output")
    success_frame = validate_success_frame(args.success_frame)
    if not np.isfinite(args.residual_scale) or args.residual_scale <= 0.0:
        raise ValueError("--residual-scale must be finite and positive")
    if (
        not np.isfinite(args.learner_norm_cap_ratio)
        or args.learner_norm_cap_ratio < 0.0
    ):
        raise ValueError("--learner-norm-cap-ratio must be finite and non-negative")
    channel_cap_ratios: tuple[float, float, float] | None = None
    if args.learner_channel_cap_ratios is not None:
        values = [
            float(item.strip())
            for item in str(args.learner_channel_cap_ratios).split(",")
            if item.strip()
        ]
        if len(values) != 3:
            raise ValueError("--learner-channel-cap-ratios requires three values")
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "--learner-channel-cap-ratios must be finite and non-negative"
            )
        channel_cap_ratios = (values[0], values[1], values[2])
    command_channel_mask: tuple[float, float, float] | None = None
    if args.learner_command_channel_mask is not None:
        values = [
            float(item.strip())
            for item in str(args.learner_command_channel_mask).split(",")
            if item.strip()
        ]
        if len(values) != 3:
            raise ValueError("--learner-command-channel-mask requires three values")
        if any(
            not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values
        ):
            raise ValueError(
                "--learner-command-channel-mask values must be finite and in [0, 1]"
            )
        command_channel_mask = (values[0], values[1], values[2])
    actuator_channel_gains: tuple[float, float, float] | None = None
    if args.actuator_residual_channel_gains is not None:
        values = [
            float(item.strip())
            for item in str(args.actuator_residual_channel_gains).split(",")
            if item.strip()
        ]
        if len(values) != 3:
            raise ValueError("--actuator-residual-channel-gains requires three values")
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "--actuator-residual-channel-gains values must be finite and non-negative"
            )
        actuator_channel_gains = (values[0], values[1], values[2])
    if args.learner_direction_gate_cos_min is not None:
        if (
            not np.isfinite(args.learner_direction_gate_cos_min)
            or args.learner_direction_gate_cos_min < -1.0
            or args.learner_direction_gate_cos_min > 1.0
        ):
            raise ValueError("--learner-direction-gate-cos-min must be in [-1, 1]")
    if not np.isfinite(args.geometry_vertical_gain) or not np.isfinite(
        args.geometry_rpm_gain
    ):
        raise ValueError("geometry gains must be finite")
    if not np.isfinite(args.geometry_max_bias) or args.geometry_max_bias <= 0.0:
        raise ValueError("--geometry-max-bias must be finite and positive")
    if (
        not np.isfinite(args.geometry_gate_enter_m)
        or args.geometry_gate_enter_m <= 0.0
    ):
        raise ValueError("--geometry-gate-enter-m must be finite and positive")
    if not np.isfinite(args.geometry_gate_full_m) or args.geometry_gate_full_m <= 0.0:
        raise ValueError("--geometry-gate-full-m must be finite and positive")
    if args.geometry_gate_full_m >= args.geometry_gate_enter_m:
        raise ValueError("--geometry-gate-full-m must be less than enter-m")
    sb3_root = _resolve(args.sb3_root) if args.sb3_root is not None else _discover_sb3_root()
    if str(sb3_root) not in sys.path:
        sys.path.insert(0, str(sb3_root))
    from stable_baselines3 import SAC

    from auv_rl.sac.pc_features import PCStatePathFeatureExtractor

    payload = load_training_payload(args.config)
    if actuator_channel_gains is None and payload.get(
        "actuator_residual_channel_gains"
    ) is not None:
        values = [
            float(item) for item in payload["actuator_residual_channel_gains"]
        ]
        if len(values) != 3:
            raise ValueError("config.actuator_residual_channel_gains requires three values")
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "config.actuator_residual_channel_gains values must be finite and non-negative"
            )
        actuator_channel_gains = (values[0], values[1], values[2])
    observation = dict(payload.get("observation", {}))
    training = dict(payload.get("training", {}))
    action_interface = str(payload.get("action_interface", "command"))
    if action_interface not in {"command", "actuator", "controller_gain"}:
        raise ValueError(
            "config.action_interface must be 'command', 'actuator', or 'controller_gain'"
        )
    actuator_residual_scale = float(payload.get("actuator_residual_scale", 0.25))
    if not np.isfinite(actuator_residual_scale) or actuator_residual_scale <= 0.0:
        raise ValueError("config.actuator_residual_scale must be finite and positive")
    actuator_max_action = float(payload.get("actuator_max_action", 1.0))
    base_residual_value = payload.get("base_residual_checkpoint")
    base_residual_checkpoint = (
        _resolve(Path(base_residual_value))
        if isinstance(base_residual_value, str) and base_residual_value
        else None
    )
    base_residual_action_mask = tuple(
        float(value)
        for value in payload.get("base_residual_action_mask", (1.0, 1.0, 1.0))
    )
    base_residual_bias = tuple(
        float(value) for value in payload.get("base_residual_bias", (0.0, 0.0, 0.0))
    )
    if len(base_residual_bias) != 3 or not all(
        np.isfinite(value) for value in base_residual_bias
    ):
        raise ValueError("base_residual_bias must contain three finite numbers")
    residual_terminal_gate_enabled = bool(
        payload.get("residual_terminal_gate_enabled", False)
    )
    residual_terminal_gate_enter_m = float(
        payload.get("residual_terminal_gate_enter_m", 10.0)
    )
    residual_terminal_gate_full_m = float(
        payload.get("residual_terminal_gate_full_m", 3.0)
    )
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
    curriculum_sampler = None
    current_speed_range_override = (
        environment_config.curriculum.current_speed_range_override_mps
    )
    shear_scale_override = environment_config.curriculum.shear_scale_override
    if args.curriculum_current_speed_range_mps is not None:
        if args.curriculum_stage is None:
            raise ValueError(
                "--curriculum-current-speed-range-mps requires --curriculum-stage"
            )
        speed_min, speed_max = (
            float(value) for value in args.curriculum_current_speed_range_mps
        )
        if not (np.isfinite(speed_min) and np.isfinite(speed_max)):
            raise ValueError("curriculum current speed range must be finite")
        if speed_min < 0.0 or speed_max < speed_min:
            raise ValueError(
                "curriculum current speed range requires 0 <= MIN <= MAX"
            )
        current_speed_range_override = (speed_min, speed_max)
    if args.curriculum_shear_scale is not None:
        if args.curriculum_stage is None:
            raise ValueError("--curriculum-shear-scale requires --curriculum-stage")
        if (
            not np.isfinite(args.curriculum_shear_scale)
            or args.curriculum_shear_scale < 0.0
        ):
            raise ValueError("curriculum shear scale must be finite and non-negative")
        shear_scale_override = float(args.curriculum_shear_scale)
    if args.curriculum_stage is not None:
        stages = curriculum_stages_with_overrides(
            (args.curriculum_stage,),
            current_speed_range_mps=current_speed_range_override,
            shear_scale=shear_scale_override,
        )
        curriculum_sampler = CurriculumStageSampler(
            stages=stages,
            stage_mix=((args.curriculum_stage, 1.0),),
            current_variation_speed_amplitude_mps=(
                0.0
                if environment_config.current.time_variation is None
                else environment_config.current.time_variation.speed_amplitude_mps
            ),
        )
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
        base_residual_checkpoint=base_residual_checkpoint,
        base_residual_action_mask=base_residual_action_mask,
        base_residual_bias=base_residual_bias,
        residual_terminal_gate_enabled=residual_terminal_gate_enabled,
        residual_terminal_gate_enter_m=residual_terminal_gate_enter_m,
        residual_terminal_gate_full_m=residual_terminal_gate_full_m,
        learner_terminal_gate_mode=args.learner_terminal_gate_mode,
        learner_norm_cap_ratio=args.learner_norm_cap_ratio,
        learner_channel_cap_ratios=channel_cap_ratios,
        learner_command_channel_mask=command_channel_mask,
        learner_actuator_channel_gains=actuator_channel_gains,
        learner_direction_gate_cos_min=args.learner_direction_gate_cos_min,
        geometry_vertical_gain=args.geometry_vertical_gain,
        geometry_rpm_gain=args.geometry_rpm_gain,
        geometry_max_bias=args.geometry_max_bias,
        geometry_gate_enter_m=args.geometry_gate_enter_m,
        geometry_gate_full_m=args.geometry_gate_full_m,
        validate_finite=False,
        eval_goal_radius_m=args.eval_goal_radius_m,
        eval_goal_plane_miss_guard_radius_m=args.eval_goal_plane_miss_guard_radius_m,
        eval_goal_plane_miss_dwell_steps=args.eval_goal_plane_miss_dwell_steps,
        curriculum_sampler=curriculum_sampler,
        curriculum_direct_commands_enabled=not args.disable_curriculum_direct_command,
    )
    env.set_action_mode(args.action_mode)
    flat_dim = int(env.observation_dim)

    model = None
    checkpoint_sha256: str | None = None
    if args.action_mode == "policy":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for policy evaluation")
        checkpoint = _resolve(args.checkpoint)
        checkpoint_sha256 = _sha256(checkpoint)
        model = SAC.load(checkpoint, device=args.device)
        if not isinstance(model.actor.features_extractor, PCStatePathFeatureExtractor):
            raise ValueError("checkpoint actor does not use PCStatePathFeatureExtractor")
        expected_shape = (history_length * flat_dim,)
        if tuple(model.observation_space.shape) != expected_shape:
            raise ValueError(
                "checkpoint observation space does not match the configured history: "
                f"{tuple(model.observation_space.shape)} != {expected_shape}"
            )
        model.policy.set_training_mode(False)

    observations, _ = env.reset(seeds.tolist())
    assert env.inner.eta is not None
    initial_heading_rad = env.inner.eta[:, 5].detach().cpu().numpy().copy()
    history = np.repeat(observations[:, None, :], history_length, axis=1)
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
    policy_action_abs_max = np.zeros(env.num_envs, dtype=np.float64)
    residual_abs_max = np.zeros(env.num_envs, dtype=np.float64)
    action_sq_sum = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
    residual_channel_sq_sum = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
    base_channel_sq_sum = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
    bc_residual_action_sq_sum = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
    learned_residual_action_sq_sum = np.zeros(
        (env.num_envs, env.action_dim), dtype=np.float64
    )
    action_saturation_steps = np.zeros(env.num_envs, dtype=np.int64)
    policy_action_saturation_steps = np.zeros(env.num_envs, dtype=np.int64)
    material_saturation_steps = np.zeros(env.num_envs, dtype=np.int64)
    actuator_saturation_steps = np.zeros(env.num_envs, dtype=np.int64)
    scale_clipped_steps = np.zeros(env.num_envs, dtype=np.int64)
    rate_clipped_steps = np.zeros(env.num_envs, dtype=np.int64)
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
    if args.trace_output is not None:
        args.trace_output.mkdir(parents=True, exist_ok=True)
    trace: dict[str, list[np.ndarray]] | None = (
        {} if args.trace_output is not None else None
    )

    for step_index in range(max_steps):
        active_before = active.copy()
        if not bool(active_before.any()):
            break
        actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
        policy_actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
        if model is not None:
            policy_actions[active_before] = model.predict(
                _stack_history(history[active_before]),
                deterministic=True,
            )[0]
            actions[active_before] = policy_actions[active_before] * args.residual_scale
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
        base = (
            env.last_controller_gain_base
            if action_interface == "controller_gain"
            else env.last_base_command
        )
        gate = env.last_residual_gate
        bc_residual_action = env.last_base_residual_action
        learned_residual_action = env.last_learned_residual_action
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
        policy_action_abs_max[active_before] = np.maximum(
            policy_action_abs_max[active_before],
            np.max(np.abs(policy_actions[active_before]), axis=1),
        )
        active_actions = actions[active_before]
        active_applied = applied[active_before]
        active_base = base[active_before]
        action_sq_sum[active_before] += active_actions**2
        residual_channel_sq_sum[active_before] += active_applied**2
        base_channel_sq_sum[active_before] += active_base**2
        bc_residual_action_sq_sum[active_before] += bc_residual_action[active_before] ** 2
        learned_residual_action_sq_sum[active_before] += (
            learned_residual_action[active_before] ** 2
        )
        action_saturation_steps[active_before] += np.any(
            np.abs(active_actions) > 0.95, axis=1
        ).astype(np.int64)
        policy_action_saturation_steps[active_before] += np.any(
            np.abs(policy_actions[active_before]) > 0.95, axis=1
        ).astype(np.int64)
        material_saturated = info.get("material_saturated")
        actuator_saturated = info.get("saturated")
        if torch.is_tensor(material_saturated):
            material_saturated = material_saturated.detach().cpu().numpy()
        if torch.is_tensor(actuator_saturated):
            actuator_saturated = actuator_saturated.detach().cpu().numpy()
        if material_saturated is not None:
            material_saturation_steps[active_before] += np.asarray(
                material_saturated, dtype=bool
            )[active_before].astype(np.int64)
        if actuator_saturated is not None:
            actuator_saturation_steps[active_before] += np.asarray(
                actuator_saturated, dtype=bool
            )[active_before].astype(np.int64)
        scale_clipped = env.last_scale_clipped
        rate_clipped = env.last_rate_clipped
        scale_clipped_steps[active_before] += np.any(
            np.asarray(scale_clipped, dtype=bool)[active_before], axis=1
        ).astype(np.int64)
        rate_clipped_steps[active_before] += np.any(
            np.asarray(rate_clipped, dtype=bool)[active_before], axis=1
        ).astype(np.int64)
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

        if trace is not None:
            rows = np.flatnonzero(active_before)
            if rows.size:
                _trace_put(trace, "step", rows, int(step_index))
                _trace_put(trace, "env_index", rows, np.arange(env.num_envs, dtype=np.int64))
                _trace_put(trace, "episode_id", rows, info["episode_ids"])
                _trace_put(trace, "seed", rows, seeds)
                _trace_put(trace, "active", rows, active_before)
                _trace_put(trace, "action", rows, actions)
                _trace_put(trace, "policy_action_raw", rows, policy_actions)
                _trace_put(trace, "policy_action_effective", rows, actions)
                _trace_put(trace, "residual", rows, applied)
                _trace_put(trace, "base_command", rows, base)
                _trace_put(trace, "gate", rows, gate)
                _trace_put(trace, "bc_residual_action", rows, bc_residual_action)
                _trace_put(trace, "learned_residual_action", rows, learned_residual_action)
                _trace_optional(
                    trace,
                    "learner_gated_action",
                    rows,
                    getattr(env, "last_learner_gated_action", None),
                )
                _trace_optional(
                    trace,
                    "learner_direction_cos",
                    rows,
                    getattr(env, "last_learner_direction_cos", None),
                )
                _trace_optional(
                    trace,
                    "geometry_bias",
                    rows,
                    getattr(env, "last_geometry_bias", None),
                )
                _trace_optional(
                    trace,
                    "geometry_gate",
                    rows,
                    getattr(env, "last_geometry_gate", None),
                )
                _trace_put(trace, "scale_clipped", rows, env.last_scale_clipped)
                _trace_put(trace, "rate_clipped", rows, env.last_rate_clipped)
                _trace_optional(trace, "actuator_saturated", rows, actuator_saturated)
                _trace_optional(trace, "material_saturated", rows, material_saturated)
                _trace_put(trace, "reward", rows, rewards)
                _trace_put(trace, "goal_distance_eta", rows, eta_distance)
                _trace_put(trace, "goal_distance_pnav", rows, p_nav_distance)
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
                _trace_put(trace, "goal_ned", rows, goal_ned.detach().cpu().numpy())
                _trace_put(
                    trace,
                    "terminal_kind",
                    rows,
                    info["terminal_kind"].detach().cpu().numpy(),
                )
                _trace_put(trace, "done", rows, done_now)
                if args.rich_trace:
                    assert inner.t_s is not None and inner.nu is not None
                    _trace_put(trace, "time_s", rows, inner.t_s.detach().cpu().numpy())
                    _trace_put(trace, "eta_rpy", rows, inner.eta[:, 3:6].detach().cpu().numpy())
                    _trace_put(trace, "nu", rows, inner.nu.detach().cpu().numpy())
                    for name in (
                        "ground_velocity_body_mps",
                        "water_velocity_body_mps",
                        "velocity_contract_code",
                        "actual_actuator",
                        "requested_action",
                        "limited_action",
                        "plant_scale_factors",
                        "command_physical",
                        "command_water_relative_speed_mps",
                        "command_reachability_feasible",
                        "guidance_terminal_phase",
                        "guidance_crab_angle_rad",
                        "guidance_speed_scale",
                    ):
                        value = info.get(name)
                        if torch.is_tensor(value):
                            _trace_put(trace, name, rows, value.detach().cpu().numpy())
                if args.trace_reward_breakdown:
                    for component, component_values in env.last_reward_breakdown.items():
                        _trace_optional(
                            trace,
                            f"reward_{component}",
                            rows,
                            component_values,
                        )

        if history_length > 1:
            history[active_before, :-1] = history[active_before, 1:]
        history[active_before, -1] = next_observations[active_before]
        active[done_now] = False

    if bool(active.any()):
        raise RuntimeError("evaluation ended before every environment terminated")

    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds.tolist()):
        kind = int(terminal_kind[index])
        p_nav_success = kind == int(PCTerminalKind.GOAL)
        row_steps = max(int(steps[index]), 1)
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
            "policy_action_abs_max": float(policy_action_abs_max[index]),
            "action_saturation_fraction": float(
                action_saturation_steps[index] / row_steps
            ),
            "policy_action_saturation_fraction": float(
                policy_action_saturation_steps[index] / row_steps
            ),
            "material_saturation_fraction": float(
                material_saturation_steps[index] / row_steps
            ),
            "actuator_saturation_fraction": float(
                actuator_saturation_steps[index] / row_steps
            ),
            "scale_clipped_fraction": float(scale_clipped_steps[index] / row_steps),
            "rate_clipped_fraction": float(rate_clipped_steps[index] / row_steps),
            "action_rms": np.sqrt(action_sq_sum[index] / row_steps).tolist(),
            "residual_channel_rms": np.sqrt(
                residual_channel_sq_sum[index] / row_steps
            ).tolist(),
            "base_channel_rms": np.sqrt(base_channel_sq_sum[index] / row_steps).tolist(),
            "residual_channel_l2_ratio": np.sqrt(
                residual_channel_sq_sum[index]
                / np.maximum(base_channel_sq_sum[index], 1e-24)
            ).tolist(),
            "bc_residual_action_rms": np.sqrt(
                bc_residual_action_sq_sum[index] / row_steps
            ).tolist(),
            "learned_residual_action_rms": np.sqrt(
                learned_residual_action_sq_sum[index] / row_steps
            ).tolist(),
            "learned_to_bc_residual_action_rms_ratio": np.sqrt(
                learned_residual_action_sq_sum[index]
                / np.maximum(bc_residual_action_sq_sum[index], 1e-24)
            ).tolist(),
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

    if env.inner.curriculum_specs is not None:
        if len(env.inner.curriculum_specs) != len(rows):
            raise ValueError("curriculum specs do not align with evaluation episodes")
        for index, (row, spec) in enumerate(
            zip(rows, env.inner.curriculum_specs, strict=True)
        ):
            row["curriculum"] = curriculum_episode_metadata(
                spec,
                initial_heading_rad=float(initial_heading_rad[index]),
            )

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
        "policy_action_abs_max_mean": float(
            np.mean([row["policy_action_abs_max"] for row in rows])
        ),
        "action_saturation_fraction_mean": float(
            np.mean([row["action_saturation_fraction"] for row in rows])
        ),
        "policy_action_saturation_fraction_mean": float(
            np.mean([row["policy_action_saturation_fraction"] for row in rows])
        ),
        "material_saturation_fraction_mean": float(
            np.mean([row["material_saturation_fraction"] for row in rows])
        ),
        "actuator_saturation_fraction_mean": float(
            np.mean([row["actuator_saturation_fraction"] for row in rows])
        ),
        "scale_clipped_fraction_mean": float(
            np.mean([row["scale_clipped_fraction"] for row in rows])
        ),
        "rate_clipped_fraction_mean": float(
            np.mean([row["rate_clipped_fraction"] for row in rows])
        ),
        "action_rms_mean": np.mean(
            np.asarray([row["action_rms"] for row in rows], dtype=np.float64), axis=0
        ).tolist(),
        "residual_channel_rms_mean": np.mean(
            np.asarray([row["residual_channel_rms"] for row in rows], dtype=np.float64),
            axis=0,
        ).tolist(),
        "base_channel_rms_mean": np.mean(
            np.asarray([row["base_channel_rms"] for row in rows], dtype=np.float64),
            axis=0,
        ).tolist(),
        "residual_channel_l2_ratio_mean": np.mean(
            np.asarray(
                [row["residual_channel_l2_ratio"] for row in rows], dtype=np.float64
            ),
            axis=0,
        ).tolist(),
        "bc_residual_action_rms_mean": np.mean(
            np.asarray([row["bc_residual_action_rms"] for row in rows], dtype=np.float64),
            axis=0,
        ).tolist(),
        "learned_residual_action_rms_mean": np.mean(
            np.asarray(
                [row["learned_residual_action_rms"] for row in rows], dtype=np.float64
            ),
            axis=0,
        ).tolist(),
        "learned_to_bc_residual_action_rms_ratio_mean": np.mean(
            np.asarray(
                [row["learned_to_bc_residual_action_rms_ratio"] for row in rows],
                dtype=np.float64,
            ),
            axis=0,
        ).tolist(),
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
    terminal_kind_counts: dict[str, int] = {}
    for row in rows:
        name = PCTerminalKind(int(row["terminal_kind"])).name
        terminal_kind_counts[name] = terminal_kind_counts.get(name, 0) + 1
    summary["terminal_kind_counts"] = terminal_kind_counts

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "episodes.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = {
        "schema": "auv_pc_sac_eval.v1",
        "training_config": str(args.config),
        "environment_config": str(environment_path),
        "curriculum_stage": args.curriculum_stage,
        "curriculum_direct_commands_enabled": (
            not args.disable_curriculum_direct_command
        ),
        "curriculum_current_speed_range_override_mps": (
            None
            if current_speed_range_override is None
            else list(current_speed_range_override)
        ),
        "curriculum_shear_scale_override": shear_scale_override,
        "environment_config_sha256": environment_config.config_sha256,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "algorithm": "SAC",
        "action_interface": action_interface,
        "controller_gain_residual": (
            {
                "heading_gain_delta": environment_config.controller_gain_residual.heading_gain_delta,
                "pitch_gain_delta": environment_config.controller_gain_residual.pitch_gain_delta,
                "terminal_gate_enabled": environment_config.controller_gain_residual.terminal_gate_enabled,
                "terminal_gate_enter_m": environment_config.controller_gain_residual.terminal_gate_enter_m,
                "terminal_gate_full_m": environment_config.controller_gain_residual.terminal_gate_full_m,
            }
            if action_interface == "controller_gain"
            else None
        ),
        "action_mode": args.action_mode,
        "residual_scale": float(args.residual_scale),
        "configured_actuator_residual_scale": actuator_residual_scale,
        "eval_residual_multiplier": float(args.residual_scale),
        "nominal_effective_actuator_residual_scale": float(
            actuator_residual_scale * args.residual_scale
        ),
        "base_residual_checkpoint": (
            None if base_residual_checkpoint is None else str(base_residual_checkpoint)
        ),
        "base_residual_action_mask": list(base_residual_action_mask),
        "base_residual_bias": list(base_residual_bias),
        "residual_terminal_gate_enabled": residual_terminal_gate_enabled,
        "residual_terminal_gate_enter_m": residual_terminal_gate_enter_m,
        "residual_terminal_gate_full_m": residual_terminal_gate_full_m,
        "learner_terminal_gate_mode": args.learner_terminal_gate_mode,
        "learner_norm_cap_ratio": float(args.learner_norm_cap_ratio),
        "learner_channel_cap_ratios": (
            list(channel_cap_ratios) if channel_cap_ratios is not None else None
        ),
        "learner_command_channel_mask": (
            list(command_channel_mask) if command_channel_mask is not None else None
        ),
        "learner_actuator_channel_gains": (
            list(actuator_channel_gains) if actuator_channel_gains is not None else None
        ),
        "learner_direction_gate_cos_min": args.learner_direction_gate_cos_min,
        "geometry_vertical_gain": float(args.geometry_vertical_gain),
        "geometry_rpm_gain": float(args.geometry_rpm_gain),
        "geometry_max_bias": float(args.geometry_max_bias),
        "geometry_gate_enter_m": float(args.geometry_gate_enter_m),
        "geometry_gate_full_m": float(args.geometry_gate_full_m),
        "path_zero": bool(args.path_zero),
        "history_length": history_length,
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
    if args.trace_output is not None:
        assert trace is not None
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
            "schema": "auv_pc_sac_trace.v1",
            "training_config": str(args.config),
            "environment_config": str(environment_path),
            "environment_config_sha256": environment_config.config_sha256,
            "curriculum_stage": args.curriculum_stage,
            "curriculum_direct_commands_enabled": (
                not args.disable_curriculum_direct_command
            ),
            "curriculum_current_speed_range_override_mps": (
                None
                if current_speed_range_override is None
                else list(current_speed_range_override)
            ),
            "curriculum_shear_scale_override": shear_scale_override,
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "algorithm": "SAC",
            "action_interface": action_interface,
            "action_mode": args.action_mode,
            "residual_scale": float(args.residual_scale),
            "configured_actuator_residual_scale": actuator_residual_scale,
            "eval_residual_multiplier": float(args.residual_scale),
            "nominal_effective_actuator_residual_scale": float(
                actuator_residual_scale * args.residual_scale
            ),
            "device": args.device,
            "dtype": args.dtype,
            "success_frame": success_frame,
            "eval_goal_radius_m": tight_radius,
            "control_dt_s": float(environment_config.plant.dt_s),
            "residual_terminal_gate_enabled": residual_terminal_gate_enabled,
            "residual_terminal_gate_enter_m": residual_terminal_gate_enter_m,
            "residual_terminal_gate_full_m": residual_terminal_gate_full_m,
            "learner_terminal_gate_mode": args.learner_terminal_gate_mode,
            "learner_norm_cap_ratio": float(args.learner_norm_cap_ratio),
            "learner_channel_cap_ratios": (
                list(channel_cap_ratios) if channel_cap_ratios is not None else None
            ),
            "learner_command_channel_mask": (
                list(command_channel_mask) if command_channel_mask is not None else None
            ),
            "learner_actuator_channel_gains": (
                list(actuator_channel_gains)
                if actuator_channel_gains is not None
                else None
            ),
            "learner_direction_gate_cos_min": args.learner_direction_gate_cos_min,
            "geometry_vertical_gain": float(args.geometry_vertical_gain),
            "geometry_rpm_gain": float(args.geometry_rpm_gain),
            "geometry_max_bias": float(args.geometry_max_bias),
            "geometry_gate_enter_m": float(args.geometry_gate_enter_m),
            "geometry_gate_full_m": float(args.geometry_gate_full_m),
            "rich_trace": bool(args.rich_trace),
            "trace_reward_breakdown": bool(args.trace_reward_breakdown),
            "seed_start": int(args.seed_start),
            "seed_count": int(args.seed_count),
            "episodes": int(len(seeds)),
            "trace_fields": sorted(arrays),
            "finite": finite,
        }
        (args.trace_output / "trace_meta.json").write_text(
            json.dumps(trace_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
