#!/usr/bin/env python3
"""Train a PC command-residual SAC policy through Stable-Baselines3.

The script is deliberately separate from the historical DDPG Goal58/59
trainer.  It reuses the same batched PC environment but exposes it as a
Stable-Baselines3 ``VecEnv`` with a fixed observation-history stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.config import REMUS100_MAX_SPEED_MPS, load_config
from auv_lowlevel.curriculum import (
    CurriculumPromotionController,
    CurriculumStageSampler,
    LinearAlphaSchedule,
    curriculum_stages_with_overrides,
)
from auv_lowlevel.curriculum_trainer import (
    CurriculumTrainerHook,
    combine_eval_summaries,
    run_eval_command,
)
from auv_lowlevel.experiment_contract import require_training_curriculum
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv
from auv_rl.training import load_training_payload

REPO_ROOT = Path(__file__).resolve().parents[1]


def _speed_contract(
    environment_config: Any,
    curriculum_sampler: CurriculumStageSampler | None,
) -> dict[str, Any]:
    """Describe and validate the REMUS100 ground-speed envelope.

    Curriculum episodes shrink the sampled current so that
    ``speed + current + variation`` never exceeds the platform cap.  For a
    fixed-speed run the same inequality must hold for the configured ranges.
    """

    variation = (
        0.0
        if environment_config.current.time_variation is None
        else float(environment_config.current.time_variation.speed_amplitude_mps)
    )
    current_upper = (
        float(environment_config.current.max_speed_mps)
        if environment_config.current.enabled
        else 0.0
    )
    reference_speed = float(environment_config.reference.speed_mps)
    stage_ranges: dict[str, list[float]] = {}
    worst_case = reference_speed + current_upper + variation
    if curriculum_sampler is not None:
        worst_case = 0.0
        for stage, spec in curriculum_sampler.stages.items():
            stage_ranges[stage] = [float(value) for value in spec.speed_range_mps]
            stage_speed = min(
                float(spec.speed_range_mps[1]),
                REMUS100_MAX_SPEED_MPS - variation,
            )
            stage_current = min(
                float(spec.current_speed_range_mps[1]),
                max(0.0, REMUS100_MAX_SPEED_MPS - stage_speed - variation),
            )
            worst_case = max(worst_case, stage_speed + stage_current + variation)
    ok = (
        reference_speed <= REMUS100_MAX_SPEED_MPS + 1e-12
        and worst_case <= REMUS100_MAX_SPEED_MPS + 1e-12
        and all(
            upper <= REMUS100_MAX_SPEED_MPS + 1e-12
            for upper in (values[1] for values in stage_ranges.values())
        )
    )
    return {
        "max_speed_mps": REMUS100_MAX_SPEED_MPS,
        "reference_speed_mps": reference_speed,
        "current_speed_range_mps": [
            float(environment_config.current.min_speed_mps),
            current_upper,
        ],
        "current_variation_amplitude_mps": variation,
        "curriculum_enabled": bool(environment_config.curriculum.enabled),
        "curriculum_speed_ranges_mps": stage_ranges,
        "worst_case_ground_speed_mps": float(worst_case),
        "ok": bool(ok),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-env-transitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--update-to-data-ratio", type=float, default=None)
    parser.add_argument(
        "--sb3-root",
        type=Path,
        default=None,
        help="stable-baselines3 checkout; auto-discovered when omitted",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Save the initialized actor as initial.zip without training.",
    )
    parser.add_argument(
        "--engineering-smoke",
        action="store_true",
        help="Mark a throughput/integration run without applying the tiny-smoke caps",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--curriculum-eval-default-config", type=Path, default=None)
    parser.add_argument("--curriculum-eval-strict-config", type=Path, default=None)
    parser.add_argument("--curriculum-eval-interval", type=int, default=100_000)
    parser.add_argument("--curriculum-eval-seed-start", type=int, default=20)
    parser.add_argument("--curriculum-eval-seed-count", type=int, default=40)
    parser.add_argument("--curriculum-eval-timeout-s", type=float, default=1800.0)
    parser.add_argument("--curriculum-mix-ramp-transitions", type=int, default=0)
    parser.add_argument("--curriculum-eval-device", default="cpu")
    parser.add_argument(
        "--actuator-residual-channel-gains",
        default=None,
        help=(
            "Comma-separated learner-only gains for [rudder, stern_plane, rpm]. "
            "The frozen base residual is not masked."
        ),
    )
    parser.add_argument(
        "--curriculum-force-current-alpha-zero",
        action="store_true",
        help=(
            "keep the frozen-BC current encoder enabled but force its actor alpha to "
            "zero for the whole run (C-no-current ablation)"
        ),
    )
    parser.add_argument(
        "--curriculum-disable-direct-command",
        action="store_true",
        help=(
            "diagnostic/repair arm: keep C0/C1 curriculum episodes but use ALOS "
            "instead of random direct commands"
        ),
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_actuator_channel_gains(
    value: str | None,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--actuator-residual-channel-gains requires three values")
    if any(not np.isfinite(item) or item < 0.0 for item in values):
        raise ValueError(
            "--actuator-residual-channel-gains values must be finite and non-negative"
        )
    return (values[0], values[1], values[2])


def _discover_sb3_root() -> Path:
    candidates: list[Path] = []
    ancestor = REPO_ROOT.parent
    for _ in range(5):
        candidates.extend(
            (
                ancestor / "third_party" / "stable-baselines3",
                ancestor / "stable-baselines3",
                ancestor / "src" / "stable-baselines3",
            )
        )
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    for candidate in candidates:
        if (candidate / "stable_baselines3" / "sac" / "sac.py").is_file():
            return candidate
    raise FileNotFoundError("could not find stable-baselines3; pass --sb3-root explicitly")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _gradient_steps_per_vector_step(num_envs: int, utd: float) -> int:
    if not math.isfinite(utd) or utd <= 0.0:
        raise ValueError("update_to_data_ratio must be finite and positive")
    return max(1, int(round(num_envs * utd)))


def _current_encoder_alpha_schedule(
    policy_config: dict[str, Any],
    *,
    current_dim: int,
) -> tuple[float, LinearAlphaSchedule | None]:
    """Resolve the SAC actor current-encoder alpha and optional linear ramp."""

    base_alpha = float(policy_config.get("current_encoder_alpha", 1.0))
    if not math.isfinite(base_alpha) or base_alpha < 0.0:
        raise ValueError("policy.current_encoder_alpha must be finite and non-negative")
    start_value = policy_config.get("current_encoder_alpha_start")
    end_value = policy_config.get("current_encoder_alpha_end")
    if (start_value is None) != (end_value is None):
        raise ValueError(
            "policy.current_encoder_alpha_start and "
            "policy.current_encoder_alpha_end must be set together"
        )
    transitions_value = policy_config.get(
        "current_encoder_alpha_ramp_transitions", 0
    )
    if isinstance(transitions_value, bool):
        raise ValueError(
            "policy.current_encoder_alpha_ramp_transitions must be a non-negative integer"
        )
    transitions = int(transitions_value)
    if transitions < 0 or transitions != transitions_value:
        raise ValueError(
            "policy.current_encoder_alpha_ramp_transitions must be a non-negative integer"
        )
    if start_value is None:
        if transitions != 0:
            raise ValueError(
                "policy.current_encoder_alpha_ramp_transitions requires start and end"
            )
        return base_alpha, None
    if current_dim == 0:
        raise ValueError(
            "actor current-encoder alpha ramp requires observation.current_dim=3"
        )
    schedule = LinearAlphaSchedule(
        start=float(start_value),
        end=float(end_value),
        transitions=transitions,
        name="policy.current_encoder_alpha",
    )
    return schedule.value_at(0), schedule


def main() -> int:
    args = _parse_args()
    sb3_root = _resolve(args.sb3_root) if args.sb3_root is not None else _discover_sb3_root()
    if str(sb3_root) not in sys.path:
        sys.path.insert(0, str(sb3_root))
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    from auv_rl.sac.pc_features import (
        PC_DADE_FEATURE_DIM,
        PCStatePathFeatureExtractor,
        initialize_sac_head_from_bc,
        zero_initialize_sac_residual,
    )
    from auv_rl.sac.authority_sac import (
        AuthorityRegularizedSAC,
        parse_authority_regularization,
    )
    from auv_rl.sac.pc_vec_env import PCHistoryVecEnv

    payload = load_training_payload(args.config)
    training = dict(payload.get("training", {}))
    observation = dict(payload.get("observation", {}))
    policy_config = dict(payload.get("policy", {}))
    authority_regularization = parse_authority_regularization(training)
    action_interface = str(payload.get("action_interface", "command"))
    if action_interface not in {"command", "actuator", "controller_gain"}:
        raise ValueError(
            "config.action_interface must be 'command', 'actuator', or 'controller_gain'"
        )
    actuator_residual_scale = float(payload.get("actuator_residual_scale", 0.25))
    actuator_max_action = float(payload.get("actuator_max_action", 1.0))
    actuator_channel_gains = _parse_actuator_channel_gains(
        args.actuator_residual_channel_gains
    )
    if actuator_channel_gains is None and payload.get(
        "actuator_residual_channel_gains"
    ) is not None:
        payload_gains = payload["actuator_residual_channel_gains"]
        actuator_channel_gains = _parse_actuator_channel_gains(
            ",".join(str(item) for item in payload_gains)
        )
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
    learner_terminal_gate_mode = str(
        payload.get("learner_terminal_gate_mode", "none")
    )
    if learner_terminal_gate_mode not in {"none", "shell"}:
        raise ValueError("learner_terminal_gate_mode must be 'none' or 'shell'")
    if learner_terminal_gate_mode == "shell" and not residual_terminal_gate_enabled:
        raise ValueError(
            "learner_terminal_gate_mode='shell' requires residual_terminal_gate_enabled=true"
        )
    environment_value = payload.get("environment_config")
    if not isinstance(environment_value, str) or not environment_value:
        raise ValueError("config.environment_config must be a non-empty string")
    environment_path = _resolve(Path(environment_value))
    environment_config = load_config(environment_path)
    require_training_curriculum(environment_config, source=environment_path)
    if args.curriculum_force_current_alpha_zero and not environment_config.curriculum.enabled:
        raise ValueError(
            "--curriculum-force-current-alpha-zero requires environment.curriculum.enabled=true"
        )
    if args.curriculum_disable_direct_command and not environment_config.curriculum.enabled:
        raise ValueError(
            "--curriculum-disable-direct-command requires environment.curriculum.enabled=true"
        )

    num_envs = int(args.num_envs if args.num_envs is not None else training.get("num_envs", 40))
    total_env_transitions = int(
        args.total_env_transitions
        if args.total_env_transitions is not None
        else training.get("total_env_transitions", 5_000_000)
    )
    history_length = int(
        args.history_length
        if args.history_length is not None
        else training.get("history_length", 10)
    )
    learning_starts = int(
        args.learning_starts
        if args.learning_starts is not None
        else training.get("learning_starts", 20_000)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else training.get("batch_size", 256)
    )
    buffer_size = int(
        args.buffer_size if args.buffer_size is not None else training.get("buffer_size", 500_000)
    )
    utd = float(
        args.update_to_data_ratio
        if args.update_to_data_ratio is not None
        else training.get("update_to_data_ratio", 0.125)
    )
    if args.smoke:
        num_envs = min(num_envs, 4)
        total_env_transitions = min(total_env_transitions, 2048)
        learning_starts = min(learning_starts, 64)
        batch_size = min(batch_size, 64)
        buffer_size = min(buffer_size, 10_000)
    if num_envs <= 0 or total_env_transitions <= 0 or history_length <= 0:
        raise ValueError("num_envs, total_env_transitions and history_length must be positive")
    seed = int(args.seed if args.seed is not None else training.get("seed", 0))
    device = _resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    gradient_steps = _gradient_steps_per_vector_step(num_envs, utd)

    output = _resolve(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    curriculum_sampler = None
    if environment_config.curriculum.enabled:
        stages = curriculum_stages_with_overrides(
            [stage for stage, _weight in environment_config.curriculum.stage_mix],
            current_speed_range_mps=(
                environment_config.curriculum.current_speed_range_override_mps
            ),
            shear_scale=environment_config.curriculum.shear_scale_override,
        )
        curriculum_sampler = CurriculumStageSampler(
            stages=stages,
            stage_mix=environment_config.curriculum.stage_mix,
            current_variation_speed_amplitude_mps=(
                0.0
                if environment_config.current.time_variation is None
                else environment_config.current.time_variation.speed_amplitude_mps
            ),
        )
    speed_contract = _speed_contract(environment_config, curriculum_sampler)
    if not speed_contract["ok"]:
        raise ValueError(
            "training speed contract violates the REMUS100 envelope: "
            + json.dumps(speed_contract, sort_keys=True)
        )
    env = PCCommandResidualOffPolicyEnv(
        environment_config,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        path_dim=int(observation.get("path_dim", 8)),
        actor_current_encoder_enabled=int(observation.get("current_dim", 3)) == 3,
        action_interface=action_interface,
        actuator_residual_scale=actuator_residual_scale,
        actuator_max_action=actuator_max_action,
        base_residual_checkpoint=base_residual_checkpoint,
        base_residual_action_mask=base_residual_action_mask,
        base_residual_bias=base_residual_bias,
        residual_terminal_gate_enabled=residual_terminal_gate_enabled,
        residual_terminal_gate_enter_m=residual_terminal_gate_enter_m,
        residual_terminal_gate_full_m=residual_terminal_gate_full_m,
        learner_terminal_gate_mode=learner_terminal_gate_mode,
        learner_actuator_channel_gains=actuator_channel_gains,
        validate_finite=False,
        curriculum_sampler=curriculum_sampler,
        curriculum_manifest_path=(
            output / environment_config.curriculum.manifest_filename
            if curriculum_sampler is not None
            else None
        ),
        curriculum_direct_commands_enabled=not args.curriculum_disable_direct_command,
    )
    vec_env = PCHistoryVecEnv(env, history_length=history_length)

    architecture = str(policy_config.get("architecture", "pc_z_t"))
    if architecture != "pc_z_t":
        raise ValueError("SAC trainer currently requires policy.architecture='pc_z_t'")
    state_dim = int(observation.get("state_dim", env.inner.actor_state_dim))
    path_dim = int(observation.get("path_dim", env.path_dim))
    command_dim = int(observation.get("command_dim", 3))
    current_dim = int(observation.get("current_dim", 3))
    if state_dim != int(env.inner.actor_state_dim):
        raise ValueError("observation.state_dim does not match the PC environment")
    if command_dim != 3:
        raise ValueError("SAC PC policy requires observation.command_dim=3")
    if current_dim not in (0, 3):
        raise ValueError("SAC PC policy requires observation.current_dim in (0, 3)")
    expected_flat = state_dim + path_dim + command_dim + current_dim
    if expected_flat != int(env.observation_dim):
        raise ValueError(
            "state/path/command/current dimensions do not match the PC observation: "
            f"{expected_flat} != {env.observation_dim}"
        )
    bc_path_dim = int(policy_config.get("bc_path_dim", path_dim))
    d_z_total = int(policy_config.get("d_z_total", 64))
    n_in_reserved = int(policy_config.get("n_in_reserved", 8))
    n_z_reserved = int(policy_config.get("n_z_reserved", 4))
    activation_name = str(policy_config.get("activation", "elu")).lower()
    activation_fns = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
    }
    if activation_name not in activation_fns:
        raise ValueError(f"unsupported policy activation: {activation_name}")

    current_encoder_alpha_initial, current_encoder_alpha_schedule = (
        _current_encoder_alpha_schedule(policy_config, current_dim=current_dim)
    )
    net_arch = [int(value) for value in training.get("net_arch", (256, 128))]
    features_extractor_kwargs = {
        "flat_observation_dim": int(env.observation_dim),
        "state_dim": state_dim,
        "path_dim": path_dim,
        "bc_path_dim": bc_path_dim,
        "command_dim": command_dim,
        "current_dim": current_dim,
        "history_length": history_length,
        "d_z_total": d_z_total,
        "n_in_reserved": n_in_reserved,
        "n_z_reserved": n_z_reserved,
        "state_encoder_hidden_dims": tuple(
            int(value) for value in policy_config.get("state_encoder_hidden_dims", (128, 128))
        ),
        "path_encoder_hidden_dims": tuple(
            int(value) for value in policy_config.get("path_encoder_hidden_dims", (64,))
        ),
        "path_encoder_alpha": float(policy_config.get("path_encoder_alpha", 1.0)),
        "endpoint_encoder_hidden_dims": tuple(
            int(value) for value in policy_config.get("endpoint_encoder_hidden_dims", (32,))
        ),
        "endpoint_encoder_alpha": float(policy_config.get("endpoint_encoder_alpha", 1.0)),
        "current_encoder_hidden_dims": tuple(
            int(value) for value in policy_config.get("current_encoder_hidden_dims", (64,))
        ),
        "current_encoder_alpha": current_encoder_alpha_initial,
        "activation": activation_name,
    }
    policy_kwargs = {
        "net_arch": {"pi": net_arch, "qf": net_arch},
        "activation_fn": activation_fns[activation_name],
        "features_extractor_class": PCStatePathFeatureExtractor,
        "features_extractor_kwargs": features_extractor_kwargs,
        "share_features_extractor": False,
        "log_std_init": float(policy_config.get("log_std_init", -3.0)),
    }
    actor_lr = float(training.get("actor_lr", 3e-4))
    critic_lr = float(training.get("critic_lr", actor_lr))
    model = AuthorityRegularizedSAC(
        "MlpPolicy",
        vec_env,
        learning_rate=actor_lr,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=float(training.get("tau", 0.005)),
        gamma=float(training.get("gamma", 0.99)),
        train_freq=(1, "step"),
        gradient_steps=gradient_steps,
        ent_coef="auto",
        target_entropy=-float(env.action_dim),
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        verbose=int(training.get("verbose", 0)),
        seed=seed,
        device=device,
        authority_rho=float(authority_regularization["rho"]),
        authority_beta=float(authority_regularization["beta"]),
        authority_form=str(authority_regularization["form"]),
    )
    for group in model.critic.optimizer.param_groups:
        group["lr"] = critic_lr

    bc_checkpoint_value = policy_config.get("bc_checkpoint")
    bc_checkpoint = (
        _resolve(Path(bc_checkpoint_value)) if bc_checkpoint_value not in (None, "") else None
    )
    initialization: dict[str, Any] = {
        "architecture": architecture,
        "features_dim": int(model.actor.features_dim),
        "net_arch": net_arch,
        "activation": activation_name,
        "zero_dade_slot_dim": PC_DADE_FEATURE_DIM,
        "feature_order": ["z_t", "zero_dade", "command"],
        "bc_checkpoint": None if bc_checkpoint is None else str(bc_checkpoint),
        "bc_path_dim": bc_path_dim,
        "endpoint_dim": path_dim - bc_path_dim,
        "base_residual_checkpoint": (
            None if base_residual_checkpoint is None else str(base_residual_checkpoint)
        ),
                "base_residual_checkpoint_sha256": (
                    None if base_residual_checkpoint is None else _sha256(base_residual_checkpoint)
                ),
        "base_residual_action_mask": list(base_residual_action_mask),
        "base_residual_bias": list(base_residual_bias),
        "residual_terminal_gate_enabled": residual_terminal_gate_enabled,
        "residual_terminal_gate_enter_m": residual_terminal_gate_enter_m,
        "residual_terminal_gate_full_m": residual_terminal_gate_full_m,
        "learner_terminal_gate_mode": learner_terminal_gate_mode,
        "authority_regularization": authority_regularization,
        "current_encoder_alpha_initial": current_encoder_alpha_initial,
        "current_encoder_alpha_schedule": (
            None
            if current_encoder_alpha_schedule is None
            else current_encoder_alpha_schedule.as_dict()
        ),
    }
    if bc_checkpoint is not None:
        if not bc_checkpoint.is_file():
            raise FileNotFoundError(f"BC checkpoint does not exist: {bc_checkpoint}")
        initialization["encoder"] = model.actor.features_extractor.load_bc_encoder(bc_checkpoint)
        initialization["head"] = initialize_sac_head_from_bc(
            model,
            bc_checkpoint,
            zero_residual=True,
        )
    elif bool(policy_config.get("zero_residual_init", True)):
        initialization["head"] = zero_initialize_sac_residual(
            model,
            log_std_init=float(policy_config.get("log_std_init", -3.0)),
        )
    _write_json(output / "initialization.json", initialization)

    if args.init_only:
        initial_checkpoint = output / "initial.zip"
        model.save(output / "initial")
        _write_json(
            output / "initial_manifest.json",
            {
                "schema_version": "auv_pc_sac_initialization.v1",
                "config": str(args.config),
                "environment_config": str(environment_path),
                "seed": seed,
                "num_envs": num_envs,
                "action_interface": action_interface,
                "actuator_residual_scale": actuator_residual_scale,
                "actuator_max_action": actuator_max_action,
                "actuator_residual_channel_gains": (
                    None
                    if actuator_channel_gains is None
                    else list(actuator_channel_gains)
                ),
                "base_residual_checkpoint": (
                    None
                    if base_residual_checkpoint is None
                    else str(base_residual_checkpoint)
                ),
                "base_residual_action_mask": list(base_residual_action_mask),
                "base_residual_bias": list(base_residual_bias),
                "residual_terminal_gate_enabled": residual_terminal_gate_enabled,
                "residual_terminal_gate_enter_m": residual_terminal_gate_enter_m,
                "residual_terminal_gate_full_m": residual_terminal_gate_full_m,
                "learner_terminal_gate_mode": learner_terminal_gate_mode,
                "authority_regularization": authority_regularization,
                "checkpoint": str(initial_checkpoint),
                "checkpoint_sha256": _sha256(initial_checkpoint),
                "policy": initialization,
            },
        )
        print(
            json.dumps(
                {
                    "init_only": True,
                    "checkpoint": str(initial_checkpoint),
                    "checkpoint_sha256": _sha256(initial_checkpoint),
                },
                ensure_ascii=False,
            )
        )
        return 0

    curriculum_hook: CurriculumTrainerHook | None = None
    if environment_config.curriculum.enabled:
        eval_script = REPO_ROOT / "scripts" / "evaluate_pc_sac.py"
        eval_root = output / "curriculum_evals"
        checkpoint_root = output / "curriculum_checkpoints"
        default_eval_config = (
            environment_path
            if args.curriculum_eval_default_config is None
            else _resolve(args.curriculum_eval_default_config)
        )
        strict_eval_config = (
            environment_path
            if args.curriculum_eval_strict_config is None
            else _resolve(args.curriculum_eval_strict_config)
        )

        def _evaluate_curriculum(stage: str, completed_transitions: int):
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_root / f"{stage}_{completed_transitions:09d}.zip"
            if not checkpoint_path.is_file():
                model.save(str(checkpoint_path))

            def _run(config_path: Path, label: str) -> dict[str, Any]:
                eval_output = eval_root / f"{stage}_{completed_transitions:09d}" / label
                command = [
                    sys.executable,
                    str(eval_script),
                    "--config",
                    str(_resolve(args.config)),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--environment-config",
                    str(config_path),
                    "--curriculum-stage",
                    stage,
                    "--output",
                    str(eval_output),
                    "--seed-start",
                    str(args.curriculum_eval_seed_start),
                    "--seed-count",
                    str(args.curriculum_eval_seed_count),
                    "--device",
                    str(args.curriculum_eval_device),
                    "--dtype",
                    str(args.dtype),
                    "--overwrite",
                ]
                if args.curriculum_disable_direct_command:
                    command.append("--disable-curriculum-direct-command")
                summary = run_eval_command(
                    command,
                    output_dir=eval_output,
                    timeout_s=args.curriculum_eval_timeout_s,
                )
                episodes_path = eval_output / "episodes.json"
                if episodes_path.is_file():
                    summary = dict(summary)
                    summary["episodes"] = json.loads(
                        episodes_path.read_text(encoding="utf-8")
                    )
                return summary

            return combine_eval_summaries(
                _run(default_eval_config, "default"),
                _run(strict_eval_config, "strict125"),
                gate=curriculum_sampler.stages[stage].gate,
            )

        curriculum_hook = CurriculumTrainerHook(
            controller=CurriculumPromotionController(
                start_stage=environment_config.curriculum.initial_stage
            ),
            sampler=curriculum_sampler,
            output_dir=output,
            eval_interval_transitions=args.curriculum_eval_interval,
            evaluate=_evaluate_curriculum,
            mix_ramp_transitions=args.curriculum_mix_ramp_transitions,
            preview_enabled=True,
            force_current_encoder_alpha_zero=args.curriculum_force_current_alpha_zero,
            promotion_enabled=environment_config.curriculum.promotion_enabled,
        )

    class MetricsCallback(BaseCallback):
        def __init__(self, hook: CurriculumTrainerHook | None = None) -> None:
            super().__init__()
            self.rows: list[dict[str, Any]] = []
            self.started = time.perf_counter()
            self._alpha = current_encoder_alpha_initial
            self._curriculum_hook = hook

        def _on_training_start(self) -> None:
            self._apply_current_encoder_alpha()

        def _apply_current_encoder_alpha(self) -> None:
            if self._curriculum_hook is not None:
                alpha = self._curriculum_hook.current_encoder_alpha(int(self.num_timesteps))
            elif current_encoder_alpha_schedule is not None:
                alpha = current_encoder_alpha_schedule.value_at(int(self.num_timesteps))
            else:
                return
            if alpha != self._alpha:
                self.model.actor.features_extractor.set_current_encoder_alpha(alpha)
                self._alpha = alpha

        def _on_step(self) -> bool:
            self._apply_current_encoder_alpha()
            if self._curriculum_hook is not None:
                self._curriculum_hook.maybe_update(int(self.num_timesteps))
            if self.n_calls == 1 or self.n_calls % 100 == 0:
                elapsed = time.perf_counter() - self.started
                row = {
                    "vector_steps": float(self.n_calls),
                    "env_transitions": float(self.num_timesteps),
                    "elapsed_s": float(elapsed),
                    "transitions_per_s": float(self.num_timesteps / max(elapsed, 1e-12)),
                    "current_encoder_alpha": float(self._alpha),
                }
                metrics = getattr(self.model, "authority_metrics", None)
                if isinstance(metrics, dict):
                    for key in ("rms_last", "rms_mean", "penalty_mean"):
                        value = metrics.get(key)
                        if value is not None:
                            row[f"authority_{key}"] = float(value)
                self.rows.append(row)
            return True

    callback = MetricsCallback(curriculum_hook)
    started = time.perf_counter()
    model.learn(
        total_timesteps=total_env_transitions,
        callback=callback,
        log_interval=int(training.get("log_interval", 100)),
    )
    if curriculum_hook is not None:
        curriculum_hook.finalize(int(model.num_timesteps))
    wall_time_s = time.perf_counter() - started
    completed = vec_env.drain_completed_infos()
    success = sum(1 for row in completed if bool(row.get("success", False)))

    checkpoint = output / "final.zip"
    model.save(output / "final")
    metrics_path = output / "training_metrics.json"
    _write_json(metrics_path, {"rows": callback.rows})
    manifest = {
        "schema_version": "auv_pc_sac_training.v1",
        "algorithm": "SAC",
        "implementation": "stable_baselines3",
        "sb3_version": getattr(__import__("stable_baselines3"), "__version__", None),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "environment_config": str(environment_path),
        "environment_config_sha256": _sha256(environment_path),
        "curriculum": environment_config.curriculum.as_dict(),
        "curriculum_promotion_enabled": bool(
            environment_config.curriculum.promotion_enabled
        ),
        "curriculum_force_current_alpha_zero": bool(
            args.curriculum_force_current_alpha_zero
        ),
        "curriculum_direct_commands_enabled": not bool(
            args.curriculum_disable_direct_command
        ),
        "direct_commands_active": bool(env.env.direct_commands),
        "speed_contract": speed_contract,
        "reference_speed_mps": speed_contract["reference_speed_mps"],
        "command_residual": (
            asdict(environment_config.command_residual)
            if action_interface == "command"
            else None
        ),
        "controller_gain_residual": (
            asdict(environment_config.controller_gain_residual)
            if action_interface == "controller_gain"
            else None
        ),
        "seed": seed,
        "device": device,
        "dtype": args.dtype,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "num_envs": num_envs,
        "action_interface": action_interface,
        "action_dim": int(env.action_dim),
        "actuator_residual_scale": actuator_residual_scale,
        "actuator_max_action": actuator_max_action,
        "actuator_residual_channel_gains": (
            None if actuator_channel_gains is None else list(actuator_channel_gains)
        ),
        "base_residual_checkpoint": (
            None if base_residual_checkpoint is None else str(base_residual_checkpoint)
        ),
        "base_residual_checkpoint_sha256": (
            None if base_residual_checkpoint is None else _sha256(base_residual_checkpoint)
        ),
        "base_residual_action_mask": list(base_residual_action_mask),
        "base_residual_bias": list(base_residual_bias),
        "residual_terminal_gate_enabled": residual_terminal_gate_enabled,
        "residual_terminal_gate_enter_m": residual_terminal_gate_enter_m,
        "residual_terminal_gate_full_m": residual_terminal_gate_full_m,
        "learner_terminal_gate_mode": learner_terminal_gate_mode,
        "authority_regularization": authority_regularization,
        "authority_training_metrics": getattr(model, "authority_metrics", None),
        "current_encoder_alpha_final": float(callback._alpha),
        "current_encoder_alpha_schedule": (
            None
            if current_encoder_alpha_schedule is None
            else current_encoder_alpha_schedule.as_dict()
        ),
        "history_length": history_length,
        "policy": initialization,
        "total_env_transitions_requested": total_env_transitions,
        "total_env_transitions_actual": int(model.num_timesteps),
        "gradient_steps_per_vector_step": gradient_steps,
        "update_to_data_ratio": utd,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "wall_time_s": wall_time_s,
        "transitions_per_s": float(model.num_timesteps / max(wall_time_s, 1e-12)),
        "completed_episodes": len(completed),
        "success_episodes": success,
        "checkpoint": str(checkpoint),
        "checkpoint_format": "sb3_sac_zip",
        "checkpoint_sha256": _sha256(checkpoint),
        "training_metrics": str(metrics_path),
        "smoke": bool(args.smoke),
        "engineering_smoke": bool(args.engineering_smoke),
        "run_kind": (
            "tiny_smoke"
            if args.smoke
            else "engineering_smoke"
            if args.engineering_smoke
            else "training"
        ),
    }
    _write_json(output / "training_manifest.json", manifest)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "algorithm",
                    "seed",
                    "num_envs",
                    "total_env_transitions_actual",
                    "wall_time_s",
                    "completed_episodes",
                    "success_episodes",
                    "checkpoint_sha256",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
