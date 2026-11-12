#!/usr/bin/env python3
"""Collect student-state teacher labels for one PC frozen-BC residual DAgger round."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.checkpoint import validate_pc_checkpoint_metadata
from auv_lowlevel.batch.pc import PC_SAFETY_TERMINAL_KINDS, PCTerminalKind
from auv_lowlevel.batch.pc_bc import PC_ACTION_ENVELOPE, pc_action_from_actuator
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import PC_PATH_FEATURE_NAMES
from auv_lowlevel.batch.pc_policy import PC_SCHEMA_VERSION, PCActorCritic, pc_schema_hash
from auv_lowlevel.batch.pc_teacher import BatchedPCTeacher
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import LowLevelConfig, load_config
from auv_lowlevel.ppo_config import load_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps-per-env", type=int, default=2048)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--episode-id-offset", type=int, default=0)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _normalization(config: LowLevelConfig) -> PCNormalization:
    configured = config.normalization
    return PCNormalization(
        position_scale_m=configured.position_scale_m,
        attitude_scale_rad=configured.attitude_scale_rad,
        angular_rate_scale_rad_s=configured.angular_rate_scale_rad_s,
        surge_speed_scale_mps=configured.surge_speed_scale_mps,
        sway_heave_speed_scale_mps=configured.sway_heave_speed_scale_mps,
        current_scale_mps=configured.current_scale_mps,
    )


def _build_policy(
    training_config_path: Path,
    checkpoint_path: Path,
    *,
    num_actor_obs: int,
    num_critic_obs: int,
    device: torch.device,
) -> tuple[PCActorCritic, dict[str, Any]]:
    training = load_training_config(training_config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = validate_pc_checkpoint_metadata(
        checkpoint,
        expected_actor_observation_dim=num_actor_obs,
        expected_critic_observation_dim=num_critic_obs,
        expected_schema_hash=pc_schema_hash(
            path_dim=training.pc.path_dim,
            dade_history_mode=training.pc.dade_history_mode,
            actor_command_dim=training.pc.actor_command_dim,
            critic_command_dim=training.pc.critic_command_dim,
            history_path_dim=training.pc.history_path_dim,
        ),
    )
    assert metadata is not None
    expected_hash = sha256_file(training_config_path)
    if metadata["training_config_sha256"] != expected_hash:
        raise ValueError(
            "student checkpoint training-config hash does not match --training-config"
        )
    policy_config = training.rsl_train_cfg()["policy"]
    policy = PCActorCritic(
        num_actor_obs,
        num_critic_obs,
        3,
        **{key: value for key, value in policy_config.items() if key != "class_name"},
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("student checkpoint does not contain model_state_dict")
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy, metadata


def _validate_environment(
    config: LowLevelConfig,
    metadata: dict[str, Any],
    *,
    env: BatchTorchPCVecEnv,
) -> None:
    normalization = _normalization(config).as_dict()
    if metadata["normalization"] != normalization:
        raise ValueError("environment config changes PC normalization scales")
    if metadata["pc_stage"] != str(config.pc_stage):
        raise ValueError("environment config changes the PC stage")
    if metadata["actuator_envelope"] != config.actuator.envelope:
        raise ValueError("environment config changes the actuator envelope")
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
        raise ValueError("environment config changes the policy action scale")
    if env.num_obs != metadata["actor_observation_dim"]:
        raise ValueError("environment actor observation dimension does not match student")
    if env.num_privileged_obs != metadata["critic_observation_dim"]:
        raise ValueError("environment critic observation dimension does not match student")


def _collect_config(
    config_path: Path,
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    num_envs: int,
    steps_per_env: int,
    seed_start: int,
    episode_id_offset: int,
    action_noise_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = load_config(config_path)
    if config.arm != "PC" or config.policy_mode != "pure_controller":
        raise ValueError(f"{config_path} is not a PC pure-controller environment")
    training = load_training_config(training_config_path)
    if training.history is None or training.pc is None:
        raise ValueError("training config must contain PC history and pc blocks")
    inner = BatchedPCLowLevelEnv(
        config,
        num_envs=num_envs,
        device=device.type,
        dtype=dtype,
        validate_finite=True,
        path_dim=training.pc.path_dim,
    )
    env = BatchTorchPCVecEnv(
        inner,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        short_stride=training.history.short_stride,
        long_stride=training.history.long_stride,
        normalization=_normalization(config),
        current_frame_dropout_prob=0.0,
        actor_command_dim=training.pc.actor_command_dim,
        critic_command_dim=training.pc.critic_command_dim,
        history_path_dim=training.pc.history_path_dim,
    )
    policy, metadata = _build_policy(
        training_config_path,
        checkpoint_path,
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        device=device,
    )
    _validate_environment(config, metadata, env=env)

    seeds = tuple(seed_start + index for index in range(num_envs))
    observations, _ = env.reset(seeds)
    teacher = BatchedPCTeacher(inner)
    teacher.reset(num_envs)

    arrays: dict[str, list[np.ndarray]] = {
        "actor_state": [],
        "command": [],
        "action": [],
        "episode_id": [],
        "seed": [],
        "step": [],
        "actual_actuator": [],
        "reward": [],
        "current_estimate_body": [],
    }
    if training.pc.path_dim:
        arrays["actor_path"] = []
    episode_ids = np.arange(
        episode_id_offset, episode_id_offset + num_envs, dtype=np.int64
    )
    next_episode_id = episode_id_offset + num_envs
    episode_steps = np.zeros(num_envs, dtype=np.int64)
    correction_l2: list[np.ndarray] = []
    goal_count = 0
    safety_count = 0

    for _ in range(steps_per_env):
        observation = inner.observation
        assert inner.episode_seed is not None
        assert inner.sensor_output is not None
        with torch.inference_mode():
            teacher_actuator = teacher.control()
            teacher_action = pc_action_from_actuator(teacher_actuator)
            student_action = policy.act_inference(observations)
            if not bool(torch.isfinite(student_action).all()):
                raise RuntimeError("student policy produced a non-finite action")
            if action_noise_std > 0.0:
                noise = torch.randn_like(student_action) * action_noise_std
                executed_action = torch.clamp(student_action + noise, -1.0, 1.0)
            else:
                executed_action = student_action

        arrays["actor_state"].append(
            observation.actor_state.detach().cpu().numpy()
        )
        command_full = torch.stack(
            [
                observation.command.heading_rad,
                observation.command.pitch_rad,
                observation.command.speed_mps,
            ],
            dim=1,
        )
        command = command_full[:, : training.pc.actor_command_dim]
        arrays["command"].append(command.detach().cpu().numpy())
        if training.pc.path_dim:
            assert observation.actor_path is not None
            arrays["actor_path"].append(
                observation.actor_path.detach().cpu().numpy()
            )
        arrays["action"].append(teacher_action.detach().cpu().numpy())
        arrays["episode_id"].append(episode_ids.copy())
        arrays["seed"].append(inner.episode_seed.detach().cpu().numpy())
        arrays["step"].append(episode_steps.copy())
        arrays["current_estimate_body"].append(
            inner.sensor_output.current_estimate_body.detach().cpu().numpy()
        )
        correction_l2.append(
            torch.linalg.norm(teacher_action - student_action, dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        observations, _, done, infos = env.step(executed_action, auto_reset=True)
        arrays["actual_actuator"].append(
            infos["actual_actuator"].detach().cpu().numpy()
        )
        arrays["reward"].append(infos["raw_reward"].detach().cpu().numpy())
        done_np = done.detach().cpu().numpy() > 0.5
        terminal_kind = infos["terminal_kind"].detach().cpu().numpy()
        goal_count += int(
            np.sum((terminal_kind == int(PCTerminalKind.GOAL)) & done_np)
        )
        safety_count += int(
            np.sum(
                np.isin(
                    terminal_kind,
                    tuple(int(kind) for kind in PC_SAFETY_TERMINAL_KINDS),
                )
                & done_np
            )
        )
        episode_steps += 1
        for row in np.flatnonzero(done_np):
            episode_ids[int(row)] = next_episode_id
            next_episode_id += 1
            episode_steps[int(row)] = 0
        teacher.reset_mask(done.detach().to(dtype=torch.bool))

    combined = {
        name: np.concatenate(values, axis=0).astype(
            np.float32 if name not in {"episode_id", "seed", "step"} else np.int64
        )
        for name, values in arrays.items()
    }
    for name, value in combined.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{config_path}: {name} contains NaN or Inf")
    correction = np.concatenate(correction_l2, axis=0)
    metadata_out = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config.config_sha256,
        "config_name": config.name,
        "pc_stage": config.pc_stage,
        "sensor_profile": config.sensors.profile if config.sensors is not None else None,
        "sensor_seed": config.sensors.seed if config.sensors is not None else None,
        "normalization": config.normalization.as_dict(),
        "actor_command_dim": training.pc.actor_command_dim,
        "path_dim": training.pc.path_dim,
        "path_feature_names": (
            list(PC_PATH_FEATURE_NAMES) if training.pc.path_dim else []
        ),
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
        "samples": int(combined["actor_state"].shape[0]),
        "episodes_started": int(next_episode_id - episode_id_offset),
        "goal_terminations": goal_count,
        "safety_terminations": safety_count,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "correction_l2_mean": float(np.mean(correction)),
        "correction_l2_p95": float(np.percentile(correction, 95)),
        "correction_l2_max": float(np.max(correction)),
    }
    return combined, metadata_out


def main() -> int:
    args = _parse_args()
    if args.num_envs <= 0 or args.steps_per_env <= 0:
        raise ValueError("--num-envs and --steps-per-env must be positive")
    if args.seed_start < 0 or args.episode_id_offset < 0:
        raise ValueError("--seed-start and --episode-id-offset must be non-negative")
    if args.action_noise_std < 0.0:
        raise ValueError("--action-noise-std must be non-negative")
    training_config_path = _resolve(args.student_training_config)
    checkpoint_path = _resolve(args.checkpoint)
    output = _resolve(args.output)
    dataset_path = output / "demos.npz"
    manifest_path = output / "demos.manifest.json"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    combined_parts: dict[str, list[np.ndarray]] = {}
    configs: list[dict[str, Any]] = []
    episode_offset = int(args.episode_id_offset)
    for index, config_path in enumerate(args.config):
        part, metadata = _collect_config(
            config_path,
            training_config_path=training_config_path,
            checkpoint_path=checkpoint_path,
            num_envs=args.num_envs,
            steps_per_env=args.steps_per_env,
            seed_start=args.seed_start + index * (args.num_envs + 1),
            episode_id_offset=episode_offset,
            action_noise_std=args.action_noise_std,
            device=device,
            dtype=dtype,
        )
        for name, value in part.items():
            combined_parts.setdefault(name, []).append(value)
        configs.append(metadata)
        episode_offset += int(metadata["episodes_started"])
        print(
            f"collected {config_path}: samples={metadata['samples']} "
            f"episodes={metadata['episodes_started']}",
            flush=True,
        )

    if not combined_parts:
        raise ValueError("at least one --config is required")
    payload = {
        name: np.concatenate(values, axis=0).astype(
            np.float32 if name not in {"episode_id", "seed", "step"} else np.int64
        )
        for name, values in combined_parts.items()
    }
    payload["observation_schema"] = np.asarray(PC_SCHEMA_VERSION)
    payload["action_envelope"] = np.asarray(PC_ACTION_ENVELOPE)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **payload)

    correction_means = [float(item["correction_l2_mean"]) for item in configs]
    manifest = {
        "schema_version": "auv_pc_dagger_dataset.v1",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "observation_schema": PC_SCHEMA_VERSION,
        "action_envelope": PC_ACTION_ENVELOPE,
        "sample_count": int(payload["actor_state"].shape[0]),
        "episode_count": int(np.unique(payload["episode_id"]).size),
        "actor_command_dim": int(payload["command"].shape[1]),
        "path_dim": 0 if "actor_path" not in payload else int(payload["actor_path"].shape[1]),
        "configs": configs,
        "student": {
            "training_config": str(training_config_path.resolve()),
            "training_config_sha256": sha256_file(training_config_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "parameters": {
            "num_envs": args.num_envs,
            "steps_per_env": args.steps_per_env,
            "seed_start": args.seed_start,
            "episode_id_offset": args.episode_id_offset,
            "action_noise_std": args.action_noise_std,
            "device": args.device,
            "dtype": args.dtype,
        },
        "correction_l2_mean": float(np.mean(correction_means)),
        "config_goal_terminations": sum(
            int(item["goal_terminations"]) for item in configs
        ),
        "config_safety_terminations": sum(
            int(item["safety_terminations"]) for item in configs
        ),
        "git": _git_state(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "manifest": str(manifest_path),
                "sample_count": manifest["sample_count"],
                "episode_count": manifest["episode_count"],
                "configs": [item["config_name"] for item in configs],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
