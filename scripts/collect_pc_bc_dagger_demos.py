#!/usr/bin/env python3
"""Collect one DAgger round for a PC behaviour-cloning checkpoint.

The student executes its own actions; the ALOS/PC teacher labels the visited
states.  This is the BC-checkpoint counterpart of collect_pc_dagger_demos.py,
which is tied to PPO training-config metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.pc import PC_SAFETY_TERMINAL_KINDS, PCTerminalKind
from auv_lowlevel.batch.pc_bc import PCBCPolicy, pc_action_from_actuator
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_path import (
    PC_PATH_DIMS,
    pc_path_feature_names_for_dim,
)
from auv_lowlevel.batch.pc_teacher import BatchedPCTeacher
from auv_lowlevel.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps-per-env", type=int, default=4096)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument(
        "--seed-list",
        default=None,
        help="comma-separated explicit seeds; length must equal --num-envs",
    )
    parser.add_argument("--episode-id-offset", type=int, default=1000000)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--seed-role", default="dagger")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_seed_list(value: str | None, num_envs: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        seeds = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--seed-list must contain comma-separated integers") from error
    if len(seeds) != num_envs:
        raise ValueError("--seed-list length must equal --num-envs")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seed-list must not contain duplicates")
    return seeds


def _collect_config(
    config_path: Path,
    *,
    checkpoint_path: Path,
    num_envs: int,
    steps_per_env: int,
    seed_start: int,
    seed_list: tuple[int, ...] | None,
    seed_role: str,
    episode_id_offset: int,
    action_noise_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = load_config(config_path)
    if config.arm != "PC" or config.policy_mode != "pure_controller":
        raise ValueError(f"{config_path} is not a PC pure-controller environment")
    policy = PCBCPolicy(checkpoint_path, device=device)
    path_dim = int(policy.model.path_dim)
    if path_dim not in (0, *PC_PATH_DIMS):
        raise ValueError(f"unsupported checkpoint path_dim={path_dim}")
    inner = BatchedPCLowLevelEnv(
        config,
        num_envs=num_envs,
        device=device.type,
        dtype=dtype,
        validate_finite=True,
        path_dim=path_dim,
    )

    seeds = seed_list or tuple(seed_start + index for index in range(num_envs))
    inner.reset(seeds)
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
        "p_nav": [],
        "position_ned_m": [],
        "current_estimate_valid": [],
    }
    if path_dim:
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
        if observation is None:
            raise RuntimeError("environment observation is not initialized")
        if path_dim and observation.actor_path is None:
            raise RuntimeError("environment observation has no actor_path")
        if not path_dim and observation.actor_path is not None:
            raise RuntimeError("environment unexpectedly produced actor_path")
        if inner.sensor_output is None or inner.episode_seed is None:
            raise RuntimeError("environment sensors are not initialized")
        with torch.inference_mode():
            teacher_actuator = teacher.control()
            teacher_action = pc_action_from_actuator(teacher_actuator)
            student_action = policy.act_inference(
                observation.actor_state,
                command=torch.zeros(
                    (num_envs, 0), device=device, dtype=observation.actor_state.dtype
                ),
                actor_path=observation.actor_path if path_dim else None,
                actor_current=(
                    observation.actor_current_body
                    if policy.model.actor_current_encoder is not None
                    else None
                ),
            )
            if not bool(torch.isfinite(student_action).all()):
                raise RuntimeError("student policy produced a non-finite action")
            student_action = student_action.to(dtype=dtype)
            if action_noise_std > 0.0:
                noise = torch.randn_like(student_action) * action_noise_std
                executed_action = torch.clamp(student_action + noise, -1.0, 1.0)
            else:
                executed_action = student_action

        arrays["actor_state"].append(observation.actor_state.detach().cpu().numpy())
        arrays["command"].append(
            np.zeros((num_envs, 0), dtype=np.float32)
        )
        if path_dim:
            arrays["actor_path"].append(
                observation.actor_path.detach().cpu().numpy()
            )
        arrays["action"].append(teacher_action.detach().cpu().numpy())
        arrays["episode_id"].append(episode_ids.copy())
        arrays["seed"].append(inner.episode_seed.detach().cpu().numpy())
        arrays["step"].append(episode_steps.copy())
        arrays["current_estimate_body"].append(
            observation.actor_current_body.detach().cpu().numpy()
        )
        correction_l2.append(
            torch.linalg.norm(teacher_action - student_action, dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        step = inner.step(executed_action, auto_reset=True)
        infos = step.info
        done = step.done
        arrays["actual_actuator"].append(
            infos["actual_actuator"].detach().cpu().numpy()
        )
        arrays["reward"].append(step.reward.total.detach().cpu().numpy())
        arrays["p_nav"].append(infos["p_nav"].detach().cpu().numpy())
        arrays["position_ned_m"].append(
            infos["position_ned_m"].detach().cpu().numpy()
        )
        arrays["current_estimate_valid"].append(
            infos["current_estimate_valid"].detach().cpu().numpy()
        )
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
    metadata = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config.config_sha256,
        "config_name": config.name,
        "pc_stage": config.pc_stage,
        "sensor_profile": config.sensors.profile,
        "sensor_seed": config.sensors.seed,
        "actor_observation_source": (
            "ideal" if config.observation.ideal_actor_observations else "nominal"
        ),
        "student": {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "path_dim": policy.model.path_dim,
            "actor_command_dim": policy.model.actor_command_dim,
            "actor_observation_source": (
                "ideal" if config.observation.ideal_actor_observations else "nominal"
            ),
            "actor_current_field": "actor_current_body",
        },
        "teacher_parameters": {
            "pitch_integral_gain": teacher.pitch_integral_gain,
            "pitch_integral_limit_rad_s": teacher.pitch_integral_limit_rad_s,
            "speed_feedback": teacher.speed_feedback,
            "speed_integral_limit_rpm": config.controller.max_speed_integral_rpm,
        },
        "normalization": config.normalization.as_dict(),
        "path_dim": path_dim,
        "path_feature_names": (
            list(pc_path_feature_names_for_dim(path_dim)) if path_dim else []
        ),
        "actor_command_dim": 0,
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
        "samples": int(combined["actor_state"].shape[0]),
        "episodes_started": int(next_episode_id - episode_id_offset),
        "goal_count": goal_count,
        "safety_events": safety_count,
        "seed_start": int(min(seeds)),
        "seed_end": int(max(seeds)),
        "seed_list": list(seeds),
        "seed_role": seed_role,
        "source": seed_role,
        "action_noise_std": action_noise_std,
        "student_correction_l2_mean": float(np.mean(correction)),
        "student_correction_l2_p95": float(np.percentile(correction, 95)),
        "sensor_statistics": {
            "nav_position_error_mean_m": float(
                np.mean(
                    np.linalg.norm(
                        combined["p_nav"] - combined["position_ned_m"], axis=1
                    )
                )
            )
            if "p_nav" in combined
            else None,
            "current_estimate_valid_rate": float(
                np.mean(combined["current_estimate_valid"])
            )
            if "current_estimate_valid" in combined
            else None,
        },
    }
    return combined, metadata


def main() -> int:
    args = _parse_args()
    if args.num_envs <= 0 or args.steps_per_env <= 0:
        raise ValueError("--num-envs and --steps-per-env must be positive")
    if args.seed_start < 0 or args.episode_id_offset < 0:
        raise ValueError("--seed-start and --episode-id-offset must be non-negative")
    if args.action_noise_std < 0.0:
        raise ValueError("--action-noise-std must be non-negative")
    seed_list = _parse_seed_list(args.seed_list, args.num_envs)
    output = args.output.resolve()
    dataset_path = output / "demos.npz"
    manifest_path = output / "demos.manifest.json"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    parts: dict[str, list[np.ndarray]] = {}
    configs: list[dict[str, Any]] = []
    episode_offset = int(args.episode_id_offset)
    for index, config_path in enumerate(args.config):
        part, metadata = _collect_config(
            config_path,
            checkpoint_path=args.checkpoint.resolve(),
            num_envs=args.num_envs,
            steps_per_env=args.steps_per_env,
            seed_start=args.seed_start + index * (args.num_envs + 1),
            seed_list=seed_list,
            seed_role=args.seed_role,
            episode_id_offset=episode_offset,
            action_noise_std=args.action_noise_std,
            device=device,
            dtype=dtype,
        )
        for name, value in part.items():
            parts.setdefault(name, []).append(value)
        configs.append(metadata)
        episode_offset += int(metadata["episodes_started"])
        print(
            f"collected {config_path}: samples={metadata['samples']} "
            f"episodes={metadata['episodes_started']}",
            flush=True,
        )

    combined = {
        name: np.concatenate(values, axis=0).astype(
            np.float32 if name not in {"episode_id", "seed", "step"} else np.int64
        )
        for name, values in parts.items()
    }
    combined["observation_schema"] = np.asarray("pc_v1")
    combined["action_envelope"] = np.asarray("pc20")
    np.savez_compressed(dataset_path, **combined)
    manifest = {
        "schema_version": "auv_pc_bc_dataset.v1",
        "observation_schema": "pc_v1",
        "action_envelope": "pc20",
        "sample_count": int(combined["actor_state"].shape[0]),
        "episode_count": int(len(np.unique(combined["episode_id"]))),
        "path_dim": (
            int(combined["actor_path"].shape[1]) if "actor_path" in combined else 0
        ),
        "actor_command_dim": int(combined["command"].shape[1]),
        "path_feature_names": (
            list(pc_path_feature_names_for_dim(int(combined["actor_path"].shape[1])))
            if "actor_path" in combined
            else []
        ),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "configs": configs,
        "git": {"commit": None, "dirty": None},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
