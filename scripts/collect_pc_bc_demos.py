from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.pc_bc import (
    PC_ACTION_ENVELOPE,
    PC_BC_DATASET_SCHEMA,
    pc_action_from_actuator,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_path import (
    PC_PATH_DIMS,
    pc_path_feature_names_for_dim,
)
from auv_lowlevel.batch.pc_teacher import BatchedPCTeacher
from auv_lowlevel.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PC20 PD/PI teacher demonstrations")
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps-per-env", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument(
        "--seed-list",
        default=None,
        help="comma-separated explicit seeds; length must equal --num-envs",
    )
    parser.add_argument("--path-dim", type=int, choices=(0, *PC_PATH_DIMS), default=0)
    parser.add_argument("--actor-command-dim", type=int, choices=(0, 3), default=3)
    parser.add_argument("--seed-role", default="training")
    parser.add_argument("--teacher-label", default=None)
    parser.add_argument("--teacher-selection", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repository,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


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
    output: Path,
    num_envs: int,
    steps_per_env: int,
    device: str,
    dtype: torch.dtype,
    seed_start: int,
    seed_list: tuple[int, ...] | None,
    episode_base: int,
    path_dim: int,
    actor_command_dim: int,
    seed_role: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = load_config(config_path)
    if config.arm != "PC" or config.policy_mode != "pure_controller":
        raise ValueError(f"{config_path} is not a PC pure-controller config")
    if config.sensors is None:
        raise ValueError(f"{config_path} requires a sensor profile")
    env = BatchedPCLowLevelEnv(
        config,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        path_dim=path_dim,
    )
    seeds = seed_list or tuple(seed_start + index for index in range(num_envs))
    env.reset(seeds)
    teacher = BatchedPCTeacher(env)
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
        "dvl_bottom_valid": [],
        "dvl_water_valid": [],
        "current_estimate_valid": [],
        "usbl_valid": [],
        "usbl_outlier": [],
        "nav_age_s": [],
        "dvl_bottom_age_s": [],
        "dvl_water_age_s": [],
        "usbl_age_s": [],
        "current_estimate_age_s": [],
    }
    if path_dim:
        arrays["actor_path"] = []
    episode_ids = np.arange(episode_base, episode_base + num_envs, dtype=np.int64)
    next_episode = episode_base + num_envs
    safety_events = 0

    for _ in range(steps_per_env):
        observation = env.observation
        episode_seeds = env.episode_seed
        episode_steps = env.step_count
        assert episode_seeds is not None and episode_steps is not None
        with torch.inference_mode():
            teacher_actuator = teacher.control()
            teacher_action = pc_action_from_actuator(teacher_actuator)
        result = env.step(teacher_action)
        arrays["actor_state"].append(observation.actor_state.detach().cpu().numpy())
        if path_dim:
            assert observation.actor_path is not None
            arrays["actor_path"].append(
                observation.actor_path.detach().cpu().numpy()
            )
        command_full = torch.stack(
            [
                observation.command.heading_rad,
                observation.command.pitch_rad,
                observation.command.speed_mps,
            ],
            dim=1,
        )
        command = command_full[:, :actor_command_dim]
        arrays["command"].append(command.detach().cpu().numpy())
        arrays["action"].append(teacher_action.detach().cpu().numpy())
        arrays["episode_id"].append(episode_ids.copy())
        arrays["seed"].append(episode_seeds.detach().cpu().numpy())
        arrays["step"].append(episode_steps.detach().cpu().numpy())
        arrays["actual_actuator"].append(
            result.info["actual_actuator"].detach().cpu().numpy()
        )
        arrays["reward"].append(result.reward.total.detach().cpu().numpy())
        sensor = env.sensor_output
        assert sensor is not None
        arrays["current_estimate_body"].append(
            observation.actor_current_body.detach().cpu().numpy()
        )
        arrays["p_nav"].append(sensor.p_nav.detach().cpu().numpy())
        assert env.eta is not None
        arrays["position_ned_m"].append(env.eta[:, :3].detach().cpu().numpy())
        arrays["dvl_bottom_valid"].append(
            sensor.bottom_lock_valid.detach().cpu().numpy()
        )
        arrays["dvl_water_valid"].append(
            sensor.water_track_valid.detach().cpu().numpy()
        )
        arrays["current_estimate_valid"].append(
            sensor.current_estimate_valid.detach().cpu().numpy()
        )
        arrays["usbl_valid"].append(sensor.usbl_valid.detach().cpu().numpy())
        arrays["usbl_outlier"].append(sensor.usbl_outlier.detach().cpu().numpy())
        arrays["nav_age_s"].append(sensor.nav_age_s.detach().cpu().numpy())
        arrays["dvl_bottom_age_s"].append(
            sensor.dvl_bottom_age_s.detach().cpu().numpy()
        )
        arrays["dvl_water_age_s"].append(
            sensor.dvl_water_age_s.detach().cpu().numpy()
        )
        arrays["usbl_age_s"].append(sensor.usbl_age_s.detach().cpu().numpy())
        arrays["current_estimate_age_s"].append(
            sensor.current_estimate_age_s.detach().cpu().numpy()
        )
        safety_events += int(result.info["safety_event"].sum().item())

        done = result.done.detach().cpu().numpy()
        for row in np.flatnonzero(done):
            episode_ids[int(row)] = next_episode
            next_episode += 1
        teacher.reset_mask(result.done)

    combined = {
        name: np.concatenate(values, axis=0).astype(
            np.float32 if name not in {"episode_id", "seed", "step"} else np.int64
        )
        for name, values in arrays.items()
    }
    for name, value in combined.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{config_path}: {name} contains NaN or Inf")
    metadata = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config.config_sha256,
        "config_file_sha256": _sha256(config_path),
        "config_name": config.name,
        "pc_stage": config.pc_stage,
        "current_crab_enabled": bool(
            getattr(config.guidance, "current_crab_enabled", False)
        ),
        "actor_current_encoder": "none",
        "actor_observation_source": (
            "ideal" if config.observation.ideal_actor_observations else "nominal"
        ),
        "actor_current_field": "actor_current_body",
        "z_t_dim": None,
        "actor_top_level_input_dim": None,
        "pc_dade_enabled": None,
        "seed_role": seed_role,
        "sensor_profile": config.sensors.profile,
        "sensor_seed": config.sensors.seed,
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
        "actor_command_dim": actor_command_dim,
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
        "samples": int(combined["actor_state"].shape[0]),
        "episodes_started": int(next_episode - episode_base),
        "safety_events": safety_events,
        "seed_start": seed_start,
        "seed_end": seed_start + num_envs - 1,
        "seed_list": list(seeds),
        "sensor_statistics": {
            "nav_position_error_mean_m": float(
                np.mean(
                    np.linalg.norm(
                        combined["p_nav"] - combined["position_ned_m"],
                        axis=1,
                    )
                )
            ),
            "nav_position_error_rmse_m": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.linalg.norm(
                                combined["p_nav"] - combined["position_ned_m"],
                                axis=1,
                            )
                        )
                    )
                )
            ),
            "dvl_bottom_outage_rate": float(
                1.0 - np.mean(combined["dvl_bottom_valid"])
            ),
            "dvl_water_outage_rate": float(
                1.0 - np.mean(combined["dvl_water_valid"])
            ),
            "current_estimate_valid_rate": float(
                np.mean(combined["current_estimate_valid"])
            ),
            "usbl_outlier_rate": float(np.mean(combined["usbl_outlier"])),
            "usbl_valid_rate": float(np.mean(combined["usbl_valid"])),
            "nav_age_s_mean": float(np.mean(combined["nav_age_s"])),
            "dvl_bottom_age_s_mean": float(np.mean(combined["dvl_bottom_age_s"])),
            "current_estimate_age_s_mean": float(
                np.mean(combined["current_estimate_age_s"])
            ),
        },
    }
    return combined, metadata


def main() -> int:
    args = _parse_args()
    if args.num_envs <= 0 or args.steps_per_env <= 0:
        raise ValueError("--num-envs and --steps-per-env must be positive")
    seed_list = _parse_seed_list(args.seed_list, args.num_envs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "demos.npz"
    manifest_path = output / "demos.manifest.json"
    if not args.overwrite and (dataset_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"output already contains PC demonstrations: {output}")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    started_at = datetime.now(timezone.utc).isoformat()
    teacher_selection: dict[str, Any] | None = None
    if args.teacher_selection is not None:
        selection_path = args.teacher_selection.resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(f"teacher selection JSON not found: {selection_path}")
        teacher_selection = {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
        }

    names = (
        "actor_state",
        "command",
        "action",
        "episode_id",
        "seed",
        "step",
        "actual_actuator",
        "reward",
        "current_estimate_body",
        "p_nav",
        "position_ned_m",
        "dvl_bottom_valid",
        "dvl_water_valid",
        "current_estimate_valid",
        "usbl_valid",
        "usbl_outlier",
        "nav_age_s",
        "dvl_bottom_age_s",
        "dvl_water_age_s",
        "usbl_age_s",
        "current_estimate_age_s",
    )
    if args.path_dim:
        names = (*names, "actor_path")
    merged: dict[str, list[np.ndarray]] = {name: [] for name in names}
    config_metadata: list[dict[str, Any]] = []
    episode_base = 0
    for config_path in args.config:
        config = load_config(config_path)
        seed_start = (
            config.evaluation.training_seed_start
            if args.seed_start is None
            else args.seed_start
        )
        arrays, metadata = _collect_config(
            config_path,
            output=output,
            num_envs=args.num_envs,
            steps_per_env=args.steps_per_env,
            device=args.device,
            dtype=dtype,
            seed_start=seed_start,
            seed_list=seed_list,
            episode_base=episode_base,
            path_dim=args.path_dim,
            actor_command_dim=args.actor_command_dim,
            seed_role=args.seed_role,
        )
        for name in names:
            merged[name].append(arrays[name])
        config_metadata.append(metadata)
        episode_base = int(arrays["episode_id"].max()) + 1
        print(
            f"collected {metadata['samples']} samples from {config_path}",
            flush=True,
        )

    payload = {
        name: np.concatenate(values, axis=0)
        for name, values in merged.items()
    }
    payload["observation_schema"] = np.asarray("pc_v1")
    payload["action_envelope"] = np.asarray(PC_ACTION_ENVELOPE)
    np.savez_compressed(dataset_path, **payload)
    manifest = {
        "schema_version": PC_BC_DATASET_SCHEMA,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "observation_schema": "pc_v1",
        "action_envelope": PC_ACTION_ENVELOPE,
        "sample_count": int(payload["actor_state"].shape[0]),
        "episode_count": int(np.unique(payload["episode_id"]).size),
        "action_full_scale": [
            float(np.deg2rad(20.0)),
            float(np.deg2rad(20.0)),
            1525.0,
        ],
        "teacher": "BatchedPCTeacher(20deg, ground-speed command, pitch PI)",
        "path_dim": args.path_dim,
        "path_feature_names": (
            list(pc_path_feature_names_for_dim(args.path_dim))
            if args.path_dim
            else []
        ),
        "actor_command_dim": args.actor_command_dim,
        "actor_observation_source": (
            "ideal"
            if all(item["actor_observation_source"] == "ideal" for item in config_metadata)
            else "mixed"
            if any(item["actor_observation_source"] == "ideal" for item in config_metadata)
            else "nominal"
        ),
        "device": args.device,
        "dtype": args.dtype,
        "seed_role": args.seed_role,
        "teacher_label": args.teacher_label,
        "teacher_selection": teacher_selection,
        "command_line": shlex.join(sys.argv),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "configs": config_metadata,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
