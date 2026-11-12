from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.config import load_config
from auv_lowlevel.env import LowLevelEnv
from auv_lowlevel.observation_schema import observation_schema_for_dim
from auv_lowlevel.randomization import training_seeds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect I-arm PD/PI teacher demonstrations")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lowlevel/odarrl_control_i_bc.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lowlevel/i_bc/demos"),
    )
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--role",
        choices=("bc_teacher", "residual_teacher"),
        default="bc_teacher",
        help=(
            "bc_teacher collects the no-current I-arm BC dataset; "
            "residual_teacher collects current-aware B0 labels with hidden-current observations"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _collect_episode(payload: tuple[str, int, int, int | None, str]) -> dict[str, Any]:
    config_path, seed, episode_id, max_steps, shard_dir_text = payload
    config = load_config(config_path)
    if max_steps is not None:
        config = replace(config, plant=replace(config.plant, max_episode_steps=max_steps))
    env = LowLevelEnv(config, expose_current=None)
    env.reset(seed)
    rows: dict[str, list[Any]] = {
        "observation": [],
        "action": [],
        "step": [],
        "time_s": [],
        "actual_actuator": [],
        "state_position_ned_m": [],
        "state_quaternion_xyzw": [],
        "state_linear_velocity_body_mps": [],
        "state_angular_velocity_body_rps": [],
        "reference_position_ned_m": [],
        "reference_quaternion_xyzw": [],
        "current_ned_mps": [],
        "teacher_post_integral_rpm": [],
    }
    termination: str | None = None
    while True:
        context = env.last_context
        result = env.step()
        info = result.info
        rows["observation"].append(context.observation)
        rows["action"].append(info["base_command"])
        rows["step"].append(env.step_count - 1)
        rows["time_s"].append(context.state.t_s)
        rows["actual_actuator"].append(info["actual_actuator"])
        rows["state_position_ned_m"].append(context.state.position_ned_m)
        rows["state_quaternion_xyzw"].append(context.state.quat_xyzw)
        rows["state_linear_velocity_body_mps"].append(
            context.state.linear_velocity_body_mps
        )
        rows["state_angular_velocity_body_rps"].append(
            context.state.angular_velocity_body_rps
        )
        rows["reference_position_ned_m"].append(context.reference.position_ned_m)
        rows["reference_quaternion_xyzw"].append(context.reference.quat_xyzw)
        rows["current_ned_mps"].append(context.current_ned_mps)
        rows["teacher_post_integral_rpm"].append(info["teacher_post_integral_rpm"])
        if result.terminated or result.truncated:
            termination = str(info["terminal_kind"])
            break

    shard_dir = Path(shard_dir_text)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"episode_{episode_id:05d}_seed_{seed}.npz"
    count = len(rows["observation"])
    arrays = {
        "observation": np.asarray(rows["observation"], dtype=np.float32),
        "action": np.asarray(rows["action"], dtype=np.float32),
        "episode_id": np.full(count, episode_id, dtype=np.int64),
        "seed": np.full(count, seed, dtype=np.int64),
        "step": np.asarray(rows["step"], dtype=np.int64),
        "time_s": np.asarray(rows["time_s"], dtype=np.float64),
        "actual_actuator": np.asarray(rows["actual_actuator"], dtype=np.float32),
        "state_position_ned_m": np.asarray(rows["state_position_ned_m"], dtype=np.float32),
        "state_quaternion_xyzw": np.asarray(rows["state_quaternion_xyzw"], dtype=np.float32),
        "state_linear_velocity_body_mps": np.asarray(
            rows["state_linear_velocity_body_mps"], dtype=np.float32
        ),
        "state_angular_velocity_body_rps": np.asarray(
            rows["state_angular_velocity_body_rps"], dtype=np.float32
        ),
        "reference_position_ned_m": np.asarray(
            rows["reference_position_ned_m"], dtype=np.float32
        ),
        "reference_quaternion_xyzw": np.asarray(
            rows["reference_quaternion_xyzw"], dtype=np.float32
        ),
        "current_ned_mps": np.asarray(rows["current_ned_mps"], dtype=np.float32),
        "teacher_post_integral_rpm": np.asarray(
            rows["teacher_post_integral_rpm"], dtype=np.float32
        ),
        "observation_schema": np.asarray(
            observation_schema_for_dim(config.observation.dim).version
        ),
    }
    np.savez_compressed(shard_path, **arrays)
    return {
        "episode_id": episode_id,
        "seed": seed,
        "samples": count,
        "termination": termination,
        "success": termination == "goal",
        "shard": str(shard_path.resolve()),
        "constraint_violations": int(env._constraint_violations),
        "max_shadow_teacher_margin_rpm": float(env._min_shadow_teacher_margin_rpm),
    }


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    schema = observation_schema_for_dim(config.observation.dim)
    if args.role == "bc_teacher":
        if config.arm != "I":
            raise ValueError("I-arm demo collection requires an I-arm config")
        if config.current.enabled:
            raise ValueError("I-arm BC teacher data must be collected without current")
    else:
        if config.arm != "B0":
            raise ValueError("residual teacher collection requires the B0 config")
        if not config.current.enabled:
            raise ValueError("residual teacher collection requires enabled current")
        if not config.guidance.compensate_current or not config.controller.use_current_field:
            raise ValueError("residual teacher must be current-aware")
        if not config.observation.expose_current:
            raise ValueError("B0 residual teacher config must expose current")
    if args.episodes <= 0 or args.workers <= 0:
        raise ValueError("--episodes and --workers must be positive")
    seed_start = (
        config.evaluation.training_seed_start
        if args.seed_start is None
        else args.seed_start
    )
    seeds = training_seeds(args.episodes, seed_start)
    output = args.output.resolve()
    shard_dir = output / "shards"
    output.mkdir(parents=True, exist_ok=True)
    payloads = [
        (str(args.config.resolve()), seed, episode_id, args.max_steps, str(shard_dir))
        for episode_id, seed in enumerate(seeds)
    ]
    metadata: list[dict[str, Any]] = []
    if args.workers == 1:
        metadata = [_collect_episode(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_collect_episode, payload): payload[1]
                for payload in payloads
            }
            for future in as_completed(futures):
                seed = futures[future]
                metadata.append(future.result())
                print(f"finished episode seed {seed}", flush=True)
    metadata.sort(key=lambda item: int(item["episode_id"]))

    names = [
        "observation",
        "action",
        "episode_id",
        "seed",
        "step",
        "time_s",
        "actual_actuator",
        "state_position_ned_m",
        "state_quaternion_xyzw",
        "state_linear_velocity_body_mps",
        "state_angular_velocity_body_rps",
        "reference_position_ned_m",
        "reference_quaternion_xyzw",
        "current_ned_mps",
        "teacher_post_integral_rpm",
    ]
    combined: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for item in metadata:
        with np.load(item["shard"]) as shard:
            for name in names:
                combined[name].append(np.asarray(shard[name]))
    demo_path = output / "demos.npz"
    demo_arrays = {
        name: np.concatenate(values, axis=0) for name, values in combined.items()
    }
    demo_arrays["observation_schema"] = np.asarray(schema.version)
    np.savez_compressed(demo_path, **demo_arrays)
    sample_count = int(sum(int(item["samples"]) for item in metadata))
    manifest = {
        "schema_version": "auv_odarrl_demos.v1",
        "role": args.role,
        "expert_type": (
            "pd_pi_current_aware" if args.role == "residual_teacher" else "pd_pi"
        ),
        "teacher_current_aware": bool(config.controller.use_current_field),
        "observation_current_exposed": config.observation.expose_current,
        "observation_speed_feedback": config.observation.speed_feedback,
        "observation_current_contract": config.observation.current_contract,
        "config_path": str(args.config.resolve()),
        "config_sha256": config.config_sha256,
        "git_commit": _git_commit(),
        "current_enabled": config.current.enabled,
        "observation_dim": config.observation.dim,
        "observation_schema": schema.version,
        "episode_count": len(metadata),
        "sample_count": sample_count,
        "success_count": sum(bool(item["success"]) for item in metadata),
        "success_rate": float(np.mean([bool(item["success"]) for item in metadata])),
        "seed_start": seed_start,
        "seed_end": seeds[-1],
        "max_steps_override": args.max_steps,
        "workers": args.workers,
        "demos_path": str(demo_path.resolve()),
        "demos_sha256": _sha256(demo_path),
        "constraint_violations": sum(int(item["constraint_violations"]) for item in metadata),
        "max_shadow_teacher_margin_rpm": max(
            float(item["max_shadow_teacher_margin_rpm"]) for item in metadata
        ),
        "episodes": metadata,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in (
        "sample_count",
        "episode_count",
        "success_rate",
        "constraint_violations",
        "max_shadow_teacher_margin_rpm",
        "demos_sha256",
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
