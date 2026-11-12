from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from auv_core.control.remus100 import Remus100TrackingController
from auv_core.scenarios import remus100_vehicle

from auv_lowlevel.bc import ACTION_SCALES, BCPolicy
from auv_lowlevel.config import load_config
from auv_lowlevel.env import LowLevelEnv
from auv_lowlevel.observation_schema import observation_schema_for_dim


class _NoisyBCPolicy:
    def __init__(self, policy: BCPolicy, noise_std: float, seed: int) -> None:
        self.policy = policy
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation, state, reference):
        command = np.asarray(self.policy(observation, state, reference), dtype=np.float64)
        if self.noise_std > 0.0:
            command = command + self.rng.normal(0.0, self.noise_std, size=3) * ACTION_SCALES
        command[0:2] = np.clip(command[0:2], -ACTION_SCALES[0], ACTION_SCALES[0])
        command[2] = np.clip(command[2], 0.0, ACTION_SCALES[2])
        return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DAgger labels for the I arm")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lowlevel/odarrl_control_i_bc.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=11000)
    parser.add_argument("--episode-id-offset", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    import hashlib

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


def _teacher(config) -> Remus100TrackingController:
    return Remus100TrackingController(
        remus100_vehicle(),
        current_field=None,
        nominal_step_s=config.controller.nominal_step_s,
        heading_gain=config.controller.heading_gain,
        heading_rate_gain=config.controller.heading_rate_gain,
        pitch_gain=config.controller.pitch_gain,
        pitch_rate_gain=config.controller.pitch_rate_gain,
        speed_feedforward_rpm_per_mps=config.controller.speed_feedforward_rpm_per_mps,
        speed_gain_rpm_per_mps=config.controller.speed_gain_rpm_per_mps,
        speed_integral_gain_rpm_per_mps_s=(
            config.controller.speed_integral_gain_rpm_per_mps_s
        ),
        max_speed_integral_rpm=config.controller.max_speed_integral_rpm,
    )


def _collect_episode(payload: tuple[str, str, int, int, int | None, float, str]) -> dict[str, Any]:
    config_path, checkpoint, seed, episode_id, max_steps, noise_std, shard_dir = payload
    from dataclasses import replace

    config = load_config(config_path)
    if max_steps is not None:
        config = replace(config, plant=replace(config.plant, max_episode_steps=max_steps))
    schema = observation_schema_for_dim(config.observation.dim)
    policy = _NoisyBCPolicy(
        BCPolicy(
            checkpoint,
            device="cpu",
            expected_observation_dim=schema.dim,
            expected_observation_schema=schema.version,
            expected_speed_feedback=config.observation.speed_feedback,
            expected_current_contract=config.observation.current_contract,
        ),
        noise_std,
        seed,
    )
    env = LowLevelEnv(
        config,
        base_mode="frozen",
        base_command_fn=policy,
        expose_current=False,
    )
    env.reset(seed)
    teacher = _teacher(config)
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
    max_margin = 0.0
    while True:
        context = env.last_context
        teacher_command = teacher.control(context.state, context.reference)
        if teacher.last_diagnostics is None:
            raise RuntimeError("teacher did not publish diagnostics")
        rows["observation"].append(context.observation)
        rows["action"].append(
            [teacher_command.rudder_angle_rad, teacher_command.stern_plane_angle_rad,
             teacher_command.propeller_rpm]
        )
        rows["step"].append(env.step_count)
        rows["time_s"].append(context.state.t_s)
        rows["actual_actuator"].append(
            [
                context.observation[20] * ACTION_SCALES[0],
                context.observation[21] * ACTION_SCALES[1],
                context.observation[22] * ACTION_SCALES[2],
            ]
        )
        rows["state_position_ned_m"].append(context.state.position_ned_m)
        rows["state_quaternion_xyzw"].append(context.state.quat_xyzw)
        rows["state_linear_velocity_body_mps"].append(context.state.linear_velocity_body_mps)
        rows["state_angular_velocity_body_rps"].append(context.state.angular_velocity_body_rps)
        rows["reference_position_ned_m"].append(context.reference.position_ned_m)
        rows["reference_quaternion_xyzw"].append(context.reference.quat_xyzw)
        rows["current_ned_mps"].append(context.current_ned_mps)
        rows["teacher_post_integral_rpm"].append(
            teacher.last_diagnostics.speed_integral_rpm
        )
        result = env.step()
        margin = abs(
            env.last_context.speed_integral_rpm
            - teacher.last_diagnostics.speed_integral_rpm
        )
        max_margin = max(max_margin, margin)
        if margin > 1e-8:
            raise RuntimeError(f"teacher/shadow integral diverged by {margin} rpm")
        if result.terminated or result.truncated:
            termination = str(result.info["terminal_kind"])
            break

    shard_path = Path(shard_dir) / f"episode_{episode_id:06d}_seed_{seed}.npz"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(rows["observation"])
    np.savez_compressed(
        shard_path,
        observation=np.asarray(rows["observation"], dtype=np.float32),
        action=np.asarray(rows["action"], dtype=np.float32),
        episode_id=np.full(count, episode_id, dtype=np.int64),
        seed=np.full(count, seed, dtype=np.int64),
        step=np.asarray(rows["step"], dtype=np.int64),
        time_s=np.asarray(rows["time_s"], dtype=np.float64),
        actual_actuator=np.asarray(rows["actual_actuator"], dtype=np.float32),
        state_position_ned_m=np.asarray(rows["state_position_ned_m"], dtype=np.float32),
        state_quaternion_xyzw=np.asarray(
            rows["state_quaternion_xyzw"], dtype=np.float32
        ),
        state_linear_velocity_body_mps=np.asarray(
            rows["state_linear_velocity_body_mps"], dtype=np.float32
        ),
        state_angular_velocity_body_rps=np.asarray(
            rows["state_angular_velocity_body_rps"], dtype=np.float32
        ),
        reference_position_ned_m=np.asarray(
            rows["reference_position_ned_m"], dtype=np.float32
        ),
        reference_quaternion_xyzw=np.asarray(
            rows["reference_quaternion_xyzw"], dtype=np.float32
        ),
        current_ned_mps=np.asarray(rows["current_ned_mps"], dtype=np.float32),
        teacher_post_integral_rpm=np.asarray(
            rows["teacher_post_integral_rpm"], dtype=np.float32
        ),
        observation_schema=np.asarray(schema.version),
    )
    return {
        "episode_id": episode_id,
        "seed": seed,
        "samples": count,
        "termination": termination,
        "success": termination == "goal",
        "shard": str(shard_path.resolve()),
        "max_shadow_teacher_margin_rpm": max_margin,
        "constraint_violations": int(env._constraint_violations),
    }


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    schema = observation_schema_for_dim(config.observation.dim)
    if config.arm != "I" or config.current.enabled:
        raise ValueError("DAgger is only defined for the no-current I arm")
    if args.episodes <= 0 or args.workers <= 0:
        raise ValueError("--episodes and --workers must be positive")
    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    output = args.output.resolve()
    shard_dir = output / "shards"
    output.mkdir(parents=True, exist_ok=True)
    payloads = [
        (
            str(args.config.resolve()),
            str(args.checkpoint.resolve()),
            seed,
            args.episode_id_offset + index,
            args.max_steps,
            args.noise_std,
            str(shard_dir),
        )
        for index, seed in enumerate(seeds)
    ]
    metadata: list[dict[str, Any]] = []
    if args.workers == 1:
        metadata = [_collect_episode(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_collect_episode, payload): payload[2]
                for payload in payloads
            }
            for future in as_completed(futures):
                metadata.append(future.result())
                print(f"finished DAgger episode seed {futures[future]}", flush=True)
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
    dagger_path = output / "dagger.npz"
    dagger_arrays = {
        name: np.concatenate(values, axis=0) for name, values in combined.items()
    }
    dagger_arrays["observation_schema"] = np.asarray(schema.version)
    np.savez_compressed(dagger_path, **dagger_arrays)
    manifest = {
        "schema_version": "auv_odarrl_dagger.v1",
        "config_path": str(args.config.resolve()),
        "config_sha256": config.config_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "git_commit": _git_commit(),
        "observation_dim": config.observation.dim,
        "observation_schema": schema.version,
        "episode_count": len(metadata),
        "sample_count": int(sum(int(item["samples"]) for item in metadata)),
        "success_count": sum(bool(item["success"]) for item in metadata),
        "success_rate": float(np.mean([bool(item["success"]) for item in metadata])),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "episode_id_offset": args.episode_id_offset,
        "noise_std": args.noise_std,
        "workers": args.workers,
        "max_steps_override": args.max_steps,
        "dagger_path": str(dagger_path.resolve()),
        "dagger_sha256": _sha256(dagger_path),
        "max_shadow_teacher_margin_rpm": max(
            float(item["max_shadow_teacher_margin_rpm"]) for item in metadata
        ),
        "constraint_violations": sum(
            int(item["constraint_violations"]) for item in metadata
        ),
        "episodes": metadata,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "episode_count", "sample_count", "success_rate",
        "max_shadow_teacher_margin_rpm", "constraint_violations", "dagger_sha256",
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
