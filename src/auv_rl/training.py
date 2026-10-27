from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from .envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig
from .reward import PaperRewardConfig

EPISODE_FIELDS: tuple[str, ...] = (
    "scenario_key",
    "scenario_name",
    "rl_step",
    "simulation_time_s",
    "success",
    "goal_reached",
    "collision",
    "out_of_bounds",
    "timeout",
    "failure_reason",
    "goal_distance_m",
    "initial_goal_distance_m",
    "min_clearance_m",
    "path_length_m",
    "front_obstacle_range_m",
    "action_clipped",
    "action_clip_count",
    "action_clipped_step_count",
    "action_u_dot_saturation_ratio",
    "action_yaw_rate_saturation_ratio",
    "action_u_dot_max_abs",
    "action_yaw_rate_max_abs",
    "action_u_dot_mean_abs",
    "action_yaw_rate_mean_abs",
    "observation_source",
    "perception_mode",
    "perception_config_hash",
    "sensor_scan_valid",
    "sensor_points_total",
    "sensor_points_used",
    "sensor_empty_rays",
    "sensor_frame_age_s",
    "sensor_scan_returned_false",
    "reward_total",
    "reward_distance",
    "reward_obstacle",
    "reward_terminal",
)


def load_training_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a JSON object")
    return payload


def env_config_from_payload(
    payload: dict[str, Any],
) -> tuple[PlanarEnvConfig, dict[str, Any]]:
    environment = dict(payload.get("environment", {}))
    perception = dict(environment.pop("perception", {}))
    reward = environment.pop("reward", {})
    if not isinstance(reward, dict):
        raise ValueError("environment.reward must be an object")
    keys = environment.get("active_scenario_keys")
    if isinstance(keys, list):
        environment["active_scenario_keys"] = tuple(str(key) for key in keys)
    return PlanarEnvConfig(reward=PaperRewardConfig(**reward), **environment), perception


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_episode_row(
    writer: csv.DictWriter,
    info: dict[str, Any],
    *,
    total_timesteps: int,
) -> None:
    row = {field: info.get(field) for field in EPISODE_FIELDS}
    row["total_timesteps"] = total_timesteps
    writer.writerow(row)


def evaluate_policy(
    model: Any,
    env: PlanarAvoidanceEnv,
    *,
    scenario_keys: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_key in scenario_keys:
        for seed in seeds:
            observation, _ = env.reset(seed=seed, options={"scenario_key": scenario_key})
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(np.asarray(action))
            rows.append(
                {
                    **{field: info.get(field) for field in EPISODE_FIELDS},
                    "scenario_key": scenario_key,
                    "eval_seed": seed,
                }
            )
    return rows


def train_sb3(
    env: PlanarAvoidanceEnv,
    payload: dict[str, Any],
    *,
    output: Path,
    seed: int,
    total_timesteps: int,
    device: str = "auto",
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    training = dict(payload.get("training", {}))

    try:
        import torch  # type: ignore[import-not-found, unused-ignore]
        from stable_baselines3 import DDPG  # type: ignore[import-not-found, unused-ignore]
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import-not-found, unused-ignore]
        from stable_baselines3.common.noise import NormalActionNoise  # type: ignore[import-not-found, unused-ignore]
    except ImportError as error:
        print(
            "SB3 training dependencies are missing. Run: uv sync --extra training",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    class EpisodeMetricsCallback(BaseCallback):
        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path = path
            self._stream: Any = None
            self._writer: csv.DictWriter | None = None

        def _on_training_start(self) -> None:
            self._stream = self.path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(
                self._stream,
                fieldnames=(*EPISODE_FIELDS, "total_timesteps"),
            )
            self._writer.writeheader()

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for info, done in zip(infos, dones, strict=False):
                if bool(done) and isinstance(info, dict) and "scenario_key" in info:
                    if self._writer is None:
                        raise RuntimeError("training callback was not initialized")
                    _write_episode_row(
                        self._writer,
                        info,
                        total_timesteps=self.num_timesteps,
                    )
            return True

        def _on_training_end(self) -> None:
            if self._stream is not None:
                self._stream.close()

    net_arch = [int(width) for width in training.get("net_arch", [160, 154])]
    noise_sigma = np.asarray(training.get("noise_sigma", [0.05, 0.02]), dtype=np.float32)
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=noise_sigma)
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=float(training.get("learning_rate", 0.0002)),
        gamma=float(training.get("gamma", 0.98)),
        batch_size=int(training.get("batch_size", 64)),
        buffer_size=int(training.get("buffer_size", 100000)),
        learning_starts=int(training.get("learning_starts", 1000)),
        policy_kwargs={"net_arch": net_arch},
        action_noise=action_noise,
        tensorboard_log=None,
        seed=seed,
        device=device,
        verbose=1,
    )
    callback = EpisodeMetricsCallback(output / "train_episodes.csv")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model_path = output / "model"
    model.save(model_path)
    model_path = model_path.with_suffix(".zip")

    eval_rows = evaluate_policy(
        model,
        env,
        scenario_keys=env.config.active_scenario_keys,
        seeds=tuple(int(item) for item in training.get("eval_seeds", [0, 1, 2])),
    )
    with (output / "eval_episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(*EPISODE_FIELDS, "eval_seed"),
        )
        writer.writeheader()
        writer.writerows(eval_rows)

    perception = getattr(env, "perception", None)
    action_space = env.action_space
    if not isinstance(action_space, Box):
        raise TypeError("P1 environment must expose a Box action space")
    manifest = {
        "config": payload,
        "environment": asdict(env.config),
        "observation_source": getattr(perception, "observation_source", None),
        "perception_mode": getattr(perception, "perception_mode", None),
        "perception_config_hash": getattr(perception, "config_hash", None),
        "total_timesteps": total_timesteps,
        "seed": seed,
        "model_path": str(model_path),
        "git_commit": _git_commit(),
        "python": sys.version,
        "torch": torch.__version__,
        "stable_baselines3": __import__("stable_baselines3").__version__,
        "gymnasium": __import__("gymnasium").__version__,
        "observation_spec": {
            "name": env.builder.spec.name,
            "version": env.builder.spec.version,
            "shape": list(env.builder.spec.shape),
            "normalization": dict(env.builder.spec.normalization),
            "critic_shape": list(env.builder.critic_spec.shape),
        },
        "action_spec": {
            "low": [float(item) for item in action_space.low],
            "high": [float(item) for item in action_space.high],
        },
        "evaluation": eval_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    successes = sum(bool(row["success"]) for row in eval_rows)
    print(f"trained model: {model_path}")
    print(f"evaluation success: {successes}/{len(eval_rows)}")
    return 0


__all__ = [
    "EPISODE_FIELDS",
    "env_config_from_payload",
    "evaluate_policy",
    "load_training_payload",
    "train_sb3",
]
