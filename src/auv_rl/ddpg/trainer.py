from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gymnasium.spaces import Box

from ..envs.planar_avoidance import PlanarAvoidanceEnv
from ..training import EPISODE_FIELDS
from .agent import DDPGAgent, DDPGConfig


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


def _ddpg_config_from_payload(
    env: PlanarAvoidanceEnv,
    payload: dict[str, Any],
    *,
    device: str,
) -> DDPGConfig:
    training = dict(payload.get("training", {}))
    action_space = env.action_space
    observation_space = env.observation_space
    if not isinstance(action_space, Box) or not isinstance(observation_space, Box):
        raise TypeError("P1 environment must expose Box observation and action spaces")

    def training_float(key: str, default: float) -> float:
        value = training.get(key, default)
        return float(default if value is None else value)

    action_low = tuple(float(item) for item in action_space.low)
    action_high = tuple(float(item) for item in action_space.high)
    observation_shape = observation_space.shape
    if observation_shape is None or len(observation_shape) != 1:
        raise ValueError("P1 observation space must be one-dimensional")
    return DDPGConfig(
        observation_dim=int(observation_shape[0]),
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=tuple(int(item) for item in training.get("net_arch", (160, 154))),
        gru_hidden_size=int(training.get("gru_hidden_size", 64)),
        history_length=int(training.get("history_length", 10)),
        gamma=float(training.get("gamma", 0.98)),
        tau=float(training.get("tau", 0.005)),
        actor_lr=training_float("actor_lr", training_float("learning_rate", 2e-4)),
        critic_lr=training_float("critic_lr", training_float("learning_rate", 2e-4)),
        batch_size=int(training.get("batch_size", 64)),
        buffer_size=int(training.get("buffer_size", 100000)),
        learning_starts=int(training.get("learning_starts", 1000)),
        use_noisy=bool(training.get("use_noisy", True)),
        use_gru=bool(training.get("use_gru", True)),
        use_pper=bool(training.get("use_pper", True)),
        per_alpha=float(training.get("per_alpha", 0.6)),
        per_beta_start=float(training.get("per_beta_start", 0.4)),
        per_beta_steps=int(training.get("per_beta_steps", 100000)),
        per_epsilon=float(training.get("per_epsilon", 1e-6)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        device=device,
    )


def _rows_from_info(info: dict[str, Any]) -> dict[str, Any]:
    return {field: info.get(field) for field in EPISODE_FIELDS}


def _write_episode_row(
    writer: csv.DictWriter,
    info: dict[str, Any],
    *,
    total_timesteps: int,
    episode_id: int,
    gradient_step: int,
    update_metrics: dict[str, float] | None,
) -> None:
    row = _rows_from_info(info)
    row["total_timesteps"] = total_timesteps
    row["episode_id"] = episode_id
    row["gradient_step"] = gradient_step
    if update_metrics:
        row.update(update_metrics)
    writer.writerow(row)


def _evaluate(
    agent: DDPGAgent,
    env: PlanarAvoidanceEnv,
    *,
    scenario_keys: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_id = 0
    for scenario_key in scenario_keys:
        for seed in seeds:
            observation, _ = env.reset(
                seed=seed,
                options={"scenario_key": scenario_key},
            )
            agent.begin_episode(episode_id)
            episode_id += 1
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            while not terminated and not truncated:
                action = agent.act(observation, evaluate=True)
                observation, _, terminated, truncated, info = env.step(action)
            rows.append({**_rows_from_info(info), "scenario_key": scenario_key, "eval_seed": seed})
    return rows


def train_ddpg(
    env: PlanarAvoidanceEnv,
    payload: dict[str, Any],
    *,
    output: Path,
    seed: int,
    total_timesteps: int,
    device: str = "cpu",
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config = _ddpg_config_from_payload(env, payload, device=device)
    agent = DDPGAgent(config, seed=seed)
    training = dict(payload.get("training", {}))
    scenario_keys = tuple(env.config.active_scenario_keys)
    total_timesteps = int(total_timesteps)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    train_path = output / "train_episodes.csv"
    gradient_step = 0
    timestep = 0
    episode_id = 0
    last_metrics: dict[str, float] | None = None
    with train_path.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            *EPISODE_FIELDS,
            "total_timesteps",
            "episode_id",
            "gradient_step",
            "critic_loss",
            "actor_loss",
            "mean_q",
            "mean_td_error",
            "beta",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while timestep < total_timesteps:
            scenario_key = scenario_keys[episode_id % len(scenario_keys)]
            observation, _ = env.reset(
                seed=seed + episode_id,
                options={"scenario_key": scenario_key},
            )
            agent.begin_episode(episode_id)
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            while not terminated and not truncated and timestep < total_timesteps:
                action = agent.act(observation, evaluate=False)
                next_observation, reward, terminated, truncated, info = env.step(action)
                agent.observe(
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                )
                observation = next_observation
                timestep += 1
                if len(agent.replay) >= config.learning_starts:
                    metrics = agent.update(gradient_step)
                    gradient_step += 1
                    if metrics is not None:
                        last_metrics = asdict(metrics)
            _write_episode_row(
                writer,
                info,
                total_timesteps=timestep,
                episode_id=episode_id,
                gradient_step=gradient_step,
                update_metrics=last_metrics,
            )
            print(
                f"episode={episode_id} timestep={timestep} scenario={scenario_key} "
                f"success={info.get('success')} reason={info.get('failure_reason')}"
            )
            episode_id += 1

    model_path = output / "model.pt"
    agent.save(model_path)
    eval_rows = _evaluate(
        agent,
        env,
        scenario_keys=scenario_keys,
        seeds=tuple(int(item) for item in training.get("eval_seeds", (0, 1, 2))),
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
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]

        torch_version = torch.__version__
    except ImportError:  # pragma: no cover - only on intentionally minimal installs
        torch_version = None
    manifest = {
        "algorithm": "auv_rl.ddpg",
        "config": payload,
        "ddpg_config": asdict(config),
        "environment": asdict(env.config),
        "observation_source": getattr(perception, "observation_source", None),
        "perception_mode": getattr(perception, "perception_mode", None),
        "perception_config_hash": getattr(perception, "config_hash", None),
        "total_timesteps": total_timesteps,
        "seed": seed,
        "model_path": str(model_path),
        "python": sys.version,
        "torch": torch_version,
        "git_commit": _git_commit(),
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


def evaluate_ddpg(
    checkpoint: Path | str,
    env: PlanarAvoidanceEnv,
    *,
    scenario_keys: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    agent = DDPGAgent.load(checkpoint)
    return _evaluate(agent, env, scenario_keys=scenario_keys, seeds=seeds)


__all__ = ["evaluate_ddpg", "train_ddpg"]
