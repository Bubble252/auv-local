from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from ..envs.planar_avoidance import PlanarAvoidanceEnv
from ..privileged import privileged_critic_features
from ..training import EPISODE_FIELDS
from .agent import DDPGAgent, DDPGConfig
from .diagnostics import diagnose_critic, write_critic_health

SATURATION_TIMELINE_FIELDS = (
    "total_timesteps",
    "gradient_step",
    "episode_id",
    "actor_raw_abs_max",
    "tanh_grad_norm",
    "action_u_dot_saturation_ratio",
    "action_yaw_rate_saturation_ratio",
    "action_u_dot_max_abs",
    "action_u_dot_mean_abs",
    "action_yaw_rate_max_abs",
    "action_yaw_rate_mean_abs",
    "action_clip_count",
    "action_clipped_step_count",
    "q_range_mean",
    "q_std_mean",
    "dq_da_mean",
    "activation_active_fraction",
    "first_layer_dead_fraction",
    "critic_loss",
    "actor_loss",
    "mean_q",
    "mean_td_error",
)


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

    action_low_array = np.asarray(action_space.low, dtype=np.float64)
    action_high_array = np.asarray(action_space.high, dtype=np.float64)
    if action_low_array.ndim > 1:
        action_low_array = action_low_array[0]
    if action_high_array.ndim > 1:
        action_high_array = action_high_array[0]
    action_low = tuple(float(item) for item in action_low_array)
    action_high = tuple(float(item) for item in action_high_array)
    observation_shape = observation_space.shape
    if observation_shape is None or len(observation_shape) != 1:
        if observation_shape is None or len(observation_shape) != 2:
            raise ValueError("P1 observation space must be one-dimensional")
        observation_dim = int(observation_shape[-1])
    else:
        observation_dim = int(observation_shape[0])
    return DDPGConfig(
        observation_dim=observation_dim,
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
        activation=str(training.get("activation", "relu")),
        actor_activation=(
            str(training["actor_activation"])
            if training.get("actor_activation") is not None
            else None
        ),
        critic_activation=(
            str(training["critic_activation"])
            if training.get("critic_activation") is not None
            else None
        ),
        layer_norm=bool(training.get("layer_norm", False)),
        actor_output_noise=bool(training.get("actor_output_noise", True)),
        actor_output_sigma_init=float(training.get("actor_output_sigma_init", 0.5)),
        actor_hidden_sigma_init=float(training.get("actor_hidden_sigma_init", 0.5)),
        actor_noise_sigma_trainable=bool(training.get("actor_noise_sigma_trainable", True)),
        actor_output_weight_init_bound=(
            None
            if training.get("actor_output_weight_init_bound") is None
            else float(training["actor_output_weight_init_bound"])
        ),
        actor_output_scale=float(training.get("actor_output_scale", 1.0)),
        actor_output_squash=str(training.get("actor_output_squash", "tanh")),
        policy_delay=int(training.get("policy_delay", 1)),
        actor_warmup_updates=int(training.get("actor_warmup_updates", 0)),
        twin_critics=bool(training.get("twin_critics", False)),
        target_policy_noise_sigma=float(training.get("target_policy_noise_sigma", 0.0)),
        target_policy_noise_clip=float(training.get("target_policy_noise_clip", 0.5)),
        saturation_penalty=float(training.get("saturation_penalty", 0.0)),
        saturation_threshold_fraction=float(training.get("saturation_threshold_fraction", 0.9)),
        privileged_dim=int(training.get("privileged_dim", 0)),
        bc_auxiliary_weight=float(training.get("bc_auxiliary_weight", 0.0)),
        bc_auxiliary_decay_updates=int(training.get("bc_auxiliary_decay_updates", 0)),
        use_pper=bool(training.get("use_pper", True)),
        per_alpha=float(training.get("per_alpha", 0.6)),
        per_beta_start=float(training.get("per_beta_start", 0.4)),
        per_beta_steps=int(training.get("per_beta_steps", 100000)),
        per_epsilon=float(training.get("per_epsilon", 1e-6)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        cuda_graph_update=bool(training.get("cuda_graph_update", False)),
        device=device,
    )


def ddpg_config_from_payload(
    env: Any,
    payload: dict[str, Any],
    *,
    device: str,
) -> DDPGConfig:
    """Build a DDPG config from a JSON payload and an env-like object."""

    return _ddpg_config_from_payload(env, payload, device=device)


def _rows_from_info(info: dict[str, Any]) -> dict[str, Any]:
    return {field: info.get(field) for field in EPISODE_FIELDS}


def _scenario_keys_for_timestep(
    training: dict[str, Any],
    *,
    timestep: int,
    default_keys: tuple[str, ...],
) -> tuple[str, ...]:
    curriculum = training.get("curriculum")
    if curriculum is None:
        return default_keys
    if not isinstance(curriculum, dict):
        raise ValueError("training.curriculum must be an object")
    stage1_keys = tuple(str(key) for key in curriculum.get("stage1_scenarios", ()))
    stage2_keys = tuple(str(key) for key in curriculum.get("stage2_scenarios", default_keys))
    stage1_steps = int(curriculum.get("stage1_steps", 0))
    if not stage1_keys or not stage2_keys:
        raise ValueError("curriculum stages must contain at least one scenario")
    if stage1_steps < 0:
        raise ValueError("curriculum.stage1_steps must be non-negative")
    return stage1_keys if timestep < stage1_steps else stage2_keys


def _write_saturation_row(
    writer: csv.DictWriter,
    *,
    agent: DDPGAgent,
    observations: deque[np.ndarray],
    privileged_observations: deque[np.ndarray] | None,
    actions: list[np.ndarray],
    episode_info: dict[str, Any] | None,
    total_timesteps: int,
    gradient_step: int,
    episode_id: int,
    metrics: dict[str, float] | None,
) -> None:
    if len(observations) < 2:
        return
    health = diagnose_critic(
        agent,
        np.stack(observations, axis=0),
        (None if privileged_observations is None else np.stack(privileged_observations, axis=0)),
    )
    action_array = (
        np.asarray(actions, dtype=np.float64)
        if actions
        else np.zeros((1, agent.config.action_dim), dtype=np.float64)
    )
    high = np.asarray(agent.config.action_high, dtype=np.float64)
    saturation = np.mean(
        np.abs(action_array) >= (high.reshape(1, -1) - 1e-9),
        axis=0,
    )
    action_abs = np.abs(action_array)
    row: dict[str, Any] = {
        "total_timesteps": total_timesteps,
        "gradient_step": gradient_step,
        "episode_id": episode_id,
        "actor_raw_abs_max": health["actor_raw_abs_max"],
        "tanh_grad_norm": health["tanh_grad_norm"],
        "action_u_dot_saturation_ratio": float(saturation[0]),
        "action_yaw_rate_saturation_ratio": float(saturation[1]),
        "action_u_dot_max_abs": float(action_abs[:, 0].max()),
        "action_u_dot_mean_abs": float(action_abs[:, 0].mean()),
        "action_yaw_rate_max_abs": float(action_abs[:, 1].max()),
        "action_yaw_rate_mean_abs": float(action_abs[:, 1].mean()),
        "action_clip_count": (
            int(episode_info.get("action_clip_count", 0)) if episode_info is not None else 0
        ),
        "action_clipped_step_count": (
            int(episode_info.get("action_clipped_step_count", 0)) if episode_info is not None else 0
        ),
        "q_range_mean": health["q_range_mean"],
        "q_std_mean": health["q_std_mean"],
        "dq_da_mean": health["dq_da_mean"],
        "activation_active_fraction": health["activation_active_fraction"],
        "first_layer_dead_fraction": health["first_layer_dead_fraction"],
    }
    if metrics is not None:
        for key in ("critic_loss", "actor_loss", "mean_q", "mean_td_error"):
            row[key] = metrics.get(key)
    writer.writerow(row)


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
            observation, reset_info = env.reset(
                seed=seed,
                options={"scenario_key": scenario_key, "jitter": False},
            )
            agent.begin_episode(episode_id)
            episode_id += 1
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            while not terminated and not truncated:
                action = agent.act(observation, evaluate=True)
                observation, _, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    **_rows_from_info(info),
                    "scenario_key": scenario_key,
                    "eval_seed": seed,
                    "reset_sensor_points_total": reset_info.get("sensor_points_total"),
                }
            )
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
    agent.set_observation_spec(
        {
            "name": env.builder.spec.name,
            "version": env.builder.spec.version,
            "shape": list(env.builder.spec.shape),
        }
    )
    training = dict(payload.get("training", {}))
    bc_payload = training.get("bc")
    actor_checkpoint: str | None = None
    demos_path: str | None = None
    if bc_payload is not None:
        if not isinstance(bc_payload, dict):
            raise ValueError("training.bc must be an object")
        checkpoint_value = bc_payload.get("actor_checkpoint")
        if checkpoint_value is not None:
            actor_checkpoint = str(checkpoint_value)
        demo_value = bc_payload.get("demos")
        if demo_value is not None:
            demos_path = str(demo_value)
        from ..bc.initialization import initialize_agent_from_bc

        bc_report = initialize_agent_from_bc(
            agent,
            actor_checkpoint=actor_checkpoint,
            demos=demos_path,
            history_length=config.history_length,
            expected_observation_version=env.builder.spec.version,
        )
    else:
        from ..bc.initialization import BCInitializationReport

        bc_report = BCInitializationReport(
            actor_checkpoint=None,
            actor_checkpoint_sha256=None,
            demos=None,
            demo_sha256=None,
            replay_transitions=0,
            auxiliary_samples=0,
        )
    scenario_keys = tuple(env.config.active_scenario_keys)
    total_timesteps = int(total_timesteps)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    train_path = output / "train_episodes.csv"
    saturation_path = output / "saturation_timeline.csv"
    gradient_step = 0
    timestep = 0
    episode_id = 0
    last_metrics: dict[str, float] | None = None
    recent_observations: deque[np.ndarray] = deque(maxlen=64)
    recent_privileged: deque[np.ndarray] | None = (
        deque(maxlen=64) if config.privileged_dim > 0 else None
    )
    recent_actions: list[np.ndarray] = []
    next_diagnostic_timestep = 1000
    with (
        train_path.open("w", encoding="utf-8", newline="") as stream,
        saturation_path.open("w", encoding="utf-8", newline="") as saturation_stream,
    ):
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
        stream.flush()
        saturation_writer = csv.DictWriter(
            saturation_stream,
            fieldnames=SATURATION_TIMELINE_FIELDS,
        )
        saturation_writer.writeheader()
        saturation_stream.flush()
        while timestep < total_timesteps:
            active_keys = _scenario_keys_for_timestep(
                training,
                timestep=timestep,
                default_keys=scenario_keys,
            )
            scenario_key = active_keys[episode_id % len(active_keys)]
            observation, _ = env.reset(
                seed=seed + episode_id,
                options={"scenario_key": scenario_key},
            )
            recent_observations.append(np.asarray(observation, dtype=np.float32).copy())
            if recent_privileged is not None:
                recent_privileged.append(privileged_critic_features(env.privileged_scene_state()))
            agent.begin_episode(episode_id)
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            while not terminated and not truncated and timestep < total_timesteps:
                privileged = (
                    privileged_critic_features(env.privileged_scene_state())
                    if config.privileged_dim > 0
                    else None
                )
                action = agent.act(observation, evaluate=False)
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_privileged = (
                    privileged_critic_features(env.privileged_scene_state())
                    if config.privileged_dim > 0
                    else None
                )
                agent.observe(
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                    privileged=privileged,
                    next_privileged=next_privileged,
                )
                recent_observations.append(np.asarray(next_observation, dtype=np.float32).copy())
                if recent_privileged is not None and next_privileged is not None:
                    recent_privileged.append(next_privileged.copy())
                recent_actions.append(np.asarray(action, dtype=np.float64).copy())
                observation = next_observation
                timestep += 1
                if len(agent.replay) >= config.learning_starts:
                    metrics = agent.update(gradient_step)
                    gradient_step += 1
                    if metrics is not None:
                        last_metrics = asdict(metrics)
                if timestep >= next_diagnostic_timestep:
                    _write_saturation_row(
                        saturation_writer,
                        agent=agent,
                        observations=recent_observations,
                        privileged_observations=recent_privileged,
                        actions=recent_actions,
                        episode_info=info,
                        total_timesteps=timestep,
                        gradient_step=gradient_step,
                        episode_id=episode_id,
                        metrics=last_metrics,
                    )
                    saturation_stream.flush()
                    recent_actions = []
                    next_diagnostic_timestep += 1000
            _write_episode_row(
                writer,
                info,
                total_timesteps=timestep,
                episode_id=episode_id,
                gradient_step=gradient_step,
                update_metrics=last_metrics,
            )
            stream.flush()
            print(
                f"episode={episode_id} timestep={timestep} scenario={scenario_key} "
                f"success={info.get('success')} reason={info.get('failure_reason')}",
                flush=True,
            )
            episode_id += 1

        for scenario_key in scenario_keys:
            reset_observation, _ = env.reset(
                seed=seed + 100000 + episode_id,
                options={"scenario_key": scenario_key, "jitter": False},
            )
            recent_observations.append(np.asarray(reset_observation, dtype=np.float32).copy())
            if recent_privileged is not None:
                recent_privileged.append(privileged_critic_features(env.privileged_scene_state()))
            episode_id += 1
        final_health = diagnose_critic(
            agent,
            np.stack(recent_observations, axis=0),
            (None if recent_privileged is None else np.stack(recent_privileged, axis=0)),
        )
        if timestep != 0:
            _write_saturation_row(
                saturation_writer,
                agent=agent,
                observations=recent_observations,
                privileged_observations=recent_privileged,
                actions=recent_actions,
                episode_info=info,
                total_timesteps=timestep,
                gradient_step=gradient_step,
                episode_id=episode_id,
                metrics=last_metrics,
            )
            saturation_stream.flush()

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
    write_critic_health(
        output / "critic_health.json",
        final_health,
        eval_rows=eval_rows,
    )

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
        "bc": {
            **bc_report.as_manifest(),
            "privileged_critic": config.privileged_dim > 0,
        },
        "evaluation": eval_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    successes = sum(bool(row["success"]) for row in eval_rows)
    print(f"trained model: {model_path}", flush=True)
    print(f"evaluation success: {successes}/{len(eval_rows)}", flush=True)
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


__all__ = ["ddpg_config_from_payload", "evaluate_ddpg", "train_ddpg"]
