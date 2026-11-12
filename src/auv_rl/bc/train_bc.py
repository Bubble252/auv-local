from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found, unused-ignore]
from torch import nn  # type: ignore[import-not-found, unused-ignore]

from ..ddpg.agent import DDPGConfig
from ..ddpg.networks import PolicyActor
from ..envs.planar_avoidance import PlanarAvoidanceEnv
from ..training import EPISODE_FIELDS
from .policy import BCActorPolicy


@dataclass(frozen=True)
class BCSequenceDataset:
    train_observations: np.ndarray
    train_actions: np.ndarray
    validation_observations: np.ndarray
    validation_actions: np.ndarray
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]


@dataclass(frozen=True)
class BCTrainingConfig:
    batch_size: int = 256
    gradient_steps: int = 10000
    learning_rate: float = 2e-4
    validation_fraction: float = 0.1
    evaluation_interval: int = 500
    patience: int = 8
    action_loss_weighting: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.gradient_steps <= 0:
            raise ValueError("batch_size and gradient_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.evaluation_interval <= 0 or self.patience <= 0:
            raise ValueError("evaluation_interval and patience must be positive")
        if not np.isfinite(self.action_loss_weighting) or self.action_loss_weighting < 0.0:
            raise ValueError("action_loss_weighting must be finite and non-negative")


def _history_sequence(
    observations: np.ndarray,
    index: int,
    history_length: int,
) -> np.ndarray:
    start = max(0, index - history_length + 1)
    history = observations[start : index + 1]
    if history.shape[0] < history_length:
        padding = np.repeat(history[:1], history_length - history.shape[0], axis=0)
        history = np.concatenate((padding, history), axis=0)
    return history


def build_sequence_dataset(
    demo_path: Path | str,
    *,
    history_length: int,
    validation_fraction: float = 0.1,
) -> BCSequenceDataset:
    payload = np.load(Path(demo_path), allow_pickle=False)
    required = {
        "observations",
        "actions",
        "episode_ids",
        "episode_steps",
        "scenario_keys",
        "teacher_success",
    }
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"demo dataset is missing fields: {sorted(missing)}")
    observations = np.asarray(payload["observations"], dtype=np.float32)
    actions = np.asarray(payload["actions"], dtype=np.float32)
    episode_ids = np.asarray(payload["episode_ids"], dtype=np.int64)
    episode_steps = np.asarray(payload["episode_steps"], dtype=np.int64)
    scenario_keys = np.asarray(payload["scenario_keys"]).astype(str)
    teacher_success = np.asarray(payload["teacher_success"], dtype=np.bool_)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be two-dimensional")
    if not (
        len(observations)
        == len(actions)
        == len(episode_ids)
        == len(episode_steps)
        == len(scenario_keys)
    ):
        raise ValueError("demo arrays must have equal length")
    if not np.all(teacher_success):
        raise ValueError("BC main dataset must contain successful teacher transitions only")

    unique_episodes = tuple(int(item) for item in np.unique(episode_ids))
    if not unique_episodes:
        raise ValueError("demo dataset contains no episodes")
    validation_episodes: list[int] = []
    for scenario_key in sorted(set(scenario_keys.tolist())):
        scenario_episodes = tuple(
            int(item) for item in np.unique(episode_ids[scenario_keys == scenario_key])
        )
        if len(scenario_episodes) < 2:
            raise ValueError(
                f"scenario {scenario_key!r} needs at least two episodes for a train/validation split"
            )
        validation_count = max(
            1,
            int(round(len(scenario_episodes) * validation_fraction)),
        )
        validation_count = min(validation_count, len(scenario_episodes) - 1)
        validation_episodes.extend(scenario_episodes[-validation_count:])
    validation_set = set(validation_episodes)
    train_observations: list[np.ndarray] = []
    train_actions: list[np.ndarray] = []
    validation_observations: list[np.ndarray] = []
    validation_actions: list[np.ndarray] = []
    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(episode_steps[indices], kind="stable")]
        episode_observations = observations[indices]
        episode_actions = actions[indices]
        for local_index in range(len(indices)):
            sequence = _history_sequence(
                episode_observations,
                local_index,
                history_length,
            )
            if episode_id in validation_set:
                validation_observations.append(sequence)
                validation_actions.append(episode_actions[local_index])
            else:
                train_observations.append(sequence)
                train_actions.append(episode_actions[local_index])
    if not train_observations or not validation_observations:
        raise ValueError("both BC train and validation splits must be non-empty")
    return BCSequenceDataset(
        train_observations=np.stack(train_observations, axis=0),
        train_actions=np.stack(train_actions, axis=0),
        validation_observations=np.stack(validation_observations, axis=0),
        validation_actions=np.stack(validation_actions, axis=0),
        train_episodes=tuple(item for item in unique_episodes if item not in validation_set),
        validation_episodes=tuple(validation_episodes),
    )


def _actor_from_config(config: DDPGConfig) -> PolicyActor:
    return PolicyActor(
        config.observation_dim,
        config.action_low,
        config.action_high,
        hidden_sizes=config.hidden_sizes,
        use_gru=config.use_gru,
        gru_hidden_size=config.gru_hidden_size,
        use_noisy=config.use_noisy,
        activation=config.resolved_actor_activation,
        layer_norm=config.layer_norm,
    ).to(torch.device(config.device))


def action_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    action_high: torch.Tensor,
    weighting: float,
) -> torch.Tensor:
    squared_error = (prediction - target).square().mean(dim=1)
    if weighting == 0.0:
        return squared_error.mean()
    normalized_magnitude = (target.abs() / action_high.reshape(1, -1)).mean(dim=1)
    weights = 1.0 + weighting * normalized_magnitude
    return (squared_error * weights).mean()


def train_bc_actor(
    config: DDPGConfig,
    dataset: BCSequenceDataset,
    training: BCTrainingConfig,
    *,
    output_dir: Path | str,
    observation_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(training.seed)
    np.random.seed(training.seed)
    device = torch.device(config.device)
    actor = _actor_from_config(config)
    optimizer = torch.optim.Adam(actor.parameters(), lr=training.learning_rate)
    train_obs = torch.as_tensor(dataset.train_observations, device=device)
    train_actions = torch.as_tensor(dataset.train_actions, device=device)
    action_high = torch.as_tensor(config.action_high, device=device, dtype=train_actions.dtype)
    validation_obs = torch.as_tensor(dataset.validation_observations, device=device)
    validation_actions = torch.as_tensor(dataset.validation_actions, device=device)
    rng = np.random.default_rng(training.seed)
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_evaluations = 0
    history: list[dict[str, float | int]] = []

    for step in range(1, training.gradient_steps + 1):
        actor.train()
        indices = rng.integers(0, len(train_obs), size=min(training.batch_size, len(train_obs)))
        sequence = train_obs[indices]
        target = train_actions[indices]
        prediction, _ = actor(sequence, sample_noise=False)
        loss = action_mse_loss(
            prediction[:, -1],
            target,
            action_high=action_high,
            weighting=training.action_loss_weighting,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        if step % training.evaluation_interval != 0 and step != training.gradient_steps:
            continue
        actor.eval()
        with torch.no_grad():
            validation_prediction, _ = actor(validation_obs, sample_noise=False)
            validation_loss = float(
                nn.functional.mse_loss(validation_prediction[:, -1], validation_actions).cpu()
            )
        history.append(
            {
                "gradient_step": step,
                "train_loss": float(loss.detach().cpu()),
                "validation_loss": validation_loss,
                "action_loss_weighting": training.action_loss_weighting,
            }
        )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_step = step
            best_state = {
                key: value.detach().cpu().clone() for key, value in actor.state_dict().items()
            }
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        if stale_evaluations >= training.patience:
            break
    if best_state is None:
        raise RuntimeError("BC training did not produce a finite validation checkpoint")
    actor.load_state_dict(best_state)
    checkpoint = output / "actor.pt"
    torch.save(
        {
            "format": "auv_rl.bc_actor.v1",
            "config": asdict(config),
            "seed": training.seed,
            "actor": best_state,
            "best_validation_loss": best_loss,
            "best_gradient_step": best_step,
            "teacher_used": True,
            "bc_pretrained": True,
            "oracle_in_actor": False,
            "observation_spec": observation_spec,
        },
        checkpoint,
    )
    result = {
        "schema": "auv_rl.bc_training.v1",
        "checkpoint": str(checkpoint),
        "best_validation_loss": best_loss,
        "best_gradient_step": best_step,
        "train_samples": int(len(dataset.train_observations)),
        "validation_samples": int(len(dataset.validation_observations)),
        "train_episodes": list(dataset.train_episodes),
        "validation_episodes": list(dataset.validation_episodes),
        "history": history,
        "teacher_used": True,
        "bc_pretrained": True,
        "oracle_in_actor": False,
        "observation_spec": observation_spec,
    }
    (output / "bc_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def evaluate_bc_policy(
    env: PlanarAvoidanceEnv,
    policy: BCActorPolicy,
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
            policy.begin_episode(episode_id)
            episode_id += 1
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            while not terminated and not truncated:
                action = policy.act(observation, evaluate=True)
                observation, _, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    **{field: info.get(field) for field in EPISODE_FIELDS},
                    "scenario_key": scenario_key,
                    "eval_seed": seed,
                    "reset_sensor_points_total": reset_info.get("sensor_points_total"),
                }
            )
    return rows


__all__ = [
    "BCSequenceDataset",
    "BCTrainingConfig",
    "action_mse_loss",
    "build_sequence_dataset",
    "evaluate_bc_policy",
    "train_bc_actor",
]
