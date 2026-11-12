"""Inference-only adapter for PPO actor checkpoints produced by rsl_rl.

The OceanSim evaluation environment is a separate conda environment and must
not receive an rsl_rl installation merely to read an actor checkpoint.  This
module reconstructs the actor MLP from the serialized ``actor.*`` tensors and
keeps the same 16-frame observation history used by the analytic PPO task.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _linear_indices(state: dict[str, torch.Tensor]) -> list[int]:
    indices: list[int] = []
    for key in state:
        if not key.endswith(".weight"):
            continue
        index_text = key.removesuffix(".weight")
        if not index_text.isdigit():
            raise ValueError(f"actor state contains a non-linear weight key: {key!r}")
        indices.append(int(index_text))
    if not indices:
        raise ValueError("actor state does not contain any linear layers")
    if len(indices) != len(set(indices)):
        raise ValueError("actor state contains duplicate layer indices")
    return sorted(indices)


def _build_actor(state: dict[str, torch.Tensor]) -> nn.Sequential:
    indices = _linear_indices(state)
    modules: list[nn.Module] = []
    for position, index in enumerate(indices):
        weight_key = f"{index}.weight"
        bias_key = f"{index}.bias"
        if weight_key not in state or bias_key not in state:
            raise ValueError(f"actor layer {index} is missing weight or bias")
        weight = state[weight_key]
        bias = state[bias_key]
        if weight.ndim != 2 or bias.ndim != 1:
            raise ValueError(f"actor layer {index} has an invalid tensor shape")
        if tuple(weight.shape) != (bias.shape[0], weight.shape[1]):
            raise ValueError(f"actor layer {index} weight and bias do not match")
        modules.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if position < len(indices) - 1:
            modules.append(nn.ELU())
    actor = nn.Sequential(*modules)
    actor.load_state_dict(state, strict=True)
    return actor


class PPOActorPolicy:
    """Deterministic PPO actor with the training-time frame history."""

    def __init__(
        self,
        actor: nn.Sequential,
        *,
        frame_dim: int,
        history_length: int,
        action_scale: tuple[float, float],
        device: str,
        checkpoint_iteration: int | None = None,
    ) -> None:
        if frame_dim <= 0:
            raise ValueError("frame_dim must be positive")
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if len(action_scale) != 2 or not all(
            math.isfinite(float(value)) and float(value) > 0.0 for value in action_scale
        ):
            raise ValueError("action_scale must contain two finite positive values")
        input_dim = frame_dim * history_length
        first = next((module for module in actor if isinstance(module, nn.Linear)), None)
        if first is None or first.in_features != input_dim:
            actual = None if first is None else first.in_features
            raise ValueError(
                f"actor input dimension must be frame_dim * history_length={input_dim}, got {actual}"
            )
        last = next((module for module in reversed(actor) if isinstance(module, nn.Linear)), None)
        if last is None or last.out_features != 2:
            actual = None if last is None else last.out_features
            raise ValueError(f"actor output dimension must be 2, got {actual}")
        self.actor = actor.eval()
        self.frame_dim = int(frame_dim)
        self.history_length = int(history_length)
        self.action_scale = np.asarray(action_scale, dtype=np.float64)
        self._device = torch.device(device)
        self.actor.to(self._device)
        self.checkpoint_iteration = checkpoint_iteration
        self._history: list[np.ndarray] = []

    def begin_episode(self, episode_id: int) -> None:
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        self._history.clear()

    def _stacked_observation(self, observation: np.ndarray) -> torch.Tensor:
        frame = np.asarray(observation, dtype=np.float32)
        if frame.shape != (self.frame_dim,):
            raise ValueError(f"observation must have shape ({self.frame_dim},), got {frame.shape}")
        if not np.all(np.isfinite(frame)):
            raise ValueError("observation must contain only finite values")
        copied = frame.copy()
        if not self._history:
            self._history = [copied] * self.history_length
        else:
            self._history.append(copied)
            if len(self._history) > self.history_length:
                del self._history[0]
        stacked = np.concatenate(self._history, axis=0)
        return torch.as_tensor(stacked, device=self._device).unsqueeze(0)

    def act(self, observation: np.ndarray, *, evaluate: bool = True) -> np.ndarray:
        if evaluate:
            self.actor.eval()
        observation_tensor = self._stacked_observation(observation)
        with torch.inference_mode():
            action = self.actor(observation_tensor).squeeze(0).cpu().numpy()
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise RuntimeError("PPO actor produced an invalid action")
        normalized = np.clip(action, -1.0, 1.0)
        return normalized * self.action_scale

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        frame_dim: int,
        history_length: int,
        action_scale: tuple[float, float],
        device: str = "cpu",
    ) -> PPOActorPolicy:
        checkpoint = Path(path)
        payload: Any = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("PPO checkpoint must be a dictionary")
        model_state = payload.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError("PPO checkpoint does not contain model_state_dict")
        if "std" not in model_state:
            raise ValueError("PPO checkpoint does not contain the actor action std")
        actor_state = {
            str(key).removeprefix("actor."): value
            for key, value in model_state.items()
            if str(key).startswith("actor.")
        }
        if not actor_state:
            raise ValueError("PPO checkpoint does not contain actor weights")
        if not all(isinstance(value, torch.Tensor) for value in actor_state.values()):
            raise ValueError("PPO actor state must contain only tensors")
        actor = _build_actor(actor_state)
        iteration_value = payload.get("iter")
        iteration = int(iteration_value) if iteration_value is not None else None
        return cls(
            actor,
            frame_dim=frame_dim,
            history_length=history_length,
            action_scale=action_scale,
            device=device,
            checkpoint_iteration=iteration,
        )


__all__ = ["PPOActorPolicy"]
