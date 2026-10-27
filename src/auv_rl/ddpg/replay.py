from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SumTree:
    """Binary sum tree over leaf priorities."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        size = 1
        while size < capacity:
            size *= 2
        self.capacity = size
        self.tree = np.zeros(2 * size - 1, dtype=np.float64)
        self.n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        if self.n_entries == 0:
            return 1.0
        return float(np.max(self.tree[-self.capacity :]))

    def update(self, index: int, priority: float) -> None:
        if not 0 <= index < self.capacity:
            raise IndexError("leaf index out of range")
        value = max(float(priority), 0.0)
        position = index + self.capacity - 1
        change = value - float(self.tree[position])
        self.tree[position] = value
        while position > 0:
            position = (position - 1) // 2
            self.tree[position] += change
        self.n_entries = max(self.n_entries, index + 1)

    def get(self, prefix: float) -> tuple[int, float]:
        if self.n_entries == 0 or self.total <= 0.0:
            raise ValueError("cannot sample from an empty SumTree")
        target = min(max(float(prefix), 0.0), self.total - 1e-12)
        position = 0
        while position < self.capacity - 1:
            left = 2 * position + 1
            if target <= float(self.tree[left]):
                position = left
            else:
                target -= float(self.tree[left])
                position = left + 1
        leaf = position - self.capacity + 1
        return leaf, float(self.tree[position])


@dataclass(frozen=True)
class ReplayBatch:
    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    rewards: NDArray[np.float32]
    next_observations: NDArray[np.float32]
    dones: NDArray[np.float32]
    valid_masks: NDArray[np.float32]
    weights: NDArray[np.float32]
    indices: NDArray[np.int64]
    episode_ids: NDArray[np.int64]

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.observations.shape[1])


@dataclass
class _Transition:
    observation: NDArray[np.float32]
    action: NDArray[np.float32]
    reward: float
    next_observation: NDArray[np.float32]
    done: bool
    episode_id: int
    episode_step: int


class PrioritizedSequenceReplayBuffer:
    """Replay buffer that samples contiguous, episode-local sequences.

    Priorities belong to the sampled anchor transition. Earlier steps are used
    only as GRU context and never borrowed from a different episode.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        *,
        history_length: int = 10,
        use_pper: bool = True,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.history_length = int(history_length)
        self.use_pper = bool(use_pper)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)
        self._tree = SumTree(self.capacity)
        self._transitions: list[_Transition | None] = [None] * self.capacity
        self._episode_index: dict[tuple[int, int], int] = {}
        self._write_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: NDArray[np.float32] | np.ndarray,
        action: NDArray[np.float32] | np.ndarray,
        reward: float,
        next_observation: NDArray[np.float32] | np.ndarray,
        done: bool,
        *,
        episode_id: int,
        episode_step: int,
    ) -> None:
        obs = np.asarray(observation, dtype=np.float32)
        act = np.asarray(action, dtype=np.float32)
        next_obs = np.asarray(next_observation, dtype=np.float32)
        if obs.shape != (self.observation_dim,):
            raise ValueError("observation shape does not match replay buffer")
        if act.shape != (self.action_dim,):
            raise ValueError("action shape does not match replay buffer")
        if next_obs.shape != (self.observation_dim,):
            raise ValueError("next_observation shape does not match replay buffer")
        if episode_id < 0 or episode_step < 0:
            raise ValueError("episode_id and episode_step must be non-negative")
        if self._size == self.capacity:
            old = self._transitions[self._write_index]
            if old is not None:
                self._episode_index.pop((old.episode_id, old.episode_step), None)
        transition = _Transition(
            observation=obs.copy(),
            action=act.copy(),
            reward=float(reward),
            next_observation=next_obs.copy(),
            done=bool(done),
            episode_id=int(episode_id),
            episode_step=int(episode_step),
        )
        index = self._write_index
        self._transitions[index] = transition
        self._episode_index[(transition.episode_id, transition.episode_step)] = index
        self._tree.update(index, self._tree.max_priority)
        self._write_index = (self._write_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _sequence_indices(self, anchor_index: int) -> list[int]:
        anchor = self._transitions[anchor_index]
        if anchor is None:
            raise ValueError("anchor transition is missing")
        first_step = max(0, anchor.episode_step - self.history_length + 1)
        indices: list[int] = []
        for step in range(first_step, anchor.episode_step + 1):
            index = self._episode_index.get((anchor.episode_id, step))
            if index is not None:
                indices.append(index)
        if not indices:
            indices.append(anchor_index)
        return indices

    def sample(self, batch_size: int, *, beta: float = 1.0) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        observations = np.zeros(
            (batch_size, self.history_length, self.observation_dim),
            dtype=np.float32,
        )
        actions = np.zeros(
            (batch_size, self.history_length, self.action_dim),
            dtype=np.float32,
        )
        rewards = np.zeros((batch_size, self.history_length), dtype=np.float32)
        next_observations = np.zeros_like(observations)
        dones = np.ones((batch_size, self.history_length), dtype=np.float32)
        masks = np.zeros((batch_size, self.history_length), dtype=np.float32)
        weights = np.ones(batch_size, dtype=np.float32)
        indices = np.zeros(batch_size, dtype=np.int64)
        episode_ids = np.full((batch_size, self.history_length), -1, dtype=np.int64)
        probabilities = np.zeros(batch_size, dtype=np.float64)

        total = self._tree.total
        if self.use_pper and total > 0.0:
            segments = np.linspace(0.0, total, batch_size + 1)
            for row in range(batch_size):
                prefix = self._rng.uniform(segments[row], segments[row + 1])
                indices[row], priority = self._tree.get(prefix)
                probabilities[row] = priority / total
        else:
            indices[:] = self._rng.integers(0, self._size, size=batch_size)
            probabilities[:] = 1.0 / float(self._size)

        for row, index in enumerate(indices.tolist()):
            sequence = self._sequence_indices(index)
            offset = self.history_length - len(sequence)
            for position, sequence_index in enumerate(sequence):
                transition = self._transitions[sequence_index]
                if transition is None:
                    continue
                slot = offset + position
                observations[row, slot] = transition.observation
                actions[row, slot] = transition.action
                rewards[row, slot] = transition.reward
                next_observations[row, slot] = transition.next_observation
                dones[row, slot] = float(transition.done)
                masks[row, slot] = 1.0
                episode_ids[row, slot] = transition.episode_id

        if self.use_pper:
            importance_weights = np.asarray(
                np.power(self._size * probabilities + 1e-12, -beta),
                dtype=np.float32,
            )
            weights[:] = importance_weights / max(float(np.max(importance_weights)), 1e-12)
        return ReplayBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            valid_masks=masks,
            weights=weights.astype(np.float32),
            indices=indices,
            episode_ids=episode_ids,
        )

    def update_priorities(
        self, indices: NDArray[np.int64], priorities: NDArray[np.float32]
    ) -> None:
        if indices.shape != priorities.shape:
            raise ValueError("indices and priorities must have the same shape")
        for index, priority in zip(indices.tolist(), priorities.tolist(), strict=True):
            self._tree.update(int(index), float(priority) + self.epsilon)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "history_length": self.history_length,
            "use_pper": self.use_pper,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "size": self._size,
            "write_index": self._write_index,
            "tree": self._tree.tree.copy(),
            "transitions": self._transitions,
            "episode_index": self._episode_index.copy(),
        }


__all__ = [
    "PrioritizedSequenceReplayBuffer",
    "ReplayBatch",
    "SumTree",
]
