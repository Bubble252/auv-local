from __future__ import annotations

import copy
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

    def update_many(self, indices: NDArray[np.int64], priorities: NDArray[np.float64]) -> None:
        """Update multiple leaves and propagate their deltas level by level."""

        index_array = np.asarray(indices, dtype=np.int64)
        priority_array = np.asarray(priorities, dtype=np.float64)
        if index_array.shape != priority_array.shape or index_array.ndim != 1:
            raise ValueError("indices and priorities must be equal one-dimensional arrays")
        if index_array.size == 0:
            return
        if np.any(index_array < 0) or np.any(index_array >= self.capacity):
            raise IndexError("leaf index out of range")
        values = np.maximum(priority_array, 0.0)
        leaves = index_array + self.capacity - 1
        deltas = values - self.tree[leaves]
        self.tree[leaves] = values
        parents = (leaves - 1) // 2
        for _ in range(self.capacity.bit_length() - 1):
            np.add.at(self.tree, parents, deltas)
            parents = (parents - 1) // 2
        self.n_entries = max(self.n_entries, int(np.max(index_array)) + 1)

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

    def get_many(
        self,
        prefixes: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Resolve many prefix sums with one vectorized traversal per tree level."""

        prefix_array = np.asarray(prefixes, dtype=np.float64)
        if prefix_array.ndim != 1:
            raise ValueError("prefixes must be a one-dimensional array")
        if prefix_array.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        if self.n_entries == 0 or self.total <= 0.0:
            raise ValueError("cannot sample from an empty SumTree")
        targets = np.clip(prefix_array, 0.0, self.total - 1e-12)
        positions = np.zeros(targets.shape, dtype=np.int64)
        for _ in range(self.capacity.bit_length() - 1):
            left = 2 * positions + 1
            left_values = self.tree[left]
            go_right = targets > left_values
            targets = np.where(go_right, targets - left_values, targets)
            positions = np.where(go_right, left + 1, left)
        leaves = positions - self.capacity + 1
        return leaves, np.asarray(self.tree[positions], dtype=np.float64)


@dataclass(frozen=True)
class ReplayBatch:
    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    rewards: NDArray[np.float32]
    next_observations: NDArray[np.float32]
    privileged: NDArray[np.float32]
    next_privileged: NDArray[np.float32]
    dones: NDArray[np.float32]
    valid_masks: NDArray[np.float32]
    weights: NDArray[np.float32]
    indices: NDArray[np.int64]
    episode_ids: NDArray[np.int64]
    env_ids: NDArray[np.int64]

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
    privileged: NDArray[np.float32]
    next_privileged: NDArray[np.float32]
    done: bool
    episode_id: int
    episode_step: int


class PrioritizedSequenceReplayBuffer:
    """Tensor ring buffer that samples contiguous, episode-local sequences.

    Priorities belong to the sampled anchor transition. Earlier steps are used
    only as GRU context and never borrowed from a different episode or env.
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
        privileged_dim: int = 0,
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
        if privileged_dim < 0:
            raise ValueError("privileged_dim must be non-negative")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.history_length = int(history_length)
        self.use_pper = bool(use_pper)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.privileged_dim = int(privileged_dim)
        self._rng = np.random.default_rng(seed)
        self._tree = SumTree(self.capacity)
        self._observations = np.zeros((self.capacity, self.observation_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_observations = np.zeros_like(self._observations)
        self._privileged = np.zeros((self.capacity, self.privileged_dim), dtype=np.float32)
        self._next_privileged = np.zeros_like(self._privileged)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._env_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._episode_steps = np.full(self.capacity, -1, dtype=np.int64)
        self._valid = np.zeros(self.capacity, dtype=bool)
        self._episode_index: dict[tuple[int, int, int], int] = {}
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
        privileged: NDArray[np.float32] | np.ndarray | None = None,
        next_privileged: NDArray[np.float32] | np.ndarray | None = None,
        env_id: int = -1,
    ) -> None:
        self.add_batch(
            np.asarray(observation, dtype=np.float32).reshape(1, self.observation_dim),
            np.asarray(action, dtype=np.float32).reshape(1, self.action_dim),
            np.asarray([reward], dtype=np.float32),
            np.asarray(next_observation, dtype=np.float32).reshape(1, self.observation_dim),
            np.asarray([done], dtype=np.bool_),
            episode_ids=np.asarray([episode_id], dtype=np.int64),
            episode_steps=np.asarray([episode_step], dtype=np.int64),
            privileged=(
                None
                if privileged is None
                else np.asarray(privileged, dtype=np.float32).reshape(1, self.privileged_dim)
            ),
            next_privileged=(
                None
                if next_privileged is None
                else np.asarray(next_privileged, dtype=np.float32).reshape(1, self.privileged_dim)
            ),
            env_ids=np.asarray([env_id], dtype=np.int64),
        )

    def add_batch(
        self,
        observations: NDArray[np.float32],
        actions: NDArray[np.float32],
        rewards: NDArray[np.float32],
        next_observations: NDArray[np.float32],
        dones: NDArray[np.bool_],
        *,
        episode_ids: NDArray[np.int64],
        episode_steps: NDArray[np.int64],
        privileged: NDArray[np.float32] | None = None,
        next_privileged: NDArray[np.float32] | None = None,
        env_ids: NDArray[np.int64] | None = None,
    ) -> None:
        obs = np.asarray(observations, dtype=np.float32)
        act = np.asarray(actions, dtype=np.float32)
        next_obs = np.asarray(next_observations, dtype=np.float32)
        reward_array = np.asarray(rewards, dtype=np.float32)
        done_array = np.asarray(dones, dtype=np.bool_)
        ids = np.asarray(episode_ids, dtype=np.int64)
        steps = np.asarray(episode_steps, dtype=np.int64)
        env_values = (
            np.full(ids.shape, -1, dtype=np.int64)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.int64)
        )
        privileged_array = (
            np.zeros((obs.shape[0], self.privileged_dim), dtype=np.float32)
            if privileged is None
            else np.asarray(privileged, dtype=np.float32)
        )
        next_privileged_array = (
            np.zeros((obs.shape[0], self.privileged_dim), dtype=np.float32)
            if next_privileged is None
            else np.asarray(next_privileged, dtype=np.float32)
        )
        batch_size = obs.shape[0] if obs.ndim == 2 else -1
        if batch_size <= 0 or obs.shape[1] != self.observation_dim:
            raise ValueError("observations shape does not match replay buffer")
        if act.shape != (batch_size, self.action_dim):
            raise ValueError("actions shape does not match replay buffer")
        if next_obs.shape != obs.shape:
            raise ValueError("next_observations shape does not match observations")
        if reward_array.shape != (batch_size,) or done_array.shape != (batch_size,):
            raise ValueError("rewards and dones must have shape (B,)")
        if ids.shape != (batch_size,) or steps.shape != (batch_size,):
            raise ValueError("episode_ids and episode_steps must have shape (B,)")
        if env_values.shape != (batch_size,):
            raise ValueError("env_ids must have shape (B,)")
        if privileged_array.shape != (batch_size, self.privileged_dim):
            raise ValueError("privileged shape does not match replay buffer")
        if next_privileged_array.shape != privileged_array.shape:
            raise ValueError("next_privileged shape does not match privileged")
        if np.any(ids < 0) or np.any(steps < 0):
            raise ValueError("episode_id and episode_step must be non-negative")
        if not all(
            np.all(np.isfinite(array))
            for array in (
                obs,
                act,
                reward_array,
                next_obs,
                privileged_array,
                next_privileged_array,
            )
        ):
            raise ValueError("replay values must contain only finite values")

        for row in range(batch_size):
            index = self._write_index
            if self._valid[index]:
                old_env = int(self._env_ids[index])
                old_episode = int(self._episode_ids[index])
                old_step = int(self._episode_steps[index])
                self._episode_index.pop((old_env, old_episode, old_step), None)
            self._observations[index] = obs[row]
            self._actions[index] = act[row]
            self._rewards[index] = reward_array[row]
            self._next_observations[index] = next_obs[row]
            self._privileged[index] = privileged_array[row]
            self._next_privileged[index] = next_privileged_array[row]
            self._dones[index] = float(done_array[row])
            self._env_ids[index] = int(env_values[row])
            self._episode_ids[index] = int(ids[row])
            self._episode_steps[index] = int(steps[row])
            self._valid[index] = True
            self._episode_index[(int(env_values[row]), int(ids[row]), int(steps[row]))] = index
            self._tree.update(index, self._tree.max_priority)
            self._write_index = (self._write_index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def _sequence_indices(self, anchor_index: int) -> list[int]:
        anchor_env = int(self._env_ids[anchor_index])
        anchor_id = int(self._episode_ids[anchor_index])
        anchor_step = int(self._episode_steps[anchor_index])
        if anchor_id < 0 or anchor_step < 0:
            raise ValueError("anchor transition is missing")
        first_step = max(0, anchor_step - self.history_length + 1)
        indices: list[int] = []
        for step in range(first_step, anchor_step + 1):
            index = self._episode_index.get((anchor_env, anchor_id, step))
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
        weights = np.ones(batch_size, dtype=np.float32)
        indices = np.zeros(batch_size, dtype=np.int64)
        probabilities = np.zeros(batch_size, dtype=np.float64)

        total = self._tree.total
        if self.use_pper and total > 0.0:
            segments = np.linspace(0.0, total, batch_size + 1)
            prefixes = self._rng.uniform(segments[:-1], segments[1:])
            indices[:], priorities = self._tree.get_many(prefixes)
            probabilities[:] = priorities / total
        else:
            indices[:] = self._rng.integers(0, self._size, size=batch_size)
            probabilities[:] = 1.0 / float(self._size)

        sequence_indices = np.full(
            (batch_size, self.history_length),
            -1,
            dtype=np.int64,
        )
        for row, index in enumerate(indices.tolist()):
            sequence = self._sequence_indices(index)
            offset = self.history_length - len(sequence)
            sequence_indices[row, offset : offset + len(sequence)] = sequence

        valid = sequence_indices >= 0
        gather_indices = np.where(valid, sequence_indices, 0)
        masks = valid.astype(np.float32)
        observations = np.where(valid[..., None], self._observations[gather_indices], 0.0).astype(
            np.float32
        )
        actions = np.where(valid[..., None], self._actions[gather_indices], 0.0).astype(np.float32)
        rewards = np.where(valid, self._rewards[gather_indices], 0.0).astype(np.float32)
        next_observations = np.where(
            valid[..., None],
            self._next_observations[gather_indices],
            0.0,
        ).astype(np.float32)
        privileged = np.where(
            valid[..., None],
            self._privileged[gather_indices],
            0.0,
        ).astype(np.float32)
        next_privileged = np.where(
            valid[..., None],
            self._next_privileged[gather_indices],
            0.0,
        ).astype(np.float32)
        dones = np.where(valid, self._dones[gather_indices], 1.0).astype(np.float32)
        sampled_episode_ids = np.where(valid, self._episode_ids[gather_indices], -1).astype(
            np.int64
        )
        sampled_env_ids = np.where(valid, self._env_ids[gather_indices], -1).astype(np.int64)

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
            privileged=privileged,
            next_privileged=next_privileged,
            dones=dones,
            valid_masks=masks,
            weights=weights.astype(np.float32),
            indices=indices,
            episode_ids=sampled_episode_ids,
            env_ids=sampled_env_ids,
        )

    def update_priorities(
        self,
        indices: NDArray[np.int64],
        priorities: NDArray[np.float32],
    ) -> None:
        index_array = np.asarray(indices, dtype=np.int64)
        priority_array = np.asarray(priorities, dtype=np.float32)
        if index_array.shape != priority_array.shape:
            raise ValueError("indices and priorities must have the same shape")
        if index_array.ndim != 1:
            raise ValueError("indices and priorities must be one-dimensional")
        if np.any(index_array < 0) or np.any(index_array >= self.capacity):
            raise IndexError("leaf index out of range")
        self._tree.update_many(index_array, priority_array.astype(np.float64) + self.epsilon)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "privileged_dim": self.privileged_dim,
            "history_length": self.history_length,
            "use_pper": self.use_pper,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "size": self._size,
            "write_index": self._write_index,
            "tree": self._tree.tree.copy(),
            "observations": self._observations.copy(),
            "actions": self._actions.copy(),
            "rewards": self._rewards.copy(),
            "next_observations": self._next_observations.copy(),
            "privileged": self._privileged.copy(),
            "next_privileged": self._next_privileged.copy(),
            "dones": self._dones.copy(),
            "env_ids": self._env_ids.copy(),
            "episode_ids": self._episode_ids.copy(),
            "episode_steps": self._episode_steps.copy(),
            "valid": self._valid.copy(),
            "episode_index": self._episode_index.copy(),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "tree_n_entries": self._tree.n_entries,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore buffer contents, priorities, episode index and sampling RNG."""

        expected = {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "privileged_dim": self.privileged_dim,
            "history_length": self.history_length,
            "use_pper": self.use_pper,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"replay checkpoint {key} does not match buffer")

        tree = np.asarray(state["tree"], dtype=np.float64)
        if tree.shape != self._tree.tree.shape:
            raise ValueError("replay checkpoint tree shape does not match buffer")
        self._tree.tree[:] = tree
        self._tree.n_entries = int(state.get("tree_n_entries", 0))

        arrays = (
            ("_observations", self._observations, np.float32),
            ("_actions", self._actions, np.float32),
            ("_rewards", self._rewards, np.float32),
            ("_next_observations", self._next_observations, np.float32),
            ("_privileged", self._privileged, np.float32),
            ("_next_privileged", self._next_privileged, np.float32),
            ("_dones", self._dones, np.float32),
            ("_env_ids", self._env_ids, np.int64),
            ("_episode_ids", self._episode_ids, np.int64),
            ("_episode_steps", self._episode_steps, np.int64),
            ("_valid", self._valid, np.bool_),
        )
        for name, destination, dtype in arrays:
            state_key = name.removeprefix("_")
            source = np.asarray(state[state_key], dtype=dtype)
            if source.shape != destination.shape:
                raise ValueError(f"replay checkpoint {state_key} shape does not match buffer")
            destination[...] = source

        size = int(state["size"])
        write_index = int(state["write_index"])
        if not 0 <= size <= self.capacity:
            raise ValueError("replay checkpoint size is out of range")
        if not 0 <= write_index < self.capacity:
            raise ValueError("replay checkpoint write_index is out of range")
        self._size = size
        self._write_index = write_index
        self._episode_index = {
            (int(env_id), int(episode_id), int(step)): int(index)
            for (env_id, episode_id, step), index in dict(state["episode_index"]).items()
        }
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, dict):
            raise ValueError("replay checkpoint is missing its RNG state")
        self._rng.bit_generator.state = copy.deepcopy(rng_state)


__all__ = [
    "PrioritizedSequenceReplayBuffer",
    "ReplayBatch",
    "SumTree",
]
