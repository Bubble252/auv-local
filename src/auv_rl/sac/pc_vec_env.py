"""Stable-Baselines3 VecEnv adapter for the batched PC residual environment.

The PC environment is already vectorized in torch.  This adapter keeps that
vectorization and adds the fixed-length observation history used by the
existing off-policy and PPO PC policies.  Policy actions are normalized
command residuals in ``[-1, 1]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    return value


def _column(value: Any) -> Any:
    """Convert an info value to a host array once, before per-row indexing."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _rows(values: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    """Split vector info into per-env rows without repeated GPU copies."""
    columns = {key: _column(value) for key, value in values.items()}
    rows: list[dict[str, Any]] = []
    for index in range(num_envs):
        row: dict[str, Any] = {}
        for key, value in values.items():
            column = columns[key]
            if column is None:
                row[key] = value
            elif column.ndim == 0:
                row[key] = column.item()
            else:
                row[key] = column[index]
        rows.append(row)
    return rows


class PCHistoryVecEnv(VecEnv):
    """Expose ``PCCommandResidualOffPolicyEnv`` to SB3 SAC.

    ``history_length=1`` gives the raw flat observation.  Values greater than
    one stack the most recent observations per environment and flatten the
    stack in time-major order, matching the sequence context used by the
    historical DDPG PC trainer.
    """

    def __init__(
        self,
        env: PCCommandResidualOffPolicyEnv,
        *,
        history_length: int = 10,
    ) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        self.env = env
        self.history_length = int(history_length)
        self._observation_dim = int(env.observation_dim)
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.history_length * self._observation_dim,),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(env.action_dim),),
            dtype=np.float32,
        )
        self._history = np.zeros(
            (int(env.num_envs), self.history_length, self._observation_dim),
            dtype=np.float32,
        )
        self._actions = np.zeros((int(env.num_envs), int(env.action_dim)), dtype=np.float64)
        self._completed: list[dict[str, Any]] = []
        super().__init__(int(env.num_envs), observation_space, action_space)

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def stacked_dim(self) -> int:
        return int(self.observation_space.shape[0])

    def _stack(self) -> np.ndarray:
        return self._history.reshape(self.num_envs, -1).copy()

    def _set_initial_history(self, observations: np.ndarray) -> None:
        array = np.asarray(observations, dtype=np.float32)
        if array.shape != (self.num_envs, self._observation_dim):
            raise ValueError("observations have the wrong shape")
        self._history[:] = array[:, None, :]

    def _append_history(self, observations: np.ndarray, done: np.ndarray) -> None:
        array = np.asarray(observations, dtype=np.float32)
        if array.shape != (self.num_envs, self._observation_dim):
            raise ValueError("observations have the wrong shape")
        if self.history_length > 1:
            self._history[:, :-1] = self._history[:, 1:]
        self._history[:, -1] = array
        if done.any():
            self._history[done] = array[done, None, :]

    def _init_reset_infos(self, info: dict[str, Any]) -> None:
        self.reset_infos = _rows(info, self.num_envs)

    def reset(self) -> np.ndarray:
        observations, info = self.env.reset()
        self._set_initial_history(observations)
        self._init_reset_infos(info)
        return self._stack()

    def step_async(self, actions: np.ndarray) -> None:
        array = np.asarray(actions, dtype=np.float64)
        if array.shape != (self.num_envs, int(self.env.action_dim)):
            raise ValueError("actions have the wrong shape")
        self._actions = array

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, rewards, terminated, truncated, info = self.env.step(self._actions)
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        self._completed.extend(self.env.drain_completed_infos())
        final_observations = np.asarray(info["final_observations"], dtype=np.float32)
        terminal_history = self._history.copy()
        if self.history_length > 1:
            terminal_history[:, :-1] = terminal_history[:, 1:]
        terminal_history[:, -1] = final_observations
        terminal_stacked = terminal_history.reshape(self.num_envs, -1)
        self._append_history(observations, done)
        infos = _rows(info, self.num_envs)
        for index, row in enumerate(infos):
            row["TimeLimit.truncated"] = bool(truncated[index] and not terminated[index])
            if done[index]:
                row["terminal_observation"] = terminal_stacked[index].copy()
        return (
            self._stack(),
            np.asarray(rewards, dtype=np.float32),
            done.astype(bool),
            infos,
        )

    def drain_completed_infos(self) -> list[dict[str, Any]]:
        rows = self._completed
        self._completed = []
        return rows

    def set_action_mode(self, mode: str) -> None:
        self.env.set_action_mode(mode)

    def set_path_zero(self, enabled: bool) -> None:
        self.env.set_path_zero(enabled)

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        values = [getattr(self.env, attr_name, None) for _ in range(self.num_envs)]
        if indices is None:
            return values
        return [values[index] for index in indices]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Any = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        method = getattr(self.env, method_name)
        result = method(*method_args, **method_kwargs)
        return [result for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return [False for _ in range(self.num_envs)]


__all__ = ["PCHistoryVecEnv"]
