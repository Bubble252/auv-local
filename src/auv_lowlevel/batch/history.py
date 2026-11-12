"""Explicit observation-history wrapper for the batched low-level environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .vec_env import BatchTorchVecEnv


@dataclass(frozen=True)
class HistoryLayout:
    """Flattened layout of ``[current, short_window, long_window]``.

    A policy sees the current observation first, followed by two explicitly
    separated history blocks. Keeping the layout in one place prevents the
    training wrapper and evaluation adapter from silently disagreeing about
    column order.
    """

    observation_dim: int
    short_window: int
    long_window: int
    short_stride: int = 1
    long_stride: int = 1

    def __post_init__(self) -> None:
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("history windows must be positive")
        if self.short_window > self.long_window:
            raise ValueError("short_window must not exceed long_window")
        if self.short_stride <= 0 or self.long_stride <= 0:
            raise ValueError("history strides must be positive")

    @property
    def policy_input_dim(self) -> int:
        return self.observation_dim * (1 + self.short_window + self.long_window)

    @property
    def short_span_steps(self) -> int:
        return self.short_window * self.short_stride

    @property
    def long_span_steps(self) -> int:
        return self.long_window * self.long_stride

    def flatten(self, current: Tensor, short: Tensor, long: Tensor) -> Tensor:
        if current.ndim != 2 or current.shape[1] != self.observation_dim:
            raise ValueError("current must have shape (N, observation_dim)")
        if short.shape != (current.shape[0], self.short_window, self.observation_dim):
            raise ValueError("short has the wrong shape")
        if long.shape != (current.shape[0], self.long_window, self.observation_dim):
            raise ValueError("long has the wrong shape")
        return torch.cat(
            [current, short.reshape(current.shape[0], -1), long.reshape(current.shape[0], -1)],
            dim=1,
        )

    def split(self, flat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if flat.ndim != 2 or flat.shape[1] != self.policy_input_dim:
            raise ValueError("flat history has the wrong shape")
        batch = flat.shape[0]
        current_end = self.observation_dim
        short_end = current_end + self.short_window * self.observation_dim
        current = flat[:, :current_end]
        short = flat[:, current_end:short_end].reshape(
            batch, self.short_window, self.observation_dim
        )
        long = flat[:, short_end:].reshape(batch, self.long_window, self.observation_dim)
        return current, short, long


class BatchTorchHistoryVecEnv:
    """rsl_rl-compatible view with short/long history in the observation.

    The wrapper keeps the low-level environment unchanged. It maintains two
    rolling buffers on the policy device, fills them with the reset observation
    at episode boundaries, and exposes the flattened layout defined by
    :class:`HistoryLayout`. This is the deterministic history-stack baseline for
    the later DADE encoder; it does not modify rsl_rl.
    """

    def __init__(
        self,
        env: BatchTorchVecEnv,
        *,
        short_window: int = 10,
        long_window: int = 50,
        short_stride: int = 1,
        long_stride: int = 1,
        privileged_current: bool = False,
        current_frame_dropout_prob: float = 0.0,
    ) -> None:
        self.env = env
        self._privileged_current = bool(privileged_current)
        if not 0.0 <= current_frame_dropout_prob <= 1.0:
            raise ValueError("current_frame_dropout_prob must lie in [0, 1]")
        self._current_frame_dropout_prob = float(current_frame_dropout_prob)
        self.layout = HistoryLayout(
            observation_dim=int(env.num_obs),
            short_window=short_window,
            long_window=long_window,
            short_stride=short_stride,
            long_stride=long_stride,
        )
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.num_actions)
        self.num_obs = self.layout.policy_input_dim
        self.critic_observation_dim = (
            self.layout.observation_dim + 3 if self._privileged_current else None
        )
        self.num_privileged_obs = self.critic_observation_dim
        self.device = env.device
        self.step_dt = env.step_dt
        self.max_episode_length = env.max_episode_length
        self.unwrapped = env.unwrapped
        self.cfg = {
            **env.cfg,
            "base_observation_dim": self.layout.observation_dim,
            "history_short_window": self.layout.short_window,
            "history_long_window": self.layout.long_window,
            "history_short_stride": self.layout.short_stride,
            "history_long_stride": self.layout.long_stride,
            "observation_dim": self.num_obs,
            "policy_input_dim": self.num_obs,
            "privileged_current_critic": self._privileged_current,
            "critic_observation_dim": self.critic_observation_dim,
            "current_frame_dropout_prob": self._current_frame_dropout_prob,
        }
        current, _ = self.env.get_observations()
        self._short_full = current.new_zeros(
            self.num_envs, self.layout.short_span_steps, self.layout.observation_dim
        )
        self._long_full = current.new_zeros(
            self.num_envs, self.layout.long_span_steps, self.layout.observation_dim
        )
        self._fill(current)
        self._reset_mask = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=current.device,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def total_env_steps(self) -> int:
        return self.env.total_env_steps

    @property
    def env_step_rate_hz(self) -> float:
        return self.env.env_step_rate_hz

    def _fill(self, observation: Tensor) -> None:
        self._short_full = observation.unsqueeze(1).expand(
            -1, self.layout.short_span_steps, -1
        ).clone()
        self._long_full = observation.unsqueeze(1).expand(
            -1, self.layout.long_span_steps, -1
        ).clone()

    @staticmethod
    def _sample_window(window: Tensor, *, size: int, stride: int) -> Tensor:
        if stride == 1:
            return window
        sampled = window[:, stride - 1 :: stride]
        if sampled.shape[1] != size:
            raise RuntimeError("history stride produced an inconsistent window size")
        return sampled

    def _advance(self, observation: Tensor, dones: Tensor) -> None:
        short = torch.cat([self._short_full[:, 1:], observation.unsqueeze(1)], dim=1)
        long = torch.cat([self._long_full[:, 1:], observation.unsqueeze(1)], dim=1)
        done_mask = dones.reshape(-1) > 0.5
        short = torch.where(
            done_mask[:, None, None],
            observation.unsqueeze(1).expand_as(short),
            short,
        )
        long = torch.where(
            done_mask[:, None, None],
            observation.unsqueeze(1).expand_as(long),
            long,
        )
        self._short_full = short
        self._long_full = long

    def _critic_observations(self, observation: Tensor) -> dict[str, Tensor]:
        if not self._privileged_current:
            return {}
        current = self.env.current_ned_mps
        if current is None:
            raise RuntimeError(
                "privileged-current critic requires env.current_ned_mps"
            )
        current = torch.as_tensor(
            current,
            device=observation.device,
            dtype=observation.dtype,
        )
        expected = (self.num_envs, 3)
        if tuple(current.shape) != expected:
            raise ValueError(
                f"true current must have shape {expected}, got {tuple(current.shape)}"
            )
        return {"critic": torch.cat([observation, current], dim=1)}

    @staticmethod
    def _merge_extras(
        extras: dict[str, Any], additions: dict[str, Tensor]
    ) -> dict[str, Any]:
        merged = dict(extras)
        observations = dict(merged.get("observations", {}))
        observations.update(additions)
        merged["observations"] = observations
        return merged

    def _mask_current_frames(
        self,
        current: Tensor,
        short: Tensor,
        long: Tensor,
        *,
        reset_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Hide both copies of the current frame in the policy observation.

        The current frame is present once directly and again as the last row
        of both history windows. Dropping only the direct copy is a no-op
        because the encoder can recover it from either window. The mask is
        baked into the returned PPO observation, rather than resampled inside
        the policy, so rollout log-probabilities and PPO updates see the exact
        same input.
        """

        if self._current_frame_dropout_prob <= 0.0:
            return current, short, long
        drop = torch.rand(
            current.shape[0],
            device=current.device,
            dtype=current.dtype,
        ) < self._current_frame_dropout_prob
        masked_current = torch.where(
            drop[:, None],
            torch.zeros_like(current),
            current,
        )

        if reset_mask is None:
            reset_mask = torch.zeros_like(drop)
        else:
            reset_mask = torch.as_tensor(
                reset_mask,
                device=current.device,
                dtype=torch.bool,
            ).reshape(-1)
            if reset_mask.shape != drop.shape:
                raise ValueError("reset_mask must have one entry per environment")
        full_window_drop = drop & reset_mask

        def mask_window(window: Tensor) -> Tensor:
            latest_hidden = torch.cat(
                [window[:, :-1], torch.zeros_like(window[:, -1:])],
                dim=1,
            )
            hidden = torch.where(
                full_window_drop[:, None, None],
                torch.zeros_like(window),
                latest_hidden,
            )
            return torch.where(drop[:, None, None], hidden, window)

        return masked_current, mask_window(short), mask_window(long)

    def _flattened(self, observation: Tensor, *, reset_mask: Tensor | None = None) -> Tensor:
        short = self._sample_window(
            self._short_full,
            size=self.layout.short_window,
            stride=self.layout.short_stride,
        )
        long = self._sample_window(
            self._long_full,
            size=self.layout.long_window,
            stride=self.layout.long_stride,
        )
        current, short, long = self._mask_current_frames(
            observation,
            short,
            long,
            reset_mask=reset_mask,
        )
        return self.layout.flatten(current, short, long)

    def get_observations(self) -> tuple[Tensor, dict[str, Any]]:
        observation, extras = self.env.get_observations()
        return self._flattened(observation, reset_mask=self._reset_mask), self._merge_extras(
            extras, self._critic_observations(observation)
        )

    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        observation, extras = self.env.reset()
        self._fill(observation)
        self._reset_mask = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=observation.device,
        )
        return self._flattened(observation, reset_mask=self._reset_mask), self._merge_extras(
            extras, self._critic_observations(observation)
        )

    def step(
        self,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        observation, rewards, dones, infos = self.env.step(actions)
        self._advance(observation, dones)
        self._reset_mask = dones.reshape(-1) > 0.5
        return (
            self._flattened(observation, reset_mask=self._reset_mask),
            rewards,
            dones,
            self._merge_extras(infos, self._critic_observations(observation)),
        )


__all__ = ["BatchTorchHistoryVecEnv", "HistoryLayout"]
