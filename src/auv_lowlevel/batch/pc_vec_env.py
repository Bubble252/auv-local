"""rsl_rl-compatible PC vector environment with separate actor/critic history."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from .pc import PC_SAFETY_TERMINAL_KINDS, PCRewardBreakdown, PCTerminalKind
from .pc_env import BatchedPCLowLevelEnv, BatchedPCObservation
from .pc_normalization import PCNormalization
from .pc_path import normalize_path_features, path_block_column_mask, validate_path_blocks
from .pc_policy import PCPolicyLayout


class BatchTorchPCVecEnv:
    """Expose PC observations as flattened actor and critic history layouts.

    The actor and critic maintain independent rolling buffers. History tokens
    contain only ``[active_state, side_channel]``; the command is supplied only
    through the top-level ``c_t`` block. Completed rows are filled with their
    post-reset observation so one environment cannot leak an old episode into
    another row's history.
    """

    def __init__(
        self,
        env: BatchedPCLowLevelEnv,
        *,
        short_window: int = 10,
        long_window: int = 50,
        short_stride: int = 1,
        long_stride: int = 1,
        actor_decision_stride: int = 1,
        actor_command_dim: int = 3,
        critic_command_dim: int = 3,
        history_path_dim: int = 0,
        actor_current_encoder_enabled: bool = False,
        normalization: PCNormalization | None = None,
        current_frame_dropout_prob: float = 0.0,
        current_frame_dropout_mode: str = "full",
        path_blocks: Iterable[str] | None = None,
        path_zero: bool = False,
        observation_dtype: torch.dtype = torch.float32,
        record_step_timings: bool = False,
    ) -> None:
        if short_window <= 0 or long_window <= 0 or short_window > long_window:
            raise ValueError("history windows must satisfy 0 < short <= long")
        if short_stride <= 0 or long_stride <= 0:
            raise ValueError("history strides must be positive")
        if actor_decision_stride <= 0:
            raise ValueError("actor decision stride must be positive")
        if short_window * short_stride > long_window * long_stride:
            raise ValueError("short history span must not exceed long history span")
        if not 0.0 <= current_frame_dropout_prob <= 1.0:
            raise ValueError("current_frame_dropout_prob must lie in [0, 1]")
        if current_frame_dropout_mode not in {"full", "side_channel", "history"}:
            raise ValueError(
                "current_frame_dropout_mode must be 'full', 'side_channel', or 'history'"
            )
        if observation_dtype != torch.float32:
            raise ValueError("policy observations must use float32")

        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.action_dim
        self.device = env.device
        self.step_dt = env.dt_s
        self.max_episode_length = int(env.config.plant.max_episode_steps)
        self.observation_dtype = observation_dtype
        configured = env.config.normalization
        self.normalization = normalization or PCNormalization(
            position_scale_m=configured.position_scale_m,
            attitude_scale_rad=configured.attitude_scale_rad,
            angular_rate_scale_rad_s=configured.angular_rate_scale_rad_s,
            surge_speed_scale_mps=configured.surge_speed_scale_mps,
            sway_heave_speed_scale_mps=configured.sway_heave_speed_scale_mps,
            current_scale_mps=configured.current_scale_mps,
        )
        self.actor_layout = PCPolicyLayout(
            state_dim=env.actor_state_dim,
            path_dim=env.actor_path_dim,
            current_dim=3 if actor_current_encoder_enabled else 0,
            short_window=short_window,
            long_window=long_window,
            command_dim=actor_command_dim,
            history_path_dim=history_path_dim,
        )
        self.critic_layout = PCPolicyLayout(
            state_dim=env.critic_state_dim,
            short_window=short_window,
            long_window=long_window,
            command_dim=critic_command_dim,
        )
        self.num_obs = self.actor_layout.flat_dim
        self.num_privileged_obs = self.critic_layout.flat_dim
        self.critic_observation_dim = self.critic_layout.flat_dim
        self.unwrapped = env
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.short_stride = int(short_stride)
        self.long_stride = int(long_stride)
        self.actor_decision_stride = int(actor_decision_stride)
        self.current_frame_dropout_prob = float(current_frame_dropout_prob)
        self.current_frame_dropout_mode = current_frame_dropout_mode
        self.path_zero = bool(path_zero)
        self.path_blocks = (
            validate_path_blocks(env.path_blocks)
            if path_blocks is None
            else validate_path_blocks(path_blocks)
        )
        self._path_column_mask = self._resolve_path_column_mask(self.path_blocks)
        self.capacity = 1 + (self.long_window - 1) * self.long_stride
        self._short_offsets = (
            torch.arange(self.short_window - 1, -1, -1, device=self.device) * self.short_stride
        )
        self._long_offsets = (
            torch.arange(self.long_window - 1, -1, -1, device=self.device) * self.long_stride
        )
        self._actor_history: Tensor | None = None
        self._critic_history: Tensor | None = None
        self._actor_observations: Tensor | None = None
        self._critic_observations: Tensor | None = None
        self._last_observation: BatchedPCObservation | None = None
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_return = self._zeros()
        self._episode_abs_heading_error = self._zeros()
        self._episode_abs_pitch_error = self._zeros()
        self._episode_abs_speed_error = self._zeros()
        self._episode_abs_roll = self._zeros()
        self._episode_material_saturation = self._zeros()
        self._episode_safety_events = self._zeros()
        self._total_env_steps = 0
        self._elapsed_s = 0.0
        self._record_step_timings = bool(record_step_timings)
        self._step_latencies_ms: list[float] = []
        self.cfg = {
            "backend": "batch_torch_pc",
            "config_sha256": env.config.config_sha256,
            "num_envs": self.num_envs,
            "policy_kind": "pc_actor_critic",
            "pc_stage": env.stage,
            "sensor_profile": env.config.sensors.profile if env.config.sensors else None,
            "sensor_source": env.config.sensor_source,
            "actor_state_dim": env.actor_state_dim,
            "actor_path_dim": env.actor_path_dim,
            "actor_path_blocks": list(self.path_blocks),
            "path_zero": self.path_zero,
            "critic_state_dim": env.critic_state_dim,
            "command_dim": self.actor_layout.command_dim,
            "critic_command_dim": self.critic_layout.command_dim,
            "history_path_dim": self.actor_layout.history_path_dim,
            "actor_current_encoder_enabled": actor_current_encoder_enabled,
            "actor_current_dim": self.actor_layout.current_dim,
            "side_channel_dim": self.actor_layout.side_channel_dim,
            "actor_token_dim": self.actor_layout.token_dim,
            "critic_token_dim": self.critic_layout.token_dim,
            "observation_dim": self.num_obs,
            "critic_observation_dim": self.critic_observation_dim,
            "history_short_window": self.short_window,
            "history_long_window": self.long_window,
            "history_short_stride": self.short_stride,
            "history_long_stride": self.long_stride,
            "current_frame_dropout_prob": self.current_frame_dropout_prob,
            "current_frame_dropout_mode": self.current_frame_dropout_mode,
            "pc_ppo_scale": self.env.config.reward.resolved_pc_ppo_scale,
            "normalization": self.normalization.as_dict(),
            "observation_dtype": str(observation_dtype).removeprefix("torch."),
            "actor_side_channel": (
                "current_estimate_body"
                if self.actor_layout.current_dim
                else "zero_when_actor_current_encoder_disabled"
            ),
            "critic_side_channel": "true_current_ned",
            "true_current_in_actor_observation": False,
        }
        self.reset()

    def _zeros(self) -> Tensor:
        return torch.zeros(self.num_envs, dtype=self.observation_dtype, device=self.device)

    def _ensure_observations(self) -> tuple[Tensor, Tensor]:
        if self._actor_observations is None or self._critic_observations is None:
            raise RuntimeError("reset() must be called before accessing observations")
        return self._actor_observations, self._critic_observations

    def _tokens(
        self, observation: BatchedPCObservation
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor, Tensor, Tensor, Tensor, Tensor]:
        actor_state = self.normalization.normalize_actor_state(observation.actor_state)
        actor_path = observation.actor_path
        if self.actor_layout.path_dim:
            if actor_path is None:
                raise RuntimeError("PC environment did not expose the path observation block")
            actor_path = normalize_path_features(actor_path)
            mask = self._path_column_mask.to(device=actor_path.device, dtype=actor_path.dtype)
            actor_path = actor_path * mask
            if self.path_zero:
                actor_path = torch.zeros_like(actor_path)
        elif actor_path is not None:
            raise ValueError("unexpected actor path observations when path_dim=0")
        actor_current = (
            self.normalization.normalize_current(observation.actor_current_body)
            if self.actor_layout.current_dim
            else None
        )
        critic_state = self.normalization.normalize_critic_state(observation.critic_state)
        # The actor current encoder is the only explicit current channel.  When
        # it is disabled, zero the history side channel too; otherwise the same
        # current estimate would leak through the actor's short/long history.
        actor_side = (
            self.normalization.normalize_current(observation.actor_current_body)
            if self.actor_layout.current_dim
            else torch.zeros_like(observation.actor_current_body)
        )
        critic_side = self.normalization.normalize_current(observation.critic_current_ned)
        command_full = observation.command.normalized()
        actor_command = self.normalization.normalize_command(
            self._select_command(command_full, self.actor_layout.command_dim)
        )
        critic_command = self.normalization.normalize_command(
            self._select_command(command_full, self.critic_layout.command_dim)
        )
        actor_token_parts = [actor_state]
        if self.actor_layout.history_path_dim:
            assert actor_path is not None
            actor_token_parts.append(actor_path)
        actor_token_parts.append(actor_side)
        actor_token = torch.cat(actor_token_parts, dim=1).to(self.observation_dtype)
        critic_token = torch.cat([critic_state, critic_side], dim=1).to(self.observation_dtype)
        return (
            actor_state.to(self.observation_dtype),
            None if actor_path is None else actor_path.to(self.observation_dtype),
            None if actor_current is None else actor_current.to(self.observation_dtype),
            critic_state.to(self.observation_dtype),
            actor_command.to(self.observation_dtype),
            critic_command.to(self.observation_dtype),
            actor_token,
            critic_token,
        )

    def _mask_history_path(self, history: Tensor) -> Tensor:
        """Apply the active path mask to cached encoded-history path columns."""

        history_path_dim = self.actor_layout.history_path_dim
        if not history_path_dim:
            return history
        start = self.actor_layout.state_dim
        end = start + history_path_dim
        if history.shape[-1] < end:
            raise ValueError("actor history is narrower than its path layout")
        masked = history.clone()
        mask = self._path_column_mask.to(device=history.device, dtype=history.dtype)
        masked[..., start:end] = masked[..., start:end] * mask
        if self.path_zero:
            masked[..., start:end] = 0.0
        return masked

    @staticmethod
    def _select_command(command: Tensor, dim: int) -> Tensor:
        if dim == 3:
            return command
        if dim == 1:
            return command[:, 2:3]
        if dim == 0:
            return command[:, :0]
        raise ValueError("command dimension must be 0, 1, or 3")

    def set_path_blocks(self, blocks: Iterable[str]) -> None:
        """Change active path blocks without changing any tensor width."""

        self.path_blocks = validate_path_blocks(blocks)
        self._path_column_mask = self._resolve_path_column_mask(self.path_blocks)
        self.cfg["actor_path_blocks"] = list(self.path_blocks)
        self._refresh_observations()

    def set_path_zero(self, enabled: bool) -> None:
        self.path_zero = bool(enabled)
        self.cfg["path_zero"] = self.path_zero
        self._refresh_observations()

    def _resolve_path_column_mask(self, blocks: Iterable[str]) -> Tensor:
        path_dim = self.actor_layout.path_dim
        if path_dim == 0:
            if tuple(blocks):
                raise ValueError("path_blocks must be empty when path_dim=0")
            return torch.empty(0, dtype=torch.float32, device=self.device)
        mask = path_block_column_mask(blocks, path_dim=path_dim).to(self.device)
        if int(mask.numel()) != path_dim:
            raise ValueError("path_blocks do not match actor path_dim")
        return mask

    def _refresh_observations(self) -> None:
        """Rebuild current observations after a path-mask change.

        The path block is not part of the P1 history token, so history buffers
        stay untouched and the curriculum switch only changes the current path
        columns before the encoder.
        """

        if self._last_observation is None or self._actor_history is None:
            return
        (
            actor_state,
            actor_path,
            actor_current,
            critic_state,
            actor_command,
            critic_command,
            _actor_token,
            _critic_token,
        ) = self._tokens(self._last_observation)
        actor_short = self._sample(self._actor_history, self._short_offsets)
        actor_long = self._sample(self._actor_history, self._long_offsets)
        actor_short = self._mask_history_path(actor_short)
        actor_long = self._mask_history_path(actor_long)
        assert self._critic_history is not None
        critic_short = self._sample(self._critic_history, self._short_offsets)
        critic_long = self._sample(self._critic_history, self._long_offsets)
        self._actor_observations = self.actor_layout.flatten(
            state=actor_state,
            path=actor_path,
            current=actor_current,
            command=actor_command,
            short=actor_short,
            long=actor_long,
        )
        self._critic_observations = self.critic_layout.flatten(
            state=critic_state,
            command=critic_command,
            short=critic_short,
            long=critic_long,
        )

    def _fill_history(
        self, actor_token: Tensor, critic_token: Tensor, *, reset_mask: Tensor
    ) -> None:
        if self._actor_history is None or self._critic_history is None:
            actor_filled = actor_token[:, None, :].expand(-1, self.capacity, -1).clone()
            critic_filled = critic_token[:, None, :].expand(-1, self.capacity, -1).clone()
        else:
            actor_filled = torch.roll(self._actor_history, shifts=-1, dims=1)
            critic_filled = torch.roll(self._critic_history, shifts=-1, dims=1)
            actor_filled[:, -1, :] = actor_token
            critic_filled[:, -1, :] = critic_token
            reset_view = reset_mask[:, None, None]
            actor_reset = actor_token[:, None, :].expand_as(actor_filled)
            critic_reset = critic_token[:, None, :].expand_as(critic_filled)
            actor_filled = torch.where(reset_view, actor_reset, actor_filled)
            critic_filled = torch.where(reset_view, critic_reset, critic_filled)
        self._actor_history = actor_filled
        self._critic_history = critic_filled

    def _sample(self, history: Tensor, offsets: Tensor) -> Tensor:
        indices = history.shape[1] - 1 - offsets
        return history[:, indices, :]

    def _mask_current(
        self,
        state: Tensor,
        short: Tensor,
        long: Tensor,
        *,
        drop: Tensor,
        reset_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.current_frame_dropout_prob <= 0.0:
            return state, short, long
        side_slice = slice(state.shape[1], short.shape[2])
        full_drop = drop & reset_mask
        if self.current_frame_dropout_mode == "side_channel":

            def mask_side_channel(window: Tensor) -> Tensor:
                latest_hidden = window.clone()
                latest_hidden[:, -1, side_slice] = 0.0
                full_hidden = window.clone()
                full_hidden[..., side_slice] = 0.0
                hidden = torch.where(full_drop[:, None, None], full_hidden, latest_hidden)
                return torch.where(drop[:, None, None], hidden, window)

            return state, mask_side_channel(short), mask_side_channel(long)
        if self.current_frame_dropout_mode == "history":

            def mask_history(window: Tensor) -> Tensor:
                latest_hidden = window.clone()
                latest_hidden[:, -1, :] = 0.0
                full_hidden = torch.zeros_like(window)
                hidden = torch.where(full_drop[:, None, None], full_hidden, latest_hidden)
                return torch.where(drop[:, None, None], hidden, window)

            return state, mask_history(short), mask_history(long)
        zero_state = torch.zeros_like(state)
        masked_state = torch.where(drop[:, None], zero_state, state)

        def mask_window(window: Tensor) -> Tensor:
            latest_hidden = torch.cat([window[:, :-1], torch.zeros_like(window[:, -1:])], dim=1)
            hidden = torch.where(full_drop[:, None, None], torch.zeros_like(window), latest_hidden)
            return torch.where(drop[:, None, None], hidden, window)

        return masked_state, mask_window(short), mask_window(long)

    def _flatten(
        self,
        actor_state: Tensor,
        actor_path: Tensor | None,
        actor_current: Tensor | None,
        critic_state: Tensor,
        actor_command: Tensor,
        critic_command: Tensor,
        *,
        reset_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self._actor_history is None or self._critic_history is None:
            raise RuntimeError("history buffers are not initialized")
        actor_short = self._sample(self._actor_history, self._short_offsets)
        actor_long = self._sample(self._actor_history, self._long_offsets)
        actor_short = self._mask_history_path(actor_short)
        actor_long = self._mask_history_path(actor_long)
        critic_short = self._sample(self._critic_history, self._short_offsets)
        critic_long = self._sample(self._critic_history, self._long_offsets)
        drop = (
            torch.rand(self.num_envs, device=self.device, dtype=torch.float32)
            < self.current_frame_dropout_prob
            if self.current_frame_dropout_prob > 0.0
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        actor_state, actor_short, actor_long = self._mask_current(
            actor_state,
            actor_short,
            actor_long,
            drop=drop,
            reset_mask=reset_mask,
        )
        if actor_current is not None and self.current_frame_dropout_prob > 0.0:
            if self.current_frame_dropout_mode in {"full", "side_channel"}:
                actor_current = torch.where(
                    drop[:, None],
                    torch.zeros_like(actor_current),
                    actor_current,
                )
        critic_state, critic_short, critic_long = self._mask_current(
            critic_state,
            critic_short,
            critic_long,
            drop=drop,
            reset_mask=reset_mask,
        )
        actor_obs = self.actor_layout.flatten(
            state=actor_state,
            path=actor_path,
            current=actor_current,
            command=actor_command,
            short=actor_short,
            long=actor_long,
        )
        critic_obs = self.critic_layout.flatten(
            state=critic_state,
            command=critic_command,
            short=critic_short,
            long=critic_long,
        )
        return actor_obs, critic_obs

    def _set_observations(self, observation: BatchedPCObservation, *, reset_mask: Tensor) -> None:
        (
            actor_state,
            actor_path,
            actor_current,
            critic_state,
            actor_command,
            critic_command,
            actor_token,
            critic_token,
        ) = self._tokens(observation)
        self._fill_history(actor_token, critic_token, reset_mask=reset_mask)
        actor_obs, critic_obs = self._flatten(
            actor_state,
            actor_path,
            actor_current,
            critic_state,
            actor_command,
            critic_command,
            reset_mask=reset_mask,
        )
        self._actor_observations = actor_obs
        self._critic_observations = critic_obs
        self._last_observation = observation

    def _extras(self) -> dict[str, Any]:
        _, critic_obs = self._ensure_observations()
        return {"observations": {"critic": critic_obs}}

    @property
    def total_env_steps(self) -> int:
        return self._total_env_steps

    @property
    def actor_observations(self) -> Tensor:
        """Return the actor observation that the next policy decision will see."""

        actor_obs, _ = self._ensure_observations()
        return actor_obs

    @property
    def env_step_rate_hz(self) -> float:
        return self._total_env_steps / self._elapsed_s if self._elapsed_s > 0.0 else 0.0

    def get_observations(self) -> tuple[Tensor, dict[str, Any]]:
        actor_obs, _ = self._ensure_observations()
        return actor_obs, self._extras()

    def reset(
        self, seeds: list[int] | tuple[int, ...] | None = None
    ) -> tuple[Tensor, dict[str, Any]]:
        observation = self.env.reset(seeds)
        self.episode_length_buf.zero_()
        reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._set_observations(observation, reset_mask=reset_mask)
        for value in (
            self._episode_return,
            self._episode_abs_heading_error,
            self._episode_abs_pitch_error,
            self._episode_abs_speed_error,
            self._episode_abs_roll,
            self._episode_material_saturation,
            self._episode_safety_events,
        ):
            value.zero_()
        actor_obs, _ = self._ensure_observations()
        return actor_obs, self._extras()

    def _update_logs(
        self,
        result: Any,
        rewards: Tensor,
        actions: Tensor,
        lengths: Tensor,
        dones: Tensor,
    ) -> dict[str, Tensor]:
        critic_state = result.observation.critic_state.to(dtype=torch.float32)
        heading_error = torch.abs(critic_state[:, 12])
        pitch_error = torch.abs(critic_state[:, 13])
        speed_error = torch.abs(critic_state[:, 14])
        roll = torch.abs(critic_state[:, 3])
        material_saturation = result.info["material_saturated"].to(dtype=torch.float32)
        terminal_kind = result.info["terminal_kind"]
        safety_kinds = torch.tensor(
            sorted(int(kind) for kind in PC_SAFETY_TERMINAL_KINDS),
            device=terminal_kind.device,
            dtype=terminal_kind.dtype,
        )
        safety = torch.isin(terminal_kind, safety_kinds).to(dtype=torch.float32) * dones.to(
            dtype=torch.float32
        )

        self._episode_return.add_(rewards)
        self._episode_abs_heading_error.add_(heading_error)
        self._episode_abs_pitch_error.add_(pitch_error)
        self._episode_abs_speed_error.add_(speed_error)
        self._episode_abs_roll.add_(roll)
        self._episode_material_saturation.add_(material_saturation)
        self._episode_safety_events.add_(safety)

        reset_count = dones.to(dtype=torch.float32).sum()
        denominator = torch.clamp(reset_count, min=1.0)
        completed_length = torch.where(
            dones, lengths.to(dtype=torch.float32), torch.zeros_like(lengths, dtype=torch.float32)
        )
        completed_return = torch.where(
            dones, self._episode_return, torch.zeros_like(self._episode_return)
        )
        completed_heading = torch.where(
            dones,
            self._episode_abs_heading_error,
            torch.zeros_like(self._episode_abs_heading_error),
        )
        completed_pitch = torch.where(
            dones,
            self._episode_abs_pitch_error,
            torch.zeros_like(self._episode_abs_pitch_error),
        )
        completed_speed = torch.where(
            dones,
            self._episode_abs_speed_error,
            torch.zeros_like(self._episode_abs_speed_error),
        )
        completed_roll = torch.where(
            dones, self._episode_abs_roll, torch.zeros_like(self._episode_abs_roll)
        )
        completed_material = torch.where(
            dones,
            self._episode_material_saturation,
            torch.zeros_like(self._episode_material_saturation),
        )
        completed_safety = torch.where(
            dones,
            self._episode_safety_events,
            torch.zeros_like(self._episode_safety_events),
        )
        step_denominator = torch.clamp(completed_length.sum(), min=1.0)
        goal_count = (
            ((terminal_kind == int(PCTerminalKind.GOAL)) & dones).to(dtype=torch.float32).sum()
        )
        timeout_count = result.truncated.to(dtype=torch.float32).sum()

        for value in (
            self._episode_return,
            self._episode_abs_heading_error,
            self._episode_abs_pitch_error,
            self._episode_abs_speed_error,
            self._episode_abs_roll,
            self._episode_material_saturation,
            self._episode_safety_events,
        ):
            value.masked_fill_(dones, 0.0)

        return {
            "/env/reset_count": reset_count,
            "/env/nonfinite_observation_count": (~torch.isfinite(result.observation.actor_state))
            .any(dim=1)
            .to(dtype=torch.float32)
            .sum(),
            "/env/nonfinite_action_count": (~torch.isfinite(actions))
            .any(dim=1)
            .to(dtype=torch.float32)
            .sum(),
            "/env/nonfinite_reward_count": (~torch.isfinite(rewards)).to(dtype=torch.float32).sum(),
            "/episode/success_rate": goal_count / denominator,
            "/episode/timeout_rate": timeout_count / denominator,
            "/episode/safety_termination_rate": completed_safety.sum() / denominator,
            "/episode/return_mean": completed_return.sum() / denominator,
            "/episode/length_mean": completed_length.sum() / denominator,
            "/episode/heading_abs_error_mean": completed_heading.sum() / step_denominator,
            "/episode/pitch_abs_error_mean": completed_pitch.sum() / step_denominator,
            "/episode/speed_abs_error_mean": completed_speed.sum() / step_denominator,
            "/episode/roll_abs_mean": completed_roll.sum() / step_denominator,
            "/episode/material_saturation_rate": completed_material.sum() / step_denominator,
        }

    def _reset_episode_logs(self, mask: Tensor) -> None:
        for value in (
            self._episode_return,
            self._episode_abs_heading_error,
            self._episode_abs_pitch_error,
            self._episode_abs_speed_error,
            self._episode_abs_roll,
            self._episode_material_saturation,
            self._episode_safety_events,
        ):
            value[mask] = 0
        self.episode_length_buf[mask] = 0

    def step(
        self, actions: Tensor, *, auto_reset: bool = True
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        if self.actor_decision_stride > 1:
            return self._step_repeated(actions, auto_reset=auto_reset)
        return self._step_once(actions, auto_reset=auto_reset)

    def step_split(
        self,
        base_residual: Tensor,
        ppo_residual: Tensor,
        *,
        gate: Tensor,
        ppo_rate_fraction: float,
        ppo_fin_gain: float,
        ppo_rpm_gain: float,
        auto_reset: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        inner = self.env
        if not hasattr(inner, "set_split_context"):
            raise TypeError("inner environment does not support split residuals")
        combined = torch.as_tensor(base_residual, device=self.device) + (
            torch.as_tensor(gate, device=self.device)[:, None]
            * torch.as_tensor(ppo_residual, device=self.device)
        )
        inner.set_split_context(
            base_residual_normalized=base_residual,
            ppo_residual_normalized=ppo_residual,
            gate=gate,
            ppo_rate_fraction=ppo_rate_fraction,
            ppo_fin_gain=ppo_fin_gain,
            ppo_rpm_gain=ppo_rpm_gain,
        )
        try:
            return self.step(combined, auto_reset=auto_reset)
        finally:
            inner.clear_split_context()

    def _step_repeated(
        self, actions: Tensor, *, auto_reset: bool
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        """Run one actor decision over ``actor_decision_stride`` physics steps.

        The inner environment is stepped for every physics step, so sensor,
        guidance, termination and actuator-rate state all advance at 50 Hz.
        A row that terminates early is reset only after the decision interval;
        its remaining physics steps are masked out of the decision reward and
        the first terminal metadata is preserved for PPO/time-out bootstrapping.
        """

        if not auto_reset:
            raise ValueError("actor decision stride > 1 requires auto_reset=True")
        started = perf_counter()
        action = torch.as_tensor(actions, device=self.device)
        if action.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"actions must have shape ({self.num_envs}, {self.num_actions})")
        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        done_any = torch.zeros_like(active)
        reward_sum = torch.zeros(self.num_envs, dtype=self.observation_dtype, device=self.device)
        reward_breakdown_sum: dict[str, Tensor] = {}
        last_infos: dict[str, Any] = {}
        terminal_fields = (
            "time_outs",
            "terminal_kind",
            "safety_event",
            "material_saturated",
            "actual_actuator",
            "hold_fraction",
        )
        first_terminal: dict[str, Tensor] = {}
        step_logs: list[dict[str, Tensor]] = []
        for _ in range(self.actor_decision_stride):
            if not bool(active.any()):
                break
            contributing = active.clone()
            _obs, reward, done, last_infos = self._step_once(action, auto_reset=False)
            reward_sum.add_(torch.where(contributing, reward, torch.zeros_like(reward)))
            for key, value in last_infos.items():
                if not key.startswith("reward_") or not torch.is_tensor(value):
                    continue
                contribution = torch.where(
                    contributing, value, torch.zeros_like(value)
                )
                if key in reward_breakdown_sum:
                    reward_breakdown_sum[key].add_(contribution)
                else:
                    reward_breakdown_sum[key] = contribution.clone()
            newly_done = done.to(dtype=torch.bool) & active
            if bool(newly_done.any()):
                for key in terminal_fields:
                    value = last_infos[key]
                    if key in first_terminal:
                        row_mask = newly_done[:, None] if value.ndim == 2 else newly_done
                        first_terminal[key] = torch.where(row_mask, value, first_terminal[key])
                    else:
                        first_terminal[key] = value.clone()
            if "log" in last_infos:
                step_logs.append(last_infos["log"])
            done_any |= newly_done
            active &= ~newly_done
        if bool(done_any.any()):
            self.env.reset_indices(done_any)
            self._set_observations(self.env.observation, reset_mask=done_any)
            self._reset_episode_logs(done_any)
        actor_obs, _ = self._ensure_observations()
        infos = dict(last_infos) if last_infos else {}
        for key, value in first_terminal.items():
            if key in infos and torch.is_tensor(value):
                current = infos[key]
                if torch.is_tensor(current) and current.shape == value.shape:
                    row_mask = done_any[:, None] if value.ndim == 2 else done_any
                    infos[key] = torch.where(row_mask, value, current)
                else:
                    infos[key] = value
            else:
                infos[key] = value
        infos.update(reward_breakdown_sum)
        if step_logs:
            infos["log"] = self._merge_decision_logs(step_logs)
        infos["observations"] = self._extras()["observations"]
        elapsed_s = perf_counter() - started
        self._elapsed_s += elapsed_s
        if self._record_step_timings:
            self._step_latencies_ms.append(1000.0 * elapsed_s)
        return (
            actor_obs,
            reward_sum,
            done_any.to(dtype=self.observation_dtype),
            infos,
        )

    @staticmethod
    def _merge_decision_logs(logs: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Merge per-physics-step episode logs at one decision boundary."""

        last = logs[-1]
        reset_counts = [torch.as_tensor(log.get("/env/reset_count", 0.0)) for log in logs]
        total_resets = torch.stack(reset_counts).sum()
        merged: dict[str, Tensor] = dict(last)
        merged["/env/reset_count"] = total_resets
        reset_weighted = {
            "/episode/success_rate",
            "/episode/timeout_rate",
            "/episode/safety_termination_rate",
            "/episode/return_mean",
            "/episode/length_mean",
        }
        for key in reset_weighted:
            if key not in last:
                continue
            weighted = torch.stack(
                [
                    torch.as_tensor(log[key]) * torch.as_tensor(count)
                    for log, count in zip(logs, reset_counts, strict=True)
                ]
            ).sum()
            merged[key] = weighted / torch.clamp(total_resets, min=1.0)
        for key in (
            "/env/nonfinite_observation_count",
            "/env/nonfinite_action_count",
            "/env/nonfinite_reward_count",
        ):
            if key in last:
                merged[key] = torch.stack(
                    [torch.as_tensor(log.get(key, 0.0)) for log in logs]
                ).sum()
        return merged

    def _step_once(
        self, actions: Tensor, *, auto_reset: bool = True
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        started = perf_counter()
        action = torch.as_tensor(actions, device=self.device)
        if action.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"actions must have shape ({self.num_envs}, {self.num_actions})")
        result = self.env.step(action, auto_reset=auto_reset)
        raw_rewards = result.reward.total.to(dtype=self.observation_dtype)
        rewards = result.reward.ppo_scaled.to(dtype=self.observation_dtype)
        dones = result.done
        lengths = self.episode_length_buf + 1
        self.episode_length_buf.add_(1)
        self.episode_length_buf.masked_fill_(dones, 0)
        reset_mask = dones
        self._set_observations(result.observation, reset_mask=reset_mask)
        log = self._update_logs(result, raw_rewards, action, lengths, dones)
        actor_obs, _ = self._ensure_observations()
        self._total_env_steps += self.num_envs
        elapsed_s = perf_counter() - started
        if self.actor_decision_stride == 1:
            self._elapsed_s += elapsed_s
            if self._record_step_timings:
                self._step_latencies_ms.append(1000.0 * elapsed_s)
        episode_cruise_speed = result.info["episode_command_speed_mps"]
        zero_speed = torch.zeros_like(episode_cruise_speed)
        one_speed = torch.ones_like(episode_cruise_speed)
        default_terminal_ratio = result.info["command_physical"][:, 2] / torch.clamp(
            episode_cruise_speed, min=1e-9
        )
        infos: dict[str, Any] = {
            **self._extras(),
            "time_outs": result.truncated.to(dtype=self.observation_dtype),
            "terminal_kind": result.info["terminal_kind"],
            "safety_event": result.info["safety_event"],
            "material_saturated": result.info["material_saturated"],
            "actual_actuator": result.info["actual_actuator"],
            "requested_action": result.info.get("requested_action", result.info["actual_actuator"]),
            "limited_action": result.info.get("limited_action", result.info["actual_actuator"]),
            "plant_scale_factors": result.info.get("plant_scale_factors"),
            "ground_route_speed_mps": result.info.get("ground_route_speed_mps", zero_speed),
            "ground_route_speed_ratio": result.info.get("ground_route_speed_ratio", zero_speed),
            "raw_reward": raw_rewards,
            "hold_fraction": result.info["hold_fraction"],
            "projection": result.info["projection"],
            "solution": result.info["solution"],
            "guidance_current_valid": result.info.get("guidance_current_valid", zero_speed),
            "guidance_current_age_s": result.info.get("guidance_current_age_s", zero_speed),
            "guidance_crab_angle_rad": result.info.get("guidance_crab_angle_rad", zero_speed),
            "guidance_speed_scale": result.info.get("guidance_speed_scale", one_speed),
            "guidance_terminal_phase": result.info.get("guidance_terminal_phase", zero_speed),
            "guidance_terminal_speed_ratio": result.info.get(
                "guidance_terminal_speed_ratio", default_terminal_ratio
            ),
            "guidance_saturation_ema": result.info.get("guidance_saturation_ema", zero_speed),
            "guidance_saturation_emergency": result.info.get(
                "guidance_saturation_emergency", zero_speed
            ),
            "guidance_route_speed_floor_active": result.info.get(
                "guidance_route_speed_floor_active", zero_speed
            ),
            "guidance_route_speed_floor_launch_active": result.info.get(
                "guidance_route_speed_floor_launch_active", zero_speed
            ),
            "guidance_route_speed_floor_valid": result.info.get(
                "guidance_route_speed_floor_valid", zero_speed
            ),
            "guidance_route_speed_floor_measured_mps": result.info.get(
                "guidance_route_speed_floor_measured_mps", zero_speed
            ),
            "guidance_route_speed_floor_filtered_mps": result.info.get(
                "guidance_route_speed_floor_filtered_mps", zero_speed
            ),
            "guidance_route_speed_floor_boost_mps": result.info.get(
                "guidance_route_speed_floor_boost_mps", zero_speed
            ),
            "guidance_route_speed_floor_target_mps": result.info.get(
                "guidance_route_speed_floor_target_mps", episode_cruise_speed
            ),
            "guidance_route_speed_floor_ratio": result.info.get(
                "guidance_route_speed_floor_ratio", one_speed
            ),
            "guidance_target_heading_rad": result.info.get(
                "guidance_target_heading_rad", result.info["command_physical"][:, 0]
            ),
            "guidance_target_pitch_rad": result.info.get(
                "guidance_target_pitch_rad", result.info["command_physical"][:, 1]
            ),
            "guidance_target_speed_mps": result.info.get(
                "guidance_target_speed_mps", result.info["command_physical"][:, 2]
            ),
            "guidance_limited_heading_rad": result.info.get(
                "guidance_limited_heading_rad", result.info["command_physical"][:, 0]
            ),
            "guidance_limited_pitch_rad": result.info.get(
                "guidance_limited_pitch_rad", result.info["command_physical"][:, 1]
            ),
            "guidance_limited_speed_mps": result.info.get(
                "guidance_limited_speed_mps", result.info["command_physical"][:, 2]
            ),
            "guidance_heading_rate_clipped": result.info.get(
                "guidance_heading_rate_clipped", zero_speed
            ),
            "guidance_pitch_rate_clipped": result.info.get(
                "guidance_pitch_rate_clipped", zero_speed
            ),
            "guidance_speed_rate_clipped": result.info.get(
                "guidance_speed_rate_clipped", zero_speed
            ),
            "episode_command_speed_mps": episode_cruise_speed,
            "log": log,
        }
        for field in fields(PCRewardBreakdown):
            value = getattr(result.reward, field.name)
            if value is not None:
                infos[f"reward_{field.name}"] = value
        for key in (
            "command_residual_applied",
            "command_residual_scale_clipped",
            "command_residual_rate_clipped",
            "command_base_physical",
            "command_corrected_physical",
            "actuator_residual_normalized",
            "actuator_residual_clipped_normalized",
            "actuator_residual_gate",
            "actuator_residual_gated_normalized",
            "actuator_residual_applied",
            "actuator_residual_effective",
            "actuator_base_physical",
            "actuator_requested_physical",
            "actuator_total_physical",
            "actuator_residual_scale_clipped",
            "actuator_residual_magnitude_clipped",
            "actuator_residual_rate_clipped",
            "actuator_residual_scale",
            "actuator_residual_schedule_distance_m",
            "actuator_residual_channel_gains",
            "actuator_residual_channel_gain_near_gate",
            "actuator_residual_bc_applied",
            "actuator_residual_bc_effective",
            "actuator_residual_ppo_applied",
            "actuator_residual_ppo_effective",
            "actuator_residual_bc_rate_clipped",
            "actuator_residual_ppo_rate_clipped",
        ):
            value = result.info.get(key)
            if value is not None:
                infos[key] = value
        return actor_obs, rewards, dones.to(dtype=self.observation_dtype), infos

    def drain_step_latency_stats(self) -> dict[str, Any]:
        if not self._record_step_timings:
            return {"mode": "disabled", "count": 0}
        samples = self._step_latencies_ms
        self._step_latencies_ms = []
        if not samples:
            return {"mode": "cpu_wall_clock", "count": 0}
        values = torch.tensor(samples, dtype=torch.float64)
        return {
            "mode": "cpu_wall_clock",
            "count": len(samples),
            "mean_ms": float(values.mean()),
            "p50_ms": float(values.quantile(0.5)),
            "p95_ms": float(values.quantile(0.95)),
        }

    def close(self) -> None:
        pass

    def __enter__(self) -> BatchTorchPCVecEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["BatchTorchPCVecEnv"]
