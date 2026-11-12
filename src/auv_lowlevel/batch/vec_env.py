"""rsl_rl-compatible adapter for the tensorized REMUS100 environment."""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from auv_lowlevel.config import LowLevelConfig

from .controllers import BatchedBaseController
from .env import BatchedLowLevelEnv, BatchedLowLevelStep
from .metrics import summarize_latencies_ms
from .observation import ObservationSchema
from .reward import TerminalKind


class BatchTorchVecEnv:
    """Synchronous GPU ``VecEnv`` backed by :class:`BatchedLowLevelEnv`.

    The class implements the rsl_rl 2.3.3 ``VecEnv`` contract without importing
    or modifying rsl_rl. Physics remains float64 by default while observations
    crossing the policy boundary are float32. The adapter never calls ``item``
    or performs a Python branch on a mask inside ``step``.
    """

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        num_envs: int,
        device: str | torch.device = "cpu",
        physics_dtype: torch.dtype = torch.float64,
        schema: ObservationSchema | None = None,
        base_controller: BatchedBaseController | None = None,
        expose_current: bool | None = None,
        projection_search_radius: int = 8,
        compile_step: bool = False,
        compile_mode: str = "default",
        randomizer_seed: int | None = None,
        validate: bool = False,
        record_step_timings: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if compile_mode != "default":
            raise ValueError("compile_mode must be 'default'")
        self.config = config
        self.num_envs = int(num_envs)
        self.num_actions = 3
        self.device = torch.device(device)
        self.physics_dtype = physics_dtype
        self.step_dt = float(config.plant.dt_s)
        self.max_episode_length = int(config.plant.max_episode_steps)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env = BatchedLowLevelEnv(
            config,
            num_envs=self.num_envs,
            device=self.device,
            dtype=physics_dtype,
            schema=schema,
            base_controller=base_controller,
            expose_current=expose_current,
            validate_shadow=validate,
            validate_finite=validate,
            projection_search_radius=projection_search_radius,
            randomizer_seed=randomizer_seed,
        )
        self.num_obs = self.env.observation_dim
        self.num_privileged_obs = None
        self.unwrapped = self
        self.cfg = {
            "backend": "batch_torch",
            "config_sha256": config.config_sha256,
            "num_envs": self.num_envs,
            "observation_schema": self.env.schema.version,
            "observation_dim": self.num_obs,
            "physics_dtype": str(physics_dtype).removeprefix("torch."),
            "projection_search_radius": int(projection_search_radius),
            "record_step_timings": bool(record_step_timings),
        }
        self._validate_actions = bool(validate)
        self._record_step_timings = bool(record_step_timings)
        self._step_latencies_ms: deque[float] = deque(maxlen=8192)
        self._step_timing_events: deque[tuple[torch.cuda.Event, torch.cuda.Event]] = deque(
            maxlen=8192
        )
        self._observations = self.env.reset().to(dtype=torch.float32)
        self._episode_return = self._zeros()
        self._episode_error_3d_sq = self._zeros()
        self._episode_horizontal_sq = self._zeros()
        self._episode_vertical_sq = self._zeros()
        self._episode_residual_abs = self._zeros()
        self._episode_material_saturation = self._zeros()
        self._episode_constraint_violations = self._zeros()
        self._episode_base_at_limit = self._zeros()
        self._total_env_steps = 0
        self._elapsed_s = 0.0
        self.compile_step = bool(compile_step)
        self.compile_mode = compile_mode
        self.cfg["compile_step"] = self.compile_step
        self.cfg["compile_mode"] = self.compile_mode
        self._compiled_step = (
            torch.compile(
                self.env.step,
                mode=self.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            if self.compile_step
            else None
        )

    def _zeros(self) -> Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    @property
    def total_env_steps(self) -> int:
        return self._total_env_steps

    @property
    def env_step_rate_hz(self) -> float:
        return self._total_env_steps / self._elapsed_s if self._elapsed_s > 0.0 else 0.0

    @property
    def current_ned_mps(self) -> Tensor | None:
        """Return the true current for an explicitly privileged critic path.

        The base ``VecEnv`` contract does not expose this value. Callers must
        opt in through a wrapper that places it only in ``observations["critic"]``;
        it must never be appended to the actor observation.
        """

        return self.env.current_ned_mps

    def get_observations(self) -> tuple[Tensor, dict[str, Any]]:
        return self._observations, {"observations": {}}

    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        self._observations = self.env.reset().to(dtype=torch.float32)
        self.episode_length_buf.zero_()
        for value in (
            self._episode_return,
            self._episode_error_3d_sq,
            self._episode_horizontal_sq,
            self._episode_vertical_sq,
            self._episode_residual_abs,
            self._episode_material_saturation,
            self._episode_constraint_violations,
            self._episode_base_at_limit,
        ):
            value.zero_()
        return self.get_observations()

    def step(
        self,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        started = perf_counter()
        start_event: torch.cuda.Event | None = None
        end_event: torch.cuda.Event | None = None
        if self._record_step_timings and self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        actions_t = torch.as_tensor(actions, device=self.device, dtype=self.physics_dtype)
        expected_shape = (self.num_envs, self.num_actions)
        if tuple(actions_t.shape) != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {tuple(actions_t.shape)}"
            )
        if self._validate_actions and not bool(torch.isfinite(actions_t).all()):
            raise ValueError("actions contain NaN or Inf")

        result = (
            self.env.step(actions_t, auto_reset=True)
            if self._compiled_step is None
            else self._compiled_step(actions_t, auto_reset=True)
        )
        observations = result.observation.to(dtype=torch.float32)
        raw_rewards = result.reward.total.to(dtype=torch.float32)
        rewards = result.reward.ppo_scaled.to(dtype=torch.float32)
        dones = result.done
        dones_f = dones.to(dtype=torch.float32)

        lengths = self.episode_length_buf + 1
        self.episode_length_buf.add_(1)
        self.episode_length_buf.masked_fill_(dones, 0)
        log = self._update_logs(result, raw_rewards, actions_t, lengths, dones)

        self._observations = observations
        self._total_env_steps += self.num_envs
        elapsed_s = perf_counter() - started
        self._elapsed_s += elapsed_s
        if self._record_step_timings:
            if start_event is not None and end_event is not None:
                end_event.record(torch.cuda.current_stream(self.device))
                self._step_timing_events.append((start_event, end_event))
            else:
                self._step_latencies_ms.append(1000.0 * elapsed_s)
        infos: dict[str, Any] = {
            "observations": {},
            "time_outs": result.truncated.to(dtype=torch.float32),
            "log": log,
        }
        return observations, rewards, dones_f, infos

    def _update_logs(
        self,
        result: BatchedLowLevelStep,
        rewards: Tensor,
        actions: Tensor,
        lengths: Tensor,
        dones: Tensor,
    ) -> dict[str, Tensor]:
        projection = result.info["projection_after"]
        error_3d = projection.error_3d_m.to(dtype=torch.float32)
        horizontal = projection.horizontal_error_m.to(dtype=torch.float32)
        vertical = projection.vertical_error_m.to(dtype=torch.float32)
        residual = result.info["residual_normalized"].to(dtype=torch.float32)
        residual_abs = torch.mean(torch.abs(residual), dim=1)
        material_saturation = result.info["material_saturated"].to(dtype=torch.float32)
        constraint_violations = result.info["constraint_violation_count"].to(dtype=torch.float32)
        base = result.info["base_command"]
        max_fin = self.env.limits.max_rudder_angle_rad - 1e-6
        max_stern = self.env.limits.max_stern_plane_angle_rad - 1e-6
        max_rpm = self.env.limits.max_propeller_rpm - 1e-3
        base_at_limit = (
            (torch.abs(base[:, 0]) >= max_fin)
            | (torch.abs(base[:, 1]) >= max_stern)
            | (base[:, 2] <= 1e-3)
            | (base[:, 2] >= max_rpm)
        ).to(dtype=torch.float32)

        self._episode_return.add_(rewards)
        self._episode_error_3d_sq.add_(torch.square(error_3d))
        self._episode_horizontal_sq.add_(torch.square(horizontal))
        self._episode_vertical_sq.add_(torch.square(vertical))
        self._episode_residual_abs.add_(residual_abs)
        self._episode_material_saturation.add_(material_saturation)
        self._episode_constraint_violations.add_(constraint_violations)
        self._episode_base_at_limit.add_(base_at_limit)

        reset_count = dones.to(dtype=torch.float32).sum()
        denominator = torch.clamp(reset_count, min=1.0)
        completed_length = torch.where(
            dones, lengths.to(torch.float32), torch.zeros_like(lengths, dtype=torch.float32)
        )
        completed_return = torch.where(
            dones, self._episode_return, torch.zeros_like(self._episode_return)
        )
        completed_error_sq = torch.where(
            dones, self._episode_error_3d_sq, torch.zeros_like(self._episode_error_3d_sq)
        )
        completed_horizontal_sq = torch.where(
            dones,
            self._episode_horizontal_sq,
            torch.zeros_like(self._episode_horizontal_sq),
        )
        completed_vertical_sq = torch.where(
            dones, self._episode_vertical_sq, torch.zeros_like(self._episode_vertical_sq)
        )
        completed_residual_abs = torch.where(
            dones, self._episode_residual_abs, torch.zeros_like(self._episode_residual_abs)
        )
        completed_material = torch.where(
            dones,
            self._episode_material_saturation,
            torch.zeros_like(self._episode_material_saturation),
        )
        completed_constraints = torch.where(
            dones,
            self._episode_constraint_violations,
            torch.zeros_like(self._episode_constraint_violations),
        )
        completed_base_at_limit = torch.where(
            dones,
            self._episode_base_at_limit,
            torch.zeros_like(self._episode_base_at_limit),
        )
        terminal_kind = result.info["terminal_kind"]
        goal_count = ((terminal_kind == TerminalKind.GOAL) & dones).to(torch.float32).sum()
        timeout_count = result.truncated.to(torch.float32).sum()
        out_of_bounds_count = (
            ((terminal_kind == TerminalKind.OUT_OF_BOUNDS) & dones).to(torch.float32).sum()
        )
        invalid_count = ((terminal_kind == TerminalKind.INVALID) & dones).to(torch.float32).sum()

        mean_length = completed_length.sum() / denominator
        step_denominator = torch.clamp(completed_length.sum(), min=1.0)
        mean_error_sq = completed_error_sq.sum() / step_denominator
        mean_horizontal_sq = completed_horizontal_sq.sum() / step_denominator
        mean_vertical_sq = completed_vertical_sq.sum() / step_denominator

        for value in (
            self._episode_return,
            self._episode_error_3d_sq,
            self._episode_horizontal_sq,
            self._episode_vertical_sq,
            self._episode_residual_abs,
            self._episode_material_saturation,
            self._episode_constraint_violations,
            self._episode_base_at_limit,
        ):
            value.masked_fill_(dones, 0.0)

        return {
            "/env/reset_count": reset_count,
            "/env/nonfinite_observation_count": (~torch.isfinite(result.observation))
            .any(dim=1)
            .to(torch.float32)
            .sum(),
            "/env/nonfinite_action_count": (~torch.isfinite(actions))
            .any(dim=1)
            .to(torch.float32)
            .sum(),
            "/env/nonfinite_reward_count": (~torch.isfinite(rewards)).to(dtype=torch.float32).sum(),
            "/episode/success_rate": goal_count / denominator,
            "/episode/timeout_rate": timeout_count / denominator,
            "/episode/out_of_bounds_rate": out_of_bounds_count / denominator,
            "/episode/invalid_rate": invalid_count / denominator,
            "/episode/return_mean": completed_return.sum() / denominator,
            "/episode/length_mean": mean_length,
            "/episode/three_d_rmse_mean": torch.sqrt(mean_error_sq),
            "/episode/horizontal_rmse_mean": torch.sqrt(mean_horizontal_sq),
            "/episode/vertical_rmse_mean": torch.sqrt(mean_vertical_sq),
            "/episode/residual_abs_mean": completed_residual_abs.sum() / step_denominator,
            "/episode/material_saturation_rate": completed_material.sum() / step_denominator,
            "/episode/constraint_violations_mean": completed_constraints.sum() / step_denominator,
            "/episode/base_at_limit_rate": completed_base_at_limit.sum() / step_denominator,
        }

    def drain_step_latency_stats(self) -> dict[str, float | int | str]:
        """Return and clear the current step-timing window.

        CUDA timings use events and synchronize only when this method is called,
        not inside the rollout loop. CPU timings are wall-clock measurements.
        """

        if not self._record_step_timings:
            return {"mode": "disabled", **summarize_latencies_ms(())}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            samples = [start.elapsed_time(end) for start, end in self._step_timing_events]
            self._step_timing_events.clear()
            return {"mode": "cuda_events", **summarize_latencies_ms(samples)}
        samples = list(self._step_latencies_ms)
        self._step_latencies_ms.clear()
        return {"mode": "cpu_wall_clock", **summarize_latencies_ms(samples)}

    def close(self) -> None:
        """Keep the rsl_rl lifecycle contract; the tensor backend owns no workers."""

    def __enter__(self) -> BatchTorchVecEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["BatchTorchVecEnv"]
