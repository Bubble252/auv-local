"""Environment-side composition for frozen-BC PPO residual policies.

The policy action is a normalized residual.  This wrapper adds it to the frozen
BC action with an optional terminal gate, then sends the resulting absolute
normalized actuator action to the ordinary PC vector environment.  Keeping the
composition outside the policy makes the PPO action the residual itself, while
the executed action can still be logged and audited step by step.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from .pc_command_residual import smooth_terminal_gate
from .pc_vec_env import BatchTorchPCVecEnv


class BatchTorchPCFrozenBCResidualVecEnv:
    """Compose ``frozen_BC + gate * residual`` before the absolute PC env step."""

    def __init__(
        self,
        env: BatchTorchPCVecEnv,
        *,
        base_action_fn: Callable[[Tensor], Tensor] | None = None,
        terminal_gate_enabled: bool = False,
        terminal_gate_enter_m: float = 20.0,
        terminal_gate_full_m: float = 5.0,
        residual_clip_min: float = -1.0,
        residual_clip_max: float = 1.0,
        rate_aware_residual_enabled: bool = False,
        ppo_rate_fraction: float = 0.2,
        ppo_fin_gain: float = 1.0,
        ppo_rpm_gain: float = 1.0,
        base_residual_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if not math.isfinite(terminal_gate_enter_m) or terminal_gate_enter_m <= 0.0:
            raise ValueError("terminal_gate_enter_m must be finite and positive")
        if (
            not math.isfinite(terminal_gate_full_m)
            or terminal_gate_full_m <= 0.0
            or terminal_gate_full_m >= terminal_gate_enter_m
        ):
            raise ValueError(
                "terminal_gate_full_m must be finite, positive, and less than enter_m"
            )
        if (
            not math.isfinite(residual_clip_min)
            or not math.isfinite(residual_clip_max)
            or residual_clip_min >= residual_clip_max
        ):
            raise ValueError("residual clip bounds must be finite and increasing")
        if not hasattr(env, "actor_observations"):
            raise TypeError("frozen residual wrapper requires a BatchTorchPCVecEnv")
        if not math.isfinite(ppo_rate_fraction) or not 0.0 < ppo_rate_fraction < 1.0:
            raise ValueError("ppo_rate_fraction must be finite and in (0, 1)")
        for name, value in (("ppo_fin_gain", ppo_fin_gain), ("ppo_rpm_gain", ppo_rpm_gain)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if len(base_residual_bias) != 3 or any(
            not math.isfinite(float(value)) for value in base_residual_bias
        ):
            raise ValueError("base_residual_bias must contain three finite numbers")
        if rate_aware_residual_enabled and not hasattr(env, "step_split"):
            raise TypeError("rate-aware residuals require BatchTorchPCVecEnv.step_split")

        self.env = env
        self.base_action_fn = base_action_fn
        self.terminal_gate_enabled = bool(terminal_gate_enabled)
        self.terminal_gate_enter_m = float(terminal_gate_enter_m)
        self.terminal_gate_full_m = float(terminal_gate_full_m)
        self.residual_clip_min = float(residual_clip_min)
        self.residual_clip_max = float(residual_clip_max)
        self.rate_aware_residual_enabled = bool(rate_aware_residual_enabled)
        self.ppo_rate_fraction = float(ppo_rate_fraction)
        self.ppo_fin_gain = float(ppo_fin_gain)
        self.ppo_rpm_gain = float(ppo_rpm_gain)
        self.base_residual_bias = torch.tensor(
            base_residual_bias, device=env.device, dtype=torch.float32
        )
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.num_obs = env.num_obs
        self.num_privileged_obs = env.num_privileged_obs
        self.device = env.device
        self.step_dt = env.step_dt
        self.max_episode_length = env.max_episode_length

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def set_base_action_fn(self, base_action_fn: Callable[[Tensor], Tensor]) -> None:
        self.base_action_fn = base_action_fn

    def set_terminal_gate(
        self,
        *,
        enabled: bool,
        enter_m: float | None = None,
        full_m: float | None = None,
    ) -> None:
        enter = self.terminal_gate_enter_m if enter_m is None else float(enter_m)
        full = self.terminal_gate_full_m if full_m is None else float(full_m)
        if not math.isfinite(enter) or enter <= 0.0:
            raise ValueError("terminal gate enter_m must be finite and positive")
        if not math.isfinite(full) or full <= 0.0 or full >= enter:
            raise ValueError("terminal gate full_m must be positive and below enter_m")
        self.terminal_gate_enabled = bool(enabled)
        self.terminal_gate_enter_m = enter
        self.terminal_gate_full_m = full

    def _gate(self) -> Tensor:
        if not self.terminal_gate_enabled:
            return torch.ones(self.num_envs, device=self.device)
        inner = self.env.env
        observation = inner.observation
        position = observation.actor_state[:, :3].to(
            device=self.device, dtype=torch.float32
        )
        goal = inner.path.goal_ned_m_for(self.num_envs).to(
            device=self.device, dtype=torch.float32
        )
        distance = torch.linalg.norm(goal - position, dim=1)
        return smooth_terminal_gate(
            distance,
            enter_m=self.terminal_gate_enter_m,
            full_m=self.terminal_gate_full_m,
        )

    def step(
        self, residual_action: Tensor, *, auto_reset: bool = True
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        if self.base_action_fn is None:
            raise RuntimeError(
                "frozen residual base action provider is not configured; "
                "call set_base_action_fn() before stepping"
            )
        residual = torch.as_tensor(residual_action, device=self.device, dtype=torch.float32)
        expected = (self.num_envs, self.num_actions)
        if tuple(residual.shape) != expected:
            raise ValueError(
                f"frozen BC residual must have shape {expected}, got {tuple(residual.shape)}"
            )
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("frozen BC residual contains NaN or Inf")

        observation = self.env.actor_observations
        with torch.no_grad():
            base = self.base_action_fn(observation)
        if base is None:
            raise RuntimeError("frozen BC base action provider returned None")
        base = torch.as_tensor(base, device=self.device, dtype=torch.float32)
        if tuple(base.shape) != expected:
            raise ValueError(
                f"frozen BC base action must have shape {expected}, got {tuple(base.shape)}"
            )

        gate = self._gate().to(device=self.device, dtype=torch.float32)
        gain = torch.tensor(
            [self.ppo_fin_gain, self.ppo_fin_gain, self.ppo_rpm_gain],
            device=self.device,
            dtype=torch.float32,
        )
        residual_with_bias = residual + self.base_residual_bias[None, :]
        ppo_term = gate[:, None] * residual_with_bias * gain[None, :]
        requested = base + ppo_term
        if self.rate_aware_residual_enabled:
            observations, reward, done, infos = self.env.step_split(
                base,
                residual_with_bias,
                gate=gate,
                ppo_rate_fraction=self.ppo_rate_fraction,
                ppo_fin_gain=self.ppo_fin_gain,
                ppo_rpm_gain=self.ppo_rpm_gain,
                auto_reset=auto_reset,
            )
            executed = torch.clamp(
                requested, min=self.residual_clip_min, max=self.residual_clip_max
            )
            clipped = torch.zeros_like(requested, dtype=torch.bool)
        else:
            executed = torch.clamp(
                requested, min=self.residual_clip_min, max=self.residual_clip_max
            )
            clipped = torch.abs(executed - requested) > 1e-6
            observations, reward, done, infos = self.env.step(
                executed, auto_reset=auto_reset
            )
        infos.update(
            {
                "frozen_residual_base_action": base,
                "frozen_residual_action": residual,
                "frozen_residual_gate": gate,
                "frozen_residual_ppo_gain": gain,
                "frozen_residual_ppo_term": ppo_term,
                "frozen_residual_requested_action": requested,
                "frozen_residual_executed_action": executed,
                "frozen_residual_clipped": clipped,
                "frozen_residual_scale": torch.full(
                    (self.num_envs,), 1.0, device=self.device, dtype=torch.float32
                ),
            }
        )
        return observations, reward, done, infos


__all__ = ["BatchTorchPCFrozenBCResidualVecEnv"]
