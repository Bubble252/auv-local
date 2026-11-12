"""Bounded H3 controller-gain residual wrapper (Goal162).

The action is a two-dimensional normalized vector ``[a_heading, a_pitch]``.
It is mapped to PD-gain deltas with a hard ``tanh`` bound and an optional
near-terminal smooth gate:

``heading_gain = base_heading_gain + heading_delta * tanh(a_heading) * gate``
``pitch_gain   = base_pitch_gain   + pitch_delta   * tanh(a_pitch)   * gate``

With a zero action and/or a zero gate, the wrapper is exactly the H3 controller
path. The shared ``ControllerConfig`` is never mutated.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from .controllers import (
    BatchedControlContext,
    BatchedRemus100Teacher,
    controller_state_from_actor_state,
)
from .geometry import quaternion_from_direction, quaternion_from_euler
from .pc_command_residual import smooth_terminal_gate
from .pc_env import BatchedPCLowLevelEnv, BatchedPCLowLevelStep


class BatchedPCControllerGainResidualEnv:
    """Expose bounded heading/pitch controller-gain residuals as actions."""

    def __init__(
        self,
        env: BatchedPCLowLevelEnv,
        *,
        heading_gain_delta: float = 1.0,
        pitch_gain_delta: float = 1.5,
        controller_state: str = "actor",
        terminal_gate_enabled: bool = True,
        terminal_gate_enter_m: float = 10.0,
        terminal_gate_full_m: float = 2.0,
    ) -> None:
        for name, value in (
            ("heading_gain_delta", heading_gain_delta),
            ("pitch_gain_delta", pitch_gain_delta),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if controller_state not in {"actor", "privileged"}:
            raise ValueError("controller_state must be 'actor' or 'privileged'")
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

        self.env = env
        self.controller_state = controller_state
        self.terminal_gate_enabled = bool(terminal_gate_enabled)
        self.terminal_gate_enter_m = float(terminal_gate_enter_m)
        self.terminal_gate_full_m = float(terminal_gate_full_m)
        self.delta_scale = torch.tensor(
            [heading_gain_delta, pitch_gain_delta],
            device=env.device,
            dtype=env.dtype,
        )
        self.base_gains = torch.tensor(
            [
                float(env.config.controller.heading_gain),
                float(env.config.controller.pitch_gain),
            ],
            device=env.device,
            dtype=env.dtype,
        )
        self.controller = BatchedRemus100Teacher(
            env.config.controller,
            env.limits,
            device=env.device,
            dtype=env.dtype,
        )
        self.controller.reset(count=env.num_envs)
        self.num_envs = env.num_envs
        self.action_dim = 2
        self.num_actions = 2
        self.device = env.device
        self.dt_s = env.dt_s
        self.dtype = env.dtype
        self.stage = env.stage
        self.config = env.config
        self.actor_state_dim = env.actor_state_dim
        self.actor_path_dim = env.actor_path_dim
        self.critic_state_dim = env.critic_state_dim
        self.path_blocks = env.path_blocks
        self.vehicle = env.vehicle
        self.limits = env.limits
        self._previous_delta = torch.zeros(
            (env.num_envs, 2), device=env.device, dtype=env.dtype
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def observation(self) -> Any:
        return self.env.observation

    def _terminal_gate(self) -> Tensor:
        observation = self.env.observation
        position = observation.actor_state[:, :3].to(
            device=self.env.device, dtype=self.env.dtype
        )
        goal = self.env.path.goal_ned_m_for(self.env.num_envs)
        distance = torch.linalg.norm(goal - position, dim=1)
        return smooth_terminal_gate(
            distance,
            enter_m=self.terminal_gate_enter_m,
            full_m=self.terminal_gate_full_m,
        )

    def _bounded_delta(
        self,
        residual_normalized: Tensor,
        *,
        active_mask: Tensor,
        gate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        bounded = torch.tanh(residual_normalized)
        candidate = bounded * gate[:, None] * self.delta_scale
        delta = torch.where(active_mask[:, None], candidate, self._previous_delta)
        self._previous_delta = delta
        return delta, gate

    def _deployable_action(self, gains: Tensor, actor_state: Tensor) -> Tensor:
        observation = self.env.observation
        controller_water_surge = (
            observation.water_velocity_body_mps[:, 0]
            if self.env.config.plant.velocity_contract == "nu_ground"
            else None
        )
        heading = observation.command.heading_rad.to(
            device=self.env.device, dtype=self.env.dtype
        )
        pitch = observation.command.pitch_rad.to(
            device=self.env.device, dtype=self.env.dtype
        )
        speed = observation.command.speed_mps.to(
            device=self.env.device, dtype=self.env.dtype
        )
        direction = torch.stack(
            (
                torch.cos(pitch) * torch.cos(heading),
                torch.cos(pitch) * torch.sin(heading),
                -torch.sin(pitch),
            ),
            dim=1,
        )
        reference_quat = quaternion_from_direction(direction)
        if self.controller_state == "privileged":
            eta = self.env.eta
            nu = self.env.nu
            current_ned_mps = self.env.current_ned_mps
            if eta is None or nu is None or current_ned_mps is None:
                raise RuntimeError(
                    "privileged controller state is unavailable before reset"
                )
            controller_eta = eta
            controller_nu = nu
            controller_quat = quaternion_from_euler(
                eta[:, 3], eta[:, 4], eta[:, 5]
            )
            controller_current = current_ned_mps
        else:
            if self.env.config.plant.velocity_contract == "nu_ground":
                controller_eta, controller_nu = controller_state_from_actor_state(
                    actor_state
                )
            else:
                controller_eta = actor_state[:, :6]
                controller_nu = actor_state[:, 6:12]
            controller_quat = reference_quat
            controller_current = torch.zeros_like(actor_state[:, :3])
        context = BatchedControlContext(
            eta=controller_eta,
            nu=controller_nu,
            quat_xyzw=controller_quat,
            current_ned_mps=controller_current,
            reference_quat_xyzw=reference_quat,
            command_speed_mps=speed,
            observation=actor_state,
            dt_s=float(self.env.dt_s),
            heading_gain=gains[:, 0],
            pitch_gain=gains[:, 1],
            water_surge_mps=controller_water_surge,
        )
        actuator = self.controller.control(context)
        action = torch.empty_like(actuator)
        action[:, 0] = actuator[:, 0] / float(
            self.env.limits.max_rudder_angle_rad
        )
        action[:, 1] = actuator[:, 1] / float(
            self.env.limits.max_stern_plane_angle_rad
        )
        action[:, 2] = (
            2.0 * actuator[:, 2] / float(self.env.limits.max_propeller_rpm) - 1.0
        )
        return action

    def step(
        self,
        residual_normalized: Tensor,
        *,
        auto_reset: bool = True,
        advance_mask: Tensor | None = None,
    ) -> BatchedPCLowLevelStep:
        residual = torch.as_tensor(
            residual_normalized, device=self.env.device, dtype=self.env.dtype
        )
        expected = (self.env.num_envs, 2)
        if tuple(residual.shape) != expected:
            raise ValueError(
                f"controller-gain residual must have shape {expected}, "
                f"got {tuple(residual.shape)}"
            )
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("controller-gain residual contains NaN or Inf")
        if advance_mask is None:
            active = torch.ones(
                self.env.num_envs, device=self.env.device, dtype=torch.bool
            )
        else:
            active = torch.as_tensor(
                advance_mask, device=self.env.device, dtype=torch.bool
            )
            if active.shape != (self.env.num_envs,):
                raise ValueError(
                    f"advance_mask must have shape ({self.env.num_envs},)"
                )
        gate = (
            self._terminal_gate()
            if self.terminal_gate_enabled
            else torch.ones_like(residual[:, 0])
        )
        delta, gate = self._bounded_delta(
            residual, active_mask=active, gate=gate
        )
        effective = self.base_gains[None, :] + delta
        action = self._deployable_action(effective, self.env.observation.actor_state)
        result = self.env.step(
            action,
            auto_reset=auto_reset,
            advance_mask=advance_mask,
        )
        done = result.terminated | result.truncated
        if auto_reset and bool(done.any()):
            self.controller.reset(mask=done)
            self._previous_delta = torch.where(
                done[:, None],
                torch.zeros_like(self._previous_delta),
                self._previous_delta,
            )
        info = dict(result.info)
        info.update(
            {
                "controller_gain_residual_normalized": residual,
                "controller_gain_residual_gate": gate,
                "controller_gain_residual_applied": delta,
                "controller_gain_base": self.base_gains[None, :].expand_as(effective),
                "controller_gain_effective": effective,
                "controller_gain_residual_actuator": action,
            }
        )
        return replace(result, info=info)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observation = self.env.reset(*args, **kwargs)
        self.controller.reset(count=self.env.num_envs)
        self._previous_delta = torch.zeros_like(self._previous_delta)
        return observation

    def reset_indices(self, mask: Tensor, *args: Any, **kwargs: Any) -> Any:
        observation = self.env.reset_indices(mask, *args, **kwargs)
        reset_mask = torch.as_tensor(mask, device=self.env.device, dtype=torch.bool)
        self.controller.reset(mask=reset_mask)
        self._previous_delta = torch.where(
            reset_mask[:, None],
            torch.zeros_like(self._previous_delta),
            self._previous_delta,
        )
        return observation


__all__ = ["BatchedPCControllerGainResidualEnv"]
