"""Optional ALOS command-space residual wrapper (Goal27 C1).

The wrapper is deliberately separate from :class:`BatchedPCLowLevelEnv` so the
existing actuator-action path stays byte-for-byte unchanged when the feature is
disabled. With zero residual it reproduces the deployable B0 controller:
``ALOS command -> PD/PI -> [rudder, stern, rpm]``.
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
from .geometry import quaternion_from_direction, quaternion_from_euler, wrap_angle
from .pc_env import BatchedPCLowLevelEnv, BatchedPCLowLevelStep


def _wrap_pi(value: Tensor) -> Tensor:
    return wrap_angle(value)


def smooth_terminal_gate(distance_m: Tensor, *, enter_m: float, full_m: float) -> Tensor:
    """Return 0 outside ``enter_m`` and 1 inside ``full_m`` with smoothstep."""

    if not math.isfinite(enter_m) or enter_m <= 0.0:
        raise ValueError("enter_m must be finite and positive")
    if not math.isfinite(full_m) or full_m <= 0.0 or full_m >= enter_m:
        raise ValueError("full_m must be finite, positive, and less than enter_m")
    progress = torch.clamp(
        (float(enter_m) - distance_m) / (float(enter_m) - float(full_m)),
        0.0,
        1.0,
    )
    return progress * progress * (3.0 - 2.0 * progress)


class BatchedPCCommandResidualEnv:
    """Expose 3-D ALOS command residuals as the environment action space."""

    def __init__(
        self,
        env: BatchedPCLowLevelEnv,
        *,
        scale_heading_rad: float = 0.35,
        scale_pitch_rad: float = 0.17,
        scale_speed_mps: float = 0.60,
        rate_heading_rad_s: float = 0.25,
        rate_pitch_rad_s: float = 0.15,
        rate_speed_mps2: float = 0.80,
        controller_state: str = "actor",
        terminal_gate_enabled: bool = False,
        terminal_gate_enter_m: float = 20.0,
        terminal_gate_full_m: float = 5.0,
    ) -> None:
        values = (
            scale_heading_rad,
            scale_pitch_rad,
            scale_speed_mps,
            rate_heading_rad_s,
            rate_pitch_rad_s,
            rate_speed_mps2,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("command-residual scales and rates must be finite and positive")
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
        self.controller_state = controller_state
        self.terminal_gate_enabled = bool(terminal_gate_enabled)
        self.terminal_gate_enter_m = float(terminal_gate_enter_m)
        self.terminal_gate_full_m = float(terminal_gate_full_m)
        self.env = env
        self.scale = torch.tensor(
            [scale_heading_rad, scale_pitch_rad, scale_speed_mps],
            device=env.device,
            dtype=env.dtype,
        )
        self.rate_limit = torch.tensor(
            [rate_heading_rad_s, rate_pitch_rad_s, rate_speed_mps2],
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
        self.action_dim = env.action_dim
        self.num_actions = env.action_dim
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
            (env.num_envs, 3), device=env.device, dtype=env.dtype
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def observation(self) -> Any:
        return self.env.observation

    def _minimum_speed(self, base_speed: Tensor) -> Tensor:
        guidance = self.config.guidance
        minimum_ratio = float(guidance.minimum_speed_ratio)
        vehicle_minimum = float(self.vehicle.min_speed_mps)
        cruise = getattr(self.env, "_episode_command_speed_mps", base_speed)
        ratio_floor = minimum_ratio * torch.as_tensor(
            cruise, device=self.env.device, dtype=self.env.dtype
        )
        return torch.maximum(
            torch.full_like(base_speed, vehicle_minimum),
            ratio_floor,
        )

    def _terminal_gate(self) -> Tensor:
        """Gate residuals by deployable navigation distance to the goal."""

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

    def _corrected_command(
        self,
        residual_normalized: Tensor,
        *,
        active_mask: Tensor | None = None,
        gate: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        observation = self.env.observation
        base = torch.stack(
            (
                observation.command.heading_rad,
                observation.command.pitch_rad,
                observation.command.speed_mps,
            ),
            dim=1,
        ).to(device=self.env.device, dtype=self.env.dtype)
        if gate is None:
            gate = torch.ones_like(residual_normalized[:, 0])
        gated_residual = residual_normalized * gate[:, None]
        scaled = gated_residual * self.scale
        raw = torch.clamp(scaled, -self.scale, self.scale)
        scale_clipped = torch.abs(raw - scaled) > 1e-6
        previous = self._previous_delta
        max_step = self.rate_limit * float(self.env.dt_s)
        rate_candidate = previous + torch.clamp(raw - previous, -max_step, max_step)
        rate_clipped = torch.abs(rate_candidate - raw) > 1e-6
        delta = torch.clamp(rate_candidate, -self.scale, self.scale)
        if active_mask is not None:
            delta = torch.where(active_mask[:, None], delta, previous)
            scale_clipped = torch.where(
                active_mask[:, None], scale_clipped, torch.zeros_like(scale_clipped)
            )
            rate_clipped = torch.where(
                active_mask[:, None], rate_clipped, torch.zeros_like(rate_clipped)
            )

        heading = _wrap_pi(base[:, 0] + delta[:, 0])
        pitch = torch.clamp(
            base[:, 1] + delta[:, 1],
            -0.5 * math.pi + 1e-3,
            0.5 * math.pi - 1e-3,
        )
        speed_floor = self._minimum_speed(base[:, 2])
        speed = torch.clamp(
            base[:, 2] + delta[:, 2],
            min=speed_floor,
            max=torch.full_like(base[:, 2], float(self.vehicle.max_speed_mps)),
        )
        corrected = torch.stack((heading, pitch, speed), dim=1)
        # Preserve exact B0 identity when the residual is exactly zero: wrapping
        # or clamping an already-valid command would introduce 1e-17 rounding.
        corrected = torch.where(
            torch.all(delta == 0.0, dim=1, keepdim=True), base, corrected
        )
        self._previous_delta = delta
        return corrected, delta, base, scale_clipped, rate_clipped, gate

    def _deployable_action(self, corrected: Tensor, actor_state: Tensor) -> Tensor:
        observation = self.env.observation
        controller_water_surge = (
            observation.water_velocity_body_mps[:, 0]
            if self.env.config.plant.velocity_contract == "nu_ground"
            else None
        )
        heading = corrected[:, 0]
        pitch = corrected[:, 1]
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
                raise RuntimeError("privileged controller state is unavailable before reset")
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
            command_speed_mps=corrected[:, 2],
            observation=actor_state,
            dt_s=float(self.env.dt_s),
            water_surge_mps=controller_water_surge,
        )
        actuator = self.controller.control(context)
        action = torch.empty_like(actuator)
        action[:, 0] = actuator[:, 0] / float(self.env.limits.max_rudder_angle_rad)
        action[:, 1] = actuator[:, 1] / float(self.env.limits.max_stern_plane_angle_rad)
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
        expected = (self.env.num_envs, 3)
        if tuple(residual.shape) != expected:
            raise ValueError(
                f"command residual must have shape {expected}, got {tuple(residual.shape)}"
            )
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("command residual contains NaN or Inf")
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
        gate = self._terminal_gate() if self.terminal_gate_enabled else torch.ones_like(
            residual[:, 0]
        )
        corrected, delta, base, scale_clipped, rate_clipped, gate = self._corrected_command(
            residual, active_mask=active, gate=gate
        )
        action = self._deployable_action(corrected, self.env.observation.actor_state)
        result = self.env.step(
            action,
            auto_reset=auto_reset,
            advance_mask=advance_mask,
        )
        done = result.terminated | result.truncated
        if auto_reset and bool(done.any()):
            self.controller.reset(mask=done)
            self._previous_delta = torch.where(
                done[:, None], torch.zeros_like(self._previous_delta), self._previous_delta
            )
        info = dict(result.info)
        info.update(
            {
                "command_residual_normalized": residual,
                "command_residual_gate": gate,
                "command_residual_applied": delta,
                "command_residual_scale_clipped": scale_clipped,
                "command_residual_rate_clipped": rate_clipped,
                "command_base_physical": base,
                "command_corrected_physical": corrected,
                "command_residual_actuator": action,
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


__all__ = ["BatchedPCCommandResidualEnv"]
