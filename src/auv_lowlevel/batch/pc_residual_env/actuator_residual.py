"""Wrap ALOS actuator-space residuals for the Goal86 T2/P-A arms.

Non-responsibility: does not change the ALOS baseline or zero-action semantics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from ..controllers import (
    BatchedControlContext,
    BatchedRemus100Teacher,
    controller_state_from_actor_state,
)
from ..geometry import quaternion_from_direction, quaternion_from_euler
from ..pc_command_residual import smooth_terminal_gate
from ..pc_env import BatchedPCLowLevelEnv, BatchedPCLowLevelStep


class BatchedPCActuatorResidualEnv:
    """Expose a normalized actuator residual around the deployable ALOS PD/PI.

    The action is ``residual_normalized`` in ``[-1, 1]^3``.  It is converted to
    a physical delta with ``residual_scale * full_scale``.  The base actuator is
    produced by the same actor-observable PD/PI controller used by the
    command-residual wrapper, so an all-zero residual is the traditional ALOS
    baseline.
    """

    def __init__(
        self,
        env: BatchedPCLowLevelEnv,
        *,
        residual_scale: float = 0.25,
        residual_channel_gains: Sequence[float] = (1.0, 1.0, 1.0),
        max_action: float = 1.0,
        controller_state: str = "actor",
        terminal_gate_enabled: bool = False,
        terminal_gate_enter_m: float = 20.0,
        terminal_gate_full_m: float = 5.0,
        residual_scale_near: float | None = None,
        residual_scale_far: float | None = None,
        residual_scale_switch_distance_m: float | None = None,
        residual_channel_gains_near: Sequence[float] | None = None,
        residual_channel_gain_near_distance_m: float | None = None,
        residual_channel_gain_far_distance_m: float | None = None,
    ) -> None:
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        gains = tuple(float(value) for value in residual_channel_gains)
        if len(gains) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in gains):
            raise ValueError("residual_channel_gains must contain three finite positive values")
        if not math.isfinite(max_action) or max_action <= 0.0:
            raise ValueError("max_action must be finite and positive")
        if controller_state not in {"actor", "privileged"}:
            raise ValueError("controller_state must be 'actor' or 'privileged'")
        if not math.isfinite(terminal_gate_enter_m) or terminal_gate_enter_m <= 0.0:
            raise ValueError("terminal_gate_enter_m must be finite and positive")
        if (
            not math.isfinite(terminal_gate_full_m)
            or terminal_gate_full_m <= 0.0
            or terminal_gate_full_m >= terminal_gate_enter_m
        ):
            raise ValueError("terminal_gate_full_m must be finite, positive, and less than enter_m")
        schedule_values = (
            residual_scale_near,
            residual_scale_far,
            residual_scale_switch_distance_m,
        )
        if any(value is not None for value in schedule_values) and not all(
            value is not None for value in schedule_values
        ):
            raise ValueError("residual scale schedule requires near, far, and switch distance")
        if residual_scale_near is not None:
            assert residual_scale_far is not None
            assert residual_scale_switch_distance_m is not None
            for name, value in (
                ("residual_scale_near", residual_scale_near),
                ("residual_scale_far", residual_scale_far),
                (
                    "residual_scale_switch_distance_m",
                    residual_scale_switch_distance_m,
                ),
            ):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
        channel_schedule_values = (
            residual_channel_gains_near,
            residual_channel_gain_near_distance_m,
            residual_channel_gain_far_distance_m,
        )
        if any(value is not None for value in channel_schedule_values) and not all(
            value is not None for value in channel_schedule_values
        ):
            raise ValueError(
                "residual channel gain schedule requires near gains, near distance, "
                "and far distance"
            )
        near_gains: tuple[float, ...] | None = None
        if residual_channel_gains_near is not None:
            near_gains = tuple(float(value) for value in residual_channel_gains_near)
            if len(near_gains) != 3 or any(
                not math.isfinite(value) or value <= 0.0 for value in near_gains
            ):
                raise ValueError(
                    "residual_channel_gains_near must contain three finite positive values"
                )
            assert residual_channel_gain_near_distance_m is not None
            assert residual_channel_gain_far_distance_m is not None
            near_distance = float(residual_channel_gain_near_distance_m)
            far_distance = float(residual_channel_gain_far_distance_m)
            if (
                not math.isfinite(near_distance)
                or not math.isfinite(far_distance)
                or near_distance <= 0.0
                or far_distance <= near_distance
            ):
                raise ValueError(
                    "residual channel gain distances must be finite with "
                    "0 < near_distance < far_distance"
                )

        self.env = env
        self.residual_scale = float(residual_scale)
        self.residual_channel_gains = gains
        self._residual_channel_gain_tensor = torch.tensor(gains, device=env.device, dtype=env.dtype)
        self.residual_channel_gains_near = near_gains
        self.residual_channel_gain_near_distance_m = (
            None
            if residual_channel_gain_near_distance_m is None
            else float(residual_channel_gain_near_distance_m)
        )
        self.residual_channel_gain_far_distance_m = (
            None
            if residual_channel_gain_far_distance_m is None
            else float(residual_channel_gain_far_distance_m)
        )
        self._residual_channel_gain_near_tensor = (
            None
            if near_gains is None
            else torch.tensor(near_gains, device=env.device, dtype=env.dtype)
        )
        self.residual_scale_near = (
            None if residual_scale_near is None else float(residual_scale_near)
        )
        self.residual_scale_far = None if residual_scale_far is None else float(residual_scale_far)
        self.residual_scale_switch_distance_m = (
            None
            if residual_scale_switch_distance_m is None
            else float(residual_scale_switch_distance_m)
        )
        self.max_action = float(max_action)
        self.controller_state = controller_state
        self.terminal_gate_enabled = bool(terminal_gate_enabled)
        self.terminal_gate_enter_m = float(terminal_gate_enter_m)
        self.terminal_gate_full_m = float(terminal_gate_full_m)
        self.controller = BatchedRemus100Teacher(
            env.config.controller,
            env.limits,
            device=env.device,
            dtype=env.dtype,
        )
        self.controller.reset(count=env.num_envs)
        self.num_envs = env.num_envs
        self.action_dim = 3
        self.num_actions = 3
        self.device = env.device
        self.dtype = env.dtype
        self.dt_s = env.dt_s
        self.config = env.config
        self.actor_state_dim = env.actor_state_dim
        self.actor_path_dim = env.actor_path_dim
        self.critic_state_dim = env.critic_state_dim
        self.path_blocks = env.path_blocks
        self.vehicle = env.vehicle
        self.limits = env.limits
        self._previous_residual = torch.zeros((env.num_envs, 3), device=env.device, dtype=env.dtype)
        self._split_context: dict[str, Any] | None = None
        self._full_scale = torch.tensor(
            [
                env.limits.max_rudder_angle_rad,
                env.limits.max_stern_plane_angle_rad,
                env.limits.max_propeller_rpm,
            ],
            device=env.device,
            dtype=env.dtype,
        )
        self._lower = torch.tensor(
            [
                -env.limits.max_rudder_angle_rad,
                -env.limits.max_stern_plane_angle_rad,
                0.0,
            ],
            device=env.device,
            dtype=env.dtype,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def observation(self) -> Any:
        return self.env.observation

    @property
    def path(self) -> Any:
        return self.env.path

    def _reference_quaternion(self, observation: Any) -> Tensor:
        heading = observation.command.heading_rad
        pitch = observation.command.pitch_rad
        direction = torch.stack(
            (
                torch.cos(pitch) * torch.cos(heading),
                torch.cos(pitch) * torch.sin(heading),
                -torch.sin(pitch),
            ),
            dim=1,
        )
        return quaternion_from_direction(direction)

    def base_actuator(self, observation: Any | None = None) -> Tensor:
        """Return the physical ALOS PD/PI action before adding the residual.

        This call advances the controller's internal integrator and rate state.
        It is therefore intended for diagnostics or for the single call made by
        :meth:`step`, not for repeated parity probes without restoring state.
        """

        value = self.env.observation if observation is None else observation
        reference_quat = self._reference_quaternion(value)
        controller_water_surge = (
            value.water_velocity_body_mps[:, 0]
            if self.env.config.plant.velocity_contract == "nu_ground"
            else None
        )
        if self.controller_state == "privileged":
            eta = self.env.eta
            nu = self.env.nu
            current_ned_mps = self.env.current_ned_mps
            if eta is None or nu is None or current_ned_mps is None:
                raise RuntimeError("privileged controller state is unavailable before reset")
            controller_eta = eta
            controller_nu = nu
            controller_quat = quaternion_from_euler(eta[:, 3], eta[:, 4], eta[:, 5])
            controller_current = current_ned_mps
        else:
            if self.env.config.plant.velocity_contract == "nu_ground":
                controller_eta, controller_nu = controller_state_from_actor_state(value.actor_state)
            else:
                controller_eta = value.actor_state[:, :6]
                controller_nu = value.actor_state[:, 6:12]
            controller_quat = reference_quat
            controller_current = torch.zeros_like(value.actor_state[:, :3])
        context = BatchedControlContext(
            eta=controller_eta,
            nu=controller_nu,
            quat_xyzw=controller_quat,
            current_ned_mps=controller_current,
            reference_quat_xyzw=reference_quat,
            command_speed_mps=value.command.speed_mps,
            observation=value.actor_state,
            dt_s=float(self.env.dt_s),
            water_surge_mps=controller_water_surge,
        )
        return self.controller.control(context)

    def _normalize_actuator(self, actuator: Tensor) -> Tensor:
        normalized = torch.empty_like(actuator)
        normalized[:, 0] = actuator[:, 0] / float(self.env.limits.max_rudder_angle_rad)
        normalized[:, 1] = actuator[:, 1] / float(self.env.limits.max_stern_plane_angle_rad)
        normalized[:, 2] = 2.0 * actuator[:, 2] / float(self.env.limits.max_propeller_rpm) - 1.0
        return normalized

    def _terminal_gate(self) -> Tensor:
        """Gate residuals by deployable navigation distance to the goal."""

        observation = self.env.observation
        position = observation.actor_state[:, :3].to(device=self.env.device, dtype=self.env.dtype)
        goal = self.env.path.goal_ned_m_for(self.env.num_envs)
        distance = torch.linalg.norm(goal - position, dim=1)
        return smooth_terminal_gate(
            distance,
            enter_m=self.terminal_gate_enter_m,
            full_m=self.terminal_gate_full_m,
        )

    def _residual_scale_state(self) -> tuple[Tensor, Tensor]:
        """Return the residual scale and the distance used to choose it."""

        observation = self.env.observation
        position = observation.actor_state[:, :3].to(device=self.device, dtype=self.dtype)
        goal = self.env.path.goal_ned_m_for(self.env.num_envs)
        distance = torch.linalg.norm(goal - position, dim=1)
        if self.residual_scale_near is None:
            scale = torch.full(
                (self.env.num_envs,),
                self.residual_scale,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            scale = torch.where(
                distance > float(self.residual_scale_switch_distance_m),
                torch.full_like(distance, float(self.residual_scale_far)),
                torch.full_like(distance, float(self.residual_scale_near)),
            )
        return scale, distance

    def _residual_scale_vector(self) -> Tensor:
        """Return the per-environment residual scale for this step."""

        scale, _ = self._residual_scale_state()
        return scale

    def _residual_channel_gain_state(self, distance: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return effective channel gains and the near-gain blend for this step."""

        far = self._residual_channel_gain_tensor.expand(self.env.num_envs, 3)
        if self._residual_channel_gain_near_tensor is None:
            return far, torch.zeros(self.env.num_envs, device=self.device, dtype=self.dtype)
        if distance is None:
            _, distance = self._residual_scale_state()
        assert self.residual_channel_gain_near_distance_m is not None
        assert self.residual_channel_gain_far_distance_m is not None
        near_gate = smooth_terminal_gate(
            distance,
            enter_m=self.residual_channel_gain_far_distance_m,
            full_m=self.residual_channel_gain_near_distance_m,
        )
        near = self._residual_channel_gain_near_tensor.expand(self.env.num_envs, 3)
        gains = far + near_gate[:, None] * (near - far)
        return gains, near_gate

    def _apply_residual(
        self,
        base: Tensor,
        residual_normalized: Tensor,
        *,
        gate: Tensor | None = None,
        scale: Tensor | None = None,
        channel_gains: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raw = residual_normalized
        if gate is None:
            gate = torch.ones_like(raw[:, 0])
        gated = raw * gate[:, None]
        residual = torch.clamp(gated, -self.max_action, self.max_action)
        scale_clipped = torch.abs(residual - gated) > 1e-6
        if scale is None:
            scale = self._residual_scale_vector()
        if channel_gains is None:
            channel_gains = self._residual_channel_gain_tensor.expand(self.env.num_envs, 3)
        delta = residual * scale[:, None] * self._full_scale * channel_gains
        requested = base + delta
        magnitude_limited = torch.clamp(requested, self._lower, self._full_scale)
        magnitude_clipped = torch.abs(magnitude_limited - requested) > torch.tensor(
            [1e-6, 1e-6, 1e-3], device=self.device, dtype=self.dtype
        )

        previous = self.env.actuator
        if previous is None:
            raise RuntimeError("reset() must be called before applying a residual")
        rate_limited = magnitude_limited.clone()
        max_fin_step = torch.tensor(
            [
                self.env.limits.max_rudder_rate_rad_s,
                self.env.limits.max_stern_plane_rate_rad_s,
            ],
            device=self.device,
            dtype=self.dtype,
        ) * float(self.env.dt_s)
        fin_change = torch.clamp(
            rate_limited[:, :2] - previous[:, :2],
            -max_fin_step,
            max_fin_step,
        )
        rate_limited[:, :2] = previous[:, :2] + fin_change
        rate_clipped = torch.abs(rate_limited - magnitude_limited) > torch.tensor(
            [1e-6, 1e-6, torch.inf], device=self.device, dtype=self.dtype
        )
        # Zero residual is the ALOS baseline, not "ALOS plus a pre-limited zero".
        # Pass the controller request through unchanged so the environment's
        # action pipeline sees exactly the same request as the command wrapper.
        # This keeps material saturation and guidance diagnostics paired.
        zero_residual = torch.all(residual == 0.0, dim=1)
        total = torch.where(zero_residual[:, None], base, rate_limited)
        zero_residual_2d = zero_residual[:, None]
        magnitude_clipped = torch.where(
            zero_residual_2d, torch.zeros_like(magnitude_clipped), magnitude_clipped
        )
        rate_clipped = torch.where(zero_residual_2d, torch.zeros_like(rate_clipped), rate_clipped)
        scale_clipped = torch.where(
            zero_residual_2d, torch.zeros_like(scale_clipped), scale_clipped
        )
        return total, {
            "raw": raw,
            "gate": gate,
            "gated": gated,
            "residual": residual,
            "applied": delta,
            "effective": total - base,
            "requested": requested,
            "scale_clipped": scale_clipped,
            "scale": scale,
            "magnitude_clipped": magnitude_clipped,
            "rate_clipped": rate_clipped,
            "channel_gains": channel_gains,
        }

    def set_split_context(
        self,
        *,
        base_residual_normalized: Tensor,
        ppo_residual_normalized: Tensor,
        gate: Tensor,
        ppo_rate_fraction: float,
        ppo_fin_gain: float,
        ppo_rpm_gain: float,
    ) -> None:
        if self._split_context is not None:
            raise RuntimeError("split residual context is already active")
        fraction = float(ppo_rate_fraction)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise ValueError("ppo_rate_fraction must be finite and in (0, 1)")
        for name, value in (
            ("ppo_fin_gain", ppo_fin_gain),
            ("ppo_rpm_gain", ppo_rpm_gain),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self._split_context = {
            "base_residual": torch.as_tensor(
                base_residual_normalized, device=self.device, dtype=self.dtype
            ),
            "ppo_residual": torch.as_tensor(
                ppo_residual_normalized, device=self.device, dtype=self.dtype
            ),
            "gate": torch.as_tensor(gate, device=self.device, dtype=self.dtype),
            "ppo_rate_fraction": fraction,
            "ppo_fin_gain": float(ppo_fin_gain),
            "ppo_rpm_gain": float(ppo_rpm_gain),
        }

    def clear_split_context(self) -> None:
        self._split_context = None

    def _apply_split_residual(
        self,
        base: Tensor,
        *,
        base_residual: Tensor,
        ppo_residual: Tensor,
        gate: Tensor,
        ppo_rate_fraction: float,
        ppo_fin_gain: float,
        ppo_rpm_gain: float,
        scale: Tensor,
        channel_gains: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        expected = (self.env.num_envs, 3)
        for name, value in (
            ("base_residual", base_residual),
            ("ppo_residual", ppo_residual),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if tuple(gate.shape) != (self.env.num_envs,):
            raise ValueError("split gate must have shape (num_envs,)")

        gain = torch.tensor(
            [ppo_fin_gain, ppo_fin_gain, ppo_rpm_gain],
            device=self.device,
            dtype=self.dtype,
        )
        bc_delta = base_residual * scale[:, None] * self._full_scale * channel_gains
        bc_requested = base + bc_delta
        bc_limited = torch.clamp(bc_requested, self._lower, self._full_scale)

        previous = self.env.actuator
        if previous is None:
            raise RuntimeError("reset() must be called before applying a residual")
        max_fin_step = torch.tensor(
            [
                self.env.limits.max_rudder_rate_rad_s,
                self.env.limits.max_stern_plane_rate_rad_s,
            ],
            device=self.device,
            dtype=self.dtype,
        ) * float(self.env.dt_s)
        bc_share = 1.0 - ppo_rate_fraction
        bc_fin_step = torch.clamp(
            bc_limited[:, :2] - previous[:, :2],
            -bc_share * max_fin_step,
            bc_share * max_fin_step,
        )
        bc_total = bc_limited.clone()
        bc_total[:, :2] = previous[:, :2] + bc_fin_step

        ppo_delta_raw = (
            gate[:, None]
            * ppo_residual
            * gain[None, :]
            * scale[:, None]
            * self._full_scale
            * channel_gains
        )
        ppo_limit = ppo_rate_fraction * max_fin_step
        ppo_delta = torch.cat(
            (
                torch.clamp(ppo_delta_raw[:, :2], -ppo_limit, ppo_limit),
                ppo_delta_raw[:, 2:3],
            ),
            dim=1,
        )
        requested = bc_total + ppo_delta
        total = torch.clamp(requested, self._lower, self._full_scale)

        bc_rate_clipped = torch.abs(bc_total - bc_limited) > torch.tensor(
            [1e-6, 1e-6, torch.inf], device=self.device, dtype=self.dtype
        )
        ppo_rate_clipped = torch.abs(ppo_delta_raw[:, :2] - ppo_delta[:, :2]) > torch.tensor(
            [1e-6, 1e-6], device=self.device, dtype=self.dtype
        )
        ppo_rate_clipped = torch.cat(
            (
                ppo_rate_clipped,
                torch.zeros((self.env.num_envs, 1), device=self.device, dtype=torch.bool),
            ),
            dim=1,
        )
        magnitude_clipped = torch.abs(total - requested) > torch.tensor(
            [1e-6, 1e-6, 1e-3], device=self.device, dtype=self.dtype
        )
        combined_residual = base_residual + gate[:, None] * ppo_residual
        scale_clipped = torch.abs(combined_residual) > self.max_action
        residual = torch.clamp(combined_residual, -self.max_action, self.max_action)
        return total, {
            "raw": combined_residual,
            "gate": gate,
            "gated": gate[:, None] * ppo_residual,
            "residual": residual,
            "applied": bc_delta + ppo_delta,
            "effective": total - base,
            "requested": base + bc_delta + ppo_delta,
            "scale_clipped": scale_clipped,
            "scale": scale,
            "magnitude_clipped": magnitude_clipped,
            "rate_clipped": bc_rate_clipped | ppo_rate_clipped,
            "bc_applied": bc_delta,
            "bc_effective": bc_total - base,
            "ppo_applied": ppo_delta,
            "ppo_effective": total - bc_total,
            "bc_rate_clipped": bc_rate_clipped,
            "ppo_rate_clipped": ppo_rate_clipped,
            "channel_gains": channel_gains,
        }

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
                f"actuator residual must have shape {expected}, got {tuple(residual.shape)}"
            )
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("actuator residual contains NaN or Inf")
        if advance_mask is None:
            active = torch.ones(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        else:
            active = torch.as_tensor(advance_mask, device=self.env.device, dtype=torch.bool)
            if active.shape != (self.env.num_envs,):
                raise ValueError(f"advance_mask must have shape ({self.env.num_envs},)")

        base = self.base_actuator()
        scale, scale_distance = self._residual_scale_state()
        channel_gains, channel_gain_near_gate = self._residual_channel_gain_state(scale_distance)
        split = self._split_context
        if split is not None:
            gate = split["gate"]
            total, diagnostics = self._apply_split_residual(
                base,
                base_residual=split["base_residual"],
                ppo_residual=split["ppo_residual"],
                gate=gate,
                ppo_rate_fraction=split["ppo_rate_fraction"],
                ppo_fin_gain=split["ppo_fin_gain"],
                ppo_rpm_gain=split["ppo_rpm_gain"],
                scale=scale,
                channel_gains=channel_gains,
            )
        else:
            gate = (
                self._terminal_gate()
                if self.terminal_gate_enabled
                else torch.ones_like(residual[:, 0])
            )
            total, diagnostics = self._apply_residual(
                base,
                residual,
                gate=gate,
                scale=scale,
                channel_gains=channel_gains,
            )
        diagnostics["scale_distance"] = scale_distance
        normalized_action = self._normalize_actuator(total)
        result = self.env.step(
            normalized_action,
            auto_reset=auto_reset,
            advance_mask=advance_mask,
        )
        done = result.terminated | result.truncated
        if auto_reset and bool(done.any()):
            self.controller.reset(mask=done)
        self._previous_residual = torch.where(
            done[:, None], torch.zeros_like(self._previous_residual), diagnostics["residual"]
        )
        info = dict(result.info)
        info.update(
            {
                "actuator_residual_normalized": residual,
                "actuator_residual_gate": diagnostics["gate"],
                "actuator_residual_gated_normalized": diagnostics["gated"],
                "actuator_residual_clipped_normalized": diagnostics["residual"],
                "actuator_residual_applied": diagnostics["applied"],
                "actuator_residual_effective": diagnostics["effective"],
                "actuator_base_physical": base,
                "actuator_requested_physical": diagnostics["requested"],
                "actuator_total_physical": total,
                "actuator_residual_scale_clipped": diagnostics["scale_clipped"],
                "actuator_residual_magnitude_clipped": diagnostics["magnitude_clipped"],
                "actuator_residual_rate_clipped": diagnostics["rate_clipped"],
                "actuator_residual_scale": diagnostics["scale"],
                "actuator_residual_channel_gains": diagnostics["channel_gains"],
                "actuator_residual_channel_gain_near_gate": channel_gain_near_gate,
                "actuator_residual_schedule_distance_m": diagnostics["scale_distance"],
            }
        )
        if split is not None:
            info.update(
                {
                    "actuator_residual_bc_applied": diagnostics["bc_applied"],
                    "actuator_residual_bc_effective": diagnostics["bc_effective"],
                    "actuator_residual_ppo_applied": diagnostics["ppo_applied"],
                    "actuator_residual_ppo_effective": diagnostics["ppo_effective"],
                    "actuator_residual_bc_rate_clipped": diagnostics["bc_rate_clipped"],
                    "actuator_residual_ppo_rate_clipped": diagnostics["ppo_rate_clipped"],
                }
            )
        return replace(result, info=info)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observation = self.env.reset(*args, **kwargs)
        self.controller.reset(count=self.env.num_envs)
        self._previous_residual = torch.zeros_like(self._previous_residual)
        return observation

    def reset_indices(self, mask: Tensor, *args: Any, **kwargs: Any) -> Any:
        observation = self.env.reset_indices(mask, *args, **kwargs)
        reset_mask = torch.as_tensor(mask, device=self.env.device, dtype=torch.bool)
        self.controller.reset(mask=reset_mask)
        self._previous_residual = torch.where(
            reset_mask[:, None],
            torch.zeros_like(self._previous_residual),
            self._previous_residual,
        )
        return observation


__all__ = ["BatchedPCActuatorResidualEnv"]
