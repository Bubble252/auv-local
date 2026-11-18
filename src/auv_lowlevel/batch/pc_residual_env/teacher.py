"""Adapt the privileged PD/PI teacher for PC20 demonstration collection.

Non-responsibility: does not train policies or write demonstration manifests.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from ..controllers import BatchedControlContext, BatchedRemus100Teacher
from ..geometry import quaternion_from_euler, yaw_pitch
from ..guidance import BatchedThreeDAlosGuidance
from ..pc_env import BatchedPCLowLevelEnv


class BatchedPCTeacher:
    """Produce privileged PC20 teacher labels for BC demonstration collection.

    The default reference path is ALOS guidance with true-current feedforward,
    followed by the existing REMUS PD/PI controller. ``reference_mode="command"``
    preserves the pre-Goal184 tangent-command behavior for diagnostics.
    """

    def __init__(
        self,
        env: BatchedPCLowLevelEnv,
        *,
        pitch_integral_gain: float = 0.1,
        pitch_integral_limit_rad_s: float = 2.0,
        speed_feedback: str | None = None,
        reference_mode: str = "alos_compensated",
        current_compensation_source: str = "truth",
    ) -> None:
        self.env = env
        self.teacher = BatchedRemus100Teacher(
            env.config.controller,
            env.limits,
            device=env.device,
            dtype=env.dtype,
        )
        self.pitch_integral_gain = float(pitch_integral_gain)
        self.pitch_integral_limit_rad_s = float(pitch_integral_limit_rad_s)
        self.speed_feedback = str(
            env.config.observation.speed_feedback if speed_feedback is None else speed_feedback
        )
        if self.speed_feedback not in {"ground", "water_relative"}:
            raise ValueError("speed_feedback must be 'ground' or 'water_relative'")
        if reference_mode not in {"alos_compensated", "command"}:
            raise ValueError("reference_mode must be 'alos_compensated' or 'command'")
        if current_compensation_source not in {"truth", "estimated", "none"}:
            raise ValueError("current_compensation_source must be 'truth', 'estimated' or 'none'")
        self.reference_mode = str(reference_mode)
        self.current_compensation_source = str(current_compensation_source)
        self.alos_guidance: BatchedThreeDAlosGuidance | None = None
        if self.reference_mode == "alos_compensated":
            guidance_source = (
                "true"
                if self.current_compensation_source == "truth"
                else self.current_compensation_source
            )
            guidance_config = replace(
                env.config.guidance,
                compensate_current=self.current_compensation_source != "none",
                current_crab_enabled=False,
                current_compensation_source=guidance_source,
            )
            self.alos_guidance = BatchedThreeDAlosGuidance(
                guidance_config,
                env.config.reference,
                device=env.device,
                dtype=env.dtype,
            )
        self._pitch_integral: Tensor | None = None
        self._last_stern_plane_angle_rad: Tensor | None = None
        self._speed_integral_rpm: Tensor | None = None
        self._projection_hint: Tensor | None = None
        self.last_solution: Any | None = None

    def reset(self, count: int) -> None:
        self.teacher.reset(count=count)
        self._pitch_integral = torch.zeros(count, device=self.env.device, dtype=self.env.dtype)
        self._last_stern_plane_angle_rad = torch.zeros_like(self._pitch_integral)
        self._speed_integral_rpm = torch.zeros_like(self._pitch_integral)
        self._projection_hint = None
        self.last_solution = None

    def reset_mask(self, mask: Tensor) -> None:
        self.teacher.reset(mask)
        if (
            self._pitch_integral is None
            or self._last_stern_plane_angle_rad is None
            or self._speed_integral_rpm is None
        ):
            raise RuntimeError("reset(count=...) must be called before reset_mask")
        mask_t = torch.as_tensor(mask, device=self.env.device, dtype=torch.bool)
        zeros = torch.zeros_like(self._pitch_integral)
        self._pitch_integral = torch.where(mask_t, zeros, self._pitch_integral)
        self._last_stern_plane_angle_rad = torch.where(
            mask_t, zeros, self._last_stern_plane_angle_rad
        )
        self._speed_integral_rpm = torch.where(mask_t, zeros, self._speed_integral_rpm)
        if self._projection_hint is not None:
            self._projection_hint = torch.where(
                mask_t, torch.zeros_like(self._projection_hint), self._projection_hint
            )

    def _current_for_guidance(self) -> tuple[Tensor, Tensor | None]:
        env = self.env
        if env.eta is None:
            raise RuntimeError("environment must be reset before teacher control")
        zeros = env.eta[:, :3] * 0.0
        if self.current_compensation_source == "none":
            return zeros, None
        if self.current_compensation_source == "truth":
            if env.current_ned_mps is None:
                raise RuntimeError("true current is required for teacher compensation")
            return env.current_ned_mps, None
        estimated, valid, _ = env._current_estimate_for_guidance()
        return torch.where(valid[:, None], estimated, zeros), valid

    def _alos_reference(self, observation: Any) -> tuple[Tensor, Tensor, Tensor, Any]:
        env = self.env
        if self.alos_guidance is None:
            raise RuntimeError("ALOS guidance is not enabled for this teacher")
        if env.sensor_output is None or env.path is None:
            raise RuntimeError("sensor output and path must be initialized")
        position = env.sensor_output.p_nav
        projection = env.path.project(
            position,
            segment_hint=self._projection_hint,
            search_radius=env.reset_projection_search_radius,
        )
        self._projection_hint = projection.segment_index
        current, estimated_valid = self._current_for_guidance()
        desired_ground_speed = env._episode_command_speed_mps
        if desired_ground_speed is None:
            desired_ground_speed = observation.command.speed_mps
        solution = self.alos_guidance.solve(
            path=env.path,
            projection=projection,
            position_ned_m=position,
            current_ned_mps=current,
            desired_ground_speed_mps=desired_ground_speed,
            estimated_current_ned_mps=current,
            estimated_current_valid=estimated_valid,
        )
        return (
            solution.reference_quat_xyzw,
            desired_ground_speed,
            solution.command_speed_mps,
            solution,
        )

    def control(self) -> Tensor:
        env = self.env
        if env.eta is None or env.nu is None:
            raise RuntimeError("environment must be reset before teacher control")
        observation = env.observation
        attitude = quaternion_from_euler(
            env.eta[:, 3],
            env.eta[:, 4],
            env.eta[:, 5],
        )
        if self.reference_mode == "alos_compensated":
            (
                reference_attitude,
                ground_speed_reference,
                water_speed_reference,
                solution,
            ) = self._alos_reference(observation)
            self.last_solution = solution
        elif observation.solution is not None:
            reference_attitude = observation.solution.reference_quat_xyzw
            ground_speed_reference = observation.command.speed_mps
            water_speed_reference = observation.command.speed_mps
        else:
            zeros = env.eta[:, 3] * 0.0
            reference_attitude = quaternion_from_euler(
                zeros,
                observation.command.pitch_rad,
                observation.command.heading_rad,
            )
            ground_speed_reference = observation.command.speed_mps
            water_speed_reference = observation.command.speed_mps
        context = BatchedControlContext(
            eta=env.eta,
            nu=env.nu,
            quat_xyzw=attitude,
            current_ned_mps=env.eta[:, :3] * 0.0,
            reference_quat_xyzw=reference_attitude,
            command_speed_mps=water_speed_reference,
            observation=observation.actor_state,
            dt_s=env.dt_s,
            water_surge_mps=(
                observation.water_velocity_body_mps[:, 0]
                if env.config.plant.velocity_contract == "nu_ground"
                else None
            ),
        )
        action = self.teacher.control(context)
        if (
            self._pitch_integral is None
            or self._last_stern_plane_angle_rad is None
            or self._speed_integral_rpm is None
        ):
            raise RuntimeError("reset() must be called before teacher control")

        desired_pitch = yaw_pitch(reference_attitude)[1]
        pitch_error = desired_pitch - env.eta[:, 4]
        unsaturated_stern = (
            -env.config.controller.pitch_gain * pitch_error
            - env.config.controller.pitch_rate_gain * env.nu[:, 4]
            + self.pitch_integral_gain * self._pitch_integral
        )
        max_stern = float(env.limits.max_stern_plane_angle_rad)
        stern = torch.clamp(unsaturated_stern, -max_stern, max_stern)
        max_step = float(env.limits.max_stern_plane_rate_rad_s) * env.dt_s
        stern = self._last_stern_plane_angle_rad + torch.clamp(
            stern - self._last_stern_plane_angle_rad, -max_step, max_step
        )
        stern = torch.clamp(stern, -max_stern, max_stern)

        at_upper = unsaturated_stern >= max_stern
        at_lower = unsaturated_stern <= -max_stern
        winding_up = (
            (at_upper & (pitch_error < 0.0))
            | (at_lower & (pitch_error > 0.0))
            | (at_upper & (self._pitch_integral > 0.0))
            | (at_lower & (self._pitch_integral < 0.0))
        )
        candidate_integral = self._pitch_integral - pitch_error * env.dt_s
        candidate_integral = torch.clamp(
            candidate_integral,
            -self.pitch_integral_limit_rad_s,
            self.pitch_integral_limit_rad_s,
        )
        self._pitch_integral = torch.where(winding_up, self._pitch_integral, candidate_integral)
        self._last_stern_plane_angle_rad = stern
        self.teacher.last_stern_plane_angle_rad = stern
        self.teacher.last_unsaturated_stern_plane = unsaturated_stern

        if self.speed_feedback == "ground":
            speed_feedback = observation.actor_state[:, 9]
            speed_reference = ground_speed_reference
        else:
            speed_feedback = env.nu[:, 0]
            speed_reference = water_speed_reference
        speed_error = speed_reference - speed_feedback
        controller = env.config.controller
        unsaturated_rpm = (
            controller.speed_feedforward_rpm_per_mps * water_speed_reference
            + self._speed_integral_rpm
            + controller.speed_gain_rpm_per_mps * speed_error
        )
        max_rpm = float(env.limits.max_propeller_rpm)
        rpm = torch.clamp(unsaturated_rpm, 0.0, max_rpm)
        saturated_high = (unsaturated_rpm > max_rpm) & (speed_error > 0.0)
        saturated_low = (unsaturated_rpm < 0.0) & (speed_error < 0.0)
        updated = ~saturated_high & ~saturated_low
        integral_candidate = self._speed_integral_rpm + (
            controller.speed_integral_gain_rpm_per_mps_s * speed_error * env.dt_s
        )
        integral_candidate = torch.clamp(
            integral_candidate,
            -controller.max_speed_integral_rpm,
            controller.max_speed_integral_rpm,
        )
        self._speed_integral_rpm = torch.where(
            updated, integral_candidate, self._speed_integral_rpm
        )
        action = action.clone()
        action[:, 1] = stern
        action[:, 2] = rpm
        return action


__all__ = ["BatchedPCTeacher"]
