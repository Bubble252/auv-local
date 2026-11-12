"""Privileged PD/PI teacher adapter for PC20 demonstration collection."""

from __future__ import annotations

import torch
from torch import Tensor

from .controllers import BatchedControlContext, BatchedRemus100Teacher
from .geometry import quaternion_from_euler, yaw_pitch
from .pc_env import BatchedPCLowLevelEnv


class BatchedPCTeacher:
    """Produce PC20 teacher labels for the environment's current command."""

    def __init__(
        self,
        env: BatchedPCLowLevelEnv,
        *,
        pitch_integral_gain: float = 0.1,
        pitch_integral_limit_rad_s: float = 2.0,
        speed_feedback: str | None = None,
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
            env.config.observation.speed_feedback
            if speed_feedback is None
            else speed_feedback
        )
        if self.speed_feedback not in {"ground", "water_relative"}:
            raise ValueError("speed_feedback must be 'ground' or 'water_relative'")
        self._pitch_integral: Tensor | None = None
        self._last_stern_plane_angle_rad: Tensor | None = None
        self._speed_integral_rpm: Tensor | None = None

    def reset(self, count: int) -> None:
        self.teacher.reset(count=count)
        self._pitch_integral = torch.zeros(
            count, device=self.env.device, dtype=self.env.dtype
        )
        self._last_stern_plane_angle_rad = torch.zeros_like(self._pitch_integral)
        self._speed_integral_rpm = torch.zeros_like(self._pitch_integral)

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
        self._speed_integral_rpm = torch.where(
            mask_t, zeros, self._speed_integral_rpm
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
        if observation.solution is not None:
            reference_attitude = observation.solution.reference_quat_xyzw
        else:
            zeros = env.eta[:, 3] * 0.0
            reference_attitude = quaternion_from_euler(
                zeros,
                observation.command.pitch_rad,
                observation.command.heading_rad,
            )
        context = BatchedControlContext(
            eta=env.eta,
            nu=env.nu,
            quat_xyzw=attitude,
            current_ned_mps=env.eta[:, :3] * 0.0,
            reference_quat_xyzw=reference_attitude,
            command_speed_mps=observation.command.speed_mps,
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
        self._pitch_integral = torch.where(
            winding_up, self._pitch_integral, candidate_integral
        )
        self._last_stern_plane_angle_rad = stern
        self.teacher.last_stern_plane_angle_rad = stern
        self.teacher.last_unsaturated_stern_plane = unsaturated_stern

        if self.speed_feedback == "ground":
            speed_feedback = observation.actor_state[:, 9]
        else:
            speed_feedback = env.nu[:, 0]
        speed_error = observation.command.speed_mps - speed_feedback
        controller = env.config.controller
        unsaturated_rpm = (
            controller.speed_feedforward_rpm_per_mps
            * observation.command.speed_mps
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
