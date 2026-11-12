"""Batched base controllers and shadow state for the low-level environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from auv_lowlevel.bc import BCActor
from auv_lowlevel.config import ControllerConfig
from auv_lowlevel.observation import PREVIOUS_RESIDUAL_OBSERVATION_INDICES
from auv_lowlevel.observation_schema import resolve_observation_schema

from .geometry import quaternion_from_euler, world_to_body, wrap_angle, yaw_pitch


@dataclass(frozen=True)
class BatchedControlContext:
    eta: Tensor
    nu: Tensor
    quat_xyzw: Tensor
    current_ned_mps: Tensor
    reference_quat_xyzw: Tensor
    command_speed_mps: Tensor
    observation: Tensor
    dt_s: float
    heading_gain: Tensor | None = None
    pitch_gain: Tensor | None = None
    heading_rate_gain: Tensor | None = None
    pitch_rate_gain: Tensor | None = None
    water_surge_mps: Tensor | None = None


def controller_state_from_actor_state(actor_state: Tensor) -> tuple[Tensor, Tensor]:
    """Split PC actor state into pose and a correctly ordered ``nu``.

    The frozen actor layout is ``[position, attitude, angular_rate, linear_velocity]``.
    The controller state is ``[position, attitude, linear_velocity, angular_rate]``.
    """

    values = torch.as_tensor(actor_state)
    if values.ndim != 2 or values.shape[1] < 12:
        raise ValueError("actor_state must have shape (N, >=12)")
    eta = values[:, :6]
    angular_rate = values[:, 6:9]
    linear_velocity = values[:, 9:12]
    return eta, torch.cat((linear_velocity, angular_rate), dim=1)


def _context_gain(
    value: Tensor | None,
    fallback: float,
    *,
    count: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | float:
    if value is None:
        return float(fallback)
    gain = torch.as_tensor(value, device=device, dtype=dtype)
    if gain.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not bool(torch.isfinite(gain).all()):
        raise ValueError(f"{name} must contain finite values")
    return gain


class BatchedBaseController(Protocol):
    def reset(self, mask: Tensor | None = None, *, count: int | None = None) -> None: ...

    def control(self, context: BatchedControlContext) -> Tensor: ...


class BatchedShadowSpeedIntegrator:
    """Tensorized version of ``ShadowSpeedIntegrator``."""

    def __init__(
        self,
        config: ControllerConfig,
        max_propeller_rpm: float,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.max_propeller_rpm = float(max_propeller_rpm)
        self.device = torch.device(device)
        self.dtype = dtype
        self.value_rpm: Tensor | None = None

    def reset(self, mask: Tensor | None = None, *, count: int | None = None) -> None:
        if mask is None:
            if count is None or count <= 0:
                raise ValueError("count must be positive when resetting the whole batch")
            self.value_rpm = torch.zeros((count,), device=self.device, dtype=self.dtype)
            return
        if self.value_rpm is None:
            raise RuntimeError("reset(count=...) must be called first")
        self.value_rpm = torch.where(
            mask,
            torch.zeros_like(self.value_rpm),
            self.value_rpm,
        )

    def advance(
        self,
        water_relative_surge_mps: Tensor,
        desired_water_speed_mps: Tensor,
        dt_s: float,
    ) -> Tensor:
        if self.value_rpm is None:
            raise RuntimeError("reset() must be called before advance()")
        water_surge = torch.as_tensor(
            water_relative_surge_mps, device=self.device, dtype=self.dtype
        )
        desired = torch.as_tensor(
            desired_water_speed_mps, device=self.device, dtype=self.dtype
        )
        if water_surge.shape != self.value_rpm.shape or desired.shape != self.value_rpm.shape:
            raise ValueError("speed tensors must have shape (N,)")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        speed_error = desired - water_surge
        unsaturated = (
            self.config.speed_feedforward_rpm_per_mps * desired
            + self.value_rpm
            + self.config.speed_gain_rpm_per_mps * speed_error
        )
        saturated_high = (unsaturated > self.max_propeller_rpm) & (speed_error > 0.0)
        saturated_low = (unsaturated < 0.0) & (speed_error < 0.0)
        updated = ~saturated_high & ~saturated_low
        candidate = self.value_rpm + (
            self.config.speed_integral_gain_rpm_per_mps_s * speed_error * dt_s
        )
        candidate = torch.clamp(
            candidate,
            -self.config.max_speed_integral_rpm,
            self.config.max_speed_integral_rpm,
        )
        self.value_rpm = torch.where(updated, candidate, self.value_rpm)
        return self.value_rpm


class BatchedRemus100Teacher:
    """Tensorized PD/PI tracking controller with per-environment internal state."""

    def __init__(
        self,
        config: ControllerConfig,
        actuator_limits: Any,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_rudder_angle_rad = float(actuator_limits.max_rudder_angle_rad)
        self.max_stern_plane_angle_rad = float(actuator_limits.max_stern_plane_angle_rad)
        self.max_rudder_rate_rad_s = float(actuator_limits.max_rudder_rate_rad_s)
        self.max_stern_plane_rate_rad_s = float(actuator_limits.max_stern_plane_rate_rad_s)
        self.max_propeller_rpm = float(actuator_limits.max_propeller_rpm)
        self.last_rudder_angle_rad: Tensor | None = None
        self.last_stern_plane_angle_rad: Tensor | None = None
        self.speed_integral_rpm: Tensor | None = None
        self.last_unsaturated_rudder: Tensor | None = None
        self.last_unsaturated_stern_plane: Tensor | None = None
        self.last_unsaturated_rpm: Tensor | None = None

    def reset(self, mask: Tensor | None = None, *, count: int | None = None) -> None:
        if mask is None:
            if count is None or count <= 0:
                raise ValueError("count must be positive when resetting the whole batch")
            zeros = torch.zeros((count,), device=self.device, dtype=self.dtype)
            self.last_rudder_angle_rad = zeros.clone()
            self.last_stern_plane_angle_rad = zeros.clone()
            self.speed_integral_rpm = zeros.clone()
            self.last_unsaturated_rudder = zeros.clone()
            self.last_unsaturated_stern_plane = zeros.clone()
            self.last_unsaturated_rpm = zeros.clone()
            return
        if any(
            value is None
            for value in (
                self.last_rudder_angle_rad,
                self.last_stern_plane_angle_rad,
                self.speed_integral_rpm,
            )
        ):
            raise RuntimeError("reset(count=...) must be called first")
        assert self.last_rudder_angle_rad is not None
        assert self.last_stern_plane_angle_rad is not None
        assert self.speed_integral_rpm is not None
        zeros = torch.zeros_like(self.last_rudder_angle_rad)
        self.last_rudder_angle_rad = torch.where(mask, zeros, self.last_rudder_angle_rad)
        self.last_stern_plane_angle_rad = torch.where(
            mask, zeros, self.last_stern_plane_angle_rad
        )
        self.speed_integral_rpm = torch.where(mask, zeros, self.speed_integral_rpm)

    def control(self, context: BatchedControlContext) -> Tensor:
        if (
            self.speed_integral_rpm is None
            or self.last_rudder_angle_rad is None
            or self.last_stern_plane_angle_rad is None
        ):
            raise RuntimeError("reset() must be called before control()")
        eta = context.eta
        nu = context.nu
        desired_yaw, desired_pitch = yaw_pitch(context.reference_quat_xyzw)
        actual_yaw, actual_pitch = yaw_pitch(quaternion_from_euler(eta[:, 3], eta[:, 4], eta[:, 5]))
        heading_error = wrap_angle(desired_yaw - actual_yaw)
        pitch_error = desired_pitch - actual_pitch
        count = int(eta.shape[0])
        heading_gain = _context_gain(
            context.heading_gain,
            self.config.heading_gain,
            count=count,
            name="context.heading_gain",
            device=self.device,
            dtype=self.dtype,
        )
        heading_rate_gain = _context_gain(
            context.heading_rate_gain,
            self.config.heading_rate_gain,
            count=count,
            name="context.heading_rate_gain",
            device=self.device,
            dtype=self.dtype,
        )
        pitch_gain = _context_gain(
            context.pitch_gain,
            self.config.pitch_gain,
            count=count,
            name="context.pitch_gain",
            device=self.device,
            dtype=self.dtype,
        )
        pitch_rate_gain = _context_gain(
            context.pitch_rate_gain,
            self.config.pitch_rate_gain,
            count=count,
            name="context.pitch_rate_gain",
            device=self.device,
            dtype=self.dtype,
        )
        unsaturated_rudder = heading_gain * heading_error - (
            heading_rate_gain * nu[:, 5]
        )
        unsaturated_stern = -pitch_gain * pitch_error - (
            pitch_rate_gain * nu[:, 4]
        )
        rudder = torch.clamp(
            unsaturated_rudder,
            -self.max_rudder_angle_rad,
            self.max_rudder_angle_rad,
        )
        stern = torch.clamp(
            unsaturated_stern,
            -self.max_stern_plane_angle_rad,
            self.max_stern_plane_angle_rad,
        )
        max_rudder_step = self.max_rudder_rate_rad_s * context.dt_s
        max_stern_step = self.max_stern_plane_rate_rad_s * context.dt_s
        rudder = self.last_rudder_angle_rad + torch.clamp(
            rudder - self.last_rudder_angle_rad,
            -max_rudder_step,
            max_rudder_step,
        )
        stern = self.last_stern_plane_angle_rad + torch.clamp(
            stern - self.last_stern_plane_angle_rad,
            -max_stern_step,
            max_stern_step,
        )
        if context.water_surge_mps is not None:
            relative_surge = torch.as_tensor(
                context.water_surge_mps,
                device=self.device,
                dtype=self.dtype,
            )
            if relative_surge.shape != (count,):
                raise ValueError(f"context.water_surge_mps must have shape ({count},)")
        else:
            current_ned = (
                context.current_ned_mps
                if self.config.use_current_field
                else torch.zeros_like(context.current_ned_mps)
            )
            current_body = world_to_body(current_ned, context.quat_xyzw)
            relative_surge = nu[:, 0] - current_body[:, 0]
        speed_error = context.command_speed_mps - relative_surge
        unsaturated_rpm = (
            self.config.speed_feedforward_rpm_per_mps * context.command_speed_mps
            + self.speed_integral_rpm
            + self.config.speed_gain_rpm_per_mps * speed_error
        )
        rpm = torch.clamp(unsaturated_rpm, 0.0, self.max_propeller_rpm)
        saturated_high = (unsaturated_rpm > self.max_propeller_rpm) & (speed_error > 0.0)
        saturated_low = (unsaturated_rpm < 0.0) & (speed_error < 0.0)
        updated = ~saturated_high & ~saturated_low
        integral_candidate = self.speed_integral_rpm + (
            self.config.speed_integral_gain_rpm_per_mps_s * speed_error * context.dt_s
        )
        integral_candidate = torch.clamp(
            integral_candidate,
            -self.config.max_speed_integral_rpm,
            self.config.max_speed_integral_rpm,
        )
        self.speed_integral_rpm = torch.where(updated, integral_candidate, self.speed_integral_rpm)
        self.last_rudder_angle_rad = rudder
        self.last_stern_plane_angle_rad = stern
        self.last_unsaturated_rudder = unsaturated_rudder
        self.last_unsaturated_stern_plane = unsaturated_stern
        self.last_unsaturated_rpm = unsaturated_rpm
        return torch.stack((rudder, stern, rpm), dim=1)


class BatchedBCPolicy:
    """Frozen BC actor evaluated on a tensor batch without NumPy round-trips."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        device: str | torch.device = "cpu",
        mask_previous_residual: bool | None = None,
        expected_observation_dim: int | None = None,
        expected_observation_schema: str | None = None,
        expected_speed_feedback: str | None = None,
        expected_current_contract: str | None = None,
    ) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
        self.device = torch.device(device)
        self.observation_dim = int(checkpoint["observation_dim"])
        stored_schema = checkpoint.get("observation_schema")
        self.observation_schema = resolve_observation_schema(
            self.observation_dim,
            None if stored_schema is None else str(stored_schema),
        )
        if (
            expected_observation_dim is not None
            and self.observation_dim != expected_observation_dim
        ):
            raise ValueError(
                "checkpoint observation dimension mismatch: "
                f"expected {expected_observation_dim}, got {self.observation_dim}"
            )
        if (
            expected_observation_schema is not None
            and self.observation_schema.version != expected_observation_schema
        ):
            raise ValueError(
                "checkpoint observation schema mismatch: "
                f"expected {expected_observation_schema!r}, "
                f"got {self.observation_schema.version!r}"
            )
        self.speed_feedback = str(
            checkpoint.get("observation_speed_feedback") or "water_relative"
        )
        self.current_contract = str(
            checkpoint.get("observation_current_contract") or "legacy"
        )
        if (
            expected_speed_feedback is not None
            and self.speed_feedback != expected_speed_feedback
        ):
            raise ValueError(
                "checkpoint observation speed-feedback mismatch: "
                f"expected {expected_speed_feedback!r}, got {self.speed_feedback!r}"
            )
        if (
            expected_current_contract is not None
            and self.current_contract != expected_current_contract
        ):
            raise ValueError(
                "checkpoint current-contract mismatch: "
                f"expected {expected_current_contract!r}, got {self.current_contract!r}"
            )
        hidden_sizes = tuple(int(value) for value in checkpoint["hidden_sizes"])
        self.model = BCActor(self.observation_dim, hidden_sizes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.observation_mean = torch.as_tensor(
            checkpoint["observation_mean"], device=self.device, dtype=torch.float32
        )
        self.observation_std = torch.as_tensor(
            checkpoint["observation_std"], device=self.device, dtype=torch.float32
        )
        self.action_scales = torch.as_tensor(
            checkpoint["action_scales"], device=self.device, dtype=torch.float32
        )
        if self.observation_mean.shape != (self.observation_dim,) or self.observation_std.shape != (
            self.observation_dim,
        ):
            raise ValueError("checkpoint observation normalization has the wrong shape")
        if self.action_scales.shape != (3,):
            raise ValueError("checkpoint action scales have the wrong shape")
        residual_std = self.observation_std[list(PREVIOUS_RESIDUAL_OBSERVATION_INDICES)]
        if mask_previous_residual is None:
            mask_previous_residual = bool(torch.all(residual_std <= 1e-4))
        self.mask_previous_residual = bool(mask_previous_residual)
        self.metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
        self.metadata["observation_schema"] = self.observation_schema.version
        self.metadata["observation_speed_feedback"] = self.speed_feedback
        self.metadata["observation_current_contract"] = self.current_contract
        self.metadata["mask_previous_residual"] = self.mask_previous_residual

    def reset(self, mask: Tensor | None = None, *, count: int | None = None) -> None:
        del mask, count

    def control(self, context: BatchedControlContext) -> Tensor:
        observation = context.observation
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(
                f"observation must have shape (N, {self.observation_dim}), got "
                f"{tuple(observation.shape)}"
            )
        normalized = (
            observation.to(device=self.device, dtype=torch.float32) - self.observation_mean
        ) / self.observation_std
        if self.mask_previous_residual:
            normalized = normalized.clone()
            normalized[:, list(PREVIOUS_RESIDUAL_OBSERVATION_INDICES)] = 0.0
        with torch.inference_mode():
            prediction = self.model(normalized)
        prediction = torch.stack(
            (
                torch.clamp(prediction[:, 0], -1.0, 1.0),
                torch.clamp(prediction[:, 1], -1.0, 1.0),
                torch.clamp(prediction[:, 2], 0.0, 1.0),
            ),
            dim=1,
        )
        return prediction * self.action_scales


__all__ = [
    "BatchedBaseController",
    "BatchedBCPolicy",
    "BatchedControlContext",
    "BatchedRemus100Teacher",
    "BatchedShadowSpeedIntegrator",
]
