"""Batched residual-to-actuator action pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from auv_lowlevel.actions import ActuatorLimits
from auv_lowlevel.config import LowLevelConfig


@dataclass(frozen=True)
class BatchedActionApplication:
    base: Tensor
    raw_residual_normalized: Tensor
    residual_normalized: Tensor
    delta: Tensor
    requested_total: Tensor
    total: Tensor
    residual_magnitude_clipped: Tensor
    magnitude_clipped: Tensor
    rate_clipped: Tensor
    material_magnitude_clipped: Tensor
    material_rate_clipped: Tensor
    constraint_violation_count: Tensor
    saturated: Tensor
    material_saturated: Tensor


@dataclass(frozen=True)
class BatchedPureActionApplication:
    raw_normalized: Tensor
    normalized: Tensor
    requested_total: Tensor
    total: Tensor
    normalized_magnitude_clipped: Tensor
    magnitude_clipped: Tensor
    rate_clipped: Tensor
    material_magnitude_clipped: Tensor
    material_rate_clipped: Tensor
    constraint_violation_count: Tensor
    saturated: Tensor
    material_saturated: Tensor


class BatchedActionPipeline:
    def __init__(
        self,
        config: LowLevelConfig,
        limits: ActuatorLimits,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        validate_finite: bool = True,
    ) -> None:
        self.config = config
        self.limits = limits
        self.device = torch.device(device)
        self.dtype = dtype
        self.validate_finite = bool(validate_finite)
        self.full_scale = torch.tensor(
            [
                limits.max_rudder_angle_rad,
                limits.max_stern_plane_angle_rad,
                limits.max_propeller_rpm,
            ],
            device=self.device,
            dtype=dtype,
        )
        self.lower = torch.tensor(
            [-limits.max_rudder_angle_rad, -limits.max_stern_plane_angle_rad, 0.0],
            device=self.device,
            dtype=dtype,
        )
        self.upper = self.full_scale
        self.scales = torch.tensor(
            [config.residual.scale_rudder, config.residual.scale_stern, config.residual.scale_rpm],
            device=self.device,
            dtype=dtype,
        )
        self.magnitude_tolerance = torch.tensor([1e-6, 1e-6, 1e-3], device=self.device, dtype=dtype)
        self.material_tolerance = limits.material_tolerance_fraction * self.full_scale
        self.rate_tolerance = torch.tensor(
            [1e-6, 1e-6, torch.inf], device=self.device, dtype=dtype
        )
        self._max_fin_rate_per_s = torch.tensor(
            [
                limits.max_rudder_rate_rad_s,
                limits.max_stern_plane_rate_rad_s,
            ],
            device=self.device,
            dtype=dtype,
        )

    def apply(
        self,
        base: Tensor,
        residual_normalized: Tensor,
        previous_total: Tensor,
        dt_s: float,
    ) -> BatchedActionApplication:
        base_t = torch.as_tensor(base, device=self.device, dtype=self.dtype)
        residual = torch.as_tensor(residual_normalized, device=self.device, dtype=self.dtype)
        previous = torch.as_tensor(previous_total, device=self.device, dtype=self.dtype)
        if base_t.ndim != 2 or base_t.shape[1] != 3:
            raise ValueError("base must have shape (N, 3)")
        if residual.shape != base_t.shape or previous.shape != base_t.shape:
            raise ValueError("residual and previous_total must have shape (N, 3)")
        if self.validate_finite and (
            not torch.isfinite(base_t).all() or not torch.isfinite(residual).all()
        ):
            raise ValueError("base and residual must contain finite values")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        raw_residual = residual
        residual = torch.clamp(residual, -self.config.residual.max_action, self.config.residual.max_action)
        residual_clipped = torch.abs(residual - raw_residual) > 1e-6
        delta = residual * self.scales * self.full_scale
        requested = base_t + delta
        magnitude_limited = torch.clamp(requested, self.lower, self.upper)
        magnitude_clipped = torch.abs(magnitude_limited - requested) > self.magnitude_tolerance
        material_magnitude_clipped = (
            torch.abs(magnitude_limited - requested) > self.material_tolerance
        )

        rate_limited = magnitude_limited.clone()
        max_fin_step = self._max_fin_rate_per_s * dt_s
        fin_change = torch.clamp(
            rate_limited[:, :2] - previous[:, :2],
            -max_fin_step,
            max_fin_step,
        )
        rate_limited[:, :2] = previous[:, :2] + fin_change
        rate_clipped = torch.abs(rate_limited - magnitude_limited) > self.rate_tolerance
        material_rate_clipped = (
            torch.abs(rate_limited - magnitude_limited) > self.material_tolerance
        )
        violation_count = torch.sum(magnitude_clipped, dim=1) + torch.sum(rate_clipped, dim=1)
        saturated = torch.any(magnitude_clipped, dim=1) | torch.any(rate_clipped, dim=1)
        material_saturated = torch.any(material_magnitude_clipped, dim=1) | torch.any(
            material_rate_clipped, dim=1
        )
        return BatchedActionApplication(
            base=base_t,
            raw_residual_normalized=raw_residual,
            residual_normalized=residual,
            delta=delta,
            requested_total=requested,
            total=rate_limited,
            residual_magnitude_clipped=residual_clipped,
            magnitude_clipped=magnitude_clipped,
            rate_clipped=rate_clipped,
            material_magnitude_clipped=material_magnitude_clipped,
            material_rate_clipped=material_rate_clipped,
            constraint_violation_count=violation_count,
            saturated=saturated,
            material_saturated=material_saturated,
        )


class BatchedPureActionPipeline:
    """Vectorized normalized-to-actuator mapping for the PC arm."""

    def __init__(
        self,
        limits: ActuatorLimits,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        validate_finite: bool = True,
    ) -> None:
        self.limits = limits
        self.device = torch.device(device)
        self.dtype = dtype
        self.validate_finite = bool(validate_finite)
        self.full_scale = torch.tensor(
            [
                limits.max_rudder_angle_rad,
                limits.max_stern_plane_angle_rad,
                limits.max_propeller_rpm,
            ],
            device=self.device,
            dtype=dtype,
        )
        self.lower = torch.tensor(
            [-limits.max_rudder_angle_rad, -limits.max_stern_plane_angle_rad, 0.0],
            device=self.device,
            dtype=dtype,
        )
        self.upper = self.full_scale
        self.magnitude_tolerance = torch.tensor(
            [1e-6, 1e-6, 1e-3], device=self.device, dtype=dtype
        )
        self.material_tolerance = limits.material_tolerance_fraction * self.full_scale
        self.rate_tolerance = torch.tensor(
            [1e-6, 1e-6, torch.inf], device=self.device, dtype=dtype
        )
        self._max_fin_rate_per_s = torch.tensor(
            [limits.max_rudder_rate_rad_s, limits.max_stern_plane_rate_rad_s],
            device=self.device,
            dtype=dtype,
        )

    def to_actuator(self, action_normalized: Tensor) -> Tensor:
        action = torch.as_tensor(action_normalized, device=self.device, dtype=self.dtype)
        if action.ndim != 2 or action.shape[1] != 3:
            raise ValueError("action_normalized must have shape (N, 3)")
        if self.validate_finite and not torch.isfinite(action).all():
            raise ValueError("action_normalized contains NaN or Inf")
        clipped = torch.clamp(action, -1.0, 1.0)
        rpm = self.limits.max_propeller_rpm * 0.5 * (clipped[:, 2] + 1.0)
        return torch.stack(
            [
                clipped[:, 0] * self.limits.max_rudder_angle_rad,
                clipped[:, 1] * self.limits.max_stern_plane_angle_rad,
                rpm,
            ],
            dim=1,
        )

    def apply(
        self,
        action_normalized: Tensor,
        previous_total: Tensor,
        dt_s: float,
    ) -> BatchedPureActionApplication:
        action = torch.as_tensor(action_normalized, device=self.device, dtype=self.dtype)
        previous = torch.as_tensor(previous_total, device=self.device, dtype=self.dtype)
        if action.ndim != 2 or action.shape[1] != 3:
            raise ValueError("action_normalized must have shape (N, 3)")
        if previous.shape != action.shape:
            raise ValueError("previous_total must match action_normalized")
        if self.validate_finite and (
            not torch.isfinite(action).all() or not torch.isfinite(previous).all()
        ):
            raise ValueError("action_normalized and previous_total contain NaN or Inf")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        clipped = torch.clamp(action, -1.0, 1.0)
        normalized_clipped = torch.abs(clipped - action) > 1e-6
        requested = self.to_actuator(clipped)
        magnitude_limited = torch.clamp(requested, self.lower, self.upper)
        magnitude_clipped = torch.abs(magnitude_limited - requested) > self.magnitude_tolerance
        material_magnitude_clipped = (
            torch.abs(magnitude_limited - requested) > self.material_tolerance
        )

        rate_limited = magnitude_limited.clone()
        max_fin_step = self._max_fin_rate_per_s * dt_s
        fin_change = torch.clamp(
            rate_limited[:, :2] - previous[:, :2],
            -max_fin_step,
            max_fin_step,
        )
        rate_limited[:, :2] = previous[:, :2] + fin_change
        rate_clipped = torch.abs(rate_limited - magnitude_limited) > self.rate_tolerance
        material_rate_clipped = (
            torch.abs(rate_limited - magnitude_limited) > self.material_tolerance
        )
        violation_count = (
            torch.sum(normalized_clipped, dim=1)
            + torch.sum(magnitude_clipped, dim=1)
            + torch.sum(rate_clipped, dim=1)
        )
        saturated = torch.any(magnitude_clipped, dim=1) | torch.any(rate_clipped, dim=1)
        material_saturated = torch.any(material_magnitude_clipped, dim=1) | torch.any(
            material_rate_clipped, dim=1
        )
        return BatchedPureActionApplication(
            raw_normalized=action,
            normalized=clipped,
            requested_total=requested,
            total=rate_limited,
            normalized_magnitude_clipped=normalized_clipped,
            magnitude_clipped=magnitude_clipped,
            rate_clipped=rate_clipped,
            material_magnitude_clipped=material_magnitude_clipped,
            material_rate_clipped=material_rate_clipped,
            constraint_violation_count=violation_count,
            saturated=saturated,
            material_saturated=material_saturated,
        )


__all__ = [
    "BatchedActionApplication",
    "BatchedActionPipeline",
    "BatchedPureActionApplication",
    "BatchedPureActionPipeline",
]
