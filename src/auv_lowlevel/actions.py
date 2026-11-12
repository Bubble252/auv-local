from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from auv_core.contracts import FloatArray
from numpy.typing import NDArray

from .config import LowLevelConfig


@dataclass(frozen=True)
class ActuatorLimits:
    max_rudder_angle_rad: float
    max_stern_plane_angle_rad: float
    max_rudder_rate_rad_s: float
    max_stern_plane_rate_rad_s: float
    max_propeller_rpm: float
    material_tolerance_fraction: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.max_rudder_angle_rad,
            self.max_stern_plane_angle_rad,
            self.max_rudder_rate_rad_s,
            self.max_stern_plane_rate_rad_s,
            self.max_propeller_rpm,
            self.material_tolerance_fraction,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("actuator limits and tolerance must be finite and positive")
        if self.material_tolerance_fraction >= 1.0:
            raise ValueError("material_tolerance_fraction must be below 1")


@dataclass(frozen=True)
class ActionApplication:
    base: FloatArray
    raw_residual_normalized: FloatArray
    residual_normalized: FloatArray
    delta: FloatArray
    requested_total: FloatArray
    total: FloatArray
    residual_magnitude_clipped: NDArray[np.bool_]
    magnitude_clipped: NDArray[np.bool_]
    rate_clipped: NDArray[np.bool_]
    material_magnitude_clipped: NDArray[np.bool_]
    material_rate_clipped: NDArray[np.bool_]
    constraint_violation_count: int

    @property
    def saturated(self) -> bool:
        return bool(np.any(self.magnitude_clipped) or np.any(self.rate_clipped))

    @property
    def residual_clipped(self) -> bool:
        return bool(np.any(self.residual_magnitude_clipped))

    @property
    def material_saturated(self) -> bool:
        return bool(
            np.any(self.material_magnitude_clipped)
            or np.any(self.material_rate_clipped)
        )


@dataclass(frozen=True)
class PureActionApplication:
    """Result of mapping a normalized pure-controller action to actuators."""

    raw_normalized: FloatArray
    normalized: FloatArray
    requested_total: FloatArray
    total: FloatArray
    normalized_magnitude_clipped: NDArray[np.bool_]
    magnitude_clipped: NDArray[np.bool_]
    rate_clipped: NDArray[np.bool_]
    material_magnitude_clipped: NDArray[np.bool_]
    material_rate_clipped: NDArray[np.bool_]
    constraint_violation_count: int

    @property
    def saturated(self) -> bool:
        return bool(np.any(self.magnitude_clipped) or np.any(self.rate_clipped))

    @property
    def material_saturated(self) -> bool:
        return bool(
            np.any(self.material_magnitude_clipped)
            or np.any(self.material_rate_clipped)
        )


class ActionPipeline:
    """Map normalized residual actions to checked REMUS100 actuator commands.

    Clipping is reported only above a physical tolerance. A network that lands
    on the limit by floating-point noise must not be counted as saturated.
    """

    def __init__(self, config: LowLevelConfig, limits: ActuatorLimits) -> None:
        self.config = config
        self.limits = limits
        self.full_scale = np.array(
            [
                limits.max_rudder_angle_rad,
                limits.max_stern_plane_angle_rad,
                limits.max_propeller_rpm,
            ],
            dtype=np.float64,
        )
        self.lower = np.array(
            [
                -limits.max_rudder_angle_rad,
                -limits.max_stern_plane_angle_rad,
                0.0,
            ],
            dtype=np.float64,
        )
        self.upper = self.full_scale.copy()

    def apply(
        self,
        base: FloatArray,
        residual_normalized: FloatArray,
        previous_total: FloatArray,
        dt_s: float,
    ) -> ActionApplication:
        base_array = np.asarray(base, dtype=np.float64)
        residual = np.asarray(residual_normalized, dtype=np.float64)
        previous = np.asarray(previous_total, dtype=np.float64)
        if base_array.shape != (3,) or residual.shape != (3,) or previous.shape != (3,):
            raise ValueError("base, residual, and previous_total must have shape (3,)")
        if not np.all(np.isfinite(base_array)) or not np.all(np.isfinite(residual)):
            raise ValueError("base and residual must contain finite values")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        raw_residual = residual.copy()
        residual = np.clip(
            residual,
            -self.config.residual.max_action,
            self.config.residual.max_action,
        )
        residual_magnitude_clipped = np.abs(residual - raw_residual) > 1e-6
        scales = np.array(
            [
                self.config.residual.scale_rudder,
                self.config.residual.scale_stern,
                self.config.residual.scale_rpm,
            ],
            dtype=np.float64,
        )
        delta = residual * scales * self.full_scale
        requested = base_array + delta
        magnitude_limited = np.clip(requested, self.lower, self.upper)
        magnitude_tolerance = np.array([1e-6, 1e-6, 1e-3], dtype=np.float64)
        magnitude_clipped = np.abs(magnitude_limited - requested) > magnitude_tolerance
        material_tolerance = self.limits.material_tolerance_fraction * self.full_scale
        material_magnitude_clipped = (
            np.abs(magnitude_limited - requested) > material_tolerance
        )

        rate_limited = magnitude_limited.copy()
        max_fin_step = np.array(
            [
                self.limits.max_rudder_rate_rad_s * dt_s,
                self.limits.max_stern_plane_rate_rad_s * dt_s,
            ],
            dtype=np.float64,
        )
        for index in range(2):
            rate_limited[index] = previous[index] + float(
                np.clip(
                    rate_limited[index] - previous[index],
                    -max_fin_step[index],
                    max_fin_step[index],
                )
            )
        rate_tolerance = np.array([1e-6, 1e-6, np.inf], dtype=np.float64)
        rate_clipped = np.abs(rate_limited - magnitude_limited) > rate_tolerance
        material_rate_clipped = np.abs(rate_limited - magnitude_limited) > material_tolerance
        violations = int(np.count_nonzero(magnitude_clipped) + np.count_nonzero(rate_clipped))
        return ActionApplication(
            base=base_array,
            raw_residual_normalized=raw_residual,
            residual_normalized=residual,
            delta=delta,
            requested_total=requested,
            total=rate_limited,
            residual_magnitude_clipped=residual_magnitude_clipped,
            magnitude_clipped=magnitude_clipped,
            rate_clipped=rate_clipped,
            material_magnitude_clipped=material_magnitude_clipped,
            material_rate_clipped=material_rate_clipped,
            constraint_violation_count=violations,
        )


class PureActionPipeline:
    """Map ``[a_r, a_s, a_n]`` in ``[-1, 1]`` to actuator commands.

    The mapping is the PC20 contract from Goal9::

        delta_r = 20 deg * clip(a_r, -1, 1)
        delta_s = 20 deg * clip(a_s, -1, 1)
        rpm     = 1525 * 0.5 * (clip(a_n, -1, 1) + 1)

    The action vector is normalized in the same units as the policy output.
    Fin rates are limited after magnitude clipping. The propeller has no
    additional rate limiter here because the plant already models first-order
    RPM lag.
    """

    def __init__(self, limits: ActuatorLimits) -> None:
        self.limits = limits
        self.full_scale = np.array(
            [
                limits.max_rudder_angle_rad,
                limits.max_stern_plane_angle_rad,
                limits.max_propeller_rpm,
            ],
            dtype=np.float64,
        )
        self.lower = np.array(
            [
                -limits.max_rudder_angle_rad,
                -limits.max_stern_plane_angle_rad,
                0.0,
            ],
            dtype=np.float64,
        )
        self.upper = self.full_scale.copy()

    def to_actuator(self, action_normalized: FloatArray) -> FloatArray:
        action = np.asarray(action_normalized, dtype=np.float64)
        if action.shape != (3,):
            raise ValueError("action_normalized must have shape (3,)")
        if not np.all(np.isfinite(action)):
            raise ValueError("action_normalized must contain finite values")
        clipped = np.clip(action, -1.0, 1.0)
        return np.array(
            [
                clipped[0] * self.limits.max_rudder_angle_rad,
                clipped[1] * self.limits.max_stern_plane_angle_rad,
                self.limits.max_propeller_rpm * 0.5 * (clipped[2] + 1.0),
            ],
            dtype=np.float64,
        )

    def apply(
        self,
        action_normalized: FloatArray,
        previous_total: FloatArray,
        dt_s: float,
    ) -> PureActionApplication:
        action = np.asarray(action_normalized, dtype=np.float64)
        previous = np.asarray(previous_total, dtype=np.float64)
        if action.shape != (3,) or previous.shape != (3,):
            raise ValueError("action_normalized and previous_total must have shape (3,)")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(previous)):
            raise ValueError("action and previous_total must contain finite values")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        clipped = np.clip(action, -1.0, 1.0)
        normalized_clipped = np.abs(clipped - action) > 1e-6
        requested = self.to_actuator(clipped)
        magnitude_limited = np.clip(requested, self.lower, self.upper)
        magnitude_tolerance = np.array([1e-6, 1e-6, 1e-3], dtype=np.float64)
        magnitude_clipped = np.abs(magnitude_limited - requested) > magnitude_tolerance
        material_tolerance = self.limits.material_tolerance_fraction * self.full_scale
        material_magnitude_clipped = (
            np.abs(magnitude_limited - requested) > material_tolerance
        )

        rate_limited = magnitude_limited.copy()
        max_fin_step = np.array(
            [
                self.limits.max_rudder_rate_rad_s * dt_s,
                self.limits.max_stern_plane_rate_rad_s * dt_s,
            ],
            dtype=np.float64,
        )
        for index in range(2):
            rate_limited[index] = previous[index] + float(
                np.clip(
                    rate_limited[index] - previous[index],
                    -max_fin_step[index],
                    max_fin_step[index],
                )
            )
        rate_tolerance = np.array([1e-6, 1e-6, np.inf], dtype=np.float64)
        rate_clipped = np.abs(rate_limited - magnitude_limited) > rate_tolerance
        material_rate_clipped = np.abs(rate_limited - magnitude_limited) > material_tolerance
        violations = int(
            np.count_nonzero(normalized_clipped)
            + np.count_nonzero(magnitude_clipped)
            + np.count_nonzero(rate_clipped)
        )
        return PureActionApplication(
            raw_normalized=action.copy(),
            normalized=clipped,
            requested_total=requested,
            total=rate_limited,
            normalized_magnitude_clipped=normalized_clipped,
            magnitude_clipped=magnitude_clipped,
            rate_clipped=rate_clipped,
            material_magnitude_clipped=material_magnitude_clipped,
            material_rate_clipped=material_rate_clipped,
            constraint_violation_count=violations,
        )


__all__ = [
    "ActionApplication",
    "ActionPipeline",
    "ActuatorLimits",
    "PureActionApplication",
    "PureActionPipeline",
]
