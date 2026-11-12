from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ControllerConfig


@dataclass(frozen=True)
class ShadowIntegratorState:
    speed_integral_rpm: float
    speed_error_mps: float
    unsaturated_rpm: float
    updated: bool


class ShadowSpeedIntegrator:
    """Reproduce the PD/PI speed integral using deployable observations only."""

    def __init__(self, config: ControllerConfig, max_propeller_rpm: float) -> None:
        if not np.isfinite(max_propeller_rpm) or max_propeller_rpm <= 0.0:
            raise ValueError("max_propeller_rpm must be finite and positive")
        self.config = config
        self.max_propeller_rpm = float(max_propeller_rpm)
        self.reset()

    @property
    def value_rpm(self) -> float:
        return self._speed_integral_rpm

    @property
    def last_state(self) -> ShadowIntegratorState | None:
        return self._last_state

    def reset(self) -> None:
        self._speed_integral_rpm = 0.0
        self._last_state: ShadowIntegratorState | None = None

    def advance(
        self,
        water_relative_surge_mps: float,
        desired_water_speed_mps: float,
        dt_s: float,
    ) -> ShadowIntegratorState:
        if not np.isfinite(water_relative_surge_mps):
            raise ValueError("water_relative_surge_mps must be finite")
        if not np.isfinite(desired_water_speed_mps):
            raise ValueError("desired_water_speed_mps must be finite")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        speed_error = float(desired_water_speed_mps - water_relative_surge_mps)
        unsaturated_rpm = float(
            self.config.speed_feedforward_rpm_per_mps * desired_water_speed_mps
            + self._speed_integral_rpm
            + self.config.speed_gain_rpm_per_mps * speed_error
        )
        saturated_high = unsaturated_rpm > self.max_propeller_rpm and speed_error > 0.0
        saturated_low = unsaturated_rpm < 0.0 and speed_error < 0.0
        updated = not saturated_high and not saturated_low
        if updated:
            self._speed_integral_rpm += (
                self.config.speed_integral_gain_rpm_per_mps_s * speed_error * dt_s
            )
            self._speed_integral_rpm = float(
                np.clip(
                    self._speed_integral_rpm,
                    -self.config.max_speed_integral_rpm,
                    self.config.max_speed_integral_rpm,
                )
            )
        self._last_state = ShadowIntegratorState(
            speed_integral_rpm=self._speed_integral_rpm,
            speed_error_mps=speed_error,
            unsaturated_rpm=unsaturated_rpm,
            updated=updated,
        )
        return self._last_state


__all__ = ["ShadowIntegratorState", "ShadowSpeedIntegrator"]
