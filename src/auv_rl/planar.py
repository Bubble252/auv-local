from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin

import numpy as np
from auv_planner.attitude import euler_from_quaternion, quaternion_from_euler
from auv_planner.contracts import FloatArray, State

from auv_local.contracts import SurgeYawAction


def _positive(value: float, name: str) -> float:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + pi) % (2.0 * pi) - pi)


@dataclass(frozen=True)
class PlanarMotionLimits:
    min_speed_mps: float = 1.0
    max_speed_mps: float = 2.5
    max_accel_mps2: float = 0.5
    max_yaw_rate_rps: float = 15.0 * pi / 180.0

    def __post_init__(self) -> None:
        minimum = _positive(self.min_speed_mps, "min_speed_mps")
        maximum = _positive(self.max_speed_mps, "max_speed_mps")
        if minimum >= maximum:
            raise ValueError("min_speed_mps must be smaller than max_speed_mps")
        object.__setattr__(self, "min_speed_mps", minimum)
        object.__setattr__(self, "max_speed_mps", maximum)
        object.__setattr__(self, "max_accel_mps2", _positive(self.max_accel_mps2, "max_accel_mps2"))
        object.__setattr__(
            self,
            "max_yaw_rate_rps",
            _positive(self.max_yaw_rate_rps, "max_yaw_rate_rps"),
        )


@dataclass(frozen=True)
class PlanarTransition:
    state: State
    applied_action: SurgeYawAction
    action_clipped: bool


@dataclass(frozen=True)
class PlanarKinematics:
    """Three-state horizontal surrogate used only for P1 pretraining."""

    limits: PlanarMotionLimits

    def __post_init__(self) -> None:
        if not isinstance(self.limits, PlanarMotionLimits):
            raise TypeError("limits must be a PlanarMotionLimits")

    def step(
        self,
        state: State,
        action: SurgeYawAction,
        dt_s: float,
        *,
        current_ned_mps: FloatArray | None = None,
    ) -> PlanarTransition:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        if not isinstance(action, SurgeYawAction):
            raise TypeError("action must be a SurgeYawAction")
        dt = _positive(dt_s, "dt_s")
        current = (
            np.zeros(3, dtype=np.float64)
            if current_ned_mps is None
            else np.asarray(current_ned_mps, dtype=np.float64)
        )
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise ValueError("current_ned_mps must be a finite shape-(3,) array")

        acceleration = float(
            np.clip(
                action.u_dot_mps2,
                -self.limits.max_accel_mps2,
                self.limits.max_accel_mps2,
            )
        )
        yaw_rate = float(
            np.clip(
                action.yaw_rate_rps,
                -self.limits.max_yaw_rate_rps,
                self.limits.max_yaw_rate_rps,
            )
        )
        clipped = (
            abs(acceleration - action.u_dot_mps2) > 1e-12
            or abs(yaw_rate - action.yaw_rate_rps) > 1e-12
        )

        yaw = float(euler_from_quaternion(state.quat_xyzw)[2])
        surge = float(
            np.clip(
                state.linear_velocity_body_mps[0] + acceleration * dt,
                self.limits.min_speed_mps,
                self.limits.max_speed_mps,
            )
        )
        next_yaw = _wrap_angle(yaw + yaw_rate * dt)
        position = state.position_ned_m + np.array(
            [
                (surge * cos(yaw) + current[0]) * dt,
                (surge * sin(yaw) + current[1]) * dt,
                0.0,
            ],
            dtype=np.float64,
        )
        next_state = State(
            t_s=state.t_s + dt,
            position_ned_m=position,
            quat_xyzw=quaternion_from_euler(0.0, 0.0, next_yaw),
            linear_velocity_body_mps=np.array([surge, 0.0, 0.0], dtype=np.float64),
            angular_velocity_body_rps=np.array([0.0, 0.0, yaw_rate], dtype=np.float64),
        )
        return PlanarTransition(
            state=next_state,
            applied_action=SurgeYawAction(acceleration, yaw_rate),
            action_clipped=clipped,
        )


__all__ = ["PlanarKinematics", "PlanarMotionLimits", "PlanarTransition"]
