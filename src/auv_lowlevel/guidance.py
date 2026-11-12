from __future__ import annotations

from dataclasses import replace
from math import atan2, hypot

import numpy as np
from auv_core.attitude import forward_quaternion
from auv_core.contracts import FloatArray, Reference, State
from auv_core.guidance import AlosSolution, ThreeDAlosGuidance


class TerminalCaptureGuidance:
    """Wrap ALOS so the last metres converge on the goal point.

    ``ThreeDAlosGuidance`` measures only lateral and vertical error at the
    clamped path end. Once the vehicle passes the end, its along-track
    overshoot is invisible and the desired direction stays on the terminal
    tangent, so the vehicle can fly past the goal forever. This wrapper keeps
    the upstream guidance untouched and switches to direct goal pursuit when
    the remaining Euclidean distance is small enough.
    """

    def __init__(
        self,
        inner: ThreeDAlosGuidance,
        goal_ned_m: FloatArray,
        capture_distance_m: float,
        terminal_speed_mps: float = 0.0,
    ) -> None:
        if not np.isfinite(capture_distance_m) or capture_distance_m <= 0.0:
            raise ValueError("capture_distance_m must be finite and positive")
        if not np.isfinite(terminal_speed_mps) or terminal_speed_mps < 0.0:
            raise ValueError("terminal_speed_mps must be finite and non-negative")
        goal = np.asarray(goal_ned_m, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal_ned_m must be a finite shape-(3,) vector")
        self.inner = inner
        self.goal_ned_m = goal
        self.capture_distance_m = float(capture_distance_m)
        self.terminal_speed_mps = float(terminal_speed_mps)

    def solve(self, state: State, trajectory, t_s: float) -> AlosSolution:
        base = self.inner.solve(state, trajectory, t_s)
        delta = self.goal_ned_m - state.position_ned_m
        distance = float(np.linalg.norm(delta))
        if distance > self.capture_distance_m or distance <= 1e-9:
            return base

        desired_ground_direction = delta / distance
        commanded_ground_speed = self.inner.speed_mps
        if self.terminal_speed_mps > 0.0:
            fraction = min(distance / self.capture_distance_m, 1.0)
            commanded_ground_speed = (
                self.terminal_speed_mps
                + (self.inner.speed_mps - self.terminal_speed_mps) * fraction
            )
        commanded_water_speed = commanded_ground_speed
        desired_body_direction = desired_ground_direction.copy()
        if self.inner.compensate_current and self.inner.current_field is not None:
            current = np.asarray(
                self.inner.current_field.sample(state.position_ned_m, t_s),
                dtype=np.float64,
            )
            water_relative = commanded_ground_speed * desired_ground_direction - current
            water_speed = float(np.linalg.norm(water_relative))
            if water_speed <= 1e-9:
                raise ValueError("current exactly cancels the terminal goal direction")
            commanded_water_speed = water_speed
            desired_body_direction = water_relative / water_speed

        heading = atan2(float(desired_body_direction[1]), float(desired_body_direction[0]))
        pitch = atan2(
            -float(desired_body_direction[2]),
            float(hypot(desired_body_direction[0], desired_body_direction[1])),
        )
        reference = Reference(
            stamp_s=float(t_s),
            position_ned_m=self.goal_ned_m.copy(),
            quat_xyzw=forward_quaternion(desired_body_direction),
            linear_velocity_body_mps=np.array([commanded_water_speed, 0.0, 0.0]),
            angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
            valid_until_s=float(t_s + self.inner.validity_period_s),
        )
        return replace(
            base,
            desired_heading_rad=heading,
            desired_pitch_rad=pitch,
            desired_body_direction_ned=desired_body_direction,
            command_speed_mps=commanded_water_speed,
            reference=reference,
        )

    def guide(self, state: State, trajectory, t_s: float) -> Reference:
        return self.solve(state, trajectory, t_s).reference


__all__ = ["TerminalCaptureGuidance"]
