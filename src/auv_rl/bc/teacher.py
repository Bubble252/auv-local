from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, pi

import numpy as np
from auv_core.attitude import euler_from_quaternion
from auv_core.contracts import Obstacle

from ..envs.planar_avoidance import PrivilegedSceneState


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + pi) % (2.0 * pi) - pi)


@dataclass(frozen=True)
class PotentialFieldTeacherConfig:
    """Deterministic oracle teacher used only to collect BC demonstrations."""

    desired_speed_mps: float = 1.2
    accel_gain: float = 1.0
    heading_gain_rps_per_rad: float = 1.4
    obstacle_influence_m: float = 4.0
    obstacle_margin_m: float = 0.8
    center_tie_break_m: float = 0.0
    preferred_turn_side: str = "right"
    obstacle_gain: float = 2.2
    boundary_influence_m: float = 1.5
    boundary_gain: float = 1.5
    max_accel_mps2: float = 0.5
    max_yaw_rate_rps: float = 0.2617993877991494

    def __post_init__(self) -> None:
        for name in (
            "desired_speed_mps",
            "accel_gain",
            "heading_gain_rps_per_rad",
            "obstacle_influence_m",
            "obstacle_margin_m",
            "center_tie_break_m",
            "obstacle_gain",
            "boundary_influence_m",
            "boundary_gain",
            "max_accel_mps2",
            "max_yaw_rate_rps",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.preferred_turn_side not in {"left", "right"}:
            raise ValueError("preferred_turn_side must be 'left' or 'right'")


def _obstacle_radius_m(obstacle: Obstacle) -> float:
    if obstacle.kind == "sphere":
        assert obstacle.radius_m is not None
        return float(obstacle.radius_m)
    if obstacle.kind == "cylinder":
        assert obstacle.radius_m is not None
        return float(obstacle.radius_m)
    assert obstacle.half_extents_m is not None
    return float(np.linalg.norm(obstacle.half_extents_m[:2]))


class PotentialFieldTeacher:
    def __init__(self, config: PotentialFieldTeacherConfig | None = None) -> None:
        self.config = PotentialFieldTeacherConfig() if config is None else config
        if not isinstance(self.config, PotentialFieldTeacherConfig):
            raise TypeError("config must be a PotentialFieldTeacherConfig")

    def act(self, scene: PrivilegedSceneState) -> np.ndarray:
        if not isinstance(scene, PrivilegedSceneState):
            raise TypeError("scene must be a PrivilegedSceneState")
        state = scene.state
        position = state.position_ned_m
        goal_delta = scene.goal_ned_m - position
        goal_distance = float(np.linalg.norm(goal_delta[:2]))
        if goal_distance <= 1e-9:
            return np.zeros(2, dtype=np.float64)
        goal_direction = goal_delta[:2] / goal_distance
        desired = goal_direction.copy()

        for obstacle in scene.obstacle_map.obstacles:
            center = obstacle.center_ned_m[:2]
            offset = position[:2] - center
            distance = float(np.linalg.norm(offset))
            surface_distance = distance - _obstacle_radius_m(obstacle)
            if surface_distance >= self.config.obstacle_influence_m:
                continue
            to_obstacle = center - position[:2]
            if float(np.dot(to_obstacle, goal_direction)) < -self.config.obstacle_margin_m:
                continue
            if distance <= 1e-9:
                radial = np.array([0.0, 1.0], dtype=np.float64)
            else:
                radial = offset / distance
            left_normal = np.array([-goal_direction[1], goal_direction[0]], dtype=np.float64)
            lateral_offset = float(np.dot(to_obstacle, left_normal))
            if (
                self.config.center_tie_break_m > 0.0
                and abs(lateral_offset) < self.config.center_tie_break_m
            ):
                tangent = left_normal if self.config.preferred_turn_side == "left" else -left_normal
            else:
                tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
                if float(np.dot(tangent, goal_direction)) < 0.0:
                    tangent = -tangent
            strength = self.config.obstacle_gain * (
                1.0 - max(surface_distance, 0.0) / self.config.obstacle_influence_m
            )
            desired += strength * (0.35 * radial + tangent)

        bounds = scene.obstacle_map.bounds_ned_m[:2]
        for axis in range(2):
            lower = position[axis] - bounds[axis, 0]
            upper = bounds[axis, 1] - position[axis]
            if lower < self.config.boundary_influence_m:
                desired[axis] += self.config.boundary_gain * (
                    1.0 - max(lower, 0.0) / self.config.boundary_influence_m
                )
            if upper < self.config.boundary_influence_m:
                desired[axis] -= self.config.boundary_gain * (
                    1.0 - max(upper, 0.0) / self.config.boundary_influence_m
                )

        desired_norm = float(np.linalg.norm(desired))
        if desired_norm <= 1e-9:
            desired = goal_direction
        else:
            desired = desired / desired_norm
        yaw = float(euler_from_quaternion(state.quat_xyzw)[2])
        heading_error = _wrap_angle(atan2(float(desired[1]), float(desired[0])) - yaw)
        yaw_rate = float(
            np.clip(
                self.config.heading_gain_rps_per_rad * heading_error,
                -self.config.max_yaw_rate_rps,
                self.config.max_yaw_rate_rps,
            )
        )
        speed_target = self.config.desired_speed_mps
        if abs(heading_error) > 0.5 * pi:
            speed_target = 1.0
        surge = float(state.linear_velocity_body_mps[0])
        acceleration = float(
            np.clip(
                self.config.accel_gain * (speed_target - surge),
                -self.config.max_accel_mps2,
                self.config.max_accel_mps2,
            )
        )
        return np.array([acceleration, yaw_rate], dtype=np.float64)


__all__ = ["PotentialFieldTeacher", "PotentialFieldTeacherConfig"]
