from __future__ import annotations

from math import atan2, cos, sin

import numpy as np
from auv_core.attitude import euler_from_quaternion
from auv_core.contracts import FloatArray, Reference, State
from auv_core.guidance import AlosSolution

from .config import LowLevelConfig
from .path import ReferencePath

PREVIOUS_RESIDUAL_OBSERVATION_INDICES = (23, 24, 25)


def wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def body_x_direction(quat_xyzw: FloatArray) -> FloatArray:
    x, y, z, w = quat_xyzw
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        dtype=np.float64,
    )


def yaw_pitch(quat_xyzw: FloatArray) -> tuple[float, float]:
    direction = body_x_direction(quat_xyzw)
    yaw = atan2(float(direction[1]), float(direction[0]))
    pitch = atan2(-float(direction[2]), float(np.hypot(direction[0], direction[1])))
    return yaw, pitch


def world_to_body(vector_ned: FloatArray, quat_xyzw: FloatArray) -> FloatArray:
    x, y, z, w = quat_xyzw
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation.T @ vector_ned


def body_heading_direction_ned(quat_xyzw: FloatArray) -> FloatArray:
    yaw, _ = yaw_pitch(quat_xyzw)
    return np.array([cos(yaw), sin(yaw), 0.0], dtype=np.float64)


def speed_feedback_surge(
    *,
    speed_feedback: str,
    linear_body_mps: FloatArray,
    current_body_mps: FloatArray,
) -> float:
    """Return the surge used for the actor's speed feedback.

    ``water_relative`` preserves the historical contract and is only valid for
    oracle or explicitly legacy experiments. ``ground`` removes the true
    current from the feedback path so the actor cannot recover it from
    ``linear_x - water_surge``. The current may still enter reward, a
    privileged critic, a teacher, or an auxiliary target.
    """

    surge = float(linear_body_mps[0])
    if speed_feedback == "ground":
        return surge
    if speed_feedback == "water_relative":
        return float(surge - current_body_mps[0])
    raise ValueError(f"unsupported observation.speed_feedback={speed_feedback!r}")


class ObservationBuilder:
    def __init__(self, config: LowLevelConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        state: State,
        reference: Reference,
        solution: AlosSolution,
        path: ReferencePath,
        current_ned_mps: FloatArray,
        previous_actuator: FloatArray,
        previous_residual: FloatArray,
        speed_integral_rpm: float,
        expose_current: bool | None = None,
    ) -> FloatArray:
        quat = state.quat_xyzw
        desired_yaw, desired_pitch = yaw_pitch(reference.quat_xyzw)
        actual_yaw, actual_pitch = yaw_pitch(quat)
        euler = euler_from_quaternion(quat)
        heading_error = wrap_angle(desired_yaw - actual_yaw)
        pitch_error = desired_pitch - actual_pitch
        current_body = world_to_body(current_ned_mps, quat)
        linear = state.linear_velocity_body_mps
        angular = state.angular_velocity_body_rps
        feedback_surge = speed_feedback_surge(
            speed_feedback=self.config.observation.speed_feedback,
            linear_body_mps=linear,
            current_body_mps=current_body,
        )
        speed_error = float(reference.linear_velocity_body_mps[0] - feedback_surge)
        show_current = (
            self.config.observation.expose_current if expose_current is None else expose_current
        )
        current_for_observation = current_body if show_current else np.zeros(3)
        full_scale = np.array(
            [
                self.config.actuator.policy_max_rudder_angle_rad,
                self.config.actuator.policy_max_stern_plane_angle_rad,
                self.config.actuator.policy_max_propeller_rpm,
            ],
            dtype=np.float64,
        )
        previous = np.asarray(previous_actuator, dtype=np.float64)
        residual = np.asarray(previous_residual, dtype=np.float64)
        if previous.shape != (3,) or residual.shape != (3,):
            raise ValueError("previous actuator and residual must have shape (3,)")

        columns = [
            heading_error / np.pi,
            sin(heading_error),
            cos(heading_error),
            pitch_error / (np.pi / 2.0),
            float(euler[0]) / (np.pi / 2.0),
            (float(reference.angular_velocity_body_rps[2]) - float(angular[2])) / 0.5,
            (float(reference.angular_velocity_body_rps[1]) - float(angular[1])) / 0.5,
            float(angular[0]) / 0.5,
            float(linear[0]) / 2.5,
            float(linear[1]) / 1.0,
            float(linear[2]) / 1.0,
            feedback_surge / 2.5,
            speed_error / 1.0,
            solution.horizontal_error_m / 2.0,
            solution.vertical_error_m / 2.0,
            solution.projection_s_m / path.total_length_m,
            path.curvature_at_s(solution.projection_s_m) / 1.0,
            float(current_for_observation[0]) / 0.5,
            float(current_for_observation[1]) / 0.5,
            float(current_for_observation[2]) / 0.5,
            float(previous[0]) / full_scale[0],
            float(previous[1]) / full_scale[1],
            float(previous[2]) / full_scale[2],
            float(residual[0]),
            float(residual[1]),
            float(residual[2]),
        ]
        if self.config.observation.dim == 27:
            columns.append(
                float(speed_integral_rpm) / self.config.controller.max_speed_integral_rpm
            )
        observation = np.asarray(columns, dtype=np.float64)
        if observation.shape != (self.config.observation.dim,):
            raise RuntimeError("observation builder violated the configured dimension")
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation contains NaN or Inf")
        return observation


__all__ = [
    "PREVIOUS_RESIDUAL_OBSERVATION_INDICES",
    "ObservationBuilder",
    "body_heading_direction_ned",
    "body_x_direction",
    "speed_feedback_surge",
    "world_to_body",
    "wrap_angle",
    "yaw_pitch",
]
