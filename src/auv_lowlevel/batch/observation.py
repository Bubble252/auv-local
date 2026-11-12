"""Batched low-level observation schemas and builder."""

from __future__ import annotations

import torch
from torch import Tensor

from auv_lowlevel.config import LowLevelConfig
from auv_lowlevel.observation_schema import (
    OBSERVATION_SCHEMA_V1,
    OBSERVATION_SCHEMA_V2,
    ObservationSchema,
    observation_schema_for_dim,
)

from .geometry import (
    euler_from_quaternion,
    quaternion_from_euler,
    world_to_body,
    wrap_angle,
    yaw_pitch,
)
from .guidance import BatchedAlosSolution
from .path import BatchedPathProjection, BatchedReferencePath, BatchedReferencePathSet


class BatchedObservationBuilder:
    """Vectorized counterpart of ``auv_lowlevel.observation.ObservationBuilder``."""

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        schema: ObservationSchema | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        validate_finite: bool = True,
    ) -> None:
        self.config = config
        self.schema = (
            observation_schema_for_dim(config.observation.dim)
            if schema is None
            else schema
        )
        self.device = torch.device(device)
        self.dtype = dtype
        self.validate_finite = bool(validate_finite)
        self.full_scale = torch.tensor(
            [
                config.actuator.policy_max_rudder_angle_rad,
                config.actuator.policy_max_stern_plane_angle_rad,
                config.actuator.policy_max_propeller_rpm,
            ],
            device=self.device,
            dtype=dtype,
        )

    def build(
        self,
        *,
        eta: Tensor,
        nu: Tensor,
        path: BatchedReferencePath | BatchedReferencePathSet,
        projection: BatchedPathProjection,
        solution: BatchedAlosSolution,
        current_ned_mps: Tensor,
        previous_actuator: Tensor,
        previous_residual: Tensor,
        speed_integral_rpm: Tensor | None,
        expose_current: bool | None = None,
    ) -> Tensor:
        eta = torch.as_tensor(eta, device=self.device, dtype=self.dtype)
        nu = torch.as_tensor(nu, device=self.device, dtype=self.dtype)
        current = torch.as_tensor(current_ned_mps, device=self.device, dtype=self.dtype)
        previous = torch.as_tensor(previous_actuator, device=self.device, dtype=self.dtype)
        residual = torch.as_tensor(previous_residual, device=self.device, dtype=self.dtype)
        if eta.ndim != 2 or eta.shape[1] != 6 or nu.shape != eta.shape:
            raise ValueError("eta and nu must have shape (N, 6)")
        if current.shape != (eta.shape[0], 3):
            raise ValueError("current_ned_mps must have shape (N, 3)")
        if previous.shape != (eta.shape[0], 3) or residual.shape != previous.shape:
            raise ValueError("previous actuator and residual must have shape (N, 3)")
        if self.schema.include_speed_integral and speed_integral_rpm is None:
            raise ValueError("v1 observation requires speed_integral_rpm")

        quat = quaternion_from_euler(eta[:, 3], eta[:, 4], eta[:, 5])
        desired_yaw, desired_pitch = yaw_pitch(solution.reference_quat_xyzw)
        actual_yaw, actual_pitch = yaw_pitch(quat)
        euler = euler_from_quaternion(quat)
        heading_error = wrap_angle(desired_yaw - actual_yaw)
        pitch_error = desired_pitch - actual_pitch
        current_body = world_to_body(current, quat)
        linear = nu[:, :3]
        angular = nu[:, 3:]
        if self.config.observation.speed_feedback == "ground":
            feedback_surge = linear[:, 0]
        elif self.config.observation.speed_feedback == "water_relative":
            feedback_surge = linear[:, 0] - current_body[:, 0]
        else:
            raise ValueError(
                "unsupported observation.speed_feedback="
                f"{self.config.observation.speed_feedback!r}"
            )
        speed_error = solution.reference_speed_mps - feedback_surge
        show_current = (
            self.config.observation.expose_current
            if expose_current is None
            else expose_current
        )
        current_for_observation = current_body if show_current else torch.zeros_like(current_body)

        columns = [
            heading_error / torch.pi,
            torch.sin(heading_error),
            torch.cos(heading_error),
            pitch_error / (torch.pi / 2.0),
            euler[:, 0] / (torch.pi / 2.0),
            -angular[:, 2] / 0.5,
            -angular[:, 1] / 0.5,
            angular[:, 0] / 0.5,
            linear[:, 0] / 2.5,
            linear[:, 1],
            linear[:, 2],
            feedback_surge / 2.5,
            speed_error,
            solution.horizontal_error_m / 2.0,
            solution.vertical_error_m / 2.0,
            solution.projection_s_m / path.total_length_m_tensor,
            path.curvature_at_s(projection.progress_m),
            current_for_observation[:, 0] / 0.5,
            current_for_observation[:, 1] / 0.5,
            current_for_observation[:, 2] / 0.5,
            previous[:, 0] / self.full_scale[0],
            previous[:, 1] / self.full_scale[1],
            previous[:, 2] / self.full_scale[2],
            residual[:, 0],
            residual[:, 1],
            residual[:, 2],
        ]
        if self.schema.include_speed_integral:
            assert speed_integral_rpm is not None
            integral = torch.as_tensor(
                speed_integral_rpm, device=self.device, dtype=self.dtype
            )
            if integral.shape != (eta.shape[0],):
                raise ValueError("speed_integral_rpm must have shape (N,)")
            columns.append(integral / self.config.controller.max_speed_integral_rpm)
        observation = torch.stack(columns, dim=1)
        if observation.shape != (eta.shape[0], self.schema.dim):
            raise RuntimeError("batched observation builder violated its schema")
        if self.validate_finite and not bool(torch.isfinite(observation).all()):
            raise ValueError("observation contains NaN or Inf")
        return observation


__all__ = [
    "BatchedObservationBuilder",
    "OBSERVATION_SCHEMA_V1",
    "OBSERVATION_SCHEMA_V2",
    "ObservationSchema",
    "observation_schema_for_dim",
]
