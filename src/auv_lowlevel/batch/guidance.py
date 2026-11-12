"""Batched 3D ALOS guidance with terminal goal capture."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from auv_lowlevel.config import GuidanceConfig, ReferenceConfig

from .geometry import quaternion_from_direction, wrap_angle
from .path import BatchedPathProjection, BatchedReferencePath


@dataclass(frozen=True)
class BatchedAlosSolution:
    segment_index: Tensor
    projection_s_m: Tensor
    lookahead_s_m: Tensor
    horizontal_error_m: Tensor
    vertical_error_m: Tensor
    horizontal_lookahead_m: Tensor
    vertical_lookahead_m: Tensor
    desired_heading_rad: Tensor
    desired_pitch_rad: Tensor
    desired_body_direction_ned: Tensor
    command_speed_mps: Tensor
    reference_speed_mps: Tensor
    reference_position_ned_m: Tensor
    reference_quat_xyzw: Tensor
    terminal_phase: Tensor | None = None
    crab_angle_rad: Tensor | None = None


class BatchedThreeDAlosGuidance:
    """Vectorized counterpart of ``TerminalCaptureGuidance(ThreeDAlosGuidance)``."""

    def __init__(
        self,
        guidance: GuidanceConfig,
        reference: ReferenceConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        validate_finite: bool = True,
    ) -> None:
        for name in (
            "horizontal_lookahead_m",
            "vertical_lookahead_m",
            "min_lookahead_m",
            "max_lookahead_m",
            "validity_period_s",
            "terminal_capture_distance_m",
        ):
            if getattr(guidance, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        self.device = torch.device(device)
        self.dtype = dtype
        self.horizontal_lookahead_m = float(guidance.horizontal_lookahead_m)
        self.vertical_lookahead_m = float(guidance.vertical_lookahead_m)
        self.min_lookahead_m = float(guidance.min_lookahead_m)
        self.max_lookahead_m = float(guidance.max_lookahead_m)
        self.horizontal_gain = float(guidance.horizontal_gain)
        self.vertical_gain = float(guidance.vertical_gain)
        self.lookahead_error_gain = float(guidance.lookahead_error_gain)
        self.compensate_current = bool(guidance.compensate_current)
        self.terminal_capture_distance_m = float(guidance.terminal_capture_distance_m)
        self.terminal_speed_mps = float(guidance.terminal_speed_mps)
        if self.terminal_speed_mps < 0.0:
            raise ValueError("terminal_speed_mps must be non-negative")
        self.speed_mps = float(reference.speed_mps)
        self.robust_terminal_enabled = bool(guidance.robust_terminal_enabled)
        self.terminal_speed_ratio = float(guidance.terminal_speed_ratio)
        self.terminal_braking_enabled = bool(guidance.terminal_braking_enabled)
        self.minimum_speed_ratio = float(guidance.minimum_speed_ratio)
        self.terminal_alignment_horizon_s = float(guidance.terminal_alignment_horizon_s)
        self.terminal_alignment_min_m = float(guidance.terminal_alignment_min_m)
        self.terminal_alignment_max_m = float(guidance.terminal_alignment_max_m)
        self.terminal_path_preserving_enabled = bool(
            guidance.terminal_path_preserving_enabled
        )
        self.terminal_reference_extension_m = float(
            guidance.terminal_reference_extension_m
        )
        if self.terminal_reference_extension_m < 0.0:
            raise ValueError("terminal_reference_extension_m must be non-negative")
        self.current_crab_enabled = bool(guidance.current_crab_enabled)
        self.current_crab_relative_speed_limit = float(
            guidance.current_crab_relative_speed_limit
        )
        self.current_crab_max_angle_rad = float(guidance.current_crab_max_angle_rad)
        self.speed_adaptive_lookahead_enabled = bool(
            guidance.speed_adaptive_lookahead_enabled
        )
        self.high_speed_threshold_mps = float(guidance.high_speed_threshold_mps)
        self.validate_finite = bool(validate_finite)

    def solve(
        self,
        *,
        path: BatchedReferencePath,
        projection: BatchedPathProjection,
        position_ned_m: Tensor,
        current_ned_mps: Tensor,
        desired_ground_speed_mps: Tensor | float | None = None,
        estimated_current_ned_mps: Tensor | None = None,
        estimated_current_valid: Tensor | None = None,
    ) -> BatchedAlosSolution:
        position = torch.as_tensor(position_ned_m, device=self.device, dtype=self.dtype)
        current = torch.as_tensor(current_ned_mps, device=self.device, dtype=self.dtype)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position_ned_m must have shape (N, 3)")
        if current.shape != position.shape:
            raise ValueError("current_ned_mps must have shape (N, 3)")
        if desired_ground_speed_mps is None:
            desired_ground_speed = torch.full(
                (position.shape[0],),
                self.speed_mps,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            desired_ground_speed = torch.as_tensor(
                desired_ground_speed_mps,
                device=self.device,
                dtype=self.dtype,
            )
            if desired_ground_speed.ndim == 0:
                desired_ground_speed = desired_ground_speed.expand(position.shape[0])
            if desired_ground_speed.shape != (position.shape[0],):
                raise ValueError(
                    "desired_ground_speed_mps must have shape (N,) or be scalar"
                )
            if self.validate_finite and not bool(torch.isfinite(desired_ground_speed).all()):
                raise ValueError("desired_ground_speed_mps must be finite")
            if bool((desired_ground_speed <= 0.0).any()):
                raise ValueError("desired_ground_speed_mps must be positive")

        horizontal_error = projection.horizontal_error_m
        vertical_error = projection.vertical_error_m
        horizontal_lookahead = torch.clamp(
            self.horizontal_lookahead_m + self.lookahead_error_gain * torch.abs(horizontal_error),
            self.min_lookahead_m,
            self.max_lookahead_m,
        )
        vertical_lookahead = torch.clamp(
            self.vertical_lookahead_m + self.lookahead_error_gain * torch.abs(vertical_error),
            self.min_lookahead_m,
            self.max_lookahead_m,
        )
        if self.speed_adaptive_lookahead_enabled:
            high_speed = desired_ground_speed > self.high_speed_threshold_mps
            speed_excess = torch.clamp(
                desired_ground_speed - self.high_speed_threshold_mps,
                min=0.0,
            )
            adaptive_horizontal = torch.clamp(
                self.horizontal_lookahead_m
                + 1.2 * speed_excess
                + torch.abs(horizontal_error),
                1.5,
                10.0,
            )
            adaptive_vertical = torch.clamp(
                self.vertical_lookahead_m
                + 0.5 * speed_excess
                + torch.abs(vertical_error),
                1.0,
                8.0,
            )
            horizontal_lookahead = torch.where(
                high_speed, adaptive_horizontal, horizontal_lookahead
            )
            vertical_lookahead = torch.where(
                high_speed, adaptive_vertical, vertical_lookahead
            )
        tangent = projection.tangent_ned
        horizontal_norm = torch.linalg.norm(tangent[:, :2], dim=1)
        path_heading = torch.atan2(tangent[:, 1], tangent[:, 0])
        heading_correction = torch.atan2(horizontal_error, horizontal_lookahead)
        desired_heading = wrap_angle(path_heading - self.horizontal_gain * heading_correction)
        path_pitch = torch.atan2(-tangent[:, 2], horizontal_norm)
        pitch_correction = torch.atan2(vertical_error, vertical_lookahead)
        desired_pitch = torch.clamp(
            path_pitch + self.vertical_gain * pitch_correction,
            -math.pi / 2.0,
            math.pi / 2.0,
        )
        ground_direction = torch.stack(
            (
                torch.cos(desired_pitch) * torch.cos(desired_heading),
                torch.cos(desired_pitch) * torch.sin(desired_heading),
                -torch.sin(desired_pitch),
            ),
            dim=1,
        )
        command_speed, body_direction = self._water_velocity(
            ground_direction,
            current,
            desired_ground_speed=desired_ground_speed,
        )
        crab_angle = torch.zeros_like(desired_ground_speed)
        if self.current_crab_enabled:
            command_speed, body_direction, crab_angle = self._crab_water_velocity(
                ground_direction,
                command_speed,
                estimated_current_ned_mps,
                estimated_current_valid,
                fallback_angle=crab_angle,
            )
            desired_heading = torch.atan2(body_direction[:, 1], body_direction[:, 0])
            desired_pitch = torch.atan2(
                -body_direction[:, 2],
                torch.linalg.norm(body_direction[:, :2], dim=1),
            )
        lookahead_distance = torch.maximum(horizontal_lookahead, vertical_lookahead)
        lookahead_s = torch.minimum(
            projection.progress_m + lookahead_distance,
            path.total_length_m_tensor.expand_as(projection.progress_m),
        )
        reference_position = path.point_at_s(lookahead_s)

        goal = path.goal_ned_m_for(position.shape[0])
        delta = goal - position
        goal_distance = torch.linalg.norm(delta, dim=1)
        terminal_target = goal
        if self.robust_terminal_enabled:
            alignment_distance = torch.clamp(
                self.terminal_alignment_horizon_s * desired_ground_speed,
                self.terminal_alignment_min_m,
                self.terminal_alignment_max_m,
            )
            capture = (goal_distance <= alignment_distance) & (goal_distance > 1e-9)
            goal_direction = delta / torch.clamp(goal_distance[:, None], min=1e-9)
            terminal_ground = goal_direction
            if self.terminal_path_preserving_enabled:
                path_target_delta = reference_position - position
                path_target_distance = torch.linalg.norm(path_target_delta, dim=1)
                path_target_valid = path_target_distance > 1e-9
                path_target_direction = path_target_delta / torch.clamp(
                    path_target_distance[:, None], min=1e-9
                )
                terminal_target = torch.where(
                    path_target_valid[:, None], reference_position, goal
                )
                terminal_ground = torch.where(
                    path_target_valid[:, None], path_target_direction, goal_direction
                )
            elif self.terminal_reference_extension_m > 0.0:
                goal_tangent = path.goal_tangent_ned_for(position.shape[0])
                goal_tangent = torch.as_tensor(
                    goal_tangent, device=self.device, dtype=self.dtype
                )
                tangent_norm = torch.linalg.norm(goal_tangent, dim=1)
                tangent_valid = tangent_norm > 1e-9
                unit_tangent = goal_tangent / torch.clamp(
                    tangent_norm[:, None], min=1e-9
                )
                extended_target = (
                    goal + self.terminal_reference_extension_m * unit_tangent
                )
                extended_delta = extended_target - position
                extended_distance = torch.linalg.norm(extended_delta, dim=1)
                extended_valid = tangent_valid & (extended_distance > 1e-9)
                extended_direction = extended_delta / torch.clamp(
                    extended_distance[:, None], min=1e-9
                )
                terminal_target = torch.where(
                    extended_valid[:, None], extended_target, goal
                )
                terminal_ground = torch.where(
                    extended_valid[:, None], extended_direction, goal_direction
                )
            terminal_ratio = torch.full_like(
                desired_ground_speed, self.terminal_speed_ratio
            )
            if self.terminal_braking_enabled:
                distance_fraction = torch.clamp(
                    goal_distance / alignment_distance, min=0.0, max=1.0
                )
                terminal_ratio = self.terminal_speed_ratio + (
                    1.0 - self.terminal_speed_ratio
                ) * distance_fraction
            terminal_ground_speed = desired_ground_speed * terminal_ratio
            terminal_ground_speed = torch.maximum(
                terminal_ground_speed,
                self.minimum_speed_ratio * desired_ground_speed,
            )
        else:
            capture = (goal_distance <= self.terminal_capture_distance_m) & (
                goal_distance > 1e-9
            )
            terminal_ground = delta / torch.clamp(goal_distance[:, None], min=1e-9)
            terminal_ground_speed = desired_ground_speed
            if self.terminal_speed_mps > 0.0:
                fraction = torch.clamp(
                    goal_distance / self.terminal_capture_distance_m,
                    max=1.0,
                )
                terminal_ground_speed = self.terminal_speed_mps + (
                    desired_ground_speed - self.terminal_speed_mps
                ) * fraction
        terminal_speed, terminal_body = self._water_velocity(
            terminal_ground,
            current,
            desired_ground_speed=terminal_ground_speed,
        )
        terminal_crab = torch.zeros_like(desired_ground_speed)
        if self.current_crab_enabled:
            terminal_speed, terminal_body, terminal_crab = self._crab_water_velocity(
                terminal_ground,
                terminal_speed,
                estimated_current_ned_mps,
                estimated_current_valid,
                fallback_angle=terminal_crab,
            )
        terminal_heading = torch.atan2(terminal_body[:, 1], terminal_body[:, 0])
        terminal_pitch = torch.atan2(
            -terminal_body[:, 2],
            torch.linalg.norm(terminal_body[:, :2], dim=1),
        )
        desired_heading = torch.where(capture, terminal_heading, desired_heading)
        desired_pitch = torch.where(capture, terminal_pitch, desired_pitch)
        body_direction = torch.where(capture[:, None], terminal_body, body_direction)
        command_speed = torch.where(capture, terminal_speed, command_speed)
        crab_angle = torch.where(capture, terminal_crab, crab_angle)
        reference_position = torch.where(
            capture[:, None],
            terminal_target,
            reference_position,
        )

        return BatchedAlosSolution(
            segment_index=projection.segment_index,
            projection_s_m=projection.progress_m,
            lookahead_s_m=lookahead_s,
            horizontal_error_m=horizontal_error,
            vertical_error_m=vertical_error,
            horizontal_lookahead_m=horizontal_lookahead,
            vertical_lookahead_m=vertical_lookahead,
            desired_heading_rad=desired_heading,
            desired_pitch_rad=desired_pitch,
            desired_body_direction_ned=body_direction,
            command_speed_mps=command_speed,
            reference_speed_mps=command_speed,
            reference_position_ned_m=reference_position,
            reference_quat_xyzw=quaternion_from_direction(body_direction),
            terminal_phase=capture,
            crab_angle_rad=crab_angle,
        )

    def _water_velocity(
        self,
        ground_direction: Tensor,
        current: Tensor,
        *,
        desired_ground_speed: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        desired_speed = torch.as_tensor(
            desired_ground_speed,
            dtype=ground_direction.dtype,
            device=ground_direction.device,
        ).expand(ground_direction.shape[0])
        if not self.compensate_current:
            return desired_speed, ground_direction
        relative = desired_speed[:, None] * ground_direction - current
        speed = torch.linalg.norm(relative, dim=1)
        if self.validate_finite and bool((speed <= 1e-9).any()):
            raise ValueError("current exactly cancels the requested ground velocity")
        safe_speed = torch.clamp(speed, min=1e-9)
        return speed, relative / safe_speed[:, None]

    def _crab_water_velocity(
        self,
        ground_direction: Tensor,
        desired_speed: Tensor,
        estimated_current: Tensor | None,
        estimated_current_valid: Tensor | None,
        *,
        fallback_angle: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            not self.current_crab_enabled
            or estimated_current is None
            or estimated_current_valid is None
        ):
            return desired_speed, ground_direction, fallback_angle
        current = torch.as_tensor(
            estimated_current, device=ground_direction.device, dtype=ground_direction.dtype
        )
        valid = torch.as_tensor(
            estimated_current_valid, device=ground_direction.device, dtype=torch.bool
        )
        if current.shape != ground_direction.shape:
            raise ValueError("estimated_current_ned_mps must have shape (N, 3)")
        if valid.shape != (ground_direction.shape[0],):
            raise ValueError("estimated_current_valid must have shape (N,)")
        valid = valid & torch.isfinite(current).all(dim=1)
        safe_current = torch.where(valid[:, None], current, torch.zeros_like(current))

        relative = desired_speed[:, None] * ground_direction - safe_current
        relative_speed = torch.linalg.norm(relative, dim=1)
        full_compensation = (
            valid
            & (relative_speed > 1e-9)
            & (relative_speed <= self.current_crab_relative_speed_limit * desired_speed)
        )
        safe_relative_speed = torch.clamp(relative_speed, min=1e-9)
        full_direction = relative / safe_relative_speed[:, None]

        horizontal_norm = torch.linalg.norm(ground_direction[:, :2], dim=1)
        safe_horizontal_norm = torch.clamp(horizontal_norm, min=1e-9)
        ground_horizontal = ground_direction[:, :2] / safe_horizontal_norm[:, None]
        right = torch.stack((-ground_horizontal[:, 1], ground_horizontal[:, 0]), dim=1)
        current_horizontal = safe_current[:, :2]
        current_along = (current_horizontal * ground_horizontal).sum(dim=1)
        current_right = (current_horizontal * right).sum(dim=1)
        beta = torch.atan2(
            -current_right,
            torch.clamp(
                desired_speed - current_along,
                min=0.5 * desired_speed,
            ),
        )
        beta = torch.clamp(
            beta,
            -self.current_crab_max_angle_rad,
            self.current_crab_max_angle_rad,
        )
        ground_heading = torch.atan2(
            ground_horizontal[:, 1], ground_horizontal[:, 0]
        )
        crab_heading = ground_heading + beta
        ground_pitch = torch.atan2(
            -ground_direction[:, 2], horizontal_norm
        )
        crab_direction = torch.stack(
            (
                torch.cos(ground_pitch) * torch.cos(crab_heading),
                torch.cos(ground_pitch) * torch.sin(crab_heading),
                -torch.sin(ground_pitch),
            ),
            dim=1,
        )
        fallback = valid & ~full_compensation
        direction = torch.where(full_compensation[:, None], full_direction, ground_direction)
        direction = torch.where(fallback[:, None], crab_direction, direction)
        speed = torch.where(full_compensation, relative_speed, desired_speed)

        full_heading = torch.atan2(full_direction[:, 1], full_direction[:, 0])
        full_crab = wrap_angle(full_heading - ground_heading)
        crab_angle = torch.where(full_compensation, full_crab, torch.zeros_like(beta))
        crab_angle = torch.where(fallback, beta, crab_angle)
        return speed, direction, crab_angle


__all__ = ["BatchedAlosSolution", "BatchedThreeDAlosGuidance"]
