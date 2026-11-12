"""Batched reward and terminal-condition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from auv_lowlevel.config import LowLevelConfig


class TerminalKind(IntEnum):
    NONE = 0
    GOAL = 1
    TIMEOUT = 2
    OUT_OF_BOUNDS = 3
    INVALID = 4


def terminal_kind_name(value: int) -> str | None:
    kind = TerminalKind(value)
    return None if kind is TerminalKind.NONE else kind.name.lower()


@dataclass(frozen=True)
class BatchedRewardBreakdown:
    progress: Tensor
    terminal_progress: Tensor
    track_3d: Tensor
    track_yaw: Tensor
    track_u: Tensor
    action: Tensor
    saturation: Tensor
    terminal: Tensor
    total: Tensor
    ppo_scaled: Tensor


class BatchedRewardComputer:
    def __init__(
        self,
        config: LowLevelConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype

    def compute(
        self,
        *,
        previous_progress_m: Tensor,
        progress_m: Tensor,
        previous_goal_distance_m: Tensor,
        goal_distance_m: Tensor,
        error_3d_m: Tensor,
        yaw_error_rad: Tensor,
        water_surge_mps: Tensor,
        reference_speed_mps: Tensor,
        residual_normalized: Tensor,
        previous_residual_normalized: Tensor,
        saturated: Tensor,
        terminal_kind: Tensor,
    ) -> BatchedRewardBreakdown:
        previous_progress = torch.as_tensor(
            previous_progress_m, device=self.device, dtype=self.dtype
        )
        progress = torch.as_tensor(progress_m, device=self.device, dtype=self.dtype)
        previous_goal_distance = torch.as_tensor(
            previous_goal_distance_m, device=self.device, dtype=self.dtype
        )
        goal_distance = torch.as_tensor(
            goal_distance_m, device=self.device, dtype=self.dtype
        )
        error = torch.as_tensor(error_3d_m, device=self.device, dtype=self.dtype)
        yaw_error = torch.as_tensor(yaw_error_rad, device=self.device, dtype=self.dtype)
        water_surge = torch.as_tensor(water_surge_mps, device=self.device, dtype=self.dtype)
        reference_speed = torch.as_tensor(
            reference_speed_mps, device=self.device, dtype=self.dtype
        )
        residual = torch.as_tensor(residual_normalized, device=self.device, dtype=self.dtype)
        previous_residual = torch.as_tensor(
            previous_residual_normalized, device=self.device, dtype=self.dtype
        )
        saturated_t = torch.as_tensor(saturated, device=self.device, dtype=self.dtype)
        kind = torch.as_tensor(terminal_kind, device=self.device, dtype=torch.long)
        count = progress.shape[0]
        expected = (count,)
        for name, value in (
            ("previous_progress_m", previous_progress),
            ("progress_m", progress),
            ("previous_goal_distance_m", previous_goal_distance),
            ("goal_distance_m", goal_distance),
            ("error_3d_m", error),
            ("yaw_error_rad", yaw_error),
            ("water_surge_mps", water_surge),
            ("reference_speed_mps", reference_speed),
            ("saturated", saturated_t),
            ("terminal_kind", kind),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape ({count},)")
        if residual.shape != (count, 3) or previous_residual.shape != (count, 3):
            raise ValueError("residual vectors must have shape (N, 3)")

        reward = self.config.reward
        progress_reward = reward.w_prog * (progress - previous_progress)
        capture_radius = float(self.config.guidance.terminal_capture_distance_m)
        terminal_progress = reward.terminal_progress_weight * (
            torch.clamp(previous_goal_distance, max=capture_radius)
            - torch.clamp(goal_distance, max=capture_radius)
        )
        track_3d = -reward.w_3d * error
        track_yaw = -reward.w_yaw * torch.abs(yaw_error)
        track_u = -reward.w_u * torch.abs(water_surge - reference_speed)
        action_reward = -reward.w_action * torch.sum(
            torch.square(residual - previous_residual), dim=1
        )
        saturation_reward = -reward.w_sat * saturated_t
        terminal = torch.zeros_like(progress)
        terminal = torch.where(kind == TerminalKind.GOAL, reward.goal_bonus, terminal)
        terminal = torch.where(kind == TerminalKind.TIMEOUT, reward.timeout_penalty, terminal)
        terminal = torch.where(
            (kind == TerminalKind.OUT_OF_BOUNDS) | (kind == TerminalKind.INVALID),
            reward.out_of_bounds_penalty,
            terminal,
        )
        total = (
            progress_reward
            + terminal_progress
            + track_3d
            + track_yaw
            + track_u
            + action_reward
            + saturation_reward
            + terminal
        )
        return BatchedRewardBreakdown(
            progress=progress_reward,
            terminal_progress=terminal_progress,
            track_3d=track_3d,
            track_yaw=track_yaw,
            track_u=track_u,
            action=action_reward,
            saturation=saturation_reward,
            terminal=terminal,
            total=total,
            ppo_scaled=total * 0.01,
        )


def batched_terminal_kind(
    *,
    position_ned_m: Tensor,
    goal_ned_m: Tensor,
    bounds_ned_m: Tensor,
    goal_radius_m: float,
    step_count: Tensor,
    max_episode_steps: int,
) -> Tensor:
    """Resolve terminal conditions with the scalar environment's priority."""

    position = torch.as_tensor(position_ned_m)
    goal = torch.as_tensor(goal_ned_m, device=position.device, dtype=position.dtype)
    bounds = torch.as_tensor(bounds_ned_m, device=position.device, dtype=position.dtype)
    steps = torch.as_tensor(step_count, device=position.device)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position_ned_m must have shape (N, 3)")
    if goal.shape != (3,) or bounds.shape != (3, 2):
        raise ValueError("goal or bounds have the wrong shape")
    if steps.shape != (position.shape[0],):
        raise ValueError("step_count must have shape (N,)")
    kind = torch.full(
        (position.shape[0],),
        TerminalKind.NONE,
        dtype=torch.long,
        device=position.device,
    )
    timeout = steps >= max_episode_steps
    kind = torch.where(timeout, torch.full_like(kind, TerminalKind.TIMEOUT), kind)
    distance = torch.linalg.norm(position - goal[None, :], dim=1)
    goal_reached = distance <= goal_radius_m
    kind = torch.where(goal_reached, torch.full_like(kind, TerminalKind.GOAL), kind)
    outside = torch.any(position < bounds[:, 0] - 1e-9, dim=1) | torch.any(
        position > bounds[:, 1] + 1e-9, dim=1
    )
    kind = torch.where(outside, torch.full_like(kind, TerminalKind.OUT_OF_BOUNDS), kind)
    invalid = ~torch.isfinite(position).all(dim=1)
    return torch.where(invalid, torch.full_like(kind, TerminalKind.INVALID), kind)


__all__ = [
    "BatchedRewardBreakdown",
    "BatchedRewardComputer",
    "TerminalKind",
    "batched_terminal_kind",
    "terminal_kind_name",
]
