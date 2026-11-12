from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LowLevelConfig, RewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    progress: float
    terminal_progress: float
    track_3d: float
    track_yaw: float
    track_u: float
    action: float
    saturation: float
    terminal: float
    total: float
    ppo_scaled: float


class RewardComputer:
    def __init__(self, config: LowLevelConfig) -> None:
        self.config = config

    def compute(
        self,
        *,
        previous_progress_m: float,
        progress_m: float,
        previous_goal_distance_m: float,
        goal_distance_m: float,
        error_3d_m: float,
        yaw_error_rad: float,
        water_surge_mps: float,
        reference_speed_mps: float,
        residual_normalized: np.ndarray,
        previous_residual_normalized: np.ndarray,
        saturated: bool,
        terminal_kind: str | None,
    ) -> RewardBreakdown:
        reward: RewardConfig = self.config.reward
        residual = np.asarray(residual_normalized, dtype=np.float64)
        previous_residual = np.asarray(previous_residual_normalized, dtype=np.float64)
        if residual.shape != (3,) or previous_residual.shape != (3,):
            raise ValueError("residual vectors must have shape (3,)")
        previous_goal_distance = float(previous_goal_distance_m)
        goal_distance = float(goal_distance_m)
        if not np.isfinite(previous_goal_distance) or previous_goal_distance < 0.0:
            raise ValueError("previous_goal_distance_m must be finite and non-negative")
        if not np.isfinite(goal_distance) or goal_distance < 0.0:
            raise ValueError("goal_distance_m must be finite and non-negative")
        progress = reward.w_prog * (float(progress_m) - float(previous_progress_m))
        capture_radius = float(self.config.guidance.terminal_capture_distance_m)
        terminal_progress = reward.terminal_progress_weight * (
            min(previous_goal_distance, capture_radius)
            - min(goal_distance, capture_radius)
        )
        track_3d = -reward.w_3d * float(error_3d_m)
        track_yaw = -reward.w_yaw * abs(float(yaw_error_rad))
        track_u = -reward.w_u * abs(float(water_surge_mps) - float(reference_speed_mps))
        action = -reward.w_action * float(np.sum(np.square(residual - previous_residual)))
        saturation = -reward.w_sat * float(saturated)
        terminal = 0.0
        if terminal_kind == "goal":
            terminal = reward.goal_bonus
        elif terminal_kind == "timeout":
            terminal = reward.timeout_penalty
        elif terminal_kind in {"out_of_bounds", "invalid"}:
            terminal = reward.out_of_bounds_penalty
        elif terminal_kind is not None:
            raise ValueError(f"unsupported terminal kind: {terminal_kind}")
        total = (
            progress
            + terminal_progress
            + track_3d
            + track_yaw
            + track_u
            + action
            + saturation
            + terminal
        )
        return RewardBreakdown(
            progress=progress,
            terminal_progress=terminal_progress,
            track_3d=track_3d,
            track_yaw=track_yaw,
            track_u=track_u,
            action=action,
            saturation=saturation,
            terminal=terminal,
            total=float(total),
            ppo_scaled=float(total * 0.01),
        )


__all__ = ["RewardBreakdown", "RewardComputer"]
