from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, tanh

import numpy as np


def _finite(value: float, name: str) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _non_negative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_vector(value: Sequence[float] | np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} must not be None")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


@dataclass(frozen=True)
class PaperRewardConfig:
    """Configuration for the four reward terms in Yuan et al. (2025).

    The paper does not publish the numeric distance coefficient. The default
    value is therefore a declared design parameter, not a paper parameter.
    """

    distance_penalty_factor: float = -1.0
    action_l2_penalty_factor: float = 0.0
    near_obstacle_penalty: float = -2.0
    far_obstacle_penalty: float = -1.0
    collision_penalty: float = -80.0
    timeout_penalty: float = 0.0
    goal_bonus: float = 50.0
    obstacle_influence_fraction: float = 0.7
    continuous_obstacle_penalty: bool = False
    continuous_penalty_range_m: float = 14.0
    reward_version: str = "paper"
    goal_progress_weight: float = 1.0
    obstacle_potential_weight: float = 1.0
    reference_speed_weight: float = 0.1
    action_smoothness_weight: float = 0.1
    safety_clearance_m: float = 2.0
    reference_speed_mps: float | None = None

    def __post_init__(self) -> None:
        distance_factor = _finite(self.distance_penalty_factor, "distance_penalty_factor")
        if distance_factor > 0.0:
            raise ValueError("distance_penalty_factor must be non-positive")
        action_l2_factor = _finite(self.action_l2_penalty_factor, "action_l2_penalty_factor")
        if action_l2_factor > 0.0:
            raise ValueError("action_l2_penalty_factor must be non-positive")
        near = _finite(self.near_obstacle_penalty, "near_obstacle_penalty")
        far = _finite(self.far_obstacle_penalty, "far_obstacle_penalty")
        collision = _finite(self.collision_penalty, "collision_penalty")
        timeout = _finite(self.timeout_penalty, "timeout_penalty")
        goal = _finite(self.goal_bonus, "goal_bonus")
        influence = _finite(self.obstacle_influence_fraction, "obstacle_influence_fraction")
        continuous_range = _positive(
            self.continuous_penalty_range_m,
            "continuous_penalty_range_m",
        )
        version = self.reward_version.strip().lower()
        goal_progress = _non_negative(self.goal_progress_weight, "goal_progress_weight")
        obstacle_potential = _non_negative(
            self.obstacle_potential_weight,
            "obstacle_potential_weight",
        )
        reference_speed = _non_negative(
            self.reference_speed_weight,
            "reference_speed_weight",
        )
        action_smoothness = _non_negative(
            self.action_smoothness_weight,
            "action_smoothness_weight",
        )
        safety_clearance = _positive(self.safety_clearance_m, "safety_clearance_m")
        configured_reference = (
            None
            if self.reference_speed_mps is None
            else _positive(self.reference_speed_mps, "reference_speed_mps")
        )
        if near > 0.0 or far > 0.0:
            raise ValueError("obstacle penalties must be non-positive")
        if collision >= 0.0:
            raise ValueError("collision_penalty must be negative")
        if timeout > 0.0:
            raise ValueError("timeout_penalty must be non-positive")
        if goal <= 0.0:
            raise ValueError("goal_bonus must be positive")
        if not 0.0 < influence < 1.0:
            raise ValueError("obstacle_influence_fraction must lie in (0, 1)")
        if version not in {"paper", "v2"}:
            raise ValueError("reward_version must be 'paper' or 'v2'")
        object.__setattr__(self, "distance_penalty_factor", distance_factor)
        object.__setattr__(self, "action_l2_penalty_factor", action_l2_factor)
        object.__setattr__(self, "near_obstacle_penalty", near)
        object.__setattr__(self, "far_obstacle_penalty", far)
        object.__setattr__(self, "collision_penalty", collision)
        object.__setattr__(self, "timeout_penalty", timeout)
        object.__setattr__(self, "goal_bonus", goal)
        object.__setattr__(self, "obstacle_influence_fraction", influence)
        object.__setattr__(self, "continuous_penalty_range_m", continuous_range)
        object.__setattr__(self, "reward_version", version)
        object.__setattr__(self, "goal_progress_weight", goal_progress)
        object.__setattr__(self, "obstacle_potential_weight", obstacle_potential)
        object.__setattr__(self, "reference_speed_weight", reference_speed)
        object.__setattr__(self, "action_smoothness_weight", action_smoothness)
        object.__setattr__(self, "safety_clearance_m", safety_clearance)
        object.__setattr__(self, "reference_speed_mps", configured_reference)


@dataclass(frozen=True)
class RewardTerms:
    distance_reward: float
    obstacle_reward: float
    terminal_reward: float
    goal_progress_reward: float = 0.0
    reference_speed_reward: float = 0.0
    action_smoothness_reward: float = 0.0
    action_l2_penalty: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "distance_reward", _finite(self.distance_reward, "R1"))
        object.__setattr__(self, "obstacle_reward", _finite(self.obstacle_reward, "R2"))
        object.__setattr__(self, "terminal_reward", _finite(self.terminal_reward, "R3+R4"))
        object.__setattr__(
            self,
            "goal_progress_reward",
            _finite(self.goal_progress_reward, "goal_progress_reward"),
        )
        object.__setattr__(
            self,
            "reference_speed_reward",
            _finite(self.reference_speed_reward, "reference_speed_reward"),
        )
        object.__setattr__(
            self,
            "action_smoothness_reward",
            _finite(self.action_smoothness_reward, "action_smoothness_reward"),
        )
        object.__setattr__(
            self,
            "action_l2_penalty",
            _finite(self.action_l2_penalty, "action_l2_penalty"),
        )

    @property
    def total(self) -> float:
        return (
            self.distance_reward
            + self.obstacle_reward
            + self.terminal_reward
            + self.goal_progress_reward
            + self.reference_speed_reward
            + self.action_smoothness_reward
            + self.action_l2_penalty
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "distance": self.distance_reward,
            "obstacle": self.obstacle_reward,
            "terminal": self.terminal_reward,
            "goal_progress": self.goal_progress_reward,
            "reference_speed": self.reference_speed_reward,
            "action_smoothness": self.action_smoothness_reward,
            "action_l2_penalty": self.action_l2_penalty,
        }


class PaperReward:
    """Compute equations (20)-(22) from the selected paper."""

    def __init__(self, config: PaperRewardConfig | None = None) -> None:
        self.config = PaperRewardConfig() if config is None else config
        if not isinstance(self.config, PaperRewardConfig):
            raise TypeError("config must be a PaperRewardConfig")

    def compute(
        self,
        *,
        goal_distance_m: float,
        initial_goal_distance_m: float,
        front_obstacle_range_m: float,
        sensor_max_range_m: float,
        reward_obstacle_range_m: float | None = None,
        reward_step_scale: float = 1.0,
        collision: bool,
        out_of_bounds: bool,
        goal_reached: bool,
        truncated: bool = False,
        previous_goal_distance_m: float | None = None,
        previous_clearance_m: float | None = None,
        current_clearance_m: float | None = None,
        surge_speed_mps: float | None = None,
        normalized_action: Sequence[float] | np.ndarray | None = None,
        previous_normalized_action: Sequence[float] | np.ndarray | None = None,
        reference_speed_mps: float | None = None,
    ) -> RewardTerms:
        goal_distance = _non_negative(goal_distance_m, "goal_distance_m")
        initial_distance = _positive(initial_goal_distance_m, "initial_goal_distance_m")
        obstacle_range = _non_negative(front_obstacle_range_m, "front_obstacle_range_m")
        sensor_range = _positive(sensor_max_range_m, "sensor_max_range_m")
        if obstacle_range > sensor_range + 1e-9:
            raise ValueError("front_obstacle_range_m must not exceed sensor_max_range_m")
        reward_range = (
            sensor_range
            if reward_obstacle_range_m is None
            else _positive(reward_obstacle_range_m, "reward_obstacle_range_m")
        )
        if reward_range > sensor_range + 1e-9:
            raise ValueError("reward_obstacle_range_m must not exceed sensor_max_range_m")
        step_scale = _positive(reward_step_scale, "reward_step_scale")
        if collision and goal_reached:
            raise ValueError("collision and goal_reached cannot both be true")

        if self.config.reward_version == "paper":
            distance_reward = self.config.distance_penalty_factor * tanh(
                goal_distance / initial_distance
            )
            threshold = self.config.obstacle_influence_fraction * reward_range
            if self.config.continuous_obstacle_penalty:
                fraction = min(
                    obstacle_range / self.config.continuous_penalty_range_m,
                    1.0,
                )
                obstacle_reward = self.config.near_obstacle_penalty * (1.0 - fraction)
            elif obstacle_range < threshold:
                obstacle_reward = self.config.near_obstacle_penalty
            elif obstacle_range < reward_range:
                obstacle_reward = self.config.far_obstacle_penalty
            else:
                obstacle_reward = 0.0
            distance_reward *= step_scale
            obstacle_reward *= step_scale
            goal_progress_reward = 0.0
            reference_speed_reward = 0.0
            action_smoothness_reward = 0.0
            action_l2_penalty = 0.0
            if self.config.action_l2_penalty_factor != 0.0 and normalized_action is not None:
                action = _finite_vector(normalized_action, "normalized_action")
                action_l2_penalty = self.config.action_l2_penalty_factor * float(
                    np.square(action).sum()
                )
        else:
            distance_reward = 0.0
            required = {
                "previous_goal_distance_m": previous_goal_distance_m,
                "previous_clearance_m": previous_clearance_m,
                "current_clearance_m": current_clearance_m,
                "surge_speed_mps": surge_speed_mps,
                "normalized_action": normalized_action,
                "previous_normalized_action": previous_normalized_action,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"reward v2 requires: {', '.join(missing)}")
            assert previous_goal_distance_m is not None
            assert previous_clearance_m is not None
            assert current_clearance_m is not None
            assert surge_speed_mps is not None
            previous_distance = _non_negative(
                previous_goal_distance_m,
                "previous_goal_distance_m",
            )
            previous_clearance = _finite(previous_clearance_m, "previous_clearance_m")
            current_clearance = _finite(current_clearance_m, "current_clearance_m")
            surge_speed = _finite(surge_speed_mps, "surge_speed_mps")
            if reference_speed_mps is None:
                configured_reference = self.config.reference_speed_mps
                if configured_reference is None:
                    raise ValueError("reward v2 requires a reference speed")
                reference_speed = configured_reference
            else:
                reference_speed = reference_speed_mps
            reference_speed = _positive(reference_speed, "reference_speed_mps")
            action = _finite_vector(normalized_action, "normalized_action")
            previous_action = _finite_vector(
                previous_normalized_action,
                "previous_normalized_action",
            )
            if action.shape != previous_action.shape:
                raise ValueError("normalized action vectors must have the same shape")
            previous_potential = max(
                0.0,
                self.config.safety_clearance_m - previous_clearance,
            )
            current_potential = max(
                0.0,
                self.config.safety_clearance_m - current_clearance,
            )
            goal_progress_reward = self.config.goal_progress_weight * (
                previous_distance - goal_distance
            )
            obstacle_reward = self.config.obstacle_potential_weight * (
                previous_potential - current_potential
            )
            obstacle_reward *= step_scale
            reference_speed_reward = (
                -self.config.reference_speed_weight * (surge_speed - reference_speed) ** 2
            )
            reference_speed_reward *= step_scale
            action_smoothness_reward = -self.config.action_smoothness_weight * float(
                ((action - previous_action) ** 2).sum()
            )
            action_smoothness_reward *= step_scale
            action_l2_penalty = 0.0

        if collision or out_of_bounds:
            terminal_reward = self.config.collision_penalty
        elif goal_reached:
            terminal_reward = self.config.goal_bonus
        elif truncated:
            terminal_reward = self.config.timeout_penalty
        else:
            terminal_reward = 0.0

        return RewardTerms(
            distance_reward=distance_reward,
            obstacle_reward=obstacle_reward,
            terminal_reward=terminal_reward,
            goal_progress_reward=goal_progress_reward,
            reference_speed_reward=reference_speed_reward,
            action_smoothness_reward=action_smoothness_reward,
            action_l2_penalty=action_l2_penalty,
        )


__all__ = ["PaperReward", "PaperRewardConfig", "RewardTerms"]
