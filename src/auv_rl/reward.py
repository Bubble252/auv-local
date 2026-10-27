from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tanh


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


@dataclass(frozen=True)
class PaperRewardConfig:
    """Configuration for the four reward terms in Yuan et al. (2025).

    The paper does not publish the numeric distance coefficient. The default
    value is therefore a declared design parameter, not a paper parameter.
    """

    distance_penalty_factor: float = -1.0
    near_obstacle_penalty: float = -2.0
    far_obstacle_penalty: float = -1.0
    collision_penalty: float = -80.0
    goal_bonus: float = 50.0
    obstacle_influence_fraction: float = 0.7

    def __post_init__(self) -> None:
        distance_factor = _finite(self.distance_penalty_factor, "distance_penalty_factor")
        if distance_factor > 0.0:
            raise ValueError("distance_penalty_factor must be non-positive")
        near = _finite(self.near_obstacle_penalty, "near_obstacle_penalty")
        far = _finite(self.far_obstacle_penalty, "far_obstacle_penalty")
        collision = _finite(self.collision_penalty, "collision_penalty")
        goal = _finite(self.goal_bonus, "goal_bonus")
        influence = _finite(self.obstacle_influence_fraction, "obstacle_influence_fraction")
        if near > 0.0 or far > 0.0:
            raise ValueError("obstacle penalties must be non-positive")
        if collision >= 0.0:
            raise ValueError("collision_penalty must be negative")
        if goal <= 0.0:
            raise ValueError("goal_bonus must be positive")
        if not 0.0 < influence < 1.0:
            raise ValueError("obstacle_influence_fraction must lie in (0, 1)")
        object.__setattr__(self, "distance_penalty_factor", distance_factor)
        object.__setattr__(self, "near_obstacle_penalty", near)
        object.__setattr__(self, "far_obstacle_penalty", far)
        object.__setattr__(self, "collision_penalty", collision)
        object.__setattr__(self, "goal_bonus", goal)
        object.__setattr__(self, "obstacle_influence_fraction", influence)


@dataclass(frozen=True)
class RewardTerms:
    distance_reward: float
    obstacle_reward: float
    terminal_reward: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "distance_reward", _finite(self.distance_reward, "R1"))
        object.__setattr__(self, "obstacle_reward", _finite(self.obstacle_reward, "R2"))
        object.__setattr__(self, "terminal_reward", _finite(self.terminal_reward, "R3+R4"))

    @property
    def total(self) -> float:
        return self.distance_reward + self.obstacle_reward + self.terminal_reward

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "distance": self.distance_reward,
            "obstacle": self.obstacle_reward,
            "terminal": self.terminal_reward,
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
        collision: bool,
        out_of_bounds: bool,
        goal_reached: bool,
    ) -> RewardTerms:
        goal_distance = _non_negative(goal_distance_m, "goal_distance_m")
        initial_distance = _positive(initial_goal_distance_m, "initial_goal_distance_m")
        obstacle_range = _non_negative(front_obstacle_range_m, "front_obstacle_range_m")
        sensor_range = _positive(sensor_max_range_m, "sensor_max_range_m")
        if obstacle_range > sensor_range + 1e-9:
            raise ValueError("front_obstacle_range_m must not exceed sensor_max_range_m")
        if collision and goal_reached:
            raise ValueError("collision and goal_reached cannot both be true")

        distance_reward = self.config.distance_penalty_factor * tanh(
            goal_distance / initial_distance
        )
        threshold = self.config.obstacle_influence_fraction * sensor_range
        if obstacle_range < threshold:
            obstacle_reward = self.config.near_obstacle_penalty
        elif obstacle_range < sensor_range:
            obstacle_reward = self.config.far_obstacle_penalty
        else:
            obstacle_reward = 0.0

        if collision or out_of_bounds:
            terminal_reward = self.config.collision_penalty
        elif goal_reached:
            terminal_reward = self.config.goal_bonus
        else:
            terminal_reward = 0.0

        return RewardTerms(
            distance_reward=distance_reward,
            obstacle_reward=obstacle_reward,
            terminal_reward=terminal_reward,
        )


__all__ = ["PaperReward", "PaperRewardConfig", "RewardTerms"]
