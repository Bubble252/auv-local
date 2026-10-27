from __future__ import annotations

import numpy as np
import pytest

from auv_rl.reward import PaperReward, PaperRewardConfig


def test_paper_reward_uses_distance_and_obstacle_thresholds() -> None:
    reward = PaperReward(PaperRewardConfig(distance_penalty_factor=-2.0))

    far = reward.compute(
        goal_distance_m=20.0,
        initial_goal_distance_m=20.0,
        front_obstacle_range_m=8.0,
        sensor_max_range_m=8.0,
        collision=False,
        out_of_bounds=False,
        goal_reached=False,
    )
    near = reward.compute(
        goal_distance_m=0.0,
        initial_goal_distance_m=20.0,
        front_obstacle_range_m=5.59,
        sensor_max_range_m=8.0,
        collision=False,
        out_of_bounds=False,
        goal_reached=False,
    )
    danger = reward.compute(
        goal_distance_m=10.0,
        initial_goal_distance_m=20.0,
        front_obstacle_range_m=5.6,
        sensor_max_range_m=8.0,
        collision=False,
        out_of_bounds=False,
        goal_reached=False,
    )

    assert far.distance_reward == pytest.approx(-2.0 * np.tanh(1.0))
    assert far.obstacle_reward == 0.0
    assert near.distance_reward == 0.0
    assert near.obstacle_reward == -2.0
    assert danger.obstacle_reward == -1.0


def test_paper_reward_terminal_terms_and_sum() -> None:
    reward = PaperReward()
    collision = reward.compute(
        goal_distance_m=5.0,
        initial_goal_distance_m=20.0,
        front_obstacle_range_m=1.0,
        sensor_max_range_m=8.0,
        collision=True,
        out_of_bounds=False,
        goal_reached=False,
    )
    goal = reward.compute(
        goal_distance_m=0.2,
        initial_goal_distance_m=20.0,
        front_obstacle_range_m=8.0,
        sensor_max_range_m=8.0,
        collision=False,
        out_of_bounds=False,
        goal_reached=True,
    )

    assert collision.terminal_reward == -80.0
    assert collision.total == pytest.approx(
        collision.distance_reward + collision.obstacle_reward - 80.0
    )
    assert goal.terminal_reward == 50.0
    assert goal.total == pytest.approx(goal.distance_reward + goal.obstacle_reward + 50.0)


def test_paper_reward_rejects_inconsistent_terminal_state() -> None:
    reward = PaperReward()
    with pytest.raises(ValueError, match="cannot both"):
        reward.compute(
            goal_distance_m=0.0,
            initial_goal_distance_m=20.0,
            front_obstacle_range_m=8.0,
            sensor_max_range_m=8.0,
            collision=True,
            out_of_bounds=False,
            goal_reached=True,
        )
