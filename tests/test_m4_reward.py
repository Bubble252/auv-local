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


def test_continuous_obstacle_penalty_is_linear_and_bounded() -> None:
    reward = PaperReward(
        PaperRewardConfig(
            continuous_obstacle_penalty=True,
            continuous_penalty_range_m=14.0,
        )
    )
    values = []
    for distance in (0.0, 7.0, 14.0, 20.0):
        terms = reward.compute(
            goal_distance_m=5.0,
            initial_goal_distance_m=20.0,
            front_obstacle_range_m=distance,
            sensor_max_range_m=20.0,
            collision=False,
            out_of_bounds=False,
            goal_reached=False,
        )
        values.append(terms.obstacle_reward)
    assert values == pytest.approx([-2.0, -1.0, 0.0, 0.0])


def test_reward_v2_combines_progress_potential_speed_and_smoothness() -> None:
    reward = PaperReward(
        PaperRewardConfig(
            reward_version="v2",
            goal_progress_weight=1.0,
            obstacle_potential_weight=2.0,
            reference_speed_weight=0.1,
            action_smoothness_weight=0.3,
            safety_clearance_m=2.0,
            reference_speed_mps=1.0,
        )
    )
    terms = reward.compute(
        goal_distance_m=18.0,
        initial_goal_distance_m=20.0,
        front_obstacle_range_m=20.0,
        sensor_max_range_m=20.0,
        collision=False,
        out_of_bounds=False,
        goal_reached=False,
        previous_goal_distance_m=20.0,
        previous_clearance_m=1.5,
        current_clearance_m=2.5,
        surge_speed_mps=1.5,
        normalized_action=np.array([0.5, 0.5]),
        previous_normalized_action=np.array([0.0, 0.0]),
        reference_speed_mps=1.0,
    )
    assert terms.distance_reward == 0.0
    assert terms.goal_progress_reward == pytest.approx(2.0)
    assert terms.obstacle_reward == pytest.approx(1.0)
    assert terms.reference_speed_reward == pytest.approx(-0.025)
    assert terms.action_smoothness_reward == pytest.approx(-0.15)
    assert terms.total == pytest.approx(2.825)


def test_reward_v2_requires_previous_state_inputs() -> None:
    reward = PaperReward(PaperRewardConfig(reward_version="v2"))
    with pytest.raises(ValueError, match="reward v2 requires"):
        reward.compute(
            goal_distance_m=18.0,
            initial_goal_distance_m=20.0,
            front_obstacle_range_m=20.0,
            sensor_max_range_m=20.0,
            collision=False,
            out_of_bounds=False,
            goal_reached=False,
        )
