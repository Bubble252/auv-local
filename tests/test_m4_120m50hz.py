from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from auv_rl.bc.teacher import PotentialFieldTeacher
from auv_rl.envs.parallel_planar import P1TensorVectorEnv
from auv_rl.envs.planar_avoidance import PlanarEnvConfig
from auv_rl.observation import ForwardSectorObservationBuilder
from auv_rl.reward import PaperReward, PaperRewardConfig
from auv_rl.reward_audit import run_reward_audit


def _config(**overrides: Any) -> PlanarEnvConfig:
    values: dict[str, Any] = {
        "dt_s": 0.02,
        "integration_step_s": 0.02,
        "max_episode_steps": 3000,
        "sensor_max_range_m": 120.0,
        "local_effective_range_m": 120.0,
        "reward_obstacle_range_m": 20.0,
        "reward_step_scale": 0.1,
        "observation_version": "2.0-120m",
        "analytic_observation_source": "analytic_13sector_120m",
        "analytic_perception_mode": "analytic_13sector_geometry_120m",
        "active_scenario_keys": ("s1", "s2"),
    }
    values.update(overrides)
    return PlanarEnvConfig(**values)


def test_120m_observation_spec_and_s2_center_sector() -> None:
    env = P1TensorVectorEnv(1, _config(), seed=0)
    observation, _ = env.reset(seed=0, options={"scenario_key": "s2"})

    assert env.builder.spec.version == "2.0-120m"
    assert env.observation_source == "analytic_13sector_120m"
    assert env.perception.perception_mode == "analytic_13sector_geometry_120m"
    assert observation.shape == (1, 17)
    assert observation[0, 6] * 120.0 == pytest.approx(10.2, abs=0.1)
    assert np.all(observation[0, :5] > 0.99)
    assert np.all(observation[0, 9:13] > 0.99)


def test_50hz_env_uses_one_integration_substep() -> None:
    env = P1TensorVectorEnv(1, _config(), seed=0)
    env.reset(seed=0, options={"scenario_key": "s1"})

    env.step(np.zeros((1, 2), dtype=np.float64))

    assert env.config.dt_s / env.config.integration_step_s == pytest.approx(1.0)
    assert env.state(0).t_s == pytest.approx(0.02)
    assert env.episode_steps.tolist() == [1]


def test_reward_obstacle_range_is_decoupled_from_observation_range() -> None:
    near = P1TensorVectorEnv(1, _config(local_effective_range_m=20.0), seed=0)
    far = P1TensorVectorEnv(1, _config(local_effective_range_m=120.0), seed=0)
    near_observation, _ = near.reset(seed=0, options={"scenario_key": "s2"})
    far_observation, _ = far.reset(seed=0, options={"scenario_key": "s2"})
    assert not np.allclose(near_observation, far_observation)

    _, _, _, _, near_info = near.step(np.zeros((1, 2), dtype=np.float64))
    _, _, _, _, far_info = far.step(np.zeros((1, 2), dtype=np.float64))

    assert near_info["reward_obstacle"][0] == pytest.approx(
        far_info["reward_obstacle"][0],
        abs=1e-12,
    )


def test_paper_reward_uses_explicit_obstacle_range() -> None:
    reward = PaperReward(PaperRewardConfig(obstacle_influence_fraction=0.7))
    common: dict[str, Any] = {
        "goal_distance_m": 20.0,
        "initial_goal_distance_m": 20.0,
        "sensor_max_range_m": 120.0,
        "reward_obstacle_range_m": 20.0,
        "collision": False,
        "out_of_bounds": False,
        "goal_reached": False,
    }

    far = reward.compute(front_obstacle_range_m=30.0, **common)
    mid = reward.compute(front_obstacle_range_m=15.0, **common)
    near = reward.compute(front_obstacle_range_m=10.0, **common)

    assert far.obstacle_reward == 0.0
    assert mid.obstacle_reward == pytest.approx(-1.0)
    assert near.obstacle_reward == pytest.approx(-2.0)


def test_paper_reward_step_scale_scales_continuous_terms_only() -> None:
    reward = PaperReward(PaperRewardConfig())
    inputs: dict[str, Any] = {
        "goal_distance_m": 20.0,
        "initial_goal_distance_m": 20.0,
        "front_obstacle_range_m": 10.0,
        "sensor_max_range_m": 120.0,
        "reward_obstacle_range_m": 20.0,
        "collision": False,
        "out_of_bounds": False,
        "goal_reached": False,
    }
    unscaled = reward.compute(reward_step_scale=1.0, **inputs)
    scaled = reward.compute(reward_step_scale=0.1, **inputs)

    assert scaled.distance_reward == pytest.approx(unscaled.distance_reward * 0.1)
    assert scaled.obstacle_reward == pytest.approx(unscaled.obstacle_reward * 0.1)
    assert scaled.terminal_reward == unscaled.terminal_reward == 0.0


def test_5hz_and_50hz_continuous_reward_sums_match() -> None:
    reward = PaperReward(PaperRewardConfig())
    inputs: dict[str, Any] = {
        "goal_distance_m": 20.0,
        "initial_goal_distance_m": 20.0,
        "front_obstacle_range_m": 10.0,
        "sensor_max_range_m": 120.0,
        "reward_obstacle_range_m": 20.0,
        "collision": False,
        "out_of_bounds": False,
        "goal_reached": False,
    }
    five_hz = sum(reward.compute(reward_step_scale=1.0, **inputs).total for _ in range(10))
    fifty_hz = sum(reward.compute(reward_step_scale=0.1, **inputs).total for _ in range(100))

    assert fifty_hz == pytest.approx(five_hz, rel=1e-12, abs=1e-12)


def test_observation_version_is_forwarded_to_builder() -> None:
    builder = ForwardSectorObservationBuilder(max_range_m=120.0, version="2.0-120m")
    assert builder.spec.version == "2.0-120m"
    assert builder.critic_spec.version == "2.0-120m"
    assert builder.spec.normalization["sector_range_scale_m"] == 120.0


def test_reward_audit_passes_p0_gates_with_selected_goal_bonus() -> None:
    config = _config(
        reward=PaperRewardConfig(goal_bonus=700.0),
    )

    result = run_reward_audit(
        config,
        PotentialFieldTeacher(),
        gamma=0.99798,
        seeds=(0,),
    )

    assert result["passed"], result
    assert result["summary"]["success_return"] > 0.0
    assert result["summary"]["timeout_return"] < 0.0
    assert result["summary"]["collision_return"] < result["summary"]["timeout_return"]
    assert result["summary"]["success_minus_timeout"] >= 10.0
