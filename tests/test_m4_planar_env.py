from __future__ import annotations

from math import pi
from typing import Any

import numpy as np
import pytest
from auv_core.contracts import State

from auv_local.contracts import SurgeYawAction
from auv_rl.envs.parallel_planar import P1TensorVectorEnv
from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig
from auv_rl.perception import AnalyticSectorPerceptionBackend
from auv_rl.planar import PlanarKinematics, PlanarMotionLimits
from auv_rl.reward import PaperRewardConfig


def make_state() -> State:
    return State(
        t_s=0.0,
        position_ned_m=np.zeros(3, dtype=np.float64),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )


def test_planar_kinematics_integrates_and_clips_action() -> None:
    model = PlanarKinematics(
        PlanarMotionLimits(
            min_speed_mps=1.0,
            max_speed_mps=2.5,
            max_accel_mps2=0.5,
            max_yaw_rate_rps=15.0 * pi / 180.0,
        )
    )
    transition = model.step(
        make_state(),
        SurgeYawAction(u_dot_mps2=10.0, yaw_rate_rps=1.0),
        1.0,
    )

    assert transition.action_clipped
    assert transition.applied_action.u_dot_mps2 == pytest.approx(0.5)
    assert transition.applied_action.yaw_rate_rps == pytest.approx(15.0 * pi / 180.0)
    assert transition.state.linear_velocity_body_mps[0] == pytest.approx(1.5)
    assert transition.state.position_ned_m[0] == pytest.approx(1.5)
    assert transition.state.angular_velocity_body_rps[2] == pytest.approx(15.0 * pi / 180.0)


def test_environment_reset_is_deterministic_and_step_reports_reward_terms() -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1",)))
    first, first_info = env.reset(seed=7, options={"scenario_key": "s1"})
    second, second_info = env.reset(seed=7, options={"scenario_key": "s1"})

    assert env.observation_space.contains(first)
    assert first == pytest.approx(second)
    assert first_info["scenario_key"] == second_info["scenario_key"] == "s1"

    observation, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["reward_total"] == pytest.approx(
        info["reward_distance"] + info["reward_obstacle"] + info["reward_terminal"]
    )
    assert info["failure_reason"] == "in_progress"


def test_planar_env_logs_reward_v2_components() -> None:
    env = PlanarAvoidanceEnv(
        PlanarEnvConfig(
            active_scenario_keys=("s1",),
            reward=PaperRewardConfig(reward_version="v2"),
        )
    )
    env.reset(seed=0, options={"scenario_key": "s1"})
    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float64))
    component_sum = sum(
        float(info[key])
        for key in (
            "reward_distance",
            "reward_obstacle",
            "reward_terminal",
            "reward_goal_progress",
            "reward_reference_speed",
            "reward_action_smoothness",
        )
    )
    assert reward == pytest.approx(component_sum)
    assert info["reward_distance"] == 0.0
    assert np.isfinite(info["reward_goal_progress"])
    assert np.isfinite(info["reward_reference_speed"])
    assert np.isfinite(info["reward_action_smoothness"])


def test_empty_scenario_can_reach_goal_with_straight_control() -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s1",)))
    env.reset(seed=0, options={"scenario_key": "s1"})
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    while not terminated and not truncated:
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float64))

    assert terminated
    assert not truncated
    assert info["goal_reached"]
    assert info["success"]
    assert info["reward_terminal"] == pytest.approx(50.0)


def test_single_cylinder_collision_is_reported_explicitly() -> None:
    env = PlanarAvoidanceEnv(PlanarEnvConfig(active_scenario_keys=("s2",)))
    env.reset(seed=0, options={"scenario_key": "s2"})
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    while not terminated and not truncated:
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float64))

    assert terminated
    assert not truncated
    assert info["collision"]
    assert info["failure_reason"] == "collision"
    assert info["reward_terminal"] == pytest.approx(-80.0)
    assert info["min_clearance_m"] < 0.0


def test_reset_jitter_override_is_recorded_and_eval_can_disable_it() -> None:
    env = PlanarAvoidanceEnv(
        PlanarEnvConfig(
            active_scenario_keys=("s1",),
            start_jitter_xy_m=0.5,
            start_jitter_yaw_rad=0.05,
        )
    )
    env.reset(seed=2, options={"scenario_key": "s1"})
    _, _, _, _, training_info = env.step(np.zeros(2, dtype=np.float64))
    assert training_info["start_jitter_xy_m"] == pytest.approx(0.5)
    assert training_info["start_jitter_yaw_rad"] == pytest.approx(0.05)

    env.reset(seed=3, options={"scenario_key": "s1", "jitter": False})
    _, _, _, _, eval_info = env.step(np.zeros(2, dtype=np.float64))
    assert eval_info["start_jitter_xy_m"] == 0.0
    assert eval_info["start_jitter_yaw_rad"] == 0.0


@pytest.mark.parametrize("scenario_key", ["s1", "s2"])
def test_tensor_vector_env_n1_matches_scalar_fixed_actions(scenario_key: str) -> None:
    config = PlanarEnvConfig(active_scenario_keys=(scenario_key,))
    scalar = PlanarAvoidanceEnv(config, perception=AnalyticSectorPerceptionBackend())
    vector = P1TensorVectorEnv(1, config, seed=0)
    scalar_observation, _ = scalar.reset(
        seed=0,
        options={"scenario_key": scenario_key, "jitter": False},
    )
    vector_observation, vector_info = vector.reset(
        seed=0,
        options={"scenario_key": scenario_key, "jitter": False},
    )
    assert np.allclose(scalar_observation, vector_observation[0])
    assert vector_info["observation_source"] == "analytic_13sector"

    for step_index in range(120):
        action = np.array(
            [0.15 * np.sin(0.2 * step_index), 0.04 * np.cos(0.13 * step_index)],
            dtype=np.float64,
        )
        (
            scalar_observation,
            scalar_reward,
            scalar_terminated,
            scalar_truncated,
            scalar_info,
        ) = scalar.step(action)
        (
            _,
            vector_reward,
            vector_terminated,
            vector_truncated,
            vector_info,
        ) = vector.step(action.reshape(1, 2))
        final_observation = vector_info["final_observations"][0]
        assert np.allclose(scalar_observation, final_observation, atol=1e-6)
        assert scalar_reward == pytest.approx(float(vector_reward[0]), abs=1e-9)
        assert scalar_terminated == bool(vector_terminated[0])
        assert scalar_truncated == bool(vector_truncated[0])
        for field in (
            "scenario_key",
            "failure_reason",
            "goal_reached",
            "collision",
            "out_of_bounds",
            "timeout",
        ):
            assert scalar_info[field] == vector_info[field][0]
        for field in (
            "goal_distance_m",
            "simulation_time_s",
            "min_clearance_m",
            "path_length_m",
            "reward_total",
            "reward_distance",
            "reward_obstacle",
            "reward_terminal",
        ):
            assert scalar_info[field] == pytest.approx(float(vector_info[field][0]), abs=1e-9)
        if scalar_terminated or scalar_truncated:
            break
    else:
        pytest.fail("fixed-action comparison did not reach a terminal state")


def test_tensor_vector_env_auto_reset_does_not_disturb_other_env() -> None:
    config = PlanarEnvConfig(active_scenario_keys=("s1", "s2"))
    env = P1TensorVectorEnv(2, config, seed=7)
    env.reset(seed=7, options={"scenario_key": ("s1", "s2"), "jitter": False})
    initial_ids = env.episode_ids.copy()
    assert len(set(initial_ids.tolist())) == 2

    for _ in range(60):
        undisturbed_state = env.state(0)
        observation, _, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float64))
        if np.any(info["reset_mask"]):
            break
    else:
        pytest.fail("S2 collision should auto-reset before S1 reaches its goal")

    assert info["reset_mask"][1]
    assert not info["reset_mask"][0]
    assert env.episode_ids[0] == initial_ids[0]
    assert env.episode_ids[1] > initial_ids[1]
    after_state = env.state(0)
    assert after_state.position_ned_m[0] == pytest.approx(undisturbed_state.position_ned_m[0] + 0.2)
    assert after_state.position_ned_m[1] == pytest.approx(undisturbed_state.position_ned_m[1])
    assert after_state.t_s == pytest.approx(undisturbed_state.t_s + 0.2)
    assert bool(terminated[1])
    assert not bool(truncated[1])
    assert not np.allclose(info["final_observations"][1], observation[1])


def test_tensor_vector_env_separates_truncation_and_new_episode_observation() -> None:
    config = PlanarEnvConfig(active_scenario_keys=("s1",), max_episode_steps=3)
    env = P1TensorVectorEnv(1, config, seed=2)
    env.reset(seed=2, options={"scenario_key": "s1", "jitter": False})
    final_observation = None
    for _ in range(3):
        observation, _, terminated, truncated, info = env.step(np.zeros((1, 2), dtype=np.float64))
        final_observation = info["final_observations"][0].copy()
    assert not bool(terminated[0])
    assert bool(truncated[0])
    assert bool(info["timeout"][0])
    assert info["failure_reason"][0] == "timeout"
    assert info["reset_mask"][0]
    assert final_observation is not None
    assert not np.allclose(observation[0], final_observation)
    assert int(info["rl_step"][0]) == 3
    assert int(env.episode_steps[0]) == 0


def test_tensor_vector_env_reward_v2_reset_uses_new_episode_bootstrap() -> None:
    config = PlanarEnvConfig(
        active_scenario_keys=("s2",),
        reward=PaperRewardConfig(reward_version="v2"),
    )
    scalar = PlanarAvoidanceEnv(config, perception=AnalyticSectorPerceptionBackend())
    vector = P1TensorVectorEnv(1, config, seed=13)
    scalar.reset(seed=13, options={"scenario_key": "s2", "jitter": False})
    vector.reset(seed=13, options={"scenario_key": "s2", "jitter": False})

    for _ in range(200):
        _, _, scalar_terminated, scalar_truncated, _ = scalar.step(np.zeros(2, dtype=np.float64))
        _, _, vector_terminated, vector_truncated, vector_info = vector.step(
            np.zeros((1, 2), dtype=np.float64)
        )
        if scalar_terminated or scalar_truncated:
            assert bool(vector_terminated[0]) == scalar_terminated
            assert bool(vector_truncated[0]) == scalar_truncated
            break
    else:
        pytest.fail("S2 straight-line collision did not terminate")

    assert bool(vector_info["reset_mask"][0])
    scalar_observation, _ = scalar.reset(
        seed=13,
        options={"scenario_key": "s2", "jitter": False},
    )
    vector_observation = vector._batch_observation()[0]  # noqa: SLF001 - reset invariant probe
    assert np.allclose(vector_observation[0], scalar_observation)

    _, scalar_reward, _, _, scalar_info = scalar.step(np.zeros(2, dtype=np.float64))
    _, vector_reward, _, _, vector_info = vector.step(np.zeros((1, 2), dtype=np.float64))
    assert float(vector_reward[0]) == pytest.approx(scalar_reward, abs=1e-9)
    for field in (
        "reward_total",
        "reward_goal_progress",
        "reward_obstacle",
        "reward_reference_speed",
        "reward_action_smoothness",
    ):
        assert float(vector_info[field][0]) == pytest.approx(float(scalar_info[field]), abs=1e-9)
