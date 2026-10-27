from __future__ import annotations

from math import pi
from typing import Any

import numpy as np
import pytest
from auv_planner.contracts import State

from auv_local.contracts import SurgeYawAction
from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig
from auv_rl.planar import PlanarKinematics, PlanarMotionLimits


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
