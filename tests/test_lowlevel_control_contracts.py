from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from auv_core.attitude import forward_quaternion
from auv_core.contracts import Reference, State
from auv_core.control.remus100 import Remus100TrackingController
from auv_core.guidance import ThreeDAlosGuidance
from auv_core.scenarios import remus100_vehicle

from auv_lowlevel.actions import ActionPipeline, ActuatorLimits
from auv_lowlevel.config import load_config
from auv_lowlevel.env import LowLevelEnv
from auv_lowlevel.guidance import TerminalCaptureGuidance
from auv_lowlevel.path import generate_sinusoidal_3d_path
from auv_lowlevel.reward import RewardComputer
from auv_lowlevel.shadow import ShadowSpeedIntegrator


def _config():
    return load_config("configs/lowlevel/odarrl_control_base.json")


def _limits() -> ActuatorLimits:
    vehicle = remus100_vehicle()
    assert vehicle.max_rudder_angle_rad is not None
    assert vehicle.max_stern_plane_angle_rad is not None
    assert vehicle.max_rudder_rate_rad_s is not None
    assert vehicle.max_stern_plane_rate_rad_s is not None
    assert vehicle.max_propeller_rpm is not None
    return ActuatorLimits(
        max_rudder_angle_rad=vehicle.max_rudder_angle_rad,
        max_stern_plane_angle_rad=vehicle.max_stern_plane_angle_rad,
        max_rudder_rate_rad_s=vehicle.max_rudder_rate_rad_s,
        max_stern_plane_rate_rad_s=vehicle.max_stern_plane_rate_rad_s,
        max_propeller_rpm=vehicle.max_propeller_rpm,
    )


def _reference(speed_mps: float, stamp_s: float) -> Reference:
    return Reference(
        stamp_s=stamp_s,
        position_ned_m=np.zeros(3),
        quat_xyzw=forward_quaternion(np.array([1.0, 0.0, 0.0])),
        linear_velocity_body_mps=np.array([speed_mps, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
        valid_until_s=stamp_s + 0.02,
    )


def _state(t_s: float, surge_mps: float) -> State:
    return State(
        t_s=t_s,
        position_ned_m=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([surge_mps, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )


def test_shadow_integral_matches_teacher_with_and_without_rpm_saturation() -> None:
    config = _config()
    controller = Remus100TrackingController(remus100_vehicle())
    shadow = ShadowSpeedIntegrator(config.controller, max_propeller_rpm=1525.0)
    for index, (speed, surge) in enumerate(
        [(1.2, 0.8), (1.2, 1.0), (2.4, 0.2), (1.2, 1.4), (1.2, 0.7)]
    ):
        t_s = index * 0.02
        command = controller.control(_state(t_s, surge), _reference(speed, t_s))
        assert controller.last_diagnostics is not None
        shadow_state = shadow.advance(surge, speed, 0.02)
        assert shadow_state.speed_integral_rpm == pytest.approx(
            controller.last_diagnostics.speed_integral_rpm,
            abs=1e-12,
        )
        assert command.propeller_rpm <= 1525.0
    assert controller.last_diagnostics is not None


def test_environment_shadow_matches_ground_speed_teacher_under_current() -> None:
    config = load_config(
        "configs/lowlevel/odarrl_control_i_residual_26d_rpm_only_hidden_timevarying_strict_capture.json"
    )
    env = LowLevelEnv(config)
    env.reset(11)
    assert env.randomization is not None
    assert np.linalg.norm(env.randomization.current_ned_mps) > 0.0
    for _ in range(8):
        result = env.step()
        assert result.info["shadow_teacher_margin_rpm"] == pytest.approx(0.0, abs=1e-10)


def test_action_pipeline_records_magnitude_and_rate_limiting() -> None:
    config = _config()
    pipeline = ActionPipeline(config, _limits())
    zero = np.zeros(3)

    no_residual = pipeline.apply(
        np.array([0.05, -0.05, 760.0]),
        zero,
        previous_total=zero,
        dt_s=0.02,
    )
    assert np.array_equal(no_residual.total, np.array([0.05, -0.05, 760.0]))
    assert not no_residual.saturated
    assert not no_residual.material_saturated
    assert not no_residual.residual_clipped

    saturated = pipeline.apply(
        np.array([0.40, -0.40, 2000.0]),
        np.array([2.0, -2.0, 2.0]),
        previous_total=zero,
        dt_s=0.02,
    )
    assert np.all(np.abs(saturated.residual_normalized) == 1.0)
    assert saturated.magnitude_clipped[0]
    assert saturated.magnitude_clipped[1]
    assert saturated.magnitude_clipped[2]
    assert saturated.rate_clipped[:2].all()
    assert saturated.residual_clipped
    assert saturated.material_saturated
    assert saturated.constraint_violation_count >= 5
    assert saturated.total[0] == pytest.approx(np.deg2rad(15.0) / 0.1 * 0.02)

    at_limit = pipeline.apply(
        np.array([np.deg2rad(15.0) + 1e-9, -np.deg2rad(15.0) - 1e-9, 760.0]),
        zero,
        previous_total=np.array([np.deg2rad(15.0), -np.deg2rad(15.0), 760.0]),
        dt_s=0.02,
    )
    assert not at_limit.saturated
    assert not at_limit.material_saturated
    assert at_limit.constraint_violation_count == 0


def test_reward_components_and_terminal_values() -> None:
    computer = RewardComputer(_config())
    common = {
        "previous_progress_m": 1.0,
        "progress_m": 2.0,
        "previous_goal_distance_m": 20.0,
        "goal_distance_m": 19.0,
        "error_3d_m": 0.5,
        "yaw_error_rad": 0.2,
        "water_surge_mps": 1.1,
        "reference_speed_mps": 1.2,
        "residual_normalized": np.array([1.0, 0.0, 0.0]),
        "previous_residual_normalized": np.zeros(3),
        "saturated": True,
    }

    goal = computer.compute(**common, terminal_kind="goal")
    timeout = computer.compute(**common, terminal_kind="timeout")
    bounds = computer.compute(**common, terminal_kind="out_of_bounds")
    assert goal.progress == pytest.approx(15.0)
    assert goal.terminal_progress == pytest.approx(0.0)
    assert goal.track_3d == pytest.approx(-1.0)
    assert goal.track_yaw == pytest.approx(-0.16)
    assert goal.track_u == pytest.approx(-0.05)
    assert goal.action == pytest.approx(-0.05)
    assert goal.saturation == pytest.approx(-0.1)
    assert goal.terminal == pytest.approx(50.0)
    assert timeout.terminal == pytest.approx(-20.0)
    assert bounds.terminal == pytest.approx(-50.0)
    assert goal.ppo_scaled == pytest.approx(goal.total * 0.01)


def test_terminal_progress_reward_is_potential_shaped_and_radius_clipped() -> None:
    config = _config()
    config = replace(
        config,
        reward=replace(config.reward, terminal_progress_weight=10.0),
    )
    computer = RewardComputer(config)
    common = {
        "previous_progress_m": 1.0,
        "progress_m": 1.0,
        "error_3d_m": 0.0,
        "yaw_error_rad": 0.0,
        "water_surge_mps": 0.0,
        "reference_speed_mps": 0.0,
        "residual_normalized": np.zeros(3),
        "previous_residual_normalized": np.zeros(3),
        "saturated": False,
        "terminal_kind": None,
    }

    entering = computer.compute(
        **common,
        previous_goal_distance_m=20.0,
        goal_distance_m=15.0,
    )
    approaching = computer.compute(
        **common,
        previous_goal_distance_m=15.0,
        goal_distance_m=14.0,
    )
    receding = computer.compute(
        **common,
        previous_goal_distance_m=14.0,
        goal_distance_m=17.0,
    )

    assert entering.terminal_progress == pytest.approx(10.0)
    assert approaching.terminal_progress == pytest.approx(10.0)
    assert receding.terminal_progress == pytest.approx(-20.0)
    assert entering.total == pytest.approx(entering.terminal_progress)


def test_single_environment_reports_observation_action_and_timing_contracts() -> None:
    env = LowLevelEnv(_config())
    first = env.reset(3)
    second = LowLevelEnv(_config()).reset(3)
    assert first.shape == (27,)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert first[26] == 0.0

    result = env.step()
    assert result.observation.shape == (27,)
    assert np.all(np.isfinite(result.observation))
    assert result.info["base_command"].shape == (3,)
    assert result.info["residual_normalized"].shape == (3,)
    assert result.info["delta_command"].shape == (3,)
    assert result.info["requested_total"].shape == (3,)
    assert result.info["limited_total"].shape == (3,)
    assert result.info["actual_actuator"].shape == (3,)
    assert result.info["shadow_teacher_margin_rpm"] == pytest.approx(0.0, abs=1e-10)
    assert result.info["pvs_step_latency_s"] >= 0.0
    assert result.info["control_loop_latency_s"] >= result.info["pvs_step_latency_s"]
    assert result.info["state_after"].t_s == pytest.approx(0.02)
    assert not result.terminated
    assert not result.truncated


def test_environment_seed_is_repeatable_for_first_steps() -> None:
    config = _config()
    first_env = LowLevelEnv(config)
    second_env = LowLevelEnv(config)
    first_obs = first_env.reset(9)
    second_obs = second_env.reset(9)
    assert np.array_equal(first_obs, second_obs)
    for _ in range(4):
        first = first_env.step()
        second = second_env.step()
        assert np.array_equal(first.observation, second.observation)
        assert first.reward.total == pytest.approx(second.reward.total)
        assert np.array_equal(first.info["limited_total"], second.info["limited_total"])
        assert np.array_equal(first.info["actual_actuator"], second.info["actual_actuator"])


def test_current_exposure_only_changes_indices_17_to_19() -> None:
    config = _config()
    exposed_env = LowLevelEnv(config, expose_current=True)
    hidden_env = LowLevelEnv(config, expose_current=False)
    exposed = exposed_env.reset(5)
    hidden = hidden_env.reset(5)

    assert np.any(np.abs(exposed[17:20]) > 1e-12)
    assert np.all(hidden[17:20] == 0.0)
    mask = np.ones(27, dtype=bool)
    mask[17:20] = False
    assert np.array_equal(exposed[mask], hidden[mask])


def test_terminal_capture_guides_toward_goal_near_path_end() -> None:
    config = _config()
    path = generate_sinusoidal_3d_path(config.reference)
    inner = ThreeDAlosGuidance(
        3.0,
        2.0,
        config.reference.speed_mps,
        current_field=None,
        compensate_current=False,
    )
    guidance = TerminalCaptureGuidance(inner, path.goal_ned_m, capture_distance_m=8.0)
    state = State(
        t_s=0.0,
        position_ned_m=path.goal_ned_m + np.array([0.0, 2.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.2, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )

    solution = guidance.solve(state, path.trajectory(), 0.0)
    desired = path.goal_ned_m - state.position_ned_m
    desired /= np.linalg.norm(desired)

    assert np.dot(solution.desired_body_direction_ned, desired) > 0.99
    assert solution.command_speed_mps == pytest.approx(config.reference.speed_mps)
    assert np.array_equal(solution.reference.position_ned_m, path.goal_ned_m)

    far_state = State(
        t_s=0.0,
        position_ned_m=path.goal_ned_m - np.array([10.0, 0.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.2, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )
    far_solution = guidance.solve(far_state, path.trajectory(), 0.0)
    base_solution = inner.solve(far_state, path.trajectory(), 0.0)
    assert np.array_equal(far_solution.reference.position_ned_m, base_solution.reference.position_ned_m)


def test_terminal_capture_speed_ramps_down_near_goal() -> None:
    config = _config()
    path = generate_sinusoidal_3d_path(config.reference)
    inner = ThreeDAlosGuidance(
        3.0,
        2.0,
        config.reference.speed_mps,
        current_field=None,
        compensate_current=False,
    )
    guidance = TerminalCaptureGuidance(
        inner,
        path.goal_ned_m,
        capture_distance_m=8.0,
        terminal_speed_mps=0.4,
    )
    state = State(
        t_s=0.0,
        position_ned_m=path.goal_ned_m + np.array([2.0, 0.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.2, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )

    solution = guidance.solve(state, path.trajectory(), 0.0)
    assert solution.command_speed_mps == pytest.approx(0.6)

    state_at_goal = State(
        t_s=0.0,
        position_ned_m=path.goal_ned_m + np.array([0.1, 0.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.2, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )
    near_solution = guidance.solve(state_at_goal, path.trajectory(), 0.0)
    assert near_solution.command_speed_mps == pytest.approx(0.41)
