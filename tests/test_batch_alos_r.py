from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.guidance import BatchedThreeDAlosGuidance
from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv, PCCommand
from auv_lowlevel.config import load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path

R1_CONFIG = "configs/lowlevel/odarrl_control_alos_r1_terminal.json"
R1B_CONFIG = "configs/lowlevel/odarrl_control_alos_r1b_legacy20_floor75.json"
R1C_CONFIG = "configs/lowlevel/odarrl_control_alos_r1c_ground_floor75.json"
R1E_CONFIG = "configs/lowlevel/odarrl_control_alos_r1e_terminal20_prespin75.json"
R1F_CONFIG = "configs/lowlevel/odarrl_control_alos_r1f_terminal20_noprespin75.json"
R1G_CONFIG = "configs/lowlevel/odarrl_control_alos_r1g_ground_floor_init110_nospin.json"
R1H_CONFIG = "configs/lowlevel/odarrl_control_alos_r1h_ground_floor_init110_prespin75.json"
R1I_CONFIG = "configs/lowlevel/odarrl_control_alos_r1i_noground_init110_nospin.json"
R1J_CONFIG = "configs/lowlevel/odarrl_control_alos_r1j_noground_init110_prespin75.json"
R1K_CONFIG = "configs/lowlevel/odarrl_control_alos_r1k_ground_floor_init100_nospin.json"
R4A_CONFIG = "configs/lowlevel/odarrl_control_alos_r4a_launch_init125_nospin.json"
R4B_CONFIG = "configs/lowlevel/odarrl_control_alos_r4b_launch_init125_prespin75.json"
R4C_CONFIG = "configs/lowlevel/odarrl_control_alos_r4c_launch_init125_prespin100.json"
R2B_CONFIG = "configs/lowlevel/odarrl_control_alos_r2b_crab_floor_init100.json"
R3B_CONFIG = "configs/lowlevel/odarrl_control_alos_r3b_governor_floor_init100.json"
R2_CONFIG = "configs/lowlevel/odarrl_control_alos_r2_crab.json"
R3_CONFIG = "configs/lowlevel/odarrl_control_alos_r3_governor.json"


def _solution(config_path: str, position: torch.Tensor, **kwargs):
    config = load_config(config_path)
    path = generate_sinusoidal_3d_path(config.reference)
    batched_path = BatchedReferencePath.from_reference_path(path)
    projection = batched_path.project(position)
    guidance = BatchedThreeDAlosGuidance(config.guidance, config.reference)
    solution = guidance.solve(
        path=batched_path,
        projection=projection,
        position_ned_m=position,
        current_ned_mps=torch.zeros_like(position),
        desired_ground_speed_mps=torch.full(
            (position.shape[0],), float(config.reference.speed_mps)
        ),
        **kwargs,
    )
    return config, batched_path, solution


def test_alos_r_terminal_keeps_cruise_speed_at_one_metre() -> None:
    config = load_config(R1_CONFIG)
    path = generate_sinusoidal_3d_path(config.reference)
    goal = torch.as_tensor(path.goal_ned_m, dtype=torch.float64)
    position = (goal - torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))[None, :]

    _, _, solution = _solution(R1_CONFIG, position)

    assert bool(solution.terminal_phase[0])
    terminal_ratio = float(solution.command_speed_mps[0] / config.reference.speed_mps)
    assert terminal_ratio >= 0.99


def test_alos_r_terminal_heading_points_at_goal_without_speed_change() -> None:
    config = load_config(R1_CONFIG)
    path = generate_sinusoidal_3d_path(config.reference)
    goal = torch.as_tensor(path.goal_ned_m, dtype=torch.float64)
    position = (goal - torch.tensor([1.0, -1.0, 0.2], dtype=torch.float64))[None, :]

    _, batched_path, solution = _solution(R1_CONFIG, position)

    target = goal - position[0]
    target_heading = float(torch.atan2(target[1], target[0]))
    assert bool(solution.terminal_phase[0])
    assert float(solution.command_speed_mps[0]) == pytest.approx(
        config.reference.speed_mps, abs=1e-6
    )
    assert abs(float(solution.desired_heading_rad[0]) - target_heading) < 1e-12
    assert torch.linalg.norm(solution.reference_position_ned_m[0] - batched_path.goal_ned_m) < 1e-12


def test_alos_r_bounded_terminal_braking_never_reaches_zero() -> None:
    config = load_config(R1_CONFIG)
    guidance = replace(
        config.guidance,
        terminal_braking_enabled=True,
        terminal_speed_ratio=0.75,
    )
    path = generate_sinusoidal_3d_path(config.reference)
    batched_path = BatchedReferencePath.from_reference_path(path)
    goal = torch.as_tensor(path.goal_ned_m, dtype=torch.float64)
    positions = torch.stack(
        (
            goal - torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            goal - torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64),
            goal - torch.tensor([20.0, 0.0, 0.0], dtype=torch.float64),
        )
    )
    alos = BatchedThreeDAlosGuidance(guidance, config.reference)
    solution = alos.solve(
        path=batched_path,
        projection=batched_path.project(positions),
        position_ned_m=positions,
        current_ned_mps=torch.zeros_like(positions),
        desired_ground_speed_mps=torch.full((3,), float(config.reference.speed_mps)),
    )

    cruise = float(config.reference.speed_mps)
    assert float(solution.command_speed_mps[0]) >= 0.75 * cruise
    assert float(solution.command_speed_mps[0]) < float(solution.command_speed_mps[1])
    assert float(solution.command_speed_mps[1]) <= cruise
    assert float(solution.command_speed_mps[2]) == pytest.approx(cruise, abs=1e-6)


def test_alos_r_route_speed_floor_holds_across_terminal_approach() -> None:
    config = load_config(R1B_CONFIG)
    path = generate_sinusoidal_3d_path(config.reference)
    goal = torch.as_tensor(path.goal_ned_m, dtype=torch.float64)
    distances = torch.linspace(0.25, 30.0, 48, dtype=torch.float64)
    positions = goal[None, :] - torch.stack(
        (distances, torch.zeros_like(distances), torch.zeros_like(distances)), dim=1
    )

    _, _, solution = _solution(R1B_CONFIG, positions)

    ratios = solution.command_speed_mps / float(config.reference.speed_mps)
    assert float(ratios.min()) >= 0.75 - 1e-9
    assert float(ratios[distances <= 20.0].min()) >= 0.75 - 1e-9
    assert bool(solution.terminal_phase[distances <= 20.0].all())


def test_alos_r_rejects_terminal_speed_below_route_floor() -> None:
    config = load_config(R1_CONFIG)

    with pytest.raises(ValueError, match="terminal_speed_ratio"):
        replace(
            config.guidance,
            minimum_speed_ratio=0.75,
            terminal_speed_ratio=0.70,
        )


def test_alos_r_environment_enforces_route_speed_floor() -> None:
    config = load_config(R3_CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=2, device="cpu", dtype=torch.float64)
    env.reset([0, 1])
    env._episode_command_speed_mps = torch.full((2,), 2.4, dtype=torch.float64)
    env._material_saturation_ema = torch.ones(2, dtype=torch.float64)
    env._high_saturation_time_s = torch.full((2,), 3.0, dtype=torch.float64)
    env.config = replace(
        env.config,
        guidance=replace(
            env.config.guidance,
            saturation_governor_min_scale=0.5,
            saturation_emergency_extra_reduction=0.2,
        ),
    )

    command, _, solution = env._alos_command()

    floor = 0.75 * 2.4
    assert torch.all(command.speed_mps >= floor)
    assert torch.all(solution.command_speed_mps >= floor)


def test_alos_r_ground_floor_boosts_below_floor_and_is_goal_independent() -> None:
    config = load_config(R1C_CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=1, device="cpu", dtype=torch.float64)
    env.reset([0])
    path = env.path
    goal = path.goal_ned_m_for(1)

    def set_route_speed(speed_mps: float) -> None:
        assert env.sensor_output is not None
        projection = env.path.project(
            env.sensor_output.p_nav,
            segment_hint=env._projection_hint,
            search_radius=env.reset_projection_search_radius,
        )
        body = projection.tangent_ned[0] * speed_mps
        env.sensor_output = replace(
            env.sensor_output,
            dvl_ground_body=body[None, :],
            bottom_lock_valid=torch.ones(1, dtype=torch.float64),
            dvl_bottom_age_s=torch.zeros(1, dtype=torch.float64),
            attitude=torch.zeros((1, 3), dtype=torch.float64),
        )

    env._reset_guidance_state(torch.ones(1, dtype=torch.bool))
    set_route_speed(0.5)
    near_command, _, _ = env._alos_command()
    near_speed = float(near_command.speed_mps[0])
    assert near_speed > float(config.reference.speed_mps) + 0.1
    assert bool(env._route_speed_floor_active[0])

    env._reset_guidance_state(torch.ones(1, dtype=torch.bool))
    assert env.sensor_output is not None
    env.sensor_output = replace(
        env.sensor_output,
        p_nav=goal - torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    set_route_speed(0.5)
    terminal_command, _, _ = env._alos_command()
    terminal_speed = float(terminal_command.speed_mps[0])
    assert terminal_speed == pytest.approx(near_speed, rel=0.0, abs=1e-9)

    env._reset_guidance_state(torch.ones(1, dtype=torch.bool))
    set_route_speed(2.0)
    fast_command, _, _ = env._alos_command()
    assert float(fast_command.speed_mps[0]) == pytest.approx(
        float(config.reference.speed_mps), abs=1e-9
    )
    assert not bool(env._route_speed_floor_active[0])


def test_alos_r_invalid_current_disables_crab_and_preserves_baseline() -> None:
    config = load_config(R2_CONFIG)
    path = generate_sinusoidal_3d_path(config.reference)
    position = torch.as_tensor(
        [path.goal_ned_m[0] - 5.0, path.goal_ned_m[1] + 1.0, path.goal_ned_m[2]],
        dtype=torch.float64,
    )[None, :]
    estimate = torch.tensor([[0.3, 0.2, 0.0]], dtype=torch.float64)

    _, _, disabled = _solution(R2_CONFIG, position)
    _, _, invalid = _solution(
        R2_CONFIG,
        position,
        estimated_current_ned_mps=estimate,
        estimated_current_valid=torch.tensor([False]),
    )

    assert torch.allclose(invalid.command_speed_mps, disabled.command_speed_mps)
    assert torch.allclose(
        invalid.desired_body_direction_ned, disabled.desired_body_direction_ned
    )
    assert torch.equal(invalid.crab_angle_rad, torch.zeros_like(invalid.crab_angle_rad))


def test_alos_r_current_age_gate_uses_estimate_only_when_fresh() -> None:
    config = load_config(R2_CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=2, device="cpu", dtype=torch.float64)
    env.reset([0, 1])
    assert env.sensor_output is not None
    estimate_body = torch.tensor(
        [[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]], dtype=torch.float64
    )
    env.sensor_output = replace(
        env.sensor_output,
        current_estimate_body=estimate_body,
        current_estimate_valid=torch.ones(2, dtype=torch.float64),
        current_estimate_age_s=torch.tensor([0.3, 0.6001], dtype=torch.float64),
    )

    estimate, valid, age = env._current_estimate_for_guidance()

    assert torch.equal(valid, torch.tensor([True, False]))
    assert torch.allclose(age, torch.tensor([0.3, 0.6001], dtype=torch.float64))
    assert torch.count_nonzero(estimate[0]) > 0
    assert torch.equal(estimate[1], torch.zeros(3, dtype=torch.float64))


def test_alos_r_saturation_governor_is_high_speed_only_and_goal_independent() -> None:
    config = load_config(R3_CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=2, device="cpu", dtype=torch.float64)
    env.reset([0, 1])
    env._material_saturation_ema = torch.tensor([0.55, 0.55], dtype=torch.float64)

    low_speed, low_scale = env._governed_speed(
        torch.tensor([1.7, 2.4], dtype=torch.float64)
    )

    assert torch.equal(low_speed[0], torch.tensor(1.7, dtype=torch.float64))
    assert torch.equal(low_scale[0], torch.tensor(1.0, dtype=torch.float64))
    assert low_speed[1] < 2.4
    assert 0.88 <= float(low_scale[1]) < 1.0


@pytest.mark.parametrize(
    ("command_rate_limit_enabled", "governor_enabled", "expect_limited"),
    (
        (None, False, False),
        (None, True, True),
        (True, False, True),
        (False, True, False),
    ),
)
def test_alos_r_command_rate_limit_is_independent_from_governor(
    command_rate_limit_enabled: bool | None,
    governor_enabled: bool,
    expect_limited: bool,
) -> None:
    config = load_config(R2B_CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=1, device="cpu", dtype=torch.float64)
    env.config = replace(
        env.config,
        guidance=replace(
            env.config.guidance,
            saturation_governor_enabled=governor_enabled,
            command_rate_limit_enabled=command_rate_limit_enabled,
        ),
    )
    env._guidance_previous_command = PCCommand(
        heading_rad=torch.zeros(1, dtype=torch.float64),
        pitch_rad=torch.zeros(1, dtype=torch.float64),
        speed_mps=torch.zeros(1, dtype=torch.float64),
    )
    env._guidance_command_initialized = torch.ones(1, dtype=torch.bool)
    target = PCCommand(
        heading_rad=torch.ones(1, dtype=torch.float64),
        pitch_rad=torch.ones(1, dtype=torch.float64),
        speed_mps=torch.ones(1, dtype=torch.float64),
    )

    actual = env._limit_guidance_command(target)

    if expect_limited:
        assert float(actual.heading_rad[0]) == pytest.approx(
            env.dt_s * config.guidance.heading_rate_limit_rad_s
        )
        assert float(actual.pitch_rad[0]) == pytest.approx(
            env.dt_s * config.guidance.pitch_rate_limit_rad_s
        )
        assert float(actual.speed_mps[0]) == pytest.approx(
            env.dt_s * config.guidance.speed_rate_limit_mps2
        )
    else:
        assert torch.equal(actual.heading_rad, target.heading_rad)
        assert torch.equal(actual.pitch_rad, target.pitch_rad)
        assert torch.equal(actual.speed_mps, target.speed_mps)


def test_existing_configs_keep_alos_r_disabled_by_default() -> None:
    config = load_config("configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json")
    guidance = config.guidance

    assert guidance.robust_terminal_enabled is False
    assert guidance.current_crab_enabled is False
    assert guidance.speed_adaptive_lookahead_enabled is False
    assert guidance.saturation_governor_enabled is False
    assert guidance.terminal_speed_ratio == 1.0
    assert guidance.route_speed_floor_enabled is False


def test_alos_r_profiles_enable_only_the_intended_stage() -> None:
    r1 = load_config(R1_CONFIG).guidance
    r1b = load_config(R1B_CONFIG).guidance
    r1c = load_config(R1C_CONFIG).guidance
    r1e = load_config(R1E_CONFIG).guidance
    r1f = load_config(R1F_CONFIG).guidance
    r1g = load_config(R1G_CONFIG).guidance
    r1h = load_config(R1H_CONFIG).guidance
    r1i = load_config(R1I_CONFIG).guidance
    r1j = load_config(R1J_CONFIG).guidance
    r1k = load_config(R1K_CONFIG).guidance
    r2b = load_config(R2B_CONFIG).guidance
    r3b = load_config(R3B_CONFIG).guidance
    r2 = load_config(R2_CONFIG).guidance
    r3 = load_config(R3_CONFIG).guidance

    assert r1.robust_terminal_enabled and not r1.current_crab_enabled
    assert not r1.speed_adaptive_lookahead_enabled and not r1.saturation_governor_enabled
    assert r2.robust_terminal_enabled and r2.current_crab_enabled
    assert not r2.speed_adaptive_lookahead_enabled and not r2.saturation_governor_enabled
    assert r3.robust_terminal_enabled and r3.current_crab_enabled
    assert r3.speed_adaptive_lookahead_enabled and r3.saturation_governor_enabled
    assert np.isclose(r1.terminal_speed_ratio, 1.0)
    assert r1b.terminal_braking_enabled
    assert np.isclose(r1b.terminal_speed_ratio, 0.75)
    assert np.isclose(r1b.minimum_speed_ratio, 0.75)
    assert r1b.terminal_alignment_min_m == r1b.terminal_alignment_max_m == 20.0
    assert r1b.route_speed_floor_enabled is False
    assert r1c.robust_terminal_enabled and not r1c.current_crab_enabled
    assert r1c.route_speed_floor_enabled
    assert np.isclose(r1c.terminal_speed_ratio, 1.0)
    assert r1c.terminal_alignment_min_m == r1c.terminal_alignment_max_m == 20.0
    assert not r1e.route_speed_floor_enabled
    assert r1e.initial_propeller_rpm_ratio == 1.0
    assert not r1f.route_speed_floor_enabled
    assert r1f.initial_propeller_rpm_ratio == 0.0
    assert r1g.route_speed_floor_enabled and r1h.route_speed_floor_enabled
    assert not r1i.route_speed_floor_enabled and not r1j.route_speed_floor_enabled
    for guidance in (r1g, r1h, r1i, r1j):
        assert np.isclose(guidance.initial_speed_ratio, 1.10)
        assert np.isclose(guidance.minimum_speed_ratio, 0.75)
    assert r1g.initial_propeller_rpm_ratio == 0.0
    assert r1h.initial_propeller_rpm_ratio == 0.75
    assert r1i.initial_propeller_rpm_ratio == 0.0
    assert r1j.initial_propeller_rpm_ratio == 0.75
    assert r1k.route_speed_floor_enabled
    assert r1k.initial_propeller_rpm_ratio == 0.0
    assert np.isclose(r1k.initial_speed_ratio, 1.0)
    assert r1k.route_speed_floor_enabled
    assert r2b.route_speed_floor_enabled and r2b.current_crab_enabled
    assert not r2b.speed_adaptive_lookahead_enabled
    assert r3b.route_speed_floor_enabled and r3b.current_crab_enabled
    assert r3b.speed_adaptive_lookahead_enabled and r3b.saturation_governor_enabled
    assert np.isclose(r2.terminal_speed_ratio, 1.0)
    assert np.isclose(r3.terminal_speed_ratio, 1.0)


def test_alos_r_launch_speed_ratio_defaults_to_noop() -> None:
    guidance = load_config(R1C_CONFIG).guidance

    assert guidance.route_speed_floor_launch_speed_ratio == 1.0
    assert guidance.route_speed_floor_launch_duration_s == 2.0
    assert guidance.initial_propeller_rpm_ratio == 1.0
    assert guidance.initial_speed_ratio == 1.0


def test_alos_r_initial_propeller_ratio_pre_spins_only_when_enabled() -> None:
    enabled = BatchedPCLowLevelEnv(
        load_config(R1C_CONFIG), num_envs=2, device="cpu", dtype=torch.float64
    )
    enabled.reset([0, 1])
    expected_rpm = (
        enabled.config.guidance.initial_propeller_rpm_ratio
        * enabled.config.controller.speed_feedforward_rpm_per_mps
        * float(enabled.config.reference.speed_mps)
    )
    assert torch.allclose(
        enabled.actuator[:, 2],
        torch.full((2,), expected_rpm, dtype=torch.float64),
    )

    disabled = BatchedPCLowLevelEnv(
        load_config(R1_CONFIG), num_envs=1, device="cpu", dtype=torch.float64
    )
    disabled.reset([0])
    assert torch.equal(disabled.actuator[:, 2], torch.zeros(1, dtype=torch.float64))


def test_alos_r_initial_speed_ratio_scales_state_and_pre_spin() -> None:
    config = load_config(R1C_CONFIG)
    config = replace(
        config,
        guidance=replace(config.guidance, initial_speed_ratio=1.15),
    )
    env = BatchedPCLowLevelEnv(config, num_envs=1, device="cpu", dtype=torch.float64)
    env.reset([0])
    expected_speed = 1.15 * float(config.reference.speed_mps)
    expected_rpm = min(
        config.guidance.initial_propeller_rpm_ratio
        * config.controller.speed_feedforward_rpm_per_mps
        * expected_speed,
        float(config.actuator.plant_max_propeller_rpm),
    )
    assert float(env.nu[0, 0]) == pytest.approx(expected_speed, abs=1e-9)
    assert float(env.actuator[0, 2]) == pytest.approx(expected_rpm, abs=1e-9)


def test_alos_r_initial_speed_and_prespin_are_clamped_to_vehicle_envelope() -> None:
    config = load_config(R4C_CONFIG)
    config = replace(
        config,
        reference=replace(config.reference, command_speed_range_mps=(2.4, 2.4)),
    )
    env = BatchedPCLowLevelEnv(config, num_envs=1, device="cpu", dtype=torch.float64)

    env.reset([0])

    expected_speed = float(env.vehicle.max_speed_mps)
    expected_rpm = min(
        config.guidance.initial_propeller_rpm_ratio
        * config.controller.speed_feedforward_rpm_per_mps
        * expected_speed,
        float(config.actuator.plant_max_propeller_rpm),
    )
    assert float(env.nu[0, 0]) == pytest.approx(expected_speed, abs=1e-9)
    assert float(env.actuator[0, 2]) == pytest.approx(expected_rpm, abs=1e-9)


def test_alos_r_launch_boost_respects_window_and_is_goal_independent() -> None:
    config = load_config(R1C_CONFIG)
    config = replace(
        config,
        guidance=replace(
            config.guidance,
            route_speed_floor_launch_speed_ratio=1.15,
            route_speed_floor_launch_duration_s=2.0,
        ),
    )
    env = BatchedPCLowLevelEnv(config, num_envs=1, device="cpu", dtype=torch.float64)
    env.reset([0])
    assert env.sensor_output is not None
    def set_route_speed(speed_mps: float) -> None:
        assert env.sensor_output is not None
        projection = env.path.project(
            env.sensor_output.p_nav,
            segment_hint=env._projection_hint,
            search_radius=env.reset_projection_search_radius,
        )
        env.sensor_output = replace(
            env.sensor_output,
            dvl_ground_body=(projection.tangent_ned[0] * speed_mps)[None, :],
            bottom_lock_valid=torch.ones(1, dtype=torch.float64),
            dvl_bottom_age_s=torch.zeros(1, dtype=torch.float64),
            attitude=torch.zeros((1, 3), dtype=torch.float64),
        )

    env._reset_guidance_state(torch.ones(1, dtype=torch.bool))
    set_route_speed(2.0)
    launch_command, _, _ = env._alos_command()
    launch_target = float(launch_command.speed_mps[0])
    assert bool(env._route_speed_floor_launch_active[0])
    assert launch_target == pytest.approx(
        1.15 * float(config.reference.speed_mps), abs=1e-9
    )

    goal = env.path.goal_ned_m_for(1)
    assert env.sensor_output is not None
    env.sensor_output = replace(
        env.sensor_output,
        p_nav=goal - torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    set_route_speed(2.0)
    terminal_command, _, _ = env._alos_command()
    assert float(terminal_command.speed_mps[0]) == pytest.approx(
        launch_target, abs=1e-9
    )

    env.t_s = torch.full((1,), 3.0, dtype=torch.float64)
    env._reset_guidance_state(torch.ones(1, dtype=torch.bool))
    set_route_speed(2.0)
    cruise_command, _, _ = env._alos_command()
    assert not bool(env._route_speed_floor_launch_active[0])
    assert float(cruise_command.speed_mps[0]) == pytest.approx(
        float(config.reference.speed_mps), abs=1e-9
    )
