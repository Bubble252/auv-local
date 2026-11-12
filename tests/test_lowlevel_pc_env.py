from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.geometry import (
    legacy_ground_velocity_body,
    quaternion_from_euler,
    world_to_body,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_path import build_pc_path_features
from auv_lowlevel.config import (
    DomainCellConfig,
    DomainConfig,
    LowLevelConfig,
    parse_config,
)

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)


def _pc_config(
    stage: str,
    *,
    profile: str = "clean",
    current_enabled: bool = False,
    max_episode_steps: int = 100,
    path_bank_size: int = 1,
    path_bank_seed: int = 0,
    command_speed_range_mps: tuple[float, float] | None = None,
) -> LowLevelConfig:
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": f"pc_env_{stage.lower()}",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": stage,
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["plant"]["max_episode_steps"] = max_episode_steps
    data["reference"]["path_bank_size"] = path_bank_size
    data["reference"]["path_bank_seed"] = path_bank_seed
    if command_speed_range_mps is not None:
        data["reference"]["command_speed_range_mps"] = list(command_speed_range_mps)
    data["current"]["enabled"] = current_enabled
    data["current"]["speed_mps"] = [0.1, 0.3] if current_enabled else [0.0, 0.0]
    data["current"]["direction_rad"] = [0.0, 6.2]
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["observation"]["expose_current"] = False
    data["observation"]["current_contract"] = "hidden"
    data["sensors"] = {"profile": profile, "seed": 17}
    data["reward"].update(
        {
            "mode": "pc_command_track",
            "hold_bonus": stage in {"C0", "C1"},
            "w_prog": 0.0,
            "w_3d": 0.0,
            "w_yaw": 0.0,
            "w_u": 0.0,
            "w_action": 0.0,
            "w_sat": 0.0,
            "terminal_progress_weight": 0.0,
            "goal_bonus": 0.0,
            "timeout_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
        }
    )
    data["actuator"] = {
        "envelope": "pc20",
        "max_rudder_angle_deg": 20.0,
        "max_stern_plane_angle_deg": 20.0,
        "max_fin_rate_deg_s": 150.0,
        "max_propeller_rpm": 1525.0,
        "plant_max_rudder_angle_deg": 20.0,
        "plant_max_stern_plane_angle_deg": 20.0,
        "plant_max_fin_rate_deg_s": 150.0,
        "plant_max_propeller_rpm": 1525.0,
        "material_tolerance_fraction": 0.01,
    }
    data["termination"] = {
        "phi_term_deg": 45.0,
        "t_phi_term_s": 1.0,
        "p_term_deg_s": 90.0,
        "t_p_term_s": 0.5,
        "safety_terminal_penalty": -50.0,
    }
    return parse_config(data)


def test_c0_clean_env_reset_and_short_rollout() -> None:
    config = _pc_config("C0", profile="clean")
    env = BatchedPCLowLevelEnv(config, num_envs=4, validate_finite=True)
    observation = env.reset([10, 11, 12, 13])
    assert observation.actor_state.shape == (4, 12)
    assert observation.critic_state.shape == (4, 15)
    assert observation.command.normalized().shape == (4, 3)
    assert torch.isfinite(observation.actor_state).all()
    assert torch.isfinite(observation.critic_state).all()

    initial_command = observation.command.normalized().clone()
    for _ in range(20):
        result = env.step(torch.zeros((4, 3), dtype=torch.float64))
        assert torch.isfinite(result.observation.actor_state).all()
        assert torch.isfinite(result.reward.total).all()
    assert result.terminated.shape == (4,)
    assert result.truncated.shape == (4,)
    assert torch.equal(
        result.observation.command.normalized(),
        initial_command,
    )


def test_ideal_actor_observations_use_true_state_current_and_endpoint_path() -> None:
    config = _pc_config("C3", profile="realistic", current_enabled=True)
    config = replace(
        config,
        observation=replace(
            config.observation,
            ideal_actor_observations=True,
            path_endpoint_context_enabled=True,
        ),
    )
    env = BatchedPCLowLevelEnv(
        config,
        num_envs=4,
        validate_finite=True,
        path_dim=12,
    )
    observation = env.reset([501, 502, 503, 504])
    assert env.eta is not None and env.nu is not None and env.current_ned_mps is not None
    expected_ground_velocity = legacy_ground_velocity_body(
        env.eta,
        env.nu,
        env.current_ned_mps,
    )
    expected_actor_state = torch.cat(
        [env.eta[:, :3], env.eta[:, 3:6], env.nu[:, 3:6], expected_ground_velocity],
        dim=1,
    )
    torch.testing.assert_close(observation.actor_state, expected_actor_state)
    expected_current_body = world_to_body(
        env.current_ned_mps,
        quaternion_from_euler(env.eta[:, 3], env.eta[:, 4], env.eta[:, 5]),
    )
    torch.testing.assert_close(observation.actor_current_body, expected_current_body)
    projection = env.path.project(env.eta[:, :3])
    expected_path = build_pc_path_features(
        projection,
        attitude=env.eta[:, 3:6],
        path=env.path,
        position_ned_m=env.eta[:, :3],
        path_blocks=env.path_blocks,
        endpoint_enter_m=env.endpoint_enter_m,
        endpoint_full_m=env.endpoint_full_m,
    )
    torch.testing.assert_close(observation.actor_path, expected_path)


def test_c0_command_speed_range_is_episode_fixed_and_initialized() -> None:
    config = _pc_config(
        "C0",
        profile="clean",
        current_enabled=True,
        command_speed_range_mps=(1.8, 2.0),
    )
    env = BatchedPCLowLevelEnv(config, num_envs=8, validate_finite=True)
    observation = env.reset([100, 101, 102, 103, 104, 105, 106, 107])

    speeds = env._episode_command_speed_mps
    assert torch.all(speeds >= 1.8 - 1e-12)
    assert torch.all(speeds <= 2.0 + 1e-12)
    assert torch.allclose(observation.command.speed_mps, speeds)
    assert env.nu is not None
    assert torch.allclose(env.nu[:, 0], speeds)
    assert not bool(env._command_speed_filtered_below_range.any())

    initial_speeds = speeds.clone()
    for _ in range(20):
        result = env.step(torch.zeros((8, 3), dtype=torch.float64))
    assert torch.allclose(result.observation.command.speed_mps, initial_speeds)


def test_pc_joint_domain_sampling_couples_speed_and_current() -> None:
    config = _pc_config(
        "C2",
        profile="clean",
        current_enabled=True,
        command_speed_range_mps=(1.0, 2.0),
    )
    config = replace(
        config,
        current=replace(
            config.current,
            min_speed_mps=0.1,
            max_speed_mps=0.45,
        ),
        domain=DomainConfig(
            cells=(
                DomainCellConfig("grid_lw", (1.0, 1.4), (0.1, 0.2)),
                DomainCellConfig("grid_ls", (1.0, 1.4), (0.3, 0.45)),
                DomainCellConfig("grid_hw", (1.8, 2.0), (0.1, 0.2)),
            )
        ),
    )
    env = BatchedPCLowLevelEnv(config, num_envs=64, validate_finite=True)
    env.reset(list(range(200, 264)))
    speeds = env._episode_command_speed_mps
    base_current = torch.linalg.norm(env.current_base_ned_mps[:, :2], dim=1)
    matched = torch.zeros(64, dtype=torch.bool)
    for cell in config.domain.cells:
        command_low, command_high = cell.command_speed_range_mps
        current_low, current_high = cell.current_speed_range_mps
        matched |= (
            (speeds >= command_low - 1e-12)
            & (speeds <= command_high + 1e-12)
            & (base_current >= current_low - 1e-12)
            & (base_current <= current_high + 1e-12)
        )
    assert bool(matched.all())
    assert not bool(((speeds > 1.4) & (base_current > 0.2)).any())
    assert env.nu is not None
    assert torch.allclose(env.nu[:, 0], speeds * config.guidance.initial_speed_ratio)


def test_c2_command_speed_range_drives_alos_and_reports_reachability() -> None:
    config = _pc_config(
        "C2",
        profile="clean",
        current_enabled=True,
        command_speed_range_mps=(1.8, 2.0),
    )
    env = BatchedPCLowLevelEnv(config, num_envs=4, validate_finite=True)
    observation = env.reset([110, 111, 112, 113])
    speeds = env._episode_command_speed_mps
    assert torch.allclose(observation.command.speed_mps, speeds)
    assert observation.solution is not None
    assert torch.allclose(observation.solution.reference_speed_mps, speeds)

    result = env.step(torch.zeros((4, 3), dtype=torch.float64))
    assert bool(result.info["command_reachability_feasible"].all())
    assert torch.all(
        result.info["command_water_relative_speed_mps"] <= 2.5 + 1e-9
    )


def test_c2_alos_uses_nav_position_and_runs_clean_closed_loop() -> None:
    config = _pc_config("C2", profile="clean")
    env = BatchedPCLowLevelEnv(config, num_envs=3, validate_finite=True)
    observation = env.reset([20, 21, 22])
    assert observation.projection is not None
    assert observation.solution is not None
    assert torch.equal(observation.sensor.p_nav, env.eta[:, :3])
    start_progress = observation.projection.progress_m.clone()
    for _ in range(40):
        result = env.step(torch.zeros((3, 3), dtype=torch.float64))
    assert torch.isfinite(result.observation.actor_state).all()
    assert torch.isfinite(result.reward.total).all()
    assert result.observation.projection is not None
    assert (result.observation.projection.progress_m > start_progress).all()


def test_pc_env_reports_ground_route_speed_and_optional_reward_terms() -> None:
    config = _pc_config("C2", profile="clean", current_enabled=True)
    config = replace(
        config,
        reward=replace(
            config.reward,
            w_route_speed_pc=1.0,
            w_material_sat_pc=0.2,
        ),
    )
    env = BatchedPCLowLevelEnv(config, num_envs=3, validate_finite=True)
    env.reset([25, 26, 27])
    result = env.step(torch.zeros((3, 3), dtype=torch.float64))

    route_speed = result.info["ground_route_speed_mps"]
    route_ratio = result.info["ground_route_speed_ratio"]
    assert route_speed.shape == (3,)
    assert route_ratio.shape == (3,)
    assert torch.isfinite(route_speed).all()
    assert torch.isfinite(route_ratio).all()
    assert torch.all(route_speed >= 0.0)
    assert torch.all(result.reward.route_speed <= 0.0)
    assert torch.all(result.reward.material_saturation <= 0.0)


def test_pc_critic_speed_error_uses_true_ground_velocity() -> None:
    config = _pc_config("C2", profile="clean", current_enabled=False)
    env = BatchedPCLowLevelEnv(config, num_envs=2, validate_finite=True)
    env.reset([23, 24])
    assert env.eta is not None and env.nu is not None
    env.eta[:, 3:6] = 0.0
    env.nu[:, 0] = 1.0
    env.current_ned_mps = torch.tensor(
        [[0.5, 0.0, 0.0], [0.0, -0.25, 0.0]],
        dtype=env.dtype,
    )

    observation = env._build_observation()

    assert observation.ground_velocity_body_mps[:, 0].tolist() == [1.5, 1.0]
    assert observation.critic_state[:, 14].tolist() == pytest.approx(
        (observation.command.speed_mps - observation.ground_velocity_body_mps[:, 0]).tolist()
    )


def test_auto_reset_uses_sensor_row_prime_without_nan() -> None:
    config = _pc_config("C2", profile="realistic", max_episode_steps=3)
    env = BatchedPCLowLevelEnv(config, num_envs=4, validate_finite=True)
    env.reset([30, 31, 32, 33])
    result = None
    for _ in range(3):
        result = env.step(torch.zeros((4, 3), dtype=torch.float64))
    assert result is not None
    assert bool(result.done.all())
    assert result.truncated.all()
    assert torch.isfinite(result.observation.actor_state).all()
    assert torch.isfinite(result.observation.sensor.p_nav).all()
    assert torch.equal(env.step_count, torch.zeros(4, dtype=torch.long))
    assert torch.equal(env.t_s, torch.zeros(4, dtype=torch.float64))


def test_c1_subset_reset_and_staggered_command_switch_keep_other_rows() -> None:
    config = _pc_config("C1", profile="mild", max_episode_steps=100)
    env = BatchedPCLowLevelEnv(config, num_envs=4, validate_finite=True)
    observation = env.reset([40, 41, 42, 43])
    initial_command = observation.command.normalized().clone()
    mask = torch.tensor([True, False, False, False])
    env.reset_indices(mask, seeds=[140])
    assert torch.isfinite(env.observation.command.normalized()).all()
    assert torch.equal(env.observation.command.normalized()[1:], initial_command[1:])

    assert env._direct_switch_time_s is not None
    env._direct_switch_time_s[0] = env.t_s[0]
    env._direct_switch_time_s[1:] = env.t_s[1:] + 1000.0
    result = env.step(torch.zeros((4, 3), dtype=torch.float64))
    assert torch.isfinite(result.observation.command.normalized()).all()


def test_c3_path_bank_selects_path_by_episode_seed_and_stays_finite() -> None:
    config = _pc_config(
        "C3",
        profile="clean",
        max_episode_steps=20,
        path_bank_size=8,
        path_bank_seed=41,
    )
    env = BatchedPCLowLevelEnv(config, num_envs=4, validate_finite=True)
    seeds = [50, 51, 52, 53]
    observation = env.reset(seeds)

    assert env._path_bank is not None
    assert env._path_index.tolist() == [seed % 8 for seed in seeds]
    assert env.path.num_paths == 4
    assert not torch.equal(env.path.start_ned_m[0], env.path.start_ned_m[1])
    assert torch.isfinite(observation.actor_state).all()
    assert torch.isfinite(observation.critic_state).all()
    assert torch.isfinite(observation.sensor.p_nav).all()

    for _ in range(10):
        result = env.step(torch.zeros((4, 3), dtype=torch.float64))
        assert torch.isfinite(result.observation.actor_state).all()
        assert torch.isfinite(result.observation.critic_state).all()
        assert torch.isfinite(result.reward.total).all()


def test_c3_path_bank_subset_reset_and_auto_reset_keep_path_selection_consistent() -> None:
    config = _pc_config(
        "C3",
        profile="realistic",
        max_episode_steps=3,
        path_bank_size=8,
        path_bank_seed=43,
    )
    env = BatchedPCLowLevelEnv(config, num_envs=4, validate_finite=True)
    env.reset([60, 61, 62, 63])

    mask = torch.tensor([True, False, True, False])
    env.reset_indices(mask, seeds=[70, 71])
    assert env._path_index.tolist() == [6, 5, 7, 7]
    assert env.path.num_paths == 4

    result = None
    for _ in range(3):
        result = env.step(torch.zeros((4, 3), dtype=torch.float64))
    assert result is not None
    assert bool(result.done.all())
    assert env._path_bank is not None
    expected = torch.remainder(env.episode_seed, env._path_bank.size)
    assert torch.equal(env._path_index, expected)
    assert torch.isfinite(result.observation.actor_state).all()
    assert torch.isfinite(result.observation.critic_state).all()
    assert torch.isfinite(result.observation.sensor.p_nav).all()


def test_v0_initial_speed_and_rpm_ranges_are_actually_sampled() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0_fixed_speed.json"
    )
    config = parse_config(json.loads(path.read_text(encoding="utf-8")))
    env = BatchedPCLowLevelEnv(
        config,
        num_envs=16,
        validate_finite=True,
        path_dim=8,
    )
    env.reset(list(range(1000, 1016)))

    speed_ratio = env._episode_initial_speed_ratio
    rpm_ratio = env._episode_initial_propeller_rpm_ratio
    assert float(speed_ratio.min()) >= 0.75 - 1e-12
    assert float(speed_ratio.max()) <= 1.25 + 1e-12
    assert float(rpm_ratio.min()) >= 0.0 - 1e-12
    assert float(rpm_ratio.max()) <= 1.0 + 1e-12
    assert float(rpm_ratio.max()) > 0.0
    assert env.actuator is not None and env.nu is not None
    assert float(env.actuator[:, 2].max()) > 0.0
    assert torch.isfinite(env.actuator).all()
    assert torch.isfinite(env.nu).all()
