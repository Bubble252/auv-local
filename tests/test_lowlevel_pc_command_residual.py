from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.controllers import BatchedControlContext, BatchedRemus100Teacher
from auv_lowlevel.batch.geometry import quaternion_from_direction
from auv_lowlevel.batch.pc_command_residual import BatchedPCCommandResidualEnv
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path_bank

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL26_CONFIG = REPO_ROOT / "configs" / "lowlevel" / "pc_goal26_env_default.json"


def _config(*, max_steps: int = 120, current_enabled: bool = True):
    config = load_config(GOAL26_CONFIG)
    return replace(
        config,
        plant=replace(config.plant, max_episode_steps=max_steps),
        current=replace(config.current, enabled=current_enabled),
    )


def _b0_action(
    controller: BatchedRemus100Teacher,
    actor_state: torch.Tensor,
    command: torch.Tensor,
    *,
    dt_s: float,
    env: BatchedPCLowLevelEnv,
) -> torch.Tensor:
    heading = command[:, 0]
    pitch = command[:, 1]
    direction = torch.stack(
        (
            torch.cos(pitch) * torch.cos(heading),
            torch.cos(pitch) * torch.sin(heading),
            -torch.sin(pitch),
        ),
        dim=1,
    )
    context = BatchedControlContext(
        eta=actor_state[:, :6],
        nu=actor_state[:, 6:12],
        quat_xyzw=quaternion_from_direction(direction),
        current_ned_mps=torch.zeros_like(actor_state[:, :3]),
        reference_quat_xyzw=quaternion_from_direction(direction),
        command_speed_mps=command[:, 2],
        observation=actor_state,
        dt_s=dt_s,
    )
    actuator = controller.control(context)
    action = torch.empty_like(actuator)
    action[:, 0] = actuator[:, 0] / float(env.limits.max_rudder_angle_rad)
    action[:, 1] = actuator[:, 1] / float(env.limits.max_stern_plane_angle_rad)
    action[:, 2] = 2.0 * actuator[:, 2] / float(env.limits.max_propeller_rpm) - 1.0
    return action


def test_zero_residual_matches_b0_stepwise() -> None:
    config = _config()
    env_c1 = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=8)
    env_b0 = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=8)
    seeds = [20, 21]
    env_c1.reset(seeds)
    env_b0.reset(seeds)
    wrapper = BatchedPCCommandResidualEnv(
        env_c1,
        scale_heading_rad=config.command_residual.scale_heading_rad,
        scale_pitch_rad=config.command_residual.scale_pitch_rad,
        scale_speed_mps=config.command_residual.scale_speed_mps,
        rate_heading_rad_s=config.command_residual.rate_heading_rad_s,
        rate_pitch_rad_s=config.command_residual.rate_pitch_rad_s,
        rate_speed_mps2=config.command_residual.rate_speed_mps2,
    )
    controller = BatchedRemus100Teacher(
        config.controller,
        env_b0.limits,
        device=env_b0.device,
        dtype=env_b0.dtype,
    )
    controller.reset(count=2)
    active = torch.ones(2, dtype=torch.bool)
    for _ in range(60):
        observation = env_b0.observation
        command = torch.stack(
            (
                observation.command.heading_rad,
                observation.command.pitch_rad,
                observation.command.speed_mps,
            ),
            dim=1,
        )
        action = _b0_action(
            controller,
            observation.actor_state,
            command,
            dt_s=env_b0.dt_s,
            env=env_b0,
        )
        result_c1 = wrapper.step(
            torch.zeros((2, 3), dtype=torch.float64),
            auto_reset=False,
            advance_mask=active,
        )
        result_b0 = env_b0.step(action, auto_reset=False, advance_mask=active)
        assert env_c1.eta is not None and env_b0.eta is not None
        assert env_c1.nu is not None and env_b0.nu is not None
        assert env_c1.actuator is not None and env_b0.actuator is not None
        assert torch.equal(env_c1.eta, env_b0.eta)
        assert torch.equal(env_c1.nu, env_b0.nu)
        assert torch.equal(env_c1.actuator, env_b0.actuator)
        assert torch.equal(result_c1.reward.total, result_b0.reward.total)
        assert torch.equal(result_c1.terminated, result_b0.terminated)
        assert torch.equal(result_c1.truncated, result_b0.truncated)
        assert torch.equal(
            result_c1.info["command_residual_applied"],
            torch.zeros_like(result_c1.info["command_residual_applied"]),
        )
        active = active & ~(result_c1.terminated | result_c1.truncated)
        if not bool(active.any()):
            break


def test_residual_is_clamped_to_scale() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([30])
    wrapper = BatchedPCCommandResidualEnv(
        env,
        scale_heading_rad=0.2,
        scale_pitch_rad=0.1,
        scale_speed_mps=0.4,
        rate_heading_rad_s=1e6,
        rate_pitch_rad_s=1e6,
        rate_speed_mps2=1e6,
    )
    result = wrapper.step(torch.tensor([[100.0, -100.0, 100.0]], dtype=torch.float64))
    delta = result.info["command_residual_applied"]
    torch.testing.assert_close(delta, torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64))
    assert bool(result.info["command_residual_scale_clipped"].all())


def test_residual_rate_limit_applies() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([31])
    wrapper = BatchedPCCommandResidualEnv(env)
    result = wrapper.step(torch.ones((1, 3), dtype=torch.float64))
    delta = result.info["command_residual_applied"]
    expected = wrapper.rate_limit * env.dt_s
    torch.testing.assert_close(delta[0], expected)
    assert bool(result.info["command_residual_rate_clipped"].all())
    assert not bool(result.info["command_residual_scale_clipped"].any())


def test_speed_residual_does_not_drop_below_task_floor() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    observation = env.reset([32])
    wrapper = BatchedPCCommandResidualEnv(
        env,
        rate_heading_rad_s=1e6,
        rate_pitch_rad_s=1e6,
        rate_speed_mps2=1e6,
    )
    base_speed = float(observation.command.speed_mps.item())
    result = wrapper.step(torch.tensor([[-100.0, 0.0, -100.0]], dtype=torch.float64))
    corrected = result.info["command_corrected_physical"]
    floor = max(
        float(env.vehicle.min_speed_mps),
        float(config.guidance.minimum_speed_ratio) * base_speed,
    )
    assert float(corrected[0, 2]) >= floor - 1e-12


def test_nonfinite_residual_is_rejected() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([33])
    wrapper = BatchedPCCommandResidualEnv(env)
    with pytest.raises(ValueError, match="NaN or Inf"):
        wrapper.step(torch.tensor([[float("nan"), 0.0, 0.0]], dtype=torch.float64))


def test_wrapper_proxies_core_environment_attributes() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=8)
    env.reset([34, 35])
    wrapper = BatchedPCCommandResidualEnv(env)
    assert wrapper.num_envs == env.num_envs
    assert wrapper.action_dim == env.action_dim == 3
    assert wrapper.observation is env.observation
    assert wrapper.config is env.config
    assert wrapper.actor_path_dim == 8


def test_wrapper_path_tracks_seed_selection_after_reset_and_reset_indices() -> None:
    config = _config()
    config = replace(
        config,
        reference=replace(config.reference, path_bank_size=8, path_bank_seed=41),
    )
    path_bank = generate_sinusoidal_3d_path_bank(config.reference)
    env = BatchedPCLowLevelEnv(config, num_envs=4, path_dim=8)
    wrapper = BatchedPCCommandResidualEnv(env)

    wrapper.reset([20, 21, 22, 23])
    assert wrapper.path is env.path
    assert env._path_index.tolist() == [4, 5, 6, 7]
    goals = wrapper.path.goal_ned_m_for(4).detach().cpu().numpy()
    for row, seed in enumerate([20, 21, 22, 23]):
        assert goals[row] == pytest.approx(path_bank[seed % len(path_bank)].goal_ned_m)

    mask = torch.tensor([True, False, True, False])
    wrapper.reset_indices(mask, seeds=[52, 53])
    assert wrapper.path is env.path
    assert env._path_index.tolist() == [4, 5, 5, 7]
    assert wrapper.path.num_paths == 4
    goals = wrapper.path.goal_ned_m_for(4).detach().cpu().numpy()
    for row, seed in enumerate([52, 21, 53, 23]):
        assert goals[row] == pytest.approx(path_bank[seed % len(path_bank)].goal_ned_m)
