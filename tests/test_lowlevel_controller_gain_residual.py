from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.controllers import BatchedControlContext, BatchedRemus100Teacher
from auv_lowlevel.batch.geometry import quaternion_from_direction
from auv_lowlevel.batch.pc_controller_gain_residual import (
    BatchedPCControllerGainResidualEnv,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
H3_CONFIG = REPO_ROOT / "configs" / "lowlevel" / "goal160" / "env_c_command_terminal_train.json"


def _config(*, max_steps: int = 120):
    config = load_config(H3_CONFIG)
    return replace(
        config,
        plant=replace(config.plant, max_episode_steps=max_steps),
    )


def _h3_action(
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
    reference_quat = quaternion_from_direction(direction)
    context = BatchedControlContext(
        eta=actor_state[:, :6],
        nu=actor_state[:, 6:12],
        quat_xyzw=reference_quat,
        current_ned_mps=torch.zeros_like(actor_state[:, :3]),
        reference_quat_xyzw=reference_quat,
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


def test_zero_gain_residual_matches_h3_stepwise() -> None:
    config = _config()
    env_gain = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=12)
    env_h3 = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=12)
    seeds = [20, 21]
    env_gain.reset(seeds)
    env_h3.reset(seeds)
    wrapper = BatchedPCControllerGainResidualEnv(
        env_gain,
        heading_gain_delta=1.0,
        pitch_gain_delta=1.5,
        terminal_gate_enabled=False,
    )
    controller = BatchedRemus100Teacher(
        config.controller,
        env_h3.limits,
        device=env_h3.device,
        dtype=env_h3.dtype,
    )
    controller.reset(count=2)
    active = torch.ones(2, dtype=torch.bool)
    for _ in range(60):
        assert env_h3.observation is not None
        observation = env_h3.observation
        command = torch.stack(
            (
                observation.command.heading_rad,
                observation.command.pitch_rad,
                observation.command.speed_mps,
            ),
            dim=1,
        )
        action = _h3_action(
            controller,
            observation.actor_state,
            command,
            dt_s=env_h3.dt_s,
            env=env_h3,
        )
        result_gain = wrapper.step(
            torch.zeros((2, 2), dtype=torch.float64),
            auto_reset=False,
            advance_mask=active,
        )
        result_h3 = env_h3.step(action, auto_reset=False, advance_mask=active)
        assert env_gain.eta is not None and env_h3.eta is not None
        assert env_gain.nu is not None and env_h3.nu is not None
        assert env_gain.actuator is not None and env_h3.actuator is not None
        assert torch.equal(env_gain.eta, env_h3.eta)
        assert torch.equal(env_gain.nu, env_h3.nu)
        assert torch.equal(env_gain.actuator, env_h3.actuator)
        assert torch.equal(result_gain.reward.total, result_h3.reward.total)
        assert torch.equal(
            result_gain.info["controller_gain_residual_applied"],
            torch.zeros_like(result_gain.info["controller_gain_residual_applied"]),
        )
        active = active & ~(result_gain.terminated | result_gain.truncated)
        if not bool(active.any()):
            break


def test_gain_residual_is_tanh_bounded() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=12)
    env.reset([30])
    wrapper = BatchedPCControllerGainResidualEnv(
        env,
        heading_gain_delta=1.0,
        pitch_gain_delta=1.5,
        terminal_gate_enabled=False,
    )
    result = wrapper.step(torch.tensor([[100.0, -100.0]], dtype=torch.float64))
    effective = result.info["controller_gain_effective"]
    torch.testing.assert_close(
        effective,
        torch.tensor([[4.0, 2.5]], dtype=torch.float64),
    )
    delta = result.info["controller_gain_residual_applied"]
    torch.testing.assert_close(
        delta,
        torch.tensor([[1.0, -1.5]], dtype=torch.float64),
    )


def test_gain_residual_honors_terminal_gate() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=12)
    env.reset([31, 32])
    wrapper = BatchedPCControllerGainResidualEnv(
        env,
        heading_gain_delta=1.0,
        pitch_gain_delta=1.5,
        terminal_gate_enabled=True,
    )
    wrapper._terminal_gate = lambda: torch.tensor([0.0, 1.0], dtype=torch.float64)
    result = wrapper.step(torch.ones((2, 2), dtype=torch.float64))
    delta = result.info["controller_gain_residual_applied"]
    torch.testing.assert_close(delta[0], torch.zeros(2, dtype=torch.float64))
    bounded = float(torch.tanh(torch.tensor(1.0)))
    torch.testing.assert_close(
        delta[1],
        torch.tensor([bounded, 1.5 * bounded], dtype=torch.float64),
    )


def test_nonfinite_gain_residual_is_rejected() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=12)
    env.reset([33])
    wrapper = BatchedPCControllerGainResidualEnv(env)
    with pytest.raises(ValueError, match="NaN or Inf"):
        wrapper.step(torch.tensor([[float("nan"), 0.0]], dtype=torch.float64))


def test_offpolicy_adapter_exposes_two_dimensional_gain_interface() -> None:
    config = _config(max_steps=8)
    env = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=12,
        action_interface="controller_gain",
        curriculum_direct_commands_enabled=False,
    )
    observations, _ = env.reset([40, 41])
    assert env.action_dim == 2
    assert observations.shape == (2, int(env.observation_dim))
    _, _, _, _, info = env.step(
        torch.ones((2, 2), dtype=torch.float64).numpy(), auto_reset=False
    )
    assert env.last_controller_gain_effective.shape == (2, 2)
    assert env.last_controller_gain_base.shape == (2, 2)
    assert info["controller_gain_residual_gate"].shape == (2,)


def test_context_gain_override_changes_controller_without_mutating_config() -> None:
    config = load_config(REPO_ROOT / "configs" / "lowlevel" / "odarrl_control_base.json")
    base_teacher = BatchedRemus100Teacher(config.controller, _limits(config))
    gain_teacher = BatchedRemus100Teacher(config.controller, _limits(config))
    base_teacher.reset(count=2)
    gain_teacher.reset(count=2)
    context = _context(config, count=2)
    base_action = base_teacher.control(context)
    gain_action = gain_teacher.control(
        replace(
            context,
            heading_gain=torch.full((2,), 4.0, dtype=torch.float64),
            pitch_gain=torch.full((2,), 5.5, dtype=torch.float64),
        )
    )
    assert not torch.allclose(base_action, gain_action)
    assert config.controller.heading_gain == pytest.approx(2.0)
    assert config.controller.pitch_gain == pytest.approx(2.0)


def _limits(config):
    class Limits:
        max_rudder_angle_rad = 0.3490658504
        max_stern_plane_angle_rad = 0.3490658504
        max_rudder_rate_rad_s = 2.617993878
        max_stern_plane_rate_rad_s = 2.617993878
        max_propeller_rpm = 1525.0

    return Limits()


def _context(config, *, count: int) -> BatchedControlContext:
    eta = torch.tensor(
        [[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]] * count,
        dtype=torch.float64,
    )
    nu = torch.tensor(
        [[1.2, 0.0, 0.0, 0.0, 0.02, -0.03]] * count,
        dtype=torch.float64,
    )
    reference = quaternion_from_direction(
        torch.tensor([[1.0, 0.002, -0.002]] * count, dtype=torch.float64)
    )
    return BatchedControlContext(
        eta=eta,
        nu=nu,
        quat_xyzw=reference,
        current_ned_mps=torch.zeros((count, 3), dtype=torch.float64),
        reference_quat_xyzw=reference,
        command_speed_mps=torch.full((count,), 1.2, dtype=torch.float64),
        observation=torch.zeros((count, 26), dtype=torch.float64),
        dt_s=float(config.controller.nominal_step_s),
    )
