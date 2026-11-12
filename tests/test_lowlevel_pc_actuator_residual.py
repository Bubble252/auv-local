from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.controllers import BatchedControlContext, BatchedRemus100Teacher
from auv_lowlevel.batch.geometry import quaternion_from_direction
from auv_lowlevel.batch.pc_actuator_residual import BatchedPCActuatorResidualEnv
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL26_CONFIG = REPO_ROOT / "configs" / "lowlevel" / "pc_goal26_env_default.json"
GOAL85_CONFIG = (
    REPO_ROOT
    / "configs"
    / "lowlevel"
    / "pc_goal85_unbiased_default_candidate_20261110.json"
)


def _config(*, max_steps: int = 120):
    config = load_config(GOAL26_CONFIG)
    return replace(config, plant=replace(config.plant, max_episode_steps=max_steps))


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
    reference = quaternion_from_direction(direction)
    context = BatchedControlContext(
        eta=actor_state[:, :6],
        nu=actor_state[:, 6:12],
        quat_xyzw=reference,
        current_ned_mps=torch.zeros_like(actor_state[:, :3]),
        reference_quat_xyzw=reference,
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


@pytest.mark.parametrize(
    ("config_path", "path_dim"), [(GOAL26_CONFIG, 8), (GOAL85_CONFIG, 12)]
)
def test_zero_actuator_residual_matches_alos_pd_pi_stepwise(
    config_path: Path, path_dim: int
) -> None:
    config = load_config(config_path)
    config = replace(config, plant=replace(config.plant, max_episode_steps=120))
    env_residual = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=path_dim)
    env_b0 = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=path_dim)
    seeds = [20, 21]
    env_residual.reset(seeds)
    env_b0.reset(seeds)
    wrapper = BatchedPCActuatorResidualEnv(env_residual, residual_scale=0.25)
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
        baseline_action = _b0_action(
            controller,
            observation.actor_state,
            command,
            dt_s=env_b0.dt_s,
            env=env_b0,
        )
        result_residual = wrapper.step(
            torch.zeros((2, 3), dtype=torch.float64),
            auto_reset=False,
            advance_mask=active,
        )
        result_b0 = env_b0.step(baseline_action, auto_reset=False, advance_mask=active)
        assert env_residual.eta is not None and env_b0.eta is not None
        assert env_residual.nu is not None and env_b0.nu is not None
        assert env_residual.actuator is not None and env_b0.actuator is not None
        torch.testing.assert_close(env_residual.eta, env_b0.eta, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(env_residual.nu, env_b0.nu, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(
            env_residual.actuator, env_b0.actuator, atol=1e-9, rtol=0.0
        )
        torch.testing.assert_close(
            result_residual.info["actuator_residual_applied"],
            torch.zeros_like(result_residual.info["actuator_residual_applied"]),
            atol=1e-12,
            rtol=0.0,
        )
        assert torch.equal(
            result_residual.info["material_saturated"],
            result_b0.info["material_saturated"],
        )
        torch.testing.assert_close(
            result_residual.info["guidance_saturation_ema"],
            result_b0.info["guidance_saturation_ema"],
            atol=1e-12,
            rtol=0.0,
        )
        active = active & ~(result_residual.terminated | result_residual.truncated)
        if not bool(active.any()):
            break


def test_actuator_residual_is_scaled_and_clipped() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([30])
    wrapper = BatchedPCActuatorResidualEnv(env, residual_scale=0.25, max_action=1.0)
    result = wrapper.step(torch.tensor([[100.0, -100.0, 100.0]], dtype=torch.float64))
    expected = wrapper._full_scale * wrapper.residual_scale * torch.tensor(
        [1.0, -1.0, 1.0], dtype=torch.float64
    )
    torch.testing.assert_close(
        result.info["actuator_residual_applied"], expected.unsqueeze(0)
    )
    assert bool(result.info["actuator_residual_scale_clipped"].all())
    assert torch.all(
        result.info["actuator_residual_applied"].abs()
        <= wrapper._full_scale * wrapper.residual_scale + 1e-12
    )


def test_actuator_terminal_gate_zeroes_far_residual_and_opens_near_goal() -> None:
    config = _config()
    env_far = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env_far.reset([30])
    far = BatchedPCActuatorResidualEnv(
        env_far,
        residual_scale=0.25,
        terminal_gate_enabled=True,
        terminal_gate_enter_m=1e-3,
        terminal_gate_full_m=1e-4,
    )
    assert torch.all(far._terminal_gate() == 0.0)
    far_result = far.step(torch.full((1, 3), 0.5, dtype=torch.float64))
    torch.testing.assert_close(
        far_result.info["actuator_residual_applied"],
        torch.zeros_like(far_result.info["actuator_residual_applied"]),
    )
    torch.testing.assert_close(
        far_result.info["actuator_residual_gate"],
        torch.zeros_like(far_result.info["actuator_residual_gate"]),
    )

    env_near = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env_near.reset([30])
    near = BatchedPCActuatorResidualEnv(
        env_near,
        residual_scale=0.25,
        terminal_gate_enabled=True,
        terminal_gate_enter_m=1e9,
        terminal_gate_full_m=5e8,
    )
    assert torch.all(near._terminal_gate() == 1.0)
    near_result = near.step(torch.full((1, 3), 0.5, dtype=torch.float64))
    expected = (
        0.5
        * near.residual_scale
        * near._full_scale
    ).unsqueeze(0)
    torch.testing.assert_close(
        near_result.info["actuator_residual_applied"], expected
    )


def test_actuator_terminal_gate_rejects_inverted_window() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([30])
    with pytest.raises(ValueError, match="less than enter_m"):
        BatchedPCActuatorResidualEnv(
            env,
            terminal_gate_enabled=True,
            terminal_gate_enter_m=5.0,
            terminal_gate_full_m=5.0,
        )


def test_actuator_residual_distance_schedule_uses_near_and_far_scales() -> None:
    config = _config()

    env_near = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env_near.reset([30])
    near = BatchedPCActuatorResidualEnv(
        env_near,
        residual_scale=0.25,
        residual_scale_near=0.75,
        residual_scale_far=1.50,
        residual_scale_switch_distance_m=1e9,
    )
    near_scale = near._residual_scale_vector()
    near_scale_again, near_distance = near._residual_scale_state()
    torch.testing.assert_close(
        near_scale, torch.tensor([0.75], dtype=torch.float64)
    )
    torch.testing.assert_close(near_scale, near_scale_again)
    assert near_distance.shape == (1,)
    near_result = near.step(torch.tensor([[0.5, -0.5, 0.25]], dtype=torch.float64))
    torch.testing.assert_close(
        near_result.info["actuator_residual_scale"], near_scale
    )
    torch.testing.assert_close(
        near_result.info["actuator_residual_schedule_distance_m"], near_distance
    )
    torch.testing.assert_close(
        near_result.info["actuator_residual_applied"],
        torch.tensor(
            [[0.5, -0.5, 0.25]], dtype=torch.float64
        )
        * 0.75
        * near._full_scale,
    )

    env_far = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env_far.reset([30])
    far = BatchedPCActuatorResidualEnv(
        env_far,
        residual_scale=0.25,
        residual_scale_near=0.75,
        residual_scale_far=1.50,
        residual_scale_switch_distance_m=1e-6,
    )
    far_scale = far._residual_scale_vector()
    far_scale_again, far_distance = far._residual_scale_state()
    torch.testing.assert_close(
        far_scale, torch.tensor([1.50], dtype=torch.float64)
    )
    torch.testing.assert_close(far_scale, far_scale_again)
    assert far_distance.shape == (1,)
    far_result = far.step(torch.tensor([[0.5, -0.5, 0.25]], dtype=torch.float64))
    torch.testing.assert_close(far_result.info["actuator_residual_scale"], far_scale)
    torch.testing.assert_close(
        far_result.info["actuator_residual_schedule_distance_m"], far_distance
    )
    torch.testing.assert_close(
        far_result.info["actuator_residual_applied"],
        torch.tensor(
            [[0.5, -0.5, 0.25]], dtype=torch.float64
        )
        * 1.50
        * far._full_scale,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residual_scale_near": 1.5},
        {"residual_scale_far": 2.0},
        {"residual_scale_switch_distance_m": 2.0},
        {
            "residual_scale_near": 1.5,
            "residual_scale_far": 2.0,
            "residual_scale_switch_distance_m": 0.0,
        },
        {
            "residual_scale_near": -1.5,
            "residual_scale_far": 2.0,
            "residual_scale_switch_distance_m": 2.0,
        },
    ],
)
def test_actuator_residual_distance_schedule_rejects_invalid_config(
    kwargs: dict[str, float],
) -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([30])
    with pytest.raises(ValueError, match="residual scale|residual_scale"):
        BatchedPCActuatorResidualEnv(env, **kwargs)


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
    ],
)
def test_nonfinite_actuator_residual_is_rejected(bad: float) -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([33])
    wrapper = BatchedPCActuatorResidualEnv(env)
    with pytest.raises(ValueError, match="NaN or Inf"):
        wrapper.step(torch.tensor([[bad, 0.0, 0.0]], dtype=torch.float64))


def test_actuator_wrapper_proxies_environment_and_resets_controller() -> None:
    config = _config()
    config = replace(
        config,
        reference=replace(config.reference, path_bank_size=8, path_bank_seed=41),
    )
    env = BatchedPCLowLevelEnv(config, num_envs=4, path_dim=8)
    wrapper = BatchedPCActuatorResidualEnv(env)
    wrapper.reset([20, 21, 22, 23])
    assert wrapper.num_envs == env.num_envs == 4
    assert wrapper.action_dim == 3
    assert wrapper.observation is env.observation
    assert wrapper.path is env.path
    assert wrapper.actor_path_dim == 8
    wrapper.reset_indices(torch.tensor([True, False, True, False]), seeds=[52, 53])
    assert wrapper.path is env.path
    assert env._path_index.tolist() == [4, 5, 5, 7]


def test_actuator_residual_channel_gains_scale_each_channel_independently() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([30])
    gains = (0.5, 1.0, 2.0)
    wrapper = BatchedPCActuatorResidualEnv(
        env,
        residual_scale=0.25,
        residual_channel_gains=gains,
    )
    result = wrapper.step(torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float64))
    expected = (
        torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float64)
        * 0.25
        * wrapper._full_scale
        * torch.tensor(gains, dtype=torch.float64)
    )
    torch.testing.assert_close(result.info["actuator_residual_applied"], expected)
    torch.testing.assert_close(
        result.info["actuator_residual_channel_gains"],
        torch.tensor([gains], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "gains",
    [
        (1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, float("nan"), 1.0),
        (1.0, float("inf"), 1.0),
    ],
)
def test_actuator_residual_channel_gains_reject_invalid_values(
    gains: tuple[float, ...],
) -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([30])
    with pytest.raises(ValueError, match="residual_channel_gains"):
        BatchedPCActuatorResidualEnv(env, residual_channel_gains=gains)


def test_actuator_residual_channel_gain_schedule_interpolates_per_channel() -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([30])
    wrapper = BatchedPCActuatorResidualEnv(
        env,
        residual_scale=0.25,
        residual_channel_gains=(1.0, 1.0, 1.0),
        residual_channel_gains_near=(1.0, 1.0, 0.5),
        residual_channel_gain_near_distance_m=5.0,
        residual_channel_gain_far_distance_m=10.0,
    )

    gains, near_gate = wrapper._residual_channel_gain_state(
        torch.tensor([4.0, 7.5, 11.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        near_gate, torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        gains,
        torch.tensor(
            [
                [1.0, 1.0, 0.5],
                [1.0, 1.0, 0.75],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        ),
    )


def test_actuator_residual_channel_gain_schedule_scales_only_near_rpm() -> None:
    config = _config()
    env_near = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env_near.reset([30])
    near = BatchedPCActuatorResidualEnv(
        env_near,
        residual_scale=0.25,
        residual_channel_gains_near=(1.0, 1.0, 0.5),
        residual_channel_gain_near_distance_m=1e9,
        residual_channel_gain_far_distance_m=2e9,
    )
    near_result = near.step(
        torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float64)
    )
    expected = (
        torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float64)
        * 0.25
        * near._full_scale
        * torch.tensor([[1.0, 1.0, 0.5]], dtype=torch.float64)
    )
    torch.testing.assert_close(near_result.info["actuator_residual_applied"], expected)
    torch.testing.assert_close(
        near_result.info["actuator_residual_channel_gain_near_gate"],
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        near_result.info["actuator_residual_channel_gains"],
        torch.tensor([[1.0, 1.0, 0.5]], dtype=torch.float64),
    )

    env_far = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env_far.reset([30])
    far = BatchedPCActuatorResidualEnv(
        env_far,
        residual_scale=0.25,
        residual_channel_gains_near=(1.0, 1.0, 0.5),
        residual_channel_gain_near_distance_m=1e-6,
        residual_channel_gain_far_distance_m=2e-6,
    )
    far_result = far.step(
        torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        far_result.info["actuator_residual_channel_gain_near_gate"],
        torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        far_result.info["actuator_residual_channel_gains"],
        torch.ones((1, 3), dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residual_channel_gains_near": (1.0, 1.0, 0.5)},
        {
            "residual_channel_gains_near": (1.0, 1.0, 0.5),
            "residual_channel_gain_near_distance_m": 5.0,
        },
        {
            "residual_channel_gains_near": (1.0, 1.0, 0.5),
            "residual_channel_gain_near_distance_m": 5.0,
            "residual_channel_gain_far_distance_m": 5.0,
        },
        {
            "residual_channel_gains_near": (1.0, 1.0, 0.0),
            "residual_channel_gain_near_distance_m": 5.0,
            "residual_channel_gain_far_distance_m": 10.0,
        },
    ],
)
def test_actuator_residual_channel_gain_schedule_rejects_invalid_config(
    kwargs: dict[str, object],
) -> None:
    config = _config()
    env = BatchedPCLowLevelEnv(config, num_envs=1, path_dim=0)
    env.reset([30])
    with pytest.raises(ValueError, match="residual channel gain|residual_channel_gains_near"):
        BatchedPCActuatorResidualEnv(env, **kwargs)
