from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.controllers import BatchedBCPolicy
from auv_lowlevel.batch.env import BatchedLowLevelEnv
from auv_lowlevel.batch.observation import OBSERVATION_SCHEMA_V2
from auv_lowlevel.batch.reward import TerminalKind
from auv_lowlevel.bc import BCPolicy
from auv_lowlevel.config import load_config
from auv_lowlevel.env import LowLevelEnv


def test_batch_reset_matches_scalar_for_multiple_seeds() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    seeds = [3, 5, 7]
    batch = BatchedLowLevelEnv(config, num_envs=len(seeds))
    actual = batch.reset(seeds).numpy()
    expected = np.stack([LowLevelEnv(config).reset(seed) for seed in seeds])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_batch_closed_loop_matches_scalar_for_500_steps() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    scalar = LowLevelEnv(config)
    batched = BatchedLowLevelEnv(config, num_envs=1)
    np.testing.assert_allclose(
        batched.reset([11]).numpy()[0],
        scalar.reset(11),
        atol=1e-12,
        rtol=1e-12,
    )
    for step in range(500):
        expected = scalar.step()
        actual = batched.step(auto_reset=False)
        np.testing.assert_allclose(
            actual.observation.numpy()[0],
            expected.observation,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"observation diverged at step {step}",
        )
        np.testing.assert_allclose(
            actual.reward.total.numpy()[0],
            expected.reward.total,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"reward diverged at step {step}",
        )
        np.testing.assert_allclose(
            actual.info["limited_total"].numpy()[0],
            expected.info["limited_total"],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"command diverged at step {step}",
        )
        np.testing.assert_allclose(
            actual.info["actual_actuator"].numpy()[0],
            expected.info["actual_actuator"],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"actuator diverged at step {step}",
        )
        assert bool(actual.terminated[0]) == expected.terminated
        assert bool(actual.truncated[0]) == expected.truncated
        if expected.terminated or expected.truncated:
            break


def test_batch_closed_loop_matches_scalar_for_full_episode() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    scalar = LowLevelEnv(config)
    batched = BatchedLowLevelEnv(config, num_envs=1)
    np.testing.assert_allclose(
        batched.reset([17]).numpy()[0],
        scalar.reset(17),
        atol=1e-12,
        rtol=1e-12,
    )
    steps = 0
    while True:
        expected = scalar.step()
        actual = batched.step(auto_reset=False)
        steps += 1
        np.testing.assert_allclose(
            actual.observation.numpy()[0],
            expected.observation,
            atol=1e-9,
            rtol=1e-9,
            err_msg=f"observation diverged at step {steps}",
        )
        np.testing.assert_allclose(
            actual.reward.total.numpy()[0],
            expected.reward.total,
            atol=1e-9,
            rtol=1e-9,
            err_msg=f"reward diverged at step {steps}",
        )
        assert bool(actual.terminated[0]) == expected.terminated
        assert bool(actual.truncated[0]) == expected.truncated
        if expected.terminated or expected.truncated:
            break
        if steps > config.plant.max_episode_steps:
            raise AssertionError("episode exceeded max_episode_steps")
    assert steps == scalar.step_count
    assert bool(actual.terminated[0])
    assert int(actual.info["terminal_kind"][0]) == int(TerminalKind.GOAL)


def test_partial_reset_only_changes_selected_environment() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    batch = BatchedLowLevelEnv(config, num_envs=3)
    initial = batch.reset([21, 22, 23]).numpy()
    for _ in range(4):
        batch.step(auto_reset=False)
    before = batch.observation.numpy()
    mask = torch.tensor([False, True, False])
    batch.reset_indices(mask, [31])
    after = batch.observation.numpy()
    expected = LowLevelEnv(config).reset(31)
    np.testing.assert_allclose(after[1], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(after[0], before[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(after[2], before[2], atol=0.0, rtol=0.0)
    assert np.any(initial[1] != expected)


def test_auto_reset_returns_fresh_observation_and_keeps_terminal_reward() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    batch = BatchedLowLevelEnv(config, num_envs=2)
    batch.reset([41, 42])
    assert batch.eta is not None
    batch.eta[0, :3] = batch.path.goal_ned_m
    result = batch.step(
        torch.zeros((2, 3), dtype=torch.float64),
        auto_reset=True,
    )
    assert bool(result.terminated[0])
    assert int(result.info["terminal_kind"][0]) == int(TerminalKind.GOAL)
    assert "final_observation" in result.info
    assert not torch.equal(result.observation[0], result.info["final_observation"][0])
    assert result.reward.terminal[0].item() == pytest.approx(50.0)


def test_frozen_bc_batch_closed_loop_matches_scalar_policy() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    checkpoint = "artifacts/lowlevel/i_bc_smoke_02/bc/pi0.pt"
    scalar_policy = BCPolicy(checkpoint, device="cpu")
    scalar = LowLevelEnv(
        config,
        base_mode="frozen",
        base_command_fn=scalar_policy,
    )
    batched = BatchedLowLevelEnv(
        config,
        num_envs=1,
        base_controller=BatchedBCPolicy(checkpoint, device="cpu"),
    )
    np.testing.assert_allclose(
        batched.reset([13]).numpy()[0],
        scalar.reset(13),
        atol=1e-6,
        rtol=1e-6,
    )
    for step in range(200):
        expected = scalar.step()
        actual = batched.step(auto_reset=False)
        np.testing.assert_allclose(
            actual.observation.numpy()[0],
            expected.observation,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"BC observation diverged at step {step}",
        )
        np.testing.assert_allclose(
            actual.info["limited_total"].numpy()[0],
            expected.info["limited_total"],
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"BC command diverged at step {step}",
        )
        assert bool(actual.terminated[0]) == expected.terminated
        assert bool(actual.truncated[0]) == expected.truncated
        if expected.terminated or expected.truncated:
            break


def test_v2_environment_is_exact_26d_prefix_of_v1() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    seeds = [51, 52]
    v1 = BatchedLowLevelEnv(config, num_envs=2).reset(seeds)
    v2 = BatchedLowLevelEnv(
        config,
        num_envs=2,
        schema=OBSERVATION_SCHEMA_V2,
    ).reset(seeds)
    assert v1.shape == (2, 27)
    assert v2.shape == (2, 26)
    assert torch.equal(v2, v1[:, :26])


def test_scalar_v2_observation_is_exact_26d_prefix_of_v1() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    v2_config = replace(
        config,
        observation=replace(config.observation, dim=26),
    )

    v1 = LowLevelEnv(config).reset(51)
    v2 = LowLevelEnv(v2_config).reset(51)

    assert v1.shape == (27,)
    assert v2.shape == (26,)
    np.testing.assert_array_equal(v2, v1[:26])


def test_v2_auto_reset_clears_residual_history_without_integral_column() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    batch = BatchedLowLevelEnv(
        config,
        num_envs=2,
        schema=OBSERVATION_SCHEMA_V2,
    )
    batch.reset([61, 62])
    assert batch.eta is not None
    assert batch.previous_residual is not None
    batch.previous_residual.fill_(0.5)
    batch.eta[0, :3] = batch.path.goal_ned_m

    result = batch.step(torch.zeros((2, 3), dtype=torch.float64), auto_reset=True)

    assert result.observation.shape == (2, 26)
    assert torch.equal(result.observation[0, 23:26], torch.zeros(3, dtype=torch.float64))
