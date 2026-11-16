from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from auv_lowlevel.batch.pc_command_residual import BatchedPCCommandResidualEnv
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.cli.evaluate_pc_ppo.runner import _wrap_alos_command_residual
from auv_lowlevel.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL26_CONFIG = REPO_ROOT / "configs" / "lowlevel" / "pc_goal26_env_default.json"


def _config():
    config = load_config(GOAL26_CONFIG)
    return replace(config, plant=replace(config.plant, max_episode_steps=8))


def test_alos_absolute_checkpoint_gets_command_residual_wrapper() -> None:
    inner = BatchedPCLowLevelEnv(_config(), num_envs=1, path_dim=0)
    wrapped = _wrap_alos_command_residual(
        inner,
        action_mode="alos",
        resolved_action_interface="absolute",
    )

    assert isinstance(wrapped, BatchedPCCommandResidualEnv)
    assert wrapped.env is inner


def test_alos_zero_residual_runs_through_alos_and_pd_pi() -> None:
    inner = BatchedPCLowLevelEnv(_config(), num_envs=1, path_dim=0)
    wrapped = _wrap_alos_command_residual(
        inner,
        action_mode="alos",
        resolved_action_interface="absolute",
    )
    assert isinstance(wrapped, BatchedPCCommandResidualEnv)
    wrapped.reset([20])

    result = wrapped.step(
        torch.zeros((1, 3), dtype=torch.float64),
        auto_reset=False,
    )

    assert torch.count_nonzero(result.info["command_residual_applied"]) == 0
    assert torch.isfinite(result.info["command_residual_actuator"]).all()
    assert inner.actuator is not None
    assert float(inner.actuator[0, 2]) > 0.0
    assert torch.count_nonzero(inner.actuator[0, :2]) > 0


def test_alos_does_not_double_wrap_existing_interfaces() -> None:
    inner = BatchedPCLowLevelEnv(_config(), num_envs=1, path_dim=0)

    assert (
        _wrap_alos_command_residual(
            inner,
            action_mode="alos",
            resolved_action_interface="command",
        )
        is inner
    )
    assert (
        _wrap_alos_command_residual(
            inner,
            action_mode="alos",
            resolved_action_interface="actuator",
        )
        is inner
    )
    assert (
        _wrap_alos_command_residual(
            inner,
            action_mode="policy",
            resolved_action_interface="absolute",
        )
        is inner
    )
