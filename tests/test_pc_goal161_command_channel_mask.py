from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.config import load_config
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = REPO_ROOT / "configs/lowlevel/pc_goal27_r1_reward_v2_default_20261109.json"


def _config():
    config = load_config(ENV_CONFIG)
    return replace(
        config,
        plant=replace(config.plant, max_episode_steps=8),
        reference=replace(config.reference, path_bank_size=8, path_bank_seed=41),
        evaluation=replace(config.evaluation, training_seed_start=240),
    )


def _command_env(mask: tuple[float, float, float] | None) -> PCCommandResidualOffPolicyEnv:
    env = PCCommandResidualOffPolicyEnv(
        _config(),
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
        action_interface="command",
        learner_command_channel_mask=mask,
    )
    env.reset([20, 21])
    return env


def test_command_channel_mask_disables_only_selected_channels() -> None:
    env = _command_env((1.0, 0.0, 1.0))
    action = np.full((2, 3), 0.4, dtype=np.float64)
    env.step(action, auto_reset=False)
    assert np.allclose(env.last_learner_gated_action, [[0.4, 0.0, 0.4], [0.4, 0.0, 0.4]])


def test_command_channel_mask_keeps_default_action_when_none() -> None:
    env = _command_env(None)
    action = np.array([[0.4, -0.2, 0.1], [-0.3, 0.2, -0.1]], dtype=np.float64)
    env.step(action, auto_reset=False)
    assert np.allclose(env.last_learner_gated_action, action)


def test_command_channel_mask_rejects_invalid_values_and_actuator_interface() -> None:
    with pytest.raises(ValueError, match="three values"):
        _command_env((1.0, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _command_env((1.0, 1.5, 0.0))
    with pytest.raises(ValueError, match="only supported for command"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            learner_command_channel_mask=(1.0, 1.0, 1.0),
        )
