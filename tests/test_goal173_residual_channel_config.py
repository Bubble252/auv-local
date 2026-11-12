from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import parse_training_config
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs"
    / "lowlevel"
    / "goal171"
    / "train_r1_bc_ppo_seed86201.json"
)
ENV_CONFIG = (
    REPO_ROOT
    / "configs"
    / "lowlevel"
    / "pc_goal27_r1_reward_v2_default_20261109.json"
)


def _payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_goal173_channel_gains_default_preserves_old_config() -> None:
    training = parse_training_config(_payload())
    assert training.pc is not None
    assert training.pc.actuator_residual_channel_gains == (1.0, 1.0, 1.0)
    assert training.pc.as_dict()["actuator_residual_channel_gains"] == [1.0, 1.0, 1.0]


def test_goal173_rudder_only_channel_gains_parse_and_roundtrip() -> None:
    payload = _payload()
    payload["pc"]["actuator_residual_channel_gains"] = [1.0, 1e-6, 1e-6]
    training = parse_training_config(payload)
    assert training.pc is not None
    assert training.pc.actuator_residual_channel_gains == (1.0, 1e-6, 1e-6)
    assert training.pc.as_dict()["actuator_residual_channel_gains"] == [
        1.0,
        1e-6,
        1e-6,
    ]


def test_goal173_channel_gains_reject_nonpositive_values() -> None:
    payload = _payload()
    payload["pc"]["actuator_residual_channel_gains"] = [1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="three finite positive values"):
        parse_training_config(payload)


def test_goal173_sac_actuator_channel_gains_are_learner_only() -> None:
    config = load_config(ENV_CONFIG)
    config = replace(
        config,
        plant=replace(config.plant, max_episode_steps=8),
        reference=replace(config.reference, path_bank_size=8, path_bank_seed=41),
        evaluation=replace(config.evaluation, training_seed_start=240),
    )
    env = PCCommandResidualOffPolicyEnv(
        config,
        num_envs=1,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
        action_interface="actuator",
        learner_actuator_channel_gains=(1.0, 0.0, 0.0),
    )
    env.reset([20])
    env.step(np.ones((1, 3), dtype=np.float64), auto_reset=False)
    assert np.allclose(env.last_learner_gated_action, [[1.0, 0.0, 0.0]])


def test_goal173_sac_actuator_channel_gains_reject_command_interface() -> None:
    config = load_config(ENV_CONFIG)
    with pytest.raises(ValueError, match="only supported for actuator"):
        PCCommandResidualOffPolicyEnv(
            config,
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="command",
            learner_actuator_channel_gains=(1.0, 0.0, 0.0),
        )
