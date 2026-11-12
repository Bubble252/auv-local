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


def _config(*, max_steps: int = 8):
    config = load_config(ENV_CONFIG)
    return replace(
        config,
        plant=replace(config.plant, max_episode_steps=max_steps),
        reference=replace(
            config.reference,
            path_bank_size=8,
            path_bank_seed=41,
        ),
        evaluation=replace(config.evaluation, training_seed_start=240),
    )


class _BaseResidual:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def act_inference(self, *_args):
        return torch.full((2, 3), self.value, dtype=torch.float32)


def _env(
    *,
    channel_cap_ratios: tuple[float, float, float] | None = None,
    direction_cos_min: float | None = None,
    base_value: float = 0.1,
) -> PCCommandResidualOffPolicyEnv:
    env = PCCommandResidualOffPolicyEnv(
        _config(),
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
        action_interface="actuator",
        actuator_residual_scale=1.0,
        residual_terminal_gate_enabled=True,
        residual_terminal_gate_enter_m=10.0,
        residual_terminal_gate_full_m=3.0,
        learner_channel_cap_ratios=channel_cap_ratios,
        learner_direction_gate_cos_min=direction_cos_min,
    )
    env.base_residual_policy = _BaseResidual(base_value)
    env.reset([20, 21])
    env._residual_gate = lambda: np.ones(2, dtype=np.float64)
    return env


def _requested_normalized(env: PCCommandResidualOffPolicyEnv, info: dict) -> np.ndarray:
    requested = info["actuator_requested_physical"].detach().cpu().numpy()
    base = info["actuator_base_physical"].detach().cpu().numpy()
    scale = env.wrapper._full_scale.detach().cpu().numpy()
    return (requested - base) / scale


def test_rpm_only_channel_cap_leaves_fin_channels_untouched() -> None:
    env = _env(channel_cap_ratios=(0.0, 0.0, 0.25), base_value=0.1)
    env.step(np.full((2, 3), 0.4, dtype=np.float64), auto_reset=False)
    expected = np.array([[0.4, 0.4, 0.025], [0.4, 0.4, 0.025]])
    assert np.allclose(env.last_learner_gated_action, expected)
    assert np.allclose(env.last_learned_residual_action, expected)


def test_per_channel_cap_clamps_each_channel_independently() -> None:
    env = _env(channel_cap_ratios=(0.25, 0.5, 0.1), base_value=0.1)
    action = np.array([[0.4, -0.4, 0.4], [0.4, -0.4, 0.4]], dtype=np.float64)
    env.step(action, auto_reset=False)
    expected = np.array([[0.025, -0.05, 0.01], [0.025, -0.05, 0.01]])
    assert np.allclose(env.last_learner_gated_action, expected)


def test_channel_cap_zeros_channel_when_bc_channel_is_zero() -> None:
    env = _env(channel_cap_ratios=(0.0, 0.0, 0.25), base_value=0.0)
    env.step(np.full((2, 3), 0.4, dtype=np.float64), auto_reset=False)
    assert np.allclose(env.last_learner_gated_action[:, 2], 0.0)
    assert np.allclose(env.last_learner_gated_action[:, :2], 0.4)


def test_direction_gate_keeps_aligned_learner_and_drops_opposite() -> None:
    env = _env(direction_cos_min=0.0, base_value=0.1)
    action = np.array([[0.4, 0.4, 0.4], [-0.4, -0.4, -0.4]], dtype=np.float64)
    env.step(action, auto_reset=False)
    assert np.allclose(env.last_learner_gated_action[0], 0.4)
    assert np.allclose(env.last_learner_gated_action[1], 0.0)
    assert env.last_learner_direction_cos[0, 0] == pytest.approx(1.0)
    assert env.last_learner_direction_cos[1, 0] == pytest.approx(-1.0)


def test_direction_gate_zeros_learner_when_bc_is_zero() -> None:
    env = _env(direction_cos_min=-0.5, base_value=0.0)
    env.step(np.full((2, 3), 0.4, dtype=np.float64), auto_reset=False)
    assert np.allclose(env.last_learner_gated_action, 0.0)
    assert np.allclose(env.last_learner_direction_cos, 0.0)


def test_disabled_switches_preserve_plain_learner_action() -> None:
    env = _env(channel_cap_ratios=None, direction_cos_min=None, base_value=0.1)
    action = np.array([[0.4, -0.2, 0.1], [-0.3, 0.2, -0.1]], dtype=np.float64)
    env.step(action, auto_reset=False)
    assert np.allclose(env.last_learner_gated_action, action)


def test_channel_cap_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="three values"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            learner_channel_cap_ratios=(0.1, 0.2),
        )
    with pytest.raises(ValueError, match="non-negative"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            learner_channel_cap_ratios=(0.1, -0.2, 0.3),
        )


def test_direction_gate_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            learner_direction_gate_cos_min=1.5,
        )
