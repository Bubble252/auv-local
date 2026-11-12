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
    gate_mode: str,
    cap_ratio: float,
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
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
        base_residual_bias=bias,
        residual_terminal_gate_enabled=True,
        residual_terminal_gate_enter_m=10.0,
        residual_terminal_gate_full_m=3.0,
        learner_terminal_gate_mode=gate_mode,
        learner_norm_cap_ratio=cap_ratio,
    )
    env.base_residual_policy = _BaseResidual(base_value)
    env.reset([20, 21])
    return env


def _requested_normalized(env: PCCommandResidualOffPolicyEnv, info: dict) -> np.ndarray:
    requested = info["actuator_requested_physical"].detach().cpu().numpy()
    base = info["actuator_base_physical"].detach().cpu().numpy()
    scale = env.wrapper._full_scale.detach().cpu().numpy()
    return (requested - base) / scale


def test_none_mode_preserves_existing_gated_learner_plus_bias() -> None:
    env = _env(gate_mode="none", cap_ratio=0.0, bias=(0.0, -0.2, 0.0))
    env._residual_gate = lambda: np.full(2, 0.25, dtype=np.float64)
    action = np.full((2, 3), 0.4, dtype=np.float64)
    _, _, _, _, info = env.step(action, auto_reset=False)
    expected = (action + np.array([0.0, -0.2, 0.0])) * 0.25
    assert np.allclose(env.last_learner_gated_action, action * 0.25)
    assert np.allclose(env.last_learned_residual_action, expected)
    assert np.allclose(_requested_normalized(env, info), expected + 0.1)


def test_shell_mode_suppresses_learner_inside_capture_and_outside_approach() -> None:
    env = _env(gate_mode="shell", cap_ratio=0.0)
    action = np.full((2, 3), 0.4, dtype=np.float64)
    for gate in (0.0, 0.25, 0.5, 0.75, 1.0):
        env._residual_gate = lambda gate=gate: np.full(2, gate, dtype=np.float64)
        _, _, _, _, info = env.step(action, auto_reset=False)
        expected = action * gate * 4.0 * (1.0 - gate)
        assert np.allclose(env.last_learner_gated_action, expected)
        assert np.allclose(_requested_normalized(env, info), expected + 0.1)


def test_norm_cap_limits_learner_to_fraction_of_frozen_bc() -> None:
    env = _env(gate_mode="none", cap_ratio=0.25, base_value=0.1)
    env._residual_gate = lambda: np.ones(2, dtype=np.float64)
    action = np.full((2, 3), 0.4, dtype=np.float64)
    env.step(action, auto_reset=False)
    base_norm = np.linalg.norm(np.full(3, 0.1))
    expected_norm = 0.25 * base_norm
    assert np.allclose(
        np.linalg.norm(env.last_learner_gated_action, axis=1), expected_norm
    )
    assert np.allclose(
        env.last_learner_gated_action,
        np.full((2, 3), expected_norm / np.sqrt(3.0)),
    )


def test_norm_cap_zeros_learner_when_bc_is_zero() -> None:
    env = _env(gate_mode="none", cap_ratio=0.25, base_value=0.0)
    env._residual_gate = lambda: np.ones(2, dtype=np.float64)
    env.step(np.full((2, 3), 0.4, dtype=np.float64), auto_reset=False)
    assert np.allclose(env.last_learner_gated_action, 0.0)
    assert np.allclose(env.last_applied_residual, 0.0)


def test_shell_and_cap_compose_and_leave_bias_gated() -> None:
    env = _env(gate_mode="shell", cap_ratio=0.25, bias=(0.0, -0.2, 0.0))
    env._residual_gate = lambda: np.full(2, 0.5, dtype=np.float64)
    _, _, _, _, info = env.step(np.full((2, 3), 0.4, dtype=np.float64), auto_reset=False)
    base_norm = np.linalg.norm(np.full(3, 0.1))
    expected_learner_norm = 0.25 * base_norm
    assert np.allclose(
        np.linalg.norm(env.last_learner_gated_action, axis=1), expected_learner_norm
    )
    expected_applied = 0.1 + env.last_learner_gated_action + np.array([0.0, -0.1, 0.0])
    assert np.allclose(_requested_normalized(env, info), expected_applied)


def test_shell_requires_terminal_gate() -> None:
    with pytest.raises(ValueError, match="shell gate requires"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            learner_terminal_gate_mode="shell",
        )


def test_norm_cap_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            learner_norm_cap_ratio=-0.1,
        )
