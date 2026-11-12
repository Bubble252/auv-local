from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.config import load_config
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv
import auv_rl.ddpg.pc_command_residual_env as env_module

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


class _ZeroResidual:
    def act_inference(self, *_args):
        return torch.zeros((2, 3), dtype=torch.float32)


def _env(
    *,
    vertical_gain: float = 0.0,
    rpm_gain: float = 0.0,
    max_bias: float = 0.15,
    gate_enter_m: float = 10.0,
    gate_full_m: float = 2.0,
    force_geometry_gate: bool = True,
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
        geometry_vertical_gain=vertical_gain,
        geometry_rpm_gain=rpm_gain,
        geometry_max_bias=max_bias,
        geometry_gate_enter_m=gate_enter_m,
        geometry_gate_full_m=gate_full_m,
    )
    env.base_residual_policy = _ZeroResidual()
    env.reset([20, 21])
    env._residual_gate = lambda: np.ones(2, dtype=np.float64)
    if force_geometry_gate:
        env._geometry_gate = lambda: np.ones(2, dtype=np.float64)
    return env


def test_vertical_gain_is_normalized_actuator_per_meter(monkeypatch) -> None:
    env = _env(vertical_gain=0.02, max_bias=0.5)

    def fake_path(_path):
        # normalized signed vertical error = 0.4 (2 m / 5 m scale)
        return torch.tensor([[0.0, 0.4] + [0.0] * 6] * 2, dtype=torch.float64)

    monkeypatch.setattr(env_module, "normalize_path_features", fake_path)
    env.step(np.zeros((2, 3), dtype=np.float64), auto_reset=False)
    expected = np.array([[0.0, 0.04, 0.0], [0.0, 0.04, 0.0]])
    assert np.allclose(env.last_geometry_bias, expected)
    assert np.allclose(env.last_learned_residual_action, expected)


def test_geometry_bias_scales_with_independent_geometry_gate(monkeypatch) -> None:
    env = _env(vertical_gain=0.05, rpm_gain=0.1)
    env._geometry_gate = lambda: np.array([0.5, 1.0], dtype=np.float64)
    monkeypatch.setattr(
        env_module,
        "normalize_path_features",
        lambda path: torch.tensor([[0.0, 0.4] + [0.0] * 6] * 2, dtype=torch.float64),
    )
    env.step(np.zeros((2, 3), dtype=np.float64), auto_reset=False)
    expected = np.array([[0.0, 0.05, 0.05], [0.0, 0.10, 0.10]])
    assert np.allclose(env.last_geometry_bias, expected)
    assert np.allclose(env.last_geometry_gate, [0.5, 1.0])


def test_geometry_gate_uses_its_own_enter_and_full_radius(monkeypatch) -> None:
    env = _env(vertical_gain=0.02, force_geometry_gate=False)
    calls: list[tuple[float, float]] = []

    def fake_gate(distance, *, enter_m, full_m):
        calls.append((float(enter_m), float(full_m)))
        return torch.ones_like(distance)

    monkeypatch.setattr(env_module, "smooth_terminal_gate", fake_gate)
    gate = env._geometry_gate()
    assert np.allclose(gate, 1.0)
    assert calls == [(10.0, 2.0)]


def test_rpm_gain_adds_gated_closure_bias(monkeypatch) -> None:
    env = _env(rpm_gain=0.05)
    monkeypatch.setattr(
        env_module,
        "normalize_path_features",
        lambda path: torch.zeros((2, 8), dtype=torch.float64),
    )
    env.step(np.zeros((2, 3), dtype=np.float64), auto_reset=False)
    assert np.allclose(env.last_geometry_bias, np.array([[0.0, 0.0, 0.05]] * 2))


def test_geometry_bias_is_capped(monkeypatch) -> None:
    env = _env(vertical_gain=10.0, max_bias=0.15)
    monkeypatch.setattr(
        env_module,
        "normalize_path_features",
        lambda path: torch.tensor([[0.0, 0.4] + [0.0] * 6] * 2, dtype=torch.float64),
    )
    env.step(np.zeros((2, 3), dtype=np.float64), auto_reset=False)
    assert np.allclose(env.last_geometry_bias[:, 1], 0.15)


def test_geometry_bias_zero_when_gains_disabled(monkeypatch) -> None:
    env = _env()
    monkeypatch.setattr(
        env_module,
        "normalize_path_features",
        lambda path: torch.tensor([[0.0, 0.4] + [0.0] * 6] * 2, dtype=torch.float64),
    )
    env.step(np.zeros((2, 3), dtype=np.float64), auto_reset=False)
    assert np.allclose(env.last_geometry_bias, 0.0)
    assert np.allclose(env.last_geometry_gate, 1.0)


def test_disabled_geometry_reproduces_base_rollout() -> None:
    base = _env()
    gated = _env(gate_enter_m=8.0, gate_full_m=1.0)
    action = np.array([[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]], dtype=np.float64)
    for _ in range(4):
        base_obs, base_reward, base_term, base_trunc, _ = base.step(
            action, auto_reset=False
        )
        gated_obs, gated_reward, gated_term, gated_trunc, _ = gated.step(
            action, auto_reset=False
        )
        assert np.array_equal(base_obs, gated_obs)
        assert np.array_equal(base_reward, gated_reward)
        assert np.array_equal(base_term, gated_term)
        assert np.array_equal(base_trunc, gated_trunc)
        assert np.array_equal(base.last_geometry_bias, gated.last_geometry_bias)


def test_geometry_bias_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="geometry gains"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            geometry_vertical_gain=float("nan"),
        )
    with pytest.raises(ValueError, match="geometry_max_bias"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            geometry_max_bias=0.0,
        )
    with pytest.raises(ValueError, match="geometry_gate_full_m"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="actuator",
            geometry_gate_enter_m=2.0,
            geometry_gate_full_m=3.0,
        )
    with pytest.raises(ValueError, match="actuator residuals"):
        PCCommandResidualOffPolicyEnv(
            _config(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
            path_dim=8,
            action_interface="command",
            geometry_vertical_gain=0.02,
        )
