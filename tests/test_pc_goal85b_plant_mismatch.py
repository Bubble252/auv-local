from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from auv_lowlevel.batch.dynamics import BatchedRemus100Dynamics, Remus100BatchState
from auv_lowlevel.batch.plant_randomization import sample_plant_scale_factors
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import (
    PlantRandomizationConfig,
    load_config,
    parse_config,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "lowlevel" / "odarrl_control_base.json"
GOAL85_CONFIG = (
    ROOT
    / "configs"
    / "lowlevel"
    / "pc_goal85_unbiased_default_baseline_20261110.json"
)


def _mismatch_config(fraction: float) -> PlantRandomizationConfig:
    scale = (1.0 - fraction, 1.0 + fraction)
    return PlantRandomizationConfig(
        enabled=fraction > 0.0,
        seed=8505,
        drag_scale=scale,
        thrust_scale=scale,
        fin_scale=scale,
        added_mass_scale=scale,
        actuator_time_scale=scale,
    )


def _state_and_command() -> tuple[Remus100BatchState, torch.Tensor, torch.Tensor]:
    eta = torch.tensor(
        [[0.0, 0.0, 10.0, 0.02, -0.03, 0.4], [1.0, 2.0, 9.0, -0.01, 0.02, -0.3]],
        dtype=torch.float64,
    )
    nu = torch.tensor(
        [[1.0, 0.05, -0.02, 0.01, -0.005, 0.02], [0.8, -0.1, 0.03, -0.02, 0.01, -0.04]],
        dtype=torch.float64,
    )
    actuator = torch.tensor(
        [[0.1, 0.05, 700.0], [-0.2, -0.04, 900.0]], dtype=torch.float64
    )
    command = torch.tensor(
        [[0.12, 0.06, 720.0], [-0.18, -0.03, 880.0]], dtype=torch.float64
    )
    current = torch.tensor([[0.1, 0.2], [0.0, 0.0]], dtype=torch.float64)
    return Remus100BatchState(eta=eta, nu=nu, actuator=actuator), command, current


def test_plant_randomization_config_parses_and_defaults_are_legacy() -> None:
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    assert parse_config(data).plant_randomization.enabled is False
    data["plant_randomization"] = {
        "enabled": True,
        "seed": 8505,
        "drag_scale": [0.95, 1.05],
        "thrust_scale": [0.95, 1.05],
        "fin_scale": [0.95, 1.05],
        "added_mass_scale": [0.95, 1.05],
        "actuator_time_scale": [1.0, 1.0],
    }
    config = parse_config(data)
    assert config.plant_randomization.enabled is True
    assert config.plant_randomization.drag_scale == (0.95, 1.05)


def test_plant_scale_sampler_is_order_independent_and_seed_stable() -> None:
    config = _mismatch_config(0.10)
    first = sample_plant_scale_factors([20, 21, 22], config)
    second = sample_plant_scale_factors([22, 20, 21], config)
    assert torch.equal(first, second[[1, 2, 0]])
    assert torch.equal(first, sample_plant_scale_factors([20, 21, 22], config))
    assert bool((first >= 0.90).all()) and bool((first <= 1.10).all())
    assert len(set(map(tuple, first.tolist()))) == 3


def test_disabled_plant_randomization_is_exact_identity() -> None:
    factors = sample_plant_scale_factors([20, 21], PlantRandomizationConfig())
    assert torch.equal(factors, torch.ones((2, 5), dtype=torch.float64))


def test_zero_mismatch_dynamics_is_bitwise_legacy_equivalent() -> None:
    dynamics = BatchedRemus100Dynamics(dtype=torch.float64)
    state, command, current = _state_and_command()
    legacy = dynamics.step(state, command, current)
    zero = dynamics.step(
        state,
        command,
        current,
        plant_scales=torch.ones((2, 5), dtype=torch.float64),
    )
    for field in ("eta", "nu", "actuator"):
        assert torch.equal(
            getattr(legacy.state, field), getattr(zero.state, field)
        )
    assert torch.equal(legacy.nu_dot, zero.nu_dot)


def test_each_scale_changes_the_intended_dynamics_channel() -> None:
    dynamics = BatchedRemus100Dynamics(dtype=torch.float64)
    state, command, current = _state_and_command()
    nominal = dynamics.step(state, command, current)

    def scaled(index: int, value: float) -> torch.Tensor:
        factors = torch.ones((2, 5), dtype=torch.float64)
        factors[:, index] = value
        return dynamics.step(state, command, current, plant_scales=factors).nu_dot

    drag = scaled(0, 1.10)
    thrust = scaled(1, 1.10)
    fin = scaled(2, 1.10)
    added_mass = scaled(3, 0.95)
    assert not torch.allclose(drag[:, 0], nominal.nu_dot[:, 0])
    assert not torch.allclose(drag[:, 1], nominal.nu_dot[:, 1])
    assert not torch.allclose(thrust[:, 0], nominal.nu_dot[:, 0])
    assert not torch.allclose(fin[:, 1], nominal.nu_dot[:, 1])
    assert not torch.allclose(fin[:, 5], nominal.nu_dot[:, 5])
    assert not torch.allclose(added_mass[:, 0], nominal.nu_dot[:, 0])
    assert torch.isfinite(added_mass).all()


def test_env_reset_is_seed_stable_and_reset_indices_updates_only_selected_rows() -> None:
    config = replace(load_config(GOAL85_CONFIG), plant_randomization=_mismatch_config(0.05))
    env = BatchedPCLowLevelEnv(
        config, num_envs=3, validate_finite=True, path_dim=12
    )
    env.reset([20, 21, 22])
    first = env.plant_scales.clone()
    expected = sample_plant_scale_factors([20, 21, 22], config.plant_randomization)
    assert torch.equal(first, expected)

    env.reset([20, 21, 22])
    assert torch.equal(env.plant_scales, first)
    env.reset_indices(
        torch.tensor([False, True, False], dtype=torch.bool), seeds=[31]
    )
    assert torch.equal(env.plant_scales[0], first[0])
    assert torch.equal(env.plant_scales[2], first[2])
    assert not torch.equal(env.plant_scales[1], first[1])


def test_plant_mismatch_does_not_change_navigation_noise_sequence() -> None:
    base = load_config(GOAL85_CONFIG)
    nominal = BatchedPCLowLevelEnv(
        base, num_envs=2, validate_finite=True, path_dim=12
    )
    mismatch = BatchedPCLowLevelEnv(
        replace(base, plant_randomization=_mismatch_config(0.10)),
        num_envs=2,
        validate_finite=True,
        path_dim=12,
    )
    seeds = [20, 21]
    nominal.reset(seeds)
    mismatch.reset(seeds)
    action = torch.zeros((2, 3), dtype=torch.float64)
    for _ in range(10):
        nominal.step(action, auto_reset=False)
        mismatch.step(action, auto_reset=False)
    assert nominal.eta is not None and nominal.sensor_output is not None
    assert mismatch.eta is not None and mismatch.sensor_output is not None
    nominal_error = nominal.sensor_output.p_nav - nominal.eta[:, :3]
    mismatch_error = mismatch.sensor_output.p_nav - mismatch.eta[:, :3]
    assert torch.allclose(nominal_error, mismatch_error, rtol=0.0, atol=1e-12)
