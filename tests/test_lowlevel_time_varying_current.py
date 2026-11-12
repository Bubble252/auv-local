from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.env import BatchedLowLevelEnv
from auv_lowlevel.config import (
    CurrentShearConfig,
    CurrentVariationConfig,
    load_config,
    parse_config,
)
from auv_lowlevel.randomization import (
    SmoothTemporalCurrent,
    SpatialShear,
    UniformShearCurrent,
    episode_current_field,
    sample_episode_randomization,
    spatial_shear_from_config,
)

BASE_CONFIG = Path("configs/lowlevel/odarrl_control_v2_26d.json")


def _raw_config() -> dict:
    return json.loads(BASE_CONFIG.read_text(encoding="utf-8"))


def test_time_variation_config_validation() -> None:
    raw = _raw_config()
    raw["current"]["constant_within_episode"] = False
    with pytest.raises(ValueError, match="requires current.time_variation"):
        parse_config(raw)

    raw = _raw_config()
    raw["current"]["time_variation"] = {
        "speed_amplitude_mps": 0.05,
        "direction_amplitude_rad": 0.4,
        "period_s": 20.0,
    }
    with pytest.raises(ValueError, match="constant_within_episode=false"):
        parse_config(raw)

    raw["current"]["constant_within_episode"] = False
    config = parse_config(raw)
    assert config.current.time_variation == CurrentVariationConfig(
        speed_amplitude_mps=0.05,
        direction_amplitude_rad=0.4,
        period_s=20.0,
    )

    raw["current"]["time_variation"]["speed_amplitude_mps"] = 0.5
    with pytest.raises(ValueError, match="must not exceed"):
        parse_config(raw)

    raw = _raw_config()
    raw["current"]["enabled"] = False
    raw["current"]["constant_within_episode"] = False
    raw["current"]["time_variation"] = {
        "speed_amplitude_mps": 0.05,
        "direction_amplitude_rad": 0.4,
        "period_s": 20.0,
    }
    with pytest.raises(ValueError, match="requires current.enabled=true"):
        parse_config(raw)


def test_smooth_current_is_periodic_and_phase_deterministic() -> None:
    field = SmoothTemporalCurrent(
        base_velocity_ned_mps=np.array([0.3, 0.1, 0.0]),
        speed_amplitude_mps=0.05,
        direction_amplitude_rad=0.4,
        period_s=20.0,
        phase_rad=0.7,
    )
    position = np.zeros(3)
    first = field.sample(position, 0.0)
    later = field.sample(position, 4.0)
    repeated = field.sample(position, 20.0)
    assert not np.allclose(first, later)
    np.testing.assert_allclose(first, repeated, atol=1e-12, rtol=1e-12)
    assert first[2] == 0.0

    config = load_config(BASE_CONFIG)
    config = replace(
        config,
        current=replace(
            config.current,
            time_variation=CurrentVariationConfig(0.05, 0.4, 20.0),
        ),
    )
    first_randomization = sample_episode_randomization(123, config.current, config.reference)
    second_randomization = sample_episode_randomization(123, config.current, config.reference)
    assert first_randomization.current_phase_rad == pytest.approx(
        second_randomization.current_phase_rad
    )
    assert 0.0 <= first_randomization.current_phase_rad < 2.0 * np.pi
    current_field = episode_current_field(first_randomization, config.current)
    assert isinstance(current_field, SmoothTemporalCurrent)


def test_spatial_shear_config_and_sampling() -> None:
    raw = _raw_config()
    gradient = [[0.002, 0.001, 0.010], [-0.001, -0.002, -0.008], [0.0, 0.0, 0.0]]
    raw["current"]["type"] = "sheared_horizontal"
    raw["current"]["shear"] = {
        "velocity_gradient_per_s": gradient,
        "origin_ned_m": [10.0, 0.0, 11.0],
    }
    config = parse_config(raw)
    assert config.current.shear is not None
    np.testing.assert_allclose(config.current.shear.velocity_gradient_per_s, gradient)
    np.testing.assert_allclose(config.current.shear.origin_ned_m, [10.0, 0.0, 11.0])

    shear = spatial_shear_from_config(config.current)
    assert isinstance(shear, SpatialShear)
    base = np.array([0.4, -0.2, 0.0])
    origin = np.array([10.0, 0.0, 11.0])
    np.testing.assert_allclose(shear.apply(base, origin), base, atol=1e-12)
    np.testing.assert_allclose(
        shear.apply(base, origin + np.array([2.0, -3.0, 1.0])),
        base + np.array([0.002 * 2.0 - 0.001 * 3.0 + 0.010, -0.001 * 2.0 + 0.002 * 3.0 - 0.008, 0.0]),
        atol=1e-12,
    )

    randomization = sample_episode_randomization(123, config.current, config.reference)
    field = episode_current_field(randomization, config.current)
    assert isinstance(field, UniformShearCurrent)
    at_origin = field.sample(origin, 0.0)
    away = field.sample(origin + np.array([0.0, 0.0, 4.0]), 0.0)
    assert not np.allclose(at_origin, away)
    assert at_origin[2] == 0.0 and away[2] == 0.0


def test_spatial_shear_config_validation() -> None:
    raw = _raw_config()
    raw["current"]["type"] = "sheared_horizontal"
    with pytest.raises(ValueError, match="requires current.shear"):
        parse_config(raw)

    raw["current"]["shear"] = {
        "velocity_gradient_per_s": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]],
        "origin_ned_m": [0.0, 0.0, 0.0],
    }
    with pytest.raises(ValueError, match="bottom row must be zero"):
        parse_config(raw)

    raw = _raw_config()
    raw["current"]["shear"] = {
        "velocity_gradient_per_s": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "origin_ned_m": [0.0, 0.0, 0.0],
    }
    with pytest.raises(ValueError, match="must not define"):
        parse_config(raw)


def test_batched_environment_current_changes_with_time() -> None:
    config = load_config(BASE_CONFIG)
    config = replace(
        config,
        current=replace(
            config.current,
            time_variation=CurrentVariationConfig(0.05, 0.4, 20.0),
        ),
    )
    env = BatchedLowLevelEnv(config, num_envs=4, device="cpu", dtype=torch.float64)
    env.reset([11, 12, 13, 14])
    initial = env.current_ned_mps.clone()
    samples = [initial]
    for _ in range(20):
        env.step(torch.zeros((4, 3), dtype=torch.float64))
        samples.append(env.current_ned_mps.clone())
    stacked = torch.stack(samples, dim=0)
    assert torch.isfinite(stacked).all()
    assert bool((torch.diff(stacked, dim=0).abs() > 1e-9).any())
    np.testing.assert_allclose(
        initial[:, 2].numpy(), np.zeros(4), atol=0.0
    )


def test_batched_environment_current_changes_with_position_under_shear() -> None:
    config = load_config(BASE_CONFIG)
    shear = CurrentShearConfig(
        velocity_gradient_per_s=np.array(
            [[0.0, 0.0, 0.02], [0.0, 0.0, -0.01], [0.0, 0.0, 0.0]]
        ),
        origin_ned_m=np.array([0.0, 0.0, 10.0]),
    )
    config = replace(config, current=replace(config.current, shear=shear))
    env = BatchedLowLevelEnv(config, num_envs=2, device="cpu", dtype=torch.float64)
    env.reset([21, 22])
    env.eta[:, 2] = torch.tensor([8.0, 14.0], dtype=torch.float64)
    env._refresh_current()
    assert env.current_ned_mps is not None
    assert torch.allclose(env.current_ned_mps[:, 2], torch.zeros(2, dtype=torch.float64))
    assert not torch.allclose(env.current_ned_mps[0, :2], env.current_ned_mps[1, :2])


def test_batched_shear_current_matches_scalar_field_at_same_state() -> None:
    config = load_config(BASE_CONFIG)
    shear = CurrentShearConfig(
        velocity_gradient_per_s=np.array(
            [[0.002, 0.001, 0.01], [-0.001, -0.002, -0.008], [0.0, 0.0, 0.0]]
        ),
        origin_ned_m=np.array([15.0, 0.0, 11.0]),
    )
    config = replace(
        config,
        current=replace(
            config.current,
            time_variation=CurrentVariationConfig(0.08, 0.5, 15.0),
            shear=shear,
        ),
    )
    env = BatchedLowLevelEnv(config, num_envs=2, device="cpu", dtype=torch.float64)
    seeds = [41, 42]
    env.reset(seeds)
    positions = np.array([[5.0, 1.0, 8.0], [25.0, -3.0, 14.0]])
    times = np.array([3.2, 12.7])
    env.eta[:, :3] = torch.from_numpy(positions)
    env.t_s = torch.from_numpy(times)
    env._refresh_current()
    assert env.current_ned_mps is not None

    for index, seed in enumerate(seeds):
        randomization = sample_episode_randomization(seed, config.current, config.reference)
        field = episode_current_field(randomization, config.current)
        expected = field.sample(positions[index], float(times[index]))
        np.testing.assert_allclose(
            env.current_ned_mps[index].numpy(),
            expected,
            atol=1e-12,
            rtol=1e-12,
        )
