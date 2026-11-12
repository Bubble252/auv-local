from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.sensors import SyntheticSensorSuite, _EpisodeParameters
from auv_lowlevel.config import load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "lowlevel" / "odarrl_control_base.json"
GOAL85_CONFIG = (
    ROOT
    / "configs"
    / "lowlevel"
    / "pc_goal85_unbiased_default_candidate_20261110.json"
)


def _pc_config(*, profile: str = "unbiased_noise_v1", sensors: dict | None = None):
    data = json.loads(GOAL85_CONFIG.read_text(encoding="utf-8"))
    sensor_data = {"profile": profile, "seed": 123}
    if sensors is not None:
        sensor_data.update(sensors)
    data["sensors"] = sensor_data
    return parse_config(data)


def _with_sensor_overrides(config, **overrides):
    assert config.sensors is not None
    return replace(config, sensors=replace(config.sensors, **overrides))


def _inputs(num_envs: int = 4):
    eta = torch.tensor(
        [[2.0, -1.0, 10.0, 0.01, -0.02, 0.3]] * num_envs,
        dtype=torch.float64,
    )
    nu = torch.tensor(
        [[1.0, 0.05, -0.02, 0.01, -0.005, 0.02]] * num_envs,
        dtype=torch.float64,
    )
    current = torch.zeros((num_envs, 3), dtype=torch.float64)
    return eta, nu, current


def _noise_deltas(output, eta, nu):
    return {
        "p_nav": output.p_nav - eta[:, :3],
        "attitude": output.attitude - eta[:, 3:6],
        "angular_rate": output.angular_rate - nu[:, 3:6],
        "dvl_ground": output.dvl_ground_body - nu[:, :3],
        "dvl_water": output.dvl_water_body - nu[:, :3],
        "current_estimate": output.current_estimate_body,
        "usbl_position": output.usbl_position - eta[:, :3],
    }


def test_noise_scale_default_is_one_and_fixed_scale_consumes_no_rng() -> None:
    config = _pc_config()
    suite = SyntheticSensorSuite(config, num_envs=6)
    suite._generator.manual_seed(9876)
    state_before = suite._generator.get_state().clone()
    scale = suite._sample_noise_scale(6)
    assert torch.equal(scale, torch.ones(6, dtype=torch.float64))
    assert torch.equal(state_before, suite._generator.get_state())

    seeds = list(range(20, 26))
    suite._params = suite._sample_parameters(seeds)
    default_state = suite._generator.get_state().clone()
    assert suite._params is not None
    default_parameters = suite._params

    scaled = replace(config, sensors=replace(config.sensors, noise_scale=0.35))
    scaled_suite = SyntheticSensorSuite(scaled, num_envs=6)
    scaled_suite._params = scaled_suite._sample_parameters(seeds)
    scaled_state = scaled_suite._generator.get_state().clone()
    assert scaled_suite._params is not None
    scaled_parameters = scaled_suite._params

    assert torch.equal(default_state, scaled_state)
    assert torch.equal(
        scaled_parameters.noise_scale,
        torch.full((6,), 0.35, dtype=torch.float64),
    )
    for field in fields(_EpisodeParameters):
        if field.name != "noise_scale":
            assert torch.equal(
                getattr(default_parameters, field.name),
                getattr(scaled_parameters, field.name),
            ), field.name


def test_degenerate_noise_scale_range_consumes_no_rng() -> None:
    config = _pc_config(sensors={"noise_scale_range": [1.25, 1.25]})
    suite = SyntheticSensorSuite(config, num_envs=5)
    suite._generator.manual_seed(123)
    state_before = suite._generator.get_state().clone()
    scale = suite._sample_noise_scale(5)
    assert torch.equal(scale, torch.full((5,), 1.25, dtype=torch.float64))
    assert torch.equal(state_before, suite._generator.get_state())


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.5, 2.0, 3.0])
def test_noise_scale_scales_only_zero_mean_noise_linearly(scale: float) -> None:
    config = _pc_config()
    nominal = SyntheticSensorSuite(config, num_envs=4)
    scaled_config = replace(config, sensors=replace(config.sensors, noise_scale=scale))
    scaled = SyntheticSensorSuite(scaled_config, num_envs=4)
    seeds = [20, 21, 22, 23]
    nominal.reset(seeds)
    scaled.reset(seeds)
    eta, nu, current = _inputs(4)

    nominal_output = nominal.step(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=current,
    )
    scaled_output = scaled.step(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=current,
    )
    nominal_deltas = _noise_deltas(nominal_output, eta, nu)
    scaled_deltas = _noise_deltas(scaled_output, eta, nu)
    for name in nominal_deltas:
        torch.testing.assert_close(
            scaled_deltas[name],
            nominal_deltas[name] * scale,
            rtol=0.0,
            atol=1e-12,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_zero_scale_matches_clean_noise_outputs() -> None:
    clean_config = _pc_config(profile="clean")
    zero_config = _pc_config(sensors={"noise_scale": 0.0})
    clean = SyntheticSensorSuite(clean_config, num_envs=4)
    zero = SyntheticSensorSuite(zero_config, num_envs=4)
    seeds = [20, 21, 22, 23]
    clean.reset(seeds)
    zero.reset(seeds)
    eta, nu, current = _inputs(4)

    for _ in range(20):
        clean_output = clean.step(
            eta_true=eta,
            nu_true=nu,
            current_true_ned_mps=current,
        )
        zero_output = zero.step(
            eta_true=eta,
            nu_true=nu,
            current_true_ned_mps=current,
        )
        clean_deltas = _noise_deltas(clean_output, eta, nu)
        zero_deltas = _noise_deltas(zero_output, eta, nu)
        for name in clean_deltas:
            torch.testing.assert_close(
                zero_deltas[name],
                clean_deltas[name],
                rtol=0.0,
                atol=1e-12,
                msg=lambda message, name=name: f"{name}: {message}",
            )


def test_noise_scale_does_not_change_bias_drift_scale_delay_or_outage_parameters() -> None:
    config = _pc_config(profile="realistic_calibrated")
    base = SyntheticSensorSuite(config, num_envs=8)
    doubled_config = replace(config, sensors=replace(config.sensors, noise_scale=2.0))
    doubled = SyntheticSensorSuite(doubled_config, num_envs=8)
    seeds = list(range(20, 28))
    base._params = base._sample_parameters(seeds)
    doubled._params = doubled._sample_parameters(seeds)
    assert base._params is not None and doubled._params is not None

    kept_fields = {
        "dvl_bottom_delay_steps",
        "dvl_bottom_bias",
        "dvl_bottom_scale",
        "dvl_bottom_outage_start_prob",
        "dvl_bottom_outage_duration_s",
        "dvl_water_delay_steps",
        "dvl_water_bias",
        "dvl_water_scale",
        "dvl_water_outage_start_prob",
        "dvl_water_outage_duration_s",
        "imu_att_delay_steps",
        "imu_rollpitch_bias",
        "imu_heading_bias",
        "imu_mag_declination",
        "imu_drift_sigma",
        "imu_gyro_delay_steps",
        "imu_gyro_bias",
        "imu_gyro_scale",
        "usbl_delay_steps",
        "usbl_outlier_prob",
        "usbl_outlier_jump",
        "current_est_delay_steps",
        "current_est_bias",
        "current_est_walk",
        "current_est_tau_s",
        "current_est_outage_start_prob",
        "current_est_outage_duration_s",
        "nav_bias",
        "nav_drift_sigma",
        "nav_drift_tau_s",
        "nav_dvl_outage_drift",
        "nav_delay_steps",
    }
    for name in kept_fields:
        assert torch.equal(
            getattr(base._params, name),
            getattr(doubled._params, name),
        ), name


def test_noise_scale_range_is_deterministic_and_reset_rows_merge_correctly() -> None:
    config = _pc_config(sensors={"noise_scale_range": [0.25, 2.0]})
    suite = SyntheticSensorSuite(config, num_envs=6)
    seeds = [20, 21, 22, 23, 24, 25]
    suite.reset(seeds)
    assert suite._params is not None
    first = suite._params.noise_scale.clone()
    assert torch.all(first >= 0.25) and torch.all(first <= 2.0)
    assert torch.unique(first).numel() > 1

    suite.reset(seeds)
    assert suite._params is not None
    assert torch.equal(first, suite._params.noise_scale)

    mask = torch.tensor([True, False, True, False, False, False])
    suite.reset_rows(mask, [120, 121])
    assert suite._params is not None
    updated = suite._params.noise_scale
    assert torch.equal(updated[~mask], first[~mask])
    assert torch.all(updated[mask] >= 0.25) and torch.all(updated[mask] <= 2.0)


def test_sensor_noise_scale_config_parser_accepts_fixed_and_range() -> None:
    fixed = _pc_config(sensors={"noise_scale": 0.0})
    assert fixed.sensors is not None
    assert fixed.sensors.noise_scale == 0.0
    assert fixed.sensors.noise_scale_range is None

    sampled = _pc_config(sensors={"noise_scale_range": [0.25, 2.0]})
    assert sampled.sensors is not None
    assert sampled.sensors.noise_scale == 1.0
    assert sampled.sensors.noise_scale_range == (0.25, 2.0)


@pytest.mark.parametrize(
    "sensors, message",
    [
        ({"noise_scale": -0.1}, "noise_scale"),
        ({"noise_scale": float("nan")}, "noise_scale"),
        ({"noise_scale_range": [-0.1, 1.0]}, "noise_scale_range"),
        ({"noise_scale_range": [2.0, 1.0]}, "min <= max"),
        (
            {"noise_scale": 2.0, "noise_scale_range": [0.5, 1.5]},
            "cannot be set together",
        ),
    ],
)
def test_sensor_noise_scale_config_parser_rejects_invalid_values(
    sensors: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _pc_config(sensors=sensors)


def test_goal85_configs_default_to_unit_noise_scale() -> None:
    config = load_config(GOAL85_CONFIG)
    assert config.sensors is not None
    assert config.sensors.noise_scale == 1.0
    assert config.sensors.noise_scale_range is None
