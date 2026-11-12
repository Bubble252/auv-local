from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.sensors import PROFILES, SyntheticSensorSuite
from auv_lowlevel.config import load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "lowlevel" / "odarrl_control_base.json"
CONFIG_DIR = ROOT / "configs" / "lowlevel"
GOAL85_CONFIGS = (
    "pc_goal85_unbiased_default_baseline_20261110.json",
    "pc_goal85_unbiased_strict125_baseline_20261110.json",
    "pc_goal85_unbiased_default_candidate_20261110.json",
    "pc_goal85_unbiased_strict125_candidate_20261110.json",
)


def _pc_config(
    *,
    profile: str = "unbiased_noise_v1",
    position_source: str = "eta",
    endpoint_stop: bool = True,
):
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": "pc_goal85_test",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": "C0",
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["current"]["enabled"] = False
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["sensors"] = {"profile": profile, "seed": 123}
    data["reward"].update(
        {
            "mode": "pc_command_track",
            "hold_bonus": True,
            "w_prog": 0.0,
            "w_3d": 0.0,
            "w_yaw": 0.0,
            "w_u": 0.0,
            "w_action": 0.0,
            "w_sat": 0.0,
            "terminal_progress_weight": 0.0,
            "goal_bonus": 0.0,
            "timeout_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
        }
    )
    data["actuator"] = {
        "envelope": "pc20",
        "max_rudder_angle_deg": 20.0,
        "max_stern_plane_angle_deg": 20.0,
        "max_fin_rate_deg_s": 150.0,
        "max_propeller_rpm": 1525.0,
        "plant_max_rudder_angle_deg": 20.0,
        "plant_max_stern_plane_angle_deg": 20.0,
        "plant_max_fin_rate_deg_s": 150.0,
        "plant_max_propeller_rpm": 1525.0,
        "material_tolerance_fraction": 0.01,
    }
    data["termination"] = {
        "phi_term_deg": 45.0,
        "t_phi_term_s": 1.0,
        "p_term_deg_s": 90.0,
        "t_p_term_s": 0.5,
        "safety_terminal_penalty": -50.0,
        "pc_goal_plane_miss_termination_enabled": endpoint_stop,
        "position_source": position_source,
    }
    return parse_config(data)


def test_unbiased_profile_has_no_systematic_navigation_terms() -> None:
    profile = PROFILES["unbiased_noise_v1"]
    assert profile.nav_bias_m == (0.0, 0.0)
    assert profile.nav_drift_sigma_m == (0.0, 0.0)
    assert profile.nav_usbl_gain == 0.0
    assert profile.nav_dvl_outage_drift_mps == (0.0, 0.0)
    assert profile.nav_delay_s == (0.0, 0.0)
    assert profile.nav_bias_calibration_fraction == 0.0
    assert profile.nav_noise_m[0] > 0.0
    assert 0.05 <= profile.nav_noise_m[0] <= profile.nav_noise_m[1] <= 0.20
    for field in (
        "dvl_bottom_delay_s",
        "dvl_water_delay_s",
        "imu_att_delay_s",
        "imu_gyro_delay_s",
        "usbl_delay_s",
        "current_est_delay_s",
    ):
        assert getattr(profile, field) == (0.0, 0.0)


def test_unbiased_nav_noise_is_zero_mean_without_drift() -> None:
    config = _pc_config()
    suite = SyntheticSensorSuite(config, num_envs=8)
    seeds = list(range(20, 28))
    suite.reset(seeds)
    eta = torch.tensor(
        [[2.0, -1.0, 10.0, 0.01, -0.02, 0.3]] * 8, dtype=torch.float64
    )
    nu = torch.tensor(
        [[1.0, 0.05, -0.02, 0.01, -0.005, 0.02]] * 8, dtype=torch.float64
    )
    current = torch.zeros((8, 3), dtype=torch.float64)
    errors = []
    for _ in range(2000):
        output = suite.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)
        errors.append((output.p_nav - eta[:, :3]).clone())
    samples = torch.stack(errors, dim=0)
    mean = samples.mean(dim=(0, 1))
    std = samples.std(dim=(0, 1), unbiased=False)
    assert torch.all(mean.abs() < 0.02), mean
    assert torch.all(std > 0.08) and torch.all(std < 0.22), std
    first = samples[:400].mean(dim=(0, 1))
    last = samples[-400:].mean(dim=(0, 1))
    assert torch.all((last - first).abs() < 0.02), (first, last)


def test_termination_position_source_eta_ignores_pnav_bias() -> None:
    config = _pc_config(position_source="eta")
    env = BatchedPCLowLevelEnv(config, num_envs=2, validate_finite=True)
    env.reset([20, 21])
    assert env.sensor_output is not None
    shifted = replace(env.sensor_output, p_nav=env.sensor_output.p_nav + 100.0)
    env.sensor_output = shifted

    termination_position = env._termination_position_ned()
    assert torch.equal(termination_position, env.eta[:, :3])
    expected = torch.linalg.norm(
        env.eta[:, :3] - env.path.goal_ned_m_for(2), dim=1
    )
    assert torch.allclose(env._goal_distance_for_reward(), expected)

    legacy = replace(
        config,
        termination=replace(config.termination, position_source="legacy"),
    )
    env_legacy = BatchedPCLowLevelEnv(legacy, num_envs=2, validate_finite=True)
    env_legacy.reset([20, 21])
    assert env_legacy.sensor_output is not None
    env_legacy.sensor_output = replace(
        env_legacy.sensor_output, p_nav=env_legacy.sensor_output.p_nav + 100.0
    )
    assert torch.equal(
        env_legacy._termination_position_ned(), env_legacy.sensor_output.p_nav
    )


def test_termination_position_source_rejects_unknown_value() -> None:
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["termination"] = {"position_source": "truthish"}
    with pytest.raises(ValueError, match="position_source"):
        parse_config(data)


def test_goal85_configs_use_unbiased_profile_and_eta_scoring() -> None:
    for name in GOAL85_CONFIGS:
        config = load_config(CONFIG_DIR / name)
        assert config.sensors is not None
        assert config.sensors.profile == "unbiased_noise_v1"
        assert config.termination.position_source == "eta"
        assert config.observation.ideal_actor_observations is False
        assert config.termination.pc_goal_plane_miss_termination_enabled is True


def test_historical_realistic_config_keeps_legacy_position_source() -> None:
    config = load_config(
        CONFIG_DIR / "pc_goal71_command_rate_limit_default_20261110.json"
    )
    assert config.sensors is not None
    assert config.sensors.profile == "realistic_calibrated"
    assert config.termination.position_source == "legacy"
