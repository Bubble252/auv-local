from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.geometry import quaternion_from_euler, world_to_body
from auv_lowlevel.batch.sensors import (
    PROFILES,
    SyntheticSensorSuite,
    pc_critic_state,
)
from auv_lowlevel.config import load_config, parse_config

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)


def _pc_config(
    profile: str = "clean",
    *,
    current_est_lpf_time_basis: str | None = None,
):
    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["experiment"].update(
        {
            "name": "pc_sensor_test",
            "arm": "PC",
            "policy_mode": "pure_controller",
            "sensor_source": "synthetic",
            "pc_stage": "C3",
        }
    )
    data["plant"]["backend"] = "batch_torch_remus100"
    data["current"]["enabled"] = True
    data["current"]["speed_mps"] = [0.0, 0.0]
    data["current"]["direction_rad"] = [0.0, 0.0]
    data["guidance"]["compensate_current"] = False
    data["base_controller"]["use_current_field"] = False
    data["observation"]["speed_feedback"] = "ground"
    data["observation"]["expose_current"] = False
    data["observation"]["current_contract"] = "hidden"
    data["sensors"] = {"profile": profile, "seed": 123}
    if current_est_lpf_time_basis is not None:
        data["sensors"]["current_est_lpf_time_basis"] = current_est_lpf_time_basis
    data["reward"].update(
        {
            "mode": "pc_command_track",
            "hold_bonus": False,
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
    }
    return parse_config(data)


def _truth(count: int, *, dtype: torch.dtype = torch.float64):
    eta = torch.tensor(
        [[1.0, 2.0, 10.0, 0.05, -0.08, 0.3]] * count,
        dtype=dtype,
    )
    eta[:, 0] += torch.arange(count, dtype=dtype)
    nu = torch.tensor(
        [[1.2, 0.1, -0.05, 0.02, -0.01, 0.04]] * count,
        dtype=dtype,
    )
    eta[:, 0] += torch.arange(count, dtype=dtype)
    return eta, nu


def test_clean_sensor_profile_is_exact_and_contains_no_true_current() -> None:
    config = _pc_config("clean")
    suite = SyntheticSensorSuite(config, num_envs=4)
    suite.reset([10, 11, 12, 13])
    eta, nu = _truth(4)
    current = torch.zeros((4, 3), dtype=torch.float64)
    output = suite.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)

    assert torch.equal(output.p_nav, eta[:, :3])
    assert torch.equal(output.attitude, eta[:, 3:6])
    assert torch.equal(output.angular_rate, nu[:, 3:6])
    assert torch.equal(output.dvl_ground_body, nu[:, :3])
    assert torch.equal(output.dvl_water_body, nu[:, :3])
    assert torch.equal(output.current_estimate_body, torch.zeros((4, 3)))
    assert torch.equal(output.bottom_lock_valid, torch.ones(4, dtype=torch.float64))
    assert torch.equal(output.current_estimate_valid, torch.ones(4, dtype=torch.float64))
    assert torch.equal(output.usbl_valid, torch.ones(4, dtype=torch.float64))
    assert not hasattr(output, "current_true_ned_mps")
    assert not hasattr(output, "p_true")


def test_dvl_ground_uses_nu_plus_current_and_water_uses_nu() -> None:
    config = _pc_config("clean")
    suite = SyntheticSensorSuite(config, num_envs=4)
    suite.reset([14, 15, 16, 17])
    eta, nu = _truth(4)
    current_ned = torch.tensor([[0.4, -0.2, 0.0]] * 4, dtype=torch.float64)
    expected_current_body = world_to_body(
        current_ned,
        quaternion_from_euler(eta[:, 3], eta[:, 4], eta[:, 5]),
    )

    output = suite.step(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=current_ned,
    )

    torch.testing.assert_close(output.dvl_water_body, nu[:, :3])
    torch.testing.assert_close(
        output.dvl_ground_body,
        nu[:, :3] + expected_current_body,
    )


def test_actor_state_uses_frozen_pc_order() -> None:
    config = _pc_config("clean")
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([100, 101])
    eta, nu = _truth(2)
    output = suite.step(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=torch.zeros((2, 3), dtype=torch.float64),
    )
    expected = torch.cat(
        [output.p_nav, output.attitude, output.angular_rate, output.dvl_ground_body],
        dim=1,
    )
    assert torch.equal(output.actor_state(), expected)
    assert output.actor_state().shape == (2, 12)


def test_delay_queue_pushes_zero_order_hold_every_control_step(monkeypatch) -> None:
    profile = replace(
        PROFILES["clean"],
        dvl_bottom_hz=(50.0, 50.0),
        dvl_bottom_delay_s=(0.04, 0.04),
    )
    monkeypatch.setitem(PROFILES, "clean", profile)
    config = _pc_config("clean")
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([200, 201])
    speeds = [0.0, 1.0, 2.0, 3.0, 4.0]
    observed: list[float] = []
    for speed in speeds:
        eta, nu = _truth(2)
        nu[:, 0] = speed
        output = suite.step(
            eta_true=eta,
            nu_true=nu,
            current_true_ned_mps=torch.zeros((2, 3), dtype=torch.float64),
        )
        observed.append(float(output.dvl_ground_body[0, 0]))

    assert observed == pytest.approx([0.0, 0.0, 0.0, 1.0, 2.0])
    assert observed[0] == pytest.approx(observed[1])
    assert observed[2] != pytest.approx(observed[3])


def test_prime_fills_delay_history_with_first_valid_sample(monkeypatch) -> None:
    profile = replace(
        PROFILES["clean"],
        nav_delay_s=(0.2, 0.2),
        dvl_bottom_delay_s=(0.2, 0.2),
        usbl_delay_s=(0.2, 0.2),
    )
    monkeypatch.setitem(PROFILES, "clean", profile)
    config = _pc_config("clean")
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([210, 211])
    eta, nu = _truth(2)

    output = suite.prime(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=torch.zeros((2, 3), dtype=torch.float64),
    )

    torch.testing.assert_close(output.p_nav, eta[:, :3])
    torch.testing.assert_close(output.dvl_ground_body, nu[:, :3])


def test_realistic_sensor_stream_is_reproducible_after_reset() -> None:
    config = _pc_config("realistic")
    suite = SyntheticSensorSuite(config, num_envs=3)
    eta, nu = _truth(3)
    current = torch.tensor([[0.2, -0.1, 0.0]] * 3, dtype=torch.float64)

    def rollout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        suite.reset([301, 302, 303])
        outputs = []
        for _ in range(12):
            outputs.append(
                suite.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)
            )
        return (
            torch.stack([item.p_nav for item in outputs]),
            torch.stack([item.actor_state() for item in outputs]),
            torch.stack([item.current_estimate_body for item in outputs]),
        )

    first = rollout()
    second = rollout()
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
    for tensor in first:
        assert torch.isfinite(tensor).all()


def test_dvl_outage_marks_invalid_and_holds_last_value(monkeypatch) -> None:
    profile = replace(
        PROFILES["clean"],
        dvl_bottom_outage_start_prob=(1.0e6, 1.0e6),
        dvl_bottom_outage_duration_s=(0.1, 0.1),
    )
    monkeypatch.setitem(PROFILES, "clean", profile)
    config = _pc_config("clean")
    suite = SyntheticSensorSuite(config, num_envs=1)
    suite.reset([401])
    eta, nu = _truth(1)
    output = suite.step(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=torch.zeros((1, 3), dtype=torch.float64),
    )
    assert output.bottom_lock_valid.item() == 0.0
    assert torch.equal(output.dvl_ground_body, torch.zeros((1, 3), dtype=torch.float64))
    assert torch.isfinite(output.actor_state()).all()


def test_realistic_outage_duty_cycle_is_not_resampled_every_control_step() -> None:
    config = _pc_config("realistic")
    suite = SyntheticSensorSuite(config, num_envs=64)
    suite.reset(list(range(900, 964)))
    eta, nu = _truth(64)
    eta[:, 0] += torch.arange(64, dtype=torch.float64)
    current = torch.zeros((64, 3), dtype=torch.float64)
    invalid = torch.zeros(64, dtype=torch.float64)
    steps = 2000
    for _ in range(steps):
        output = suite.step(
            eta_true=eta,
            nu_true=nu,
            current_true_ned_mps=current,
        )
        invalid += 1.0 - output.bottom_lock_valid
    duty_cycle = float(invalid.sum() / (64 * steps))
    assert 0.0 < duty_cycle < 0.20


def test_usbl_outlier_is_rejected_before_nav_update(monkeypatch) -> None:
    profile = replace(
        PROFILES["clean"],
        usbl_outlier_prob=(1.0, 1.0),
        usbl_outlier_jump_m=(100.0, 100.0),
        usbl_reject_outliers=True,
        nav_usbl_gain=1.0,
    )
    monkeypatch.setitem(PROFILES, "clean", profile)
    config = _pc_config("clean")
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([501, 502])
    eta, nu = _truth(2)
    output = suite.step(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=torch.zeros((2, 3), dtype=torch.float64),
    )
    assert torch.equal(output.p_nav, eta[:, :3])
    assert torch.equal(output.usbl_outlier, torch.ones(2, dtype=torch.float64))
    assert torch.equal(output.usbl_valid, torch.zeros(2, dtype=torch.float64))


def test_usbl_outlier_is_rejected_with_high_noise(monkeypatch) -> None:
    profile = replace(
        PROFILES["clean"],
        usbl_noise_m=(5.0, 5.0),
        usbl_outlier_prob=(1.0, 1.0),
        usbl_outlier_jump_m=(10.0, 10.0),
        usbl_reject_outliers=True,
        nav_usbl_gain=1.0,
    )
    monkeypatch.setitem(PROFILES, "clean", profile)
    config = _pc_config("clean")
    suite = SyntheticSensorSuite(config, num_envs=2)
    eta = torch.zeros((2, 6), dtype=torch.float64)
    nu = torch.zeros((2, 6), dtype=torch.float64)
    suite.reset([0, 1])
    output = suite.step(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=torch.zeros((2, 3), dtype=torch.float64),
    )
    assert torch.equal(output.usbl_outlier, torch.ones(2, dtype=torch.float64))
    assert torch.equal(output.usbl_valid, torch.zeros(2, dtype=torch.float64))
    assert torch.linalg.norm(output.p_nav, dim=1).max() < 5.0


def test_realistic_calibrated_profile_applies_premission_bias_calibration() -> None:
    config = _pc_config("realistic_calibrated")
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([0, 1])
    assert suite._params is not None
    expected = -0.8 * suite._params.nav_bias
    assert torch.allclose(suite._nav_event, expected)


def test_realistic_calibrated_is_the_nominal_sensor_domain() -> None:
    profile = PROFILES["realistic_calibrated"]

    assert profile.current_est_tau_s == (1.5, 3.0)
    assert profile.current_est_delay_s == (0.0, 0.3)
    assert profile.current_est_hz == (1.0, 3.0)
    assert profile.current_est_noise_mps == (0.005, 0.02)
    assert profile.current_est_bias_mps == (0.005, 0.02)
    assert profile.current_est_outage_duration_s == (1.0, 3.0)
    assert profile.nav_delay_s == (0.02, 0.15)
    assert profile.nav_bias_m == (0.1, 0.5)
    assert profile.usbl_noise_m == (0.3, 1.5)
    assert profile.usbl_delay_s == (0.1, 0.3)
    assert profile.dvl_bottom_outage_duration_s == (1.0, 3.0)
    assert profile.usbl_outlier_jump_m == (5.0, 10.0)
    assert profile.usbl_outlier_prob == (0.005, 0.01)
    assert profile.nav_usbl_gain == 0.03
    assert profile.nav_bias_calibration_fraction == 0.8


def test_legacy_calibrated_and_stress_profiles_keep_old_extreme_ranges() -> None:
    for name in ("realistic_calibrated_legacy", "realistic_stress"):
        profile = PROFILES[name]
        assert profile.current_est_tau_s == (1.0, 10.0)
        assert profile.current_est_delay_s == (0.0, 2.0)
        assert profile.current_est_hz == (0.1, 5.0)
        assert profile.current_est_noise_mps == (0.01, 0.05)
        assert profile.current_est_bias_mps == (0.01, 0.03)
        assert profile.current_est_outage_duration_s == (1.0, 10.0)
        assert profile.nav_delay_s == (0.05, 0.5)
        assert profile.nav_bias_m == (0.2, 2.0)
        assert profile.usbl_noise_m == (0.5, 5.0)
        assert profile.usbl_delay_s == (0.1, 0.5)
        assert profile.dvl_bottom_outage_duration_s == (1.0, 10.0)
        assert profile.usbl_outlier_jump_m == (10.0, 25.0)
        assert profile.usbl_outlier_prob == (0.01, 0.03)


def test_sensor_profile_entry_configs_are_explicit() -> None:
    root = Path(__file__).resolve().parents[1]
    nominal = load_config(
        root / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_20261107.json"
    )
    strict = load_config(
        root
        / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_strict125_20261107.json"
    )
    stress = load_config(
        root / "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_stress_20261107.json"
    )

    assert nominal.sensors.profile == "realistic_calibrated"
    assert strict.sensors.profile == "realistic_calibrated"
    assert stress.sensors.profile == "realistic_stress"
    assert nominal.guidance.current_estimate_max_age_s == 0.9
    assert strict.guidance.current_estimate_max_age_s == 0.9
    assert stress.guidance.current_estimate_max_age_s == 0.6
    assert nominal.plant.goal_radius_m == 2.0
    assert strict.plant.goal_radius_m == 1.25
    assert stress.plant.goal_radius_m == 2.0


def test_sensor_ages_are_finite_and_non_negative() -> None:
    config = _pc_config("realistic")
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([601, 602])
    eta, nu = _truth(2)
    output = suite.step(
        eta_true=eta,
        nu_true=nu,
        current_true_ned_mps=torch.zeros((2, 3), dtype=torch.float64),
    )
    ages = torch.stack(
        [
            output.nav_age_s,
            output.dvl_bottom_age_s,
            output.dvl_water_age_s,
            output.usbl_age_s,
            output.current_estimate_age_s,
        ],
        dim=1,
    )
    assert torch.isfinite(ages).all()
    assert (ages >= 0.0).all()


def _constant_current_estimate_ratio(time_basis: str) -> float:
    config = _pc_config(
        "clean",
        current_est_lpf_time_basis=time_basis,
    )
    suite = SyntheticSensorSuite(config, num_envs=1)
    suite.profile = replace(
        PROFILES["clean"],
        current_est_hz=(1.0, 1.0),
        current_est_tau_s=(3.0, 3.0),
    )
    suite.reset([901])
    eta = torch.zeros((1, 6), dtype=torch.float64)
    nu = torch.zeros((1, 6), dtype=torch.float64)
    current_ned = torch.tensor([[0.4, 0.2, 0.0]], dtype=torch.float64)
    for _ in range(2500):
        output = suite.step(
            eta_true=eta,
            nu_true=nu,
            current_true_ned_mps=current_ned,
        )
    return float(torch.linalg.norm(output.current_estimate_body[0]) / torch.linalg.norm(current_ned[0]))


def test_current_lpf_time_basis_controls_low_rate_attenuation() -> None:
    dt = 0.02
    tau_s = 3.0
    samples = 50
    legacy_ratio = _constant_current_estimate_ratio("control_dt_legacy")
    corrected_ratio = _constant_current_estimate_ratio("elapsed_sample")

    alpha_legacy = 1.0 - math.exp(-dt / tau_s)
    alpha_corrected = 1.0 - math.exp(-1.0 / tau_s)
    expected_legacy = 1.0 - (1.0 - alpha_legacy) ** samples
    expected_corrected = 1.0 - (1.0 - alpha_corrected) ** samples

    assert legacy_ratio == pytest.approx(expected_legacy, abs=1e-9)
    assert corrected_ratio == pytest.approx(expected_corrected, abs=1e-9)
    assert legacy_ratio < 0.30
    assert corrected_ratio > 0.99


def test_current_lpf_time_basis_defaults_to_legacy_and_rejects_unknown() -> None:
    assert (
        _pc_config("clean").sensors.current_est_lpf_time_basis
        == "control_dt_legacy"
    )
    assert (
        _pc_config(
            "clean",
            current_est_lpf_time_basis="elapsed_sample",
        ).sensors.current_est_lpf_time_basis
        == "elapsed_sample"
    )

    data = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    data["sensors"] = {
        "profile": "clean",
        "current_est_lpf_time_basis": "unknown",
    }
    with pytest.raises(ValueError, match="current_est_lpf_time_basis"):
        parse_config(data)


def test_critic_state_order_and_errors() -> None:
    eta, nu = _truth(3)
    heading_error = torch.full((3,), math.radians(5.0), dtype=torch.float64)
    pitch_error = torch.full((3,), math.radians(-2.0), dtype=torch.float64)
    speed_error = torch.full((3,), 0.15, dtype=torch.float64)
    critic = pc_critic_state(
        eta=eta,
        nu=nu,
        heading_error_rad=heading_error,
        pitch_error_rad=pitch_error,
        speed_error_mps=speed_error,
    )
    assert critic.shape == (3, 15)
    assert torch.equal(critic[:, :6], eta)
    assert torch.equal(critic[:, 6:9], nu[:, :3])
    assert torch.equal(critic[:, 9:12], nu[:, 3:])
    assert torch.equal(critic[:, 12], heading_error)
    assert torch.equal(critic[:, 13], pitch_error)
    assert torch.equal(critic[:, 14], speed_error)


def test_reset_and_prime_rows_preserve_non_reset_environments() -> None:
    config = _pc_config("realistic")
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([701, 702])
    eta, nu = _truth(2)
    current = torch.tensor([[0.2, -0.1, 0.0], [0.2, -0.1, 0.0]], dtype=torch.float64)
    previous = suite.prime(eta_true=eta, nu_true=nu, current_true_ned_mps=current)
    for _ in range(6):
        previous = suite.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)

    mask = torch.tensor([True, False])
    suite.reset_rows(mask, seeds=[801])
    eta_new = eta.clone()
    eta_new[0, :3] += torch.tensor([5.0, -2.0, 1.0], dtype=torch.float64)
    primed = suite.prime_rows(
        mask,
        eta_true=eta_new,
        nu_true=nu,
        current_true_ned_mps=current,
        previous=previous,
    )

    for field in (
        "p_nav",
        "attitude",
        "angular_rate",
        "dvl_ground_body",
        "dvl_water_body",
        "current_estimate_body",
        "usbl_position",
        "bottom_lock_valid",
        "water_track_valid",
        "current_estimate_valid",
        "usbl_valid",
        "usbl_outlier",
        "nav_age_s",
        "dvl_bottom_age_s",
        "dvl_water_age_s",
        "usbl_age_s",
        "current_estimate_age_s",
    ):
        assert torch.equal(getattr(primed, field)[1], getattr(previous, field)[1]), field
    assert torch.isfinite(primed.actor_state()).all()
    assert not torch.equal(primed.p_nav[0], previous.p_nav[0])
