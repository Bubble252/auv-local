from __future__ import annotations

from pathlib import Path

import pytest

from auv_lowlevel.batch.pc_calibration import calibrate_pc20_high_speed
from auv_lowlevel.config import load_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "lowlevel"


def test_h1_calibration_accounts_for_worst_case_current() -> None:
    path = CONFIG_ROOT / "odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json"
    config = load_config(path)
    report = calibrate_pc20_high_speed(
        config,
        config_path=path,
        settle_seconds=20.0,
        window_seconds=5.0,
        step_seconds=1.0,
        rpm_points=6,
    )

    assert report["current_envelope"]["worst_case_uniform_current_mps"] == pytest.approx(0.5)
    assert report["reachability"]["command_high_mps"] == pytest.approx(2.0)
    assert report["reachability"]["passed"] is True
    assert report["reachability"]["margin_mps"] > 0.0
    assert len(report["rpm_steady_speed_curve"]) == 6
    assert len(report["rudder_step_response"]) == 2
    assert report["rudder_step_response"][-1]["positive_fin"]["turn_radius_m"] is not None


def test_h4_static_envelope_reaches_full_command() -> None:
    path = CONFIG_ROOT / "odarrl_control_pc_c3_random_h4_speed_2p5_2p5.json"
    config = load_config(path)
    report = calibrate_pc20_high_speed(
        config,
        config_path=path,
        settle_seconds=20.0,
        window_seconds=5.0,
        step_seconds=1.0,
        rpm_points=6,
    )

    assert report["reachability"]["static_max_ground_speed_mps"] >= 2.5
    assert report["reachability"]["passed"] is True


def test_h1_directional_current_sweep_exposes_head_current_boundary() -> None:
    path = CONFIG_ROOT / "odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json"
    config = load_config(path)
    report = calibrate_pc20_high_speed(
        config,
        config_path=path,
        settle_seconds=20.0,
        window_seconds=5.0,
        step_seconds=1.0,
        rpm_points=6,
        current_sweep_mps=(0.0, 0.5),
        current_directions_deg=(0.0, 180.0),
    )

    rows = report["current_reachability_sweep"]
    tail = next(
        row
        for row in rows
        if row["command_speed_mps"] == pytest.approx(2.0)
        and row["current_magnitude_mps"] == pytest.approx(0.5)
        and row["current_direction_deg"] == pytest.approx(0.0)
    )
    head = next(
        row
        for row in rows
        if row["command_speed_mps"] == pytest.approx(2.0)
        and row["current_magnitude_mps"] == pytest.approx(0.5)
        and row["current_direction_deg"] == pytest.approx(180.0)
    )
    assert tail["max_rpm_forward_ground_speed_mps"] > head["max_rpm_forward_ground_speed_mps"]
    assert head["reachable_at_max_rpm"] is True
    assert head["forward_margin_mps"] == pytest.approx(
        report["reachability"]["margin_mps"], abs=1e-5
    )


def test_coastdown_reports_stopping_distance_and_time() -> None:
    path = CONFIG_ROOT / "odarrl_control_pc_c3_random_h4_speed_2p5_2p5.json"
    config = load_config(path)
    report = calibrate_pc20_high_speed(
        config,
        config_path=path,
        settle_seconds=20.0,
        window_seconds=5.0,
        step_seconds=1.0,
        rpm_points=6,
        current_sweep_mps=(0.0,),
        current_directions_deg=(0.0,),
        coastdown_seconds=30.0,
    )

    row = report["coastdown_response"][-1]
    assert row["initial_ground_speed_mps"] > 2.2
    assert row["coastdown_distance_to_threshold_m"] > 0.0
    assert row["coastdown_time_to_threshold_s"] > 0.0
    assert row["final_ground_speed_mps"] <= row["initial_ground_speed_mps"]
