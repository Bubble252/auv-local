from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/analyze_pc_goal30_failure.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal30_failure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(
    *,
    terminal_phase_fraction: float,
    yaw: float,
    saturation: float,
    offset: float,
    three_d_rmse: float,
    vertical_rmse: float,
) -> dict[str, float]:
    return {
        "terminal_phase_fraction": terminal_phase_fraction,
        "terminal_yaw_error_abs_mean_rad": yaw,
        "terminal_material_saturation_rate": saturation,
        "terminal_rudder_abs_mean_rad": 0.1,
        "terminal_stern_abs_mean_rad": 0.1,
        "terminal_eta_offset_m": offset,
        "pre_goal_3d_rmse_m": three_d_rmse,
        "path_three_d_rmse_m": three_d_rmse,
        "path_horizontal_rmse_m": 1.0,
        "path_vertical_rmse_m": vertical_rmse,
        "miss_distance_at_plane_m": offset,
    }


def test_paired_reports_window_path_and_terminal_deltas() -> None:
    baseline = {
        20: _row(
            terminal_phase_fraction=0.5,
            yaw=0.2,
            saturation=0.2,
            offset=1.0,
            three_d_rmse=2.0,
            vertical_rmse=1.5,
        ),
        21: _row(
            terminal_phase_fraction=0.5,
            yaw=0.2,
            saturation=0.2,
            offset=1.0,
            three_d_rmse=2.0,
            vertical_rmse=1.5,
        ),
    }
    candidate = {
        20: _row(
            terminal_phase_fraction=0.8,
            yaw=0.25,
            saturation=0.1,
            offset=1.2,
            three_d_rmse=2.4,
            vertical_rmse=1.9,
        ),
        21: _row(
            terminal_phase_fraction=0.8,
            yaw=0.15,
            saturation=0.3,
            offset=0.8,
            three_d_rmse=2.4,
            vertical_rmse=1.9,
        ),
    }

    paired = MODULE._paired(baseline, candidate)["numeric_delta"]

    assert paired["terminal_phase_fraction"]["median"] == pytest.approx(0.3)
    assert paired["path_three_d_rmse_m"]["median"] == pytest.approx(0.4)
    assert paired["path_vertical_rmse_m"]["median"] == pytest.approx(0.4)
    assert paired["terminal_eta_offset_m"]["median"] == pytest.approx(0.0)
    assert paired["terminal_material_saturation_rate"]["median"] == pytest.approx(0.0)


def test_yaw_offset_correlation_is_finite() -> None:
    rows = {
        20: _row(
            terminal_phase_fraction=0.5,
            yaw=0.1,
            saturation=0.1,
            offset=0.5,
            three_d_rmse=1.0,
            vertical_rmse=0.5,
        ),
        21: _row(
            terminal_phase_fraction=0.5,
            yaw=0.2,
            saturation=0.1,
            offset=1.5,
            three_d_rmse=1.0,
            vertical_rmse=0.5,
        ),
    }

    correlation = MODULE._correlations(rows)["yaw_vs_terminal_eta_offset"]

    assert correlation is not None
    assert 0.99 <= correlation <= 1.0
