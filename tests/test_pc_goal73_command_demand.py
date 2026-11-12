from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal73_command_demand.py"
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal73_command_demand", ANALYSIS_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def _rows(steps: int = 120) -> dict[str, np.ndarray]:
    times = np.arange(steps, dtype=np.float64) * 0.02
    signed = np.full(steps, -1.0, dtype=np.float64)
    signed[80:] = 0.01
    command = np.zeros((steps, 3), dtype=np.float64)
    command[:, 0] = np.linspace(0.0, 0.8, steps)
    command[:, 1] = np.linspace(0.0, -0.2, steps)
    command[:, 2] = 1.2
    nu = np.zeros((steps, 6), dtype=np.float64)
    nu[:, 0] = 1.5
    nu[:, 5] = 0.15
    return {
        "time_s": times,
        "signed_offset": signed,
        "command_physical": command,
        "nu": nu,
        "material_saturated": np.zeros(steps, dtype=bool),
        "guidance_speed_scale": np.ones(steps, dtype=np.float64),
        "guidance_crab_angle_rad": np.full(steps, 0.1, dtype=np.float64),
    }


def test_window_metrics_computes_rates_and_curvature() -> None:
    metrics = ANALYSIS._window_metrics(_rows(), 1.0)
    assert metrics["heading_rate_p90_rad_s"] is not None
    assert metrics["heading_rate_p90_rad_s"] > 0.0
    assert metrics["pitch_rate_p90_rad_s"] is not None
    assert metrics["material_saturation_fraction"] == 0.0
    assert metrics["mean_command_speed_mps"] == pytest.approx(1.2)
    assert metrics["mean_abs_curvature_1_per_m"] == pytest.approx(0.15 / 1.5)
    assert math.isclose(metrics["mean_abs_crab_angle_rad"], 0.1)


def test_h73a_heading_gap_supports_default() -> None:
    result = ANALYSIS._decide_h73a(
        "default",
        {
            "heading_rate_p90_rad_s": 0.11,
            "pitch_rate_p90_rad_s": 0.0,
            "material_saturation_fraction": 0.0,
            "mean_abs_curvature_1_per_m": 0.0,
        },
        8,
    )
    assert result["supported"] is True
    assert result["checks"]["heading"] is True


def test_h73a_requires_minimum_samples() -> None:
    result = ANALYSIS._decide_h73a(
        "strict125",
        {
            "heading_rate_p90_rad_s": 0.06,
            "pitch_rate_p90_rad_s": 0.03,
            "material_saturation_fraction": 0.03,
            "mean_abs_curvature_1_per_m": 0.02,
        },
        2,
    )
    assert result["supported"] is False


def test_h73a_strict125_uses_half_thresholds() -> None:
    result = ANALYSIS._decide_h73a(
        "strict125",
        {
            "heading_rate_p90_rad_s": 0.05,
            "pitch_rate_p90_rad_s": 0.0,
            "material_saturation_fraction": 0.0,
            "mean_abs_curvature_1_per_m": 0.0,
        },
        3,
    )
    assert result["supported"] is True


def test_metric_gap_uses_group_medians() -> None:
    newly = [{"x": 1.0}, {"x": 3.0}, {"x": 5.0}]
    both = [{"x": 0.0}, {"x": 1.0}, {"x": 1.0}]
    assert ANALYSIS._metric_gap(newly, both, "x") == 2.0


def test_aggregate_reports_group_counts_and_gap() -> None:
    records = []
    for name, value in (
        ("newly_lost", 0.3),
        ("newly_lost", 0.4),
        ("newly_lost", 0.5),
        ("both_true", 0.1),
    ):
        record = {"group": name}
        for delta in (1.0, 2.0):
            tag = f"{delta:g}s"
            record[f"limiter_on_{tag}_heading_rate_p90_rad_s"] = value
            record[f"limiter_on_{tag}_pitch_rate_p90_rad_s"] = value
            record[f"limiter_on_{tag}_material_saturation_fraction"] = value
            record[f"limiter_on_{tag}_mean_command_speed_mps"] = value
            record[f"limiter_on_{tag}_mean_guidance_speed_scale"] = value
            record[f"limiter_on_{tag}_mean_abs_curvature_1_per_m"] = value
            record[f"limiter_on_{tag}_mean_abs_crab_angle_rad"] = value
        records.append(record)
    result = ANALYSIS._aggregate(records, "default")
    assert result["groups"]["newly_lost"]["n"] == 3
    assert result["groups"]["both_true"]["n"] == 1
    assert result["gaps_newly_lost_minus_both_true"][
        "limiter_on_1s_heading_rate_p90_rad_s"
    ] == pytest.approx(0.3)
    assert result["h73a"]["supported"] is True
