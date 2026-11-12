from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/analyze_pc_goal34_global_vertical.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal34_global_vertical", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _global_record(
    seed: int,
    *,
    abs40: float,
    abs20: float,
    terminal: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "vertical_miss": True,
        "abs_vertical_40m": abs40,
        "abs_vertical_20m": abs20,
        "terminal_abs_vertical_m": terminal,
    }


def _phase_record(
    seed: int,
    *,
    sign_fraction: float,
    terminal: float,
    latency: float | None,
) -> dict[str, object]:
    return {
        "seed": seed,
        "vertical_miss": True,
        "phase_sign_matched_fraction": sign_fraction,
        "terminal_abs_vertical_m": terminal,
        "phase_active_sample_count": 20,
        "phase_reversal_latency_s": latency,
    }


def _slope_record(
    seed: int,
    *,
    slope: float,
    abs40: float,
    abs20: float,
    terminal: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "vertical_miss": True,
        "path_vertical_slope": slope,
        "abs_vertical_40m": abs40,
        "abs_vertical_20m": abs20,
        "terminal_abs_vertical_m": terminal,
    }


def _block(global_pass: bool, phase_pass: bool, slope_pass: bool) -> dict[str, object]:
    return {
        "h_global": {"passes_all": global_pass},
        "h_phase": {"passes_all": phase_pass},
        "h_slope": {"passes_all": slope_pass},
    }


def test_crossing_metrics_uses_first_threshold_crossing() -> None:
    count = 51
    time_s = np.linspace(0.0, 1.0, count)
    distance = 45.0 - 35.0 * time_s
    vertical = 1.0 + time_s
    vertical_velocity = -np.ones(count, dtype=np.float64)
    trace = {"time_s": time_s}

    result = MODULE._crossing_metrics(
        trace, np.arange(count), distance, vertical, vertical_velocity, 40.0
    )

    assert result["distance_m"] <= 40.0
    assert result["signed_vertical_m"] > 1.0
    assert result["sign_matched_fraction"] == 1.0
    assert result["window_sample_count"] == count


def test_reversal_latency_measures_time_to_sign_matched_velocity() -> None:
    time_s = np.arange(0.0, 5.01, 0.1)
    error = np.where(time_s < 1.0, 1.0, -1.0)
    velocity = np.where(time_s < 3.0, 0.0, 0.2)

    latency, count = MODULE._reversal_latency(time_s, error, velocity)

    assert count == 1
    assert latency == pytest.approx(2.0)


def test_reversal_latency_returns_none_without_sign_change() -> None:
    time_s = np.arange(0.0, 2.01, 0.1)
    error = np.ones(time_s.size, dtype=np.float64)
    velocity = -np.ones(time_s.size, dtype=np.float64)

    latency, count = MODULE._reversal_latency(time_s, error, velocity)

    assert latency is None
    assert count == 0


def test_phase_metrics_collects_active_sample_sign_fraction() -> None:
    count = 101
    time_s = np.arange(count, dtype=np.float64) * 0.1
    eta = np.zeros((count, 6), dtype=np.float64)
    eta[:, 0] = np.linspace(-10.0, 10.0, count)
    trace = {
        "time_s": time_s,
        "eta": eta,
        "goal_ned_m": np.zeros((count, 3), dtype=np.float64),
        "goal_tangent_ned": np.tile(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), (count, 1)
        ),
        "guidance_terminal_phase": time_s >= 3.0,
    }
    vertical = np.ones(count, dtype=np.float64)
    vertical_velocity = -np.full(count, 0.1, dtype=np.float64)

    result = MODULE._phase_metrics(
        trace, np.arange(count), vertical, vertical_velocity
    )

    assert result["terminal_abs_vertical_m"] == 1.0
    assert result["phase_active_sample_count"] > 10
    assert result["phase_sign_matched_fraction"] == 1.0
    assert result["phase_reversal_latency_s"] is None


def test_path_slope_metrics_uses_40_to_20m_segment() -> None:
    count = 101
    distance = np.linspace(45.0, 10.0, count)
    trace = {
        "route_tangent_ned": np.tile(
            np.asarray([[1.0, 0.0, 0.05]], dtype=np.float64), (count, 1)
        ),
        "goal_tangent_ned": np.tile(
            np.asarray([[1.0, 0.0, 0.2]], dtype=np.float64), (count, 1)
        ),
    }

    result = MODULE._path_slope_metrics(trace, np.arange(count), distance)

    assert result["path_vertical_slope"] == pytest.approx(0.05)
    assert result["goal_vertical_slope"] == pytest.approx(0.2)
    assert result["path_slope_sample_count"] > 10


def test_h_global_supports_early_carryover() -> None:
    records = [
        _global_record(
            seed,
            abs40=2.5 + 0.2 * index,
            abs20=3.0 + 0.2 * index,
            terminal=2.0 + 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_global(records, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["rho40"] == pytest.approx(1.0)
    assert result["ratio20"] > 1.0


def test_h_global_fails_when_error_is_not_established_at_40m() -> None:
    records = [
        _global_record(
            seed,
            abs40=0.1,
            abs20=2.0 + 0.2 * index,
            terminal=2.5 + 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_global(records, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["criteria"]["median_40_ge_half_radius"] is False


def test_h_phase_supports_long_latency_and_low_sign_fraction() -> None:
    records = [
        _phase_record(
            seed,
            sign_fraction=0.50 - 0.02 * index,
            terminal=3.0 + 0.2 * index,
            latency=1.0 + 0.1 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_phase(records)

    assert result["passes_all"] is True
    assert result["reversal_coverage"] == 1.0
    assert result["reversal_latency_median_s"] >= 1.0


def test_h_phase_fails_when_reversals_are_not_observed() -> None:
    records = [
        _phase_record(
            seed,
            sign_fraction=0.40,
            terminal=3.0 + 0.2 * index,
            latency=None,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_phase(records)

    assert result["passes_all"] is False
    assert result["criteria"]["reversal_coverage_and_latency"] is False


def test_h_slope_supports_slope_conditioned_vertical_error() -> None:
    records = [
        _slope_record(
            seed,
            slope=0.02 + 0.01 * index,
            abs40=1.0 + 0.2 * index,
            abs20=1.5 + 0.2 * index,
            terminal=2.0 + 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_slope(records)

    assert result["passes_all"] is True
    assert result["slope_vertical40_rho"] == pytest.approx(1.0)
    assert result["high_minus_low_abs_vertical_20m"] > 0.2


def test_h_slope_fails_without_slope_variation() -> None:
    records = [
        _slope_record(
            seed,
            slope=0.05,
            abs40=1.0 + 0.2 * index,
            abs20=1.5 + 0.2 * index,
            terminal=2.0 + 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_slope(records)

    assert result["passes_all"] is False
    assert result["criteria"]["slope_vertical40_rho_ge_0_30"] is False


@pytest.mark.parametrize(
    ("global_pass", "phase_pass", "slope_pass", "expected"),
    (
        (True, False, False, "GLOBAL_CARRYOVER_SUPPORTED"),
        (False, True, False, "PHASE_LAG_SUPPORTED"),
        (False, False, True, "VERTICAL_SLOPE_SUPPORTED"),
        (True, True, False, "MULTI_FACTOR"),
        (False, False, False, "NO_SUPPORTED_MECHANISM"),
    ),
)
def test_overall_verdict_mapping(
    global_pass: bool,
    phase_pass: bool,
    slope_pass: bool,
    expected: str,
) -> None:
    blocks = [_block(global_pass, phase_pass, slope_pass) for _ in range(4)]

    result = MODULE._overall_verdict(blocks, invalid=False)

    assert result["decision"] == expected


def test_overall_verdict_reserves_invalid_diagnostic() -> None:
    result = MODULE._overall_verdict([], invalid=True)

    assert result["decision"] == "INVALID_DIAGNOSTIC"
