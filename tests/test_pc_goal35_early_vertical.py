from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/analyze_pc_goal35_early_vertical.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal35_early_vertical", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _initial_record(
    seed: int,
    *,
    initial: float,
    terminal: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "vertical_miss": True,
        "initial_abs_vertical_m": initial,
        "terminal_abs_vertical_m": terminal,
    }


def _track_record(
    seed: int,
    *,
    command_pitch: float,
    tracking_error: float,
    reference_match: float,
    terminal: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "vertical_miss": True,
        "median_abs_command_pitch_rad": command_pitch,
        "median_abs_tracking_error_rad": tracking_error,
        "reference_sign_match_fraction": reference_match,
        "actual_sign_match_fraction": reference_match,
        "terminal_abs_vertical_m": terminal,
    }


def _handoff_record(
    seed: int,
    *,
    jump: float,
    worsening: float,
    terminal: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "vertical_miss": True,
        "pitch_command_jump_rad": jump,
        "abs_pitch_command_jump_rad": abs(jump),
        "vertical_worsening_m": worsening,
        "closure_drop": 0.1,
        "terminal_abs_vertical_m": terminal,
    }


def _block(initial_pass: bool, track_pass: bool, handoff_pass: bool) -> dict[str, object]:
    return {
        "h_initial": {"passes_all": initial_pass},
        "h_track": {"passes_all": track_pass},
        "h_handoff": {"passes_all": handoff_pass},
    }


def test_initial_metrics_reads_first_row() -> None:
    trace = {"time_s": np.asarray([0.0, 1.0, 2.0])}
    vertical = np.asarray([2.0, 1.0, 0.0])
    distance = np.asarray([34.0, 20.0, 5.0])

    result = MODULE._initial_metrics(
        trace, np.arange(3), vertical, distance, terminal_abs_vertical_m=0.5
    )

    assert result["initial_time_s"] == 0.0
    assert result["initial_distance_to_goal_m"] == 34.0
    assert result["initial_abs_vertical_m"] == 2.0
    assert result["terminal_abs_vertical_m"] == 0.5


def test_pitch_tracking_metrics_uses_40_to_20m_segment() -> None:
    count = 21
    distance = np.linspace(45.0, 5.0, count)
    trace = {
        "command": np.tile(np.asarray([[0.0, -0.1, 1.0]]), (count, 1)),
        "eta": np.zeros((count, 6), dtype=np.float64),
    }
    trace["eta"][:, 4] = -0.05
    vertical = np.ones(count, dtype=np.float64)

    result = MODULE._pitch_tracking_metrics(
        trace, np.arange(count), vertical, distance
    )

    assert result["early_segment_sample_count"] >= 10
    assert result["median_abs_command_pitch_rad"] == pytest.approx(0.1)
    assert result["median_abs_tracking_error_rad"] == pytest.approx(0.05)
    assert result["reference_sign_match_fraction"] == 1.0
    assert result["actual_sign_match_fraction"] == 1.0


def test_handoff_metrics_compares_pre_and_post_windows() -> None:
    count = 101
    time_s = np.arange(count, dtype=np.float64) * 0.1
    command = np.zeros((count, 3), dtype=np.float64)
    command[time_s < 5.0, 1] = -0.1
    command[time_s >= 5.0, 1] = -0.2
    trace = {
        "time_s": time_s,
        "command": command,
        "guidance_terminal_phase": time_s >= 5.0,
    }
    vertical = np.where(time_s < 5.0, 1.0, 2.0)
    velocity = np.full(count, -0.1)
    distance = np.full(count, 20.0)

    result = MODULE._handoff_metrics(
        trace, np.arange(count), vertical, velocity, distance
    )

    assert result["pitch_command_jump_rad"] == pytest.approx(-0.1)
    assert result["vertical_worsening_m"] == pytest.approx(1.0)
    assert result["closure_drop"] == 0.0


def test_h_initial_supports_initial_offset_dominance() -> None:
    records = [
        _initial_record(seed, initial=3.0 + 0.1 * index, terminal=2.0 + 0.1 * index)
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_initial(records, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["initial_terminal_rho"] == pytest.approx(1.0)
    assert result["initial_outside_radius_fraction"] == 1.0


def test_h_initial_fails_when_initial_error_is_inside_radius() -> None:
    records = [
        _initial_record(seed, initial=0.5, terminal=3.0 + 0.1 * index)
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_initial(records, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["criteria"]["initial_outside_radius_fraction_ge_0_60"] is False


def test_h_track_supports_reference_correct_but_tracking_error() -> None:
    records = [
        _track_record(
            seed,
            command_pitch=0.12,
            tracking_error=0.05 + 0.01 * index,
            reference_match=0.80,
            terminal=2.0 + 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_track(records)

    assert result["passes_all"] is True
    assert result["tracking_terminal_rho"] == pytest.approx(1.0)
    assert result["high_minus_low_terminal_abs_m"] > 0.25


def test_h_track_fails_when_reference_direction_is_wrong() -> None:
    records = [
        _track_record(
            seed,
            command_pitch=0.12,
            tracking_error=0.10,
            reference_match=0.40,
            terminal=2.0 + 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_track(records)

    assert result["passes_all"] is False
    assert result["criteria"]["reference_sign_match_ge_0_60"] is False


def test_h_handoff_supports_discontinuity_and_terminal_ranking() -> None:
    records = [
        _handoff_record(
            seed,
            jump=0.06 + 0.01 * index,
            worsening=0.10 + 0.02 * index,
            terminal=2.0 + 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_handoff(records)

    assert result["passes_all"] is True
    assert result["worsening_terminal_rho"] == pytest.approx(1.0)
    assert result["high_minus_low_terminal_abs_m"] > 0.25


def test_h_handoff_fails_without_vertical_worsening() -> None:
    records = [
        _handoff_record(
            seed,
            jump=0.10,
            worsening=0.0,
            terminal=2.0 + 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_handoff(records)

    assert result["passes_all"] is False
    assert result["criteria"]["worsening_or_closure_drop"] is True
    assert result["criteria"]["worsening_terminal_rho_ge_0_30"] is False


@pytest.mark.parametrize(
    ("initial_pass", "track_pass", "handoff_pass", "expected"),
    (
        (True, False, False, "INITIAL_OFFSET_DOMINANT"),
        (False, True, False, "PITCH_TRACKING_SUPPORTED"),
        (False, False, True, "HANDOFF_TRANSITION_SUPPORTED"),
        (True, True, False, "MULTI_FACTOR"),
        (False, False, False, "NO_SUPPORTED_MECHANISM"),
    ),
)
def test_overall_verdict_mapping(
    initial_pass: bool,
    track_pass: bool,
    handoff_pass: bool,
    expected: str,
) -> None:
    blocks = [_block(initial_pass, track_pass, handoff_pass) for _ in range(4)]

    result = MODULE._overall_verdict(blocks, invalid=False)

    assert result["decision"] == expected


def test_overall_verdict_reserves_invalid_diagnostic() -> None:
    result = MODULE._overall_verdict([], invalid=True)

    assert result["decision"] == "INVALID_DIAGNOSTIC"
