from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = (
    REPO_ROOT / "scripts" / "analyze_pc_goal69_precrossing_tracking_channels.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal69_precrossing_tracking_channels", ANALYSIS_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


class _Trace(dict[str, np.ndarray]):
    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self.keys())


def _rows(steps: int = 200) -> dict[str, np.ndarray]:
    times = np.arange(steps, dtype=np.float64) * 0.02
    crossing = min(100, steps - 1)
    distance = np.empty(steps, dtype=np.float64)
    distance[:crossing] = np.linspace(2.2, 1.7, crossing)
    distance[crossing:] = 1.7 + np.linspace(0.0, 0.5, steps - crossing)
    eta = np.zeros((steps, 3), dtype=np.float64)
    eta[:, 0] = distance
    p_nav = eta.copy()
    p_nav[:, 0] += 0.20
    signed = np.full(steps, -1.0, dtype=np.float64)
    signed[crossing:] = 0.01
    command = np.tile(np.asarray([math.pi, 0.0, 2.0]), (steps, 1))
    velocity = np.tile(np.asarray([-1.5, 0.0, 0.0]), (steps, 1))
    eta_rpy = np.zeros((steps, 3), dtype=np.float64)
    eta_rpy[:, 2] = math.pi
    return {
        "step": np.arange(steps, dtype=np.int64),
        "done": np.r_[np.zeros(steps - 1, dtype=bool), True],
        "time_s": times,
        "signed_offset": signed,
        "eta_ned": eta,
        "p_nav_ned": p_nav,
        "goal_ned": np.zeros((steps, 3), dtype=np.float64),
        "eta_rpy": eta_rpy,
        "nu": np.tile(np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (steps, 1)),
        "ground_velocity_body_mps": velocity.copy(),
        "ground_velocity_ned_mps": velocity,
        "actual_actuator": np.zeros((steps, 3), dtype=np.float64),
        "command_physical": command,
        "material_saturated": np.zeros(steps, dtype=bool),
        "residual": np.zeros((steps, 3), dtype=np.float64),
        "base_command": command.copy(),
        "gate": np.ones(steps, dtype=np.float64),
        "scale_clipped": np.zeros((steps, 3), dtype=bool),
        "rate_clipped": np.zeros((steps, 3), dtype=bool),
    }


def _group(**overrides: object) -> dict[str, object]:
    group: dict[str, object] = {
        "n": 5,
        "abs_heading_error_median_rad": 0.05,
        "abs_pitch_error_median_rad": 0.02,
        "abs_speed_error_median_mps": 0.05,
        "command_goal_cosine_median": 0.99,
        "actual_command_cosine_median": 0.95,
        "actual_ground_speed_median_mps": 2.0,
        "eta_tight_before_crossing_fraction": 0.0,
        "p_nav_tight_before_crossing_fraction": 0.0,
        "eta_tight_but_p_nav_not_fraction": 0.0,
        "eta_tight_p_nav_minus_eta_distance_median_m": None,
    }
    group.update(overrides)
    return group


def _hypotheses(
    *,
    lost: dict[str, object] | None = None,
    retained: dict[str, object] | None = None,
    consistency: dict[str, object] | None = None,
    scenario: str = "default",
) -> dict[str, object]:
    primary_lost = _group(
        abs_heading_error_median_rad=0.30,
        actual_command_cosine_median=0.70,
    ) if lost is None else lost
    primary_retained = _group() if retained is None else retained
    consistency_lost = (
        _group(abs_heading_error_median_rad=0.30)
        if consistency is None
        else consistency
    )
    return ANALYSIS._hypotheses(
        primary_lost,
        primary_retained,
        consistency_lost,
        scenario=scenario,
    )


def test_crossing_and_primary_window_stop_before_future_rows() -> None:
    rows = _rows()
    assert ANALYSIS._crossing_index(rows) == 100
    window, crossing = ANALYSIS._window_slice(rows, delta_s=1.0)
    assert crossing == 100
    assert window.stop == crossing + 1
    assert window.start == 50
    metrics = ANALYSIS._window_metrics(rows, tight_radius_m=1.8, delta_s=1.0)
    assert metrics["window_start_step"] == 50
    assert metrics["crossing_step"] == 100
    assert metrics["abs_heading_error_median_rad"] == pytest.approx(0.0)
    assert metrics["abs_pitch_error_median_rad"] == pytest.approx(0.0)
    assert metrics["abs_speed_error_median_mps"] == pytest.approx(0.0)
    assert metrics["command_goal_cosine_median"] == pytest.approx(1.0)
    assert metrics["actual_command_cosine_median"] == pytest.approx(1.0)


def test_h69_a_supported_and_has_priority_over_b() -> None:
    lost = _group(
        abs_heading_error_median_rad=0.30,
        abs_pitch_error_median_rad=0.20,
        actual_command_cosine_median=0.70,
    )
    result = _hypotheses(lost=lost)
    assert result["H69_A_supported"] is True
    assert result["H69_B_supported"] is True
    assert result["classification"] == "PRECROSSING_HEADING_TRACKING_LAG"


def test_h69_a_requires_2s_persistence() -> None:
    lost = _group(
        abs_heading_error_median_rad=0.30,
        actual_command_cosine_median=0.70,
    )
    consistency = _group(abs_heading_error_median_rad=0.10)
    result = _hypotheses(lost=lost, consistency=consistency)
    assert result["H69_A_supported"] is False
    assert result["heading_persistent_at_2s"] is False


def test_h69_b_pitch_or_speed_gate_and_priority() -> None:
    lost = _group(abs_heading_error_median_rad=0.10, abs_pitch_error_median_rad=0.20)
    result = _hypotheses(lost=lost)
    assert result["pitch_supported"] is True
    assert result["H69_B_supported"] is True
    assert result["classification"] == "PRECROSSING_MULTICHANNEL_TRACKING_LAG"

    lost = _group(
        abs_heading_error_median_rad=0.10,
        abs_speed_error_median_mps=0.35,
    )
    result = _hypotheses(lost=lost)
    assert result["speed_supported"] is True
    assert result["H69_B_supported"] is True


def test_h69_c_only_evaluated_for_default() -> None:
    lost = _group(
        abs_heading_error_median_rad=0.10,
        eta_tight_before_crossing_fraction=0.60,
        p_nav_tight_before_crossing_fraction=0.0,
        eta_tight_but_p_nav_not_fraction=0.60,
        eta_tight_p_nav_minus_eta_distance_median_m=0.25,
    )
    default = _hypotheses(lost=lost, scenario="default")
    assert default["H69_C_supported"] is True
    assert default["classification"] == "PRECROSSING_PNAV_TIMING"

    strict = _hypotheses(lost=lost, scenario="strict125")
    assert strict["H69_C_evaluated"] is False
    assert strict["H69_C_supported"] is False
    assert strict["classification"] == "PRECROSSING_TRACKING_CHANNEL_UNRESOLVED"


def test_tight_entry_uses_first_pre_crossing_entry() -> None:
    rows = _rows()
    result = ANALYSIS._tight_entry(rows, tight_radius_m=1.8)
    assert result["eta_tight_before_crossing"] is True
    assert result["p_nav_tight_before_crossing"] is False
    assert result["eta_tight_but_p_nav_not"] is True
    assert result["eta_tight_entry_step"] is not None
    assert result["eta_tight_entry_step"] <= 100
    assert result["eta_tight_p_nav_minus_eta_distance_m"] == pytest.approx(0.20)
    assert result["eta_tight_lead_before_crossing_s"] >= 0.0


def test_seed_rows_sorts_and_requires_rich_fields() -> None:
    rows = _rows(steps=20)
    trace = _Trace({name: value[::-1] for name, value in rows.items()})
    trace["seed"] = np.full(20, 20, dtype=np.int64)
    selected = ANALYSIS._seed_rows(trace, 20)
    assert np.array_equal(selected["step"], np.arange(20))

    del trace["nu"]
    with pytest.raises(ANALYSIS.DiagnosticError, match="missing fields"):
        ANALYSIS._seed_rows(trace, 20)
