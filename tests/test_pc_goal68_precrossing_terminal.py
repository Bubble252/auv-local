from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal68_precrossing_terminal.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal68_precrossing_terminal", ANALYSIS_SCRIPT
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
    distance[:crossing] = np.linspace(3.0, 2.0, crossing)
    distance[crossing:] = 2.0 + np.linspace(0.0, 0.5, steps - crossing)
    eta = np.zeros((steps, 3), dtype=np.float64)
    eta[:, 0] = distance
    p_nav = eta.copy()
    p_nav[:, 0] += 0.10
    signed = np.full(steps, -1.0, dtype=np.float64)
    signed[crossing:] = 0.01
    command = np.tile(np.asarray([math.pi, 0.0, 2.0]), (steps, 1))
    velocity = np.tile(np.asarray([-1.5, 0.0, 0.0]), (steps, 1))
    return {
        "step": np.arange(steps, dtype=np.int64),
        "done": np.r_[np.zeros(steps - 1, dtype=bool), True],
        "time_s": times,
        "signed_offset": signed,
        "eta_ned": eta,
        "p_nav_ned": p_nav,
        "goal_ned": np.zeros((steps, 3), dtype=np.float64),
        "eta_rpy": np.zeros((steps, 3), dtype=np.float64),
        "nu": np.tile(np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.2]), (steps, 1)),
        "ground_velocity_body_mps": velocity.copy(),
        "ground_velocity_ned_mps": velocity,
        "actual_actuator": np.zeros((steps, 3), dtype=np.float64),
        "command_physical": command,
        "material_saturated": np.zeros(steps, dtype=bool),
    }


def _group(**overrides: object) -> dict[str, object]:
    group: dict[str, object] = {
        "eta_outside_count": 4,
        "eta_outside_closure_deficit_fraction": ANALYSIS.DEFICIT_FRACTION_MIN,
        "eta_outside_closure_deficit_median_mps": 0.20,
        "command_goal_cosine_median": ANALYSIS.COMMAND_COSINE_MIN,
        "actual_goal_cosine_median": 0.50,
        "actual_ground_speed_median_mps": 1.0,
        "eta_tight_before_crossing_fraction": 0.50,
        "p_nav_tight_before_crossing_fraction": 0.20,
        "eta_min_distance_median_m": 1.10,
    }
    group.update(overrides)
    return group


def _hypotheses(
    lost: dict[str, object] | None = None,
    retained: dict[str, object] | None = None,
) -> dict[str, object]:
    primary_lost = _group() if lost is None else lost
    primary_retained = _group(
        eta_outside_closure_deficit_median_mps=0.0,
        eta_outside_closure_deficit_fraction=0.0,
        actual_goal_cosine_median=0.80,
    ) if retained is None else retained
    secondary_lost = {
        1.0: {**primary_lost, "eta_outside_closure_deficit_median_mps": 0.20},
        3.0: {**primary_lost, "eta_outside_closure_deficit_median_mps": 0.20},
    }
    secondary_retained = {
        1.0: {**primary_retained, "eta_outside_closure_deficit_median_mps": -0.50},
        3.0: {**primary_retained, "eta_outside_closure_deficit_median_mps": -0.50},
    }
    return ANALYSIS._hypotheses(
        primary_lost,
        primary_retained,
        secondary_lost,
        secondary_retained,
        tight_radius_m=1.0,
    )


def test_crossing_and_window_do_not_use_future_rows() -> None:
    rows = _rows()
    assert ANALYSIS._crossing_index(rows) == 100
    window, crossing = ANALYSIS._window_slice(rows, delta_s=2.0)
    assert crossing == 100
    assert window.start <= crossing
    assert window.stop == crossing + 1
    metrics = ANALYSIS._window_metrics(rows, tight_radius_m=1.75, delta_s=2.0)
    assert metrics["window_start_step"] == 0
    assert metrics["crossing_step"] == 100
    assert metrics["required_closure_mps"] == pytest.approx(1.25 / 2.0)
    assert metrics["command_goal_cosine_median"] == pytest.approx(1.0)
    assert metrics["actual_goal_cosine_median"] == pytest.approx(1.0)
    assert metrics["eta_tight_before_crossing"] is False
    assert metrics["p_nav_tight_before_crossing"] is False


def test_h68_thresholds_and_priority() -> None:
    result = _hypotheses()
    assert result["H68_A_supported"] is True
    assert result["classification"] == "PRECROSSING_AUTHORITY_LIMITED"

    lost = _group(eta_outside_closure_deficit_fraction=0.50)
    result = _hypotheses(lost=lost)
    assert result["H68_A_supported"] is False
    assert result["H68_B_supported"] is True
    assert result["classification"] == "PRECROSSING_TRACKING_LAG"

    lost = _group(
        eta_outside_closure_deficit_fraction=0.50,
        actual_goal_cosine_median=0.95,
    )
    result = _hypotheses(lost=lost)
    assert result["H68_B_supported"] is False
    assert result["H68_C_supported"] is True
    assert result["classification"] == "PRECROSSING_CAPTURE_TIMING"

    lost = _group(
        eta_outside_closure_deficit_fraction=0.50,
        actual_goal_cosine_median=0.95,
        eta_tight_before_crossing_fraction=0.20,
    )
    result = _hypotheses(lost=lost)
    assert result["classification"] == "PRECROSSING_MECHANISM_UNRESOLVED"


def test_h68_a_requires_secondary_direction_consistency() -> None:
    lost = _group()
    retained = _group(
        eta_outside_closure_deficit_median_mps=0.0,
        eta_outside_closure_deficit_fraction=0.0,
    )
    secondary_lost = {
        1.0: {**lost, "eta_outside_closure_deficit_median_mps": -0.01},
        3.0: {**lost, "eta_outside_closure_deficit_median_mps": 0.20},
    }
    secondary_retained = {
        1.0: {**retained, "eta_outside_closure_deficit_median_mps": -0.50},
        3.0: {**retained, "eta_outside_closure_deficit_median_mps": -0.50},
    }
    result = ANALYSIS._hypotheses(
        lost,
        retained,
        secondary_lost,
        secondary_retained,
        tight_radius_m=1.0,
    )
    assert result["H68_A_supported"] is False
    assert result["secondary_delta_consistency"] is False


def test_seed_rows_sorts_and_requires_rich_fields() -> None:
    rows = _rows(steps=20)
    rows["done"] = np.r_[np.zeros(19, dtype=bool), True]
    trace = _Trace({name: value[::-1] for name, value in rows.items()})
    trace["seed"] = np.full(20, 20, dtype=np.int64)
    selected = ANALYSIS._seed_rows(trace, 20)
    assert np.array_equal(selected["step"], np.arange(20))

    del trace["nu"]
    with pytest.raises(ANALYSIS.DiagnosticError, match="missing fields"):
        ANALYSIS._seed_rows(trace, 20)


def test_baseline_rows_require_only_outcome_fields(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    rows = [
        {"seed": seed, "success": True, "terminal_eta_capture": True}
        for seed in range(20, 60)
    ]
    (baseline / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    loaded = ANALYSIS._read_episodes(baseline, ANALYSIS.BASELINE_FIELDS)
    assert [row["seed"] for row in loaded] == list(range(20, 60))
