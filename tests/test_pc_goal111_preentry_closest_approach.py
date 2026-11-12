from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "goal111_preentry_closest_approach",
    SCRIPTS / "analyze_pc_goal111_preentry_closest_approach.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _trace(
    *,
    distance: np.ndarray,
    vertical: np.ndarray | None = None,
    pitch: np.ndarray | None = None,
    heading: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    steps = int(distance.size)
    time_s = np.arange(steps, dtype=np.float64) * 0.1
    eta = np.zeros((steps, 6), dtype=np.float64)
    if vertical is not None:
        eta[:, 2] = np.asarray(vertical, dtype=np.float64)
    goal = np.zeros((steps, 3), dtype=np.float64)
    return {
        "time_s": time_s,
        "goal_distance_eta_m": np.asarray(distance, dtype=np.float64),
        "eta": eta,
        "goal_ned_m": goal,
        "heading_error_rad": (
            np.zeros(steps, dtype=np.float64)
            if heading is None
            else np.asarray(heading, dtype=np.float64)
        ),
        "pitch_error_rad": (
            np.zeros(steps, dtype=np.float64)
            if pitch is None
            else np.asarray(pitch, dtype=np.float64)
        ),
    }


def _approach_deficit_trace() -> dict[str, np.ndarray]:
    steps = 40
    return _trace(distance=np.linspace(3.5, 2.55, steps))


def _entered_trace(*, vertical_after_entry: float | None) -> dict[str, np.ndarray]:
    steps = 40
    distance = np.linspace(3.2, 1.55, steps)
    vertical = np.zeros(steps, dtype=np.float64)
    if vertical_after_entry is None:
        vertical[:] = 1.3
    else:
        entry = int(np.flatnonzero(distance <= 2.0)[0])
        vertical[entry + 1 :] = vertical_after_entry
    return _trace(distance=distance, vertical=vertical)


def test_first_entry_returns_earliest_shell_crossing() -> None:
    distance = np.asarray([4.0, 3.0, 2.7, 2.4, 1.9], dtype=np.float64)
    assert audit._first_entry(distance, 3.0) == 1
    assert audit._first_entry(distance, 2.5) == 3
    assert audit._first_entry(distance, 2.0) == 4
    assert audit._first_entry(distance, 1.5) is None


def test_rolling_median_at_uses_only_available_history() -> None:
    values = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    assert audit._rolling_median_at(values, 4, 5) == pytest.approx(0.0)
    assert audit._rolling_median_at(values, 1, 5) == pytest.approx(0.0)


def test_never_entering_two_meters_is_approach_deficit() -> None:
    metrics = audit._episode_metrics(_approach_deficit_trace(), seed=20, success=False)
    assert metrics["min_distance_m"] > 2.0
    assert metrics["classification"] == "APPROACH_DEFICIT"


def test_geometry_already_bad_at_entry_is_preentry_failure() -> None:
    metrics = audit._episode_metrics(
        _entered_trace(vertical_after_entry=None), seed=21, success=False
    )
    assert metrics["min_distance_m"] <= 2.0
    assert metrics["shells"]["2.0"]["geometry_bad_median5"] is True
    assert metrics["classification"] == "PREENTRY_GEOMETRY_BAD"


def test_geometry_ok_at_entry_then_bad_is_late_failure() -> None:
    metrics = audit._episode_metrics(
        _entered_trace(vertical_after_entry=1.3), seed=22, success=False
    )
    assert metrics["min_distance_m"] <= 2.0
    assert metrics["shells"]["2.0"]["geometry_bad_median5"] is False
    assert metrics["closest_vertical_miss_m"] >= 1.0
    assert metrics["classification"] == "ENTRY_GEOMETRY_OK_LATE_FAILURE"


def test_success_has_no_failure_classification() -> None:
    metrics = audit._episode_metrics(
        _entered_trace(vertical_after_entry=1.3), seed=23, success=True
    )
    assert metrics["classification"] is None


def _primary(counts: dict[str, int]) -> dict[str, object]:
    total = sum(counts.values())
    return {
        "failure_n": total,
        "classification_counts": counts,
        "classification_fractions": {
            name: counts[name] / total for name in audit.CLASSES
        },
    }


@pytest.mark.parametrize(
    ("counts", "expected_status"),
    (
        (
            {
                "APPROACH_DEFICIT": 10,
                "PREENTRY_GEOMETRY_BAD": 6,
                "ENTRY_GEOMETRY_OK_LATE_FAILURE": 4,
            },
            "APPROACH_DEFICIT_MAJORITY",
        ),
        (
            {
                "APPROACH_DEFICIT": 4,
                "PREENTRY_GEOMETRY_BAD": 11,
                "ENTRY_GEOMETRY_OK_LATE_FAILURE": 5,
            },
            "PREENTRY_GEOMETRY_BAD_MAJORITY",
        ),
        (
            {
                "APPROACH_DEFICIT": 3,
                "PREENTRY_GEOMETRY_BAD": 3,
                "ENTRY_GEOMETRY_OK_LATE_FAILURE": 14,
            },
            "LATE_TERMINAL_FAILURE_MAJORITY",
        ),
    ),
)
def test_gate_selects_unique_majority(counts: dict[str, int], expected_status: str) -> None:
    gate = audit._classify_gate(_primary(counts))
    assert gate["status"] == expected_status
    assert gate["maximum_fraction"] >= 0.5
    assert gate["winner"] is not None


def test_gate_tie_is_mixed_even_when_fraction_is_half() -> None:
    gate = audit._classify_gate(
        _primary(
            {
                "APPROACH_DEFICIT": 10,
                "PREENTRY_GEOMETRY_BAD": 10,
                "ENTRY_GEOMETRY_OK_LATE_FAILURE": 0,
            }
        )
    )
    assert gate["status"] == "MIXED_CLOSEST_APPROACH"
    assert gate["reason"] == "no_unique_class_at_least_50pct"
    assert gate["H111_D_supported"] is True


def test_gate_without_half_majority_is_mixed() -> None:
    gate = audit._classify_gate(
        _primary(
            {
                "APPROACH_DEFICIT": 7,
                "PREENTRY_GEOMETRY_BAD": 7,
                "ENTRY_GEOMETRY_OK_LATE_FAILURE": 6,
            }
        )
    )
    assert gate["status"] == "MIXED_CLOSEST_APPROACH"
    assert gate["maximum_fraction"] < 0.5


def test_missing_trace_field_is_rejected() -> None:
    trace = _approach_deficit_trace()
    del trace["heading_error_rad"]
    with pytest.raises(audit.AuditError, match="missing fields"):
        audit._episode_metrics(trace, seed=24, success=False)


def test_episode_contract_rejects_wrong_success_count(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    episodes = [
        {"seed": seed, "success": seed < 52, "timeout": False, "safety": False}
        for seed in audit.DEV_SEEDS
    ]
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA,
                "actuator_residual_scale_override": audit.EXPECTED_EVAL_SCALE,
                "actuator_residual_channel_gains": [1.0, 1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(audit.AuditError, match="success 32/40"):
        audit._load_episodes(run, scenario="default", profile="C0")
