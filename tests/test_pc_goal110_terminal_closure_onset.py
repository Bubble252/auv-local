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
    "goal110_terminal_closure_onset",
    SCRIPTS / "analyze_pc_goal110_terminal_closure_onset.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _trace(*, rpm_flip_step: int | None, geometry_step: int, stall_step: int) -> dict[str, np.ndarray]:
    steps = 40
    dt = 0.1
    time_s = np.arange(steps, dtype=np.float64) * dt
    distance = 1.9 - 0.05 * np.minimum(np.arange(steps), stall_step)
    eta = np.zeros((steps, 6), dtype=np.float64)
    eta[:, 2] = 0.0
    goal = np.zeros((steps, 3), dtype=np.float64)
    pitch_error = np.zeros(steps, dtype=np.float64)
    pitch_error[geometry_step:] = 0.6
    heading_error = np.zeros(steps, dtype=np.float64)
    speed_error = np.full(steps, 0.2, dtype=np.float64)
    rpm = np.full(steps, 100.0, dtype=np.float64)
    if rpm_flip_step is not None:
        rpm[rpm_flip_step:] = -100.0
    residual = np.zeros((steps, 3), dtype=np.float64)
    residual[:, 2] = rpm
    return {
        "time_s": time_s,
        "goal_distance_eta_m": distance,
        "eta": eta,
        "goal_ned_m": goal,
        "heading_error_rad": heading_error,
        "pitch_error_rad": pitch_error,
        "speed_error_mps": speed_error,
        "actuator_residual_effective": residual,
    }


def test_trailing_median_and_sustained_detection() -> None:
    values = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    median = audit._trailing_median(values, window=5)
    assert np.isnan(median[:4]).all()
    assert median[4] == pytest.approx(0.0)
    mask = np.asarray([False, False, True, True, True, True], dtype=bool)
    assert audit._first_sustained(mask, start=0, window=4) == 5


def test_geometry_precedes_rpm_and_stall() -> None:
    event = audit._episode_events(
        _trace(rpm_flip_step=15, geometry_step=5, stall_step=20),
        seed=20,
    )
    assert event["evaluable"] is True
    assert event["precedence"] == "geometry_first"
    assert event["t_geom_s"] < event["t_rpm_flip_s"] < event["t_stall_s"]
    assert event["stall_lead_over_geometry_s"] > event["stall_lead_over_rpm_s"]


def test_rpm_precedes_geometry_and_stall() -> None:
    event = audit._episode_events(
        _trace(rpm_flip_step=5, geometry_step=15, stall_step=20),
        seed=21,
    )
    assert event["evaluable"] is True
    assert event["precedence"] == "rpm_first"
    assert event["t_rpm_flip_s"] < event["t_geom_s"] < event["t_stall_s"]
    assert event["rpm_lead_over_geometry_s"] < 0.0


def test_stall_first_is_not_forced_into_action_branch() -> None:
    event = audit._episode_events(
        _trace(rpm_flip_step=30, geometry_step=30, stall_step=20),
        seed=22,
    )
    assert event["evaluable"] is True
    assert event["precedence"] == "stall_first"


def test_missing_rpm_flip_makes_episode_insufficient() -> None:
    event = audit._episode_events(
        _trace(rpm_flip_step=None, geometry_step=5, stall_step=20),
        seed=23,
    )
    assert event["evaluable"] is False
    assert event["precedence"] == "insufficient_events"


def _event(precedence: str) -> dict[str, object]:
    if precedence == "geometry_first":
        geometry, rpm, stall = 0.9, 1.9, 2.4
    elif precedence == "rpm_first":
        geometry, rpm, stall = 1.9, 0.9, 2.4
    elif precedence == "stall_first":
        geometry, rpm, stall = 3.0, 3.0, 2.4
    else:
        geometry, rpm, stall = 1.9, 1.9, 2.4
    return {
        "evaluable": precedence != "insufficient_events",
        "no_entry": False,
        "t_stall_s": stall,
        "t_rpm_flip_s": rpm,
        "t_geom_s": geometry,
        "stall_lead_over_rpm_s": stall - rpm,
        "stall_lead_over_geometry_s": stall - geometry,
        "rpm_lead_over_geometry_s": rpm - geometry,
        "precedence": precedence,
    }


def _records(precedence: str, count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "success": False,
            "timeout": False,
            "safety": False,
            "events": _event(precedence),
        }
        for seed in range(count)
    ]


def test_gate_selects_geometry_branch() -> None:
    summary = audit._precedence_summary(_records("geometry_first"))
    gate = audit._classify(summary)
    assert gate["status"] == "TERMINAL_GEOMETRY_PRECEDES_RPM"
    assert gate["H110_A_supported"] is True
    assert gate["H110_B_supported"] is False


def test_gate_selects_rpm_branch() -> None:
    summary = audit._precedence_summary(_records("rpm_first"))
    gate = audit._classify(summary)
    assert gate["status"] == "RPM_REVERSAL_PRECEDES_STALL"
    assert gate["H110_B_supported"] is True


def test_gate_selects_stall_branch() -> None:
    summary = audit._precedence_summary(_records("stall_first"))
    gate = audit._classify(summary)
    assert gate["status"] == "CLOSURE_STALL_FIRST"


def test_gate_marks_mixed_when_no_branch_reaches_threshold() -> None:
    records = _records("geometry_first", 10) + _records("rpm_first", 10)
    summary = audit._precedence_summary(records)
    gate = audit._classify(summary)
    assert gate["status"] == "MIXED_OR_AMBIGUOUS"
    assert gate["H110_C_supported"] is True


def test_gate_marks_insufficient_events() -> None:
    records = _records("geometry_first", 10) + _records("insufficient_events", 10)
    summary = audit._precedence_summary(records)
    gate = audit._classify(summary)
    assert gate["status"] == "INSUFFICIENT_EVENTS"


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
