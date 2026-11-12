from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "goal114_pitch_onset",
    SCRIPTS / "analyze_pc_goal114_pitch_onset.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _thresholds() -> dict[str, float]:
    return {
        "pitch_error_abs": 0.45,
        "pitch_cmd_rate_abs": 0.20,
        "pitch_rate_abs": 10.0,
        "rate_error_abs": 0.10,
        "rate_deficit_abs": 0.10,
        "stern_norm": 0.95,
        "speed_error_abs": 0.40,
        "current_mps": 0.35,
    }


def _trace(steps: int = 160, *, stop_distance: float = 1.0) -> dict[str, np.ndarray]:
    time_s = np.arange(steps, dtype=np.float64) * 0.05
    distance = np.linspace(4.0, stop_distance, steps, dtype=np.float64)
    eta = np.zeros((steps, 6), dtype=np.float64)
    nu = np.zeros((steps, 6), dtype=np.float64)
    command = np.zeros((steps, 3), dtype=np.float64)
    command[:, 2] = 1.5
    return {
        "time_s": time_s,
        "goal_distance_eta_m": distance,
        "eta": eta,
        "nu": nu,
        "command": command,
        "actuator_total_physical": np.zeros((steps, 3), dtype=np.float64),
        "speed_error_mps": np.zeros(steps, dtype=np.float64),
        "current_ned_mps": np.zeros((steps, 3), dtype=np.float64),
        "material_saturated": np.zeros(steps, dtype=bool),
    }


def _ramp_pitch(trace: dict[str, np.ndarray], *, start_s: float = 2.0, rate: float = 0.10) -> None:
    time_s = trace["time_s"]
    command = trace["command"]
    command[:, 1] = np.maximum(0.0, rate * (time_s - start_s))
    trace["pitch_error_rad"] = command[:, 1] - trace["eta"][:, 4]


def _step_stern(trace: dict[str, np.ndarray], *, start_s: float = 2.0, value: float = math.radians(19.8)) -> None:
    trace["actuator_total_physical"][trace["time_s"] >= start_s, 1] = value


def _step_current(trace: dict[str, np.ndarray], *, start_s: float = 2.0, value: float = 1.0) -> None:
    trace["current_ned_mps"][trace["time_s"] >= start_s, 0] = value


def test_first_sustained_returns_first_of_three_frame_run() -> None:
    values = np.zeros(20, dtype=np.float64)
    values[7:9] = 1.0
    values[9:12] = 1.0
    assert audit._first_sustained(values, 0.5, start=0, stop=19) == 7
    assert audit._first_sustained(values, 0.5, start=9, stop=19) == 9
    assert audit._first_sustained(values, 2.0, start=0, stop=19) is None


def test_envelope_thresholds_apply_physical_bounds() -> None:
    traces = []
    for _ in range(12):
        trace = _trace()
        # Success control has large pitch error but quiet rate/stern/current.
        trace["command"][:, 1] = 0.8
        trace["eta"][:, 4] = 0.0
        trace["actuator_total_physical"][:, 1] = 0.0
        traces.append(trace)
    thresholds = audit._envelope_thresholds(traces, scenario="synthetic", profile="C0")
    assert thresholds["pitch_error_abs"] >= 0.45
    assert thresholds["pitch_cmd_rate_abs"] >= 0.20
    assert thresholds["stern_norm"] == 0.0


def test_classify_rate_first() -> None:
    classification, reason = audit._classify_groups(
        onset_rel_s={},
        group_onsets={"rate": -0.8, "authority": None, "speed_current": None, "tracking": 0.5},
        alternating_handoff=False,
    )
    assert classification == "RATE_DEMAND_FIRST"
    assert "rate" in reason


def test_classify_authority_first() -> None:
    classification, _ = audit._classify_groups(
        onset_rel_s={},
        group_onsets={"rate": None, "authority": -0.8, "speed_current": None, "tracking": 0.5},
        alternating_handoff=False,
    )
    assert classification == "AUTHORITY_FIRST"


def test_classify_speed_current_first() -> None:
    classification, _ = audit._classify_groups(
        onset_rel_s={},
        group_onsets={"rate": None, "authority": None, "speed_current": -0.8, "tracking": 0.5},
        alternating_handoff=False,
    )
    assert classification == "SPEED_CURRENT_FIRST"


def test_classify_tracking_first() -> None:
    classification, _ = audit._classify_groups(
        onset_rel_s={},
        group_onsets={"rate": None, "authority": None, "speed_current": None, "tracking": -0.8},
        alternating_handoff=False,
    )
    assert classification == "TRACKING_ERROR_FIRST"


def test_classify_mixed_when_onsets_are_close() -> None:
    classification, reason = audit._classify_groups(
        onset_rel_s={},
        group_onsets={"rate": -0.8, "authority": -0.7, "speed_current": None, "tracking": 0.5},
        alternating_handoff=False,
    )
    assert classification == "MIXED_PITCH_HANDOFF"
    assert "within" in reason


def test_classify_no_onset() -> None:
    classification, _ = audit._classify_groups(
        onset_rel_s={},
        group_onsets={"rate": None, "authority": None, "speed_current": None, "tracking": None},
        alternating_handoff=False,
    )
    assert classification == "NO_ONSET_ENVELOPE"


def test_episode_onset_rate_first_on_ramp() -> None:
    trace = _trace()
    _ramp_pitch(trace, start_s=0.0, rate=0.15)
    result = audit._episode_onset(
        trace,
        seed=20,
        thresholds=_thresholds(),
        apply_fallback=False,
    )
    assert result["evaluable"] is True
    assert result["classification"] == "RATE_DEMAND_FIRST"
    assert result["group_onsets"]["rate"] < result["group_onsets"]["tracking"]


def test_episode_onset_authority_first() -> None:
    trace = _trace()
    _step_stern(trace, start_s=3.0)
    _ramp_pitch(trace, start_s=4.5, rate=0.30)
    result = audit._episode_onset(
        trace,
        seed=20,
        thresholds=_thresholds(),
        apply_fallback=False,
    )
    assert result["classification"] == "AUTHORITY_FIRST"
    assert result["group_onsets"]["authority"] < result["group_onsets"]["tracking"]


def test_episode_onset_speed_current_first() -> None:
    trace = _trace()
    _step_current(trace, start_s=3.0)
    _ramp_pitch(trace, start_s=4.5, rate=0.30)
    result = audit._episode_onset(
        trace,
        seed=20,
        thresholds=_thresholds(),
        apply_fallback=False,
    )
    assert result["classification"] == "SPEED_CURRENT_FIRST"
    assert result["group_onsets"]["speed_current"] < result["group_onsets"]["tracking"]


def test_episode_onset_tracking_first_when_pitch_error_is_already_bad() -> None:
    trace = _trace()
    trace["command"][:, 1] = 0.60
    trace["eta"][:, 4] = 0.0
    result = audit._episode_onset(
        trace,
        seed=20,
        thresholds=_thresholds(),
        apply_fallback=False,
    )
    assert result["classification"] == "TRACKING_ERROR_FIRST"


def test_episode_onset_no_2m_entry() -> None:
    trace = _trace(stop_distance=2.5)
    result = audit._episode_onset(
        trace,
        seed=20,
        thresholds=_thresholds(),
        apply_fallback=True,
    )
    assert result["evaluable"] is False
    assert result["classification"] == "NO_2M_ENTRY"


def test_gate_rate_majority() -> None:
    records = [
        {
            "evaluable": True,
            "classification": "RATE_DEMAND_FIRST",
        }
        for _ in range(11)
    ]
    gate = audit._gate(records)
    assert gate["status"] == "PITCH_RATE_DEMAND_FIRST"
    assert gate["H114_A_supported"] is True


def test_gate_mixed_majority() -> None:
    records = [
        {
            "evaluable": True,
            "classification": "MIXED_PITCH_HANDOFF",
        }
        for _ in range(11)
    ]
    gate = audit._gate(records)
    assert gate["status"] == "MIXED_PITCH_HANDOFF"
    assert gate["H114_D_supported"] is True


def test_gate_insufficient_when_fewer_than_nine_evaluable() -> None:
    records = [
        {"evaluable": True, "classification": "RATE_DEMAND_FIRST"}
        for _ in range(8)
    ] + [{"evaluable": False, "classification": "NO_2M_ENTRY"} for _ in range(3)]
    gate = audit._gate(records)
    assert gate["status"] == "INSUFFICIENT_2M_EVENTS"
    assert gate["evaluable_n"] == 8


def test_segment_deltas() -> None:
    snapshots = {
        "3.0": {name: 0.0 for name in audit.SIGNALS},
        "2.5": {name: 1.0 for name in audit.SIGNALS},
        "2.0": {name: 2.0 for name in audit.SIGNALS},
        "1.5": {name: 3.0 for name in audit.SIGNALS},
    }
    deltas = audit._segment_deltas(snapshots)
    assert deltas["3.0->2.5"]["pitch_error_abs"] == 1.0
    assert deltas["2.5->2.0"]["pitch_error_abs"] == 1.0
    assert deltas["2.0->1.5"]["pitch_error_abs"] == 1.0
