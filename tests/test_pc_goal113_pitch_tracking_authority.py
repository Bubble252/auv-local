from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "goal113_pitch_tracking_authority",
    SCRIPTS / "analyze_pc_goal113_pitch_tracking_authority.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _trace(
    *,
    steps: int = 120,
    pitch_err: float = 0.0,
    pitch_err_slope: float = 0.0,
    actual_pitch: float | None = None,
    stern: float = 0.0,
    saturated: bool = False,
    speed_error: float = 0.0,
    current: float = 0.0,
    heave: float = 0.0,
) -> dict[str, np.ndarray]:
    time_s = np.arange(steps, dtype=np.float64) * 0.05
    distance = 4.0 - 0.03 * np.arange(steps, dtype=np.float64)
    eta = np.zeros((steps, 6), dtype=np.float64)
    nu = np.zeros((steps, 6), dtype=np.float64)
    command = np.zeros((steps, 3), dtype=np.float64)
    command[:, 2] = 1.5
    if actual_pitch is None:
        command[:, 1] = pitch_err + pitch_err_slope * time_s
        eta[:, 4] = pitch_err_slope * time_s
    else:
        eta[:, 4] = actual_pitch
        command[:, 1] = eta[:, 4] + pitch_err
    total = np.zeros((steps, 3), dtype=np.float64)
    total[:, 1] = stern
    return {
        "time_s": time_s,
        "goal_distance_eta_m": distance,
        "eta": eta,
        "nu": nu,
        "command": command,
        "ground_speed_mps": np.full(steps, 1.5, dtype=np.float64),
        "speed_error_mps": np.full(steps, speed_error, dtype=np.float64),
        "pitch_error_rad": command[:, 1] - eta[:, 4],
        "actuator_base_physical": np.zeros((steps, 3), dtype=np.float64),
        "actuator_residual_effective": np.zeros((steps, 3), dtype=np.float64),
        "actuator_total_physical": total,
        "current_ned_mps": np.tile(np.asarray([current, 0.0, 0.0]), (steps, 1)),
        "material_saturated": np.full(steps, saturated, dtype=bool),
        "guidance_terminal_phase": np.ones(steps, dtype=np.float64),
    }


def _entry(trace: dict[str, np.ndarray], shell: float = 3.0) -> dict[str, object]:
    arrays = audit._validate_trace(trace, seed=20)
    return audit._entry_metrics(arrays, shell_m=shell)


def test_validate_rejects_pitch_error_mismatch() -> None:
    trace = _trace(pitch_err=0.1)
    trace["pitch_error_rad"][10] = 0.2
    with pytest.raises(audit.AuditError, match="pitch_error_rad disagrees"):
        audit._validate_trace(trace, seed=20)


def test_validate_rejects_bad_shape() -> None:
    trace = _trace()
    trace["nu"] = np.zeros((120, 5), dtype=np.float64)
    with pytest.raises(audit.AuditError, match="invalid shape"):
        audit._validate_trace(trace, seed=20)


def test_pitch_threshold_is_inclusive() -> None:
    result = _entry(_trace(pitch_err=0.45, actual_pitch=0.0))
    assert result["pitch_tracking_bad"] is True
    assert result["cause"] == "PREENTRY_PITCH_LAG"


def test_preexisting_pitch_lag_class() -> None:
    result = _entry(_trace(pitch_err=0.60, actual_pitch=0.0))
    assert result["cause"] == "PREENTRY_PITCH_LAG"
    assert result["rate_unreachable_bad"] is False
    assert result["authority_bad"] is False
    assert result["coupling_strong"] is False


def test_no_pitch_lag_class() -> None:
    result = _entry(_trace(pitch_err=0.10, actual_pitch=0.0))
    assert result["cause"] == "NO_PITCH_LAG"
    assert result["pitch_tracking_bad"] is False


def test_authority_limited_class_from_saturation() -> None:
    result = _entry(_trace(pitch_err=0.60, actual_pitch=0.0, saturated=True))
    assert result["authority_bad"] is True
    assert result["cause"] == "AUTHORITY_LIMITED"


def test_authority_limited_class_from_stern_norm() -> None:
    result = _entry(
        _trace(
            pitch_err=0.60,
            actual_pitch=0.0,
            stern=math.radians(19.5),
        )
    )
    assert result["stern_norm"] >= 0.95
    assert result["cause"] == "AUTHORITY_LIMITED"


def test_command_rate_unreachable_class() -> None:
    steps = 120
    time_s = np.arange(steps, dtype=np.float64) * 0.05
    trace = _trace(steps=steps, pitch_err=0.0, actual_pitch=0.0)
    trace["command"][:, 1] = 0.5 + 0.10 * time_s
    trace["eta"][:, 4] = 0.0
    trace["pitch_error_rad"] = trace["command"][:, 1] - trace["eta"][:, 4]
    result = _entry(trace)
    assert result["pitch_err_rad"] >= 0.45
    assert result["rate_error_abs_rad_s"] >= 0.05
    assert result["rate_sign_align"] <= 0.0
    assert result["cause"] == "COMMAND_RATE_UNREACHABLE"


def test_speed_current_coupled_class() -> None:
    steps = 120
    time_s = np.arange(steps, dtype=np.float64) * 0.05
    trace = _trace(steps=steps, pitch_err=0.0, actual_pitch=0.0)
    speed_error = 0.5 + 0.2 * np.sin(time_s * 0.1)
    trace["speed_error_mps"] = speed_error
    trace["eta"][:, 4] = 0.0
    trace["command"][:, 1] = 0.5 + 0.01 * np.sin(time_s * 0.1)
    trace["pitch_error_rad"] = trace["command"][:, 1] - trace["eta"][:, 4]
    result = _entry(trace)
    assert abs(result["rho_speed_error"]) >= 0.5
    assert result["cause"] == "SPEED_CURRENT_COUPLED"


def test_mixed_class_when_two_mechanisms_fire() -> None:
    steps = 120
    time_s = np.arange(steps, dtype=np.float64) * 0.05
    trace = _trace(steps=steps, pitch_err=0.0, actual_pitch=0.0, saturated=True)
    trace["command"][:, 1] = 0.5 + 0.10 * time_s
    trace["eta"][:, 4] = 0.0
    trace["pitch_error_rad"] = trace["command"][:, 1] - trace["eta"][:, 4]
    result = _entry(trace)
    assert "AUTHORITY_LIMITED" in result["mechanisms"]
    assert "COMMAND_RATE_UNREACHABLE" in result["mechanisms"]
    assert result["cause"] == "MIXED_OR_AMBIGUOUS"


def test_spearman_constant_is_nan() -> None:
    assert math.isnan(audit._spearman(np.ones(10), np.arange(10)))
    assert math.isnan(audit._spearman(np.arange(10), np.ones(10)))


def _record(
    cause: str,
    mechanisms: list[str],
    *,
    pitch_bad: bool = True,
    rate_deficit_bad: bool = False,
    rate_align_bad: bool = False,
) -> dict[str, object]:
    return {
        "primary": {
            "cause": cause,
            "mechanisms": mechanisms,
            "pitch_tracking_bad": pitch_bad,
            "rate_deficit_bad": rate_deficit_bad,
            "rate_align_bad": rate_align_bad,
        }
    }


def _records(mechanisms: list[str], *, n: int = 11) -> list[dict[str, object]]:
    return [_record(mechanisms[0] if mechanisms else "MIXED_OR_AMBIGUOUS", mechanisms) for _ in range(n)]


def test_gate_authority_majority() -> None:
    gate = audit._gate(_records(["AUTHORITY_LIMITED"]))
    assert gate["status"] == "PITCH_AUTHORITY_LIMITED"
    assert gate["H113_B_supported"] is True


def test_gate_rate_majority() -> None:
    gate = audit._gate(_records(["COMMAND_RATE_UNREACHABLE"]))
    assert gate["status"] == "PITCH_RATE_UNREACHABLE"
    assert gate["H113_A_supported"] is True


def test_gate_coupling_majority() -> None:
    gate = audit._gate(_records(["SPEED_CURRENT_COUPLED"]))
    assert gate["status"] == "PITCH_SPEED_CURRENT_COUPLED"
    assert gate["H113_C_supported"] is True


def test_gate_preexisting_majority() -> None:
    gate = audit._gate(_records(["PREENTRY_PITCH_LAG"]))
    assert gate["status"] == "PREENTRY_PITCH_LAG"
    assert gate["H113_D_supported"] is True


def test_gate_no_pitch_lag_majority() -> None:
    gate = audit._gate(_records(["NO_PITCH_LAG"]))
    assert gate["status"] == "PITCH_COMMAND_REACHABLE"
    assert gate["H113_E_supported"] is True


def test_gate_mixed_when_two_mechanisms_majority() -> None:
    records = _records(["AUTHORITY_LIMITED", "COMMAND_RATE_UNREACHABLE"])
    gate = audit._gate(records)
    assert gate["status"] == "MIXED_PITCH_TRACKING"
    assert gate["multi_mechanism_n"] == 11


def test_gate_ambiguous_when_no_majority() -> None:
    records = [
        _record("PREENTRY_PITCH_LAG", ["PREENTRY_PITCH_LAG"]),
        _record("NO_PITCH_LAG", ["NO_PITCH_LAG"], pitch_bad=False),
        _record("MIXED_OR_AMBIGUOUS", []),
    ] * 3
    records = records[:11]
    gate = audit._gate(records)
    assert gate["status"] == "PITCH_TRACKING_AMBIGUOUS"
