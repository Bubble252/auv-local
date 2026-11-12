from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal44_terminal_bias_persistence.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_goal44_terminal_bias_persistence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_error_trajectory_uses_queue_delay() -> None:
    time_s = np.arange(0.0, 6.02, 0.02, dtype=np.float64)
    eta = np.column_stack([time_s, np.zeros_like(time_s), np.zeros_like(time_s)])
    p_nav = np.column_stack(
        [time_s - 1.0 + 0.2, np.zeros_like(time_s), np.zeros_like(time_s)]
    )

    errors, valid = MODULE._exact_error_trajectory(
        p_nav=p_nav, time_s=time_s, eta_trajectory_xyz=eta, queue_age_s=1.0
    )

    assert int(valid.sum()) == len(time_s) - 50
    assert np.allclose(errors[valid, 0], 0.2)


def test_late_window_metrics_separates_persistence_and_amplification() -> None:
    time_s = np.arange(0.0, 6.02, 0.02, dtype=np.float64)
    exact_error = np.zeros((time_s.size, 3), dtype=np.float64)
    exact_error[:, 0] = 0.2
    exact_error[-1, 0] = 0.5
    valid = np.ones(time_s.size, dtype=bool)

    metrics = MODULE._late_window_metrics(
        exact_error=exact_error,
        valid_rows=valid,
        time_s=time_s,
        eta_terminal=np.asarray([6.0, 0.0, 0.0]),
        goal=np.asarray([7.0, 0.0, 0.0]),
    )

    assert metrics["late_projection_m"] == pytest.approx(0.2)
    assert metrics["terminal_projection_m"] == pytest.approx(0.5)
    assert metrics["terminal_delta_m"] == pytest.approx(0.3)
    assert metrics["late_share"] == pytest.approx(0.4)
    assert metrics["window_rows"] == 226
    assert metrics["late_toward_goal"] is True
    assert metrics["terminal_toward_goal"] is True


def test_late_window_allows_non_discordant_negative_terminal_projection() -> None:
    time_s = np.arange(0.0, 6.02, 0.02, dtype=np.float64)
    exact_error = np.zeros((time_s.size, 3), dtype=np.float64)
    exact_error[:, 0] = -0.1
    valid = np.ones(time_s.size, dtype=bool)

    metrics = MODULE._late_window_metrics(
        exact_error=exact_error,
        valid_rows=valid,
        time_s=time_s,
        eta_terminal=np.asarray([6.0, 0.0, 0.0]),
        goal=np.asarray([7.0, 0.0, 0.0]),
    )

    assert metrics["terminal_toward_goal"] is False
    assert metrics["late_share"] == pytest.approx(1.0)


def test_late_window_rejects_too_few_rows() -> None:
    time_s = np.arange(0.0, 1.02, 0.02, dtype=np.float64)
    exact_error = np.zeros((time_s.size, 3), dtype=np.float64)
    valid = np.ones(time_s.size, dtype=bool)

    with pytest.raises(ValueError, match="late window"):
        MODULE._late_window_metrics(
            exact_error=exact_error,
            valid_rows=valid,
            time_s=time_s,
            eta_terminal=np.zeros(3),
            goal=np.asarray([1.0, 0.0, 0.0]),
        )


def _record(
    seed: int,
    *,
    discordant: bool,
    late_projection: float = 0.2,
    terminal_projection: float = 0.4,
) -> dict[str, Any]:
    return {
        "scenario": "default",
        "mode": "baseline",
        "seed": seed,
        "success": True,
        "goal_capture_eta": not discordant,
        "discordant": discordant,
        "queue_age_s": 0.08,
        "window_rows": 225,
        "late_projection_m": late_projection,
        "terminal_projection_m": terminal_projection,
        "terminal_delta_m": terminal_projection - late_projection,
        "late_share": late_projection / terminal_projection,
        "late_toward_goal": late_projection > 0.0,
        "terminal_toward_goal": True,
    }


def test_aggregate_freezes_persistence_and_amplification() -> None:
    discordant = [_record(seed, discordant=True) for seed in range(5)]
    non_discordant = [_record(seed, discordant=False) for seed in range(5, 15)]
    aggregate = MODULE._aggregate([*discordant, *non_discordant])

    assert aggregate["discordant_seed_count"] == 5
    assert aggregate["late_toward_goal_fraction"] == pytest.approx(1.0)
    assert aggregate["late_share_median"] == pytest.approx(0.5)
    assert aggregate["h_persistent_late_bias_pass"] is True
    assert aggregate["h_terminal_amplification_pass"] is True


@pytest.mark.parametrize(
    ("persistent", "amplification", "invalid", "expected"),
    [
        (
            "SUPPORTED",
            "SUPPORTED",
            False,
            "PERSISTENT_BIAS_WITH_TERMINAL_AMPLIFICATION",
        ),
        ("SUPPORTED", "FALSIFIED", False, "PERSISTENT_LATE_BIAS"),
        ("FALSIFIED", "SUPPORTED", False, "TERMINAL_TRANSIENT_BIAS"),
        ("PARTIAL", "FALSIFIED", False, "PARTIAL_PERSISTENCE_EVIDENCE"),
        ("FALSIFIED", "FALSIFIED", False, "NO_STABLE_LATE_SIGNATURE"),
        ("SUPPORTED", "SUPPORTED", True, "INVALID_DIAGNOSTIC"),
    ],
)
def test_overall_decision_mapping(
    persistent: str, amplification: str, invalid: bool, expected: str
) -> None:
    result = MODULE._overall_decision(persistent, amplification, invalid=invalid)

    assert result["decision"] == expected


def test_verdict_requires_all_blocks() -> None:
    assert MODULE._verdict(4) == "SUPPORTED"
    assert MODULE._verdict(1) == "PARTIAL"
    assert MODULE._verdict(0) == "FALSIFIED"
