from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal41_terminal_nav_bias.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_goal41_terminal_nav_bias", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_terminal_metrics_detect_closer_and_toward_goal() -> None:
    result = MODULE._terminal_nav_metrics(
        p_nav=np.asarray([9.5, 0.0, 0.0]),
        eta=np.asarray([9.0, 0.0, 0.0]),
        goal=np.asarray([10.0, 0.0, 0.0]),
        pnav_goal_distance_m=0.5,
        eta_goal_distance_m=1.0,
    )

    assert result["p_nav_closer"] is True
    assert result["distance_gap_m"] == pytest.approx(0.5)
    assert result["nav_error_m"] == pytest.approx(0.5)
    assert result["direction_cos"] == pytest.approx(1.0)
    assert result["toward_goal"] is True


def test_terminal_metrics_reject_inconsistent_distance() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        MODULE._terminal_nav_metrics(
            p_nav=np.asarray([9.5, 0.0, 0.0]),
            eta=np.asarray([9.0, 0.0, 0.0]),
            goal=np.asarray([10.0, 0.0, 0.0]),
            pnav_goal_distance_m=0.4,
            eta_goal_distance_m=1.0,
        )


def test_terminal_metrics_sideways_error_is_not_toward_goal() -> None:
    result = MODULE._terminal_nav_metrics(
        p_nav=np.asarray([9.0, 1.0, 0.0]),
        eta=np.asarray([9.0, 0.0, 0.0]),
        goal=np.asarray([10.0, 0.0, 0.0]),
        pnav_goal_distance_m=float(np.sqrt(2.0)),
        eta_goal_distance_m=1.0,
    )

    assert result["p_nav_closer"] is False
    assert result["direction_cos"] == pytest.approx(0.0)
    assert result["toward_goal"] is False


def test_record_for_seed_accepts_six_state_eta_rows() -> None:
    trace = {
        "p_nav": np.asarray([[9.5, 0.0, 0.0]]),
        "eta": np.asarray([[9.0, 0.0, 0.0, 0.1, 0.2, 0.3]]),
        "goal_ned_m": np.asarray([[10.0, 0.0, 0.0]]),
        "goal_distance_pnav_m": np.asarray([0.5]),
        "goal_distance_eta_m": np.asarray([1.0]),
        "current_estimate_age_s": np.asarray([1.25]),
    }

    record = MODULE._record_for_seed(
        20,
        0,
        trace,
        {"success": True, "goal_capture_eta": False},
        "default",
        "baseline",
    )

    assert record["discordant"] is True
    assert record["nav_error_m"] == pytest.approx(0.5)
    assert record["current_estimate_age_s"] == pytest.approx(1.25)


def _record(
    seed: int,
    *,
    p_nav_closer: bool,
    toward_goal: bool,
    age_s: float,
) -> dict[str, Any]:
    distance_gap = 0.2 if p_nav_closer else -0.2
    return {
        "seed": seed,
        "p_nav_closer": p_nav_closer,
        "toward_goal": toward_goal,
        "pnav_goal_distance_m": 1.0,
        "eta_goal_distance_m": 1.0 + distance_gap,
        "distance_gap_m": distance_gap,
        "nav_error_m": 0.4,
        "direction_cos": 0.5 if toward_goal else -0.5,
        "current_estimate_age_s": age_s,
    }


def test_aggregate_and_frozen_block_threshold() -> None:
    records = [
        _record(i, p_nav_closer=True, toward_goal=True, age_s=float(i - 19))
        for i in range(20, 24)
    ]
    records.append(_record(24, p_nav_closer=False, toward_goal=True, age_s=5.0))

    aggregate = MODULE._aggregate(records)
    passes = MODULE._block_hypothesis_passes(aggregate)

    assert aggregate["discordant_seed_count"] == 5
    assert aggregate["p_nav_closer_fraction"] == pytest.approx(0.8)
    assert aggregate["toward_goal_fraction"] == pytest.approx(1.0)
    assert aggregate["median_current_estimate_age_s"] == pytest.approx(3.0)
    assert passes == {"distance": True, "direction": True}

    weaker = [
        _record(i, p_nav_closer=(i < 23), toward_goal=True, age_s=1.0)
        for i in range(20, 25)
    ]
    weak_aggregate = MODULE._aggregate(weaker)
    assert MODULE._block_hypothesis_passes(weak_aggregate)["distance"] is False


def test_mode_and_overall_verdict_mapping() -> None:
    assert MODULE._mode_verdict(4) == "SUPPORTED"
    assert MODULE._mode_verdict(2) == "PARTIAL"
    assert MODULE._mode_verdict(0) == "FALSIFIED"

    assert MODULE._overall_decision("SUPPORTED", "SUPPORTED", False)["decision"] == (
        "NAV_BIAS_MECHANISM_SUPPORTED"
    )
    assert MODULE._overall_decision("SUPPORTED", "PARTIAL", False)["decision"] == (
        "DISTANCE_METRIC_BIAS_SUPPORTED"
    )
    assert MODULE._overall_decision("PARTIAL", "SUPPORTED", False)["decision"] == (
        "DIRECTION_BIAS_SUPPORTED"
    )
    assert MODULE._overall_decision("PARTIAL", "FALSIFIED", False)["decision"] == (
        "PARTIAL_NAV_BIAS"
    )
    assert MODULE._overall_decision("FALSIFIED", "FALSIFIED", False)["decision"] == (
        "NO_SUPPORTED_NAV_BIAS"
    )
    assert MODULE._overall_decision("SUPPORTED", "SUPPORTED", True)["decision"] == (
        "INVALID_DIAGNOSTIC"
    )
