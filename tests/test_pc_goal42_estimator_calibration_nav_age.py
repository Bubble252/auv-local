from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal42_estimator_calibration_nav_age.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pc_goal42_estimator_calibration_nav_age", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_interpolate_position_is_linear() -> None:
    result = MODULE._interpolate_position(
        time_s=np.asarray([0.0, 1.0, 2.0]),
        eta_xyz=np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0], [4.0, 2.0, -2.0]]),
        query_time_s=0.5,
    )

    np.testing.assert_allclose(result, [1.0, 0.5, -0.5])


def test_interpolate_position_rejects_out_of_range_query() -> None:
    with pytest.raises(ValueError, match="outside the trajectory"):
        MODULE._interpolate_position(
            time_s=np.asarray([0.0, 1.0]),
            eta_xyz=np.zeros((2, 3)),
            query_time_s=-0.1,
        )


def test_terminal_age_metrics_decomposes_raw_delay_and_residual() -> None:
    result = MODULE._terminal_age_metrics(
        p_nav=np.asarray([1.5, 0.0, 0.0]),
        eta=np.asarray([1.0, 0.0, 0.0]),
        goal=np.asarray([2.0, 0.0, 0.0]),
        nav_age_s=1.0,
        usbl_age_s=0.5,
        dvl_bottom_age_s=0.2,
        current_estimate_age_s=0.3,
        time_s=np.asarray([0.0, 1.0]),
        eta_trajectory_xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        terminal_time_s=1.0,
    )

    assert result["raw_projection_m"] == pytest.approx(0.5)
    assert result["delay_projection_m"] == pytest.approx(-1.0)
    assert result["residual_projection_m"] == pytest.approx(1.5)
    assert result["raw_toward_goal"] is True
    assert result["residual_toward_goal"] is True


def test_terminal_age_metrics_rejects_age_beyond_terminal_time() -> None:
    with pytest.raises(ValueError, match="exceeds terminal time"):
        MODULE._terminal_age_metrics(
            p_nav=np.asarray([1.5, 0.0, 0.0]),
            eta=np.asarray([1.0, 0.0, 0.0]),
            goal=np.asarray([2.0, 0.0, 0.0]),
            nav_age_s=1.1,
            usbl_age_s=0.5,
            dvl_bottom_age_s=0.2,
            current_estimate_age_s=0.3,
            time_s=np.asarray([0.0, 1.0]),
            eta_trajectory_xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            terminal_time_s=1.0,
        )


def test_bootstrap_median_difference_is_deterministic_and_separated() -> None:
    first = MODULE._bootstrap_median_difference([1.0] * 8, [0.0] * 12)
    second = MODULE._bootstrap_median_difference([1.0] * 8, [0.0] * 12)

    assert first == second
    assert first == pytest.approx((1.0, 1.0))


def _record(seed: int, *, discordant: bool, age_s: float) -> dict[str, Any]:
    return {
        "seed": seed,
        "discordant": discordant,
        "nav_age_s": age_s,
        "usbl_age_s": 0.1,
        "raw_projection_m": 0.5,
        "delay_projection_m": -0.2,
        "residual_projection_m": 0.3,
        "raw_toward_goal": True,
        "residual_toward_goal": True,
    }


def test_aggregate_applies_frozen_sample_and_fraction_rules() -> None:
    records = [
        _record(seed, discordant=True, age_s=0.5) for seed in range(20, 25)
    ]
    records.extend(
        _record(seed, discordant=False, age_s=0.1) for seed in range(25, 40)
    )

    aggregate = MODULE._aggregate(records)

    assert aggregate["discordant_seed_count"] == 5
    assert aggregate["non_discordant_seed_count"] == 15
    assert aggregate["nav_age_median_difference_s"] == pytest.approx(0.4)
    assert aggregate["residual_toward_goal_fraction"] == pytest.approx(1.0)


def test_mode_and_overall_verdict_mapping() -> None:
    assert MODULE._mode_verdict(4) == "SUPPORTED"
    assert MODULE._mode_verdict(2) == "PARTIAL"
    assert MODULE._mode_verdict(0) == "FALSIFIED"

    assert MODULE._overall_decision("SUPPORTED", "SUPPORTED", False)["decision"] == (
        "AGE_ASSOCIATED_WITH_RESIDUAL_BIAS"
    )
    assert MODULE._overall_decision("FALSIFIED", "SUPPORTED", False)["decision"] == (
        "RESIDUAL_CALIBRATION_BIAS_SUPPORTED"
    )
    assert MODULE._overall_decision("SUPPORTED", "FALSIFIED", False)["decision"] == (
        "NAV_AGE_ASSOCIATION_ONLY"
    )
    assert MODULE._overall_decision("PARTIAL", "FALSIFIED", False)["decision"] == (
        "PARTIAL_AGE_BIAS"
    )
    assert MODULE._overall_decision("FALSIFIED", "FALSIFIED", False)["decision"] == (
        "NO_SINGLE_SOURCE_SUPPORTED"
    )
    assert MODULE._overall_decision("SUPPORTED", "SUPPORTED", True)["decision"] == (
        "INVALID_REPLAY"
    )
