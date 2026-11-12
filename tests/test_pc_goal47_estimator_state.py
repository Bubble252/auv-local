from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal47_estimator_state.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_goal47_estimator_state", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_verdict_uses_frozen_block_counts() -> None:
    assert MODULE._verdict([True, True, True, True]) == "SUPPORTED"
    assert MODULE._verdict([True, False, False, False]) == "PARTIAL"
    assert MODULE._verdict([False, False, False, False]) == "FALSIFIED"


def test_overall_decision_prioritizes_realized_ou_mediation() -> None:
    assert (
        MODULE._overall_decision(
            h_ou="SUPPORTED",
            h_usbl="FALSIFIED",
            h_dvl="FALSIFIED",
            h_sigma_partial="SUPPORTED",
        )
        == "REALIZED_OU_EXPLAINS_SIGMA_ASSOCIATION"
    )
    assert (
        MODULE._overall_decision(
            h_ou="SUPPORTED",
            h_usbl="SUPPORTED",
            h_dvl="FALSIFIED",
            h_sigma_partial="SUPPORTED",
        )
        == "MIXED_REALIZED_OU_USBL"
    )
    assert (
        MODULE._overall_decision(
            h_ou="PARTIAL",
            h_usbl="FALSIFIED",
            h_dvl="FALSIFIED",
            h_sigma_partial="FALSIFIED",
        )
        == "PARTIAL_ESTIMATOR_COMPONENT_EVIDENCE"
    )


def test_rankdata_averages_ties() -> None:
    ranks = MODULE._rankdata([10.0, 20.0, 20.0, 30.0])
    np.testing.assert_allclose(ranks, [1.0, 2.5, 2.5, 4.0])


def test_partial_spearman_removes_realized_ou_control() -> None:
    sigma = np.linspace(-2.0, 2.0, 31)
    realized_ou = 3.0 * sigma + np.sin(sigma)
    dynamic = realized_ou + 0.02 * np.cos(3.0 * sigma)

    unadjusted, partial, drop = MODULE._partial_spearman(
        sigma, dynamic, realized_ou
    )

    assert unadjusted > 0.95
    assert abs(partial) < 0.30
    assert drop > 0.20


def test_component_shares_are_signed_and_sum_to_one() -> None:
    dynamic = 0.40
    ou = 0.30
    event = 0.10
    ou_share, defined = MODULE._share(ou, dynamic)
    event_share, event_defined = MODULE._share(event, dynamic)

    assert defined and event_defined
    assert ou_share == pytest.approx(0.75)
    assert event_share == pytest.approx(0.25)
    assert ou_share + event_share == pytest.approx(1.0)

    undefined_share, undefined = MODULE._share(ou, 0.0)
    assert undefined_share == 0.0
    assert undefined is False


def test_projected_component_means_preserve_additive_identity() -> None:
    rng = np.random.default_rng(47)
    direction = np.asarray([0.3, -0.5, 0.81], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    nav_ou = rng.normal(scale=0.08, size=(257, 3))
    nav_event_usbl = rng.normal(scale=0.03, size=(257, 3))
    nav_event_dvl = rng.normal(scale=0.01, size=(257, 3))
    nav_event = nav_event_usbl + nav_event_dvl
    dynamic = nav_ou + nav_event

    means = MODULE._projected_component_means(
        exact_dynamic_rows=dynamic,
        nav_ou_rows=nav_ou,
        event_delta_rows=nav_event,
        nav_event_usbl_rows=nav_event_usbl,
        nav_event_dvl_rows=nav_event_dvl,
        goal_direction=direction,
    )

    dynamic_mean = means["dynamic_projection_mean_m"]
    ou_mean = means["ou_projection_mean_m"]
    event_mean = means["event_projection_mean_m"]
    usbl_mean = means["usbl_projection_mean_m"]
    dvl_mean = means["dvl_projection_mean_m"]

    assert abs(dynamic_mean - ou_mean - event_mean) <= 1.0e-14
    assert abs(event_mean - usbl_mean - dvl_mean) <= 1.0e-14
    rowwise_error = np.max(np.abs((dynamic - nav_ou - nav_event) @ direction))
    assert rowwise_error <= 1.0e-14


def test_late_window_rejects_short_window() -> None:
    times = np.arange(40, dtype=np.float64) * MODULE.DT_S
    exact_error = np.zeros((times.size, 3), dtype=np.float64)
    valid = np.ones(times.size, dtype=bool)

    with pytest.raises(ValueError, match="late window"):
        MODULE._late_window_indices(
            exact_error=exact_error,
            valid_rows=valid,
            time_s=times,
        )
