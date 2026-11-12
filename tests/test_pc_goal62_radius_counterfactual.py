from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal62_terminal_radius_counterfactual.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal62_terminal_radius_counterfactual", ANALYSIS_SCRIPT
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


def _row(
    seed: int,
    *,
    success: bool,
    capture: bool,
    p_nav_distance: float,
    eta_distance: float,
    radius: float = 2.0,
) -> dict[str, object]:
    return {
        "scenario": "default",
        "run": "test",
        "seed": seed,
        "success": success,
        "capture": capture,
        "p_nav_goal_distance_m": p_nav_distance,
        "eta_goal_distance_m": eta_distance,
        "goal_radius_m": radius,
    }


def test_metrics_removes_false_success_and_retains_true_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ANALYSIS, "BOOTSTRAP_DRAWS", 128)
    rows = [
        _row(20, success=True, capture=False, p_nav_distance=1.9, eta_distance=2.3),
        _row(21, success=True, capture=False, p_nav_distance=1.8, eta_distance=2.2),
        _row(22, success=True, capture=True, p_nav_distance=1.5, eta_distance=1.0),
        _row(23, success=True, capture=True, p_nav_distance=1.6, eta_distance=1.1),
    ]

    stats = ANALYSIS._metrics(rows, 0.25, seed=1)

    assert stats["tight_radius_m"] == pytest.approx(1.75)
    assert stats["baseline_false_success_count"] == 2
    assert stats["candidate_false_success_count"] == 0
    assert stats["false_success_removed_fraction"] == pytest.approx(1.0)
    assert stats["true_success_retention"] == pytest.approx(1.0)
    assert stats["candidate_precision"] == pytest.approx(1.0)


def test_metrics_reports_zero_removal_when_margin_keeps_false_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ANALYSIS, "BOOTSTRAP_DRAWS", 128)
    rows = [
        _row(20, success=True, capture=False, p_nav_distance=1.9, eta_distance=2.3),
        _row(21, success=True, capture=True, p_nav_distance=1.5, eta_distance=1.0),
    ]

    stats = ANALYSIS._metrics(rows, 0.10, seed=1)

    assert stats["false_success_removed_fraction"] == pytest.approx(0.0)
    assert stats["true_success_retention"] == pytest.approx(1.0)


def test_metrics_returns_none_for_zero_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ANALYSIS, "BOOTSTRAP_DRAWS", 128)
    rows = [_row(20, success=True, capture=True, p_nav_distance=1.5, eta_distance=1.0)]

    stats = ANALYSIS._metrics(rows, 0.25, seed=1)

    assert stats["false_success_removed_fraction"] is None
    assert stats["false_success_removed_ci"] is None
    assert stats["true_success_retention"] == pytest.approx(1.0)


def test_rule_boundaries_are_inclusive() -> None:
    stats = {
        "baseline_false_success_count": 10,
        "false_success_removed_fraction": ANALYSIS.FALSE_REMOVAL_MIN,
        "true_success_retention": ANALYSIS.TRUE_RETENTION_MIN,
    }
    assert ANALYSIS._rule_pass(stats) is True
    stats["false_success_removed_fraction"] -= 1.0e-9
    assert ANALYSIS._rule_pass(stats) is False


def test_checkpoint_and_zero_boundaries() -> None:
    checkpoint = {
        "false_success_removed_fraction": ANALYSIS.CHECKPOINT_FALSE_REMOVAL_MIN,
        "true_success_retention": ANALYSIS.CHECKPOINT_TRUE_RETENTION_MIN,
    }
    assert ANALYSIS._checkpoint_pass(checkpoint) is True
    zero = {
        "false_success_removed_fraction": 1.0e-12,
        "true_success_retention": ANALYSIS.ZERO_TRUE_RETENTION_MIN,
    }
    assert ANALYSIS._zero_pass(zero) is True
    zero["false_success_removed_fraction"] = 0.0
    assert ANALYSIS._zero_pass(zero) is False


def test_degenerate_boundary_is_invalid_not_fail() -> None:
    stats = {
        "baseline_success_count": 12,
        "candidate_success_count": 0,
    }
    scenario = {
        "pooled": {
            "primary_m025": stats,
            "sens_m015": stats,
            "sens_m035": stats,
        },
        "hypotheses": {
            "H_R3_supported": False,
            "sensitivity_pass": {"sens_m015": False, "sens_m035": False},
        },
    }
    scenarios = {"default": scenario, "strict125": scenario}

    assert ANALYSIS._degenerate_boundary(scenario) is True
    assert ANALYSIS._classify_status(scenarios) == (
        "INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY"
    )


def test_non_degenerate_failure_keeps_fail_status() -> None:
    primary = {
        "baseline_success_count": 12,
        "candidate_success_count": 2,
    }
    other = {
        "baseline_success_count": 12,
        "candidate_success_count": 0,
    }
    scenario = {
        "pooled": {
            "primary_m025": primary,
            "sens_m015": other,
            "sens_m035": other,
        },
        "hypotheses": {
            "H_R3_supported": False,
            "sensitivity_pass": {"sens_m015": False, "sens_m035": False},
        },
    }

    assert ANALYSIS._degenerate_boundary(scenario) is False
    assert ANALYSIS._classify_status({"default": scenario, "strict125": scenario}) == (
        "FAIL_STATIC_RADIUS"
    )


def test_terminal_rows_replays_baseline_labels(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.npz"
    np.savez(
        trace_path,
        seed=np.asarray([20, 20, 21, 21], dtype=np.int64),
        done=np.asarray([False, True, False, True]),
        p_nav_ned=np.asarray(
            [
                [9.0, 9.0, 9.0],
                [1.5, 0.0, 0.0],
                [9.0, 9.0, 9.0],
                [2.5, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        eta_ned=np.asarray(
            [
                [9.0, 9.0, 9.0],
                [2.5, 0.0, 0.0],
                [9.0, 9.0, 9.0],
                [1.5, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        goal_ned=np.zeros((4, 3), dtype=np.float64),
    )
    episodes = [
        {"seed": 20, "success": True, "terminal_eta_capture": False},
        {"seed": 21, "success": False, "terminal_eta_capture": True},
    ]

    rows = ANALYSIS._terminal_rows(
        trace_path,
        episodes,
        goal_radius=2.0,
        label="test",
        scenario="default",
    )

    assert [row["seed"] for row in rows] == [20, 21]
    assert rows[0]["p_nav_goal_distance_m"] == pytest.approx(1.5)
    assert rows[0]["eta_goal_distance_m"] == pytest.approx(2.5)
    assert rows[1]["p_nav_goal_distance_m"] == pytest.approx(2.5)
    assert rows[1]["eta_goal_distance_m"] == pytest.approx(1.5)
