from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal63_terminal_suppression.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal63_terminal_suppression", ANALYSIS_SCRIPT
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


def _row(
    seed: int,
    *,
    success: bool,
    capture: bool,
    steps: int = 100,
    timeout: bool = False,
    plane: bool = False,
    safety: bool = False,
    p_nav: float = 1.0,
    eta: float = 1.0,
) -> dict[str, object]:
    return {
        "seed": seed,
        "steps": steps,
        "success": success,
        "terminal_eta_capture": capture,
        "goal_plane_miss": plane,
        "timeout": timeout,
        "safety": safety,
        "terminal_p_nav_distance_m": p_nav,
        "terminal_eta_distance_m": eta,
    }


def test_paired_metrics_removes_false_and_retains_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ANALYSIS, "BOOTSTRAP_DRAWS", 64)
    base = [
        _row(20, success=True, capture=False, p_nav=1.9, eta=2.2),
        _row(21, success=True, capture=False, p_nav=1.8, eta=2.1),
        _row(22, success=True, capture=True, p_nav=1.2, eta=1.0),
        _row(23, success=True, capture=True, p_nav=1.1, eta=0.9),
    ]
    cand = [
        _row(20, success=True, capture=True, steps=140, p_nav=1.6, eta=1.1),
        _row(21, success=False, capture=False, timeout=True, steps=180, p_nav=1.9, eta=2.0),
        _row(22, success=True, capture=True, steps=120, p_nav=1.5, eta=1.2),
        _row(23, success=True, capture=True, steps=125, p_nav=1.4, eta=1.1),
    ]

    stats = ANALYSIS._paired_metrics(base, cand, seed=1)

    assert stats["baseline_false_success_count"] == 2
    assert stats["candidate_false_success_count"] == 0
    assert stats["false_success_removed_fraction"] == pytest.approx(1.0)
    assert stats["false_to_true_count"] == 1
    assert stats["false_to_none_count"] == 1
    assert stats["true_success_retention"] == pytest.approx(1.0)
    assert stats["candidate_precision"] == pytest.approx(1.0)
    assert stats["candidate_timeout_rate"] == pytest.approx(0.25)
    assert stats["extra_steps_mean"] == pytest.approx(41.25)


def test_rule_and_checkpoint_boundaries() -> None:
    stats = {
        "baseline_false_success_count": 10,
        "false_success_removed_fraction": ANALYSIS.FALSE_REMOVAL_MIN,
        "true_success_retention": ANALYSIS.TRUE_RETENTION_MIN,
        "baseline_safety_rate": 0.0,
        "candidate_safety_rate": ANALYSIS.SAFETY_DELTA_MAX,
    }
    assert ANALYSIS._rule_pass(stats) is True
    stats["candidate_safety_rate"] += 1e-9
    assert ANALYSIS._rule_pass(stats) is False

    checkpoint = {
        "false_success_removed_fraction": ANALYSIS.CHECKPOINT_FALSE_REMOVAL_MIN,
        "true_success_retention": ANALYSIS.CHECKPOINT_TRUE_RETENTION_MIN,
    }
    assert ANALYSIS._checkpoint_pass(checkpoint) is True
    checkpoint["true_success_retention"] -= 1e-9
    assert ANALYSIS._checkpoint_pass(checkpoint) is False


def test_zero_boundary() -> None:
    stats = {
        "false_success_removed_fraction": 1e-12,
        "true_success_retention": ANALYSIS.ZERO_TRUE_RETENTION_MIN,
    }
    assert ANALYSIS._zero_pass(stats) is True
    stats["false_success_removed_fraction"] = 0.0
    assert ANALYSIS._zero_pass(stats) is False


def test_hypotheses_require_checkpoint_count() -> None:
    pooled = {
        "baseline_false_success_count": 10,
        "false_success_removed_fraction": 0.6,
        "true_success_retention": 0.95,
        "baseline_safety_rate": 0.0,
        "candidate_safety_rate": 0.0,
    }
    passing = {
        "false_success_removed_fraction": 0.5,
        "true_success_retention": 0.9,
    }
    failing = {
        "false_success_removed_fraction": 0.1,
        "true_success_retention": 0.9,
    }
    runs = {str(seed): passing for seed in ANALYSIS.POLICY_SEEDS[:3]}
    runs[str(ANALYSIS.POLICY_SEEDS[3])] = failing
    zero = {"false_success_removed_fraction": 0.1, "true_success_retention": 0.8}

    result = ANALYSIS._hypotheses(pooled, runs, zero)

    assert result["H63_A_supported"] is True
    assert result["H63_B_supported"] is True
    assert result["H63_C_supported"] is True
