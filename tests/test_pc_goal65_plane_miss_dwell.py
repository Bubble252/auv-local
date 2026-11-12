from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal65_plane_miss_dwell.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal65_plane_miss_dwell", ANALYSIS_SCRIPT
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
    safety: bool = False,
    p_nav: float = 1.0,
    eta: float = 1.0,
) -> dict[str, object]:
    return {
        "seed": seed,
        "steps": steps,
        "success": success,
        "terminal_eta_capture": capture,
        "goal_plane_miss": False,
        "timeout": timeout,
        "safety": safety,
        "terminal_p_nav_distance_m": p_nav,
        "terminal_eta_distance_m": eta,
    }


def test_paired_metrics_tracks_immediate_false_success_and_deltas() -> None:
    base = [
        _row(20, success=True, capture=False),
        _row(21, success=True, capture=False),
        _row(22, success=True, capture=True),
        _row(23, success=True, capture=True),
    ]
    immediate = [
        _row(20, success=False, capture=False),
        _row(21, success=True, capture=True),
        _row(22, success=True, capture=True),
        _row(23, success=True, capture=True),
    ]
    candidate = [
        _row(20, success=True, capture=True, steps=130),
        _row(21, success=False, capture=False, timeout=True, steps=180),
        _row(22, success=True, capture=True, steps=120),
        _row(23, success=True, capture=True, steps=125),
    ]

    stats = ANALYSIS._paired_metrics(base, immediate, candidate)

    assert stats["baseline_false_success_count"] == 2
    assert stats["immediate_false_success_count"] == 0
    assert stats["candidate_false_success_count"] == 0
    assert stats["false_success_removed_fraction"] == pytest.approx(1.0)
    assert stats["true_success_retention"] == pytest.approx(1.0)
    assert stats["candidate_precision"] == pytest.approx(1.0)
    assert stats["timeout_delta"] == pytest.approx(0.25)
    assert stats["extra_steps_mean"] == pytest.approx(38.75)


def test_h65_a_and_c_threshold_boundaries() -> None:
    default_stats = {
        "baseline_false_success_count": 10,
        "true_success_retention": ANALYSIS.TRUE_RETENTION_MIN["default"],
        "false_success_removed_fraction": ANALYSIS.FALSE_REMOVAL_MIN,
    }
    assert ANALYSIS._h65_a(default_stats, "default") is True
    default_stats["true_success_retention"] -= 1e-9
    assert ANALYSIS._h65_a(default_stats, "default") is False

    strict_stats = {
        "baseline_false_success_count": 10,
        "true_success_retention": ANALYSIS.TRUE_RETENTION_MIN["strict125"],
        "false_success_removed_fraction": ANALYSIS.FALSE_REMOVAL_MIN,
    }
    assert ANALYSIS._h65_a(strict_stats, "strict125") is True
    assert ANALYSIS._h65_c({"true_success_retention": ANALYSIS.ZERO_RETENTION_MIN}) is True
    assert ANALYSIS._h65_c({"true_success_retention": 0.79}) is False


def test_h65_b_false_success_and_checkpoint_guards() -> None:
    stats = {
        "candidate_false_success_count": 4,
        "immediate_false_success_count": 2,
        "candidate_precision": ANALYSIS.PRECISION_MIN,
        "timeout_delta": ANALYSIS.TERMINAL_DELTA_MAX,
        "safety_delta": ANALYSIS.TERMINAL_DELTA_MAX,
    }
    passing = {"true_success_retention": ANALYSIS.CHECKPOINT_RETENTION_MIN}
    failing = {"true_success_retention": 0.79}
    runs = {str(seed): passing for seed in ANALYSIS.POLICY_SEEDS[:3]}
    runs[str(ANALYSIS.POLICY_SEEDS[3])] = failing
    result = ANALYSIS._h65_b(stats, runs)
    assert result["H65_B_supported"] is True
    assert result["checkpoint_pass_count"] == 3

    stats["candidate_false_success_count"] = 5
    result = ANALYSIS._h65_b(stats, runs)
    assert result["false_success_guard_passed"] is False
    assert result["H65_B_supported"] is False


def test_overall_status_priority() -> None:
    def scenario(*, a: bool, b: bool, c: bool, false_guard: bool = True) -> dict:
        return {
            "hypotheses": {
                "H65_A_supported": a,
                "H65_B_supported": b,
                "H65_C_supported": c,
                "false_success_guard_passed": false_guard,
                "terminal_guard_passed": True,
            }
        }

    passing = {"default": scenario(a=True, b=True, c=True)}
    assert ANALYSIS._overall_status(passing) == "DWELL_PASS_DIAGNOSTIC"
    insufficient = {"default": scenario(a=False, b=True, c=True)}
    assert ANALYSIS._overall_status(insufficient) == "DWELL_NOT_SUFFICIENT"
    partial = {"default": scenario(a=True, b=False, c=True)}
    assert ANALYSIS._overall_status(partial) == "DWELL_PARTIAL_DIAGNOSTIC"
    invalid = {"default": scenario(a=False, b=False, c=True, false_guard=False)}
    assert ANALYSIS._overall_status(invalid) == "DWELL_INVALID_FALSE_SUCCESS"
