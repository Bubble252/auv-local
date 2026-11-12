from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal59_paired.py"
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal59_paired", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _summary(*, capture: float, success: float, plane: float, rmse: float = 2.6):
    return {
        "success_rate": success,
        "goal_plane_miss_rate": plane,
        "safety_termination_rate": 0.0,
        "timeout_rate": 0.0,
        "terminal_eta_capture_rate": capture,
        "pre_goal_3d_rmse_m": rmse,
        "gate_active_rate": 0.25,
        "residual_l2_ratio_gate_active": 0.14,
        "action_saturation_fraction": 0.05,
        "finite": True,
    }


def _row(seed: int, *, capture_delta: float, success_delta: float, plane_delta: float):
    zero = _summary(capture=0.4, success=0.72, plane=0.28)
    policy = _summary(
        capture=0.4 + capture_delta,
        success=0.72 + success_delta,
        plane=0.28 + plane_delta,
    )
    scenarios = {
        "default": MODULE._scenario_row(policy, zero),
        "strict125": MODULE._scenario_row(policy, zero),
    }
    return MODULE._seed_row(seed, scenarios, probe={"raw_saturation_fraction_any_dim": 0.05})


def test_paired_row_computes_deltas_and_gate_checks() -> None:
    row = _row(240, capture_delta=0.08, success_delta=0.0, plane_delta=-0.05)
    assert row["scenarios"]["default"]["delta_capture"] == pytest.approx(0.08)
    assert row["checks"]["main_effect"] is True
    assert row["checks"]["success"] is True
    assert row["checks"]["plane_miss"] is True
    assert row["seed_pass"] is True


def test_success_guardrail_can_fail_seed_even_with_capture_gain() -> None:
    row = _row(240, capture_delta=0.08, success_delta=-0.20, plane_delta=0.20)
    assert row["checks"]["main_effect"] is True
    assert row["checks"]["success"] is False
    assert row["seed_pass"] is False


def test_exact_threshold_capture_gain_is_not_lost_to_float_roundoff() -> None:
    row = _row(240, capture_delta=0.25 - 0.20, success_delta=0.0, plane_delta=0.0)
    assert row["checks"]["main_effect"] is True


def test_overall_status_is_pass_with_six_seed_passes() -> None:
    rows = [
        _row(seed, capture_delta=0.08, success_delta=0.0, plane_delta=0.0)
        for seed in (240, 368, 496, 624, 752, 880)
    ]
    result = MODULE._overall(rows, ["default", "strict125"], draws=200)
    assert result["status"] == "PASS"
    assert result["seed_pass_count"] == 6
    assert result["per_scenario"]["default"]["delta_capture_ci95"][0] > 0.0


def test_overall_status_fails_without_main_effect() -> None:
    rows = [
        _row(seed, capture_delta=0.0, success_delta=0.0, plane_delta=0.0)
        for seed in (240, 368, 496, 624)
    ]
    result = MODULE._overall(rows, ["default", "strict125"], draws=200)
    assert result["status"] == "FAIL"
    assert result["main_effect_seed_count"] == 0
