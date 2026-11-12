from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_pc_goal26_gates import EXPECTED_SEEDS, evaluate_goal26_screen  # noqa: E402


def _metrics(
    *,
    success: float,
    pre_goal: float,
    safety: float = 0.0,
    plane_miss: float = 0.0,
    timeout: float = 0.0,
) -> dict[str, float]:
    return {
        "success_rate": success,
        "pre_goal_3d_rmse_m": pre_goal,
        "legacy_full_3d_rmse_m": pre_goal,
        "safety_rate": safety,
        "goal_plane_miss_rate": plane_miss,
        "timeout_rate": timeout,
        "material_saturation_rate": 0.2,
    }


def _baseline() -> dict[str, object]:
    return {
        "schema_version": "auv_pc_eval_matrix_comparison.v1",
        "cells": [
            {
                "scenario": "default",
                "path_mode": "normal",
                "student": _metrics(success=0.725, pre_goal=2.85, plane_miss=0.275),
            },
            {
                "scenario": "strict125",
                "path_mode": "normal",
                "student": _metrics(success=0.50, pre_goal=2.83, plane_miss=0.50),
            },
        ],
    }


def _candidate(
    *,
    default: dict[str, float] | None = None,
    strict: dict[str, float] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    normal = {
        "default": default or _metrics(success=0.725, pre_goal=2.70, plane_miss=0.275),
        "strict125": strict or _metrics(success=0.50, pre_goal=2.80, plane_miss=0.50),
    }
    zero = {
        "default": _metrics(success=0.0, pre_goal=6.30, safety=0.125, plane_miss=0.875),
        "strict125": _metrics(success=0.0, pre_goal=6.30, safety=0.125, plane_miss=0.875),
    }
    return normal, zero


def _evaluate(
    normal: dict[str, dict[str, float]], zero: dict[str, dict[str, float]]
) -> dict[str, object]:
    return evaluate_goal26_screen(baseline=_baseline(), normal=normal, path_zero=zero)


def test_goal26_gate_passes_with_guardrails_and_effect() -> None:
    normal, zero = _candidate()
    result = _evaluate(normal, zero)

    assert result["overall"] is True
    assert result["decision"] == "PASS"
    assert result["checks"]["effect_gate"] is True


def test_goal26_gate_rejects_success_drop() -> None:
    normal, zero = _candidate(default=_metrics(success=0.68, pre_goal=2.60, plane_miss=0.32))
    result = _evaluate(normal, zero)

    assert result["overall"] is False
    assert result["scenarios"]["default"]["checks"]["success_ge_floor"] is False


def test_goal26_gate_rejects_no_effect() -> None:
    normal, zero = _candidate(
        default=_metrics(success=0.725, pre_goal=2.80, plane_miss=0.275),
        strict=_metrics(success=0.50, pre_goal=2.80, plane_miss=0.50),
    )
    result = _evaluate(normal, zero)

    assert result["overall"] is False
    assert result["checks"]["effect_gate"] is False


def test_goal26_gate_rejects_other_scenario_worsening_over_three_percent() -> None:
    normal, zero = _candidate(
        default=_metrics(success=0.725, pre_goal=2.60, plane_miss=0.275),
        strict=_metrics(success=0.50, pre_goal=2.92, plane_miss=0.50),
    )
    result = _evaluate(normal, zero)

    assert result["overall"] is False
    assert result["effect_checks"]["default_improved_with_strict_worsen_le_3pct"] is False


def test_goal26_gate_does_not_mutate_inputs() -> None:
    normal, zero = _candidate()
    normal_before = copy.deepcopy(normal)
    zero_before = copy.deepcopy(zero)

    _evaluate(normal, zero)

    assert normal == normal_before
    assert zero == zero_before
    assert EXPECTED_SEEDS == list(range(20, 60))
