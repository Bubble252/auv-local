from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_pc_goal24_stress_gates import evaluate_arm  # noqa: E402


def _cell(
    *,
    success: float,
    rmse: float,
    safety: float = 0.0,
) -> dict[str, float]:
    return {
        "success_rate": success,
        "pre_goal_3d_rmse_m": rmse,
        "safety_rate": safety,
    }


def _comparison(
    *,
    normal_default: dict[str, float],
    normal_strict: dict[str, float],
) -> dict[str, object]:
    return {
        "schema_version": "auv_pc_eval_matrix_comparison.v1",
        "cells": [
            {"scenario": "default", "path_mode": "normal", "student": normal_default},
            {
                "scenario": "default",
                "path_mode": "path_zero",
                "student": _cell(success=0.0, rmse=10.0),
            },
            {"scenario": "strict125", "path_mode": "normal", "student": normal_strict},
            {
                "scenario": "strict125",
                "path_mode": "path_zero",
                "student": _cell(success=0.0, rmse=10.0),
            },
        ],
    }


def _reference() -> dict[str, object]:
    return _comparison(
        normal_default=_cell(success=0.70, rmse=2.80),
        normal_strict=_cell(success=0.35, rmse=2.80),
    )


def _stress_reference() -> dict[str, object]:
    return _comparison(
        normal_default=_cell(success=0.45, rmse=3.20),
        normal_strict=_cell(success=0.25, rmse=3.20),
    )


def test_stress_gate_passes_with_nominal_guardrail_and_success_gain() -> None:
    nominal = _comparison(
        normal_default=_cell(success=0.69, rmse=2.82),
        normal_strict=_cell(success=0.34, rmse=2.82),
    )
    stress = _comparison(
        normal_default=_cell(success=0.56, rmse=3.00),
        normal_strict=_cell(success=0.36, rmse=3.00),
    )

    result = evaluate_arm(
        nominal=nominal,
        stress=stress,
        nominal_reference=_reference(),
        stress_reference=_stress_reference(),
        arm="s10",
    )

    assert result["overall"] is True
    assert all(item["overall"] for item in result["scenarios"].values())


def test_stress_gate_rejects_nominal_rmse_above_two_percent() -> None:
    nominal = _comparison(
        normal_default=_cell(success=0.70, rmse=2.90),
        normal_strict=_cell(success=0.35, rmse=2.82),
    )
    stress = _comparison(
        normal_default=_cell(success=0.80, rmse=2.50),
        normal_strict=_cell(success=0.45, rmse=2.50),
    )

    result = evaluate_arm(
        nominal=nominal,
        stress=stress,
        nominal_reference=_reference(),
        stress_reference=_stress_reference(),
        arm="s20",
    )

    assert result["scenarios"]["default"]["checks"]["nominal_pre_goal_within_2pct"] is False
    assert result["overall"] is False


def test_stress_gate_accepts_rmse_improvement_without_success_gain() -> None:
    nominal = _comparison(
        normal_default=_cell(success=0.70, rmse=2.80),
        normal_strict=_cell(success=0.35, rmse=2.80),
    )
    stress = _comparison(
        normal_default=_cell(success=0.45, rmse=3.09),
        normal_strict=_cell(success=0.25, rmse=3.09),
    )

    result = evaluate_arm(
        nominal=nominal,
        stress=stress,
        nominal_reference=_reference(),
        stress_reference=_stress_reference(),
        arm="s10",
    )

    assert result["overall"] is True
    assert result["scenarios"]["default"]["checks"]["stress_gain_either"] is True


def test_stress_gate_rejects_no_meaningful_stress_gain() -> None:
    nominal = _comparison(
        normal_default=_cell(success=0.70, rmse=2.80),
        normal_strict=_cell(success=0.35, rmse=2.80),
    )
    stress = _comparison(
        normal_default=_cell(success=0.46, rmse=3.15),
        normal_strict=_cell(success=0.26, rmse=3.15),
    )

    result = evaluate_arm(
        nominal=nominal,
        stress=stress,
        nominal_reference=_reference(),
        stress_reference=_stress_reference(),
        arm="s20",
    )

    assert result["scenarios"]["default"]["checks"]["stress_gain_either"] is False
    assert result["overall"] is False
