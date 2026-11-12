from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_pc_goal25_d3_gates import evaluate_d3_gates  # noqa: E402


def _cell(*, success: float, rmse: float, safety: float = 0.0) -> dict[str, float]:
    return {"success_rate": success, "pre_goal_3d_rmse_m": rmse, "safety_rate": safety}


def _comparison(default: dict[str, float], strict: dict[str, float]) -> dict[str, object]:
    return {
        "schema_version": "auv_pc_eval_matrix_comparison.v1",
        "cells": [
            {"scenario": "default", "path_mode": "normal", "student": default},
            {
                "scenario": "default",
                "path_mode": "path_zero",
                "student": _cell(success=0.0, rmse=default["pre_goal_3d_rmse_m"] * 1.2),
            },
            {"scenario": "strict125", "path_mode": "normal", "student": strict},
            {
                "scenario": "strict125",
                "path_mode": "path_zero",
                "student": _cell(success=0.0, rmse=strict["pre_goal_3d_rmse_m"] * 1.2),
            },
        ],
    }


def test_d3_gate_passes_when_all_thresholds_hold() -> None:
    d2 = _comparison(
        default=_cell(success=0.725, rmse=2.7883),
        strict=_cell(success=0.375, rmse=2.7648),
    )
    d3 = _comparison(
        default=_cell(success=0.725, rmse=2.6400),
        strict=_cell(success=0.400, rmse=2.6900),
    )

    result = evaluate_d3_gates(d3=d3, d2=d2, manifest_present=True)

    assert result["overall"] is True
    assert result["decision"] == "PASS"


def test_d3_gate_rejects_pre_goal_regression() -> None:
    d2 = _comparison(
        default=_cell(success=0.725, rmse=2.7883),
        strict=_cell(success=0.375, rmse=2.7648),
    )
    d3 = _comparison(
        default=_cell(success=0.725, rmse=2.8492),
        strict=_cell(success=0.500, rmse=2.8291),
    )

    result = evaluate_d3_gates(d3=d3, d2=d2, manifest_present=True)

    assert result["overall"] is False
    assert result["decision"] == "FAIL"
    assert result["scenarios"]["default"]["checks"]["pre_goal_le_limit"] is False
    assert result["scenarios"]["strict125"]["checks"]["pre_goal_le_limit"] is False
