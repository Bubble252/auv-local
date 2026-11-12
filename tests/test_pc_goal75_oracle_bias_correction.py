from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal75_oracle_bias_correction.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal75_oracle_bias_correction", ANALYSIS_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def _fields(
    *,
    eta: tuple[float, float, float],
    p_nav: tuple[float, float, float],
    tight_radius: float = 1.75,
    protocol_radius: float = 2.0,
) -> dict[str, Any]:
    return ANALYSIS._correction_fields(
        eta=np.asarray(eta, dtype=np.float64),
        p_nav=np.asarray(p_nav, dtype=np.float64),
        goal=np.zeros(3, dtype=np.float64),
        tight_radius=tight_radius,
        protocol_radius=protocol_radius,
    )


def _record(
    *,
    seed: int,
    group: str,
    full_success: bool,
    deadband_success: bool,
    full_true_success: bool,
    deadband_true_success: bool,
) -> dict[str, Any]:
    is_false = group == "false"
    return {
        "scenario": "default",
        "seed": seed,
        "actual_success": True,
        "actual_true_capture": not is_false,
        "success_none": True,
        "true_success_none": not is_false,
        "false_success_none": is_false,
        "success_full": full_success,
        "true_success_full": full_true_success,
        "false_success_full": full_success and not full_true_success,
        "success_deadband": deadband_success,
        "true_success_deadband": deadband_true_success,
        "false_success_deadband": deadband_success and not deadband_true_success,
        "bias_projection_m": 0.55 if is_false else 0.05,
        "p_nav_eta_error_m": 0.73 if is_false else 0.38,
        "p_nav_margin_to_tight_m": 0.01,
        "eta_outside_protocol_m": 0.2 if is_false else -0.3,
    }


def test_full_correction_removes_optimistic_false_success() -> None:
    fields = _fields(eta=(2.1, 0.0, 0.0), p_nav=(1.7, 0.0, 0.0))
    assert fields["bias_projection_m"] > 0.0
    assert fields["success_none"] is True
    assert fields["false_success_none"] is True
    assert fields["success_full"] is False
    assert fields["success_deadband"] is False


def test_deadband_preserves_small_positive_bias_true_success() -> None:
    fields = _fields(eta=(1.8, 0.0, 0.0), p_nav=(1.74, 0.0, 0.0))
    assert 0.0 < fields["bias_projection_m"] < ANALYSIS.DEADBAND_M
    assert fields["true_success_none"] is True
    assert fields["true_success_full"] is False
    assert fields["true_success_deadband"] is True


def test_negative_bias_is_not_corrected() -> None:
    fields = _fields(eta=(1.5, 0.0, 0.0), p_nav=(1.7, 0.0, 0.0))
    assert fields["bias_projection_m"] < 0.0
    assert fields["p_nav_full_distance_m"] == fields["p_nav_none_distance_m"]
    assert fields["true_success_full"] is True
    assert fields["true_success_deadband"] is True


def test_deadband_gate_can_support_while_full_gate_is_also_supported() -> None:
    records = []
    for index in range(10):
        records.append(
            _record(
                seed=20 + index,
                group="false",
                full_success=index < 4,
                deadband_success=index < 4,
                full_true_success=False,
                deadband_true_success=False,
            )
        )
    for index in range(10):
        records.append(
            _record(
                seed=40 + index,
                group="true",
                full_success=index < 9,
                deadband_success=True,
                full_true_success=index < 9,
                deadband_true_success=True,
            )
        )
    result = ANALYSIS._aggregate(records, "default")
    assert result["base"]["false_success"] == 10
    assert result["base"]["true_success"] == 10
    assert result["gates"]["H75-A"]["false_success_reduction"] == 6
    assert result["gates"]["H75-A"]["true_success_retention"] == 0.9
    assert result["gates"]["H75-A"]["supported"] is True
    assert result["gates"]["H75-B"]["false_success_reduction"] == 6
    assert result["gates"]["H75-B"]["true_success_retention"] == 1.0
    assert result["gates"]["H75-B"]["supported"] is True
    assert result["reclassification_mismatch"] == 0


def test_small_sample_is_not_supported() -> None:
    records = [
        _record(
            seed=20,
            group="false",
            full_success=False,
            deadband_success=False,
            full_true_success=False,
            deadband_true_success=False,
        ),
        _record(
            seed=21,
            group="true",
            full_success=True,
            deadband_success=True,
            full_true_success=True,
            deadband_true_success=True,
        ),
    ]
    result = ANALYSIS._aggregate(records, "default")
    assert result["gates"]["H75-A"]["supported"] is False
    assert result["gates"]["H75-B"]["supported"] is False
