from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("scripts/audit_pc_goal45_fixed_calibration_residual.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_goal45_fixed_calibration_residual", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sensor_parameter_provenance_is_deterministic() -> None:
    default, default_meta = MODULE._load_sensor_parameters("default")
    strict, strict_meta = MODULE._load_sensor_parameters("strict125")

    assert default["nav_bias"].shape == (40, 3)
    assert default_meta["profile"] == "realistic_calibrated"
    assert default_meta["nav_bias_calibration_fraction"] == pytest.approx(0.8)
    assert np.allclose(default["nav_bias"], strict["nav_bias"], atol=0.0, rtol=0.0)
    assert strict_meta["sensor_seed"] == 81300


def test_fixed_residual_is_calibration_remainder() -> None:
    nav_bias = np.asarray([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])

    residual = MODULE._fixed_residual(nav_bias, 0.8)

    assert np.allclose(residual, [[0.02, 0.02, 0.02], [0.1, 0.1, 0.1]])


def test_late_error_vector_uses_frozen_window() -> None:
    time_s = np.arange(0.0, 6.02, 0.02, dtype=np.float64)
    exact_error = np.zeros((time_s.size, 3), dtype=np.float64)
    exact_error[:, 0] = 0.2
    exact_error[-1, 0] = 1.0
    valid = np.ones(time_s.size, dtype=bool)

    late_error, rows = MODULE._late_error_vector(
        exact_error=exact_error,
        valid_rows=valid,
        time_s=time_s,
    )

    assert rows == 226
    assert late_error == pytest.approx(np.asarray([0.2, 0.0, 0.0]))


def _record(
    seed: int,
    *,
    discordant: bool,
    fixed_aligned: bool = True,
    fixed_norm_share: float = 0.6,
    fixed_projection_share: float = 0.6,
    dynamic_projection: float = 0.2,
) -> dict[str, Any]:
    return {
        "scenario": "default",
        "mode": "baseline",
        "seed": seed,
        "success": True,
        "goal_capture_eta": not discordant,
        "discordant": discordant,
        "nav_bias_n": 0.1,
        "total_fixed_norm_m": 0.06,
        "late_error_norm_m": 0.2,
        "fixed_norm_share": fixed_norm_share,
        "late_projection_m": 0.3,
        "fixed_projection_m": 0.18,
        "fixed_projection_share": fixed_projection_share,
        "dynamic_projection_m": dynamic_projection,
        "fixed_aligned": fixed_aligned,
        "dynamic_toward_goal": dynamic_projection > 0.0,
        "window_rows": 226,
    }


def test_aggregate_freezes_fixed_and_dynamic_hypotheses() -> None:
    discordant = [_record(seed, discordant=True) for seed in range(5)]
    non_discordant = [_record(seed, discordant=False) for seed in range(5, 15)]
    aggregate = MODULE._aggregate([*discordant, *non_discordant])

    assert aggregate["discordant_seed_count"] == 5
    assert aggregate["fixed_aligned_fraction"] == pytest.approx(1.0)
    assert aggregate["dynamic_toward_fraction"] == pytest.approx(1.0)
    assert aggregate["h_fixed_residual_pass"] is True
    assert aggregate["h_dynamic_residual_pass"] is True


@pytest.mark.parametrize(
    ("fixed", "dynamic", "invalid", "expected"),
    [
        ("SUPPORTED", "SUPPORTED", False, "MIXED_FIXED_DYNAMIC_RESIDUAL"),
        ("SUPPORTED", "FALSIFIED", False, "FIXED_CALIBRATION_RESIDUAL_SUPPORTED"),
        ("FALSIFIED", "SUPPORTED", False, "DYNAMIC_RESIDUAL_DOMINANT"),
        ("PARTIAL", "FALSIFIED", False, "PARTIAL_COMPONENT_EVIDENCE"),
        ("FALSIFIED", "FALSIFIED", False, "NO_KNOWN_COMPONENT_SUPPORTED"),
        ("SUPPORTED", "SUPPORTED", True, "INVALID_DIAGNOSTIC"),
    ],
)
def test_overall_decision_mapping(
    fixed: str, dynamic: str, invalid: bool, expected: str
) -> None:
    result = MODULE._overall_decision(fixed, dynamic, invalid=invalid)

    assert result["decision"] == expected


def test_verdict_requires_all_blocks() -> None:
    assert MODULE._verdict(4) == "SUPPORTED"
    assert MODULE._verdict(1) == "PARTIAL"
    assert MODULE._verdict(0) == "FALSIFIED"
