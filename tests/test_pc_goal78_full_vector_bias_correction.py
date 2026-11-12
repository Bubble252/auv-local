from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = (
    REPO_ROOT / "scripts" / "analyze_pc_goal78_full_vector_bias_correction.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal78_full_vector_bias_correction", ANALYSIS_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def test_correction_arms_use_along_and_full_vector_deadbands() -> None:
    arms = ANALYSIS._correction_arms(
        eta=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        p_nav=np.asarray([2.6, 0.3, 0.0], dtype=np.float64),
        goal=np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
    )
    assert math.isclose(arms["error_norm"], math.sqrt(2.6**2 + 0.3**2))
    assert math.isclose(arms["along_goal"], 2.6)
    assert math.isclose(arms["along_correction"], 2.35)
    assert math.isclose(arms["full_correction"], math.sqrt(2.6**2 + 0.3**2) - 0.25)
    assert np.allclose(arms["p_nav_along"], np.asarray([0.25, 0.3, 0.0]))
    assert np.allclose(
        arms["p_nav_full"],
        np.asarray([2.6, 0.3, 0.0])
        - (math.sqrt(2.6**2 + 0.3**2) - 0.25)
        * np.asarray([2.6, 0.3, 0.0])
        / math.sqrt(2.6**2 + 0.3**2),
    )


def _record(
    *,
    seed: int,
    tight: bool,
    a1: bool,
    a2: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "scenario": "default",
        "run_key": "240",
        "seed": seed,
        "success_A0": True,
        "tight_capture": tight,
        "protocol_capture": True,
        "mismatch": False,
        "error_norm": 0.3,
        "along_goal": 0.1,
        "along_correction": 0.0,
        "full_correction": 0.05,
    }
    for arm, success in (("A0", True), ("A1", a1), ("A2", a2)):
        record[f"success_{arm}"] = success
        record[f"tight_true_{arm}"] = bool(success and tight)
        record[f"tight_false_{arm}"] = bool(success and not tight)
        record[f"protocol_true_{arm}"] = success
    return record


def test_aggregate_supports_full_vector_deadband() -> None:
    records = []
    for index in range(10):
        records.append(_record(seed=20 + index, tight=False, a1=index >= 6, a2=index < 2))
    for index in range(10):
        records.append(_record(seed=40 + index, tight=True, a1=True, a2=True))
    result = ANALYSIS._aggregate(records, "default")
    assert result["base"] == {
        "success": 20,
        "tight_true": 10,
        "tight_false": 10,
        "protocol_true": 20,
    }
    assert result["arms"]["A1"]["tight_false"] == 4
    assert result["arms"]["A2"]["tight_false"] == 2
    assert math.isclose(result["arms"]["A2"]["tight_false_reduction_fraction"], 0.8)
    assert math.isclose(result["arms"]["A2"]["tight_true_retention"], 1.0)
    assert result["gates"]["H78-A"]["supported"] is True
    assert result["mismatch"] == 0


def test_aggregate_rejects_true_loss() -> None:
    records = []
    for index in range(10):
        records.append(_record(seed=20 + index, tight=False, a1=False, a2=False))
    for index in range(10):
        records.append(
            _record(seed=40 + index, tight=True, a1=True, a2=index < 9)
        )
    result = ANALYSIS._aggregate(records, "default")
    assert math.isclose(result["arms"]["A2"]["tight_true_retention"], 0.9)
    assert result["gates"]["H78-A"]["supported"] is False


def test_h78b_requires_material_gain_without_true_loss() -> None:
    scenarios = {
        "default": {
            "arms": {
                "A1": {"tight_false_reduction_fraction": 0.4, "tight_true_retention": 1.0},
                "A2": {"tight_false_reduction_fraction": 0.8, "tight_true_retention": 1.0},
            }
        },
        "strict125": {
            "arms": {
                "A1": {"tight_false_reduction_fraction": 0.5, "tight_true_retention": 1.0},
                "A2": {"tight_false_reduction_fraction": 0.7, "tight_true_retention": 0.96},
            }
        },
    }
    assert ANALYSIS._h78b_supported(scenarios) is True

    scenarios["strict125"]["arms"]["A2"]["tight_true_retention"] = 0.90
    assert ANALYSIS._h78b_supported(scenarios) is False
