from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal77_terminal_geometry_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal77_terminal_geometry_gate", ANALYSIS_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def test_goal_frame_offsets_reconstruct_norm_and_sign() -> None:
    frame = ANALYSIS._goal_frame_offsets(
        np.asarray([2.0, 1.0, 0.5], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    )
    assert frame["signed_m"] == 2.0
    assert frame["vertical_m"] == 0.5
    assert frame["lateral_m"] == -1.0
    assert frame["identity_error"] <= ANALYSIS.IDENTITY_TOL
    assert math.isclose(
        frame["signed_m"] ** 2 + frame["vertical_m"] ** 2 + frame["lateral_m"] ** 2,
        2.0**2 + 1.0**2 + 0.5**2,
        abs_tol=1e-12,
    )


def test_fixed_gate_flags_use_fractional_margins() -> None:
    accepted = ANALYSIS._gate_flags(
        signed=0.1, vertical=0.4, lateral=0.7, tight_radius=1.0
    )
    rejected_lateral = ANALYSIS._gate_flags(
        signed=0.1, vertical=0.4, lateral=0.8, tight_radius=1.0
    )
    rejected_vertical = ANALYSIS._gate_flags(
        signed=0.1, vertical=0.6, lateral=0.2, tight_radius=1.0
    )
    rejected_plane = ANALYSIS._gate_flags(
        signed=-0.1, vertical=0.2, lateral=0.2, tight_radius=1.0
    )
    assert accepted == {"G1": True, "G2": True, "G3": True, "G4": True}
    assert rejected_lateral["G2"] is False
    assert rejected_lateral["G4"] is False
    assert rejected_vertical["G3"] is False
    assert rejected_vertical["G4"] is False
    assert rejected_plane["G1"] is False
    assert rejected_plane["G4"] is False


def _record(
    *,
    seed: int,
    tight: bool,
    g1: bool,
    g2: bool,
    g3: bool,
    g4: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "scenario": "default",
        "run_key": "240",
        "seed": seed,
        "success_none": True,
        "tight_capture": tight,
        "protocol_capture": True,
        "identity_error": 0.0,
        "signed_m": 0.1 if tight else -0.5,
        "lateral_m": 0.1 if tight else 0.8,
        "vertical_m": 0.1,
        "mismatch": False,
    }
    for arm, gate in (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)):
        success = gate
        record[f"success_{arm}"] = success
        record[f"tight_true_{arm}"] = bool(success and tight)
        record[f"tight_false_{arm}"] = bool(success and not tight)
        record[f"protocol_true_{arm}"] = success
    return record


def test_aggregate_supports_plane_and_anisotropic_gates() -> None:
    records = []
    for index in range(10):
        records.append(
            _record(seed=20 + index, tight=False, g1=False, g2=False, g3=False, g4=False)
        )
    for index in range(10):
        records.append(
            _record(seed=40 + index, tight=True, g1=True, g2=True, g3=True, g4=True)
        )
    for index in range(2):
        records.append(
            _record(seed=50 + index, tight=False, g1=True, g2=True, g3=True, g4=False)
        )

    result = ANALYSIS._aggregate(records, "default")
    assert result["base"] == {
        "success": 22,
        "tight_true": 10,
        "tight_false": 12,
        "protocol_true": 22,
    }
    assert result["arms"]["G1"]["tight_false"] == 2
    assert result["arms"]["G4"]["tight_false"] == 0
    assert math.isclose(
        result["arms"]["G4"]["tight_false_reduction_fraction"], 1.0
    )
    assert math.isclose(result["arms"]["G4"]["tight_true_retention"], 1.0)
    assert result["gates"]["H77-A"]["supported"] is True
    assert result["gates"]["H77-B"]["supported"] is True
    assert result["mismatch"] == 0
    assert result["max_identity_error"] == 0.0


def test_h77b_requires_material_gain_over_plane_gate() -> None:
    scenarios = {
        "default": {
            "arms": {
                "G1": {
                    "tight_false_reduction_fraction": 0.80,
                    "tight_true_retention": 1.0,
                },
                "G4": {
                    "tight_false_reduction_fraction": 0.95,
                    "tight_true_retention": 1.0,
                },
            }
        },
        "strict125": {
            "arms": {
                "G1": {
                    "tight_false_reduction_fraction": 0.80,
                    "tight_true_retention": 1.0,
                },
                "G4": {
                    "tight_false_reduction_fraction": 0.92,
                    "tight_true_retention": 0.98,
                },
            }
        },
    }
    assert ANALYSIS._h77b_gain_supported(scenarios) is True

    scenarios["default"]["arms"]["G1"] = {
        "tight_false_reduction_fraction": 0.90,
        "tight_true_retention": 0.99,
    }
    scenarios["strict125"]["arms"]["G1"] = {
        "tight_false_reduction_fraction": 0.90,
        "tight_true_retention": 0.99,
    }
    assert ANALYSIS._h77b_gain_supported(scenarios) is False
