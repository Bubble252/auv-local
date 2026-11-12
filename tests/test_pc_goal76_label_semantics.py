from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal76_label_semantics.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal76_label_semantics", ANALYSIS_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def _label_record(
    *,
    seed: int,
    tight: bool,
    deadband_tight: bool,
    protocol: bool = True,
    deadband_success: bool = True,
    bias: float = 0.0,
) -> dict[str, Any]:
    success = tight or (protocol and not tight)
    return {
        "scenario": "default",
        "seed": seed,
        "actual_success": success,
        "actual_true_capture": protocol,
        "success_none": success,
        "success_full": success,
        "success_deadband": deadband_success,
        "protocol_true_none": success and protocol,
        "tight_true_none": success and tight,
        "tight_false_none": success and not tight,
        "protocol_true_full": success and protocol,
        "tight_true_full": success and tight,
        "tight_false_full": success and not tight,
        "protocol_true_deadband": deadband_success and protocol,
        "tight_true_deadband": deadband_success and deadband_tight,
        "tight_false_deadband": deadband_success and not deadband_tight,
        "bias_projection_m": bias,
        "p_nav_eta_error_m": 0.5,
        "signed_offset_m": -0.1,
        "path_error_3d_m": 0.2,
        "goal_distance_m": 1.7,
        "base_command_l2": 1.0,
        "residual_l2": 0.1,
        "action_l2": 0.3,
        "residual_gate": 0.5,
    }


def test_tight_label_differs_from_protocol_label() -> None:
    fields = ANALYSIS.G75._correction_fields(
        eta=np.asarray([1.9, 0.0, 0.0], dtype=np.float64),
        p_nav=np.asarray([1.7, 0.0, 0.0], dtype=np.float64),
        goal=np.zeros(3, dtype=np.float64),
        tight_radius=1.75,
        protocol_radius=2.0,
    )
    fields["tight_radius_m"] = 1.75
    labels = ANALYSIS._label_fields(fields)
    assert labels["protocol_capture"] is True
    assert labels["tight_capture"] is False
    assert labels["protocol_true_none"] is True
    assert labels["tight_true_none"] is False
    assert labels["tight_false_none"] is True


def test_aggregate_supports_tight_label_and_protocol_conflation() -> None:
    records = []
    for index in range(10):
        records.append(
            _label_record(
                seed=20 + index,
                tight=False,
                deadband_tight=False,
                deadband_success=False,
                bias=0.6,
            )
        )
    for index in range(10):
        records.append(
            _label_record(seed=40 + index, tight=True, deadband_tight=True, bias=0.05)
        )
    result = ANALYSIS._aggregate(records, "default")
    assert result["base"]["protocol_true"] == 20
    assert result["base"]["tight_true"] == 10
    assert result["base"]["tight_false"] == 10
    assert result["protocol_only"] == 10
    assert result["gates"]["H76-A"]["tight_false_reduction"] == 10
    assert result["gates"]["H76-A"]["tight_true_retention"] == 1.0
    assert result["gates"]["H76-A"]["supported"] is True
    assert result["gates"]["H76-B"]["tight_to_protocol_ratio"] == 0.5
    assert result["gates"]["H76-B"]["supported"] is True
    assert result["reclassification_mismatch"] == 0


def test_protocol_ratio_gate_rejects_weak_conflation() -> None:
    records = []
    for index in range(15):
        records.append(
            _label_record(seed=20 + index, tight=True, deadband_tight=True)
        )
    for index in range(5):
        records.append(
            _label_record(seed=40 + index, tight=False, deadband_tight=False)
        )
    result = ANALYSIS._aggregate(records, "default")
    assert result["gates"]["H76-B"]["tight_to_protocol_ratio"] == 0.75
    assert result["gates"]["H76-B"]["supported"] is False


def test_deadband_tight_false_gate_can_fail() -> None:
    records = []
    for index in range(10):
        records.append(
            _label_record(seed=20 + index, tight=False, deadband_tight=False)
        )
    for index in range(10):
        records.append(
            _label_record(
                seed=40 + index,
                tight=True,
                deadband_tight=index < 9,
            )
        )
    result = ANALYSIS._aggregate(records, "default")
    assert result["gates"]["H76-A"]["tight_true_retention"] == 0.9
    assert result["gates"]["H76-A"]["supported"] is False
