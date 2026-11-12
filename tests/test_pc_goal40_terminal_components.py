from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/analyze_pc_goal40_terminal_components.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal40_terminal_components", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_decompose_and_dominant_component() -> None:
    offset = np.asarray([0.0, 3.0, 4.0])
    tangent = np.asarray([1.0, 0.0, 0.0])

    longitudinal, horizontal, vertical = MODULE._decompose(offset, tangent)

    assert longitudinal == pytest.approx(0.0)
    assert horizontal == pytest.approx(3.0)
    assert vertical == pytest.approx(4.0)
    assert MODULE._dominant_component(horizontal, vertical) == "vertical"
    assert MODULE._dominant_component(3.0, 3.0) == "horizontal"


def test_decompose_rejects_zero_tangent() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        MODULE._decompose(np.zeros(3), np.zeros(3))


def _record(
    seed: int,
    *,
    plane_miss: bool,
    dominant: str,
    success: bool,
    eta_capture: bool,
) -> dict[str, object]:
    return {
        "seed": seed,
        "goal_plane_miss": plane_miss,
        "dominant_component": dominant,
        "success": success,
        "goal_capture_eta": eta_capture,
        "protocol_success_without_eta_capture": success and not eta_capture,
        "eta_capture_without_protocol_success": eta_capture and not success,
        "abs_horizontal_over_radius": 0.7,
        "abs_vertical_over_radius": 1.2,
    }


def test_aggregate_counts_components_and_discordance() -> None:
    records = [
        _record(i, plane_miss=True, dominant="vertical", success=True, eta_capture=True)
        for i in range(20, 24)
    ]
    records += [
        _record(24, plane_miss=True, dominant="horizontal", success=True, eta_capture=True)
    ]
    records += [
        _record(i, plane_miss=False, dominant="vertical", success=True, eta_capture=(i < 32))
        for i in range(25, 35)
    ]

    result = MODULE._aggregate(records)

    assert result["plane_miss_count"] == 5
    assert result["plane_miss_vertical_dominant_fraction"] == pytest.approx(0.8)
    assert result["protocol_success_count"] == 15
    assert result["protocol_success_without_eta_capture_count"] == 3
    assert result["protocol_success_without_eta_capture_fraction"] == pytest.approx(0.2)


def test_mode_and_overall_verdict_mapping() -> None:
    assert MODULE._mode_verdict(4) == "SUPPORTED"
    assert MODULE._mode_verdict(2) == "PARTIAL"
    assert MODULE._mode_verdict(0) == "FALSIFIED"

    assert MODULE._overall_decision("SUPPORTED", "SUPPORTED", False)["decision"] == (
        "MULTI_FACTOR"
    )
    assert MODULE._overall_decision("SUPPORTED", "PARTIAL", False)["decision"] == (
        "VERTICAL_COMPONENT_SUPPORTED"
    )
    assert MODULE._overall_decision("PARTIAL", "SUPPORTED", False)["decision"] == (
        "PROTOCOL_ETA_DISCORDANCE_SUPPORTED"
    )
    assert MODULE._overall_decision("PARTIAL", "FALSIFIED", False)["decision"] == (
        "PARTIAL_ONLY"
    )
    assert MODULE._overall_decision("FALSIFIED", "FALSIFIED", False)["decision"] == (
        "NO_SUPPORTED_COMPONENT_SIGNAL"
    )
    assert MODULE._overall_decision("SUPPORTED", "SUPPORTED", True)["decision"] == (
        "INVALID_DIAGNOSTIC"
    )
