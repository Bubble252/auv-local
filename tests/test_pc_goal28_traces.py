from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/analyze_pc_goal28_traces.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal28_traces", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(seed: int, outcome: str, *, horizontal: float, vertical: float) -> object:
    return MODULE.TerminalRecord(
        arm="d3",
        scenario="default",
        seed=seed,
        outcome=outcome,
        success_protocol=outcome == "success",
        success_eta=outcome == "success",
        pnav_only_success=False,
        terminal_kind=1 if outcome == "success" else 9,
        final_time_s=20.0,
        pnav_distance_m=float(np.hypot(horizontal, vertical)),
        eta_distance_m=float(np.hypot(horizontal, vertical)),
        longitudinal_pnav_m=0.0,
        horizontal_normal_pnav_m=horizontal,
        vertical_pnav_m=vertical,
        longitudinal_eta_m=0.0,
        horizontal_normal_eta_m=horizontal,
        vertical_eta_m=vertical,
        nav_error_m=0.2,
        terminal_speed_ratio=1.1,
        terminal_yaw_error_abs_mean_rad=0.3,
        terminal_pitch_error_abs_mean_rad=0.1,
        terminal_material_saturation_rate=0.1,
        terminal_rudder_abs_mean_rad=0.1,
        terminal_stern_abs_mean_rad=0.1,
    )


def test_decompose_uses_goal_tangent() -> None:
    offset = np.array([3.0, 4.0, -2.0])
    tangent = np.array([1.0, 0.0, 0.0])
    longitudinal, horizontal, vertical = MODULE._decompose(offset, tangent)
    assert longitudinal == pytest.approx(3.0)
    assert horizontal == pytest.approx(4.0)
    assert vertical == pytest.approx(-2.0)


def test_aggregate_reports_horizontal_share_and_control_deltas() -> None:
    records = [
        _record(20, "success", horizontal=0.5, vertical=0.1),
        _record(21, "plane_miss", horizontal=2.5, vertical=0.2),
        _record(22, "plane_miss", horizontal=3.0, vertical=0.4),
    ]
    aggregate = MODULE._aggregate(records, goal_radius_m=2.0)

    assert aggregate["plane_miss"]["count"] == 2
    assert aggregate["plane_miss_distance_to_radius_median"] > 1.25
    assert aggregate["plane_miss_horizontal_share_median"] > 0.9
    assert aggregate["plane_miss_minus_success_terminal_speed_ratio_median"] == pytest.approx(0.0)
