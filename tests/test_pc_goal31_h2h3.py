from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path("scripts/analyze_pc_goal31_h2h3.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal31_h2h3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(
    seed: int,
    *,
    eta_plane: float,
    nav_plane: float,
    share: float,
    horizontal: float = 1.0,
    vertical: float = 4.0,
    pitch_error: float = 0.01,
    closure: float = 0.5,
    saturation: float = 0.0,
    stern: float = 0.1,
    pitch_command: float = 0.1,
    terminal_capture: bool = False,
) -> dict[str, object]:
    return {
        "seed": seed,
        "outcome": "success" if terminal_capture else "plane_miss",
        "success_protocol": terminal_capture,
        "success_eta": terminal_capture,
        "terminal_eta_capture": terminal_capture,
        "terminal_eta_miss": not terminal_capture,
        "terminal_eta_plane_miss_m": eta_plane,
        "nav_plane_error_m": nav_plane,
        "estimator_plane_share": share,
        "estimator_dominated": share >= 0.5 and nav_plane >= 1.0,
        "horizontal_normal_eta_m": horizontal,
        "vertical_eta_m": vertical,
        "terminal_abs_vertical_eta_m": abs(vertical),
        "terminal_abs_pitch_error_mean_rad": pitch_error,
        "terminal_abs_pitch_command_mean_rad": pitch_command,
        "terminal_stern_abs_mean_rad": stern,
        "terminal_material_saturation_rate": saturation,
        "terminal_vertical_closure_rate_mps": closure,
        "terminal_phase_duration_s": 10.0,
    }


def test_decompose_many_separates_longitudinal_horizontal_and_vertical() -> None:
    offset = np.asarray([[2.0, 3.0, 4.0], [-1.0, 0.0, 0.5]])
    tangent = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    longitudinal, horizontal, vertical, normal = MODULE._decompose_many(offset, tangent)

    assert np.allclose(longitudinal, [2.0, -1.0])
    assert np.allclose(horizontal, [3.0, 0.0])
    assert np.allclose(vertical, [4.0, 0.5])
    assert np.allclose(normal, [[0.0, 3.0, 4.0], [0.0, 0.0, 0.5]])


def test_terminal_extras_uses_seed_local_last_row() -> None:
    count = 6
    trace = {
        "time_s": np.arange(count, dtype=np.float64) * 0.5,
        "eta": np.zeros((count, 6), dtype=np.float64),
        "p_nav": np.zeros((count, 3), dtype=np.float64),
        "goal_ned_m": np.zeros((count, 3), dtype=np.float64),
        "goal_tangent_ned": np.tile(np.asarray([[1.0, 0.0, 0.0]]), (count, 1)),
        "command": np.zeros((count, 3), dtype=np.float64),
        "actual_actuator": np.zeros((count, 3), dtype=np.float64),
        "pitch_error_rad": np.zeros(count, dtype=np.float64),
        "material_saturated": np.zeros(count, dtype=bool),
        "guidance_terminal_phase": np.ones(count, dtype=bool),
    }

    extras = MODULE._terminal_extras(
        trace,
        np.arange(count, dtype=np.int64),
        goal=np.zeros(3, dtype=np.float64),
        tangent=np.asarray([1.0, 0.0, 0.0]),
        goal_radius_m=2.0,
    )

    assert extras["terminal_final_time_s"] == 2.5
    assert extras["terminal_phase_duration_s"] == 2.5


def test_h2_estimator_supports_dominant_navigation_error() -> None:
    records = [
        _record(seed, eta_plane=eta, nav_plane=0.6 * eta, share=0.6)
        for seed, eta in zip(range(20, 25), (3.0, 3.5, 4.0, 4.5, 5.0), strict=True)
    ]

    result = MODULE._h2_estimator(records, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["estimator_share_median"] == 0.6
    assert result["estimator_dominated_fraction"] == 1.0


def test_h2_estimator_falsifies_small_navigation_error() -> None:
    records = [
        _record(seed, eta_plane=eta, nav_plane=0.1 * eta, share=0.1)
        for seed, eta in zip(range(20, 25), (3.0, 3.5, 4.0, 4.5, 5.0), strict=True)
    ]

    result = MODULE._h2_estimator(records, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["criteria"]["estimator_share_median_ge_0_50"] is False


def test_h2_lag_supports_tracking_and_saturation_signature() -> None:
    records = [
        _record(
            seed,
            eta_plane=5.0,
            nav_plane=0.2,
            share=0.04,
            vertical=vertical,
            pitch_error=pitch,
            closure=0.02,
            saturation=0.10,
            stern=0.20,
        )
        for seed, vertical, pitch in zip(
            range(20, 25),
            (3.0, 3.5, 4.0, 4.5, 5.0),
            (0.06, 0.07, 0.08, 0.09, 0.10),
            strict=True,
        )
    ]

    result = MODULE._h2_lag(records, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["spearman_pitch_error_vs_vertical_abs"] == 1.0


def test_h3_supports_stern_and_saturation_deltas() -> None:
    baseline = [
        _record(seed, eta_plane=3.0, nav_plane=0.2, share=0.1, stern=0.1, saturation=0.0)
        for seed in range(20, 25)
    ]
    candidate = [
        _record(
            seed,
            eta_plane=4.0,
            nav_plane=0.2,
            share=0.1,
            vertical=4.0 + 0.5 * index,
            stern=0.2 + 0.1 * index,
            saturation=0.05 + 0.01 * index,
            pitch_command=0.12,
        )
        for index, seed in enumerate(range(20, 25))
    ]

    result = MODULE._h3(baseline, candidate, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["delta_vertical_abs_m"]["median"] == 1.0
    assert result["max_correlation"] == 1.0
    assert result["insufficient_explanation"] is False


def test_h3_no_saturation_group_can_cap_explanation() -> None:
    baseline = [
        _record(seed, eta_plane=3.0, nav_plane=0.2, share=0.1, vertical=3.0)
        for seed in range(20, 25)
    ]
    candidate = [
        _record(seed, eta_plane=4.0, nav_plane=0.2, share=0.1, vertical=4.0)
        for seed in range(20, 25)
    ]

    result = MODULE._h3(baseline, candidate, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["insufficient_explanation"] is True


def test_overall_verdict_reports_multi_factor() -> None:
    scenarios = {}
    for scenario in MODULE.SCENARIOS:
        scenarios[scenario] = {
            "h2_estimator": {"passes_all": True},
            "h2_lag": {"passes_all": False},
            "h3": {"passes_all": True, "insufficient_explanation": False},
        }

    result = MODULE._overall_verdict(scenarios)

    assert result["decision"] == "MULTI_FACTOR"


def test_overall_verdict_refuses_goal_without_supported_mechanism() -> None:
    scenarios = {}
    for scenario in MODULE.SCENARIOS:
        scenarios[scenario] = {
            "h2_estimator": {"passes_all": False},
            "h2_lag": {"passes_all": False},
            "h3": {"passes_all": False, "insufficient_explanation": False},
        }

    result = MODULE._overall_verdict(scenarios)

    assert result["decision"] == "NO_SUPPORTED_MECHANISM"
