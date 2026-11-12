from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path("scripts/analyze_pc_goal32_terminal_entry.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal32_terminal_entry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(
    seed: int,
    *,
    entry_abs: float = 4.0,
    terminal_abs: float = 3.0,
    entry_signed: float | None = None,
    terminal_signed: float | None = None,
    required: float = 1.0,
    authority: float = 0.2,
    achieved: float = 0.1,
    saturation: float = 0.2,
    pitch: float = 0.2,
    vertical_miss: bool = True,
) -> dict[str, object]:
    if entry_signed is None:
        entry_signed = entry_abs
    if terminal_signed is None:
        terminal_signed = terminal_abs
    return {
        "seed": seed,
        "vertical_miss": vertical_miss,
        "entry_abs_vertical_m": entry_abs,
        "terminal_abs_vertical_m": terminal_abs,
        "entry_vertical_m": entry_signed,
        "terminal_vertical_m": terminal_signed,
        "vertical_closure_entry_to_terminal_m": entry_abs - terminal_abs,
        "required_closure_mps": required,
        "authority_p95_mps": authority,
        "required_minus_authority_mps": required - authority,
        "achieved_closure_mps": achieved,
        "achieved_minus_required_mps": achieved - required,
        "early_material_saturation_rate": saturation,
        "early_abs_pitch_error_mean_rad": pitch,
    }


def _trace() -> dict[str, np.ndarray]:
    count = 6
    time_s = np.arange(count, dtype=np.float64) * 0.5
    eta = np.zeros((count, 6), dtype=np.float64)
    eta[:, 0] = [-22.0, -21.0, -20.0, -1.0, 1.0, 2.0]
    eta[:, 2] = [3.0, 3.0, 4.0, 3.0, 2.5, 3.0]
    p_nav = eta[:, :3].copy()
    goal = np.zeros((count, 3), dtype=np.float64)
    tangent = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (count, 1))
    phase = np.asarray([False, False, True, True, True, True], dtype=bool)
    saturation = np.asarray([False, False, True, True, False, False], dtype=bool)
    pitch_error = np.full(count, 0.2, dtype=np.float64)
    actuator = np.tile(np.asarray([[0.1, 0.1, 100.0]]), (count, 1))
    ground_body = np.zeros((count, 3), dtype=np.float64)
    ground_body[:, 0] = 1.0
    ground_body[:, 2] = -0.3
    return {
        "time_s": time_s,
        "eta": eta,
        "p_nav": p_nav,
        "goal_ned_m": goal,
        "goal_tangent_ned": tangent,
        "guidance_terminal_phase": phase,
        "material_saturated": saturation,
        "pitch_error_rad": pitch_error,
        "actual_actuator": actuator,
        "ground_velocity_body_mps": ground_body,
    }


def test_decompose_many_separates_longitudinal_horizontal_and_vertical() -> None:
    offset = np.asarray([[2.0, 3.0, 4.0], [-1.0, 0.0, 0.5]])
    tangent = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    longitudinal, horizontal, vertical, normal = MODULE._decompose_many(offset, tangent)

    assert np.allclose(longitudinal, [2.0, -1.0])
    assert np.allclose(horizontal, [3.0, 0.0])
    assert np.allclose(vertical, [4.0, 0.5])
    assert np.allclose(normal, [[0.0, 3.0, 4.0], [0.0, 0.0, 0.5]])


def test_body_to_world_ned_matches_identity_and_yaw_rotation() -> None:
    eta = np.zeros((2, 6), dtype=np.float64)
    eta[1, 5] = 0.5 * np.pi
    body = np.asarray([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    rotated = MODULE._body_to_world_ned(body, eta)

    assert np.allclose(rotated[0], body[0])
    assert np.allclose(rotated[1], [0.0, 1.0, 0.0], atol=1e-12)


def test_entry_metrics_computes_required_and_demonstrated_closure() -> None:
    trace = _trace()
    indices = np.arange(trace["time_s"].size, dtype=np.int64)

    result = MODULE._entry_metrics(trace, indices, goal_radius_m=2.0)

    assert result["entry_time_s"] == 1.0
    assert result["entry_vertical_m"] == 4.0
    assert result["terminal_vertical_m"] == 3.0
    assert result["time_to_plane_s"] == 1.0
    assert result["plane_crossed"] is True
    assert np.isclose(result["required_closure_mps"], 2.0)
    assert np.isclose(result["authority_p95_mps"], 0.3)
    assert np.isclose(result["achieved_closure_mps"], 1.5)
    assert np.isclose(result["required_minus_authority_mps"], 1.7)
    assert np.isclose(result["early_material_saturation_rate"], 0.5)


def test_entry_metrics_rejects_goal_change_within_episode() -> None:
    trace = _trace()
    trace["goal_ned_m"][3, 2] = 1.0

    try:
        MODULE._entry_metrics(
            trace,
            np.arange(trace["time_s"].size, dtype=np.int64),
            goal_radius_m=2.0,
        )
    except ValueError as exc:
        assert "goal changes" in str(exc)
    else:
        raise AssertionError("expected goal-change validation to fail")


def test_h_entry_supports_error_already_outside_radius() -> None:
    records = [
        _record(
            seed,
            entry_abs=3.0 + 0.1 * index,
            terminal_abs=2.5 + 0.1 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_entry(records, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["entry_outside_radius_fraction"] == 1.0
    assert result["entry_terminal_spearman"] == 1.0


def test_h_entry_falsifies_small_entry_error() -> None:
    records = [
        _record(
            seed,
            entry_abs=0.5 + 0.01 * index,
            terminal_abs=3.0 + 0.1 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_entry(records, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["entry_outside_radius_fraction"] == 0.0


def test_h_authority_supports_entry_shortfall() -> None:
    records = [
        _record(
            seed,
            terminal_abs=2.5 + 0.1 * index,
            required=0.8 + 0.05 * index,
            authority=0.2,
            achieved=0.1,
            saturation=0.2,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_authority(records)

    assert result["passes_all"] is True
    assert result["required_gt_authority_fraction"] == 1.0
    assert result["required_terminal_spearman"] == 1.0


def test_h_authority_falsifies_available_closure() -> None:
    records = [
        _record(
            seed,
            terminal_abs=2.5 + 0.1 * index,
            required=0.2,
            authority=1.0,
            achieved=1.0,
            saturation=0.0,
            pitch=0.01,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_authority(records)

    assert result["passes_all"] is False
    assert result["required_minus_authority_median_mps"] < 0.0


def test_h_geometry_supports_entry_inherited_candidate_shift() -> None:
    baseline = [
        _record(seed, entry_abs=2.0, terminal_abs=2.5, entry_signed=2.0, terminal_signed=2.5)
        for seed in range(20, 32)
    ]
    candidate = [
        _record(
            seed,
            entry_abs=2.0 + 0.1 * (index + 1),
            terminal_abs=2.5 + 0.1 * (index + 1),
            entry_signed=2.0 + 0.1 * (index + 1),
            terminal_signed=2.5 + 0.1 * (index + 1),
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_geometry(baseline, candidate, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["signed_spearman"] == 1.0
    assert result["sign_agreement_fraction"] == 1.0
    assert result["delta_gap_median_m"] == 0.0


def test_h_geometry_falsifies_zero_entry_delta() -> None:
    baseline = [_record(seed, entry_abs=4.0, terminal_abs=3.0) for seed in range(20, 32)]
    candidate = [_record(seed, entry_abs=4.0, terminal_abs=3.0) for seed in range(20, 32)]

    result = MODULE._h_geometry(baseline, candidate, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["non_deadband_pair_count"] == 0
    assert result["sign_agreement_fraction"] is None


def test_overall_verdict_requires_both_entry_and_authority() -> None:
    scenarios = {}
    for scenario in MODULE.SCENARIOS:
        scenarios[scenario] = {
            "baseline": {
                "h_entry": {"passes_all": True},
                "h_authority": {"passes_all": True},
            },
            "candidate": {
                "h_entry": {"passes_all": True},
                "h_authority": {"passes_all": True},
            },
            "h_geometry": {"passes_all": False},
        }

    result = MODULE._overall_verdict(scenarios)

    assert result["decision"] == "ENTRY_REACHABILITY_SUPPORTED"


def test_overall_verdict_refuses_unsupported_mechanism() -> None:
    scenarios = {}
    for scenario in MODULE.SCENARIOS:
        scenarios[scenario] = {
            "baseline": {
                "h_entry": {"passes_all": False},
                "h_authority": {"passes_all": False},
            },
            "candidate": {
                "h_entry": {"passes_all": False},
                "h_authority": {"passes_all": False},
            },
            "h_geometry": {"passes_all": False},
        }

    result = MODULE._overall_verdict(scenarios)

    assert result["decision"] == "NO_SUPPORTED_MECHANISM"
