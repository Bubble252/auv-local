from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/analyze_pc_goal36_initial_geometry.py")
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal36_initial_geometry", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _trace(
    *,
    offset: tuple[float, float, float] = (-34.0, 0.0, -5.0),
    goal_tangent: tuple[float, float, float] = (1.0, 0.0, -0.1),
    route_tangent: tuple[float, float, float] = (1.0, 0.0, -0.2),
    pitch: float = 0.05,
    pitch_rate: float = 0.0,
    vertical_velocity: float = 0.0,
) -> dict[str, np.ndarray]:
    goal = np.asarray([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    eta = np.zeros((2, 6), dtype=np.float64)
    eta[:, :3] = goal
    eta[:, :3] += np.asarray(offset, dtype=np.float64)
    eta[:, 4] = pitch
    nu = np.zeros((2, 6), dtype=np.float64)
    nu[:, 4] = pitch_rate
    ground_body = np.zeros((2, 3), dtype=np.float64)
    ground_body[:, 2] = vertical_velocity
    return {
        "time_s": np.asarray([0.0, 0.02]),
        "eta": eta,
        "nu": nu,
        "goal_ned_m": goal,
        "goal_tangent_ned": np.tile(np.asarray(goal_tangent), (2, 1)),
        "route_tangent_ned": np.tile(np.asarray(route_tangent), (2, 1)),
        "ground_velocity_body_mps": ground_body,
    }


def _record(seed: int, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "seed": seed,
        "vertical_miss": True,
        "world_vertical_m": 3.0 + 0.1 * (seed - 20),
        "reference_vertical_m": 2.0 + 0.1 * (seed - 20),
        "initial_vertical_m": 5.0 + 0.2 * (seed - 20),
        "reference_share": 0.4,
        "goal_pitch_rad": 0.10,
        "route_pitch_rad": 0.12,
        "initial_pitch_rad": 0.04,
        "initial_pitch_rate": 0.0,
        "initial_vertical_v_mps": 0.0,
        "abs_reference_vertical_m": 2.0 + 0.1 * (seed - 20),
        "abs_initial_pitch_rad": 0.04,
        "abs_initial_pitch_rate": 0.0,
        "abs_initial_vertical_v_mps": 0.0,
        "terminal_abs_vertical_m": 3.0 + 0.1 * (seed - 20),
    }
    base.update(overrides)
    return base


def test_initial_geometry_matches_goal_tangent_projection() -> None:
    trace = _trace()
    result = MODULE._initial_geometry(trace, np.arange(2), 2.0, 5.0)

    goal_tangent = np.asarray([1.0, 0.0, -0.1])
    goal_tangent /= np.linalg.norm(goal_tangent)
    route_tangent = np.asarray([1.0, 0.0, -0.2])
    route_tangent /= np.linalg.norm(route_tangent)
    expected_longitudinal = float(np.dot(np.asarray([-34.0, 0.0, -5.0]), goal_tangent))
    expected_reference = float(-expected_longitudinal * goal_tangent[2])
    assert result["world_vertical_m"] == pytest.approx(-5.0)
    assert result["longitudinal_m"] == pytest.approx(expected_longitudinal)
    assert result["reference_vertical_m"] == pytest.approx(expected_reference)
    assert result["initial_vertical_m"] == pytest.approx(
        -5.0 + expected_reference
    )
    assert result["goal_pitch_rad"] == pytest.approx(
        np.arctan2(-goal_tangent[2], goal_tangent[0]), abs=1e-6
    )
    assert result["route_pitch_rad"] == pytest.approx(
        np.arctan2(-route_tangent[2], route_tangent[0]), abs=1e-6
    )


def test_initial_geometry_rejects_nonconstant_goal() -> None:
    trace = _trace()
    trace["goal_ned_m"] = trace["goal_ned_m"].copy()
    trace["goal_ned_m"][1, 2] += 0.1

    with pytest.raises(ValueError, match="goal changes"):
        MODULE._initial_geometry(trace, np.arange(2), 2.0, 5.0)


def test_geometry_differences_detects_candidate_mismatch() -> None:
    baseline = [_record(20)]
    candidate = [_record(20, world_vertical_m=3.1)]

    result = MODULE._geometry_differences(baseline, candidate)

    assert result["seed_sets_match"] is True
    assert result["max_abs_difference"] == pytest.approx(0.1)


def test_partial_spearman_detects_independent_feature() -> None:
    x = list(np.linspace(1.0, 2.0, 12))
    control = [5.0, 1.0, 9.0, 3.0, 12.0, 2.0, 8.0, 4.0, 11.0, 6.0, 10.0, 7.0]
    y = list(np.asarray(x) * 3.0)

    result = MODULE._partial_spearman(x, y, control)

    assert result == pytest.approx(1.0)


def test_partial_spearman_is_none_when_feature_is_constant() -> None:
    result = MODULE._partial_spearman(
        [1.0] * 12,
        list(np.linspace(1.0, 2.0, 12)),
        list(np.linspace(3.0, 4.0, 12)),
    )

    assert result is None


def test_h_state_supports_world_vertical_offset() -> None:
    records = [
        _record(
            seed,
            world_vertical_m=3.0 + 0.2 * (seed - 20),
            initial_vertical_m=3.0 + 0.2 * (seed - 20),
            terminal_abs_vertical_m=2.0 + 0.2 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_state(records, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["world_initial_rho"] == pytest.approx(1.0)
    assert result["world_terminal_rho"] == pytest.approx(1.0)


def test_h_state_fails_when_world_offset_is_inside_radius() -> None:
    records = [
        _record(
            seed,
            world_vertical_m=0.5,
            initial_vertical_m=0.5,
            terminal_abs_vertical_m=3.0 + 0.1 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_state(records, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["criteria"]["world_outside_radius_fraction_ge_0_60"] is False
    assert result["criteria"]["world_terminal_ratio_ge_0_80"] is False


def test_h_reference_supports_material_goal_tangent_projection() -> None:
    records = [
        _record(
            seed,
            goal_pitch_rad=0.10,
            route_pitch_rad=0.12,
            reference_vertical_m=2.0 + 0.2 * (seed - 20),
            abs_reference_vertical_m=2.0 + 0.2 * (seed - 20),
            initial_vertical_m=3.0 + 0.2 * (seed - 20),
            reference_share=0.6,
            terminal_abs_vertical_m=2.0 + 0.2 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_reference(records)

    assert result["passes_all"] is True
    assert result["reference_initial_rho"] == pytest.approx(1.0)
    assert result["reference_terminal_rho"] == pytest.approx(1.0)
    assert result["reference_high_low_terminal_delta_m"] > 0.5


def test_h_reference_fails_with_small_share() -> None:
    records = [
        _record(
            seed,
            goal_pitch_rad=0.10,
            route_pitch_rad=0.12,
            reference_vertical_m=0.1,
            abs_reference_vertical_m=0.1,
            initial_vertical_m=3.0 + 0.1 * (seed - 20),
            reference_share=0.05,
            terminal_abs_vertical_m=3.0 + 0.1 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_reference(records)

    assert result["passes_all"] is False
    assert result["criteria"]["reference_vertical_median_ge_1_00"] is False
    assert result["criteria"]["reference_share_median_ge_0_25"] is False


def test_h_start_dynamics_supports_independent_pitch_effect() -> None:
    controls = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 12.0]
    records = [
        _record(
            seed,
            abs_initial_pitch_rad=0.04 + 0.01 * (seed - 20),
            abs_reference_vertical_m=controls[seed - 20],
            terminal_abs_vertical_m=2.0 + 0.2 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_start_dynamics(records)

    assert result["passes_all"] is True
    pitch = next(
        feature for feature in result["features"] if feature["name"] == "pitch"
    )
    assert pitch["passes_all"] is True
    assert pitch["partial_spearman"] > 0.5


def test_h_start_dynamics_fails_when_start_is_static() -> None:
    records = [
        _record(
            seed,
            abs_initial_pitch_rad=0.0,
            abs_initial_pitch_rate=0.0,
            abs_initial_vertical_v_mps=0.0,
            terminal_abs_vertical_m=2.0 + 0.2 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_start_dynamics(records)

    assert result["passes_all"] is False
    assert result["criteria"]["any_dynamic_feature_passes"] is False


def _block(state: bool, reference: bool, start: bool) -> dict[str, object]:
    return {
        "h_state": {"passes_all": state},
        "h_reference": {"passes_all": reference},
        "h_start_dynamics": {"passes_all": start},
    }


@pytest.mark.parametrize(
    ("state", "reference", "start", "shares", "expected"),
    (
        (True, False, False, [0.2, 0.2, 0.2, 0.2], "WORLD_STATE_DOMINANT"),
        (False, True, False, [0.8, 0.8, 0.8, 0.8], "REFERENCE_GEOMETRY_DOMINANT"),
        (False, False, True, [0.2, 0.2, 0.2, 0.2], "START_DYNAMICS_DOMINANT"),
        (True, True, False, [0.8, 0.8, 0.8, 0.8], "REFERENCE_GEOMETRY_DOMINANT"),
        (True, True, False, [0.2, 0.2, 0.2, 0.2], "WORLD_STATE_DOMINANT"),
        (True, True, False, [0.45, 0.45, 0.45, 0.45], "MIXED_STATE_REFERENCE"),
        (True, False, True, [0.2, 0.2, 0.2, 0.2], "MULTI_FACTOR"),
        (False, False, False, [0.2, 0.2, 0.2, 0.2], "NO_SUPPORTED_MECHANISM"),
    ),
)
def test_overall_verdict_mapping(
    state: bool,
    reference: bool,
    start: bool,
    shares: list[float],
    expected: str,
) -> None:
    blocks = [_block(state, reference, start) for _ in range(4)]

    result = MODULE._overall_verdict(
        blocks,
        invalid=False,
        reference_share_block_medians=shares,
    )

    assert result["decision"] == expected


def test_overall_verdict_reserves_invalid_diagnostic() -> None:
    result = MODULE._overall_verdict(
        [_block(True, True, True) for _ in range(4)],
        invalid=True,
        reference_share_block_medians=[0.5] * 4,
    )

    assert result["decision"] == "INVALID_DIAGNOSTIC"
