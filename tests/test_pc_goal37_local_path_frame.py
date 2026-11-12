from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from auv_lowlevel.path import ReferencePath

SCRIPT = Path("scripts/analyze_pc_goal37_local_path_frame.py")
SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal37_local_path_frame", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _line_path() -> ReferencePath:
    return ReferencePath(
        name="goal37_test_line",
        points_ned_m=np.asarray(
            [[0.0, 0.0, 10.0], [50.0, 0.0, 10.0], [100.0, 0.0, 10.0]]
        ),
        cumulative_s_m=np.asarray([0.0, 50.0, 100.0]),
        speed_mps=1.0,
    )


def _local_trace() -> dict[str, np.ndarray]:
    eta = np.zeros((4, 6), dtype=np.float64)
    eta[:, 0] = [0.0, 60.0, 80.0, 100.0]
    eta[:, 2] = 10.5
    goal = np.tile(np.asarray([[100.0, 0.0, 10.0]]), (4, 1))
    tangent = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (4, 1))
    return {
        "time_s": np.asarray([0.0, 0.02, 0.04, 0.06]),
        "eta": eta,
        "goal_ned_m": goal,
        "goal_tangent_ned": tangent,
        "route_tangent_ned": tangent.copy(),
    }


def _record(seed: int, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "seed": seed,
        "vertical_miss": True,
        "goal_initial_abs_m": 8.0,
        "local_initial_abs_m": 0.5,
        "goal_minus_local_abs_m": 7.5,
        "local40_abs_m": 6.0,
        "local20_abs_m": 5.0,
        "local_terminal_abs_m": 5.0,
        "goal_terminal_abs_m": 5.0,
    }
    base.update(overrides)
    return base


def test_first_crossing_returns_first_hit() -> None:
    distance = np.asarray([50.0, 39.0, 19.0, 5.0])

    assert MODULE._first_crossing(distance, 40.0) == 1
    assert MODULE._first_crossing(distance, 20.0) == 2


def test_first_crossing_rejects_missing_threshold() -> None:
    with pytest.raises(ValueError, match="never reaches"):
        MODULE._first_crossing(np.asarray([50.0, 30.0]), 20.0)


def test_path_for_seed_uses_modulo() -> None:
    bank = tuple(range(32))
    path, index = MODULE._path_for_seed(bank, 59)

    assert path == 27
    assert index == 27


def test_local_geometry_projects_to_local_path_frame() -> None:
    path = _line_path()
    trace = _local_trace()
    indices = np.arange(4)

    result, provenance = MODULE._local_geometry(path, trace, indices, 2.0, 3.0)

    assert result["goal_initial_vertical_m"] == pytest.approx(0.5)
    assert result["local_initial_vertical_m"] == pytest.approx(0.5)
    assert result["local40_abs_m"] == pytest.approx(0.5)
    assert result["local20_abs_m"] == pytest.approx(0.5)
    assert result["local_terminal_abs_m"] == pytest.approx(0.5)
    assert result["goal_terminal_abs_m"] == pytest.approx(3.0)
    assert result["start_path_distance_m"] == pytest.approx(0.5)
    assert provenance["trace_goal_matches_selected_path"] is True
    assert provenance["trace_goal_tangent_matches_selected_path"] is True
    assert provenance["start_within_jitter"] is True
    assert provenance["route_tangent_matches_selected_path"] is True
    assert provenance["route_tangent_matches_path0"] is True


def test_local_geometry_rejects_start_outside_jitter() -> None:
    path = _line_path()
    trace = _local_trace()
    trace["eta"][0, 2] = 12.0

    _, provenance = MODULE._local_geometry(path, trace, np.arange(4), 2.0, 3.0)

    assert provenance["start_within_jitter"] is False


def test_local_geometry_flags_trace_goal_mismatch_without_raising() -> None:
    path = _line_path()
    trace = _local_trace()
    trace["goal_ned_m"][:] = np.asarray([100.0, 0.0, 12.0])

    result, provenance = MODULE._local_geometry(
        path, trace, np.arange(4), 2.0, 3.0
    )

    assert result["local_initial_abs_m"] == pytest.approx(0.5)
    assert provenance["trace_goal_matches_selected_path"] is False
    assert provenance["goal_distance_to_selected_path_m"] == pytest.approx(2.0)


def test_local_geometry_flags_route_tangent_mismatch_rows() -> None:
    path = _line_path()
    trace = _local_trace()
    trace["route_tangent_ned"][:] = np.asarray([0.0, 1.0, 0.0])

    _, provenance = MODULE._local_geometry(path, trace, np.arange(4), 2.0, 3.0)

    assert provenance["route_tangent_matches_selected_path"] is False
    assert provenance["route_tangent_mismatch_rows"] == [0, 2, 3]
    assert provenance["route_tangent_matches_path0"] is False
    assert provenance["route_tangent_path0_mismatch_rows"] == [0, 2, 3]


def test_h_frame_supports_goal_tangent_projection_artifact() -> None:
    records = [_record(seed) for seed in range(20, 32)]

    result = MODULE._h_frame(records, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["goal_initial_abs_median_m"] == pytest.approx(8.0)
    assert result["local_initial_abs_median_m"] == pytest.approx(0.5)


def test_h_frame_fails_when_local_initial_is_large() -> None:
    records = [
        _record(seed, local_initial_abs_m=3.0, goal_minus_local_abs_m=5.0)
        for seed in range(20, 32)
    ]

    result = MODULE._h_frame(records, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["criteria"]["local_initial_median_le_1_00"] is False
    assert result["criteria"]["local_outside_fraction_le_0_20"] is False


def test_h_local_carryover_supports_early_error_ranking() -> None:
    records = [
        _record(
            seed,
            local40_abs_m=2.0 + 0.2 * (seed - 20),
            local20_abs_m=1.5 + 0.2 * (seed - 20),
            local_terminal_abs_m=3.0 + 0.2 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_local_carryover(records)

    assert result["passes_all"] is True
    assert result["local40_terminal_rho"] == pytest.approx(1.0)
    assert result["local20_terminal_rho"] == pytest.approx(1.0)


def test_h_local_carryover_fails_when_early_error_is_flat() -> None:
    records = [
        _record(
            seed,
            local40_abs_m=0.2,
            local20_abs_m=0.1,
            local_terminal_abs_m=3.0 + 0.2 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_local_carryover(records)

    assert result["passes_all"] is False
    assert result["criteria"]["local40_terminal_rho_ge_0_50"] is False
    assert result["criteria"]["local40_terminal_ratio_ge_0_50"] is False


def test_h_local_initial_weak_supports_small_uncorrelated_initial_error() -> None:
    records = [
        _record(
            seed,
            local_initial_abs_m=0.2 + 0.1 * ((seed - 20) % 2),
            local_terminal_abs_m=4.0 + 0.2 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_local_initial_weak(records, goal_radius_m=2.0)

    assert result["passes_all"] is True
    assert result["local_initial_terminal_ratio"] < 0.25


def test_h_local_initial_weak_fails_when_initial_error_ranks() -> None:
    records = [
        _record(
            seed,
            local_initial_abs_m=1.0 + 0.2 * (seed - 20),
            local_terminal_abs_m=2.0 + 0.2 * (seed - 20),
        )
        for seed in range(20, 32)
    ]

    result = MODULE._h_local_initial_weak(records, goal_radius_m=2.0)

    assert result["passes_all"] is False
    assert result["criteria"]["local_initial_terminal_rho_le_0_30"] is False
    assert result["criteria"]["local_initial_terminal_ratio_le_0_25"] is False


def _block(frame: bool, carryover: bool, weak: bool) -> dict[str, object]:
    return {
        "h_frame": {"passes_all": frame},
        "h_local_carryover": {"passes_all": carryover},
        "h_local_initial_weak": {"passes_all": weak},
    }


@pytest.mark.parametrize(
    ("frame", "carryover", "weak", "expected"),
    (
        (True, True, True, "FRAME_ARTIFACT_WITH_LOCAL_CARRYOVER"),
        (True, True, False, "FRAME_ARTIFACT_LOCAL_INITIAL_RANKS"),
        (True, False, True, "FRAME_ARTIFACT_NO_LOCAL_CARRYOVER"),
        (False, True, False, "INITIAL_OFFSET_SURVIVES_LOCAL_FRAME"),
        (False, False, False, "NO_SUPPORTED_MECHANISM"),
    ),
)
def test_overall_verdict_mapping(
    frame: bool,
    carryover: bool,
    weak: bool,
    expected: str,
) -> None:
    blocks = [_block(frame, carryover, weak) for _ in range(4)]

    result = MODULE._overall_verdict(blocks, invalid=False)

    assert result["decision"] == expected


def test_overall_verdict_reserves_invalid_diagnostic() -> None:
    result = MODULE._overall_verdict(
        [_block(True, True, True) for _ in range(4)],
        invalid=True,
        invalid_reasons=["trace path provenance mismatch"],
    )

    assert result["decision"] == "INVALID_DIAGNOSTIC"
    assert result["invalid_reasons"] == ["trace path provenance mismatch"]
