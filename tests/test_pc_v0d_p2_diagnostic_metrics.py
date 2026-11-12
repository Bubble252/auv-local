from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

SCRIPT = Path("scripts/audit_pc_v0d_p2_diagnostic_metrics_20261107.py")
SPEC = importlib.util.spec_from_file_location("audit_pc_v0d_p2_diagnostic_metrics_20261107", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _traces() -> dict[str, np.ndarray]:
    return {
        "goal_distance_m": np.array(
            [
                [10.0, 8.0, 1.5, 1.0, 0.5],
                [10.0, 9.0, 8.0, 8.0, 8.0],
            ]
        ),
        "goal_tangent_offset_m": np.array(
            [
                [-10.0, -8.0, -1.0, 1.0, 2.0],
                [-10.0, -9.0, -8.0, -8.0, -8.0],
            ]
        ),
        "hint_error_3d_m": np.array(
            [
                [3.0, 4.0, 1.0, 2.0, 5.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
            ]
        ),
        "hint_longitudinal_error_m": np.array(
            [
                [-10.0, -8.0, -1.0, 1.0, 2.0],
                [-10.0, -9.0, -8.0, -8.0, -8.0],
            ]
        ),
    }


def _metadata() -> dict:
    return {
        "seeds": [20, 21],
        "active_steps": [5, 5],
        "success": [True, False],
        "timeout": [False, True],
        "safety": [False, False],
        "terminal_reason": ["goal", "timeout"],
    }


def test_endpoint_entry_uses_earliest_radius_or_plane_crossing() -> None:
    entry, reason = MODULE._endpoint_entry(
        np.array([10.0, 8.0, 1.5, 1.0]),
        np.array([-10.0, -8.0, -1.0, 1.0]),
        4,
        2.0,
    )
    assert entry == 2
    assert reason == "goal_radius"

    entry, reason = MODULE._endpoint_entry(
        np.array([10.0, 9.0, 8.0, 7.0]),
        np.array([-3.0, -2.0, -1.0, 0.0]),
        4,
        2.0,
    )
    assert entry == 3
    assert reason == "goal_tangent_plane"

    entry, reason = MODULE._endpoint_entry(
        np.array([10.0, 9.0, 8.0, 7.0]),
        np.array([-3.0, -2.0, -1.0, -0.5]),
        4,
        2.0,
    )
    assert entry is None
    assert reason is None


def test_episode_diagnostics_splits_success_and_timeout() -> None:
    success = MODULE._episode_diagnostics(_traces(), _metadata(), 0, 2.0)
    assert success["endpoint_entry_step"] == 2
    assert success["endpoint_entry_reason"] == "goal_radius"
    assert success["success_only_3d_rmse_m"] == math.sqrt(11.0)
    assert success["pre_goal_3d_rmse_m"] == math.sqrt(12.5)
    assert success["post_goal_overshoot_m"] == 2.0
    assert success["post_goal_plane_crossing"] is True
    assert success["terminal_longitudinal_rmse_m"] == math.sqrt(2.0)
    assert success["final_goal_tangent_offset_m"] == 2.0

    timeout = MODULE._episode_diagnostics(_traces(), _metadata(), 1, 2.0)
    assert timeout["endpoint_entry_step"] is None
    assert timeout["success_only_3d_rmse_m"] is None
    assert timeout["pre_goal_3d_rmse_m"] == 2.0
    assert timeout["post_goal_overshoot_m"] is None
    assert timeout["terminal_longitudinal_rmse_m"] is None


def test_aggregate_reports_na_counts_and_diagnostic_only() -> None:
    records = [
        MODULE._episode_diagnostics(_traces(), _metadata(), 0, 2.0),
        MODULE._episode_diagnostics(_traces(), _metadata(), 1, 2.0),
    ]
    aggregate = MODULE._aggregate(records)

    assert aggregate["success_only_3d_rmse_m"]["episode_count"] == 1
    assert aggregate["success_only_3d_rmse_m"]["diagnostic_only"] is True
    assert aggregate["post_goal_overshoot_m"]["endpoint_episode_count"] == 1
    assert aggregate["post_goal_overshoot_m"]["plane_crossing_episode_count"] == 1
    assert aggregate["post_goal_overshoot_m"]["mean"] == 2.0
    assert aggregate["terminal_longitudinal_rmse_m"]["valid_step_count"] == 3


def test_aggregate_reports_na_when_no_success_or_endpoint_episode() -> None:
    traces = _traces()
    traces["goal_distance_m"] = np.full((2, 5), 8.0)
    traces["goal_tangent_offset_m"] = np.full((2, 5), -8.0)
    metadata = {
        "seeds": [20, 21],
        "active_steps": [5, 5],
        "success": [False, False],
        "timeout": [True, True],
        "safety": [False, False],
        "terminal_reason": ["timeout", "timeout"],
    }
    records = [
        MODULE._episode_diagnostics(traces, metadata, 0, 2.0),
        MODULE._episode_diagnostics(traces, metadata, 1, 2.0),
    ]
    aggregate = MODULE._aggregate(records)
    success = aggregate["success_only_3d_rmse_m"]
    overshoot = aggregate["post_goal_overshoot_m"]

    assert success["episode_count"] == 0
    assert success["valid_step_count"] == 0
    assert success["aggregate_value"] is None
    assert success["mean"] is None
    assert success["denominator"] == 0
    assert overshoot["episode_count"] == 0
    assert overshoot["valid_step_count"] == 0
    assert overshoot["aggregate_value"] is None
    assert overshoot["mean"] is None
    assert overshoot["denominator"] == 0


def test_validate_summary_rejects_non_dev_seeds() -> None:
    summary = {
        "seeds": list(range(19, 39)),
        "metadata": {"active_steps": [5] * 20},
        "aggregate": {"episode_count": 20, "hint_rmse_3d_m": 1.0},
        "path_dim": 8,
        "success_group": {},
    }
    try:
        MODULE._validate_summary(summary, [{}] * 20, {})
    except ValueError as exc:
        assert "unexpected seed set" in str(exc)
    else:
        raise AssertionError("non-dev seeds must be rejected")
