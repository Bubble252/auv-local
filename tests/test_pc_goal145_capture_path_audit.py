from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analyze_pc_goal145_capture_path_audit import (
    _paired,
    _trace_episode_metrics,
)


def test_trace_metrics_support_t2_eta_ned_and_final_geometry(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.npz"
    np.savez_compressed(
        trace_path,
        seed=np.asarray([20, 20, 20], dtype=np.int64),
        path_vertical_error=np.asarray([0.5, -0.5, 0.5]),
        path_horizontal_error=np.asarray([1.0, 1.0, 1.0]),
        eta_ned=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
            ]
        ),
        goal_ned=np.asarray([[0.0, 0.0, 0.0]] * 3),
        goal_distance_eta=np.asarray([2.0, 1.5, 2.2360679775]),
        guidance_terminal_phase=np.asarray([0.0, 1.0, 1.0]),
        learned_residual_action=np.asarray([[0.1, 0.2, 0.3]] * 3),
        actual_actuator=np.asarray([[1.0, -1.0, 0.5]] * 3),
    )

    metrics = _trace_episode_metrics(trace_path, interface="t2")[20]

    assert metrics["path_vertical_rmse_m"] == np.float64(0.5)
    assert metrics["path_horizontal_rmse_m"] == np.float64(1.0)
    assert metrics["final_goal_distance_m"] == np.float64(2.2360679775)
    assert metrics["final_vertical_abs_m"] == np.float64(1.0)
    assert metrics["final_horizontal_m"] == np.float64(2.0)
    assert metrics["terminal_phase_rate"] == np.float64(2.0 / 3.0)


def test_paired_vertical_share_is_vertical_when_horizontal_is_equal() -> None:
    policy = {
        "episodes": {
            20: {"success_eta_tight_at_done": True},
            21: {"success_eta_tight_at_done": True},
        },
        "trace": {
            20: {
                "path_vertical_mse_m2": 4.0,
                "path_horizontal_mse_m2": 1.0,
                "final_goal_distance_m": 1.0,
            },
            21: {
                "path_vertical_mse_m2": 4.0,
                "path_horizontal_mse_m2": 1.0,
                "final_goal_distance_m": 1.0,
            },
        },
    }
    zero = {
        "episodes": {
            20: {"success_eta_tight_at_done": False},
            21: {"success_eta_tight_at_done": False},
        },
        "trace": {
            20: {
                "path_vertical_mse_m2": 1.0,
                "path_horizontal_mse_m2": 1.0,
                "final_goal_distance_m": 2.0,
            },
            21: {
                "path_vertical_mse_m2": 1.0,
                "path_horizontal_mse_m2": 1.0,
                "final_goal_distance_m": 2.0,
            },
        },
    }

    paired = _paired(policy, zero, samples=100, seed=0)

    assert paired["vertical_mse_share"] == 1.0
    assert paired["policy_success_count"] == 2
    assert paired["zero_success_count"] == 0
    assert paired["flips"]["policy_only"] == [20, 21]
    assert paired["deltas"]["final_goal_distance_m"]["mean"] == -1.0
