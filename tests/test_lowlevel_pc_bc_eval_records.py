from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pc_bc.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_pc_bc", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_episode_records_align_with_seeds_and_path_metrics() -> None:
    module = _load_module()
    metrics = {
        "active_steps": np.array([10, 20], dtype=np.int64),
        "reward_sum": np.array([2.0, 4.0], dtype=np.float64),
        "goal_terminations": np.array([1, 0], dtype=np.int64),
        "timeout_terminations": np.array([0, 1], dtype=np.int64),
        "safety_terminations": np.array([0, 0], dtype=np.int64),
        "material_saturation_steps": np.array([2, 4], dtype=np.int64),
        "path_steps": np.array([10, 20], dtype=np.int64),
        "horizontal_error_sq_sum": np.array([10.0, 80.0], dtype=np.float64),
        "vertical_error_sq_sum": np.array([20.0, 20.0], dtype=np.float64),
        "three_d_error_sq_sum": np.array([30.0, 100.0], dtype=np.float64),
        "final_progress_m": np.array([30.0, 35.0], dtype=np.float64),
    }

    rows = module._episode_records(
        metrics,
        (7, 9),
        path_enabled=True,
        path_length_m=35.0,
    )

    assert [row["seed"] for row in rows] == [7, 9]
    assert rows[0]["success"] is True
    assert rows[1]["timeout"] is True
    assert rows[0]["material_saturation_rate"] == 0.2
    assert rows[1]["material_saturation_rate"] == 0.2
    assert rows[0]["path_three_d_rmse_m"] == np.sqrt(3.0)
    assert rows[1]["path_three_d_rmse_m"] == np.sqrt(5.0)
    assert rows[0]["path_completion_ratio"] == np.clip(30.0 / 35.0, 0.0, 1.0)
    assert rows[1]["path_completion_ratio"] == 1.0


def test_episode_records_skip_path_fields_for_non_path_task() -> None:
    module = _load_module()
    metrics = {
        "active_steps": np.array([5], dtype=np.int64),
        "reward_sum": np.array([1.0], dtype=np.float64),
        "goal_terminations": np.array([1], dtype=np.int64),
        "timeout_terminations": np.array([0], dtype=np.int64),
        "safety_terminations": np.array([0], dtype=np.int64),
        "material_saturation_steps": np.array([0], dtype=np.int64),
    }

    rows = module._episode_records(
        metrics,
        (3,),
        path_enabled=False,
        path_length_m=0.0,
    )

    assert rows == [
        {
            "seed": 3,
            "steps": 5,
            "success": True,
            "goal_plane_miss": False,
            "timeout": False,
            "safety": False,
            "goal_capture_pnav": True,
            "goal_capture_eta": False,
            "material_saturation_rate": 0.0,
            "mean_reward": 0.2,
        }
    ]


def test_episode_records_expose_pnav_and_eta_goal_capture() -> None:
    module = _load_module()
    metrics = {
        "active_steps": np.array([5, 5], dtype=np.int64),
        "reward_sum": np.array([1.0, 1.0], dtype=np.float64),
        "goal_terminations": np.array([1, 0], dtype=np.int64),
        "goal_plane_miss_terminations": np.array([0, 1], dtype=np.int64),
        "timeout_terminations": np.array([0, 0], dtype=np.int64),
        "safety_terminations": np.array([0, 0], dtype=np.int64),
        "pnav_goal_capture_seen": np.array([False, True], dtype=bool),
        "eta_goal_capture_seen": np.array([True, True], dtype=bool),
        "material_saturation_steps": np.array([0, 0], dtype=np.int64),
    }

    rows = module._episode_records(
        metrics,
        (3, 4),
        path_enabled=False,
        path_length_m=0.0,
    )

    assert rows[0]["success"] is True
    assert rows[0]["goal_capture_pnav"] is False
    assert rows[0]["goal_capture_eta"] is True
    assert rows[1]["success"] is False
    assert rows[1]["goal_capture_pnav"] is True
    assert rows[1]["goal_capture_eta"] is True


def test_actor_current_ablation_zero_and_shuffle_preserve_contract() -> None:
    module = _load_module()
    current = torch.arange(24, dtype=torch.float32).reshape(8, 3)

    zeroed = module._ablate_actor_current(current, "current-zero")
    assert torch.equal(zeroed, torch.zeros_like(current))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    shuffled = module._ablate_actor_current(
        current,
        "current-shuffle",
        generator=generator,
    )
    assert shuffled.shape == current.shape
    assert not torch.equal(shuffled, current)
    assert torch.equal(
        torch.sort(shuffled, dim=0).values,
        torch.sort(current, dim=0).values,
    )

    with pytest.raises(ValueError, match="at least two"):
        module._ablate_actor_current(current[:1], "current-shuffle")
    with pytest.raises(ValueError, match="unsupported"):
        module._ablate_actor_current(current, "current-drop")


def test_path_progress_bucket_indices_cover_closed_interval() -> None:
    module = _load_module()
    buckets = module._path_progress_bucket_indices(
        np.asarray([0.0, 0.249, 0.25, 0.749, 0.75, 1.0])
    )
    assert buckets.tolist() == [0, 0, 1, 2, 3, 3]


def test_path_progress_trace_is_optional_and_reported() -> None:
    module = _load_module()
    metrics = module._new_episode_metrics(1, write_trace=True)
    metrics["active_steps"][:] = 4
    metrics["path_steps"][:] = 4
    metrics["pre_goal_path_steps"][:] = 4
    metrics["pre_goal_three_d_error_sq_sum"][:] = 16.0
    metrics["three_d_error_sq_sum"][:] = 16.0
    metrics["horizontal_error_sq_sum"][:] = 9.0
    metrics["vertical_error_sq_sum"][:] = 7.0
    metrics["goal_terminations"][:] = 1
    metrics["pnav_goal_capture_seen"][:] = True
    metrics["eta_goal_capture_seen"][:] = True
    metrics["path_progress_bucket_steps"][0] = [1, 1, 1, 1]
    metrics["path_progress_bucket_three_d_error_sq_sum"][0] = [1.0, 3.0, 5.0, 7.0]
    for name in (
        "roll_abs_values",
        "roll_rate_abs_values",
        "sensor_p_nav_error_m",
        "sensor_nav_age_s",
        "sensor_dvl_bottom_age_s",
        "sensor_current_estimate_age_s",
    ):
        metrics[name] = [np.array([0.0])]
    for name in (
        "sensor_dvl_bottom_valid",
        "sensor_current_estimate_valid",
        "sensor_usbl_valid",
        "sensor_usbl_outlier",
    ):
        metrics[name] = [np.array([True])]

    rows = module._episode_records(
        metrics,
        (7,),
        path_enabled=True,
        path_length_m=10.0,
        termination_mode="endpoint_stop",
    )
    assert rows[0]["path_progress_3d_rmse_m"]["0-25%"] == pytest.approx(1.0)
    assert rows[0]["path_progress_3d_rmse_m"]["25-50%"] == pytest.approx(np.sqrt(3.0))
    assert rows[0]["path_progress_3d_rmse_m"]["50-75%"] == pytest.approx(np.sqrt(5.0))
    assert rows[0]["path_progress_3d_rmse_m"]["75-100%"] == pytest.approx(np.sqrt(7.0))
    summary = module._episode_summary(
        metrics,
        path_enabled=True,
        path_length_m=10.0,
        termination_mode="endpoint_stop",
    )
    assert summary["path_progress_3d_rmse_m"]["0-25%"] == pytest.approx(1.0)
    assert summary["path_progress_3d_rmse_m"]["75-100%"] == pytest.approx(np.sqrt(7.0))
    assert "path_progress_3d_rmse_m" not in module._new_episode_metrics(1)
