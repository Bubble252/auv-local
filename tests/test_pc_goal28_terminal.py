from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/analyze_pc_goal28_terminal.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal28_terminal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _episode(
    seed: int,
    *,
    success: bool = True,
    eta: bool = True,
    plane_miss: bool = False,
    timeout: bool = False,
    safety: bool = False,
    miss_distance: float | None = None,
    final_goal_distance: float | None = 1.5,
) -> dict:
    return {
        "seed": seed,
        "success": success,
        "goal_capture_eta": eta,
        "goal_plane_miss": plane_miss,
        "timeout": timeout,
        "safety": safety,
        "miss_distance_at_plane_m": miss_distance,
        "final_goal_distance_m": final_goal_distance,
        "pre_goal_3d_rmse_m": 2.0,
        "path_three_d_rmse_m": 1.8,
        "material_saturation_rate": 0.2,
        "residual_to_base_l2_ratio_mean": 0.0,
        "steps": 100,
    }


def _summary(episodes: list[dict]) -> dict:
    return {
        "schema_version": MODULE.SUMMARY_SCHEMA,
        "seeds": [item["seed"] for item in episodes],
        "endpoint_stop_enabled": True,
        "episodes": episodes,
        "metrics": {
            "terminal_speed_ratio_mean": 1.1,
            "nav_position_error_mean_m": 0.3,
            "nav_position_error_p95_m": 0.7,
        },
    }


def test_outcome_requires_exactly_one_terminal_flag() -> None:
    assert MODULE._outcome(_episode(20)) == "success"
    assert (
        MODULE._outcome(
            _episode(20, success=False, eta=False, plane_miss=True)
        )
        == "plane_miss"
    )
    with pytest.raises(ValueError, match="non-exclusive"):
        MODULE._outcome(_episode(20, plane_miss=True))


def test_load_summary_rejects_non_dev_seeds(tmp_path: Path) -> None:
    episodes = [_episode(100 + index) for index in range(40)]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary(episodes)), encoding="utf-8")
    with pytest.raises(ValueError, match="dev seeds 20-59"):
        MODULE.load_summary(path)


def test_aggregate_separates_pnav_only_success_and_eta_capture(tmp_path: Path) -> None:
    episodes = [
        _episode(20, eta=True),
        _episode(21, eta=False),
        _episode(
            22,
            success=False,
            eta=False,
            plane_miss=True,
            miss_distance=2.5,
            final_goal_distance=2.2,
        ),
    ]
    arm = MODULE._aggregate(_summary(episodes), source=tmp_path / "summary.json")

    assert arm["success_protocol_rate"] == pytest.approx(2 / 3)
    assert arm["success_eta_rate"] == pytest.approx(1 / 3)
    assert arm["pnav_only_success_rate"] == pytest.approx(1 / 3)
    assert arm["pnav_only_share_of_protocol_success"] == pytest.approx(0.5)
    assert arm["goal_plane_miss_rate"] == pytest.approx(1 / 3)
    assert arm["miss_distance_at_plane_m"]["median"] == pytest.approx(2.5)


def test_paired_reports_outcome_transitions_and_rate_deltas(tmp_path: Path) -> None:
    base_episodes = [
        _episode(20),
        _episode(21),
        _episode(22, success=False, eta=False, plane_miss=True, miss_distance=2.0),
    ]
    target_episodes = [
        _episode(20),
        _episode(21, success=False, eta=False, plane_miss=True, miss_distance=3.0),
        _episode(22),
    ]
    base = MODULE._aggregate(_summary(base_episodes), source=tmp_path / "base.json")
    target = MODULE._aggregate(_summary(target_episodes), source=tmp_path / "target.json")
    pair = MODULE._paired(base, target)

    assert pair["rate_delta"]["success_protocol"] == pytest.approx(0.0)
    assert pair["rate_delta"]["goal_plane_miss"] == pytest.approx(0.0)
    assert pair["outcome_transitions"]["success->plane_miss"] == 1
    assert pair["outcome_transitions"]["plane_miss->success"] == 1
    assert pair["final_goal_distance_m"]["n"] == 3
