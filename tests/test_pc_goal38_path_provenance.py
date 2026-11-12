from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.config import load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path_bank

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_pc_goal38_path_provenance.py"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "lowlevel"
    / "pc_goal27_c1_command_residual_default_20261109.json"
)


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("goal38_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["goal38_audit"] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_trace(seed_start: int = 20, seed_count: int = 2) -> tuple[dict[str, np.ndarray], Any]:
    config = load_config(DEFAULT_CONFIG)
    bank = tuple(generate_sinusoidal_3d_path_bank(config.reference))
    seeds = np.repeat(np.arange(seed_start, seed_start + seed_count), 3)
    steps = np.tile(np.arange(3, dtype=np.int64), seed_count)
    eta = np.zeros((seeds.size, 6), dtype=np.float64)
    goal = np.zeros((seeds.size, 3), dtype=np.float64)
    goal_tangent = np.zeros((seeds.size, 3), dtype=np.float64)
    route_tangent = np.zeros((seeds.size, 3), dtype=np.float64)
    progress = np.zeros(seeds.size, dtype=np.float64)
    for row, seed in enumerate(seeds.tolist()):
        path = bank[int(seed) % len(bank)]
        position = path.point_at_s(0.2 + 0.1 * float(steps[row]))
        projection = path.project(position)
        eta[row, :3] = position
        goal[row] = path.goal_ned_m
        goal_tangent[row] = path.tangents_ned[-1]
        route_tangent[row] = projection.tangent_ned
        progress[row] = projection.progress_m
    trace = {
        "episode_seed": seeds,
        "step": steps,
        "eta": eta,
        "goal_ned_m": goal,
        "goal_tangent_ned": goal_tangent,
        "route_tangent_ned": route_tangent,
        "path_progress_m": progress,
    }
    jitter = math.sqrt(
        2.0 * config.reference.start_jitter_xy_m**2 + config.reference.start_jitter_z_m**2
    )
    return trace, (bank, jitter)


def test_goal38_provenance_accepts_seed_matched_trace() -> None:
    audit = _module()
    trace, (bank, jitter) = _synthetic_trace()
    result = audit._audit_provenance(trace, bank, {20, 21}, jitter)
    assert result["valid"]
    assert result["counts"]["goal"] == 2
    assert result["counts"]["route_tangent"] == 2
    assert result["counts"]["path_progress"] == 2


def test_goal38_provenance_rejects_path_zero_goal() -> None:
    audit = _module()
    trace, (bank, jitter) = _synthetic_trace()
    trace["goal_ned_m"][trace["episode_seed"] == 21] = bank[0].goal_ned_m
    result = audit._audit_provenance(trace, bank, {20, 21}, jitter)
    assert not result["valid"]
    assert result["counts"]["goal"] == 1
    assert result["mismatch_seeds"]["goal"] == [21]


def test_goal38_control_comparison_detects_actuator_change() -> None:
    audit = _module()
    common = {
        "episode_seed": np.array([20, 20, 21, 21], dtype=np.int64),
        "step": np.array([0, 1, 0, 1], dtype=np.int64),
        "actual_actuator": np.zeros((4, 3), dtype=np.float64),
        "eta": np.zeros((4, 6), dtype=np.float64),
        "nu": np.zeros((4, 6), dtype=np.float64),
        "requested_action": np.zeros((4, 3), dtype=np.float64),
        "terminal_kind": np.array([0, 0, 0, 1], dtype=np.int64),
    }
    reference = {key: value.copy() for key, value in common.items()}
    fresh = {key: value.copy() for key, value in common.items()}
    assert audit._compare_control_fields(fresh, reference)["valid"]

    fresh["actual_actuator"][2, 0] = 1e-6
    result = audit._compare_control_fields(fresh, reference)
    assert not result["valid"]
    assert result["field_max_abs_diff"]["actual_actuator"] == 1e-6


def test_goal38_control_comparison_handles_trace_ordering() -> None:
    audit = _module()
    reference = {
        "episode_seed": np.array([20, 20, 21, 21], dtype=np.int64),
        "step": np.array([0, 1, 0, 1], dtype=np.int64),
        "actual_actuator": np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        ),
        "eta": np.zeros((4, 6), dtype=np.float64),
        "nu": np.zeros((4, 6), dtype=np.float64),
        "requested_action": np.zeros((4, 3), dtype=np.float64),
        "terminal_kind": np.arange(4, dtype=np.int64),
    }
    order = np.array([2, 0, 3, 1], dtype=np.int64)
    fresh = {key: value[order].copy() for key, value in reference.items()}
    result = audit._compare_control_fields(fresh, reference)
    assert result["valid"]
    assert result["common_rows"] == 4
