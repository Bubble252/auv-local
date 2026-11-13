#!/usr/bin/env python3
"""Read-only local path-frame diagnostic for Goal37.

The script consumes the four frozen Goal31 formal trace directories and
reconstructs the deterministic Goal31 path bank. It never loads a policy,
steps an environment, trains, or reads held-out seeds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.config import load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path_bank

SCHEMA_VERSION = "auv_pc_goal37_local_path_frame_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}
PATH_BANK_SIZE = 32
MAX_START_JITTER_M = math.sqrt(2.0 * 0.5**2 + 0.5**2)
MIN_VERTICAL_MISS_SEEDS = 10
GEOMETRY_TOLERANCE = 1e-8
PROJECTION_TANGENT_TOLERANCE = 1e-6
EPS = 1e-12

FRAME_GOAL_INITIAL_MIN_M = 5.00
FRAME_LOCAL_INITIAL_MAX_M = 1.00
FRAME_LOCAL_OUTSIDE_MAX = 0.20
FRAME_DELTA_MIN_M = 5.00

LOCAL_CARRYOVER_RHO_MIN = 0.50
LOCAL_CARRYOVER_RATIO_MIN = 0.50

LOCAL_INITIAL_RHO_MAX = 0.30
LOCAL_INITIAL_RATIO_MAX = 0.25
LOCAL_INITIAL_OUTSIDE_MAX = 0.20

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "path_index",
    "outcome",
    "vertical_miss",
    "goal_initial_vertical_m",
    "local_initial_vertical_m",
    "goal_initial_abs_m",
    "local_initial_abs_m",
    "goal_minus_local_abs_m",
    "local40_vertical_m",
    "local40_abs_m",
    "local20_vertical_m",
    "local20_abs_m",
    "local_terminal_vertical_m",
    "local_terminal_abs_m",
    "goal_terminal_abs_m",
    "route_goal_tangent_angle_rad",
    "start_path_distance_m",
    "trace_goal_matches_selected_path",
    "trace_goal_tangent_matches_selected_path",
    "trace_goal_matches_path0",
    "trace_goal_tangent_matches_path0",
    "start_within_jitter",
    "route_tangent_matches_selected_path",
    "route_tangent_mismatch_rows",
    "route_tangent_matches_path0",
    "route_tangent_path0_mismatch_rows",
    "goal_distance_to_selected_path_m",
    "goal_tangent_distance_to_selected_path",
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
_GOAL32 = _load_module(
    _HERE / "analyze_pc_goal32_terminal_entry.py", "_goal37_goal32_helpers"
)
_GOAL34 = _load_module(
    _HERE / "analyze_pc_goal34_global_vertical.py", "_goal37_goal34_helpers"
)
_median = _GOAL34._median
_fraction = _GOAL34._fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-default",
        type=Path,
        default=Path(
            "configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json"
        ),
    )
    parser.add_argument(
        "--config-strict125",
        type=Path,
        default=Path(
            "configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json"
        ),
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_path_bank(config_path: Path) -> tuple[Any, ...]:
    config = load_config(config_path)
    bank = tuple(generate_sinusoidal_3d_path_bank(config.reference))
    if len(bank) != PATH_BANK_SIZE:
        raise ValueError(f"expected {PATH_BANK_SIZE} paths, got {len(bank)}")
    return bank


def _path_for_seed(path_bank: tuple[Any, ...], seed: int) -> tuple[Any, int]:
    index = int(seed) % len(path_bank)
    return path_bank[index], index


def _first_crossing(distance: np.ndarray, threshold_m: float) -> int:
    hits = np.flatnonzero(distance <= threshold_m)
    if hits.size == 0:
        raise ValueError(f"episode never reaches {threshold_m:.1f} m to goal")
    return int(hits[0])


def _goal_frame_initial(
    eta: np.ndarray,
    goal: np.ndarray,
    goal_tangent: np.ndarray,
) -> float:
    tangent_norm = float(np.linalg.norm(goal_tangent))
    if tangent_norm <= EPS:
        raise ValueError("goal tangent is a zero vector")
    unit_tangent = goal_tangent / tangent_norm
    offset = eta[0, :3] - goal
    longitudinal = float(np.dot(offset, unit_tangent))
    return float(offset[2] - longitudinal * unit_tangent[2])


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError("vector is a zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


def _angle_between(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = _unit(left)
    right_unit = _unit(right)
    return float(math.acos(float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))))


def _projection_tangent_matches(
    route_tangent: np.ndarray,
    projection_tangent: np.ndarray,
) -> bool:
    return float(
        np.linalg.norm(_unit(route_tangent) - _unit(projection_tangent))
    ) <= PROJECTION_TANGENT_TOLERANCE


def _local_geometry(
    path: Any,
    trace: Any,
    indices: np.ndarray,
    goal_radius_m: float,
    goal_terminal_abs_vertical_m: float,
    *,
    path0: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eta = np.asarray(trace["eta"][indices], dtype=np.float64)
    goal_rows = np.asarray(trace["goal_ned_m"][indices], dtype=np.float64)
    tangent_rows = np.asarray(trace["goal_tangent_ned"][indices], dtype=np.float64)
    route_rows = np.asarray(trace["route_tangent_ned"][indices], dtype=np.float64)
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)

    if time_s.size < 2:
        raise ValueError("episode has fewer than two trace rows")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("episode time is not finite and strictly increasing")
    if eta.shape != (time_s.size, 6):
        raise ValueError("eta shape does not match the episode rows")
    if goal_rows.shape != (time_s.size, 3) or tangent_rows.shape != (time_s.size, 3):
        raise ValueError("goal/tangent shapes do not match the episode rows")
    if route_rows.shape != (time_s.size, 3):
        raise ValueError("route tangent shape does not match the episode rows")
    if any(
        not np.all(np.isfinite(array))
        for array in (eta, goal_rows, tangent_rows, route_rows)
    ):
        raise ValueError("local geometry contains non-finite values")
    if float(np.max(np.ptp(goal_rows, axis=0))) > GEOMETRY_TOLERANCE:
        raise ValueError("goal changes within an episode")
    if float(np.max(np.ptp(tangent_rows, axis=0))) > GEOMETRY_TOLERANCE:
        raise ValueError("goal tangent changes within an episode")

    goal = goal_rows[0]
    goal_tangent = tangent_rows[0]
    if float(np.linalg.norm(goal_tangent)) <= EPS:
        raise ValueError("goal tangent is a zero vector")
    goal_distance_to_path = float(np.linalg.norm(goal - path.goal_ned_m))
    goal_tangent_distance_to_path = float(
        np.linalg.norm(_unit(goal_tangent) - _unit(path.tangents_ned[-1]))
    )
    goal_matches_path = goal_distance_to_path <= GEOMETRY_TOLERANCE
    goal_tangent_matches_path = (
        goal_tangent_distance_to_path <= GEOMETRY_TOLERANCE
    )

    start_distance = float(np.linalg.norm(eta[0, :3] - path.start_ned_m))
    start_within_jitter = start_distance <= MAX_START_JITTER_M + 1e-6

    distance = np.linalg.norm(eta[:, :3] - goal, axis=1)
    i40 = _first_crossing(distance, 40.0)
    i20 = _first_crossing(distance, 20.0)
    if i20 < i40:
        raise ValueError("40 m crossing occurs after 20 m crossing")
    last = int(time_s.size - 1)

    goal_initial = _goal_frame_initial(eta, goal, goal_tangent)
    projection_rows: dict[int, Any] = {}
    route_tangent_mismatch_rows: list[int] = []
    route_tangent_path0_mismatch_rows: list[int] = []
    reference_path0 = path if path0 is None else path0
    for row in (0, i40, i20, last):
        projection = path.project(eta[row, :3])
        if not _projection_tangent_matches(route_rows[row], projection.tangent_ned):
            if int(row) not in route_tangent_mismatch_rows:
                route_tangent_mismatch_rows.append(int(row))
        projection_path0 = reference_path0.project(eta[row, :3])
        if not _projection_tangent_matches(
            route_rows[row], projection_path0.tangent_ned
        ):
            if int(row) not in route_tangent_path0_mismatch_rows:
                route_tangent_path0_mismatch_rows.append(int(row))
        projection_rows[row] = projection

    local_initial = float(projection_rows[0].vertical_error_m)
    local40 = float(projection_rows[i40].vertical_error_m)
    local20 = float(projection_rows[i20].vertical_error_m)
    local_terminal = float(projection_rows[last].vertical_error_m)
    local_terminal_abs = abs(local_terminal)
    goal_terminal_abs = abs(float(goal_terminal_abs_vertical_m))
    route_goal_angle = _angle_between(route_rows[0], goal_tangent)

    values = {
        "goal_initial_vertical_m": goal_initial,
        "local_initial_vertical_m": local_initial,
        "goal_initial_abs_m": abs(goal_initial),
        "local_initial_abs_m": abs(local_initial),
        "goal_minus_local_abs_m": abs(goal_initial) - abs(local_initial),
        "local40_vertical_m": local40,
        "local40_abs_m": abs(local40),
        "local20_vertical_m": local20,
        "local20_abs_m": abs(local20),
        "local_terminal_vertical_m": local_terminal,
        "local_terminal_abs_m": local_terminal_abs,
        "goal_terminal_abs_m": goal_terminal_abs,
        "route_goal_tangent_angle_rad": route_goal_angle,
        "start_path_distance_m": start_distance,
        "goal_radius_m": float(goal_radius_m),
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("local geometry metric is not finite")
    validation = {
        "trace_goal_matches_selected_path": goal_matches_path,
        "trace_goal_tangent_matches_selected_path": goal_tangent_matches_path,
        "start_within_jitter": start_within_jitter,
        "route_tangent_matches_selected_path": not route_tangent_mismatch_rows,
        "route_tangent_mismatch_rows": route_tangent_mismatch_rows,
        "route_tangent_matches_path0": not route_tangent_path0_mismatch_rows,
        "route_tangent_path0_mismatch_rows": route_tangent_path0_mismatch_rows,
        "goal_distance_to_selected_path_m": goal_distance_to_path,
        "goal_tangent_distance_to_selected_path": goal_tangent_distance_to_path,
    }
    return values, validation


def _load_records(
    formal_root: Path,
    scenario: str,
    mode: str,
    path_bank: tuple[Any, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    summary, records, trace = _GOAL32._load_records(formal_root, scenario, mode)
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    enriched: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{scenario}/{mode}: no trace rows for seed {seed}")
        path, path_index = _path_for_seed(path_bank, seed)
        values, provenance = _local_geometry(
            path,
            trace,
            indices,
            GOAL_RADIUS_M[scenario],
            float(record["terminal_abs_vertical_m"]),
            path0=path_bank[0],
        )
        item = dict(record)
        item.update(values)
        item.update(provenance)
        item["path_index"] = path_index
        trace_goal = np.asarray(trace["goal_ned_m"][indices[0]], dtype=np.float64)
        trace_goal_tangent = np.asarray(
            trace["goal_tangent_ned"][indices[0]], dtype=np.float64
        )
        item["trace_goal_matches_path0"] = bool(
            np.linalg.norm(trace_goal - path_bank[0].goal_ned_m)
            <= GEOMETRY_TOLERANCE
        )
        item["trace_goal_tangent_matches_path0"] = bool(
            np.linalg.norm(
                _unit(trace_goal_tangent) - _unit(path_bank[0].tangents_ned[-1])
            )
            <= GEOMETRY_TOLERANCE
        )
        enriched.append(item)
    return summary, enriched, trace


def _provenance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def matching(key: str) -> int:
        return int(sum(bool(row[key]) for row in records))

    mismatch_seeds = [
        int(row["seed"])
        for row in records
        if not all(
            bool(row[key])
            for key in (
                "trace_goal_matches_selected_path",
                "trace_goal_tangent_matches_selected_path",
                "start_within_jitter",
                "route_tangent_matches_selected_path",
            )
        )
    ]
    return {
        "seed_count": len(records),
        "goal_matches_selected_path_seed_count": matching(
            "trace_goal_matches_selected_path"
        ),
        "goal_tangent_matches_selected_path_seed_count": matching(
            "trace_goal_tangent_matches_selected_path"
        ),
        "start_within_jitter_seed_count": matching("start_within_jitter"),
        "route_tangent_matches_selected_path_seed_count": matching(
            "route_tangent_matches_selected_path"
        ),
        "route_tangent_matches_path0_seed_count": matching(
            "route_tangent_matches_path0"
        ),
        "goal_matches_path0_seed_count": matching("trace_goal_matches_path0"),
        "goal_tangent_matches_path0_seed_count": matching(
            "trace_goal_tangent_matches_path0"
        ),
        "max_goal_distance_to_selected_path_m": max(
            float(row["goal_distance_to_selected_path_m"]) for row in records
        ),
        "max_goal_tangent_distance_to_selected_path": max(
            float(row["goal_tangent_distance_to_selected_path"]) for row in records
        ),
        "max_start_path_distance_m": max(
            float(row["start_path_distance_m"]) for row in records
        ),
        "mismatch_seeds": mismatch_seeds,
    }


def _initial_geometry_differences(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    left = {int(row["seed"]): row for row in baseline}
    right = {int(row["seed"]): row for row in candidate}
    if set(left) != set(right):
        return {"seed_sets_match": False, "max_abs_difference": None}
    fields = (
        "goal_initial_vertical_m",
        "local_initial_vertical_m",
        "goal_initial_abs_m",
        "local_initial_abs_m",
        "start_path_distance_m",
    )
    differences = {
        field: max(abs(float(left[seed][field]) - float(right[seed][field])) for seed in left)
        for field in fields
    }
    return {
        "seed_sets_match": True,
        "max_abs_difference": max(differences.values()),
        "fields": differences,
    }


def _stats(values: list[float | None]) -> dict[str, Any]:
    finite = _GOAL34._finite_values(values)
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5.0)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "goal_initial_abs_m",
        "local_initial_abs_m",
        "goal_minus_local_abs_m",
        "local40_abs_m",
        "local20_abs_m",
        "local_terminal_abs_m",
        "goal_terminal_abs_m",
        "route_goal_tangent_angle_rad",
        "start_path_distance_m",
    )
    return {
        "seed_count": len(records),
        "vertical_miss_seed_count": int(sum(row["vertical_miss"] for row in records)),
        "stats": {key: _stats([row.get(key) for row in records]) for key in keys},
    }


def _h_frame(records: list[dict[str, Any]], goal_radius_m: float) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    goal_median = _median([float(row["goal_initial_abs_m"]) for row in miss])
    local_median = _median([float(row["local_initial_abs_m"]) for row in miss])
    outside_fraction = _fraction(
        [float(row["local_initial_abs_m"]) > goal_radius_m for row in miss]
    )
    delta_median = _median(
        [float(row["goal_minus_local_abs_m"]) for row in miss]
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "goal_initial_median_ge_5_00": bool(
            goal_median is not None and goal_median >= FRAME_GOAL_INITIAL_MIN_M
        ),
        "local_initial_median_le_1_00": bool(
            local_median is not None and local_median <= FRAME_LOCAL_INITIAL_MAX_M
        ),
        "local_outside_fraction_le_0_20": bool(
            outside_fraction is not None
            and outside_fraction <= FRAME_LOCAL_OUTSIDE_MAX
        ),
        "goal_minus_local_median_ge_5_00": bool(
            delta_median is not None and delta_median >= FRAME_DELTA_MIN_M
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "goal_initial_abs_median_m": goal_median,
        "local_initial_abs_median_m": local_median,
        "local_outside_radius_fraction": outside_fraction,
        "goal_minus_local_median_m": delta_median,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_local_carryover(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    terminal = [float(row["local_terminal_abs_m"]) for row in miss]
    rho40 = _GOAL32._spearman(
        [float(row["local40_abs_m"]) for row in miss], terminal
    )
    rho20 = _GOAL32._spearman(
        [float(row["local20_abs_m"]) for row in miss], terminal
    )
    med40 = _median([float(row["local40_abs_m"]) for row in miss])
    med20 = _median([float(row["local20_abs_m"]) for row in miss])
    med_terminal = _median(terminal)
    ratio40 = (
        None
        if med40 is None or med_terminal is None or med_terminal <= EPS
        else med40 / med_terminal
    )
    ratio20 = (
        None
        if med20 is None or med_terminal is None or med_terminal <= EPS
        else med20 / med_terminal
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "local40_terminal_rho_ge_0_50": bool(
            rho40 is not None and rho40 >= LOCAL_CARRYOVER_RHO_MIN
        ),
        "local20_terminal_rho_ge_0_50": bool(
            rho20 is not None and rho20 >= LOCAL_CARRYOVER_RHO_MIN
        ),
        "local40_terminal_ratio_ge_0_50": bool(
            ratio40 is not None and ratio40 >= LOCAL_CARRYOVER_RATIO_MIN
        ),
        "local20_terminal_ratio_ge_0_50": bool(
            ratio20 is not None and ratio20 >= LOCAL_CARRYOVER_RATIO_MIN
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "local40_terminal_rho": rho40,
        "local20_terminal_rho": rho20,
        "local40_abs_median_m": med40,
        "local20_abs_median_m": med20,
        "local_terminal_abs_median_m": med_terminal,
        "local40_terminal_ratio": ratio40,
        "local20_terminal_ratio": ratio20,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_local_initial_weak(
    records: list[dict[str, Any]], goal_radius_m: float
) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    initial = [float(row["local_initial_abs_m"]) for row in miss]
    terminal = [float(row["local_terminal_abs_m"]) for row in miss]
    rho = _GOAL32._spearman(initial, terminal)
    initial_median = _median(initial)
    terminal_median = _median(terminal)
    ratio = (
        None
        if initial_median is None
        or terminal_median is None
        or terminal_median <= EPS
        else initial_median / terminal_median
    )
    outside_fraction = _fraction(
        [value > goal_radius_m for value in initial]
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "local_initial_terminal_rho_le_0_30": bool(
            rho is not None and rho <= LOCAL_INITIAL_RHO_MAX
        ),
        "local_initial_terminal_ratio_le_0_25": bool(
            ratio is not None and ratio <= LOCAL_INITIAL_RATIO_MAX
        ),
        "local_initial_outside_fraction_le_0_20": bool(
            outside_fraction is not None
            and outside_fraction <= LOCAL_INITIAL_OUTSIDE_MAX
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "local_initial_terminal_rho": rho,
        "local_initial_abs_median_m": initial_median,
        "local_terminal_abs_median_m": terminal_median,
        "local_initial_terminal_ratio": ratio,
        "local_initial_outside_radius_fraction": outside_fraction,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _mode_verdict(blocks: list[dict[str, Any]], key: str) -> str:
    passes = [bool(block[key]["passes_all"]) for block in blocks]
    if all(passes):
        return "SUPPORTED"
    if any(passes):
        return "PARTIAL"
    return "FALSIFIED"


def _overall_verdict(
    blocks: list[dict[str, Any]],
    *,
    invalid: bool,
    invalid_reasons: list[str] | None = None,
) -> dict[str, Any]:
    h_frame = _mode_verdict(blocks, "h_frame")
    h_carryover = _mode_verdict(blocks, "h_local_carryover")
    h_initial_weak = _mode_verdict(blocks, "h_local_initial_weak")
    if invalid:
        decision = "INVALID_DIAGNOSTIC"
        recommendation = "路径重建、几何一致性或数据完整性失败；修复后重跑，不解释机制。"
    else:
        supported = {
            name
            for name, verdict in (
                ("H_FRAME", h_frame),
                ("H_LOCAL_CARRYOVER", h_carryover),
                ("H_LOCAL_INITIAL_WEAK", h_initial_weak),
            )
            if verdict == "SUPPORTED"
        }
        partial = {
            name
            for name, verdict in (
                ("H_FRAME", h_frame),
                ("H_LOCAL_CARRYOVER", h_carryover),
                ("H_LOCAL_INITIAL_WEAK", h_initial_weak),
            )
            if verdict == "PARTIAL"
        }
        if not supported:
            decision = "PARTIAL_MECHANISM" if partial else "NO_SUPPORTED_MECHANISM"
            recommendation = (
                "没有机制达到四 block 全支持；只补最便宜的 path/data 验证，"
                "不预注册控制器候选。"
            )
        elif (
            h_frame == "SUPPORTED"
            and h_carryover == "SUPPORTED"
            and h_initial_weak == "SUPPORTED"
        ):
            decision = "FRAME_ARTIFACT_WITH_LOCAL_CARRYOVER"
            recommendation = (
                "初始偏置主要是 goal-tangent 投影伪差；局部 40/20 m 误差仍有"
                "carryover。停止初始偏置线，预注册局部垂直跟踪诊断。"
            )
        elif h_frame == "SUPPORTED" and h_carryover == "SUPPORTED":
            decision = "FRAME_ARTIFACT_LOCAL_INITIAL_RANKS"
            recommendation = (
                "初始偏置主要是投影伪差，但局部初始误差仍排序 terminal；"
                "预注册局部初始条件诊断，不直接调控制器。"
            )
        elif h_frame == "SUPPORTED" and h_carryover == "FALSIFIED":
            decision = "FRAME_ARTIFACT_NO_LOCAL_CARRYOVER"
            recommendation = (
                "初始偏置是投影伪差，且局部 40/20 m 不排序 terminal；"
                "预注册 20 m 后局部相位诊断，不改内环增益。"
            )
        elif h_frame == "FALSIFIED" and h_carryover == "SUPPORTED":
            decision = "INITIAL_OFFSET_SURVIVES_LOCAL_FRAME"
            recommendation = (
                "初始偏置在局部路径坐标系中仍成立；重新预注册真实初始状态诊断。"
            )
        else:
            decision = "PARTIAL_MECHANISM"
            recommendation = "机制只部分满足；补最便宜的局部投影/数据验证，不预注册候选。"
    return {
        "H_FRAME": h_frame,
        "H_LOCAL_CARRYOVER": h_carryover,
        "H_LOCAL_INITIAL_WEAK": h_initial_weak,
        "decision": decision,
        "recommendation": recommendation,
        "invalid_reasons": list(invalid_reasons or ()),
    }


def analyze(
    config_paths: dict[str, Path],
    formal_root: Path,
) -> dict[str, Any]:
    missing_configs = sorted(set(SCENARIOS) - set(config_paths))
    if missing_configs:
        raise ValueError(f"missing scenario configs: {missing_configs}")
    path_banks = {
        scenario: _load_path_bank(config_paths[scenario]) for scenario in SCENARIOS
    }
    scenarios: dict[str, Any] = {}
    blocks: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    validation: dict[str, Any] = {
        "configs": {scenario: str(config_paths[scenario]) for scenario in SCENARIOS},
        "formal_root": str(formal_root),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "path_bank_sizes": {
            scenario: len(path_banks[scenario]) for scenario in SCENARIOS
        },
        "heldout_opened": False,
        "scenarios": {},
    }
    invalid = False
    for scenario in SCENARIOS:
        path_bank = path_banks[scenario]
        baseline_summary, baseline_records, _ = _load_records(
            formal_root, scenario, "baseline", path_bank
        )
        candidate_summary, candidate_records, _ = _load_records(
            formal_root, scenario, "candidate", path_bank
        )
        if {row["seed"] for row in baseline_records} != {
            row["seed"] for row in candidate_records
        }:
            raise ValueError(f"{scenario}: baseline/candidate seed sets differ")
        consistency = _initial_geometry_differences(
            baseline_records, candidate_records
        )
        if (
            not consistency["seed_sets_match"]
            or consistency["max_abs_difference"] is None
            or consistency["max_abs_difference"] > GEOMETRY_TOLERANCE
        ):
            invalid = True
            invalid_reasons.append(
                f"{scenario}: baseline/candidate initial geometry is inconsistent"
            )
        provenance = {
            "baseline": _provenance_summary(baseline_records),
            "candidate": _provenance_summary(candidate_records),
        }
        validation["scenarios"][scenario] = {
            "config": str(config_paths[scenario]),
            "goal_radius_m": GOAL_RADIUS_M[scenario],
            "path_bank_size": len(path_bank),
            "baseline_seed_count": len(baseline_records),
            "candidate_seed_count": len(candidate_records),
            "baseline_checkpoint_sha256": baseline_summary.get("checkpoint_sha256"),
            "candidate_checkpoint_sha256": candidate_summary.get("checkpoint_sha256"),
            "initial_geometry_consistency": consistency,
            "provenance": provenance,
        }
        for mode in MODES:
            mismatch_seeds = provenance[mode]["mismatch_seeds"]
            if mismatch_seeds:
                invalid = True
                invalid_reasons.append(
                    f"{scenario}/{mode}: trace path provenance mismatch for "
                    f"{len(mismatch_seeds)}/40 seeds"
                )
        scenario_blocks: dict[str, Any] = {}
        for mode, records, summary in (
            ("baseline", baseline_records, baseline_summary),
            ("candidate", candidate_records, candidate_summary),
        ):
            block = {
                "scenario": scenario,
                "mode": mode,
                "source": str(formal_root / scenario / mode),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "provenance": provenance[mode],
                "aggregate": _aggregate(records),
                "h_frame": _h_frame(records, GOAL_RADIUS_M[scenario]),
                "h_local_carryover": _h_local_carryover(records),
                "h_local_initial_weak": _h_local_initial_weak(
                    records, GOAL_RADIUS_M[scenario]
                ),
                "records": records,
            }
            if block["h_frame"]["vertical_miss_seed_count"] < MIN_VERTICAL_MISS_SEEDS:
                invalid = True
                invalid_reasons.append(
                    f"{scenario}/{mode}: fewer than {MIN_VERTICAL_MISS_SEEDS} "
                    "vertical-miss seeds"
                )
            scenario_blocks[mode] = block
            blocks.append(block)
        scenarios[scenario] = scenario_blocks
    validation["block_count"] = len(blocks)
    validation["invalid"] = invalid
    validation["invalid_reasons"] = invalid_reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "validation": validation,
        "scenarios": scenarios,
        "overall": _overall_verdict(
            blocks, invalid=invalid, invalid_reasons=invalid_reasons
        ),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal37 局部路径坐标系诊断",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block verdicts",
        "",
        "| scenario | mode | miss n | H-FRAME | H-LOCAL-CARRYOVER | H-LOCAL-INITIAL-WEAK |",
        "|---|---|---:|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            miss = block["h_frame"]["vertical_miss_seed_count"]
            verdicts = [
                block[key]["passes_all"]
                for key in ("h_frame", "h_local_carryover", "h_local_initial_weak")
            ]
            lines.append(
                f"| {scenario} | {mode} | {miss} | "
                + " | ".join("PASS" if item else "FAIL" for item in verdicts)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- H_FRAME: `{result['overall']['H_FRAME']}`",
            f"- H_LOCAL_CARRYOVER: `{result['overall']['H_LOCAL_CARRYOVER']}`",
            f"- H_LOCAL_INITIAL_WEAK: `{result['overall']['H_LOCAL_INITIAL_WEAK']}`",
            f"- decision: `{result['overall']['decision']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## Invalid reasons",
            "",
        ]
    )
    invalid_reasons = result["overall"].get("invalid_reasons", [])
    lines.extend(
        [f"- {reason}" for reason in invalid_reasons]
        if invalid_reasons
        else ["- none"]
    )
    lines.extend(
        [
            "",
            "## Frame and local geometry",
            "",
            "| scenario | mode | goal init med | local init med | outside | delta med | local40 rho | local20 rho | local40 ratio | local20 ratio | local init rho | local/term ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            frame = block["h_frame"]
            carry = block["h_local_carryover"]
            weak = block["h_local_initial_weak"]
            lines.append(
                f"| {scenario} | {mode} | "
                f"{_fmt(frame['goal_initial_abs_median_m'])} | "
                f"{_fmt(frame['local_initial_abs_median_m'])} | "
                f"{_fmt(frame['local_outside_radius_fraction'])} | "
                f"{_fmt(frame['goal_minus_local_median_m'])} | "
                f"{_fmt(carry['local40_terminal_rho'])} | "
                f"{_fmt(carry['local20_terminal_rho'])} | "
                f"{_fmt(carry['local40_terminal_ratio'])} | "
                f"{_fmt(carry['local20_terminal_ratio'])} | "
                f"{_fmt(weak['local_initial_terminal_rho'])} | "
                f"{_fmt(weak['local_initial_terminal_ratio'])} |"
            )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- invalid: `{result['validation']['invalid']}`",
            f"- block count: `{result['validation']['block_count']}`",
            f"- held-out opened: `{result['validation']['heldout_opened']}`",
        ]
    )
    for scenario in SCENARIOS:
        item = result["validation"]["scenarios"][scenario]
        lines.append(
            f"- {scenario}: config=`{item['config']}`, "
            f"path bank={item['path_bank_size']}, "
            f"baseline={item['baseline_seed_count']} seeds, "
            f"candidate={item['candidate_seed_count']} seeds, "
            f"initial geometry max diff={_fmt(item['initial_geometry_consistency']['max_abs_difference'], 12)}"
        )
        for mode in MODES:
            provenance = item["provenance"][mode]
            lines.append(
                f"- {scenario}/{mode}: goal={provenance['goal_matches_selected_path_seed_count']}/40, "
                f"goal tangent={provenance['goal_tangent_matches_selected_path_seed_count']}/40, "
                f"start jitter={provenance['start_within_jitter_seed_count']}/40, "
                f"route tangent={provenance['route_tangent_matches_selected_path_seed_count']}/40, "
                f"route tangent path0={provenance['route_tangent_matches_path0_seed_count']}/40, "
                f"goal matches path0={provenance['goal_matches_path0_seed_count']}/40"
            )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "local_path_frame_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (output / "local_path_frame_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "local_path_frame_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for scenario in SCENARIOS:
            for mode in MODES:
                for record in result["scenarios"][scenario][mode]["records"]:
                    writer.writerow({key: record.get(key) for key in CSV_FIELDS})


def main() -> None:
    args = parse_args()
    result = analyze(
        {
            "default": args.config_default,
            "strict125": args.config_strict125,
        },
        args.formal_root,
    )
    write_outputs(result, args.output)
    print(json.dumps(result["overall"], indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
