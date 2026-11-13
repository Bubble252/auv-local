#!/usr/bin/env python3
"""Read-only initial vertical geometry diagnostic for Goal36.

The script consumes the four frozen Goal31 formal trace directories. It never
loads a policy, steps an environment, trains, or reads held-out seeds.
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

SCHEMA_VERSION = "auv_pc_goal36_initial_geometry_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}
MIN_VERTICAL_MISS_SEEDS = 10
GEOMETRY_TOLERANCE = 1e-9
GEOMETRY_CONSISTENCY_TOLERANCE = 1e-8
EPS = 1e-12

STATE_OUTSIDE_RADIUS_MIN = 0.60
STATE_INITIAL_RHO_MIN = 0.50
STATE_TERMINAL_RHO_MIN = 0.50
STATE_TERMINAL_RATIO_MIN = 0.80

REFERENCE_GOAL_PITCH_MIN_RAD = 0.03
REFERENCE_ROUTE_PITCH_MIN_RAD = 0.03
REFERENCE_VERTICAL_MIN_M = 1.00
REFERENCE_SHARE_MIN = 0.25
REFERENCE_INITIAL_RHO_MIN = 0.50
REFERENCE_TERMINAL_RHO_MIN = 0.40
REFERENCE_HIGH_LOW_TERMINAL_DELTA_MIN_M = 0.50

START_PITCH_MIN_RAD = 0.03
START_PITCH_RATE_MIN_RAD_S = 0.01
START_VERTICAL_VELOCITY_MIN_MPS = 0.02
START_PARTIAL_RHO_MIN = 0.30
START_HIGH_LOW_TERMINAL_DELTA_MIN_M = 0.50

DOMINANCE_REFERENCE_SHARE_HIGH = 0.60
DOMINANCE_REFERENCE_SHARE_LOW = 0.30

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "outcome",
    "vertical_miss",
    "world_vertical_m",
    "longitudinal_m",
    "reference_vertical_m",
    "initial_vertical_m",
    "reference_share",
    "goal_pitch_rad",
    "route_pitch_rad",
    "initial_pitch_rad",
    "initial_pitch_rate",
    "initial_vertical_v_mps",
    "terminal_abs_vertical_m",
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
_GOAL35 = _load_module(
    _HERE / "analyze_pc_goal35_early_vertical.py", "_goal36_goal35_helpers"
)
_GOAL32 = _GOAL35._GOAL32
_median = _GOAL35._median
_fraction = _GOAL35._fraction
_group_median_delta = _GOAL35._group_median_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _initial_geometry(
    trace: Any,
    indices: np.ndarray,
    goal_radius_m: float,
    terminal_abs_vertical_m: float,
) -> dict[str, Any]:
    """Compute first-row geometry and validate the goal-tangent identity."""

    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    eta = np.asarray(trace["eta"][indices], dtype=np.float64)
    nu = np.asarray(trace["nu"][indices], dtype=np.float64)
    goal_rows = np.asarray(trace["goal_ned_m"][indices], dtype=np.float64)
    tangent_rows = np.asarray(trace["goal_tangent_ned"][indices], dtype=np.float64)
    route_rows = np.asarray(trace["route_tangent_ned"][indices], dtype=np.float64)
    ground_body = np.asarray(
        trace["ground_velocity_body_mps"][indices], dtype=np.float64
    )

    if time_s.size < 2:
        raise ValueError("episode has fewer than two trace rows")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("episode time is not finite and strictly increasing")
    if eta.shape != (time_s.size, 6) or nu.shape != (time_s.size, 6):
        raise ValueError("eta/nu shapes do not match the episode rows")
    if goal_rows.shape != (time_s.size, 3):
        raise ValueError("goal_ned_m shape does not match the episode rows")
    if tangent_rows.shape != (time_s.size, 3):
        raise ValueError("goal_tangent_ned shape does not match the episode rows")
    if route_rows.shape != (time_s.size, 3):
        raise ValueError("route_tangent_ned shape does not match the episode rows")
    if ground_body.shape != (time_s.size, 3):
        raise ValueError("ground_velocity_body_mps shape does not match the episode rows")
    if any(
        not np.all(np.isfinite(array))
        for array in (eta, nu, goal_rows, tangent_rows, route_rows, ground_body)
    ):
        raise ValueError("initial geometry contains non-finite values")
    if float(np.max(np.ptp(goal_rows, axis=0))) > GEOMETRY_TOLERANCE:
        raise ValueError("goal changes within an episode")
    if float(np.max(np.ptp(tangent_rows, axis=0))) > GEOMETRY_TOLERANCE:
        raise ValueError("goal tangent changes within an episode")

    goal = goal_rows[0]
    tangent_norm = float(np.linalg.norm(tangent_rows[0]))
    route_norm = float(np.linalg.norm(route_rows[0]))
    if tangent_norm <= EPS or route_norm <= EPS:
        raise ValueError("goal or route tangent is a zero vector")
    goal_tangent = tangent_rows[0] / tangent_norm
    route_tangent = route_rows[0] / route_norm

    offset = eta[0, :3] - goal
    world_vertical = float(offset[2])
    longitudinal = float(np.dot(offset, goal_tangent))
    reference_vertical = float(-longitudinal * goal_tangent[2])
    initial_vertical = float(world_vertical + reference_vertical)

    _, _, projected_vertical, _ = _GOAL32._decompose_many(
        offset[None, :], goal_tangent[None, :]
    )
    if abs(initial_vertical - float(projected_vertical[0])) > GEOMETRY_TOLERANCE:
        raise ValueError("initial vertical identity does not match goal-tangent projection")

    goal_horizontal = float(np.hypot(goal_tangent[0], goal_tangent[1]))
    route_horizontal = float(np.hypot(route_tangent[0], route_tangent[1]))
    goal_pitch = float(np.arctan2(-goal_tangent[2], goal_horizontal))
    route_pitch = float(np.arctan2(-route_tangent[2], route_horizontal))
    ground_ned = _GOAL32._body_to_world_ned(ground_body[:1], eta[:1])
    initial_vertical_velocity = float(ground_ned[0, 2])

    values = {
        "world_vertical_m": world_vertical,
        "longitudinal_m": longitudinal,
        "reference_vertical_m": reference_vertical,
        "initial_vertical_m": initial_vertical,
        "reference_share": abs(reference_vertical)
        / max(abs(initial_vertical), EPS),
        "goal_pitch_rad": goal_pitch,
        "route_pitch_rad": route_pitch,
        "initial_pitch_rad": float(eta[0, 4]),
        "initial_pitch_rate": float(nu[0, 4]),
        "initial_vertical_v_mps": initial_vertical_velocity,
        "terminal_abs_vertical_m": float(abs(terminal_abs_vertical_m)),
        "goal_radius_m": float(goal_radius_m),
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("initial geometry metric is not finite")

    values.update(
        {
            "abs_world_vertical_m": abs(world_vertical),
            "abs_reference_vertical_m": abs(reference_vertical),
            "abs_initial_vertical_m": abs(initial_vertical),
            "abs_initial_pitch_rad": abs(float(eta[0, 4])),
            "abs_initial_pitch_rate": abs(float(nu[0, 4])),
            "abs_initial_vertical_v_mps": abs(initial_vertical_velocity),
        }
    )
    return values


def _load_records(
    formal_root: Path,
    scenario: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    summary, records, trace = _GOAL32._load_records(formal_root, scenario, mode)
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    enriched: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{scenario}/{mode}: no trace rows for seed {seed}")
        item = dict(record)
        item.update(
            _initial_geometry(
                trace,
                indices,
                GOAL_RADIUS_M[scenario],
                float(record["terminal_abs_vertical_m"]),
            )
        )
        enriched.append(item)
    return summary, enriched, trace


def _geometry_differences(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    left = {int(row["seed"]): row for row in baseline}
    right = {int(row["seed"]): row for row in candidate}
    if set(left) != set(right):
        return {"seed_sets_match": False, "max_abs_difference": None}
    fields = (
        "world_vertical_m",
        "reference_vertical_m",
        "initial_vertical_m",
        "goal_pitch_rad",
        "route_pitch_rad",
        "initial_pitch_rad",
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
    return _GOAL35._stats(values)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "world_vertical_m",
        "reference_vertical_m",
        "initial_vertical_m",
        "reference_share",
        "goal_pitch_rad",
        "route_pitch_rad",
        "initial_pitch_rad",
        "initial_pitch_rate",
        "initial_vertical_v_mps",
        "terminal_abs_vertical_m",
    )
    return {
        "seed_count": len(records),
        "vertical_miss_seed_count": int(sum(row["vertical_miss"] for row in records)),
        "stats": {key: _stats([row.get(key) for row in records]) for key in keys},
    }


def _h_state(records: list[dict[str, Any]], goal_radius_m: float) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    outside_fraction = _fraction(
        [abs(float(row["world_vertical_m"])) > goal_radius_m for row in miss]
    )
    initial_rho = _GOAL32._spearman(
        [abs(float(row["world_vertical_m"])) for row in miss],
        [abs(float(row["initial_vertical_m"])) for row in miss],
    )
    terminal_rho = _GOAL32._spearman(
        [abs(float(row["world_vertical_m"])) for row in miss],
        [float(row["terminal_abs_vertical_m"]) for row in miss],
    )
    world_median = _median([abs(float(row["world_vertical_m"])) for row in miss])
    terminal_median = _median(
        [float(row["terminal_abs_vertical_m"]) for row in miss]
    )
    ratio = (
        None
        if world_median is None or terminal_median is None or terminal_median <= EPS
        else world_median / terminal_median
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "world_outside_radius_fraction_ge_0_60": bool(
            outside_fraction is not None
            and outside_fraction >= STATE_OUTSIDE_RADIUS_MIN
        ),
        "world_initial_rho_ge_0_50": bool(
            initial_rho is not None and initial_rho >= STATE_INITIAL_RHO_MIN
        ),
        "world_terminal_rho_ge_0_50": bool(
            terminal_rho is not None and terminal_rho >= STATE_TERMINAL_RHO_MIN
        ),
        "world_terminal_ratio_ge_0_80": bool(
            ratio is not None and ratio >= STATE_TERMINAL_RATIO_MIN
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "world_outside_radius_fraction": outside_fraction,
        "world_initial_rho": initial_rho,
        "world_terminal_rho": terminal_rho,
        "world_abs_vertical_median_m": world_median,
        "terminal_abs_vertical_median_m": terminal_median,
        "world_terminal_ratio": ratio,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_reference(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    goal_pitch_median = _median(
        [abs(float(row["goal_pitch_rad"])) for row in miss]
    )
    route_pitch_median = _median(
        [abs(float(row["route_pitch_rad"])) for row in miss]
    )
    reference_median = _median(
        [abs(float(row["reference_vertical_m"])) for row in miss]
    )
    share_median = _median([float(row["reference_share"]) for row in miss])
    initial_rho = _GOAL32._spearman(
        [abs(float(row["reference_vertical_m"])) for row in miss],
        [abs(float(row["initial_vertical_m"])) for row in miss],
    )
    terminal_rho = _GOAL32._spearman(
        [abs(float(row["reference_vertical_m"])) for row in miss],
        [float(row["terminal_abs_vertical_m"]) for row in miss],
    )
    terminal_delta = _group_median_delta(
        miss, "abs_reference_vertical_m", "terminal_abs_vertical_m"
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "goal_pitch_median_ge_0_03": bool(
            goal_pitch_median is not None
            and goal_pitch_median >= REFERENCE_GOAL_PITCH_MIN_RAD
        ),
        "route_pitch_median_ge_0_03": bool(
            route_pitch_median is not None
            and route_pitch_median >= REFERENCE_ROUTE_PITCH_MIN_RAD
        ),
        "reference_vertical_median_ge_1_00": bool(
            reference_median is not None
            and reference_median >= REFERENCE_VERTICAL_MIN_M
        ),
        "reference_share_median_ge_0_25": bool(
            share_median is not None and share_median >= REFERENCE_SHARE_MIN
        ),
        "reference_initial_rho_ge_0_50": bool(
            initial_rho is not None and initial_rho >= REFERENCE_INITIAL_RHO_MIN
        ),
        "reference_terminal_rho_ge_0_40": bool(
            terminal_rho is not None and terminal_rho >= REFERENCE_TERMINAL_RHO_MIN
        ),
        "reference_high_low_terminal_delta_ge_0_50": bool(
            terminal_delta is not None
            and terminal_delta >= REFERENCE_HIGH_LOW_TERMINAL_DELTA_MIN_M
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "goal_pitch_abs_median_rad": goal_pitch_median,
        "route_pitch_abs_median_rad": route_pitch_median,
        "reference_abs_vertical_median_m": reference_median,
        "reference_share_median": share_median,
        "reference_initial_rho": initial_rho,
        "reference_terminal_rho": terminal_rho,
        "reference_high_low_terminal_delta_m": terminal_delta,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _partial_spearman(
    x: list[float | None],
    y: list[float | None],
    control: list[float | None],
) -> float | None:
    pairs = [
        (float(left), float(right), float(covariate))
        for left, right, covariate in zip(x, y, control, strict=True)
        if left is not None
        and right is not None
        and covariate is not None
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isfinite(float(covariate))
    ]
    if len(pairs) < 3:
        return None
    left = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    covariate = np.asarray([pair[2] for pair in pairs], dtype=np.float64)
    if (
        np.ptp(left) <= EPS
        or np.ptp(right) <= EPS
        or np.ptp(covariate) <= EPS
    ):
        return None
    left_rank = _GOAL32._rankdata(left)
    right_rank = _GOAL32._rankdata(right)
    control_rank = _GOAL32._rankdata(covariate)
    control_centered = control_rank - np.mean(control_rank)
    denominator = float(np.dot(control_centered, control_centered))
    if denominator <= EPS:
        return None
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    left_residual = left_centered - (
        float(np.dot(left_centered, control_centered)) / denominator
    ) * control_centered
    right_residual = right_centered - (
        float(np.dot(right_centered, control_centered)) / denominator
    ) * control_centered
    residual_denominator = math.sqrt(
        float(np.dot(left_residual, left_residual))
        * float(np.dot(right_residual, right_residual))
    )
    if residual_denominator <= EPS:
        return None
    return float(
        np.dot(left_residual, right_residual) / residual_denominator
    )


def _dynamic_feature(
    records: list[dict[str, Any]],
    *,
    name: str,
    magnitude_key: str,
    minimum_magnitude: float,
) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    terminal = [float(row["terminal_abs_vertical_m"]) for row in miss]
    magnitude = [float(row[magnitude_key]) for row in miss]
    control = [float(row["abs_reference_vertical_m"]) for row in miss]
    median_magnitude = _median(magnitude)
    partial_rho = _partial_spearman(magnitude, terminal, control)
    terminal_delta = _group_median_delta(
        miss, magnitude_key, "terminal_abs_vertical_m"
    )
    criteria = {
        "median_magnitude_ge_threshold": bool(
            median_magnitude is not None and median_magnitude >= minimum_magnitude
        ),
        "partial_spearman_ge_0_30": bool(
            partial_rho is not None and partial_rho >= START_PARTIAL_RHO_MIN
        ),
        "high_low_terminal_delta_ge_0_50": bool(
            terminal_delta is not None
            and terminal_delta >= START_HIGH_LOW_TERMINAL_DELTA_MIN_M
        ),
    }
    return {
        "name": name,
        "magnitude_key": magnitude_key,
        "minimum_magnitude": minimum_magnitude,
        "median_magnitude": median_magnitude,
        "partial_spearman": partial_rho,
        "high_low_terminal_delta_m": terminal_delta,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_start_dynamics(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss_count = sum(row["vertical_miss"] for row in records)
    features = [
        _dynamic_feature(
            records,
            name="pitch",
            magnitude_key="abs_initial_pitch_rad",
            minimum_magnitude=START_PITCH_MIN_RAD,
        ),
        _dynamic_feature(
            records,
            name="pitch_rate",
            magnitude_key="abs_initial_pitch_rate",
            minimum_magnitude=START_PITCH_RATE_MIN_RAD_S,
        ),
        _dynamic_feature(
            records,
            name="vertical_velocity",
            magnitude_key="abs_initial_vertical_v_mps",
            minimum_magnitude=START_VERTICAL_VELOCITY_MIN_MPS,
        ),
    ]
    criteria = {
        "enough_vertical_miss_seeds": miss_count >= MIN_VERTICAL_MISS_SEEDS,
        "any_dynamic_feature_passes": any(
            feature["passes_all"] for feature in features
        ),
    }
    return {
        "vertical_miss_seed_count": miss_count,
        "features": features,
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
    reference_share_block_medians: list[float | None],
) -> dict[str, Any]:
    h_state = _mode_verdict(blocks, "h_state")
    h_reference = _mode_verdict(blocks, "h_reference")
    h_start = _mode_verdict(blocks, "h_start_dynamics")
    if invalid:
        decision = "INVALID_DIAGNOSTIC"
        recommendation = (
            "数据、几何恒等式或 baseline/candidate 初始几何一致性不满足契约；"
            "修复后重跑，不解释机制。"
        )
    else:
        supported = {
            name
            for name, verdict in (
                ("H_STATE", h_state),
                ("H_REFERENCE", h_reference),
                ("H_START_DYNAMICS", h_start),
            )
            if verdict == "SUPPORTED"
        }
        partial = {
            name
            for name, verdict in (
                ("H_STATE", h_state),
                ("H_REFERENCE", h_reference),
                ("H_START_DYNAMICS", h_start),
            )
            if verdict == "PARTIAL"
        }
        share_median = _median(reference_share_block_medians)
        if not supported:
            if partial:
                decision = "PARTIAL_MECHANISM"
                recommendation = (
                    "没有机制达到四个 block 全支持；只补最便宜的数据/分段验证，"
                    "不预注册控制器候选。"
                )
            else:
                decision = "NO_SUPPORTED_MECHANISM"
                recommendation = (
                    "世界垂直状态、起始参考几何和起始动力学均未满足冻结判据；"
                    "停止该初始偏置机制线，不预注册控制器候选。"
                )
        elif supported == {"H_START_DYNAMICS"}:
            decision = "START_DYNAMICS_DOMINANT"
            recommendation = (
                "预注册起始姿态/速度初始化诊断；不要直接调内环增益或重扫 reward。"
            )
        elif "H_START_DYNAMICS" in supported and len(supported) > 1:
            decision = "MULTI_FACTOR"
            recommendation = (
                "起始动力学与其他几何机制同时达到判据；拆成最小单因素，"
                "不允许合并修改。"
            )
        elif supported == {"H_STATE", "H_REFERENCE"}:
            if share_median is not None and share_median >= DOMINANCE_REFERENCE_SHARE_HIGH:
                decision = "REFERENCE_GEOMETRY_DOMINANT"
                recommendation = (
                    "世界垂直差和参考投影都通过判据，但参考份额占优；"
                    "预注册起始参考几何/坐标契约修复，不调增益。"
                )
            elif share_median is not None and share_median <= DOMINANCE_REFERENCE_SHARE_LOW:
                decision = "WORLD_STATE_DOMINANT"
                recommendation = (
                    "世界垂直差和参考投影都通过判据，但世界差份额占优；"
                    "预注册初始状态/数据分布诊断，不调控制器。"
                )
            else:
                decision = "MIXED_STATE_REFERENCE"
                recommendation = (
                    "世界垂直差和 goal-tangent 投影都实质存在且份额接近；"
                    "只允许拆成两个独立诊断，不允许直接改控制器。"
                )
        elif supported == {"H_STATE"}:
            decision = "WORLD_STATE_DOMINANT"
            recommendation = (
                "预注册初始状态/数据分布诊断；不要用内环增益掩盖起始状态差。"
            )
        elif supported == {"H_REFERENCE"}:
            decision = "REFERENCE_GEOMETRY_DOMINANT"
            recommendation = (
                "预注册起始参考几何/坐标契约修复；不要直接调内环增益。"
            )
        else:
            decision = "MULTI_FACTOR"
            recommendation = (
                "多个机制达到判据；拆成最小单因素，不允许合并修改。"
            )
    return {
        "H_STATE": h_state,
        "H_REFERENCE": h_reference,
        "H_START_DYNAMICS": h_start,
        "reference_share_block_median": _median(reference_share_block_medians),
        "decision": decision,
        "recommendation": recommendation,
    }


def analyze(formal_root: Path) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    blocks: list[dict[str, Any]] = []
    validation: dict[str, Any] = {
        "formal_root": str(formal_root),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "heldout_opened": False,
        "scenarios": {},
    }
    invalid = False
    for scenario in SCENARIOS:
        baseline_summary, baseline_records, _ = _load_records(
            formal_root, scenario, "baseline"
        )
        candidate_summary, candidate_records, _ = _load_records(
            formal_root, scenario, "candidate"
        )
        if {row["seed"] for row in baseline_records} != {
            row["seed"] for row in candidate_records
        }:
            raise ValueError(f"{scenario}: baseline/candidate seed sets differ")
        consistency = _geometry_differences(baseline_records, candidate_records)
        if (
            not consistency["seed_sets_match"]
            or consistency["max_abs_difference"] is None
            or consistency["max_abs_difference"] > GEOMETRY_CONSISTENCY_TOLERANCE
        ):
            invalid = True
        validation["scenarios"][scenario] = {
            "goal_radius_m": GOAL_RADIUS_M[scenario],
            "baseline_seed_count": len(baseline_records),
            "candidate_seed_count": len(candidate_records),
            "baseline_checkpoint_sha256": baseline_summary.get("checkpoint_sha256"),
            "candidate_checkpoint_sha256": candidate_summary.get("checkpoint_sha256"),
            "initial_geometry_consistency": consistency,
        }
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
                "aggregate": _aggregate(records),
                "h_state": _h_state(records, GOAL_RADIUS_M[scenario]),
                "h_reference": _h_reference(records),
                "h_start_dynamics": _h_start_dynamics(records),
                "records": records,
            }
            if block["h_state"]["vertical_miss_seed_count"] < MIN_VERTICAL_MISS_SEEDS:
                invalid = True
            scenario_blocks[mode] = block
            blocks.append(block)
        scenarios[scenario] = scenario_blocks

    share_medians = [
        block["h_reference"]["reference_share_median"] for block in blocks
    ]
    overall = _overall_verdict(
        blocks,
        invalid=invalid,
        reference_share_block_medians=share_medians,
    )
    validation["block_count"] = len(blocks)
    validation["invalid"] = invalid
    validation["geometry_consistency_tolerance"] = GEOMETRY_CONSISTENCY_TOLERANCE
    return {
        "schema_version": SCHEMA_VERSION,
        "validation": validation,
        "scenarios": scenarios,
        "overall": overall,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal36 初始垂直几何诊断",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block verdicts",
        "",
        "| scenario | mode | miss n | H-STATE | H-REFERENCE | H-START-DYNAMICS |",
        "|---|---|---:|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            miss = block["h_state"]["vertical_miss_seed_count"]
            verdicts = [
                block[key]["passes_all"]
                for key in ("h_state", "h_reference", "h_start_dynamics")
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
            f"- H_STATE: `{result['overall']['H_STATE']}`",
            f"- H_REFERENCE: `{result['overall']['H_REFERENCE']}`",
            f"- H_START_DYNAMICS: `{result['overall']['H_START_DYNAMICS']}`",
            f"- reference share block median: `{_fmt(result['overall']['reference_share_block_median'])}`",
            f"- decision: `{result['overall']['decision']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## State and reference metrics",
            "",
            "| scenario | mode | world med | world rho init | world rho term | ratio | outside | ref med | share | ref rho init | ref rho term | ref delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            state = block["h_state"]
            reference = block["h_reference"]
            lines.append(
                f"| {scenario} | {mode} | "
                f"{_fmt(state['world_abs_vertical_median_m'])} | "
                f"{_fmt(state['world_initial_rho'])} | "
                f"{_fmt(state['world_terminal_rho'])} | "
                f"{_fmt(state['world_terminal_ratio'])} | "
                f"{_fmt(state['world_outside_radius_fraction'])} | "
                f"{_fmt(reference['reference_abs_vertical_median_m'])} | "
                f"{_fmt(reference['reference_share_median'])} | "
                f"{_fmt(reference['reference_initial_rho'])} | "
                f"{_fmt(reference['reference_terminal_rho'])} | "
                f"{_fmt(reference['reference_high_low_terminal_delta_m'])} |"
            )
    lines.extend(
        [
            "",
            "## Start-dynamics metrics",
            "",
            "| scenario | mode | feature | magnitude med | partial rho | high-low delta | verdict |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            for feature in block["h_start_dynamics"]["features"]:
                lines.append(
                    f"| {scenario} | {mode} | {feature['name']} | "
                    f"{_fmt(feature['median_magnitude'])} | "
                    f"{_fmt(feature['partial_spearman'])} | "
                    f"{_fmt(feature['high_low_terminal_delta_m'])} | "
                    f"{'PASS' if feature['passes_all'] else 'FAIL'} |"
                )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- invalid: `{result['validation']['invalid']}`",
            f"- block count: `{result['validation']['block_count']}`",
            f"- held-out opened: `{result['validation']['heldout_opened']}`",
            f"- geometry consistency tolerance: `{result['validation']['geometry_consistency_tolerance']}`",
        ]
    )
    for scenario in SCENARIOS:
        item = result["validation"]["scenarios"][scenario]
        lines.append(
            f"- {scenario}: baseline={item['baseline_seed_count']} seeds, "
            f"candidate={item['candidate_seed_count']} seeds, "
            f"geometry max diff={_fmt(item['initial_geometry_consistency']['max_abs_difference'], 12)}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "initial_geometry_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (output / "initial_geometry_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "initial_geometry_per_seed.csv").open(
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
    result = analyze(args.formal_root)
    write_outputs(result, args.output)
    print(json.dumps(result["overall"], indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
