#!/usr/bin/env python3
"""Read-only whole-episode vertical/phase diagnostic for Goal34.

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

SCHEMA_VERSION = "auv_pc_goal34_global_vertical_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}
DISTANCE_THRESHOLDS_M = (40.0, 30.0, 20.0)
CROSSING_WINDOW_S = 2.0
MIN_CROSSING_WINDOW_SAMPLES = 10
MIN_VERTICAL_MISS_SEEDS = 10
MIN_PHASE_ACTIVE_SAMPLES = 10
PHASE_ERROR_DEADBAND_M = 0.05
PHASE_VELOCITY_DEADBAND_MPS = 0.02
REVERSAL_ERROR_DEADBAND_M = 0.10
REVERSAL_SEARCH_HORIZON_S = 5.0

GLOBAL_RHO_MIN = 0.50
GLOBAL_RATIO_20_MIN = 0.60
GLOBAL_40_RADIUS_FRACTION_MIN = 0.50

PHASE_SIGN_FRACTION_MAX = 0.55
PHASE_SIGN_CORRELATION_MAX = -0.30
PHASE_REVERSAL_COVERAGE_MIN = 0.50
PHASE_REVERSAL_LATENCY_MIN_S = 1.0
PHASE_HIGH_LOW_SIGN_DELTA_MAX = -0.10

SLOPE_MIN_ABS = 0.03
SLOPE_CORRELATION_MIN = 0.30
SLOPE_HIGH_LOW_VERTICAL_DELTA_MIN_M = 0.20
SLOPE_HORIZONTAL_TANGENT_MIN = 0.10

EPS = 1e-12

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "outcome",
    "vertical_miss",
    "terminal_abs_vertical_m",
    "abs_vertical_40m",
    "abs_vertical_30m",
    "abs_vertical_20m",
    "signed_vertical_40m",
    "signed_vertical_30m",
    "signed_vertical_20m",
    "sign_matched_fraction_40m",
    "sign_matched_fraction_30m",
    "sign_matched_fraction_20m",
    "phase_sign_matched_fraction",
    "phase_reversal_latency_s",
    "phase_reversal_count",
    "phase_active_sample_count",
    "path_vertical_slope",
    "goal_vertical_slope",
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
    _HERE / "analyze_pc_goal32_terminal_entry.py", "_goal34_goal32_helpers"
)


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


def _finite_values(values: list[float | None]) -> np.ndarray:
    return np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )


def _median(values: list[float | None]) -> float | None:
    finite = _finite_values(values)
    return float(np.median(finite)) if finite.size else None


def _fraction(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def _rank_halves(
    records: list[dict[str, Any]],
    key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda row: float(row[key]))
    half = len(ordered) // 2
    if half == 0:
        raise ValueError("cannot split fewer than two records into halves")
    return ordered[:half], ordered[-half:]


def _group_median_delta(
    records: list[dict[str, Any]],
    group_key: str,
    value_key: str,
) -> float | None:
    if len(records) < 2:
        return None
    low, high = _rank_halves(records, group_key)
    low_median = _median([row[value_key] for row in low])
    high_median = _median([row[value_key] for row in high])
    if low_median is None or high_median is None:
        return None
    return high_median - low_median


def _distance_geometry(
    trace: Any,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    goal_rows = np.asarray(trace["goal_ned_m"][indices], dtype=np.float64)
    tangent_rows = np.asarray(trace["goal_tangent_ned"][indices], dtype=np.float64)
    eta = np.asarray(trace["eta"][indices], dtype=np.float64)
    if float(np.max(np.ptp(goal_rows, axis=0))) > 1e-9:
        raise ValueError("goal changes within an episode")
    if float(np.max(np.ptp(tangent_rows, axis=0))) > 1e-9:
        raise ValueError("goal tangent changes within an episode")
    goal = goal_rows[0]
    tangent = tangent_rows[0]
    tangent_rows_for_decomposition = np.broadcast_to(tangent, (eta.shape[0], 3))
    _, _, vertical, _ = _GOAL32._decompose_many(
        eta[:, :3] - goal, tangent_rows_for_decomposition
    )
    distance = np.linalg.norm(eta[:, :3] - goal, axis=1)
    ground_ned = _GOAL32._body_to_world_ned(
        np.asarray(trace["ground_velocity_body_mps"][indices], dtype=np.float64),
        eta,
    )
    return goal, tangent, vertical, distance, ground_ned[:, 2]


def _crossing_metrics(
    trace: Any,
    indices: np.ndarray,
    distance: np.ndarray,
    vertical: np.ndarray,
    vertical_velocity: np.ndarray,
    threshold_m: float,
) -> dict[str, Any]:
    hits = np.flatnonzero(distance <= threshold_m)
    if hits.size == 0:
        raise ValueError(f"episode never reaches {threshold_m:.1f} m to goal")
    position = int(hits[0])
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    crossing_time = float(time_s[position])
    window = np.flatnonzero(np.abs(time_s - crossing_time) <= CROSSING_WINDOW_S)
    if window.size < MIN_CROSSING_WINDOW_SAMPLES:
        raise ValueError(
            f"{threshold_m:.1f} m crossing window has {window.size} samples"
        )

    error = vertical[window]
    velocity = vertical_velocity[window]
    active = (np.abs(error) > PHASE_ERROR_DEADBAND_M) & (
        np.abs(velocity) > PHASE_VELOCITY_DEADBAND_MPS
    )
    closing = np.maximum(0.0, -np.sign(error) * velocity)
    sign_fraction = (
        float(np.mean(closing[active] > 0.0)) if np.any(active) else None
    )
    return {
        "distance_m": float(distance[position]),
        "time_s": crossing_time,
        "signed_vertical_m": float(vertical[position]),
        "abs_vertical_m": float(abs(vertical[position])),
        "window_sample_count": int(window.size),
        "active_sample_count": int(np.count_nonzero(active)),
        "median_abs_vertical_velocity_mps": float(np.median(np.abs(velocity))),
        "median_closing_speed_mps": float(np.median(closing)),
        "sign_matched_fraction": sign_fraction,
    }


def _reversal_latency(
    time_s: np.ndarray,
    vertical_error: np.ndarray,
    vertical_velocity: np.ndarray,
) -> tuple[float | None, int]:
    strict_sign = np.where(
        vertical_error > REVERSAL_ERROR_DEADBAND_M,
        1,
        np.where(vertical_error < -REVERSAL_ERROR_DEADBAND_M, -1, 0),
    )
    latencies: list[float] = []
    last_sign = 0
    for index, current_sign in enumerate(strict_sign):
        if current_sign == 0:
            continue
        if last_sign != 0 and current_sign != last_sign:
            start_time = float(time_s[index])
            for candidate in range(index, time_s.size):
                elapsed = float(time_s[candidate] - start_time)
                if elapsed > REVERSAL_SEARCH_HORIZON_S:
                    break
                if vertical_error[candidate] * current_sign <= PHASE_ERROR_DEADBAND_M:
                    continue
                if -current_sign * vertical_velocity[candidate] > PHASE_VELOCITY_DEADBAND_MPS:
                    latencies.append(elapsed)
                    break
        last_sign = int(current_sign)
    if not latencies:
        return None, 0
    return float(np.median(np.asarray(latencies, dtype=np.float64))), len(latencies)


def _phase_metrics(
    trace: Any,
    indices: np.ndarray,
    vertical: np.ndarray,
    vertical_velocity: np.ndarray,
) -> dict[str, Any]:
    phase = np.asarray(trace["guidance_terminal_phase"][indices], dtype=np.bool_)
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    phase_rows = np.flatnonzero(phase)
    if phase_rows.size == 0:
        raise ValueError("episode never enters guidance terminal phase")
    start = int(phase_rows[0])
    longitudinal = np.asarray(
        _GOAL32._decompose_many(
            np.asarray(trace["eta"][indices, :3], dtype=np.float64)
            - np.asarray(trace["goal_ned_m"][indices], dtype=np.float64),
            np.broadcast_to(
                np.asarray(trace["goal_tangent_ned"][indices][0], dtype=np.float64),
                (indices.size, 3),
            ),
        )[0],
        dtype=np.float64,
    )
    crossing = np.flatnonzero(longitudinal[start:] >= 0.0)
    plane = int(start + crossing[0]) if crossing.size else int(indices.size - 1)
    window = np.arange(start, plane + 1)
    error = vertical[window]
    velocity = vertical_velocity[window]
    active = (np.abs(error) > PHASE_ERROR_DEADBAND_M) & (
        np.abs(velocity) > PHASE_VELOCITY_DEADBAND_MPS
    )
    closing = np.maximum(0.0, -np.sign(error) * velocity)
    sign_fraction = (
        float(np.mean(closing[active] > 0.0)) if np.any(active) else None
    )
    latency, reversal_count = _reversal_latency(time_s[window], error, velocity)
    return {
        "terminal_abs_vertical_m": float(abs(vertical[-1])),
        "phase_sample_count": int(window.size),
        "phase_active_sample_count": int(np.count_nonzero(active)),
        "phase_sign_matched_fraction": sign_fraction,
        "phase_reversal_latency_s": latency,
        "phase_reversal_count": reversal_count,
    }


def _path_slope_metrics(
    trace: Any,
    indices: np.ndarray,
    distance: np.ndarray,
) -> dict[str, Any]:
    hits40 = np.flatnonzero(distance <= 40.0)
    hits20 = np.flatnonzero(distance <= 20.0)
    if hits40.size == 0 or hits20.size == 0:
        raise ValueError("path slope segment is incomplete")
    start = int(hits40[0])
    end = int(hits20[0])
    if end < start:
        raise ValueError("distance-to-goal is not monotonic at segment boundaries")
    segment = np.arange(start, end + 1)
    route = np.asarray(trace["route_tangent_ned"][indices][segment], dtype=np.float64)
    goal = np.asarray(trace["goal_tangent_ned"][indices][segment], dtype=np.float64)
    route_horizontal = np.hypot(route[:, 0], route[:, 1])
    valid = route_horizontal >= SLOPE_HORIZONTAL_TANGENT_MIN
    path_slope = None
    if np.count_nonzero(valid) >= MIN_CROSSING_WINDOW_SAMPLES:
        path_slope = float(
            np.median(route[valid, 2] / route_horizontal[valid])
        )
    goal_horizontal = np.hypot(goal[:, 0], goal[:, 1])
    goal_slope = None
    goal_valid = goal_horizontal >= SLOPE_HORIZONTAL_TANGENT_MIN
    if np.any(goal_valid):
        goal_slope = float(
            np.median(goal[goal_valid, 2] / goal_horizontal[goal_valid])
        )
    return {
        "path_vertical_slope": path_slope,
        "goal_vertical_slope": goal_slope,
        "path_slope_sample_count": int(np.count_nonzero(valid)),
        "path_slope_segment_sample_count": int(segment.size),
    }


def _enrich_record(
    trace: Any,
    indices: np.ndarray,
    record: dict[str, Any],
) -> dict[str, Any]:
    goal, tangent, vertical, distance, vertical_velocity = _distance_geometry(
        trace, indices
    )
    crossing = {
        threshold: _crossing_metrics(
            trace,
            indices,
            distance,
            vertical,
            vertical_velocity,
            threshold,
        )
        for threshold in DISTANCE_THRESHOLDS_M
    }
    phase = _phase_metrics(trace, indices, vertical, vertical_velocity)
    slope = _path_slope_metrics(trace, indices, distance)
    row = dict(record)
    row["goal_ned_m"] = goal.tolist()
    row["goal_tangent_ned"] = tangent.tolist()
    for threshold, metrics in crossing.items():
        suffix = f"{int(threshold)}m"
        row[f"distance_{suffix}_m"] = metrics["distance_m"]
        row[f"crossing_time_{suffix}_s"] = metrics["time_s"]
        row[f"signed_vertical_{suffix}"] = metrics["signed_vertical_m"]
        row[f"abs_vertical_{suffix}"] = metrics["abs_vertical_m"]
        row[f"window_sample_count_{suffix}"] = metrics["window_sample_count"]
        row[f"active_sample_count_{suffix}"] = metrics["active_sample_count"]
        row[f"median_abs_vertical_velocity_{suffix}_mps"] = metrics[
            "median_abs_vertical_velocity_mps"
        ]
        row[f"median_closing_speed_{suffix}_mps"] = metrics[
            "median_closing_speed_mps"
        ]
        row[f"sign_matched_fraction_{suffix}"] = metrics["sign_matched_fraction"]
    row.update(phase)
    row.update(slope)
    row["vertical_miss"] = bool(
        float(phase["terminal_abs_vertical_m"]) > GOAL_RADIUS_M[row["scenario"]]
    )
    return row


def _load_records(
    formal_root: Path,
    scenario: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    directory = formal_root / scenario / mode
    summary, records, raw = _GOAL32._load_records(formal_root, scenario, mode)
    trace = raw
    missing = {"route_tangent_ned"} - set(trace.files)
    if missing:
        raise ValueError(f"{directory / 'trace.npz'}: missing fields {sorted(missing)}")
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    enriched: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{scenario}/{mode}: no trace rows for seed {seed}")
        item = dict(record)
        item["scenario"] = scenario
        item["mode"] = mode
        enriched.append(_enrich_record(trace, indices, item))
    return summary, enriched, trace


def _stats(values: list[float | None]) -> dict[str, Any]:
    finite = _finite_values(values)
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
        "terminal_abs_vertical_m",
        "abs_vertical_40m",
        "abs_vertical_30m",
        "abs_vertical_20m",
        "phase_sign_matched_fraction",
        "phase_reversal_latency_s",
        "path_vertical_slope",
        "goal_vertical_slope",
    )
    return {
        "seed_count": len(records),
        "vertical_miss_seed_count": int(sum(row["vertical_miss"] for row in records)),
        "stats": {key: _stats([row.get(key) for row in records]) for key in keys},
    }


def _h_global(records: list[dict[str, Any]], goal_radius_m: float) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    rho40 = _GOAL32._spearman(
        [row["abs_vertical_40m"] for row in miss],
        [row["terminal_abs_vertical_m"] for row in miss],
    )
    rho20 = _GOAL32._spearman(
        [row["abs_vertical_20m"] for row in miss],
        [row["terminal_abs_vertical_m"] for row in miss],
    )
    median_40 = _median([row["abs_vertical_40m"] for row in miss])
    median_20 = _median([row["abs_vertical_20m"] for row in miss])
    median_terminal = _median([row["terminal_abs_vertical_m"] for row in miss])
    ratio20 = (
        None
        if median_20 is None or median_terminal is None or median_terminal <= EPS
        else median_20 / median_terminal
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "rho40_ge_0_50": bool(rho40 is not None and rho40 >= GLOBAL_RHO_MIN),
        "rho20_ge_0_50": bool(rho20 is not None and rho20 >= GLOBAL_RHO_MIN),
        "ratio20_ge_0_60": bool(ratio20 is not None and ratio20 >= GLOBAL_RATIO_20_MIN),
        "median_40_ge_half_radius": bool(
            median_40 is not None
            and median_40 >= GLOBAL_40_RADIUS_FRACTION_MIN * goal_radius_m
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "rho40": rho40,
        "rho20": rho20,
        "median_abs_vertical_40m": median_40,
        "median_abs_vertical_20m": median_20,
        "median_terminal_abs_vertical_m": median_terminal,
        "ratio20": ratio20,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_phase(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    valid = [
        row
        for row in miss
        if row.get("phase_sign_matched_fraction") is not None
        and row.get("phase_active_sample_count", 0) >= MIN_PHASE_ACTIVE_SAMPLES
    ]
    sign_values = [row["phase_sign_matched_fraction"] for row in valid]
    sign_median = _median(sign_values)
    rho = _GOAL32._spearman(
        sign_values,
        [row["terminal_abs_vertical_m"] for row in valid],
    )
    latencies = [row.get("phase_reversal_latency_s") for row in valid]
    latency_median = _median(latencies)
    coverage = _fraction(
        [row.get("phase_reversal_latency_s") is not None for row in miss]
    )
    sign_delta = _group_median_delta(
        valid, "terminal_abs_vertical_m", "phase_sign_matched_fraction"
    )
    criteria = {
        "enough_valid_phase_seeds": len(valid) >= MIN_VERTICAL_MISS_SEEDS,
        "sign_fraction_median_le_0_55": bool(
            sign_median is not None and sign_median <= PHASE_SIGN_FRACTION_MAX
        ),
        "sign_fraction_terminal_rho_le_minus_0_30": bool(
            rho is not None and rho <= PHASE_SIGN_CORRELATION_MAX
        ),
        "reversal_coverage_and_latency": bool(
            coverage is not None
            and coverage >= PHASE_REVERSAL_COVERAGE_MIN
            and latency_median is not None
            and latency_median >= PHASE_REVERSAL_LATENCY_MIN_S
        ),
        "high_minus_low_sign_fraction_le_minus_0_10": bool(
            sign_delta is not None and sign_delta <= PHASE_HIGH_LOW_SIGN_DELTA_MAX
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "valid_phase_seed_count": len(valid),
        "sign_fraction_median": sign_median,
        "sign_fraction_terminal_rho": rho,
        "reversal_coverage": coverage,
        "reversal_latency_median_s": latency_median,
        "high_minus_low_sign_fraction": sign_delta,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_slope(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    valid = [row for row in miss if row.get("path_vertical_slope") is not None]
    slopes = [abs(float(row["path_vertical_slope"])) for row in valid]
    median_abs_slope = _median(slopes)
    rho40 = _GOAL32._spearman(slopes, [row["abs_vertical_40m"] for row in valid])
    rho_terminal = _GOAL32._spearman(
        slopes, [row["terminal_abs_vertical_m"] for row in valid]
    )
    high_low_delta = _group_median_delta(
        [{**row, "abs_path_vertical_slope": abs(float(row["path_vertical_slope"]))} for row in valid],
        "abs_path_vertical_slope",
        "abs_vertical_20m",
    )
    criteria = {
        "enough_valid_slope_seeds": len(valid) >= MIN_VERTICAL_MISS_SEEDS,
        "median_abs_slope_ge_0_03": bool(
            median_abs_slope is not None and median_abs_slope >= SLOPE_MIN_ABS
        ),
        "slope_vertical40_rho_ge_0_30": bool(
            rho40 is not None and rho40 >= SLOPE_CORRELATION_MIN
        ),
        "slope_terminal_rho_ge_0_30": bool(
            rho_terminal is not None and rho_terminal >= SLOPE_CORRELATION_MIN
        ),
        "high_minus_low_vertical20_ge_0_20m": bool(
            high_low_delta is not None
            and high_low_delta >= SLOPE_HIGH_LOW_VERTICAL_DELTA_MIN_M
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "valid_slope_seed_count": len(valid),
        "median_abs_path_vertical_slope": median_abs_slope,
        "slope_vertical40_rho": rho40,
        "slope_terminal_rho": rho_terminal,
        "high_minus_low_abs_vertical_20m": high_low_delta,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _hypothesis_verdict(blocks: list[dict[str, Any]], key: str) -> str:
    passes = [bool(block[key]["passes_all"]) for block in blocks]
    if all(passes):
        return "SUPPORTED"
    if any(passes):
        return "PARTIAL"
    return "FALSIFIED"


def _overall_verdict(
    blocks: list[dict[str, Any]],
    invalid: bool,
) -> dict[str, Any]:
    h_global = _hypothesis_verdict(blocks, "h_global")
    h_phase = _hypothesis_verdict(blocks, "h_phase")
    h_slope = _hypothesis_verdict(blocks, "h_slope")
    if invalid:
        return {
            "H_GLOBAL": h_global,
            "H_PHASE": h_phase,
            "H_SLOPE": h_slope,
            "decision": "INVALID_DIAGNOSTIC",
            "recommendation": "修复数据、分段或 seed 完整性后重跑，不解释机制。",
        }
    supported = {
        name
        for name, verdict in (
            ("H_GLOBAL", h_global),
            ("H_PHASE", h_phase),
            ("H_SLOPE", h_slope),
        )
        if verdict == "SUPPORTED"
    }
    if supported == {"H_GLOBAL"}:
        decision = "GLOBAL_CARRYOVER_SUPPORTED"
        recommendation = "预注册更早的全程垂直跟踪/交接诊断；不要延长 terminal window。"
    elif supported == {"H_PHASE"}:
        decision = "PHASE_LAG_SUPPORTED"
        recommendation = "预注册固定参数的符号/换向相位诊断；不要重开 authority-magnitude 扫描。"
    elif supported == {"H_SLOPE"}:
        decision = "VERTICAL_SLOPE_SUPPORTED"
        recommendation = "预注册只改参考斜率/前馈的最小单因素诊断；不要扫 reward。"
    elif len(supported) >= 2:
        decision = "MULTI_FACTOR"
        recommendation = "至少两个机制达到预注册判据；拆成最小单因素候选。"
    else:
        decision = "NO_SUPPORTED_MECHANISM"
        recommendation = "停止该全程垂直机制线，不预注册控制器候选。"
    return {
        "H_GLOBAL": h_global,
        "H_PHASE": h_phase,
        "H_SLOPE": h_slope,
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
        radius = GOAL_RADIUS_M[scenario]
        loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]], Any]] = {}
        for mode in MODES:
            loaded[mode] = _load_records(formal_root, scenario, mode)
        baseline_summary, baseline_records, _ = loaded["baseline"]
        candidate_summary, candidate_records, _ = loaded["candidate"]
        if {row["seed"] for row in baseline_records} != {
            row["seed"] for row in candidate_records
        }:
            raise ValueError(f"{scenario}: baseline/candidate seed sets differ")
        validation["scenarios"][scenario] = {
            "goal_radius_m": radius,
            "baseline_seed_count": len(baseline_records),
            "candidate_seed_count": len(candidate_records),
            "baseline_checkpoint_sha256": baseline_summary.get("checkpoint_sha256"),
            "candidate_checkpoint_sha256": candidate_summary.get("checkpoint_sha256"),
            "thresholds_m": list(DISTANCE_THRESHOLDS_M),
        }
        scenario_blocks: dict[str, Any] = {}
        for mode, (summary, records, _trace) in loaded.items():
            block = {
                "scenario": scenario,
                "mode": mode,
                "source": str(formal_root / scenario / mode),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "aggregate": _aggregate(records),
                "h_global": _h_global(records, radius),
                "h_phase": _h_phase(records),
                "h_slope": _h_slope(records),
                "records": records,
            }
            if block["h_global"]["vertical_miss_seed_count"] < MIN_VERTICAL_MISS_SEEDS:
                invalid = True
            scenario_blocks[mode] = block
            blocks.append(block)
        scenarios[scenario] = scenario_blocks
    overall = _overall_verdict(blocks, invalid)
    validation["block_count"] = len(blocks)
    validation["invalid"] = invalid
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
        "# Goal34 全程垂直/相位/斜率诊断",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block verdicts",
        "",
        "| scenario | mode | miss n | H-GLOBAL | H-PHASE | H-SLOPE |",
        "|---|---|---:|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            miss = block["h_global"]["vertical_miss_seed_count"]
            verdicts = [
                block[key]["passes_all"] for key in ("h_global", "h_phase", "h_slope")
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
            f"- H_GLOBAL: `{result['overall']['H_GLOBAL']}`",
            f"- H_PHASE: `{result['overall']['H_PHASE']}`",
            f"- H_SLOPE: `{result['overall']['H_SLOPE']}`",
            f"- decision: `{result['overall']['decision']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## Key metrics",
            "",
            "| scenario | mode | rho40 | rho20 | ratio20 | sign frac med | reversal latency | slope abs med |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            g = block["h_global"]
            p = block["h_phase"]
            s = block["h_slope"]
            lines.append(
                f"| {scenario} | {mode} | {_fmt(g['rho40'])} | {_fmt(g['rho20'])} | "
                f"{_fmt(g['ratio20'])} | {_fmt(p['sign_fraction_median'])} | "
                f"{_fmt(p['reversal_latency_median_s'])} | "
                f"{_fmt(s['median_abs_path_vertical_slope'])} |"
            )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- heldout_opened: `{result['validation']['heldout_opened']}`",
            f"- invalid: `{result['validation']['invalid']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for scenario in SCENARIOS:
            for mode in MODES:
                for record in result["scenarios"][scenario][mode]["records"]:
                    writer.writerow({key: record.get(key) for key in CSV_FIELDS})


def main() -> int:
    args = parse_args()
    result = analyze(args.formal_root)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "global_vertical_diagnostic.json"
    md_path = args.output / "global_vertical_diagnostic.md"
    csv_path = args.output / "global_vertical_per_seed.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(result), encoding="utf-8")
    _write_csv(csv_path, result)
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
