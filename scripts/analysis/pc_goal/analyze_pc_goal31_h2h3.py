#!/usr/bin/env python3
"""Read-only H2/H3 terminal diagnostic for Goal31 formal traces.

The script consumes the four frozen Goal31 formal trace directories and never
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

SCHEMA_VERSION = "auv_pc_goal31_h2h3_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}
TERMINAL_WINDOW_S = 2.0
EPS = 1e-12
ESTIMATOR_SHARE_MIN = 0.50
ESTIMATOR_ABS_RADIUS_FRACTION_MIN = 0.50
CORRELATION_MIN = 0.30
LAG_PITCH_ERROR_MIN_RAD = 0.05
LAG_CLOSURE_RATE_MAX_MPS = 0.10
LAG_SATURATION_MIN = 0.05
LABEL_SATURATION_MIN = 0.10
H3_NO_SATURATION_RADIUS_FRACTION = 0.05
H3_NO_SATURATION_POSITIVE_FRACTION = 0.50
CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "outcome",
    "terminal_eta_capture",
    "terminal_eta_miss",
    "eta_distance_m",
    "terminal_eta_plane_miss_m",
    "horizontal_normal_eta_m",
    "vertical_eta_m",
    "terminal_abs_vertical_eta_m",
    "nav_plane_error_m",
    "nav_error_horizontal_normal_m",
    "nav_error_vertical_m",
    "estimator_plane_share",
    "estimator_dominated",
    "vertical_dominated",
    "horizontal_dominated",
    "actuator_limited",
    "lag_signature",
    "terminal_abs_pitch_error_mean_rad",
    "terminal_abs_pitch_command_mean_rad",
    "terminal_stern_abs_mean_rad",
    "terminal_rudder_abs_mean_rad",
    "terminal_material_saturation_rate",
    "terminal_vertical_closure_rate_mps",
    "terminal_vertical_offset_change_m",
    "terminal_vertical_required_closure_rate_mps",
    "terminal_phase_duration_s",
    "terminal_window_duration_s",
    "terminal_vertical_miss_window_fraction",
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
_GOAL30 = _load_module(_HERE / "analyze_pc_goal30_traces.py", "_goal31_h2h3_helpers")


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


def _mean(values: list[float | None]) -> float | None:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite.size else None


def _fraction(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _spearman(x: list[float | None], y: list[float | None]) -> float | None:
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y, strict=True)
        if a is not None
        and b is not None
        and math.isfinite(float(a))
        and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return None
    left = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.ptp(left) <= EPS or np.ptp(right) <= EPS:
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator <= EPS:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _max_correlation(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _decompose_many(offset: np.ndarray, tangent: np.ndarray) -> tuple[np.ndarray, ...]:
    offset_array = np.asarray(offset, dtype=np.float64)
    tangent_array = np.asarray(tangent, dtype=np.float64)
    if offset_array.ndim != 2 or offset_array.shape[1] != 3:
        raise ValueError("offset must have shape (N, 3)")
    if tangent_array.shape != offset_array.shape:
        raise ValueError("tangent must have shape (N, 3)")
    norms = np.linalg.norm(tangent_array, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("goal tangent contains a zero vector")
    unit_tangent = tangent_array / norms
    longitudinal = np.sum(offset_array * unit_tangent, axis=1)
    normal = offset_array - longitudinal[:, None] * unit_tangent
    horizontal = np.hypot(normal[:, 0], normal[:, 1])
    vertical = normal[:, 2]
    return longitudinal, horizontal, vertical, normal


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size < 2:
        raise ValueError("at least two samples are required for a slope")
    if not (np.all(np.isfinite(x_array)) and np.all(np.isfinite(y_array))):
        raise ValueError("slope inputs must be finite")
    centered_x = x_array - np.mean(x_array)
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= EPS:
        raise ValueError("slope input has no time variation")
    return float(np.dot(centered_x, y_array - np.mean(y_array)) / denominator)


def _terminal_extras(
    trace: Any,
    indices: np.ndarray,
    goal: np.ndarray,
    tangent: np.ndarray,
    goal_radius_m: float,
) -> dict[str, Any]:
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    eta = np.asarray(trace["eta"][indices, :3], dtype=np.float64)
    p_nav = np.asarray(trace["p_nav"][indices], dtype=np.float64)
    goal_rows = np.asarray(trace["goal_ned_m"][indices], dtype=np.float64)
    tangent_rows = np.asarray(trace["goal_tangent_ned"][indices], dtype=np.float64)
    command = np.asarray(trace["command"][indices], dtype=np.float64)
    actuator = np.asarray(trace["actual_actuator"][indices], dtype=np.float64)
    pitch_error = np.asarray(trace["pitch_error_rad"][indices], dtype=np.float64)
    saturation = np.asarray(trace["material_saturated"][indices], dtype=np.bool_)
    phase = np.asarray(trace["guidance_terminal_phase"][indices], dtype=np.bool_)

    _, horizontal_eta, vertical_eta, _ = _decompose_many(eta - goal_rows, tangent_rows)
    _, horizontal_pnav, vertical_pnav, _ = _decompose_many(
        p_nav - goal_rows, tangent_rows
    )
    nav_longitudinal, nav_horizontal, nav_vertical, _ = _decompose_many(
        eta - p_nav, tangent_rows
    )
    nav_plane = np.hypot(nav_horizontal, nav_vertical)

    last_window = time_s >= time_s[-1] - TERMINAL_WINDOW_S
    if not np.any(last_window):
        raise ValueError("terminal window is empty")
    if np.count_nonzero(last_window) < 2:
        raise ValueError("terminal window has fewer than two samples")
    window_time = time_s[last_window]
    window_vertical_abs = np.abs(vertical_eta[last_window])
    window_actuator = actuator[last_window]
    window_command = command[last_window]
    window_pitch_error = pitch_error[last_window]
    window_saturation = saturation[last_window]

    phase_at_end = bool(phase[-1])
    phase_duration_s = 0.0
    phase_start_s = None
    if phase_at_end:
        phase_start = int(np.flatnonzero(phase)[-1])
        while phase_start > 0 and phase[phase_start - 1]:
            phase_start -= 1
        phase_duration_s = float(time_s[-1] - time_s[phase_start])
        phase_start_s = float(time_s[phase_start])

    vertical_end_abs = float(abs(vertical_eta[-1]))
    vertical_start_abs = float(window_vertical_abs[0])
    closure_rate = -_linear_slope(window_time, window_vertical_abs)
    effective_duration_s = max(
        phase_duration_s, float(window_time[-1] - window_time[0]), EPS
    )
    required_closure_rate = max(0.0, vertical_end_abs - goal_radius_m) / effective_duration_s

    arrays = (
        time_s,
        eta,
        p_nav,
        command,
        actuator,
        pitch_error,
        horizontal_eta,
        vertical_eta,
        horizontal_pnav,
        vertical_pnav,
        nav_longitudinal,
        nav_horizontal,
        nav_vertical,
        nav_plane,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("terminal extras contain non-finite values")

    return {
        "terminal_eta_distance_m": float(np.linalg.norm(eta[-1] - goal)),
        "terminal_eta_plane_miss_m": float(
            math.hypot(horizontal_eta[-1], vertical_eta[-1])
        ),
        "terminal_pnav_plane_miss_m": float(
            math.hypot(horizontal_pnav[-1], vertical_pnav[-1])
        ),
        "nav_longitudinal_error_m": float(nav_longitudinal[-1]),
        "nav_error_horizontal_normal_m": float(nav_horizontal[-1]),
        "nav_error_vertical_m": float(nav_vertical[-1]),
        "nav_plane_error_m": float(nav_plane[-1]),
        "terminal_abs_pitch_command_mean_rad": float(
            np.mean(np.abs(window_command[:, 1]))
        ),
        "terminal_pitch_command_slew_abs_mean_rad_s": float(
            np.mean(
                np.abs(
                    np.diff(window_command[:, 1])
                    / np.maximum(np.diff(window_time), EPS)
                )
            )
        ),
        "terminal_abs_pitch_error_mean_rad": float(
            np.mean(np.abs(window_pitch_error))
        ),
        "terminal_abs_pitch_error_max_rad": float(
            np.max(np.abs(window_pitch_error))
        ),
        "terminal_stern_abs_mean_rad": float(np.mean(np.abs(window_actuator[:, 1]))),
        "terminal_rudder_abs_mean_rad": float(np.mean(np.abs(window_actuator[:, 0]))),
        "terminal_material_saturation_rate": float(np.mean(window_saturation)),
        "terminal_vertical_closure_rate_mps": float(closure_rate),
        "terminal_vertical_window_start_abs_m": vertical_start_abs,
        "terminal_vertical_window_end_abs_m": vertical_end_abs,
        "terminal_vertical_offset_change_m": float(vertical_start_abs - vertical_end_abs),
        "terminal_vertical_required_closure_rate_mps": float(required_closure_rate),
        "terminal_phase_at_end": phase_at_end,
        "terminal_phase_start_s": phase_start_s,
        "terminal_phase_duration_s": float(phase_duration_s),
        "terminal_time_since_phase_entry_s": (
            None if phase_start_s is None else float(time_s[-1] - phase_start_s)
        ),
        "terminal_window_duration_s": float(window_time[-1] - window_time[0]),
        "terminal_vertical_miss_window_fraction": float(
            np.mean(window_vertical_abs > goal_radius_m)
        ),
        "terminal_final_time_s": float(time_s[-1]),
        "terminal_sample_count": int(indices.size),
        "terminal_window_sample_count": int(np.count_nonzero(last_window)),
    }


def _load_trace(
    formal_root: Path,
    scenario: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    directory = formal_root / scenario / mode
    summary, records, raw = _GOAL30._trace_records(scenario, mode, directory)
    seeds = tuple(int(seed) for seed in summary.get("seeds", ()))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{directory}: expected dev seeds 20-59, got {seeds}")
    episode_seeds = tuple(int(item["seed"]) for item in summary.get("episodes", ()))
    if episode_seeds != EXPECTED_SEEDS:
        raise ValueError(f"{directory}: episode seeds do not match 20-59")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{directory}: checkpoint SHA256 mismatch")
    if len(records) != len(EXPECTED_SEEDS):
        raise ValueError(f"{directory}: expected 40 terminal records")
    return summary, records, raw["trace"]


def _enrich_record(
    record: dict[str, Any],
    extras: dict[str, Any],
    goal_radius_m: float,
) -> dict[str, Any]:
    enriched = dict(record)
    enriched.update(extras)
    eta_plane = float(extras["terminal_eta_plane_miss_m"])
    nav_plane = float(extras["nav_plane_error_m"])
    share = None if eta_plane <= EPS else nav_plane / eta_plane
    vertical_abs = float(abs(record["vertical_eta_m"]))
    enriched["terminal_abs_vertical_eta_m"] = vertical_abs
    enriched["estimator_plane_share"] = share
    enriched["terminal_eta_miss"] = not bool(record["terminal_eta_capture"])
    enriched["estimator_dominated"] = bool(
        share is not None
        and share >= ESTIMATOR_SHARE_MIN
        and nav_plane >= ESTIMATOR_ABS_RADIUS_FRACTION_MIN * goal_radius_m
    )
    enriched["vertical_dominated"] = bool(
        vertical_abs >= float(record["horizontal_normal_eta_m"])
    )
    enriched["horizontal_dominated"] = bool(
        float(record["horizontal_normal_eta_m"]) > vertical_abs
    )
    enriched["actuator_limited"] = bool(
        float(extras["terminal_material_saturation_rate"]) >= LABEL_SATURATION_MIN
    )
    enriched["lag_signature"] = bool(
        float(extras["terminal_abs_pitch_error_mean_rad"]) >= LAG_PITCH_ERROR_MIN_RAD
        and float(extras["terminal_vertical_closure_rate_mps"])
        <= LAG_CLOSURE_RATE_MAX_MPS
    )
    return enriched


def _load_enriched_records(
    formal_root: Path,
    scenario: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    goal_radius = GOAL_RADIUS_M[scenario]
    summary, records, trace = _load_trace(formal_root, scenario, mode)
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    enriched: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{scenario}/{mode}: no trace rows for seed {seed}")
        last = int(indices[-1])
        goal = np.asarray(trace["goal_ned_m"][last], dtype=np.float64)
        tangent = np.asarray(trace["goal_tangent_ned"][last], dtype=np.float64)
        extras = _terminal_extras(
            trace, indices, goal, tangent, goal_radius
        )
        enriched.append(_enrich_record(record, extras, goal_radius))
    return summary, enriched, trace


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in records]))

    def stats(key: str) -> dict[str, Any]:
        return _GOAL30._term_stats(float(row[key]) for row in records if row[key] is not None)

    return {
        "seed_count": len(records),
        "success_protocol_rate": rate("success_protocol"),
        "success_eta_rate": rate("success_eta"),
        "terminal_eta_capture_rate": rate("terminal_eta_capture"),
        "terminal_eta_miss_rate": rate("terminal_eta_miss"),
        "plane_miss_rate": float(
            np.mean([row["outcome"] == "plane_miss" for row in records])
        ),
        "timeout_rate": float(np.mean([row["outcome"] == "timeout" for row in records])),
        "safety_rate": float(np.mean([row["outcome"] == "safety" for row in records])),
        "terminal_eta_distance_m": stats("terminal_eta_distance_m"),
        "terminal_eta_plane_miss_m": stats("terminal_eta_plane_miss_m"),
        "horizontal_normal_eta_m": stats("horizontal_normal_eta_m"),
        "terminal_abs_vertical_eta_m": stats("terminal_abs_vertical_eta_m"),
        "nav_plane_error_m": stats("nav_plane_error_m"),
        "estimator_plane_share": stats("estimator_plane_share"),
        "terminal_abs_pitch_error_mean_rad": stats(
            "terminal_abs_pitch_error_mean_rad"
        ),
        "terminal_abs_pitch_command_mean_rad": stats(
            "terminal_abs_pitch_command_mean_rad"
        ),
        "terminal_stern_abs_mean_rad": stats("terminal_stern_abs_mean_rad"),
        "terminal_material_saturation_rate": stats(
            "terminal_material_saturation_rate"
        ),
        "terminal_vertical_closure_rate_mps": stats(
            "terminal_vertical_closure_rate_mps"
        ),
        "terminal_vertical_required_closure_rate_mps": stats(
            "terminal_vertical_required_closure_rate_mps"
        ),
        "terminal_phase_duration_s": stats("terminal_phase_duration_s"),
    }


def _label_summary(records: list[dict[str, Any]], radius: float) -> dict[str, Any]:
    miss = [row for row in records if row["terminal_eta_miss"]]
    labels = (
        "estimator_dominated",
        "vertical_dominated",
        "horizontal_dominated",
        "actuator_limited",
        "lag_signature",
    )
    counts = {label: int(sum(bool(row[label]) for row in miss)) for label in labels}
    return {
        "terminal_miss_seed_count": len(miss),
        "counts": counts,
        "rates": {
            label: (None if not miss else counts[label] / len(miss))
            for label in labels
        },
        "vertical_miss_seed_count": int(
            sum(row["terminal_abs_vertical_eta_m"] > radius for row in miss)
        ),
    }


def _h2_estimator(
    records: list[dict[str, Any]], goal_radius_m: float
) -> dict[str, Any]:
    miss = [row for row in records if row["terminal_eta_miss"]]
    shares = [row["estimator_plane_share"] for row in miss]
    nav_plane = [row["nav_plane_error_m"] for row in miss]
    eta_plane = [row["terminal_eta_plane_miss_m"] for row in miss]
    dominated = [bool(row["estimator_dominated"]) for row in miss]
    share_median = _median(shares)
    nav_median = _median(nav_plane)
    dominated_fraction = _fraction(dominated)
    correlation = _spearman(nav_plane, eta_plane)
    criteria = {
        "estimator_share_median_ge_0_50": bool(
            share_median is not None and share_median >= ESTIMATOR_SHARE_MIN
        ),
        "nav_plane_median_ge_half_radius": bool(
            nav_median is not None
            and nav_median >= ESTIMATOR_ABS_RADIUS_FRACTION_MIN * goal_radius_m
        ),
        "dominated_fraction_ge_0_50": bool(
            dominated_fraction is not None and dominated_fraction >= 0.50
        ),
        "spearman_ge_0_30": bool(
            correlation is not None and correlation >= CORRELATION_MIN
        ),
    }
    return {
        "terminal_miss_seed_count": len(miss),
        "estimator_share_median": share_median,
        "nav_plane_error_median_m": nav_median,
        "nav_plane_error_half_radius_m": 0.5 * goal_radius_m,
        "estimator_dominated_seed_count": int(sum(dominated)),
        "estimator_dominated_fraction": dominated_fraction,
        "spearman_nav_plane_vs_eta_plane": correlation,
        "criteria": criteria,
        "passes_all": bool(miss and all(criteria.values())),
    }


def _h2_lag(records: list[dict[str, Any]], goal_radius_m: float) -> dict[str, Any]:
    vertical_miss = [
        row
        for row in records
        if row["terminal_abs_vertical_eta_m"] > goal_radius_m
    ]
    pitch = [row["terminal_abs_pitch_error_mean_rad"] for row in vertical_miss]
    closure = [row["terminal_vertical_closure_rate_mps"] for row in vertical_miss]
    saturation = [row["terminal_material_saturation_rate"] for row in vertical_miss]
    shares = [row["estimator_plane_share"] for row in vertical_miss]
    vertical_abs = [row["terminal_abs_vertical_eta_m"] for row in vertical_miss]
    stern = [row["terminal_stern_abs_mean_rad"] for row in vertical_miss]
    pitch_correlation = _spearman(pitch, vertical_abs)
    stern_correlation = _spearman(stern, vertical_abs)
    max_correlation = _max_correlation([pitch_correlation, stern_correlation])
    pitch_median = _median(pitch)
    closure_median = _median(closure)
    saturation_median = _median(saturation)
    estimator_share_median = _median(shares)
    phase_duration_median = _median(
        [row["terminal_phase_duration_s"] for row in vertical_miss]
    )
    criteria = {
        "pitch_error_median_ge_0_05_rad": bool(
            pitch_median is not None and pitch_median >= LAG_PITCH_ERROR_MIN_RAD
        ),
        "closure_rate_median_le_0_10_mps": bool(
            closure_median is not None and closure_median <= LAG_CLOSURE_RATE_MAX_MPS
        ),
        "saturation_median_ge_0_05": bool(
            saturation_median is not None
            and saturation_median >= LAG_SATURATION_MIN
        ),
        "pitch_or_stern_spearman_ge_0_30": bool(
            max_correlation is not None and max_correlation >= CORRELATION_MIN
        ),
        "estimator_share_median_lt_0_50": bool(
            estimator_share_median is not None
            and estimator_share_median < ESTIMATOR_SHARE_MIN
        ),
    }
    return {
        "vertical_miss_seed_count": len(vertical_miss),
        "pitch_error_median_rad": pitch_median,
        "vertical_closure_rate_median_mps": closure_median,
        "saturation_median": saturation_median,
        "estimator_share_median": estimator_share_median,
        "spearman_pitch_error_vs_vertical_abs": pitch_correlation,
        "spearman_stern_vs_vertical_abs": stern_correlation,
        "max_correlation": max_correlation,
        "terminal_phase_duration_median_s": phase_duration_median,
        "criteria": criteria,
        "passes_all": bool(vertical_miss and all(criteria.values())),
    }


def _delta_stats(values: list[float]) -> dict[str, Any]:
    finite = _finite_values(values)
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "positive_seeds": 0,
            "negative_seeds": 0,
            "bootstrap_95_ci": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "positive_seeds": int(np.sum(finite > 0.0)),
        "negative_seeds": int(np.sum(finite < 0.0)),
        "bootstrap_95_ci": _GOAL30._bootstrap_ci(finite),
    }


def _h3(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    goal_radius_m: float,
) -> dict[str, Any]:
    base_by_seed = {int(row["seed"]): row for row in baseline}
    candidate_by_seed = {int(row["seed"]): row for row in candidate}
    if set(base_by_seed) != set(candidate_by_seed):
        raise ValueError("baseline and candidate seed sets differ")
    seeds = sorted(base_by_seed)
    delta_vertical_abs = [
        float(candidate_by_seed[seed]["terminal_abs_vertical_eta_m"])
        - float(base_by_seed[seed]["terminal_abs_vertical_eta_m"])
        for seed in seeds
    ]
    delta_vertical_signed = [
        float(candidate_by_seed[seed]["vertical_eta_m"])
        - float(base_by_seed[seed]["vertical_eta_m"])
        for seed in seeds
    ]
    delta_pitch_command = [
        float(candidate_by_seed[seed]["terminal_abs_pitch_command_mean_rad"])
        - float(base_by_seed[seed]["terminal_abs_pitch_command_mean_rad"])
        for seed in seeds
    ]
    delta_pitch_error = [
        float(candidate_by_seed[seed]["terminal_abs_pitch_error_mean_rad"])
        - float(base_by_seed[seed]["terminal_abs_pitch_error_mean_rad"])
        for seed in seeds
    ]
    delta_stern = [
        float(candidate_by_seed[seed]["terminal_stern_abs_mean_rad"])
        - float(base_by_seed[seed]["terminal_stern_abs_mean_rad"])
        for seed in seeds
    ]
    delta_saturation = [
        float(candidate_by_seed[seed]["terminal_material_saturation_rate"])
        - float(base_by_seed[seed]["terminal_material_saturation_rate"])
        for seed in seeds
    ]
    stern_correlation = _spearman(delta_stern, delta_vertical_abs)
    saturation_correlation = _spearman(delta_saturation, delta_vertical_abs)
    pitch_correlation = _spearman(delta_pitch_command, delta_vertical_abs)
    max_correlation = _max_correlation([stern_correlation, saturation_correlation])
    stern_stats = _delta_stats(delta_stern)
    saturation_stats = _delta_stats(delta_saturation)
    pitch_command_stats = _delta_stats(delta_pitch_command)
    no_saturation_increase = [
        delta_vertical_abs[index]
        for index, delta in enumerate(delta_saturation)
        if delta <= 0.0
    ]
    no_sat_median = _median(no_saturation_increase)
    no_sat_positive_fraction = (
        _fraction([value > 0.0 for value in no_saturation_increase])
        if no_saturation_increase
        else None
    )
    insufficient_explanation = bool(
        no_sat_median is not None
        and no_sat_median > H3_NO_SATURATION_RADIUS_FRACTION * goal_radius_m
        and no_sat_positive_fraction is not None
        and no_sat_positive_fraction > H3_NO_SATURATION_POSITIVE_FRACTION
    )
    criteria = {
        "stern_delta_median_gt_0": bool(
            stern_stats["median"] is not None and stern_stats["median"] > 0.0
        ),
        "saturation_delta_median_gt_0": bool(
            saturation_stats["median"] is not None
            and saturation_stats["median"] > 0.0
        ),
        "pitch_command_delta_median_ge_0": bool(
            pitch_command_stats["median"] is not None
            and pitch_command_stats["median"] >= 0.0
        ),
        "stern_or_saturation_spearman_ge_0_30": bool(
            max_correlation is not None and max_correlation >= CORRELATION_MIN
        ),
    }
    return {
        "seed_count": len(seeds),
        "delta_vertical_abs_m": _delta_stats(delta_vertical_abs),
        "delta_vertical_signed_m": _delta_stats(delta_vertical_signed),
        "delta_abs_pitch_command_rad": pitch_command_stats,
        "delta_abs_pitch_error_rad": _delta_stats(delta_pitch_error),
        "delta_stern_abs_mean_rad": stern_stats,
        "delta_material_saturation_rate": saturation_stats,
        "spearman_delta_stern_vs_delta_vertical_abs": stern_correlation,
        "spearman_delta_saturation_vs_delta_vertical_abs": saturation_correlation,
        "spearman_delta_pitch_command_vs_delta_vertical_abs": pitch_correlation,
        "max_correlation": max_correlation,
        "no_saturation_increase_seed_count": len(no_saturation_increase),
        "no_saturation_increase_delta_vertical_abs_median_m": no_sat_median,
        "no_saturation_increase_delta_vertical_abs_positive_fraction": (
            no_sat_positive_fraction
        ),
        "insufficient_explanation": insufficient_explanation,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _scenario_verdict(blocks: dict[str, dict[str, Any]], key: str) -> str:
    passes = [bool(blocks[scenario][key]["passes_all"]) for scenario in SCENARIOS]
    if all(passes):
        return "SUPPORTED"
    if any(passes):
        return "PARTIAL"
    return "FALSIFIED"


def _overall_verdict(scenarios: dict[str, Any]) -> dict[str, Any]:
    h2_estimator = _scenario_verdict(scenarios, "h2_estimator")
    h2_lag = _scenario_verdict(scenarios, "h2_lag")
    h3 = _scenario_verdict(scenarios, "h3")
    if h3 == "SUPPORTED" and any(
        scenarios[scenario]["h3"]["insufficient_explanation"]
        for scenario in SCENARIOS
    ):
        h3 = "PARTIAL"
    supported = [
        name
        for name, verdict in (
            ("H2_ESTIMATOR", h2_estimator),
            ("H2_LAG", h2_lag),
            ("H3", h3),
        )
        if verdict == "SUPPORTED"
    ]
    if len(supported) >= 2:
        decision = "MULTI_FACTOR"
        recommendation = (
            "至少两个机制达到预注册判据；先拆成最小单因素候选，不允许把多个修改"
            "合并后直接跑 formal。"
        )
    elif supported == ["H2_ESTIMATOR"]:
        decision = "H2_ESTIMATOR_SUPPORTED"
        recommendation = (
            "先修导航/状态估计链，不先改 terminal reference；Goal32 未获授权。"
        )
    elif supported == ["H2_LAG"]:
        decision = "H2_LAG_SUPPORTED"
        recommendation = (
            "预注册固定参数 goal-plane feedback，并显式验证内环跟踪/权限；不继续扫"
            " path-lookahead。"
        )
    elif supported == ["H3"]:
        decision = "H3_SUPPORTED"
        recommendation = (
            "去掉 path-lookahead；只考虑固定参数 goal-plane vertical/cross-track "
            "feedback，且先做 seed 41 screen。"
        )
    else:
        decision = "NO_SUPPORTED_MECHANISM"
        recommendation = (
            "现有 trace 不支持 H2/H3 的预注册判据；不预注册 Goal32，先补新观测或"
            "新可证伪假设。"
        )
    return {
        "H2_ESTIMATOR": h2_estimator,
        "H2_LAG": h2_lag,
        "H3": h3,
        "decision": decision,
        "recommendation": recommendation,
    }


def analyze(formal_root: Path) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "formal_root": str(formal_root),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        radius = GOAL_RADIUS_M[scenario]
        loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]], Any]] = {}
        for mode in MODES:
            loaded[mode] = _load_enriched_records(formal_root, scenario, mode)
        baseline_summary, baseline_records, _ = loaded["baseline"]
        candidate_summary, candidate_records, _ = loaded["candidate"]
        if {row["seed"] for row in baseline_records} != {
            row["seed"] for row in candidate_records
        }:
            raise ValueError(f"{scenario}: baseline/candidate seed sets differ")
        validation["scenarios"][scenario] = {
            "goal_radius_m": radius,
            "baseline_trace": str(formal_root / scenario / "baseline" / "trace.npz"),
            "candidate_trace": str(
                formal_root / scenario / "candidate" / "trace.npz"
            ),
            "baseline_seed_count": len(baseline_records),
            "candidate_seed_count": len(candidate_records),
            "checkpoint_sha256": candidate_summary.get("checkpoint_sha256"),
        }
        scenarios[scenario] = {
            "goal_radius_m": radius,
            "baseline": {
                "source": str(formal_root / scenario / "baseline"),
                "aggregate": _aggregate(baseline_records),
                "failure_labels": _label_summary(baseline_records, radius),
                "records": baseline_records,
            },
            "candidate": {
                "source": str(formal_root / scenario / "candidate"),
                "aggregate": _aggregate(candidate_records),
                "failure_labels": _label_summary(candidate_records, radius),
                "records": candidate_records,
            },
            "h2_estimator": _h2_estimator(candidate_records, radius),
            "h2_lag": _h2_lag(candidate_records, radius),
            "h3": _h3(baseline_records, candidate_records, radius),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "terminal_window_s": TERMINAL_WINDOW_S,
        "thresholds": {
            "estimator_share_min": ESTIMATOR_SHARE_MIN,
            "estimator_abs_radius_fraction_min": ESTIMATOR_ABS_RADIUS_FRACTION_MIN,
            "correlation_min": CORRELATION_MIN,
            "lag_pitch_error_min_rad": LAG_PITCH_ERROR_MIN_RAD,
            "lag_closure_rate_max_mps": LAG_CLOSURE_RATE_MAX_MPS,
            "lag_saturation_min": LAG_SATURATION_MIN,
            "label_saturation_min": LABEL_SATURATION_MIN,
            "h3_no_saturation_radius_fraction": H3_NO_SATURATION_RADIUS_FRACTION,
            "h3_no_saturation_positive_fraction": H3_NO_SATURATION_POSITIVE_FRACTION,
        },
        "validation": validation,
        "scenarios": scenarios,
        "verdicts": _overall_verdict(scenarios),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(result: dict[str, Any]) -> str:
    verdicts = result["verdicts"]
    lines = [
        "# Goal31 H2/H3 terminal diagnostic",
        "",
        "Read-only analysis of the frozen Goal31 formal traces; no rollout, training, ",
        "parameter change, or held-out access.",
        "",
        "## Verdict",
        "",
        "| hypothesis | verdict |",
        "|---|---|",
        f"| H2 estimator | {verdicts['H2_ESTIMATOR']} |",
        f"| H2 inner-loop lag | {verdicts['H2_LAG']} |",
        f"| H3 vertical demand / saturation | {verdicts['H3']} |",
        f"| overall decision | **{verdicts['decision']}** |",
        "",
        f"Recommendation: {verdicts['recommendation']}",
        "",
        "## Validation",
        "",
        "| scenario | seeds baseline/candidate | checkpoint |",
        "|---|---:|---|",
    ]
    for scenario in SCENARIOS:
        block = result["validation"]["scenarios"][scenario]
        lines.append(
            f"| {scenario} | {block['baseline_seed_count']}/"
            f"{block['candidate_seed_count']} | `{block['checkpoint_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## H2 estimator",
            "",
            "| scenario | miss n | share med | nav plane med (m) | dominated | rho | verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        stats = result["scenarios"][scenario]["h2_estimator"]
        lines.append(
            f"| {scenario} | {stats['terminal_miss_seed_count']} | "
            f"{_fmt(stats['estimator_share_median'])} | "
            f"{_fmt(stats['nav_plane_error_median_m'])} | "
            f"{_fmt(stats['estimator_dominated_fraction'])} | "
            f"{_fmt(stats['spearman_nav_plane_vs_eta_plane'])} | "
            f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## H2 inner-loop lag proxy",
            "",
            "| scenario | vertical miss n | pitch err med (rad) | closure med (m/s) | sat. med | max rho | estimator share med | phase dur. med (s) | verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        stats = result["scenarios"][scenario]["h2_lag"]
        lines.append(
            f"| {scenario} | {stats['vertical_miss_seed_count']} | "
            f"{_fmt(stats['pitch_error_median_rad'])} | "
            f"{_fmt(stats['vertical_closure_rate_median_mps'])} | "
            f"{_fmt(stats['saturation_median'])} | "
            f"{_fmt(stats['max_correlation'])} | "
            f"{_fmt(stats['estimator_share_median'])} | "
            f"{_fmt(stats['terminal_phase_duration_median_s'])} | "
            f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## H3 candidate minus baseline",
            "",
            "| scenario | d vertical abs med (m) | d stern med (rad) | d sat. med | d |pitch cmd| med (rad) | max rho | no-sat d vertical med (m) | no-sat positive frac | verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        stats = result["scenarios"][scenario]["h3"]
        lines.append(
            f"| {scenario} | "
            f"{_fmt(stats['delta_vertical_abs_m']['median'])} | "
            f"{_fmt(stats['delta_stern_abs_mean_rad']['median'])} | "
            f"{_fmt(stats['delta_material_saturation_rate']['median'])} | "
            f"{_fmt(stats['delta_abs_pitch_command_rad']['median'])} | "
            f"{_fmt(stats['max_correlation'])} | "
            f"{_fmt(stats['no_saturation_increase_delta_vertical_abs_median_m'])} | "
            f"{_fmt(stats['no_saturation_increase_delta_vertical_abs_positive_fraction'])} | "
            f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Failure labels",
            "",
            "| scenario | mode | eta miss n | vertical | horizontal | estimator | actuator | lag |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            labels = result["scenarios"][scenario][mode]["failure_labels"]
            counts = labels["counts"]
            lines.append(
                f"| {scenario} | {mode} | {labels['terminal_miss_seed_count']} | "
                f"{counts['vertical_dominated']} | "
                f"{counts['horizontal_dominated']} | "
                f"{counts['estimator_dominated']} | "
                f"{counts['actuator_limited']} | "
                f"{counts['lag_signature']} |"
            )
    lines.extend(
        [
            "",
            "## Candidate terminal aggregates",
            "",
            "| scenario | vertical abs med (m) | nav plane med (m) | estimator share med | pitch err med (rad) | stern med (rad) | saturation med | closure med (m/s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        aggregate = result["scenarios"][scenario]["candidate"]["aggregate"]
        lines.append(
            f"| {scenario} | "
            f"{_fmt(aggregate['terminal_abs_vertical_eta_m']['median'])} | "
            f"{_fmt(aggregate['nav_plane_error_m']['median'])} | "
            f"{_fmt(aggregate['estimator_plane_share']['median'])} | "
            f"{_fmt(aggregate['terminal_abs_pitch_error_mean_rad']['median'])} | "
            f"{_fmt(aggregate['terminal_stern_abs_mean_rad']['median'])} | "
            f"{_fmt(aggregate['terminal_material_saturation_rate']['median'])} | "
            f"{_fmt(aggregate['terminal_vertical_closure_rate_mps']['median'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest candidate vertical misses",
            "",
            "| scenario | seed | outcome | vertical abs (m) | horizontal (m) | estimator share | labels |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        records = sorted(
            result["scenarios"][scenario]["candidate"]["records"],
            key=lambda row: float(row["terminal_abs_vertical_eta_m"]),
            reverse=True,
        )[:8]
        for row in records:
            labels = [
                name
                for name in (
                    "estimator_dominated",
                    "vertical_dominated",
                    "horizontal_dominated",
                    "actuator_limited",
                    "lag_signature",
                )
                if row[name]
            ]
            lines.append(
                f"| {scenario} | {row['seed']} | {row['outcome']} | "
                f"{_fmt(row['terminal_abs_vertical_eta_m'])} | "
                f"{_fmt(row['horizontal_normal_eta_m'])} | "
                f"{_fmt(row['estimator_plane_share'])} | {', '.join(labels) or '-'} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- PASS/FAIL above are diagnostic criteria, not a new formal Gate.",
            "- Correlation is not causation; one scenario alone is not a general result.",
            "- `pitch_error` is command minus true pitch and is only a proxy for tracking lag.",
            "- `eta` and `p_nav` come from the same simulation step, not independent held-out evidence.",
            "- Held-out seeds `100-149` remain closed.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for scenario in SCENARIOS:
            for mode in MODES:
                for row in result["scenarios"][scenario][mode]["records"]:
                    writer.writerow({key: row.get(key) for key in CSV_FIELDS})


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args.formal_root)
    except Exception as exc:
        print(f"INVALID_DIAGNOSTIC: {exc}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "h2h3_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "h2h3_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    _write_csv(args.output / "h2h3_per_seed.csv", result)
    verdicts = result["verdicts"]
    print(
        "H2_ESTIMATOR={H2_ESTIMATOR} H2_LAG={H2_LAG} H3={H3} decision={decision}".format(
            **verdicts
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
