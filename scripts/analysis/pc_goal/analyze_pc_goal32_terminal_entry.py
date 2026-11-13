#!/usr/bin/env python3
"""Read-only terminal-entry reachability diagnostic for Goal32.

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

SCHEMA_VERSION = "auv_pc_goal32_terminal_entry_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}
EARLY_WINDOW_S = 2.0
MIN_PLANE_TIME_S = 0.5
MIN_VERTICAL_MISS_SEEDS = 10

ENTRY_CORRELATION_MIN = 0.50
ENTRY_PRESENT_FRACTION_MIN = 0.60
ENTRY_CLOSURE_RADIUS_FRACTION_MAX = 0.50

AUTHORITY_REQUIRED_GT_AUTHORITY_FRACTION_MIN = 0.60
AUTHORITY_ACHIEVED_LT_REQUIRED_FRACTION_MIN = 0.60
AUTHORITY_CORRELATION_MIN = 0.30
AUTHORITY_SATURATION_MIN = 0.10
AUTHORITY_PITCH_ERROR_MIN_RAD = 0.10

GEOMETRY_CORRELATION_MIN = 0.50
GEOMETRY_SIGN_AGREEMENT_MIN = 0.65
GEOMETRY_SIGN_DEADBAND_M = 0.10
GEOMETRY_DELTA_GAP_RADIUS_FRACTION_MAX = 0.50

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "outcome",
    "vertical_miss",
    "entry_time_s",
    "entry_longitudinal_m",
    "entry_horizontal_m",
    "entry_vertical_m",
    "entry_abs_vertical_m",
    "terminal_vertical_m",
    "terminal_abs_vertical_m",
    "vertical_closure_entry_to_terminal_m",
    "time_to_plane_s",
    "plane_crossed",
    "required_closure_mps",
    "authority_p95_mps",
    "required_minus_authority_mps",
    "achieved_closure_mps",
    "achieved_minus_required_mps",
    "early_material_saturation_rate",
    "early_abs_pitch_error_mean_rad",
    "early_stern_abs_mean_rad",
)

EPS = 1e-12


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
_GOAL30 = _load_module(_HERE / "analyze_pc_goal30_traces.py", "_goal32_helpers")


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


def _quaternion_from_euler(
    roll_rad: np.ndarray, pitch_rad: np.ndarray, yaw_rad: np.ndarray
) -> np.ndarray:
    roll, pitch, yaw = np.broadcast_arrays(roll_rad, pitch_rad, yaw_rad)
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)
    quat = np.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def _body_to_world_ned(vector_body: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors into NED using ZYX Euler angles in ``eta``."""

    vectors = np.asarray(vector_body, dtype=np.float64)
    states = np.asarray(eta, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vector_body must have shape (N, 3)")
    if states.shape != (vectors.shape[0], 6):
        raise ValueError("eta must have shape (N, 6)")
    x, y, z, w = np.moveaxis(
        _quaternion_from_euler(states[:, 3], states[:, 4], states[:, 5]), -1, 0
    )
    rotation = np.empty((vectors.shape[0], 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.einsum("nij,nj->ni", rotation, vectors)


def _entry_metrics(
    trace: Any,
    indices: np.ndarray,
    goal_radius_m: float,
) -> dict[str, Any]:
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    eta = np.asarray(trace["eta"][indices], dtype=np.float64)
    p_nav = np.asarray(trace["p_nav"][indices], dtype=np.float64)
    goal_rows = np.asarray(trace["goal_ned_m"][indices], dtype=np.float64)
    tangent_rows = np.asarray(trace["goal_tangent_ned"][indices], dtype=np.float64)
    phase = np.asarray(trace["guidance_terminal_phase"][indices], dtype=np.bool_)
    saturation = np.asarray(trace["material_saturated"][indices], dtype=np.bool_)
    pitch_error = np.asarray(trace["pitch_error_rad"][indices], dtype=np.float64)
    actuator = np.asarray(trace["actual_actuator"][indices], dtype=np.float64)
    ground_body = np.asarray(
        trace["ground_velocity_body_mps"][indices], dtype=np.float64
    )

    if time_s.size < 2:
        raise ValueError("episode has fewer than two trace rows")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("episode time contains non-finite values")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("episode time is not strictly increasing")
    terminal_phase_rows = np.flatnonzero(phase)
    if terminal_phase_rows.size == 0:
        raise ValueError("episode never enters the guidance terminal phase")
    start = int(terminal_phase_rows[0])

    if float(np.max(np.ptp(goal_rows, axis=0))) > 1e-9:
        raise ValueError("goal changes within an episode")
    if float(np.max(np.ptp(tangent_rows, axis=0))) > 1e-9:
        raise ValueError("goal tangent changes within an episode")

    goal = goal_rows[0]
    tangent = tangent_rows[0]
    tangent_rows_for_decomposition = np.broadcast_to(tangent, (eta.shape[0], 3))
    longitudinal, horizontal, vertical, _ = _decompose_many(
        eta[:, :3] - goal, tangent_rows_for_decomposition
    )
    _, nav_horizontal, nav_vertical, _ = _decompose_many(
        p_nav - goal, tangent_rows_for_decomposition
    )
    nav_plane = np.hypot(nav_horizontal, nav_vertical)

    crossing = np.flatnonzero(longitudinal[start:] >= 0.0)
    plane_crossed = bool(crossing.size)
    plane = int(start + crossing[0]) if plane_crossed else int(time_s.size - 1)
    time_to_plane = float(time_s[plane] - time_s[start])
    if time_to_plane < -EPS:
        raise ValueError("plane crossing precedes terminal entry")
    effective_time = max(time_to_plane, MIN_PLANE_TIME_S)

    ground_ned = _body_to_world_ned(ground_body, eta)
    vertical_velocity = ground_ned[:, 2]
    entry_vertical = float(vertical[start])
    if abs(entry_vertical) <= EPS:
        toward_zero_sign = 0.0
    else:
        toward_zero_sign = -math.copysign(1.0, entry_vertical)
    closing_speed = np.maximum(0.0, toward_zero_sign * vertical_velocity[start : plane + 1])
    authority_p95 = float(np.percentile(closing_speed, 95.0))

    required_closure = max(0.0, abs(entry_vertical) - goal_radius_m) / effective_time
    vertical_at_plane = float(vertical[plane])
    achieved_closure = (
        abs(entry_vertical) - abs(vertical_at_plane)
    ) / effective_time
    early = (time_s >= time_s[start]) & (time_s <= time_s[start] + EARLY_WINDOW_S)
    if not np.any(early):
        raise ValueError("entry early window is empty")

    arrays = (
        eta,
        p_nav,
        goal_rows,
        tangent_rows,
        saturation.astype(np.float64),
        pitch_error,
        actuator,
        ground_body,
        ground_ned,
        longitudinal,
        horizontal,
        vertical,
        nav_plane,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("entry diagnostic contains non-finite values")

    terminal_vertical = float(vertical[-1])
    return {
        "entry_time_s": float(time_s[start]),
        "entry_index": start,
        "entry_longitudinal_m": float(longitudinal[start]),
        "entry_horizontal_m": float(horizontal[start]),
        "entry_vertical_m": entry_vertical,
        "entry_abs_vertical_m": float(abs(entry_vertical)),
        "entry_nav_plane_error_m": float(nav_plane[start]),
        "entry_nav_vertical_error_m": float(nav_vertical[start]),
        "terminal_vertical_m": terminal_vertical,
        "terminal_abs_vertical_m": float(abs(terminal_vertical)),
        "terminal_horizontal_m": float(horizontal[-1]),
        "vertical_at_plane_m": vertical_at_plane,
        "vertical_closure_entry_to_terminal_m": float(
            abs(entry_vertical) - abs(terminal_vertical)
        ),
        "time_to_plane_s": time_to_plane,
        "plane_crossed": plane_crossed,
        "required_closure_mps": float(required_closure),
        "authority_p95_mps": authority_p95,
        "required_minus_authority_mps": float(required_closure - authority_p95),
        "achieved_closure_mps": float(achieved_closure),
        "achieved_minus_required_mps": float(achieved_closure - required_closure),
        "early_material_saturation_rate": float(np.mean(saturation[early])),
        "early_abs_pitch_error_mean_rad": float(np.mean(np.abs(pitch_error[early]))),
        "early_stern_abs_mean_rad": float(np.mean(np.abs(actuator[early, 1]))),
        "terminal_phase_sample_count": int(np.count_nonzero(phase)),
        "early_window_sample_count": int(np.count_nonzero(early)),
    }


def _load_records(
    formal_root: Path,
    scenario: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    directory = formal_root / scenario / mode
    summary, records, raw = _GOAL30._trace_records(scenario, mode, directory)
    trace = raw["trace"]
    required_extra = {
        "guidance_terminal_phase",
        "ground_velocity_body_mps",
        "current_ned_mps",
        "nu",
    }
    missing = sorted(required_extra - set(trace.files))
    if missing:
        raise ValueError(f"{directory / 'trace.npz'}: missing fields {missing}")

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

    trace_seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    enriched: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        indices = np.flatnonzero(trace_seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{scenario}/{mode}: no trace rows for seed {seed}")
        extras = _entry_metrics(trace, indices, GOAL_RADIUS_M[scenario])
        row = dict(record)
        row.update(extras)
        row["scenario"] = scenario
        row["mode"] = mode
        row["vertical_miss"] = bool(
            extras["terminal_abs_vertical_m"] > GOAL_RADIUS_M[scenario]
        )
        enriched.append(row)
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
    vertical_miss = [row for row in records if row["vertical_miss"]]
    keys = (
        "entry_abs_vertical_m",
        "terminal_abs_vertical_m",
        "time_to_plane_s",
        "required_closure_mps",
        "authority_p95_mps",
        "required_minus_authority_mps",
        "achieved_closure_mps",
        "early_material_saturation_rate",
        "early_abs_pitch_error_mean_rad",
        "early_stern_abs_mean_rad",
    )
    return {
        "seed_count": len(records),
        "vertical_miss_seed_count": len(vertical_miss),
        "plane_crossed_rate": float(np.mean([bool(row["plane_crossed"]) for row in records])),
        "stats": {key: _stats([row[key] for row in records]) for key in keys},
        "vertical_miss_stats": {
            key: _stats([row[key] for row in vertical_miss]) for key in keys
        },
    }


def _h_entry(records: list[dict[str, Any]], goal_radius_m: float) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    correlation = _spearman(
        [row["entry_abs_vertical_m"] for row in records],
        [row["terminal_abs_vertical_m"] for row in records],
    )
    entry_outside = [row["entry_abs_vertical_m"] > goal_radius_m for row in miss]
    entry_outside_fraction = _fraction(entry_outside)
    closure_median = _median(
        [row["vertical_closure_entry_to_terminal_m"] for row in miss]
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "entry_terminal_spearman_ge_0_50": bool(
            correlation is not None and correlation >= ENTRY_CORRELATION_MIN
        ),
        "entry_outside_radius_fraction_ge_0_60": bool(
            entry_outside_fraction is not None
            and entry_outside_fraction >= ENTRY_PRESENT_FRACTION_MIN
        ),
        "closure_median_le_half_radius": bool(
            closure_median is not None
            and closure_median
            <= ENTRY_CLOSURE_RADIUS_FRACTION_MAX * goal_radius_m
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "entry_terminal_spearman": correlation,
        "entry_outside_radius_seed_count": int(sum(entry_outside)),
        "entry_outside_radius_fraction": entry_outside_fraction,
        "entry_to_terminal_closure_median_m": closure_median,
        "closure_half_radius_m": ENTRY_CLOSURE_RADIUS_FRACTION_MAX * goal_radius_m,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_authority(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    shortfalls = [row["required_minus_authority_mps"] for row in miss]
    required_gt_authority = [value > 0.0 for value in shortfalls]
    achieved_lt_required = [row["achieved_minus_required_mps"] < 0.0 for row in miss]
    required_gt_fraction = _fraction(required_gt_authority)
    achieved_lt_fraction = _fraction(achieved_lt_required)
    shortfall_median = _median(shortfalls)
    correlation = _spearman(
        [row["required_closure_mps"] for row in miss],
        [row["terminal_abs_vertical_m"] for row in miss],
    )
    saturation_median = _median(
        [row["early_material_saturation_rate"] for row in miss]
    )
    pitch_median = _median([row["early_abs_pitch_error_mean_rad"] for row in miss])
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "shortfall_median_gt_0": bool(
            shortfall_median is not None and shortfall_median > 0.0
        ),
        "required_gt_authority_fraction_ge_0_60": bool(
            required_gt_fraction is not None
            and required_gt_fraction >= AUTHORITY_REQUIRED_GT_AUTHORITY_FRACTION_MIN
        ),
        "achieved_lt_required_fraction_ge_0_60": bool(
            achieved_lt_fraction is not None
            and achieved_lt_fraction >= AUTHORITY_ACHIEVED_LT_REQUIRED_FRACTION_MIN
        ),
        "required_terminal_spearman_ge_0_30": bool(
            correlation is not None and correlation >= AUTHORITY_CORRELATION_MIN
        ),
        "early_saturation_or_pitch_signature": bool(
            (saturation_median is not None and saturation_median >= AUTHORITY_SATURATION_MIN)
            or (
                pitch_median is not None
                and pitch_median >= AUTHORITY_PITCH_ERROR_MIN_RAD
            )
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "required_minus_authority_median_mps": shortfall_median,
        "required_gt_authority_fraction": required_gt_fraction,
        "achieved_lt_required_fraction": achieved_lt_fraction,
        "required_terminal_spearman": correlation,
        "early_material_saturation_median": saturation_median,
        "early_abs_pitch_error_median_rad": pitch_median,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _direction(value: float) -> int:
    if value > GEOMETRY_SIGN_DEADBAND_M:
        return 1
    if value < -GEOMETRY_SIGN_DEADBAND_M:
        return -1
    return 0


def _h_geometry(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    goal_radius_m: float,
) -> dict[str, Any]:
    base_by_seed = {int(row["seed"]): row for row in baseline}
    candidate_by_seed = {int(row["seed"]): row for row in candidate}
    if set(base_by_seed) != set(candidate_by_seed):
        raise ValueError("baseline and candidate seed sets differ")
    seeds = sorted(base_by_seed)
    delta_entry_signed: list[float] = []
    delta_terminal_signed: list[float] = []
    delta_entry_abs: list[float] = []
    delta_terminal_abs: list[float] = []
    for seed in seeds:
        base = base_by_seed[seed]
        candidate_row = candidate_by_seed[seed]
        delta_entry_signed.append(
            float(candidate_row["entry_vertical_m"]) - float(base["entry_vertical_m"])
        )
        delta_terminal_signed.append(
            float(candidate_row["terminal_vertical_m"])
            - float(base["terminal_vertical_m"])
        )
        delta_entry_abs.append(
            float(candidate_row["entry_abs_vertical_m"])
            - float(base["entry_abs_vertical_m"])
        )
        delta_terminal_abs.append(
            float(candidate_row["terminal_abs_vertical_m"])
            - float(base["terminal_abs_vertical_m"])
        )

    signed_correlation = _spearman(delta_entry_signed, delta_terminal_signed)
    abs_correlation = _spearman(delta_entry_abs, delta_terminal_abs)
    comparable = [
        (entry, terminal)
        for entry, terminal in zip(delta_entry_abs, delta_terminal_abs, strict=True)
        if _direction(entry) != 0 and _direction(terminal) != 0
    ]
    sign_agreement = (
        _fraction([_direction(entry) == _direction(terminal) for entry, terminal in comparable])
        if comparable
        else None
    )
    delta_gap = [
        float(abs(terminal - entry))
        for entry, terminal in zip(delta_entry_abs, delta_terminal_abs, strict=True)
    ]
    gap_median = _median(delta_gap)
    criteria = {
        "signed_spearman_ge_0_50": bool(
            signed_correlation is not None
            and signed_correlation >= GEOMETRY_CORRELATION_MIN
        ),
        "abs_spearman_ge_0_50": bool(
            abs_correlation is not None and abs_correlation >= GEOMETRY_CORRELATION_MIN
        ),
        "sign_agreement_ge_0_65": bool(
            sign_agreement is not None
            and sign_agreement >= GEOMETRY_SIGN_AGREEMENT_MIN
        ),
        "delta_gap_median_le_half_radius": bool(
            gap_median is not None
            and gap_median
            <= GEOMETRY_DELTA_GAP_RADIUS_FRACTION_MAX * goal_radius_m
        ),
    }
    return {
        "seed_count": len(seeds),
        "delta_entry_signed_m": _stats(delta_entry_signed),
        "delta_terminal_signed_m": _stats(delta_terminal_signed),
        "delta_entry_abs_m": _stats(delta_entry_abs),
        "delta_terminal_abs_m": _stats(delta_terminal_abs),
        "signed_spearman": signed_correlation,
        "abs_spearman": abs_correlation,
        "non_deadband_pair_count": len(comparable),
        "sign_agreement_fraction": sign_agreement,
        "delta_gap_median_m": gap_median,
        "delta_gap_half_radius_m": GEOMETRY_DELTA_GAP_RADIUS_FRACTION_MAX
        * goal_radius_m,
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


def _mode_verdict(scenarios: dict[str, Any], key: str) -> str:
    passes = [
        bool(scenarios[scenario][mode][key]["passes_all"])
        for scenario in SCENARIOS
        for mode in MODES
    ]
    if all(passes):
        return "SUPPORTED"
    if any(passes):
        return "PARTIAL"
    return "FALSIFIED"


def _overall_verdict(scenarios: dict[str, Any]) -> dict[str, Any]:
    h_entry = _mode_verdict(scenarios, "h_entry")
    h_authority = _mode_verdict(scenarios, "h_authority")
    h_geometry = _scenario_verdict(scenarios, "h_geometry")
    supported = {
        name
        for name, verdict in (
            ("H_ENTRY", h_entry),
            ("H_AUTHORITY", h_authority),
            ("H_GEOMETRY", h_geometry),
        )
        if verdict == "SUPPORTED"
    }
    if supported == {"H_ENTRY", "H_AUTHORITY"}:
        decision = "ENTRY_REACHABILITY_SUPPORTED"
        recommendation = (
            "可预注册固定参数、authority-aware 的入口垂直修正；只改这一个因素，"
            "不要同时改 lookahead、reward 或残差接口。"
        )
    elif supported == {"H_ENTRY"}:
        decision = "ENTRY_STATE_SUPPORTED"
        recommendation = (
            "先诊断入口观测/交接，不改终端增益；下一步候选必须解释入口观测误差。"
        )
    elif supported == {"H_AUTHORITY"}:
        decision = "AUTHORITY_LIMITED_SUPPORTED"
        recommendation = (
            "先做速度/舵权限与 terminal 时间预算诊断，不假设入口误差是主因。"
        )
    elif supported == {"H_GEOMETRY"}:
        decision = "GEOMETRY_SHIFT_SUPPORTED"
        recommendation = (
            "先修参考几何；不重跑 Goal31 lookahead，也不把入口变化写成内环滞后。"
        )
    elif len(supported) >= 2:
        decision = "MULTI_FACTOR"
        recommendation = (
            "至少两个机制达到预注册判据；拆成最小单因素候选，不允许合并修改。"
        )
    else:
        decision = "NO_SUPPORTED_MECHANISM"
        recommendation = (
            "入口状态、可达性和 candidate-baseline 几何延续都不满足预注册判据；"
            "停止该 terminal 机制线，不预注册控制器候选。"
        )
    return {
        "H_ENTRY": h_entry,
        "H_AUTHORITY": h_authority,
        "H_GEOMETRY": h_geometry,
        "decision": decision,
        "recommendation": recommendation,
    }


def analyze(formal_root: Path) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "formal_root": str(formal_root),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "heldout_opened": False,
        "scenarios": {},
    }
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
            "baseline_trace": str(formal_root / scenario / "baseline" / "trace.npz"),
            "candidate_trace": str(
                formal_root / scenario / "candidate" / "trace.npz"
            ),
            "baseline_seed_count": len(baseline_records),
            "candidate_seed_count": len(candidate_records),
            "baseline_checkpoint_sha256": baseline_summary.get("checkpoint_sha256"),
            "candidate_checkpoint_sha256": candidate_summary.get("checkpoint_sha256"),
        }
        scenarios[scenario] = {
            "goal_radius_m": radius,
            "baseline": {
                "source": str(formal_root / scenario / "baseline"),
                "aggregate": _aggregate(baseline_records),
                "h_entry": _h_entry(baseline_records, radius),
                "h_authority": _h_authority(baseline_records),
                "records": baseline_records,
            },
            "candidate": {
                "source": str(formal_root / scenario / "candidate"),
                "aggregate": _aggregate(candidate_records),
                "h_entry": _h_entry(candidate_records, radius),
                "h_authority": _h_authority(candidate_records),
                "records": candidate_records,
            },
            "h_geometry": _h_geometry(baseline_records, candidate_records, radius),
        }

    entry_blocks = [
        scenarios[scenario][mode]["h_entry"] for scenario in SCENARIOS for mode in MODES
    ]
    authority_blocks = [
        scenarios[scenario][mode]["h_authority"]
        for scenario in SCENARIOS
        for mode in MODES
    ]
    invalid = any(
        not block["criteria"]["enough_vertical_miss_seeds"]
        for block in entry_blocks + authority_blocks
    )
    verdicts = _overall_verdict(scenarios)
    if invalid:
        verdicts["decision"] = "INVALID_DIAGNOSTIC"
        verdicts["recommendation"] = (
            "至少一个 block 的可用 seed 数不足；不得挑选部分 seed 或回改阈值，"
            "诊断为无效。"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": {
            "entry_correlation_min": ENTRY_CORRELATION_MIN,
            "entry_present_fraction_min": ENTRY_PRESENT_FRACTION_MIN,
            "entry_closure_radius_fraction_max": ENTRY_CLOSURE_RADIUS_FRACTION_MAX,
            "authority_required_gt_authority_fraction_min": (
                AUTHORITY_REQUIRED_GT_AUTHORITY_FRACTION_MIN
            ),
            "authority_achieved_lt_required_fraction_min": (
                AUTHORITY_ACHIEVED_LT_REQUIRED_FRACTION_MIN
            ),
            "authority_correlation_min": AUTHORITY_CORRELATION_MIN,
            "authority_saturation_min": AUTHORITY_SATURATION_MIN,
            "authority_pitch_error_min_rad": AUTHORITY_PITCH_ERROR_MIN_RAD,
            "geometry_correlation_min": GEOMETRY_CORRELATION_MIN,
            "geometry_sign_agreement_min": GEOMETRY_SIGN_AGREEMENT_MIN,
            "geometry_sign_deadband_m": GEOMETRY_SIGN_DEADBAND_M,
            "geometry_delta_gap_radius_fraction_max": (
                GEOMETRY_DELTA_GAP_RADIUS_FRACTION_MAX
            ),
            "min_vertical_miss_seeds": MIN_VERTICAL_MISS_SEEDS,
            "early_window_s": EARLY_WINDOW_S,
            "min_plane_time_s": MIN_PLANE_TIME_S,
        },
        "validation": validation,
        "scenarios": scenarios,
        "verdicts": verdicts,
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
        "# Goal32 terminal-entry reachability diagnostic",
        "",
        "Read-only analysis of the frozen Goal31 formal traces; no rollout, training, ",
        "parameter change, or held-out access.",
        "",
        "## Verdict",
        "",
        "| hypothesis | verdict |",
        "|---|---|",
        f"| H-ENTRY | {verdicts['H_ENTRY']} |",
        f"| H-AUTHORITY | {verdicts['H_AUTHORITY']} |",
        f"| H-GEOMETRY | {verdicts['H_GEOMETRY']} |",
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
            f"{block['candidate_seed_count']} | `{block['candidate_checkpoint_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## H-ENTRY",
            "",
            "| scenario | mode | vertical miss n | entry>r frac | rho | closure med (m) | verdict |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            stats = result["scenarios"][scenario][mode]["h_entry"]
            lines.append(
                f"| {scenario} | {mode} | {stats['vertical_miss_seed_count']} | "
                f"{_fmt(stats['entry_outside_radius_fraction'])} | "
                f"{_fmt(stats['entry_terminal_spearman'])} | "
                f"{_fmt(stats['entry_to_terminal_closure_median_m'])} | "
                f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## H-AUTHORITY",
            "",
            "| scenario | mode | vertical miss n | req-auth med (m/s) | req>auth frac | ach<req frac | rho | sat med | pitch err med (rad) | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            stats = result["scenarios"][scenario][mode]["h_authority"]
            lines.append(
                f"| {scenario} | {mode} | {stats['vertical_miss_seed_count']} | "
                f"{_fmt(stats['required_minus_authority_median_mps'])} | "
                f"{_fmt(stats['required_gt_authority_fraction'])} | "
                f"{_fmt(stats['achieved_lt_required_fraction'])} | "
                f"{_fmt(stats['required_terminal_spearman'])} | "
                f"{_fmt(stats['early_material_saturation_median'])} | "
                f"{_fmt(stats['early_abs_pitch_error_median_rad'])} | "
                f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## H-GEOMETRY candidate minus baseline",
            "",
            "| scenario | signed rho | abs rho | sign agreement | delta gap med (m) | verdict |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        stats = result["scenarios"][scenario]["h_geometry"]
        lines.append(
            f"| {scenario} | {_fmt(stats['signed_spearman'])} | "
            f"{_fmt(stats['abs_spearman'])} | "
            f"{_fmt(stats['sign_agreement_fraction'])} | "
            f"{_fmt(stats['delta_gap_median_m'])} | "
            f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
        )
    lines.extend(["", "## Entry aggregates", ""])
    lines.extend(
        [
            "| scenario | mode | entry abs vert med (m) | terminal abs vert med (m) | time to plane med (s) | req closure med (m/s) | authority p95 med (m/s) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            aggregate = result["scenarios"][scenario][mode]["aggregate"]
            stats = aggregate["stats"]
            lines.append(
                f"| {scenario} | {mode} | "
                f"{_fmt(stats['entry_abs_vertical_m']['median'])} | "
                f"{_fmt(stats['terminal_abs_vertical_m']['median'])} | "
                f"{_fmt(stats['time_to_plane_s']['median'])} | "
                f"{_fmt(stats['required_closure_mps']['median'])} | "
                f"{_fmt(stats['authority_p95_mps']['median'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for scenario in SCENARIOS:
            for mode in MODES:
                for row in result["scenarios"][scenario][mode]["records"]:
                    writer.writerow(row)


def main() -> int:
    args = parse_args()
    result = analyze(args.formal_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "entry_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "entry_diagnostic.md").write_text(
        _markdown(result) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output / "entry_per_seed.csv", result)
    print(json.dumps(result["verdicts"], indent=2, sort_keys=True))
    return 0 if result["verdicts"]["decision"] != "INVALID_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
