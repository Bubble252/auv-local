#!/usr/bin/env python3
"""Read-only terminal closure, action and reward attribution for Goal117."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal117_terminal_closure_attribution.v1"
DEFAULT_ROOT = Path("artifacts/lowlevel/goal117_terminal_closure_attribution_20261111")
RUN_DIR = DEFAULT_ROOT / "R100_RTRACE"
PARITY_PATH = Path("artifacts/lowlevel/goal116_reference_filter_20261111/R100/trace.npz")
EXPECTED_CHECKPOINT_SHA = "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616"
EXPECTED_SCALE = 2.0
DEV_SEEDS = tuple(range(20, 60))
EXPECTED_FAILURE_SEEDS = (25, 26, 27, 29, 31, 33, 36, 49, 50, 51, 54)
ACTION_FULL_SCALE = np.asarray([0.3490658503988659, 0.3490658503988659, 1525.0])
CHANNEL_NAMES = ("rudder_heading", "stern_pitch", "rpm_speed")
CHANNEL_ERRORS = ("heading_error_rad", "pitch_error_rad", "speed_error_mps")
PRIMARY_SHELL_M = 2.0
LAST_WINDOW_S = 2.0
LAST_SHELL_M = 0.5
STALL_CLOSURE_MPS = 0.30
BASE_SATURATION_THRESHOLD = 0.80
RESIDUAL_SMALL_RATIO = 0.10
ALIGNMENT_GOOD = 0.50
VERTICAL_MISS_M = 1.0
PITCH_ERROR_RAD = 0.45
HEADING_ERROR_RAD = 0.35
GEOMETRY_WINDOW_STEPS = 5
TERMINAL_REWARD_FIELDS = (
    "reward_terminal",
    "reward_terminal_progress",
    "reward_goal_plane_miss",
)
REWARD_EXCLUDE = {"reward_total", "reward_ppo_scaled"}
PARITY_FIELDS = (
    "time_s",
    "eta",
    "nu",
    "command",
    "actuator_total_physical",
    "actuator_base_physical",
    "frozen_residual_executed_action",
    "goal_distance_eta_m",
    "terminal_kind",
    "episode_seed",
)


class AuditError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "phase0_summary.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_ROOT / "phase0_summary.md",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AuditError(f"missing trace: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _seed_rows(trace: dict[str, np.ndarray], seed: int) -> np.ndarray:
    rows = np.flatnonzero(np.asarray(trace["episode_seed"], dtype=np.int64) == int(seed))
    if rows.size == 0:
        raise AuditError(f"trace has no rows for seed {seed}")
    return rows


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _median(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else math.nan


def _quantile(values: np.ndarray, q: float) -> float:
    finite = _finite(values)
    return float(np.quantile(finite, q)) if finite.size else math.nan


def _cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    left = np.asarray([v for v in a if math.isfinite(float(v))], dtype=np.float64)
    right = np.asarray([v for v in b if math.isfinite(float(v))], dtype=np.float64)
    if left.size == 0 or right.size == 0:
        return math.nan
    greater = sum(float(np.sum(left_value > right)) for left_value in left)
    less = sum(float(np.sum(left_value < right)) for left_value in left)
    return (greater - less) / float(left.size * right.size)


def _first_entry(distance: np.ndarray, threshold_m: float) -> int | None:
    hits = np.flatnonzero(np.asarray(distance, dtype=np.float64) <= float(threshold_m))
    return int(hits[0]) if hits.size else None


def _closest_index(distance: np.ndarray) -> int:
    return int(np.argmin(np.asarray(distance, dtype=np.float64)))


def _closure(distance: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    dt = np.diff(np.asarray(time_s, dtype=np.float64))
    closure = np.full(distance.size, np.nan, dtype=np.float64)
    valid = dt > 0.0
    values = np.full(dt.size, np.nan, dtype=np.float64)
    values[valid] = -np.diff(distance)[valid] / dt[valid]
    closure[:-1] = values
    return closure


def _window_mask(time_s: np.ndarray, *, start_s: float, end_s: float) -> np.ndarray:
    return (time_s >= start_s) & (time_s <= end_s)


def _rolling_median(values: np.ndarray, index: int, window: int) -> float:
    start = max(0, index - window + 1)
    return _median(values[start : index + 1])


def _slope(residual: np.ndarray, delta_error: np.ndarray) -> float:
    mask = np.isfinite(residual) & np.isfinite(delta_error)
    x = residual[mask]
    y = delta_error[mask]
    if x.size < 3:
        return math.nan
    x_centered = x - float(np.mean(x))
    variance = float(np.sum(x_centered * x_centered))
    if variance <= 1e-12:
        return math.nan
    return float(np.sum(x_centered * (y - float(np.mean(y)))) / variance)


def _alignment(residual: np.ndarray, error: np.ndarray) -> float:
    mask = np.isfinite(residual) & np.isfinite(error) & (np.abs(error) > 1e-9)
    if not np.any(mask):
        return math.nan
    return float(np.mean(-np.sign(error[mask]) * residual[mask]))


def _phase_stats(
    *,
    residual_scaled: np.ndarray,
    base_scaled: np.ndarray,
    errors: np.ndarray,
    closure: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "frames": 0,
            "closure_mean_mps": math.nan,
            "closure_min_mps": math.nan,
            "base_saturation_fraction": math.nan,
            "residual_magnitude_ratio": math.nan,
            "alignment": [math.nan] * 3,
            "slope": [math.nan] * 3,
        }
    base_norm = np.linalg.norm(base_scaled[mask], axis=1)
    residual_norm = np.linalg.norm(residual_scaled[mask], axis=1)
    ratios = residual_norm / np.maximum(base_norm, 1e-9)
    saturation = np.any(np.abs(base_scaled[mask]) >= 0.98, axis=1)
    alignments: list[float] = []
    slopes: list[float] = []
    for channel in range(3):
        error = errors[mask, channel]
        delta = np.full(error.size, np.nan, dtype=np.float64)
        delta[:-1] = error[1:] - error[:-1]
        alignments.append(_alignment(residual_scaled[mask, channel], error))
        slopes.append(_slope(residual_scaled[mask, channel], delta))
    closure_values = _finite(closure[mask])
    return {
        "frames": int(np.sum(mask)),
        "closure_mean_mps": float(np.mean(closure_values)) if closure_values.size else math.nan,
        "closure_min_mps": float(np.min(closure_values)) if closure_values.size else math.nan,
        "base_saturation_fraction": float(np.mean(saturation)),
        "residual_magnitude_ratio": float(np.median(ratios)) if ratios.size else math.nan,
        "alignment": alignments,
        "slope": slopes,
    }


def _reward_summary(
    trace: dict[str, np.ndarray], rows: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    reward_fields = sorted(
        name
        for name in trace
        if name.startswith("reward_") and name not in REWARD_EXCLUDE
    )
    if not reward_fields:
        raise AuditError("trace has no reward_* fields; was --trace-reward-breakdown used?")
    sums: dict[str, float] = {}
    for name in reward_fields:
        values = np.asarray(trace[name], dtype=np.float64)[rows][mask]
        sums[name] = float(np.sum(values[np.isfinite(values)]))
    terminal_terms = [name for name in TERMINAL_REWARD_FIELDS if name in sums]
    if not terminal_terms:
        raise AuditError("no terminal reward fields found")
    terminal = float(sum(sums[name] for name in terminal_terms))
    dense = float(
        sum(abs(sums[name]) for name in reward_fields if name not in terminal_terms)
    )
    absolute_total = float(sum(abs(sums[name]) for name in reward_fields))
    return {
        "sums": sums,
        "terminal_sum": terminal,
        "dense_abs_sum": dense,
        "terminal_share": terminal / absolute_total if absolute_total > 0.0 else math.nan,
        "dense_terminal_ratio": dense / (abs(terminal) + 1e-9),
    }


def _episode_record(
    trace: dict[str, np.ndarray], seed: int, episode: dict[str, Any]
) -> dict[str, Any]:
    rows = _seed_rows(trace, seed)
    time_s = np.asarray(trace["time_s"], dtype=np.float64)[rows]
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)[rows]
    eta = np.asarray(trace["eta"], dtype=np.float64)[rows]
    goal_ned = np.asarray(trace["goal_ned_m"], dtype=np.float64)[rows]
    residual_physical = np.asarray(trace["actuator_residual_applied"], dtype=np.float64)[rows]
    base_physical = np.asarray(trace["actuator_base_physical"], dtype=np.float64)[rows]
    residual_scaled = residual_physical / ACTION_FULL_SCALE
    base_scaled = base_physical / ACTION_FULL_SCALE
    errors = np.stack(
        [
            np.asarray(trace[CHANNEL_ERRORS[0]], dtype=np.float64)[rows],
            np.asarray(trace[CHANNEL_ERRORS[1]], dtype=np.float64)[rows],
            np.asarray(trace[CHANNEL_ERRORS[2]], dtype=np.float64)[rows],
        ],
        axis=1,
    )
    closure = _closure(distance, time_s)
    terminal_kind = int(np.asarray(trace["terminal_kind"], dtype=np.int64)[rows][-1])
    closest = _closest_index(distance)
    entry2 = _first_entry(distance, PRIMARY_SHELL_M)
    last_time = float(time_s[-1])
    masks = {
        "preentry": np.ones(time_s.size, dtype=bool)
        if entry2 is None
        else np.arange(time_s.size) <= entry2,
        "handoff": np.zeros(time_s.size, dtype=bool)
        if entry2 is None
        else (np.arange(time_s.size) >= entry2) & (np.arange(time_s.size) <= closest),
        "postclosest": np.arange(time_s.size) >= closest,
        "last2s": time_s >= last_time - LAST_WINDOW_S,
        "last05m": distance <= LAST_SHELL_M,
    }
    phases = {
        name: _phase_stats(
            residual_scaled=residual_scaled,
            base_scaled=base_scaled,
            errors=errors,
            closure=closure,
            mask=mask,
        )
        for name, mask in masks.items()
    }
    rewards = {
        name: _reward_summary(trace, rows, mask)
        for name, mask in {
            "postclosest": masks["postclosest"],
            "last2s": masks["last2s"],
            "episode": np.ones(time_s.size, dtype=bool),
        }.items()
    }
    vertical = np.abs(eta[:, 2] - goal_ned[:, 2])
    heading = np.abs(np.asarray(trace["heading_error_rad"], dtype=np.float64)[rows])
    pitch = np.abs(np.asarray(trace["pitch_error_rad"], dtype=np.float64)[rows])
    geometry = {
        "closest_step": closest,
        "closest_distance_m": float(distance[closest]),
        "closest_time_s": float(time_s[closest]),
        "closest_vertical_median5_m": _rolling_median(vertical, closest, GEOMETRY_WINDOW_STEPS),
        "closest_heading_median5_rad": _rolling_median(heading, closest, GEOMETRY_WINDOW_STEPS),
        "closest_pitch_median5_rad": _rolling_median(pitch, closest, GEOMETRY_WINDOW_STEPS),
        "final_distance_m": float(distance[-1]),
        "final_vertical_miss_m": float(vertical[-1]),
        "final_abs_heading_error_rad": float(heading[-1]),
        "final_abs_pitch_error_rad": float(pitch[-1]),
        "post_closest_recede_m": float(distance[-1] - distance[closest]),
        "closure_last2s_mps": _median(closure[masks["last2s"]]),
        "closure_last_mps": float(closure[-2]) if closure.size >= 2 and math.isfinite(closure[-2]) else math.nan,
    }
    geometry["closest_geometry_bad"] = bool(
        geometry["closest_vertical_median5_m"] >= VERTICAL_MISS_M
        or geometry["closest_heading_median5_rad"] >= HEADING_ERROR_RAD
        or geometry["closest_pitch_median5_rad"] >= PITCH_ERROR_RAD
    )
    if entry2 is None:
        geometry.update(
            {
                "entry2_step": None,
                "entry2_time_s": None,
                "entry2_distance_m": None,
                "entry2_vertical_median5_m": None,
                "entry2_heading_median5_rad": None,
                "entry2_pitch_median5_rad": None,
                "entry2_geometry_bad": None,
                "entry2_to_closest_s": None,
            }
        )
    else:
        geometry.update(
            {
                "entry2_step": int(entry2),
                "entry2_time_s": float(time_s[entry2]),
                "entry2_distance_m": float(distance[entry2]),
                "entry2_vertical_median5_m": _rolling_median(
                    vertical, entry2, GEOMETRY_WINDOW_STEPS
                ),
                "entry2_heading_median5_rad": _rolling_median(
                    heading, entry2, GEOMETRY_WINDOW_STEPS
                ),
                "entry2_pitch_median5_rad": _rolling_median(
                    pitch, entry2, GEOMETRY_WINDOW_STEPS
                ),
                "entry2_to_closest_s": float(time_s[closest] - time_s[entry2]),
            }
        )
        geometry["entry2_geometry_bad"] = bool(
            geometry["entry2_vertical_median5_m"] >= VERTICAL_MISS_M
            or geometry["entry2_heading_median5_rad"] >= HEADING_ERROR_RAD
            or geometry["entry2_pitch_median5_rad"] >= PITCH_ERROR_RAD
        )
    terminal_names = {
        0: "none",
        1: "goal",
        2: "time_limit",
        3: "out_of_bounds",
        4: "nan_or_inf",
        5: "invalid_state",
        6: "depth_limit",
        7: "safety_roll_angle",
        8: "safety_roll_rate",
        9: "goal_plane_miss",
    }
    return {
        "seed": int(seed),
        "success": bool(episode.get("success")),
        "timeout": bool(episode.get("timeout")),
        "safety": bool(episode.get("safety", False)),
        "terminal_kind": terminal_kind,
        "terminal_name": terminal_names.get(terminal_kind, f"kind_{terminal_kind}"),
        "frames": int(time_s.size),
        "episode_time_s": float(time_s[-1] - time_s[0]) if time_s.size > 1 else 0.0,
        "phases": phases,
        "rewards": rewards,
        "geometry": geometry,
    }


def _classify(record: dict[str, Any]) -> str:
    last = record["phases"]["last2s"]
    geometry = record["geometry"]
    alignment = last["alignment"]
    if geometry["closest_distance_m"] >= PRIMARY_SHELL_M:
        return "APPROACH_DEFICIT"
    direction_wrong = bool(
        (math.isfinite(alignment[0]) and alignment[0] < 0.0)
        or (math.isfinite(alignment[1]) and alignment[1] < 0.0)
        or (math.isfinite(last["slope"][0]) and last["slope"][0] >= 0.0)
        or (math.isfinite(last["slope"][1]) and last["slope"][1] <= 0.0)
    )
    if (
        math.isfinite(last["base_saturation_fraction"])
        and last["base_saturation_fraction"] >= BASE_SATURATION_THRESHOLD
        and not direction_wrong
    ):
        return "BASE_AUTHORITY_LIMITED"
    if (
        math.isfinite(last["base_saturation_fraction"])
        and last["base_saturation_fraction"] < BASE_SATURATION_THRESHOLD
        and direction_wrong
    ):
        return "RESIDUAL_DIRECTION_WRONG"
    direction_good = bool(
        (not math.isfinite(alignment[0]) or alignment[0] >= ALIGNMENT_GOOD)
        and (not math.isfinite(alignment[1]) or alignment[1] >= ALIGNMENT_GOOD)
    )
    stall = math.isfinite(geometry["closure_last2s_mps"]) and geometry["closure_last2s_mps"] < STALL_CLOSURE_MPS
    if (
        geometry["closest_geometry_bad"]
        and direction_good
        and stall
        and math.isfinite(last["base_saturation_fraction"])
        and last["base_saturation_fraction"] < BASE_SATURATION_THRESHOLD
    ):
        return "TERMINAL_GEOMETRY_LIMITED"
    if (
        direction_good
        and stall
        and math.isfinite(last["residual_magnitude_ratio"])
        and last["residual_magnitude_ratio"] < RESIDUAL_SMALL_RATIO
    ):
        return "RESIDUAL_TOO_SMALL"
    return "UNCLASSIFIED"


def _group_summary(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    if key == "success":
        selected = [record for record in records if record["success"]]
    elif key == "failure":
        selected = [record for record in records if not record["success"]]
    elif key == "primary_failure":
        selected = [
            record
            for record in records
            if not record["success"] and int(record["seed"]) in EXPECTED_FAILURE_SEEDS
        ]
    else:
        raise ValueError(f"unknown group key: {key}")
    if not selected:
        return {}
    fields = {
        "closest_distance_m": [r["geometry"]["closest_distance_m"] for r in selected],
        "final_distance_m": [r["geometry"]["final_distance_m"] for r in selected],
        "post_closest_recede_m": [r["geometry"]["post_closest_recede_m"] for r in selected],
        "closure_last2s_mps": [r["geometry"]["closure_last2s_mps"] for r in selected],
        "closest_vertical_median5_m": [r["geometry"]["closest_vertical_median5_m"] for r in selected],
        "closest_heading_median5_rad": [r["geometry"]["closest_heading_median5_rad"] for r in selected],
        "closest_pitch_median5_rad": [r["geometry"]["closest_pitch_median5_rad"] for r in selected],
        "entry2_vertical_median5_m": [r["geometry"]["entry2_vertical_median5_m"] for r in selected],
        "entry2_heading_median5_rad": [r["geometry"]["entry2_heading_median5_rad"] for r in selected],
        "entry2_pitch_median5_rad": [r["geometry"]["entry2_pitch_median5_rad"] for r in selected],
        "entry2_to_closest_s": [r["geometry"]["entry2_to_closest_s"] for r in selected],
        "base_saturation_fraction": [r["phases"]["last2s"]["base_saturation_fraction"] for r in selected],
        "residual_magnitude_ratio": [r["phases"]["last2s"]["residual_magnitude_ratio"] for r in selected],
        "alignment_heading": [r["phases"]["last2s"]["alignment"][0] for r in selected],
        "alignment_pitch": [r["phases"]["last2s"]["alignment"][1] for r in selected],
        "slope_heading": [r["phases"]["last2s"]["slope"][0] for r in selected],
        "slope_pitch": [r["phases"]["last2s"]["slope"][1] for r in selected],
        "terminal_share_last2s": [r["rewards"]["last2s"]["terminal_share"] for r in selected],
        "dense_terminal_ratio_last2s": [r["rewards"]["last2s"]["dense_terminal_ratio"] for r in selected],
        "terminal_share_episode": [r["rewards"]["episode"]["terminal_share"] for r in selected],
    }
    return {
        "count": len(selected),
        "seeds": [int(r["seed"]) for r in selected],
        "metrics": {
            name: {
                "median": _median(np.asarray(values, dtype=np.float64)),
                "p05": _quantile(np.asarray(values, dtype=np.float64), 0.05),
                "p95": _quantile(np.asarray(values, dtype=np.float64), 0.95),
            }
            for name, values in fields.items()
        },
    }


def _sign_gate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    success = [record for record in records if record["success"]]
    slopes = {"heading": [], "pitch": []}
    for record in success:
        slopes["heading"].append(record["phases"]["last2s"]["slope"][0])
        slopes["pitch"].append(record["phases"]["last2s"]["slope"][1])
    pooled = {
        name: _median(np.asarray(values, dtype=np.float64))
        for name, values in slopes.items()
    }
    # REMUS: positive stern produces negative pitch moment, so a corrective
    # pitch residual yields a positive d(pitch_error)/d(residual) slope.
    expected = {"heading": -1.0, "pitch": 1.0}
    passed = bool(
        math.isfinite(pooled["heading"])
        and math.isfinite(pooled["pitch"])
        and pooled["heading"] * expected["heading"] > 0.0
        and pooled["pitch"] * expected["pitch"] > 0.0
    )
    return {
        "pass": passed,
        "success_pooled_slope": pooled,
        "expected_slope_sign": expected,
    }


def _protocol_check(summary: dict[str, Any], parity: dict[str, Any]) -> dict[str, Any]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(DEV_SEEDS):
        raise AuditError("summary must contain 40 episodes")
    by_seed = {int(row["seed"]): row for row in episodes}
    if tuple(sorted(by_seed)) != DEV_SEEDS:
        raise AuditError("summary seeds must be 20-59")
    failures = sorted(seed for seed, row in by_seed.items() if not bool(row["success"]))
    success = sum(bool(row["success"]) for row in by_seed.values())
    timeout = sum(bool(row["timeout"]) for row in by_seed.values())
    safety = sum(bool(row.get("safety", False)) for row in by_seed.values())
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise AuditError("checkpoint SHA mismatch")
    if abs(float(summary.get("actuator_residual_scale_override", math.nan)) - EXPECTED_SCALE) > 1e-12:
        raise AuditError("eval scale mismatch")
    if summary.get("action_mode") != "frozen-base":
        raise AuditError("action_mode is not frozen-base")
    if not bool(parity.get("pass")):
        raise AuditError("Goal116 R100 trace parity failed")
    return {
        "success_count": success,
        "timeout_count": timeout,
        "safety_count": safety,
        "failure_seeds": failures,
        "expected_primary_failure_seeds": list(EXPECTED_FAILURE_SEEDS),
        "primary_subset_is_failure": all(
            seed in failures for seed in EXPECTED_FAILURE_SEEDS
        ),
        "parity_pass": bool(parity.get("pass")),
    }


def _gate(records: Sequence[dict[str, Any]], sign: dict[str, Any]) -> dict[str, Any]:
    for record in records:
        record["label"] = _classify(record)
    failures = [record for record in records if not record["success"]]
    primary = [
        record for record in failures if int(record["seed"]) in EXPECTED_FAILURE_SEEDS
    ]
    all_counts: dict[str, int] = {}
    for record in failures:
        all_counts[record["label"]] = all_counts.get(record["label"], 0) + 1
    primary_counts: dict[str, int] = {}
    for record in primary:
        primary_counts[record["label"]] = primary_counts.get(record["label"], 0) + 1
    success_counts: dict[str, int] = {}
    for record in records:
        if record["success"]:
            success_counts[record["label"]] = success_counts.get(record["label"], 0) + 1
    if not sign["pass"]:
        return {
            "status": "INVALID_SIGN",
            "reason": "success-control pooled residual-error slope sign does not match the plant convention",
            "label_counts_primary": primary_counts,
            "label_counts_all_failures": all_counts,
            "label_counts_success_control": success_counts,
        }
    majority = [
        (label, count)
        for label, count in primary_counts.items()
        if count >= 8 and success_counts.get(label, 0) < 8
    ]
    if majority:
        label, count = max(majority, key=lambda item: item[1])
        status = {
            "RESIDUAL_DIRECTION_WRONG": "TERMINAL_MECHANISM_IDENTIFIED",
            "BASE_AUTHORITY_LIMITED": "TERMINAL_MECHANISM_IDENTIFIED",
            "TERMINAL_GEOMETRY_LIMITED": "TERMINAL_MECHANISM_IDENTIFIED",
            "APPROACH_DEFICIT": "TERMINAL_MECHANISM_IDENTIFIED",
            "RESIDUAL_TOO_SMALL": "PPO_OPTIMIZATION_SUSPECTED",
        }.get(label, "TERMINAL_MECHANISM_IDENTIFIED")
        return {
            "status": status,
            "majority_label": label,
            "majority_count_primary": count,
            "label_counts_primary": primary_counts,
            "label_counts_all_failures": all_counts,
            "label_counts_success_control": success_counts,
        }
    shared = [
        label
        for label, count in primary_counts.items()
        if count >= 8 and success_counts.get(label, 0) >= 8
    ]
    return {
        "status": "MIXED_TERMINAL_MECHANISM",
        "reason": "no primary-subset label reaches 8/11 without being shared by success controls",
        "shared_with_success": shared,
        "label_counts_primary": primary_counts,
        "label_counts_all_failures": all_counts,
        "label_counts_success_control": success_counts,
    }


def _markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Goal117 phase0：terminal closure / action / reward 归因",
        "",
        f"- gate: `{gate['status']}`",
        f"- protocol: `{json.dumps(payload['protocol'], ensure_ascii=False)}`",
        f"- sign gate: `{json.dumps(payload['sign_gate'], ensure_ascii=False)}`",
        "",
        "## 11-seed 预注册子集标签",
        "",
        "| label | count | seeds |",
        "|---|---:|---|",
    ]
    by_label: dict[str, list[int]] = {}
    for record in payload["episodes"]:
        if not record["success"] and int(record["seed"]) in EXPECTED_FAILURE_SEEDS:
            by_label.setdefault(record["label"], []).append(int(record["seed"]))
    for label, seeds in sorted(by_label.items()):
        lines.append(f"| `{label}` | {len(seeds)} | {', '.join(str(s) for s in seeds)} |")
    lines.extend(
        [
            "",
            "## 20 个 failure 的全集标签",
            "",
            "| label | count | seeds |",
            "|---|---:|---|",
        ]
    )
    by_label_all: dict[str, list[int]] = {}
    for record in payload["episodes"]:
        if not record["success"]:
            by_label_all.setdefault(record["label"], []).append(int(record["seed"]))
    for label, seeds in sorted(by_label_all.items()):
        lines.append(f"| `{label}` | {len(seeds)} | {', '.join(str(s) for s in seeds)} |")
    lines.extend(
        [
            "",
            "## success control 的同判据标签",
            "",
            "| label | count |",
            "|---|---:|",
        ]
    )
    for label, count in sorted(payload["gate"].get("label_counts_success_control", {}).items()):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(
        [
            "",
            "## 分组中位数",
            "",
            "| metric | success | failure | Cliff's delta (failure - success) |",
            "|---|---:|---:|---:|",
        ]
    )
    success = payload["groups"].get("success", {}).get("metrics", {})
    failure = payload["groups"].get("failure", {}).get("metrics", {})
    for name in sorted(set(success) & set(failure)):
        lines.append(
            f"| `{name}` | {success[name]['median']:.4f} | {failure[name]['median']:.4f} | "
            f"{payload['cliffs_delta'].get(name, math.nan):.3f} |"
        )
    entry = payload["entry2_geometry"]
    lines.extend(
        [
            "",
            "## 2 m entry geometry（最便宜 falsification）",
            "",
            f"- success: {entry['success_entry2_geometry_bad']}/{entry['success_with_entry2']} 已在 2 m 处 geometry bad",
            f"- all failure: {entry['failure_entry2_geometry_bad']}/{entry['failure_with_entry2']} 已在 2 m 处 geometry bad",
            f"- 11-seed primary failure: {entry['primary_failure_entry2_geometry_bad']}/11 已在 2 m 处 geometry bad",
            "",
            "## 逐 failure",
            "",
            "| seed | terminal | closest_m | closure_last2s | base_sat | residual_ratio | align_h | align_p | label |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in payload["episodes"]:
        if record["success"]:
            continue
        geometry = record["geometry"]
        last = record["phases"]["last2s"]
        lines.append(
            f"| {record['seed']} | {record['terminal_name']} | {geometry['closest_distance_m']:.3f} | "
            f"{geometry['closure_last2s_mps']:.3f} | {last['base_saturation_fraction']:.3f} | "
            f"{last['residual_magnitude_ratio']:.3f} | {last['alignment'][0]:.3f} | "
            f"{last['alignment'][1]:.3f} | `{record['label']}` |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir or (args.root / "R100_RTRACE")
    summary = _read_json(run_dir / "summary.json")
    trace = _load_trace(run_dir / "trace.npz")
    parity_path = args.root / "parity.json"
    if parity_path.is_file():
        parity = _read_json(parity_path)
    else:
        parity = _read_json(Path("artifacts/lowlevel/goal117_terminal_closure_attribution_20261111/parity.json"))
    protocol = _protocol_check(summary, parity)
    by_seed = {int(row["seed"]): row for row in summary["episodes"]}
    records = [_episode_record(trace, seed, by_seed[seed]) for seed in DEV_SEEDS]
    groups = {
        "success": _group_summary(records, "success"),
        "failure": _group_summary(records, "failure"),
        "primary_failure": _group_summary(records, "primary_failure"),
    }
    metric_names = sorted(groups["success"].get("metrics", {}))
    cliffs: dict[str, float] = {}
    cliffs_primary: dict[str, float] = {}
    for name in metric_names:
        success_values = _metric_values(records, name, success=True)
        failure_values = _metric_values(records, name, success=False)
        primary_values = _metric_values(
            records, name, success=False, seeds=set(EXPECTED_FAILURE_SEEDS)
        )
        cliffs[name] = _cliffs_delta(failure_values, success_values)
        cliffs_primary[name] = _cliffs_delta(primary_values, success_values)
    entry_geometry = {
        "success_entry2_geometry_bad": sum(
            1
            for record in records
            if record["success"] and record["geometry"]["entry2_geometry_bad"] is True
        ),
        "success_with_entry2": sum(
            1
            for record in records
            if record["success"] and record["geometry"]["entry2_geometry_bad"] is not None
        ),
        "failure_entry2_geometry_bad": sum(
            1
            for record in records
            if not record["success"] and record["geometry"]["entry2_geometry_bad"] is True
        ),
        "failure_with_entry2": sum(
            1
            for record in records
            if not record["success"] and record["geometry"]["entry2_geometry_bad"] is not None
        ),
        "primary_failure_entry2_geometry_bad": sum(
            1
            for record in records
            if not record["success"]
            and int(record["seed"]) in EXPECTED_FAILURE_SEEDS
            and record["geometry"]["entry2_geometry_bad"] is True
        ),
    }
    sign_gate = _sign_gate(records)
    gate = _gate(records, sign_gate)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "protocol": protocol,
        "parity": parity,
        "sign_gate": sign_gate,
        "groups": groups,
        "cliffs_delta": cliffs,
        "cliffs_delta_primary_failure": cliffs_primary,
        "entry2_geometry": entry_geometry,
        "gate": gate,
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": gate["status"], "parity": protocol["parity_pass"], "sign": sign_gate["pass"]}, ensure_ascii=False))
    return 0


def _metric_values(
    records: Sequence[dict[str, Any]],
    name: str,
    *,
    success: bool,
    seeds: set[int] | None = None,
) -> list[float]:
    values: list[float] = []
    for record in records:
        if bool(record["success"]) is not bool(success):
            continue
        if seeds is not None and int(record["seed"]) not in seeds:
            continue
        if name.startswith("alignment_") or name.startswith("slope_"):
            index = 0 if name.endswith("heading") else 1
            key = name.rsplit("_", 1)[0]
            value = record["phases"]["last2s"][key][index]
        elif name in {"terminal_share_last2s", "dense_terminal_ratio_last2s"}:
            key = "terminal_share" if name.startswith("terminal_share") else "dense_terminal_ratio"
            value = record["rewards"]["last2s"][key]
        elif name == "terminal_share_episode":
            value = record["rewards"]["episode"]["terminal_share"]
        elif name in {"base_saturation_fraction", "residual_magnitude_ratio"}:
            value = record["phases"]["last2s"][name]
        else:
            value = record["geometry"].get(name, math.nan)
        values.append(math.nan if value is None else float(value))
    return values


if __name__ == "__main__":
    raise SystemExit(main())
