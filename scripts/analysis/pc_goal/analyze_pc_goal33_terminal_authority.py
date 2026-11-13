#!/usr/bin/env python3
"""Read-only terminal vertical-authority diagnostic for Goal33.

Consumes the four frozen Goal31 formal traces and splits the Goal32
authority shortfall into speed-dependent authority, stern limit/rate binding,
and entry-excess versus time-budget contributions.
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

SCHEMA_VERSION = "auv_pc_goal33_terminal_authority_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}
MIN_VERTICAL_MISS_SEEDS = 10
MIN_PLANE_TIME_S = 0.5

EXPECTED_MAX_STERN_DEG = 20.0
EXPECTED_MAX_FIN_RATE_DEG_S = 150.0
EXPECTED_PITCH_GAIN = 2.0
EXPECTED_PITCH_RATE_GAIN = 1.0
CONFIG_ATOL = 1e-9

SPEED_CORRELATION_MIN = 0.30
SPEED_AUTHORITY_GROUP_DELTA_MIN_MPS = 0.05

STERN_SHORTFALL_CORRELATION_MIN = 0.30
STERN_BINDING_GROUP_DELTA_MIN = 0.05
STERN_HIGH_GROUP_BINDING_MIN = 0.05

EXCESS_CORRELATION_MIN = 0.50
TIME_CORRELATION_MIN = -0.30
TIME_MEDIAN_MIN_S = 6.0

EPS = 1e-12

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "outcome",
    "vertical_miss",
    "entry_excess_m",
    "time_to_plane_s",
    "plane_crossed",
    "required_closure_mps",
    "authority_p95_mps",
    "required_minus_authority_mps",
    "horizontal_speed_median_mps",
    "stern_binding_rate",
    "stern_limit_bound_rate",
    "stern_rate_bound_rate",
    "stern_demand_abs_median_rad",
    "stern_actual_abs_median_rad",
    "pitch_error_abs_median_rad",
    "material_saturation_rate",
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
    _HERE / "analyze_pc_goal32_terminal_entry.py", "_goal33_goal32_helpers"
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


def _median(values: list[float | None]) -> float | None:
    return _GOAL32._median(values)


def _stats(values: list[float | None]) -> dict[str, Any]:
    finite = _GOAL32._finite_values(values)
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5.0)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _resolve_config_path(summary: dict[str, Any], directory: Path) -> Path:
    value = summary.get("environment_config_override") or summary.get(
        "environment_config"
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{directory}: summary has no environment config path")
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise ValueError(f"{directory}: environment config does not exist: {path}")
    return path


def _load_limits(summary: dict[str, Any], directory: Path) -> dict[str, float]:
    path = _resolve_config_path(summary, directory)
    payload = json.loads(path.read_text(encoding="utf-8"))
    actuator = payload.get("actuator")
    controller = payload.get("base_controller")
    if not isinstance(actuator, dict) or not isinstance(controller, dict):
        raise ValueError(f"{path}: missing actuator/base_controller config")
    values = {
        "max_stern_plane_angle_deg": float(actuator["max_stern_plane_angle_deg"]),
        "max_fin_rate_deg_s": float(actuator["max_fin_rate_deg_s"]),
        "pitch_gain": float(controller["pitch_gain"]),
        "pitch_rate_gain": float(controller["pitch_rate_gain"]),
    }
    expected = {
        "max_stern_plane_angle_deg": EXPECTED_MAX_STERN_DEG,
        "max_fin_rate_deg_s": EXPECTED_MAX_FIN_RATE_DEG_S,
        "pitch_gain": EXPECTED_PITCH_GAIN,
        "pitch_rate_gain": EXPECTED_PITCH_RATE_GAIN,
    }
    mismatches = {
        key: values[key]
        for key, expected_value in expected.items()
        if not math.isclose(values[key], expected_value, abs_tol=CONFIG_ATOL)
    }
    if mismatches:
        raise ValueError(f"{path}: unexpected controller/actuator values {mismatches}")
    return {
        **values,
        "max_stern_plane_angle_rad": math.radians(
            values["max_stern_plane_angle_deg"]
        ),
        "max_fin_rate_rad_s": math.radians(values["max_fin_rate_deg_s"]),
        "config_path": str(path),
    }


def _segment_bounds(
    trace: Any,
    indices: np.ndarray,
    record: dict[str, Any],
) -> tuple[int, int]:
    start = int(record["entry_index"])
    if start < 0 or start >= indices.size:
        raise ValueError("entry index is outside the episode")
    eta = np.asarray(trace["eta"][indices, :3], dtype=np.float64)
    goal = np.asarray(trace["goal_ned_m"][indices[start]], dtype=np.float64)
    tangent = np.asarray(
        trace["goal_tangent_ned"][indices[start]], dtype=np.float64
    )
    tangent_rows = np.broadcast_to(tangent, (indices.size, 3))
    longitudinal, _, _, _ = _GOAL32._decompose_many(eta - goal, tangent_rows)
    crossing = np.flatnonzero(longitudinal[start:] >= 0.0)
    end = int(start + crossing[0]) if crossing.size else int(indices.size - 1)
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    measured = float(time_s[end] - time_s[start])
    if not math.isclose(measured, float(record["time_to_plane_s"]), abs_tol=1e-8):
        raise ValueError("Goal32 and Goal33 time-to-plane disagree")
    return start, end


def _stern_segment_metrics(
    trace: Any,
    indices: np.ndarray,
    start: int,
    end: int,
    limits: dict[str, float],
) -> dict[str, Any]:
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    pitch_error = np.asarray(trace["pitch_error_rad"][indices], dtype=np.float64)
    pitch_rate = np.asarray(trace["nu"][indices, 4], dtype=np.float64)
    stern = np.asarray(trace["actual_actuator"][indices, 1], dtype=np.float64)
    saturation = np.asarray(trace["material_saturated"][indices], dtype=np.bool_)
    demand = (
        -limits["pitch_gain"] * pitch_error
        - limits["pitch_rate_gain"] * pitch_rate
    )
    segment = np.arange(start, end + 1)
    if segment.size < 2:
        raise ValueError("terminal stern segment has fewer than two samples")
    limit_bound = np.abs(demand[segment]) > limits["max_stern_plane_angle_rad"] + 1e-6

    rate_rows = np.arange(max(start, 1), end + 1)
    if rate_rows.size == 0:
        raise ValueError("terminal stern segment has no rate pairs")
    dt = time_s[rate_rows] - time_s[rate_rows - 1]
    if np.any(dt <= 0.0):
        raise ValueError("terminal stern segment has non-positive dt")
    stern_change = stern[rate_rows] - stern[rate_rows - 1]
    demand_change_needed = np.abs(demand[rate_rows] - stern[rate_rows - 1])
    max_step = limits["max_fin_rate_rad_s"] * dt
    rate_bound = (np.abs(stern_change) >= max_step - 1e-6) & (
        demand_change_needed > max_step + 1e-6
    )

    binding = limit_bound.copy()
    binding[rate_rows - start] |= rate_bound
    return {
        "stern_binding_rate": float(np.mean(binding)),
        "stern_limit_bound_rate": float(np.mean(limit_bound)),
        "stern_rate_bound_rate": float(np.mean(rate_bound)),
        "stern_demand_abs_median_rad": float(np.median(np.abs(demand[segment]))),
        "stern_actual_abs_median_rad": float(np.median(np.abs(stern[segment]))),
        "pitch_error_abs_median_rad": float(
            np.median(np.abs(pitch_error[segment]))
        ),
        "material_saturation_rate": float(np.mean(saturation[segment])),
    }


def _speed_segment_metrics(
    trace: Any,
    indices: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    eta = np.asarray(trace["eta"][indices], dtype=np.float64)
    ground_body = np.asarray(
        trace["ground_velocity_body_mps"][indices], dtype=np.float64
    )
    ground_ned = _GOAL32._body_to_world_ned(ground_body, eta)
    horizontal_speed = np.hypot(ground_ned[:, 0], ground_ned[:, 1])
    segment = np.arange(start, end + 1)
    return {
        "horizontal_speed_median_mps": float(
            np.median(horizontal_speed[segment])
        ),
        "ground_speed_median_mps": float(
            np.median(np.linalg.norm(ground_ned[segment], axis=1))
        ),
    }


def _enrich_record(
    record: dict[str, Any],
    trace: Any,
    indices: np.ndarray,
    goal_radius_m: float,
    limits: dict[str, float],
) -> dict[str, Any]:
    start, end = _segment_bounds(trace, indices, record)
    stern_metrics = _stern_segment_metrics(trace, indices, start, end, limits)
    speed_metrics = _speed_segment_metrics(trace, indices, start, end)
    row = dict(record)
    row.update(stern_metrics)
    row.update(speed_metrics)
    row["entry_excess_m"] = max(
        0.0, float(record["entry_abs_vertical_m"]) - goal_radius_m
    )
    row["vertical_miss"] = bool(
        float(record["terminal_abs_vertical_m"]) > goal_radius_m
    )
    return row


def _load_records(
    formal_root: Path,
    scenario: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    directory = formal_root / scenario / mode
    summary, records, raw_trace = _GOAL32._load_records(formal_root, scenario, mode)
    trace = {key: raw_trace[key] for key in raw_trace.files}
    limits = _load_limits(summary, directory)
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    enriched: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{scenario}/{mode}: no trace rows for seed {seed}")
        enriched.append(
            _enrich_record(
                record,
                trace,
                indices,
                GOAL_RADIUS_M[scenario],
                limits,
            )
        )
    return {**summary, "validated_limits": limits}, enriched, trace


def _rank_halves(
    records: list[dict[str, Any]],
    key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda row: float(row[key]))
    half = len(ordered) // 2
    if half == 0:
        raise ValueError("cannot split fewer than two records into halves")
    return ordered[:half], ordered[-half:]


def _h_speed(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    speed = [row["horizontal_speed_median_mps"] for row in miss]
    authority = [row["authority_p95_mps"] for row in miss]
    correlation = _GOAL32._spearman(speed, authority)
    low, high = _rank_halves(miss, "horizontal_speed_median_mps")
    low_authority = _median([row["authority_p95_mps"] for row in low])
    high_authority = _median([row["authority_p95_mps"] for row in high])
    group_delta = (
        None
        if low_authority is None or high_authority is None
        else high_authority - low_authority
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "speed_authority_spearman_ge_0_30": bool(
            correlation is not None and correlation >= SPEED_CORRELATION_MIN
        ),
        "high_minus_low_authority_ge_0_05_mps": bool(
            group_delta is not None
            and group_delta >= SPEED_AUTHORITY_GROUP_DELTA_MIN_MPS
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "speed_authority_spearman": correlation,
        "low_speed_authority_median_mps": low_authority,
        "high_speed_authority_median_mps": high_authority,
        "high_minus_low_authority_mps": group_delta,
        "group_size": len(low),
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_stern(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    shortfall = [row["required_minus_authority_mps"] for row in miss]
    binding = [row["stern_binding_rate"] for row in miss]
    correlation = _GOAL32._spearman(shortfall, binding)
    low, high = _rank_halves(miss, "required_minus_authority_mps")
    low_binding = _median([row["stern_binding_rate"] for row in low])
    high_binding = _median([row["stern_binding_rate"] for row in high])
    group_delta = (
        None
        if low_binding is None or high_binding is None
        else high_binding - low_binding
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "shortfall_binding_spearman_ge_0_30": bool(
            correlation is not None
            and correlation >= STERN_SHORTFALL_CORRELATION_MIN
        ),
        "high_minus_low_binding_ge_0_05": bool(
            group_delta is not None
            and group_delta >= STERN_BINDING_GROUP_DELTA_MIN
        ),
        "high_shortfall_binding_median_ge_0_05": bool(
            high_binding is not None and high_binding >= STERN_HIGH_GROUP_BINDING_MIN
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "shortfall_binding_spearman": correlation,
        "low_shortfall_binding_median": low_binding,
        "high_shortfall_binding_median": high_binding,
        "high_minus_low_binding": group_delta,
        "group_size": len(low),
        "stern_binding_median": _median(binding),
        "stern_limit_bound_median": _median(
            [row["stern_limit_bound_rate"] for row in miss]
        ),
        "stern_rate_bound_median": _median(
            [row["stern_rate_bound_rate"] for row in miss]
        ),
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_excess(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    excess = [row["entry_excess_m"] for row in miss]
    required = [row["required_closure_mps"] for row in miss]
    time_to_plane = [row["time_to_plane_s"] for row in miss]
    excess_correlation = _GOAL32._spearman(excess, required)
    time_correlation = _GOAL32._spearman(time_to_plane, required)
    time_median = _median(time_to_plane)
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "excess_required_spearman_ge_0_50": bool(
            excess_correlation is not None
            and excess_correlation >= EXCESS_CORRELATION_MIN
        ),
        "time_required_spearman_ge_minus_0_30": bool(
            time_correlation is not None
            and time_correlation >= TIME_CORRELATION_MIN
        ),
        "time_median_ge_6_s": bool(
            time_median is not None and time_median >= TIME_MEDIAN_MIN_S
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "excess_required_spearman": excess_correlation,
        "time_required_spearman": time_correlation,
        "time_to_plane_median_s": time_median,
        "entry_excess_median_m": _median(excess),
        "required_closure_median_mps": _median(required),
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    keys = (
        "horizontal_speed_median_mps",
        "authority_p95_mps",
        "required_closure_mps",
        "required_minus_authority_mps",
        "stern_binding_rate",
        "stern_limit_bound_rate",
        "stern_rate_bound_rate",
        "stern_demand_abs_median_rad",
        "stern_actual_abs_median_rad",
        "pitch_error_abs_median_rad",
        "material_saturation_rate",
        "entry_excess_m",
        "time_to_plane_s",
    )
    return {
        "seed_count": len(records),
        "vertical_miss_seed_count": len(miss),
        "plane_crossed_rate": float(
            np.mean([bool(row["plane_crossed"]) for row in records])
        ),
        "stats": {key: _stats([row[key] for row in records]) for key in keys},
        "vertical_miss_stats": {
            key: _stats([row[key] for row in miss]) for key in keys
        },
    }


def _mode_verdict(blocks: dict[str, dict[str, dict[str, Any]]], key: str) -> str:
    passes = [
        bool(blocks[scenario][mode][key]["passes_all"])
        for scenario in SCENARIOS
        for mode in MODES
    ]
    if all(passes):
        return "SUPPORTED"
    if any(passes):
        return "PARTIAL"
    return "FALSIFIED"


def _overall_verdict(blocks: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    h_speed = _mode_verdict(blocks, "h_speed")
    h_stern = _mode_verdict(blocks, "h_stern")
    h_excess = _mode_verdict(blocks, "h_excess")
    supported = {
        name
        for name, verdict in (
            ("H_SPEED", h_speed),
            ("H_STERN", h_stern),
            ("H_EXCESS", h_excess),
        )
        if verdict == "SUPPORTED"
    }
    if supported == {"H_SPEED"}:
        decision = "SPEED_DEPENDENT_AUTHORITY_SUPPORTED"
        recommendation = (
            "预注册速度/terminal 时间预算的最小单因素诊断；不要先改 reward 或 terminal"
            " reference。"
        )
    elif supported == {"H_STERN"}:
        decision = "STERN_LIMIT_SUPPORTED"
        recommendation = (
            "预注册固定参数 stern/姿态权限诊断；先确认角度限幅还是速率限幅，不先改"
            " reward。"
        )
    elif supported == {"H_EXCESS"}:
        decision = "ENTRY_EXCESS_DOMINANT"
        recommendation = (
            "预注册入口垂直 excess 修正或更早交接诊断；单纯延长 terminal window"
            " 不是首要方向。"
        )
    elif len(supported) >= 2:
        decision = "MULTI_FACTOR"
        recommendation = (
            "至少两个机制达到预注册判据；拆成最小单因素候选，不允许合并修改。"
        )
    else:
        decision = "NO_SUPPORTED_MECHANISM"
        recommendation = (
            "速度包络、stern 限幅/速率和入口 excess 都不满足预注册判据；停止该"
            " authority 机制线，不预注册控制器候选。"
        )
    return {
        "H_SPEED": h_speed,
        "H_STERN": h_stern,
        "H_EXCESS": h_excess,
        "decision": decision,
        "recommendation": recommendation,
    }


def analyze(formal_root: Path) -> dict[str, Any]:
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    validation: dict[str, Any] = {
        "formal_root": str(formal_root),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "heldout_opened": False,
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        blocks[scenario] = {}
        validation["scenarios"][scenario] = {
            "goal_radius_m": GOAL_RADIUS_M[scenario],
            "modes": {},
        }
        for mode in MODES:
            summary, records, _ = _load_records(formal_root, scenario, mode)
            limits = summary["validated_limits"]
            validation["scenarios"][scenario]["modes"][mode] = {
                "seed_count": len(records),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "config_path": limits["config_path"],
                "max_stern_plane_angle_deg": limits[
                    "max_stern_plane_angle_deg"
                ],
                "max_fin_rate_deg_s": limits["max_fin_rate_deg_s"],
                "pitch_gain": limits["pitch_gain"],
                "pitch_rate_gain": limits["pitch_rate_gain"],
            }
            blocks[scenario][mode] = {
                "source": str(formal_root / scenario / mode),
                "aggregate": _aggregate(records),
                "h_speed": _h_speed(records),
                "h_stern": _h_stern(records),
                "h_excess": _h_excess(records),
                "records": records,
            }

    invalid = any(
        not blocks[scenario][mode][key]["criteria"]["enough_vertical_miss_seeds"]
        for scenario in SCENARIOS
        for mode in MODES
        for key in ("h_speed", "h_stern", "h_excess")
    )
    verdicts = _overall_verdict(blocks)
    if invalid:
        verdicts["decision"] = "INVALID_DIAGNOSTIC"
        verdicts["recommendation"] = (
            "至少一个 block 的 vertical-miss seed 数不足；不得挑选 seed 或回改阈值，"
            "诊断为无效。"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": {
            "speed_correlation_min": SPEED_CORRELATION_MIN,
            "speed_authority_group_delta_min_mps": (
                SPEED_AUTHORITY_GROUP_DELTA_MIN_MPS
            ),
            "stern_shortfall_correlation_min": STERN_SHORTFALL_CORRELATION_MIN,
            "stern_binding_group_delta_min": STERN_BINDING_GROUP_DELTA_MIN,
            "stern_high_group_binding_min": STERN_HIGH_GROUP_BINDING_MIN,
            "excess_correlation_min": EXCESS_CORRELATION_MIN,
            "time_correlation_min": TIME_CORRELATION_MIN,
            "time_median_min_s": TIME_MEDIAN_MIN_S,
            "min_vertical_miss_seeds": MIN_VERTICAL_MISS_SEEDS,
            "min_plane_time_s": MIN_PLANE_TIME_S,
        },
        "validation": validation,
        "scenarios": blocks,
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
        "# Goal33 terminal vertical-authority diagnostic",
        "",
        "Read-only analysis of the frozen Goal31 formal traces; no rollout, training, ",
        "parameter change, or held-out access.",
        "",
        "## Verdict",
        "",
        "| hypothesis | verdict |",
        "|---|---|",
        f"| H-SPEED | {verdicts['H_SPEED']} |",
        f"| H-STERN | {verdicts['H_STERN']} |",
        f"| H-EXCESS | {verdicts['H_EXCESS']} |",
        f"| overall decision | **{verdicts['decision']}** |",
        "",
        f"Recommendation: {verdicts['recommendation']}",
        "",
        "## Validation",
        "",
        "| scenario | mode | seeds | max stern | max fin rate | checkpoint |",
        "|---|---|---:|---:|---:|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["validation"]["scenarios"][scenario]["modes"][mode]
            lines.append(
                f"| {scenario} | {mode} | {block['seed_count']} | "
                f"{_fmt(block['max_stern_plane_angle_deg'], 1)}° | "
                f"{_fmt(block['max_fin_rate_deg_s'], 1)}°/s | "
                f"`{block['checkpoint_sha256']}` |"
            )
    lines.extend(
        [
            "",
            "## H-SPEED",
            "",
            "| scenario | mode | vertical miss n | speed rho | low auth med | high auth med | delta | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            stats = result["scenarios"][scenario][mode]["h_speed"]
            lines.append(
                f"| {scenario} | {mode} | {stats['vertical_miss_seed_count']} | "
                f"{_fmt(stats['speed_authority_spearman'])} | "
                f"{_fmt(stats['low_speed_authority_median_mps'])} | "
                f"{_fmt(stats['high_speed_authority_median_mps'])} | "
                f"{_fmt(stats['high_minus_low_authority_mps'])} | "
                f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## H-STERN",
            "",
            "| scenario | mode | vertical miss n | shortfall rho | low bind med | high bind med | delta | high group bind | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            stats = result["scenarios"][scenario][mode]["h_stern"]
            lines.append(
                f"| {scenario} | {mode} | {stats['vertical_miss_seed_count']} | "
                f"{_fmt(stats['shortfall_binding_spearman'])} | "
                f"{_fmt(stats['low_shortfall_binding_median'])} | "
                f"{_fmt(stats['high_shortfall_binding_median'])} | "
                f"{_fmt(stats['high_minus_low_binding'])} | "
                f"{_fmt(stats['high_shortfall_binding_median'])} | "
                f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## H-EXCESS",
            "",
            "| scenario | mode | vertical miss n | excess rho | time rho | time med (s) | entry excess med (m) | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            stats = result["scenarios"][scenario][mode]["h_excess"]
            lines.append(
                f"| {scenario} | {mode} | {stats['vertical_miss_seed_count']} | "
                f"{_fmt(stats['excess_required_spearman'])} | "
                f"{_fmt(stats['time_required_spearman'])} | "
                f"{_fmt(stats['time_to_plane_median_s'])} | "
                f"{_fmt(stats['entry_excess_median_m'])} | "
                f"{'PASS' if stats['passes_all'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Vertical-miss aggregates",
            "",
            "| scenario | mode | speed med (m/s) | authority med (m/s) | shortfall med (m/s) | stern bind med | limit bind med | rate bind med |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            stats = result["scenarios"][scenario][mode]["aggregate"][
                "vertical_miss_stats"
            ]
            lines.append(
                f"| {scenario} | {mode} | "
                f"{_fmt(stats['horizontal_speed_median_mps']['median'])} | "
                f"{_fmt(stats['authority_p95_mps']['median'])} | "
                f"{_fmt(stats['required_minus_authority_mps']['median'])} | "
                f"{_fmt(stats['stern_binding_rate']['median'])} | "
                f"{_fmt(stats['stern_limit_bound_rate']['median'])} | "
                f"{_fmt(stats['stern_rate_bound_rate']['median'])} |"
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
    (args.output / "authority_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "authority_diagnostic.md").write_text(
        _markdown(result) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output / "authority_per_seed.csv", result)
    print(json.dumps(result["verdicts"], indent=2, sort_keys=True))
    return 0 if result["verdicts"]["decision"] != "INVALID_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
