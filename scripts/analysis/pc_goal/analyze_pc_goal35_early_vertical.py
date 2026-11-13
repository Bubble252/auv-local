#!/usr/bin/env python3
"""Read-only early vertical tracking and handoff diagnostic for Goal35.

The script consumes the four frozen Goal31 formal trace directories. It never
loads a policy, steps an environment, trains, or reads held-out seeds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal35_early_vertical_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}
MIN_VERTICAL_MISS_SEEDS = 10
MIN_SEGMENT_SAMPLES = 10
PITCH_ACTIVE_ERROR_M = 0.05
PITCH_ACTIVE_COMMAND_RAD = 0.02
HANDOFF_VELOCITY_DEADBAND_MPS = 0.02
SIGN_MATCH_MIN = 0.60
COMMAND_PITCH_MIN_RAD = 0.03
TRACKING_ERROR_MIN_RAD = 0.05
TRACKING_TERMINAL_RHO_MIN = 0.30
TRACKING_HIGH_LOW_TERMINAL_DELTA_MIN_M = 0.25
HANDOFF_WINDOW_S = 5.0
HANDOFF_PITCH_JUMP_MIN_RAD = 0.05
HANDOFF_WORSENING_MIN_M = 0.10
HANDOFF_CLOSURE_DROP_MIN = 0.10
HANDOFF_TERMINAL_RHO_MIN = 0.30
HANDOFF_HIGH_LOW_TERMINAL_DELTA_MIN_M = 0.25
INITIAL_RHO_MIN = 0.50
INITIAL_OUTSIDE_FRACTION_MIN = 0.60
INITIAL_TERMINAL_RATIO_MIN = 0.80
EPS = 1e-12

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "outcome",
    "vertical_miss",
    "initial_abs_vertical_m",
    "initial_distance_to_goal_m",
    "terminal_abs_vertical_m",
    "median_abs_command_pitch_rad",
    "median_abs_tracking_error_rad",
    "reference_sign_match_fraction",
    "actual_sign_match_fraction",
    "pitch_command_jump_rad",
    "vertical_worsening_m",
    "closure_drop",
    "handoff_distance_m",
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
    _HERE / "analyze_pc_goal32_terminal_entry.py", "_goal35_goal32_helpers"
)
_GOAL34 = _load_module(
    _HERE / "analyze_pc_goal34_global_vertical.py", "_goal35_goal34_helpers"
)
_median = _GOAL34._median
_fraction = _GOAL34._fraction
_group_median_delta = _GOAL34._group_median_delta


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


def _first_crossing(distance: np.ndarray, threshold_m: float) -> int:
    hits = np.flatnonzero(distance <= threshold_m)
    if hits.size == 0:
        raise ValueError(f"episode never reaches {threshold_m:.1f} m to goal")
    return int(hits[0])


def _window_sign_fraction(
    vertical_error: np.ndarray,
    vertical_velocity: np.ndarray,
) -> float | None:
    active = (np.abs(vertical_error) > PITCH_ACTIVE_ERROR_M) & (
        np.abs(vertical_velocity) > HANDOFF_VELOCITY_DEADBAND_MPS
    )
    if not np.any(active):
        return None
    closing = np.maximum(0.0, -np.sign(vertical_error[active]) * vertical_velocity[active])
    return float(np.mean(closing > 0.0))


def _initial_metrics(
    trace: Any,
    indices: np.ndarray,
    vertical: np.ndarray,
    distance: np.ndarray,
    terminal_abs_vertical_m: float,
) -> dict[str, Any]:
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    return {
        "initial_time_s": float(time_s[0]),
        "initial_distance_to_goal_m": float(distance[0]),
        "initial_signed_vertical_m": float(vertical[0]),
        "initial_abs_vertical_m": float(abs(vertical[0])),
        "terminal_abs_vertical_m": float(terminal_abs_vertical_m),
    }


def _pitch_tracking_metrics(
    trace: Any,
    indices: np.ndarray,
    vertical: np.ndarray,
    distance: np.ndarray,
) -> dict[str, Any]:
    index40 = _first_crossing(distance, 40.0)
    index20 = _first_crossing(distance, 20.0)
    if index20 < index40:
        raise ValueError("40 m crossing occurs after 20 m crossing")
    segment = np.arange(index40, index20 + 1)
    if segment.size < MIN_SEGMENT_SAMPLES:
        raise ValueError("40-to-20 m segment has too few samples")

    command_pitch = np.asarray(trace["command"][indices[segment], 1], dtype=np.float64)
    actual_pitch = np.asarray(trace["eta"][indices[segment], 4], dtype=np.float64)
    error = vertical[segment]
    tracking_error = command_pitch - actual_pitch
    active = (np.abs(error) > PITCH_ACTIVE_ERROR_M) & (
        np.abs(command_pitch) > PITCH_ACTIVE_COMMAND_RAD
    )
    desired_sign = -np.sign(error)
    reference_match = None
    actual_match = None
    if np.any(active):
        reference_match = float(
            np.mean(np.sign(command_pitch[active]) == desired_sign[active])
        )
        actual_match = float(
            np.mean(np.sign(actual_pitch[active]) == desired_sign[active])
        )
    return {
        "early_segment_sample_count": int(segment.size),
        "early_active_sample_count": int(np.count_nonzero(active)),
        "median_abs_command_pitch_rad": float(np.median(np.abs(command_pitch))),
        "median_abs_tracking_error_rad": float(np.median(np.abs(tracking_error))),
        "reference_sign_match_fraction": reference_match,
        "actual_sign_match_fraction": actual_match,
    }


def _handoff_metrics(
    trace: Any,
    indices: np.ndarray,
    vertical: np.ndarray,
    vertical_velocity: np.ndarray,
    distance: np.ndarray,
) -> dict[str, Any]:
    phase = np.asarray(trace["guidance_terminal_phase"][indices], dtype=np.bool_)
    phase_rows = np.flatnonzero(phase)
    if phase_rows.size == 0:
        raise ValueError("episode never enters guidance terminal phase")
    handoff = int(phase_rows[0])
    time_s = np.asarray(trace["time_s"][indices], dtype=np.float64)
    handoff_time = float(time_s[handoff])
    pre = np.flatnonzero(
        (time_s >= handoff_time - HANDOFF_WINDOW_S) & (time_s < handoff_time)
    )
    post = np.flatnonzero(
        (time_s >= handoff_time) & (time_s <= handoff_time + HANDOFF_WINDOW_S)
    )
    if pre.size < MIN_SEGMENT_SAMPLES or post.size < MIN_SEGMENT_SAMPLES:
        raise ValueError("handoff pre/post window has too few samples")

    command_pitch = np.asarray(trace["command"][indices, 1], dtype=np.float64)
    pre_pitch = _median(command_pitch[pre].tolist())
    post_pitch = _median(command_pitch[post].tolist())
    pre_abs = _median(np.abs(vertical[pre]).tolist())
    post_abs = _median(np.abs(vertical[post]).tolist())
    if pre_pitch is None or post_pitch is None or pre_abs is None or post_abs is None:
        raise ValueError("handoff metric is not finite")
    pre_fraction = _window_sign_fraction(vertical[pre], vertical_velocity[pre])
    post_fraction = _window_sign_fraction(vertical[post], vertical_velocity[post])
    closure_drop = (
        None
        if pre_fraction is None or post_fraction is None
        else max(0.0, pre_fraction - post_fraction)
    )
    return {
        "handoff_time_s": handoff_time,
        "handoff_distance_m": float(distance[handoff]),
        "pre_command_pitch_median_rad": float(pre_pitch),
        "post_command_pitch_median_rad": float(post_pitch),
        "pitch_command_jump_rad": float(post_pitch - pre_pitch),
        "pre_abs_vertical_m": float(pre_abs),
        "post_abs_vertical_m": float(post_abs),
        "vertical_worsening_m": float(max(0.0, post_abs - pre_abs)),
        "pre_sign_matched_fraction": pre_fraction,
        "post_sign_matched_fraction": post_fraction,
        "closure_drop": closure_drop,
    }


def _enrich_record(
    trace: Any,
    indices: np.ndarray,
    record: dict[str, Any],
) -> dict[str, Any]:
    _, _, vertical, distance, vertical_velocity = _GOAL34._distance_geometry(
        trace, indices
    )
    terminal_abs = float(record["terminal_abs_vertical_m"])
    row = dict(record)
    row.update(_initial_metrics(trace, indices, vertical, distance, terminal_abs))
    row.update(_pitch_tracking_metrics(trace, indices, vertical, distance))
    row.update(_handoff_metrics(trace, indices, vertical, vertical_velocity, distance))
    row["abs_pitch_command_jump_rad"] = abs(float(row["pitch_command_jump_rad"]))
    return row


def _load_records(
    formal_root: Path,
    scenario: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    directory = formal_root / scenario / mode
    summary, records, trace = _GOAL32._load_records(formal_root, scenario, mode)
    required = {
        "command",
        "command_base_physical",
        "command_corrected_physical",
        "nu",
        "pitch_error_rad",
        "guidance_terminal_phase",
    }
    missing = sorted(required - set(trace.files))
    if missing:
        raise ValueError(f"{directory / 'trace.npz'}: missing fields {missing}")
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    enriched: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{scenario}/{mode}: no trace rows for seed {seed}")
        command_pitch = np.asarray(trace["command"][indices, 1], dtype=np.float64)
        actual_pitch = np.asarray(trace["eta"][indices, 4], dtype=np.float64)
        pitch_error = np.asarray(trace["pitch_error_rad"][indices], dtype=np.float64)
        if float(np.max(np.abs((command_pitch - actual_pitch) - pitch_error))) > 1e-6:
            raise ValueError(f"{scenario}/{mode}: pitch_error_rad does not match command")
        item = dict(record)
        item["scenario"] = scenario
        item["mode"] = mode
        enriched.append(_enrich_record(trace, indices, item))
    return summary, enriched, trace


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
        "initial_abs_vertical_m",
        "terminal_abs_vertical_m",
        "median_abs_command_pitch_rad",
        "median_abs_tracking_error_rad",
        "reference_sign_match_fraction",
        "actual_sign_match_fraction",
        "pitch_command_jump_rad",
        "vertical_worsening_m",
        "closure_drop",
    )
    return {
        "seed_count": len(records),
        "vertical_miss_seed_count": int(sum(row["vertical_miss"] for row in records)),
        "stats": {key: _stats([row.get(key) for row in records]) for key in keys},
    }


def _h_initial(records: list[dict[str, Any]], goal_radius_m: float) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    rho = _GOAL32._spearman(
        [row["initial_abs_vertical_m"] for row in miss],
        [row["terminal_abs_vertical_m"] for row in miss],
    )
    outside_fraction = _fraction(
        [row["initial_abs_vertical_m"] > goal_radius_m for row in miss]
    )
    initial_median = _median([row["initial_abs_vertical_m"] for row in miss])
    terminal_median = _median([row["terminal_abs_vertical_m"] for row in miss])
    ratio = (
        None
        if initial_median is None or terminal_median is None or terminal_median <= EPS
        else initial_median / terminal_median
    )
    criteria = {
        "enough_vertical_miss_seeds": len(miss) >= MIN_VERTICAL_MISS_SEEDS,
        "initial_terminal_rho_ge_0_50": bool(rho is not None and rho >= INITIAL_RHO_MIN),
        "initial_outside_radius_fraction_ge_0_60": bool(
            outside_fraction is not None
            and outside_fraction >= INITIAL_OUTSIDE_FRACTION_MIN
        ),
        "initial_terminal_ratio_ge_0_80": bool(
            ratio is not None and ratio >= INITIAL_TERMINAL_RATIO_MIN
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "initial_terminal_rho": rho,
        "initial_outside_radius_fraction": outside_fraction,
        "initial_abs_vertical_median_m": initial_median,
        "terminal_abs_vertical_median_m": terminal_median,
        "initial_terminal_ratio": ratio,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_track(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    valid = [
        row
        for row in miss
        if row.get("median_abs_tracking_error_rad") is not None
        and row.get("median_abs_command_pitch_rad") is not None
    ]
    command_pitch_median = _median([row["median_abs_command_pitch_rad"] for row in valid])
    tracking_error_median = _median([row["median_abs_tracking_error_rad"] for row in valid])
    reference_match_median = _median([row["reference_sign_match_fraction"] for row in valid])
    actual_match_median = _median([row["actual_sign_match_fraction"] for row in valid])
    rho = _GOAL32._spearman(
        [row["median_abs_tracking_error_rad"] for row in valid],
        [row["terminal_abs_vertical_m"] for row in valid],
    )
    high_low_delta = _group_median_delta(
        valid, "median_abs_tracking_error_rad", "terminal_abs_vertical_m"
    )
    criteria = {
        "enough_valid_tracking_seeds": len(valid) >= MIN_VERTICAL_MISS_SEEDS,
        "command_pitch_median_ge_0_03": bool(
            command_pitch_median is not None
            and command_pitch_median >= COMMAND_PITCH_MIN_RAD
        ),
        "reference_sign_match_ge_0_60": bool(
            reference_match_median is not None and reference_match_median >= SIGN_MATCH_MIN
        ),
        "tracking_error_median_ge_0_05": bool(
            tracking_error_median is not None
            and tracking_error_median >= TRACKING_ERROR_MIN_RAD
        ),
        "tracking_terminal_rho_ge_0_30": bool(
            rho is not None and rho >= TRACKING_TERMINAL_RHO_MIN
        ),
        "high_low_terminal_delta_ge_0_25m": bool(
            high_low_delta is not None
            and high_low_delta >= TRACKING_HIGH_LOW_TERMINAL_DELTA_MIN_M
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "valid_tracking_seed_count": len(valid),
        "command_pitch_median_rad": command_pitch_median,
        "tracking_error_median_rad": tracking_error_median,
        "reference_sign_match_median": reference_match_median,
        "actual_sign_match_median": actual_match_median,
        "tracking_terminal_rho": rho,
        "high_minus_low_terminal_abs_m": high_low_delta,
        "criteria": criteria,
        "passes_all": bool(all(criteria.values())),
    }


def _h_handoff(records: list[dict[str, Any]]) -> dict[str, Any]:
    miss = [row for row in records if row["vertical_miss"]]
    valid = [
        row
        for row in miss
        if row.get("pitch_command_jump_rad") is not None
        and row.get("vertical_worsening_m") is not None
    ]
    jump_median = _median([abs(row["pitch_command_jump_rad"]) for row in valid])
    worsening_median = _median([row["vertical_worsening_m"] for row in valid])
    closure_drop_median = _median([row.get("closure_drop") for row in valid])
    rho = _GOAL32._spearman(
        [row["vertical_worsening_m"] for row in valid],
        [row["terminal_abs_vertical_m"] for row in valid],
    )
    high_low_delta = _group_median_delta(
        valid, "abs_pitch_command_jump_rad", "terminal_abs_vertical_m"
    )
    closure_or_worsening = bool(
        (worsening_median is not None and worsening_median >= HANDOFF_WORSENING_MIN_M)
        or (closure_drop_median is not None and closure_drop_median >= HANDOFF_CLOSURE_DROP_MIN)
    )
    criteria = {
        "enough_valid_handoff_seeds": len(valid) >= MIN_VERTICAL_MISS_SEEDS,
        "pitch_jump_median_ge_0_05": bool(
            jump_median is not None and jump_median >= HANDOFF_PITCH_JUMP_MIN_RAD
        ),
        "worsening_or_closure_drop": closure_or_worsening,
        "worsening_terminal_rho_ge_0_30": bool(
            rho is not None and rho >= HANDOFF_TERMINAL_RHO_MIN
        ),
        "high_low_terminal_delta_ge_0_25m": bool(
            high_low_delta is not None
            and high_low_delta >= HANDOFF_HIGH_LOW_TERMINAL_DELTA_MIN_M
        ),
    }
    return {
        "vertical_miss_seed_count": len(miss),
        "valid_handoff_seed_count": len(valid),
        "pitch_jump_median_rad": jump_median,
        "vertical_worsening_median_m": worsening_median,
        "closure_drop_median": closure_drop_median,
        "worsening_terminal_rho": rho,
        "high_minus_low_terminal_abs_m": high_low_delta,
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


def _overall_verdict(blocks: list[dict[str, Any]], invalid: bool) -> dict[str, Any]:
    h_initial = _hypothesis_verdict(blocks, "h_initial")
    h_track = _hypothesis_verdict(blocks, "h_track")
    h_handoff = _hypothesis_verdict(blocks, "h_handoff")
    if invalid:
        return {
            "H_INITIAL": h_initial,
            "H_TRACK": h_track,
            "H_HANDOFF": h_handoff,
            "decision": "INVALID_DIAGNOSTIC",
            "recommendation": "修复数据、分段或 seed 完整性后重跑，不解释机制。",
        }
    supported = {
        name
        for name, verdict in (
            ("H_INITIAL", h_initial),
            ("H_TRACK", h_track),
            ("H_HANDOFF", h_handoff),
        )
        if verdict == "SUPPORTED"
    }
    if supported == {"H_INITIAL"}:
        decision = "INITIAL_OFFSET_DOMINANT"
        recommendation = "预注册初始状态/起始参考几何诊断；不要先改 pitch 内环增益。"
    elif supported == {"H_TRACK"}:
        decision = "PITCH_TRACKING_SUPPORTED"
        recommendation = "预注册固定参数的 pitch 内环跟踪诊断；不要重开 authority/reward 扫描。"
    elif supported == {"H_HANDOFF"}:
        decision = "HANDOFF_TRANSITION_SUPPORTED"
        recommendation = "预注册交接平滑/参考切换单因素诊断；不要延长 terminal window。"
    elif len(supported) >= 2:
        decision = "MULTI_FACTOR"
        recommendation = "至少两个机制达到预注册判据；拆成最小单因素候选。"
    else:
        decision = "NO_SUPPORTED_MECHANISM"
        recommendation = "停止该早期垂直机制线，不预注册控制器候选。"
    return {
        "H_INITIAL": h_initial,
        "H_TRACK": h_track,
        "H_HANDOFF": h_handoff,
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
        }
        scenario_blocks: dict[str, Any] = {}
        for mode, (summary, records, _trace) in loaded.items():
            block = {
                "scenario": scenario,
                "mode": mode,
                "source": str(formal_root / scenario / mode),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "aggregate": _aggregate(records),
                "h_initial": _h_initial(records, radius),
                "h_track": _h_track(records),
                "h_handoff": _h_handoff(records),
                "records": records,
            }
            if block["h_initial"]["vertical_miss_seed_count"] < MIN_VERTICAL_MISS_SEEDS:
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
        "# Goal35 早期垂直跟踪/交接诊断",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block verdicts",
        "",
        "| scenario | mode | miss n | H-INITIAL | H-TRACK | H-HANDOFF |",
        "|---|---|---:|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            miss = block["h_initial"]["vertical_miss_seed_count"]
            verdicts = [
                block[key]["passes_all"] for key in ("h_initial", "h_track", "h_handoff")
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
            f"- H_INITIAL: `{result['overall']['H_INITIAL']}`",
            f"- H_TRACK: `{result['overall']['H_TRACK']}`",
            f"- H_HANDOFF: `{result['overall']['H_HANDOFF']}`",
            f"- decision: `{result['overall']['decision']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## Key metrics",
            "",
            "| scenario | mode | init rho | init outside | ratio | cmd pitch | track err | ref sign | handoff jump | worsen |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            i = block["h_initial"]
            t = block["h_track"]
            h = block["h_handoff"]
            lines.append(
                f"| {scenario} | {mode} | {_fmt(i['initial_terminal_rho'])} | "
                f"{_fmt(i['initial_outside_radius_fraction'])} | {_fmt(i['initial_terminal_ratio'])} | "
                f"{_fmt(t['command_pitch_median_rad'])} | {_fmt(t['tracking_error_median_rad'])} | "
                f"{_fmt(t['reference_sign_match_median'])} | {_fmt(h['pitch_jump_median_rad'])} | "
                f"{_fmt(h['vertical_worsening_median_m'])} |"
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
    json_path = args.output / "early_vertical_diagnostic.json"
    md_path = args.output / "early_vertical_diagnostic.md"
    csv_path = args.output / "early_vertical_per_seed.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(result), encoding="utf-8")
    _write_csv(csv_path, result)
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
