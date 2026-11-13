#!/usr/bin/env python3
"""Audit closest approach and the pre-entry window for frozen Goal107 traces.

Goal111 is read-only.  It loads no policy, starts no rollout, changes no reward
or termination contract, and never reads held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from analyze_pc_goal110_terminal_closure_onset import (
    ARM,
    DEV_SEEDS,
    EXPECTED_CHECKPOINT_SHA,
    EXPECTED_EVAL_SCALE,
    EXPECTED_GAINS,
    AuditError,
    _load_episodes,
    _read_json,
    _trace_by_seed,
)

SCHEMA_VERSION = "auv_pc_goal111_closest_approach_preentry.v1"
SCENARIOS = ("default", "strict125")
PROFILES = ("C0", "U050")
SHELLS_M = (3.0, 2.5, 2.0, 1.5, 1.25)
PRIMARY_SHELL_M = 2.0
STALL_CLOSURE_MPS = 0.20
VERTICAL_MISS_M = 1.0
PITCH_ERROR_RAD = 0.45
HEADING_ERROR_RAD = 0.35
PREENTRY_WINDOW_S = 1.0
GEOMETRY_MEDIAN_WINDOW_STEPS = 5
MAJORITY_FRACTION = 0.50
CLASSES = (
    "APPROACH_DEFICIT",
    "PREENTRY_GEOMETRY_BAD",
    "ENTRY_GEOMETRY_OK_LATE_FAILURE",
)
STATUS_BY_CLASS = {
    "APPROACH_DEFICIT": "APPROACH_DEFICIT_MAJORITY",
    "PREENTRY_GEOMETRY_BAD": "PREENTRY_GEOMETRY_BAD_MAJORITY",
    "ENTRY_GEOMETRY_OK_LATE_FAILURE": "LATE_TERMINAL_FAILURE_MAJORITY",
}
REQUIRED_TRACE_FIELDS = (
    "time_s",
    "goal_distance_eta_m",
    "eta",
    "goal_ned_m",
    "heading_error_rad",
    "pitch_error_rad",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal107_terminal_reward_credit_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal111_closest_approach_preentry_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal111_closest_approach_preentry_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _finite_median(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else math.nan


def _finite_quantile(values: Sequence[float] | np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(np.quantile(finite, quantile)) if finite.size else math.nan


def _quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    return {
        f"p{int(round(quantile * 100)):02d}": _finite_quantile(values, quantile)
        for quantile in (0.05, 0.25, 0.50, 0.75, 0.95)
    }


def _first_entry(distance: np.ndarray, shell_m: float) -> int | None:
    indices = np.flatnonzero(np.asarray(distance, dtype=np.float64) <= shell_m)
    return int(indices[0]) if indices.size else None


def _rolling_median_at(values: np.ndarray, index: int, window: int = 5) -> float:
    if window <= 0:
        raise ValueError("window must be positive")
    if index < 0 or index >= values.size:
        raise IndexError("index is outside the trace")
    start = max(0, index - window + 1)
    return _finite_median(values[start : index + 1])


def _window_indices(index: int, steps: int) -> np.ndarray:
    if steps <= 0:
        raise ValueError("window steps must be positive")
    return np.arange(max(0, index - steps + 1), index + 1, dtype=np.int64)


def _closure_speed(distance: np.ndarray, dt_s: float) -> np.ndarray:
    closure = np.full(distance.size, np.nan, dtype=np.float64)
    closure[:-1] = (distance[:-1] - distance[1:]) / dt_s
    return closure


def _shell_metrics(
    *,
    distance: np.ndarray,
    time_s: np.ndarray,
    vertical: np.ndarray,
    heading: np.ndarray,
    pitch: np.ndarray,
    closure: np.ndarray,
    entry: int | None,
) -> dict[str, Any]:
    if entry is None:
        return {
            "entered": False,
            "entry_time_s": None,
            "distance_m": None,
            "vertical_miss_m": None,
            "abs_heading_error_rad": None,
            "abs_pitch_error_rad": None,
            "vertical_median5_m": None,
            "abs_heading_median5_rad": None,
            "abs_pitch_median5_rad": None,
            "vertical_max5_m": None,
            "abs_heading_max5_rad": None,
            "abs_pitch_max5_rad": None,
            "closure_mps": None,
            "geometry_bad_median5": None,
        }
    vertical_window = vertical[max(0, entry - 4) : entry + 1]
    heading_window = np.abs(heading[max(0, entry - 4) : entry + 1])
    pitch_window = np.abs(pitch[max(0, entry - 4) : entry + 1])
    vertical_median = _rolling_median_at(vertical, entry, GEOMETRY_MEDIAN_WINDOW_STEPS)
    heading_median = _rolling_median_at(
        np.abs(heading), entry, GEOMETRY_MEDIAN_WINDOW_STEPS
    )
    pitch_median = _rolling_median_at(
        np.abs(pitch), entry, GEOMETRY_MEDIAN_WINDOW_STEPS
    )
    geometry_bad = bool(
        vertical_median >= VERTICAL_MISS_M
        or heading_median >= HEADING_ERROR_RAD
        or pitch_median >= PITCH_ERROR_RAD
    )
    return {
        "entered": True,
        "entry_time_s": float(time_s[entry]),
        "distance_m": float(distance[entry]),
        "vertical_miss_m": float(vertical[entry]),
        "abs_heading_error_rad": float(abs(heading[entry])),
        "abs_pitch_error_rad": float(abs(pitch[entry])),
        "vertical_median5_m": vertical_median,
        "abs_heading_median5_rad": heading_median,
        "abs_pitch_median5_rad": pitch_median,
        "vertical_max5_m": float(np.nanmax(vertical_window)),
        "abs_heading_max5_rad": float(np.nanmax(heading_window)),
        "abs_pitch_max5_rad": float(np.nanmax(pitch_window)),
        "closure_mps": float(closure[entry]) if math.isfinite(closure[entry]) else None,
        "geometry_bad_median5": geometry_bad,
    }


def _classify_failure(metrics: dict[str, Any]) -> str:
    if float(metrics["min_distance_m"]) > PRIMARY_SHELL_M:
        return "APPROACH_DEFICIT"
    entry_geometry = metrics["shells"][f"{PRIMARY_SHELL_M:.1f}"]["geometry_bad_median5"]
    if entry_geometry is None:
        raise AuditError("entered 2.0 m but entry geometry is unavailable")
    return (
        "PREENTRY_GEOMETRY_BAD"
        if bool(entry_geometry)
        else "ENTRY_GEOMETRY_OK_LATE_FAILURE"
    )


def _episode_metrics(trace: dict[str, np.ndarray], *, seed: int, success: bool) -> dict[str, Any]:
    missing = [name for name in REQUIRED_TRACE_FIELDS if name not in trace]
    if missing:
        raise AuditError(f"seed={seed}: trace is missing fields {missing}")

    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)
    eta = np.asarray(trace["eta"], dtype=np.float64)
    goal = np.asarray(trace["goal_ned_m"], dtype=np.float64)
    heading = np.asarray(trace["heading_error_rad"], dtype=np.float64)
    pitch = np.asarray(trace["pitch_error_rad"], dtype=np.float64)

    if time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise AuditError(f"seed={seed}: trace is too short or has non-finite time")
    if distance.shape != time_s.shape or heading.shape != time_s.shape or pitch.shape != time_s.shape:
        raise AuditError(f"seed={seed}: trace scalar arrays are misaligned")
    if eta.shape != (time_s.size, 6) or goal.shape != (time_s.size, 3):
        raise AuditError(f"seed={seed}: eta/goal shapes are invalid")
    if not np.all(np.isfinite(distance)) or np.any(distance < 0.0):
        raise AuditError(f"seed={seed}: goal distance must be finite and non-negative")
    diffs = np.diff(time_s)
    dt_s = float(np.median(diffs))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise AuditError(f"seed={seed}: trace time must be strictly increasing")
    if not np.allclose(diffs, dt_s, rtol=0.05, atol=1e-6):
        raise AuditError(f"seed={seed}: trace time is not uniformly sampled")
    window_steps = max(1, int(round(PREENTRY_WINDOW_S / dt_s)))

    vertical = np.abs(eta[:, 2] - goal[:, 2])
    closure = _closure_speed(distance, dt_s)
    closest = int(np.argmin(distance))
    entry_by_shell = {f"{shell:.1f}": _first_entry(distance, shell) for shell in SHELLS_M}
    entry_primary = entry_by_shell[f"{PRIMARY_SHELL_M:.1f}"]
    anchor = entry_primary if entry_primary is not None else closest
    window = _window_indices(anchor, window_steps)
    window_closure = closure[window]
    finite_window_closure = window_closure[np.isfinite(window_closure)]
    nearest_closure = closure[closest]
    if not math.isfinite(nearest_closure) and closest > 0:
        nearest_closure = closure[closest - 1]

    metrics: dict[str, Any] = {
        "seed": seed,
        "success": bool(success),
        "frames": int(time_s.size),
        "min_distance_m": float(distance[closest]),
        "closest_index": closest,
        "time_to_closest_s": float(time_s[closest]),
        "final_distance_m": float(distance[-1]),
        "closest_vertical_miss_m": float(vertical[closest]),
        "closest_abs_heading_error_rad": float(abs(heading[closest])),
        "closest_abs_pitch_error_rad": float(abs(pitch[closest])),
        "closest_closure_mps": float(nearest_closure) if math.isfinite(nearest_closure) else None,
        "post_closest_net_closure_m": float(distance[closest] - distance[-1]),
        "preentry_window_anchor": "entry_2m" if entry_primary is not None else "closest",
        "preentry_window_s": float(time_s[anchor] - time_s[window[0]]),
        "preentry_min_closure_mps": (
            float(np.min(finite_window_closure)) if finite_window_closure.size else None
        ),
        "preentry_stall_fraction": (
            float(np.mean(finite_window_closure < STALL_CLOSURE_MPS))
            if finite_window_closure.size
            else None
        ),
        "shells": {
            key: _shell_metrics(
                distance=distance,
                time_s=time_s,
                vertical=vertical,
                heading=heading,
                pitch=pitch,
                closure=closure,
                entry=entry,
            )
            for key, entry in entry_by_shell.items()
        },
    }
    metrics["classification"] = None if success else _classify_failure(metrics)
    return metrics


def _descriptive_summary(records: Sequence[dict[str, Any]], *, success: bool) -> dict[str, Any]:
    selected = [row for row in records if bool(row["success"]) == success]
    entry_primary = [row for row in selected if row["shells"]["2.0"]["entered"]]
    entry_bad_median = [
        row
        for row in entry_primary
        if bool(row["shells"]["2.0"]["geometry_bad_median5"])
    ]
    entry_bad_single = [
        row
        for row in entry_primary
        if (
            float(row["shells"]["2.0"]["vertical_miss_m"]) >= VERTICAL_MISS_M
            or float(row["shells"]["2.0"]["abs_heading_error_rad"]) >= HEADING_ERROR_RAD
            or float(row["shells"]["2.0"]["abs_pitch_error_rad"]) >= PITCH_ERROR_RAD
        )
    ]
    return {
        "n": len(selected),
        "closest_distance_m": _quantiles([row["min_distance_m"] for row in selected]),
        "time_to_closest_s": _quantiles([row["time_to_closest_s"] for row in selected]),
        "final_distance_m": _quantiles([row["final_distance_m"] for row in selected]),
        "entered_2m_n": len(entry_primary),
        "entered_3m_n": sum(row["shells"]["3.0"]["entered"] for row in selected),
        "entry_geometry_bad_median5_n": len(entry_bad_median),
        "entry_geometry_bad_single_frame_n": len(entry_bad_single),
        "preentry_min_closure_mps": _quantiles(
            [row["preentry_min_closure_mps"] for row in selected]
        ),
        "preentry_stall_fraction": _quantiles(
            [row["preentry_stall_fraction"] for row in selected]
        ),
    }


def _profile_summary(run_dir: Path, *, scenario: str, profile: str) -> dict[str, Any]:
    episodes = _load_episodes(run_dir, scenario=scenario, profile=profile)
    traces = _trace_by_seed(run_dir)
    if tuple(sorted(traces)) != DEV_SEEDS:
        raise AuditError(f"{run_dir}: trace seeds must be exactly {DEV_SEEDS}")
    records = [
        _episode_metrics(
            traces[seed],
            seed=seed,
            success=bool(episodes[seed]["success"]),
        )
        for seed in DEV_SEEDS
    ]
    failures = [row for row in records if not row["success"]]
    counts = {
        name: sum(1 for row in failures if row["classification"] == name)
        for name in CLASSES
    }
    if sum(counts.values()) != len(failures):
        raise AuditError(f"{run_dir}: failure classes do not partition all failures")
    fractions = {
        name: counts[name] / len(failures) if failures else math.nan for name in CLASSES
    }
    return {
        "failure_n": len(failures),
        "classification_counts": counts,
        "classification_fractions": fractions,
        "failures": _descriptive_summary(records, success=False),
        "successes": _descriptive_summary(records, success=True),
        "records": records,
    }


def _classify_gate(primary: dict[str, Any]) -> dict[str, Any]:
    counts = primary["classification_counts"]
    fractions = primary["classification_fractions"]
    if primary["failure_n"] <= 0:
        raise AuditError("strict C0 has no failures to classify")
    maximum = max(float(value) for value in fractions.values())
    winners = [name for name in CLASSES if fractions[name] == maximum]
    if maximum < MAJORITY_FRACTION or len(winners) != 1:
        status = "MIXED_CLOSEST_APPROACH"
        reason = "no_unique_class_at_least_50pct"
        winner = None
    else:
        winner = winners[0]
        status = STATUS_BY_CLASS[winner]
        reason = "unique_class_at_least_50pct"
    return {
        "status": status,
        "reason": reason,
        "primary_scenario": "strict125",
        "primary_profile": "C0",
        "entry_shell_m": PRIMARY_SHELL_M,
        "shells_m": list(SHELLS_M),
        "preentry_window_s": PREENTRY_WINDOW_S,
        "geometry_median_window_steps": GEOMETRY_MEDIAN_WINDOW_STEPS,
        "majority_fraction": MAJORITY_FRACTION,
        "failure_n": primary["failure_n"],
        "counts": counts,
        "fractions": fractions,
        "winner": winner,
        "maximum_fraction": maximum,
        "H111_A_supported": winner == "APPROACH_DEFICIT",
        "H111_B_supported": winner == "PREENTRY_GEOMETRY_BAD",
        "H111_C_supported": winner == "ENTRY_GEOMETRY_OK_LATE_FAILURE",
        "H111_D_supported": status == "MIXED_CLOSEST_APPROACH",
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}" if math.isfinite(float(value)) else "—"
    return str(value)


def _render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Goal111 Phase 0：closest approach 与 pre-entry 窗口审计",
        "",
        "本文件由 `scripts/analyze_pc_goal111_preentry_closest_approach.py` 生成；",
        "只读 Goal107 dev40 trace，不训练、不改 reward/terminal/success、不打开 held-out。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- reason: `{gate['reason']}`",
        f"- winner: `{gate['winner'] or 'MIXED'}`",
        f"- strict C0 failure: `{gate['failure_n']}`",
        "",
        "## 分类",
        "",
        "| class | count | fraction |",
        "|---|---:|---:|",
    ]
    for name in CLASSES:
        lines.append(
            f"| `{name}` | {gate['counts'][name]} | "
            f"{_fmt(gate['fractions'][name])} |"
        )
    lines.extend(
        [
            "",
            "## 全部分组",
            "",
            "| scenario | profile | success | failure | approach | preentry | late | 未进入 2 m |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for profile in PROFILES:
            summary = payload["summary"][scenario][profile]
            counts = summary["classification_counts"]
            lines.append(
                f"| {scenario} | {profile} | {summary['successes']['n']}/40 | "
                f"{summary['failure_n']} | {counts['APPROACH_DEFICIT']} | "
                f"{counts['PREENTRY_GEOMETRY_BAD']} | "
                f"{counts['ENTRY_GEOMETRY_OK_LATE_FAILURE']} | "
                f"{summary['failures']['n'] - summary['failures']['entered_2m_n']} |"
            )
    primary = payload["summary"][gate["primary_scenario"]][gate["primary_profile"]]
    lines.extend(
        [
            "",
            "## strict C0 closest approach",
            "",
            "| group | n | closest p05 | closest p50 | closest p95 | entered 2 m | entry geom bad (median5) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (("failure", "failures"), ("success", "successes")):
        group = primary[key]
        lines.append(
            f"| {label} | {group['n']} | {_fmt(group['closest_distance_m']['p05'])} | "
            f"{_fmt(group['closest_distance_m']['p50'])} | "
            f"{_fmt(group['closest_distance_m']['p95'])} | "
            f"{group['entered_2m_n']} | {group['entry_geometry_bad_median5_n']} |"
        )
    lines.extend(
        [
            "",
            "## 判读边界",
            "",
            "- 分类是描述性互斥分区，不写成因果；",
            "- entry 几何使用 entry 帧及前 4 帧中位数，单帧计数只作对照；",
            "- pre-entry 窗口为进入 2 m 前 1.0 s；未进入 2 m 时以 closest 为锚点；",
            "- default/U050 不参与主 Gate 改判；",
            "- held-out `100–149` 未打开。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for profile in PROFILES:
            run_dir = args.root / f"{ARM}_{scenario}_{profile}"
            summary[scenario][profile] = _profile_summary(
                run_dir,
                scenario=scenario,
                profile=profile,
            )
    gate = _classify_gate(summary["strict125"]["C0"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "arm": ARM,
        "scenarios": list(SCENARIOS),
        "profiles": list(PROFILES),
        "shells_m": list(SHELLS_M),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "eval_scale": EXPECTED_EVAL_SCALE,
        "expected_channel_gains": {name: list(values) for name, values in EXPECTED_GAINS.items()},
        "gate": gate,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
