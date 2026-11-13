#!/usr/bin/env python3
"""Goal115: read-only joint pitch handoff audit.

The script compares a fresh instrumented eval trace with the frozen Goal107
trace, then separates guidance-target demand, command-rate filtering, plant
tracking, and frozen-BC residual direction.  It never loads a policy or steps
an environment.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal115_joint_pitch_handoff.v1"
PRIMARY_SCENARIO = "strict125"
PRIMARY_PROFILE = "C0"
PRIMARY_SEEDS = (25, 26, 27, 29, 31, 33, 36, 49, 50, 51, 54)
MIXED_SEEDS = (25, 27, 29, 31, 36, 49, 50, 54)
START_SHELL_M = 3.0
PRIMARY_SHELL_M = 2.0
POST_ENTRY_WINDOW_S = 1.0
PITCH_RATE_LIMIT_RAD_S = math.radians(15.0)
CLIP_FRACTION_THRESHOLD = 0.25
TARGET_EXCESS_THRESHOLD = 0.05
TRACKING_DEFICIT_THRESHOLD = 0.05
MAJORITY_N = 6
MIN_EVALUABLE_N = 9
CLASSES = (
    "REFERENCE_FILTER_LIMITED",
    "PLANT_RESPONSE_LIMITED",
    "RESIDUAL_DIRECTION_LIMITED",
    "MIXED_HANDOFF_CONTINUE",
    "NOT_EVALUABLE",
)
PARITY_FIELDS = (
    "time_s",
    "goal_distance_eta_m",
    "eta",
    "nu",
    "command",
    "actuator_total_physical",
    "terminal_kind",
)


class AuditError(RuntimeError):
    """Raised when the frozen trace or instrumented trace is invalid."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal107_terminal_reward_credit_20261111"),
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal115_joint_pitch_handoff_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal115_joint_pitch_handoff_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal115_joint_pitch_handoff_20261111/"
            "phase0_summary.md"
        ),
    )
    parser.add_argument(
        "--parity-atol",
        type=float,
        default=1e-10,
    )
    return parser.parse_args(argv)


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AuditError(f"missing trace: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_rows(trace: dict[str, np.ndarray], seed: int) -> np.ndarray:
    if "episode_seed" not in trace:
        raise AuditError("trace is missing episode_seed")
    rows = np.flatnonzero(np.asarray(trace["episode_seed"], dtype=np.int64) == int(seed))
    if rows.size == 0:
        raise AuditError(f"trace has no rows for seed {seed}")
    return rows


def _first_entry(distance: np.ndarray, threshold_m: float) -> int | None:
    hits = np.flatnonzero(np.asarray(distance, dtype=np.float64) <= float(threshold_m))
    return int(hits[0]) if hits.size else None


def _window(trace: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, int, int] | None:
    rows = _seed_rows(trace, seed)
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)[rows]
    start = _first_entry(distance, START_SHELL_M)
    anchor = _first_entry(distance, PRIMARY_SHELL_M)
    if start is None or anchor is None:
        return None
    time_s = np.asarray(trace["time_s"], dtype=np.float64)[rows]
    stop = int(np.searchsorted(time_s, time_s[anchor] + POST_ENTRY_WINDOW_S, side="left"))
    stop = max(stop, anchor + 3)
    stop = min(stop, len(rows))
    if stop - start < 4:
        return None
    return rows[start:stop], int(rows[anchor]), start


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # Average ranks for ties without requiring scipy.
    unique, inverse = np.unique(values, return_inverse=True)
    for index in range(unique.size):
        selected = inverse == index
        if selected.sum() > 1:
            ranks[selected] = float(np.mean(ranks[selected]))
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 4:
        return None
    ranked_left = _rankdata(left[mask])
    ranked_right = _rankdata(right[mask])
    if float(np.std(ranked_left)) == 0.0 or float(np.std(ranked_right)) == 0.0:
        return None
    return float(np.corrcoef(ranked_left, ranked_right)[0, 1])


def _episode_metrics(
    trace: dict[str, np.ndarray],
    seed: int,
    *,
    pitch_rate_limit: float,
) -> dict[str, Any] | None:
    window = _window(trace, seed)
    if window is None:
        return None
    rows, anchor_row, _start_row = window
    time_s = np.asarray(trace["time_s"], dtype=np.float64)[rows]
    target_pitch = np.asarray(trace["guidance_target_pitch_rad"], dtype=np.float64)[rows]
    limited_pitch = np.asarray(trace["guidance_limited_pitch_rad"], dtype=np.float64)[rows]
    actual_pitch = np.asarray(trace["eta"], dtype=np.float64)[rows, 4]
    actual_rate = np.asarray(trace["nu"], dtype=np.float64)[rows, 4]
    target_rate = np.gradient(target_pitch, time_s)
    limited_rate = np.gradient(limited_pitch, time_s)
    pitch_error = np.asarray(trace["pitch_error_rad"], dtype=np.float64)[rows]
    clip = np.asarray(trace["guidance_pitch_rate_clipped"], dtype=bool)[rows]
    stern = np.asarray(trace["actuator_total_physical"], dtype=np.float64)[rows, 1]
    residual_stern = np.asarray(trace["actuator_residual_applied"], dtype=np.float64)[rows, 1]
    stern_norm = np.abs(stern) / math.radians(20.0)
    target_excess = np.maximum(np.abs(target_rate) - float(pitch_rate_limit), 0.0)
    tracking_deficit = np.maximum(np.abs(limited_rate) - np.abs(actual_rate), 0.0)
    finite = (
        np.isfinite(target_rate).all()
        and np.isfinite(limited_rate).all()
        and np.isfinite(actual_rate).all()
        and np.isfinite(pitch_error).all()
        and np.isfinite(residual_stern).all()
    )
    if not finite:
        return None
    return {
        "seed": int(seed),
        "frames": int(rows.size),
        "anchor_row": int(anchor_row),
        "anchor_time_s": float(time_s[int(np.flatnonzero(rows == anchor_row)[0])]),
        "clip_fraction": float(np.mean(clip)),
        "median_target_excess_rate": float(np.median(target_excess)),
        "median_limited_rate": float(np.median(np.abs(limited_rate))),
        "median_actual_rate": float(np.median(np.abs(actual_rate))),
        "median_tracking_deficit": float(np.median(tracking_deficit)),
        "limited_at_limit_fraction": float(
            np.mean(np.abs(limited_rate) >= 0.95 * float(pitch_rate_limit))
        ),
        "target_rate_p95": float(np.percentile(np.abs(target_rate), 95)),
        "limited_rate_p95": float(np.percentile(np.abs(limited_rate), 95)),
        "actual_rate_p95": float(np.percentile(np.abs(actual_rate), 95)),
        "pitch_error_abs_median": float(np.median(np.abs(pitch_error))),
        "stern_norm_p95": float(np.percentile(stern_norm, 95)),
        "residual_stern_p95_abs": float(np.percentile(np.abs(residual_stern), 95)),
        "residual_pitch_error_spearman": _spearman(residual_stern, pitch_error),
        "pitch_target": target_pitch.tolist(),
        "pitch_limited": limited_pitch.tolist(),
        "pitch_actual": actual_pitch.tolist(),
        "time_s": time_s.tolist(),
    }


def _parity(
    old_trace: dict[str, np.ndarray],
    new_trace: dict[str, np.ndarray],
    *,
    atol: float,
) -> dict[str, Any]:
    missing = [name for name in PARITY_FIELDS if name not in old_trace or name not in new_trace]
    if missing:
        raise AuditError(f"parity traces missing fields: {missing}")
    failures: list[dict[str, Any]] = []
    seeds = sorted(set(np.asarray(old_trace["episode_seed"], dtype=np.int64)))
    for seed in seeds:
        old_rows = _seed_rows(old_trace, int(seed))
        new_rows = _seed_rows(new_trace, int(seed))
        if old_rows.size != new_rows.size:
            failures.append(
                {
                    "seed": int(seed),
                    "reason": "row_count",
                    "old": int(old_rows.size),
                    "new": int(new_rows.size),
                }
            )
            continue
        for name in PARITY_FIELDS:
            old_value = np.asarray(old_trace[name])[old_rows]
            new_value = np.asarray(new_trace[name])[new_rows]
            if name == "terminal_kind":
                equal = np.array_equal(old_value, new_value)
            else:
                equal = bool(np.allclose(old_value, new_value, rtol=0.0, atol=float(atol)))
            if not equal:
                diff = (
                    float(np.max(np.abs(old_value.astype(np.float64) - new_value.astype(np.float64))))
                    if np.issubdtype(old_value.dtype, np.number)
                    else None
                )
                failures.append(
                    {
                        "seed": int(seed),
                        "field": name,
                        "max_abs_diff": diff,
                    }
                )
    return {
        "pass": not failures,
        "atol": float(atol),
        "seeds": [int(seed) for seed in seeds],
        "failures": failures,
    }


def _success_seeds(summary: dict[str, Any]) -> list[int]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        raise AuditError("summary is missing episodes")
    return [int(row["seed"]) for row in episodes if bool(row.get("success"))]


def _classify(metrics: dict[str, Any], *, reference_sign: float | None) -> str:
    clip_fraction = float(metrics["clip_fraction"])
    target_excess = float(metrics["median_target_excess_rate"])
    tracking_deficit = float(metrics["median_tracking_deficit"])
    limited_at_limit = float(metrics["limited_at_limit_fraction"])
    corr = metrics["residual_pitch_error_spearman"]
    filter_limit = (
        clip_fraction >= CLIP_FRACTION_THRESHOLD
        and target_excess >= TARGET_EXCESS_THRESHOLD
        and limited_at_limit >= 0.50
    )
    plant_limit = tracking_deficit >= TRACKING_DEFICIT_THRESHOLD and not filter_limit
    if filter_limit:
        return "REFERENCE_FILTER_LIMITED"
    if plant_limit:
        return "PLANT_RESPONSE_LIMITED"
    residual_aligned = (
        corr is not None
        and reference_sign is not None
        and float(corr) * float(reference_sign) > 0.0
        and abs(float(corr)) >= 0.10
    )
    if not residual_aligned:
        return "RESIDUAL_DIRECTION_LIMITED"
    return "MIXED_HANDOFF_CONTINUE"


def _gate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in records if row["classification"] != "NOT_EVALUABLE"]
    n = len(evaluable)
    counts = {name: sum(1 for row in evaluable if row["classification"] == name) for name in CLASSES}
    if n < MIN_EVALUABLE_N:
        status = "INSUFFICIENT_EVALUABLE"
    elif counts["REFERENCE_FILTER_LIMITED"] >= MAJORITY_N:
        status = "REFERENCE_FILTER_LIMITED"
    elif counts["PLANT_RESPONSE_LIMITED"] >= MAJORITY_N:
        status = "PLANT_RESPONSE_LIMITED"
    elif counts["RESIDUAL_DIRECTION_LIMITED"] >= MAJORITY_N:
        status = "RESIDUAL_DIRECTION_LIMITED"
    elif counts["MIXED_HANDOFF_CONTINUE"] >= MAJORITY_N:
        status = "MIXED_HANDOFF_CONTINUE"
    else:
        status = "JOINT_HANDOFF_AMBIGUOUS"
    return {
        "status": status,
        "primary_n": len(records),
        "evaluable_n": n,
        "minimum_evaluable_n": MIN_EVALUABLE_N,
        "majority_n": MAJORITY_N,
        "class_counts": counts,
        "class_fractions": {name: (counts[name] / n if n else math.nan) for name in CLASSES},
        "H115_A_supported": counts["REFERENCE_FILTER_LIMITED"] >= MAJORITY_N,
        "H115_B_supported": counts["PLANT_RESPONSE_LIMITED"] >= MAJORITY_N,
        "H115_C_supported": counts["RESIDUAL_DIRECTION_LIMITED"] >= MAJORITY_N,
        "H115_D_supported": status in {"MIXED_HANDOFF_CONTINUE", "JOINT_HANDOFF_AMBIGUOUS"},
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    records = payload["records"]
    lines = [
        "# Goal115 Phase 0：joint pitch handoff 审计",
        "",
        "本文件由 `scripts/analyze_pc_goal115_joint_pitch_handoff.py` 生成；只读 trace，不训练。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- primary/evaluable: `{gate['primary_n']}/{gate['evaluable_n']}`",
        f"- parity: `{payload['parity']['pass']}`",
        f"- pitch rate limit: `{payload['pitch_rate_limit_rad_s']:.6f} rad/s`",
        "",
        "## 逐 seed 机制",
        "",
        "| seed | class | clip frac | target excess | tracking deficit | limited at limit | BC residual corr |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        metrics = row.get("metrics") or {}
        corr = metrics.get("residual_pitch_error_spearman")
        corr_text = "—" if corr is None else f"{float(corr):+.3f}"
        lines.append(
            "| {seed} | `{cls}` | {clip} | {excess} | {deficit} | {limit} | {corr} |".format(
                seed=row["seed"],
                cls=row["classification"],
                clip=f"{metrics.get('clip_fraction', float('nan')):.3f}",
                excess=f"{metrics.get('median_target_excess_rate', float('nan')):.3f}",
                deficit=f"{metrics.get('median_tracking_deficit', float('nan')):.3f}",
                limit=f"{metrics.get('limited_at_limit_fraction', float('nan')):.3f}",
                corr=corr_text,
            )
        )
    lines.extend(
        [
            "",
            "## 分类计数",
            "",
            "| class | count |",
            "|---|---:|",
        ]
    )
    for name in CLASSES:
        lines.append(f"| `{name}` | {gate['class_counts'][name]} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    old_run = args.old_root / f"P_D3_{PRIMARY_SCENARIO}_{PRIMARY_PROFILE}"
    new_run = args.new_root / f"P_D3_{PRIMARY_SCENARIO}_{PRIMARY_PROFILE}"
    old_trace = _load_trace(old_run / "trace.npz")
    new_trace = _load_trace(new_run / "trace.npz")
    old_summary = _load_summary(old_run / "summary.json")
    new_summary = _load_summary(new_run / "summary.json")

    parity = _parity(old_trace, new_trace, atol=float(args.parity_atol))
    if not parity["pass"]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "INVALID_INSTRUMENTATION_PARITY",
            "parity": parity,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        args.markdown.write_text(
            "# Goal115 Phase 0\n\nINVALID_INSTRUMENTATION_PARITY\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "INVALID_INSTRUMENTATION_PARITY"}, ensure_ascii=False))
        return 1

    success_controls = _success_seeds(old_summary)
    control_correlations: list[float] = []
    for seed in success_controls:
        metrics = _episode_metrics(
            new_trace,
            seed,
            pitch_rate_limit=float(PITCH_RATE_LIMIT_RAD_S),
        )
        if metrics is not None:
            corr = metrics.get("residual_pitch_error_spearman")
            if corr is not None:
                control_correlations.append(float(corr))
    reference_sign = (
        float(np.sign(np.median(control_correlations))) if control_correlations else None
    )

    records: list[dict[str, Any]] = []
    for seed in PRIMARY_SEEDS:
        metrics = _episode_metrics(
            new_trace,
            seed,
            pitch_rate_limit=float(PITCH_RATE_LIMIT_RAD_S),
        )
        if metrics is None:
            records.append({"seed": int(seed), "classification": "NOT_EVALUABLE", "metrics": None})
            continue
        records.append(
            {
                "seed": int(seed),
                "classification": _classify(metrics, reference_sign=reference_sign),
                "metrics": metrics,
            }
        )
    gate = _gate(records)
    mixed_counts = {
        name: sum(
            1
            for row in records
            if int(row["seed"]) in MIXED_SEEDS and row["classification"] == name
        )
        for name in CLASSES
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": gate["status"],
        "primary_scenario": PRIMARY_SCENARIO,
        "primary_profile": PRIMARY_PROFILE,
        "primary_seeds": list(PRIMARY_SEEDS),
        "mixed_seeds": list(MIXED_SEEDS),
        "pitch_rate_limit_rad_s": float(PITCH_RATE_LIMIT_RAD_S),
        "parity": parity,
        "success_controls": success_controls,
        "control_residual_correlations": control_correlations,
        "control_residual_reference_sign": reference_sign,
        "gate": gate,
        "mixed_subset_counts": mixed_counts,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": gate["status"], "counts": gate["class_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
