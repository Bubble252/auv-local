#!/usr/bin/env python3
"""Analyze Goal116's frozen rate-limit sweep without stepping an environment."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pc_goal110_terminal_closure_onset import _trace_by_seed
from analyze_pc_goal111_preentry_closest_approach import _episode_metrics
from analyze_pc_goal115_joint_pitch_handoff import _load_trace, _parity

SCHEMA_VERSION = "auv_pc_goal116_reference_filter_screen.v1"
ARMS = ("R050", "R100", "R200", "ROFF")
CONTROL = "R100"
HIGH_ARMS = ("R200", "ROFF")
LOW_ARM = "R050"
ORIGINAL_FAILURE_SEEDS = (25, 26, 27, 29, 31, 33, 36, 49, 50, 51, 54)
EXPECTED_CHECKPOINT_SHA = "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616"
EXPECTED_SCALE = 2.0
EXPECTED_SEEDS = tuple(range(20, 60))
MIN_EFFICIENCY_GAIN = 3
SAFETY_SLACK = 1
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
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal116_reference_filter_20261111"),
    )
    parser.add_argument(
        "--parity-trace",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal115_joint_pitch_handoff_20261111/"
            "P_D3_strict125_C0/trace.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal116_reference_filter_20261111/phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal116_reference_filter_20261111/phase0_summary.md"
        ),
    )
    parser.add_argument("--parity-atol", type=float, default=1e-10)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _episodes_by_seed(summary: dict[str, Any], *, arm: str) -> dict[int, dict[str, Any]]:
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise AuditError(f"{arm}: checkpoint SHA mismatch")
    if abs(float(summary.get("actuator_residual_scale_override", math.nan)) - EXPECTED_SCALE) > 1e-12:
        raise AuditError(f"{arm}: eval scale mismatch")
    if summary.get("action_mode") != "frozen-base":
        raise AuditError(f"{arm}: action_mode is not frozen-base")
    gains = summary.get("actuator_residual_channel_gains")
    if not isinstance(gains, list) or len(gains) != 3:
        raise AuditError(f"{arm}: channel gains missing")
    if any(abs(float(gain) - 1.0) > 1e-12 for gain in gains):
        raise AuditError(f"{arm}: channel gains are not 1,1,1")
    rows = summary.get("episodes")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_SEEDS):
        raise AuditError(f"{arm}: expected {len(EXPECTED_SEEDS)} episodes")
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{arm}: malformed episode row")
        seed = int(row["seed"])
        if seed in by_seed:
            raise AuditError(f"{arm}: duplicate seed {seed}")
        by_seed[seed] = row
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise AuditError(f"{arm}: seeds must be exactly {EXPECTED_SEEDS[0]}-{EXPECTED_SEEDS[-1]}")
    return by_seed


def _distance_metrics(trace: dict[str, np.ndarray], *, arm: str) -> dict[int, float]:
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    result: dict[int, float] = {}
    for seed in EXPECTED_SEEDS:
        rows = np.flatnonzero(seeds == seed)
        if rows.size == 0:
            raise AuditError(f"{arm}: trace has no rows for seed {seed}")
        values = distance[rows]
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise AuditError(f"{arm}: invalid distance for seed {seed}")
        result[seed] = float(np.min(values))
    return result


def _quantile(values: Sequence[float], q: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(finite, q)) if finite else math.nan


def _arm_summary(
    *,
    arm: str,
    episodes: dict[int, dict[str, Any]],
    episode_traces: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    success = sum(1 for row in episodes.values() if bool(row["success"]))
    timeout = sum(1 for row in episodes.values() if bool(row.get("timeout", False)))
    safety = sum(1 for row in episodes.values() if bool(row.get("safety", False)))
    plane_miss = sum(1 for row in episodes.values() if bool(row.get("goal_plane_miss", False)))
    other = len(EXPECTED_SEEDS) - success - timeout - safety - plane_miss
    distances = [float(np.min(episode_traces[seed]["goal_distance_eta_m"])) for seed in EXPECTED_SEEDS]
    subset: list[dict[str, Any]] = []
    for seed in ORIGINAL_FAILURE_SEEDS:
        row = episodes[seed]
        metrics = _episode_metrics(
            episode_traces[seed],
            seed=seed,
            success=bool(row["success"]),
        )
        subset.append(
            {
                "seed": seed,
                "success": bool(row["success"]),
                "terminal": {
                    "timeout": bool(row.get("timeout", False)),
                    "safety": bool(row.get("safety", False)),
                    "goal_plane_miss": bool(row.get("goal_plane_miss", False)),
                },
                "min_distance_m": float(metrics["min_distance_m"]),
                "goal111_classification": metrics["classification"],
            }
        )
    return {
        "arm": arm,
        "success_count": int(success),
        "timeout_count": int(timeout),
        "safety_count": int(safety),
        "goal_plane_miss_count": int(plane_miss),
        "other_terminal_count": int(other),
        "closest_approach_median_m": _quantile(distances, 0.50),
        "closest_approach_p05_m": _quantile(distances, 0.05),
        "original_failure_subset": subset,
        "original_failure_success_count": sum(1 for row in subset if row["success"]),
        "original_failure_geometry_bad_count": sum(
            1 for row in subset if row["goal111_classification"] == "PREENTRY_GEOMETRY_BAD"
        ),
        "original_failure_min_distance_median_m": _quantile(
            [row["min_distance_m"] for row in subset], 0.50
        ),
    }


def _decide(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    control = arms[CONTROL]
    control_unsafe = control["timeout_count"] + control["safety_count"]
    evaluations: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        row = arms[arm]
        efficiency = row["success_count"] >= control["success_count"] + MIN_EFFICIENCY_GAIN
        safety_ok = row["timeout_count"] + row["safety_count"] <= control_unsafe + SAFETY_SLACK
        subset_gain = (
            row["original_failure_success_count"]
            >= control["original_failure_success_count"] + 2
        )
        evaluations[arm] = {
            "success_delta": row["success_count"] - control["success_count"],
            "safety_delta": (
                row["timeout_count"]
                + row["safety_count"]
                - control["timeout_count"]
                - control["safety_count"]
            ),
            "efficiency_gain": bool(efficiency),
            "safety_guard": bool(safety_ok),
            "subset_gain": bool(subset_gain),
            "qualifies": bool(efficiency and safety_ok and subset_gain),
        }
    unsafe_gain = [
        arm
        for arm in ARMS
        if arm != CONTROL
        and arms[arm]["success_count"] > control["success_count"]
        and not evaluations[arm]["safety_guard"]
    ]
    high_qual = [arm for arm in HIGH_ARMS if evaluations[arm]["qualifies"]]
    low_qual = [LOW_ARM] if evaluations[LOW_ARM]["qualifies"] else []
    if unsafe_gain:
        status = "HARM"
        reason = "success_gain_with_safety_or_timeout_degradation"
    elif high_qual and low_qual:
        status = "NONMONOTONIC_FILTER_RESPONSE"
        reason = "both_higher_and_lower_limits_qualify"
    elif high_qual:
        status = "FILTER_CAUSAL_UP"
        reason = "higher_limit_or_off_qualifies"
    elif low_qual:
        status = "REFERENCE_NOISE_DOMINANT"
        reason = "lower_limit_qualifies"
    elif all(abs(evaluations[arm]["success_delta"]) <= 2 for arm in ARMS if arm != CONTROL):
        status = "FILTER_NONCAUSAL"
        reason = "all_success_deltas_within_two"
    else:
        status = "INCONCLUSIVE"
        reason = "no_arm_qualifies_and_not_all_within_noise"
    return {
        "status": status,
        "reason": reason,
        "control": CONTROL,
        "min_efficiency_gain": MIN_EFFICIENCY_GAIN,
        "safety_slack": SAFETY_SLACK,
        "arms": evaluations,
    }


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal116 Phase 0：guidance pitch rate-limit 因果筛选",
        "",
        "本文件由 `scripts/analyze_pc_goal116_reference_filter_screen.py` 生成；只读 trace，不训练。",
        "",
        f"- Gate: `{payload['gate']['status']}`",
        f"- reason: `{payload['gate']['reason']}`",
        f"- R100 parity: `{payload['parity']['pass']}`",
        f"- seeds: `20–59`，40 episodes/arm",
        "",
        "## Arm 结果",
        "",
        "| arm | success | timeout | safety | plane miss | closest median | closest p05 | 11-seed success | 11-seed bad geometry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["arms"][arm]
        lines.append(
            f"| `{arm}` | {row['success_count']}/40 | {row['timeout_count']} | "
            f"{row['safety_count']} | {row['goal_plane_miss_count']} | "
            f"{row['closest_approach_median_m']:.3f} | {row['closest_approach_p05_m']:.3f} | "
            f"{row['original_failure_success_count']}/11 | "
            f"{row['original_failure_geometry_bad_count']}/11 |"
        )
    lines.extend(["", "## 与 R100 的差异", "", "| arm | Δsuccess | Δtimeout+safety | efficiency | safety | subset | qualifies |", "|---|---:|---:|---:|---:|---:|---:|"])
    for arm in ARMS:
        ev = payload["gate"]["arms"][arm]
        lines.append(
            f"| `{arm}` | {ev['success_delta']:+d} | {ev['safety_delta']:+d} | "
            f"{ev['efficiency_gain']} | {ev['safety_guard']} | {ev['subset_gain']} | {ev['qualifies']} |"
        )
    lines.extend(["", "## 原 11 子集", "", "| seed | arm success | timeout | safety | plane miss | min distance | Goal111 |", "|---:|---|---:|---:|---:|---:|---|"])
    for seed in ORIGINAL_FAILURE_SEEDS:
        for arm in ARMS:
            row = next(item for item in payload["arms"][arm]["original_failure_subset"] if item["seed"] == seed)
            lines.append(
                f"| {seed} | `{arm}`={row['success']} | {row['terminal']['timeout']} | "
                f"{row['terminal']['safety']} | {row['terminal']['goal_plane_miss']} | "
                f"{row['min_distance_m']:.3f} | {row['goal111_classification']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = _read_json(args.root / "run_manifest.json")
    if manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise AuditError("manifest checkpoint SHA mismatch")
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run_dir = args.root / arm
        summary = _read_json(run_dir / "summary.json")
        episodes = _episodes_by_seed(summary, arm=arm)
        trace = _trace_by_seed(run_dir)
        if tuple(sorted(trace)) != EXPECTED_SEEDS:
            raise AuditError(f"{arm}: trace seeds do not cover 20-59")
        arms[arm] = _arm_summary(arm=arm, episodes=episodes, episode_traces=trace)
    old_trace = _load_trace(args.parity_trace)
    new_trace = _load_trace(args.root / CONTROL / "trace.npz")
    parity = _parity(old_trace, new_trace, atol=float(args.parity_atol))
    if not parity["pass"]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "INVALID",
            "parity": parity,
        }
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "OK",
            "parity": parity,
            "arms": arms,
            "gate": _decide(arms),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_render(payload), encoding="utf-8")
    print(json.dumps({"status": payload.get("gate", {}).get("status", payload["status"]), "parity": parity["pass"]}, ensure_ascii=False))
    return 0 if parity["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
