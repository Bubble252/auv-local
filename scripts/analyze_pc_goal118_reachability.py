#!/usr/bin/env python3
"""Read-only 2 m entry reachability screen for Goal118."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal118_approach_reachability.v1"
DEFAULT_ROOT = Path("artifacts/lowlevel/goal117_terminal_closure_attribution_20261111")
DEFAULT_RUN = DEFAULT_ROOT / "R100_RTRACE"
PRIMARY_SEEDS = (25, 26, 27, 29, 31, 33, 36, 49, 50, 51, 54)
EXPECTED_SEEDS = tuple(range(20, 60))
SHELL_M = 2.0
AMBIGUOUS_LOW = 1.0
AMBIGUOUS_HIGH = 1.2
TERMINAL_SHARE_THRESHOLD = 0.20


class AuditError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "goal118_reachability.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_ROOT / "goal118_reachability.md",
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


def _first_entry(distance: np.ndarray, threshold_m: float) -> int | None:
    hits = np.flatnonzero(np.asarray(distance, dtype=np.float64) <= float(threshold_m))
    return int(hits[0]) if hits.size else None


def _finite(values: Sequence[float | None]) -> np.ndarray:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        resolved = float(value)
        if math.isfinite(resolved):
            result.append(resolved)
    return np.asarray(result, dtype=np.float64)


def _p95(values: Sequence[float | None]) -> float:
    finite = _finite(values)
    return float(np.quantile(finite, 0.95)) if finite.size else math.nan


def _median(values: Sequence[float | None]) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else math.nan


def _episode_entry(
    trace: dict[str, np.ndarray], seed: int, phase: dict[str, Any]
) -> dict[str, Any]:
    rows = _seed_rows(trace, seed)
    time_s = np.asarray(trace["time_s"], dtype=np.float64)[rows]
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)[rows]
    eta = np.asarray(trace["eta"], dtype=np.float64)[rows]
    goal = np.asarray(trace["goal_ned_m"], dtype=np.float64)[rows]
    pitch_error = np.asarray(trace["pitch_error_rad"], dtype=np.float64)[rows]
    entry = _first_entry(distance, SHELL_M)
    closest = int(np.argmin(distance))
    if entry is None:
        return {
            "seed": int(seed),
            "success": bool(phase.get("success")),
            "entry2": False,
            "closest_distance_m": float(distance[closest]),
            "vertical_error_m": None,
            "pitch_error_rad": None,
            "T_to_closest_s": None,
            "vertical_rate_req_mps": None,
            "pitch_rate_req_rad_s": None,
        }
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else math.nan
    horizon = float(time_s[closest] - time_s[entry])
    if not math.isfinite(horizon) or horizon <= 0.0:
        horizon = dt
    vertical = float(abs(eta[entry, 2] - goal[entry, 2]))
    pitch = float(abs(pitch_error[entry]))
    return {
        "seed": int(seed),
        "success": bool(phase.get("success")),
        "entry2": True,
        "entry2_time_s": float(time_s[entry]),
        "closest_time_s": float(time_s[closest]),
        "closest_distance_m": float(distance[closest]),
        "T_to_closest_s": horizon,
        "vertical_error_m": vertical,
        "pitch_error_rad": pitch,
        "vertical_rate_req_mps": vertical / max(horizon, 1e-9),
        "pitch_rate_req_rad_s": pitch / max(horizon, 1e-9),
        "terminal_share_last2s": float(
            phase["rewards"]["last2s"]["terminal_share"]
        ),
        "dense_terminal_ratio_last2s": float(
            phase["rewards"]["last2s"]["dense_terminal_ratio"]
        ),
    }


def _classify(margin: float | None) -> str:
    if margin is None or not math.isfinite(margin):
        return "NO_2M_ENTRY"
    if margin <= AMBIGUOUS_LOW:
        return "REACHABLE_AT_2M"
    if margin <= AMBIGUOUS_HIGH:
        return "AMBIGUOUS_AT_2M"
    return "UNREACHABLE_AT_2M"


def _credit_label(entry: dict[str, Any], reach: str) -> str:
    if reach != "REACHABLE_AT_2M":
        return "NOT_CREDIT_EVALUABLE"
    share = float(entry["terminal_share_last2s"])
    return (
        "TERMINAL_CREDIT_PRIMARY"
        if share < TERMINAL_SHARE_THRESHOLD
        else "REACHABLE_CONTROL_LIMITED"
    )


def _markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Goal118：2 m entry reachability",
        "",
        f"- gate: `{gate['status']}`",
        f"- success envelope: `{json.dumps(payload['success_envelope'], ensure_ascii=False)}`",
        "",
        "## 11 个 primary failure",
        "",
        "| seed | vertical_m | pitch_rad | T_s | margin_v | margin_p | margin_max | reach | credit |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["primary"]:
        lines.append(
            f"| {row['seed']} | {row['vertical_error_m']:.3f} | {row['pitch_error_rad']:.3f} | "
            f"{row['T_to_closest_s']:.3f} | {row['margin_vertical']:.3f} | "
            f"{row['margin_pitch']:.3f} | {row['margin_max']:.3f} | "
            f"`{row['reachability']}` | `{row['credit_label']}` |"
        )
    lines.extend(
        [
            "",
            "## 分组",
            "",
            "| group | n | reachable | ambiguous | unreachable | no entry |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, group in payload["groups"].items():
        counts = group["reachability_counts"]
        lines.append(
            f"| {name} | {group['count']} | {counts.get('REACHABLE_AT_2M', 0)} | "
            f"{counts.get('AMBIGUOUS_AT_2M', 0)} | {counts.get('UNREACHABLE_AT_2M', 0)} | "
            f"{counts.get('NO_2M_ENTRY', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir or (args.root / "R100_RTRACE")
    phase0 = _read_json(args.root / "phase0_summary.json")
    trace = _load_trace(run_dir / "trace.npz")
    episodes = phase0.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(EXPECTED_SEEDS):
        raise AuditError("phase0 summary must contain 40 episodes")
    by_seed = {int(row["seed"]): row for row in episodes}
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise AuditError("phase0 summary seeds must be 20-59")
    protocol = phase0.get("protocol", {})
    if not bool(protocol.get("parity_pass")):
        raise AuditError("Goal117 parity did not pass")
    if int(protocol.get("success_count", -1)) != 20:
        raise AuditError("Goal117 success count is not 20/40")
    entries = [_episode_entry(trace, seed, by_seed[seed]) for seed in EXPECTED_SEEDS]
    success_entries = [row for row in entries if row["success"] and row["entry2"]]
    if len(success_entries) < 10:
        raise AuditError("too few success controls with a 2 m entry")
    success_envelope = {
        "vertical_p95_mps": _p95([row["vertical_rate_req_mps"] for row in success_entries]),
        "pitch_p95_rad_s": _p95([row["pitch_rate_req_rad_s"] for row in success_entries]),
        "count": len(success_entries),
    }
    for row in entries:
        if not row["entry2"]:
            row["margin_vertical"] = None
            row["margin_pitch"] = None
            row["margin_max"] = None
            row["reachability"] = "NO_2M_ENTRY"
            row["credit_label"] = "NOT_CREDIT_EVALUABLE"
            continue
        v = row["vertical_rate_req_mps"]
        p = row["pitch_rate_req_rad_s"]
        mv = v / success_envelope["vertical_p95_mps"] if success_envelope["vertical_p95_mps"] > 0 else math.nan
        mp = p / success_envelope["pitch_p95_rad_s"] if success_envelope["pitch_p95_rad_s"] > 0 else math.nan
        margin = max(mv, mp)
        row["margin_vertical"] = mv
        row["margin_pitch"] = mp
        row["margin_max"] = margin
        row["reachability"] = _classify(margin)
        row["credit_label"] = _credit_label(row, row["reachability"])

    primary = [row for row in entries if row["seed"] in PRIMARY_SEEDS]
    failures = [row for row in entries if not row["success"]]
    successes = [row for row in entries if row["success"]]

    def counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in rows:
            result[row["reachability"]] = result.get(row["reachability"], 0) + 1
        return result

    def group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "reachability_counts": counts(rows),
            "margin_vertical_median": _median([row["margin_vertical"] for row in rows]),
            "margin_pitch_median": _median([row["margin_pitch"] for row in rows]),
            "margin_max_median": _median([row["margin_max"] for row in rows]),
            "terminal_share_last2s_median": _median(
                [row["terminal_share_last2s"] for row in rows if row["entry2"]]
            ),
            "dense_terminal_ratio_last2s_median": _median(
                [row["dense_terminal_ratio_last2s"] for row in rows if row["entry2"]]
            ),
        }

    groups = {
        "success": group(successes),
        "failure_all": group(failures),
        "primary_failure": group(primary),
    }
    primary_counts = groups["primary_failure"]["reachability_counts"]
    credit_counts: dict[str, int] = {}
    for row in primary:
        credit_counts[row["credit_label"]] = credit_counts.get(row["credit_label"], 0) + 1
    reachable = primary_counts.get("REACHABLE_AT_2M", 0)
    unreachable = primary_counts.get("UNREACHABLE_AT_2M", 0)
    if unreachable >= 8:
        status = "APPROACH_GEOMETRY_PRIMARY"
    elif reachable >= 8 and credit_counts.get("TERMINAL_CREDIT_PRIMARY", 0) >= 8:
        status = "TERMINAL_CREDIT_PRIMARY"
    elif reachable >= 8 and credit_counts.get("REACHABLE_CONTROL_LIMITED", 0) >= 8:
        status = "REACHABLE_CONTROL_LIMITED"
    else:
        status = "MIXED_REACHABILITY"
    gate = {
        "status": status,
        "primary_reachability_counts": primary_counts,
        "primary_credit_counts": credit_counts,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "protocol": protocol,
        "success_envelope": success_envelope,
        "groups": groups,
        "gate": gate,
        "primary": primary,
        "episodes": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": status, "primary": primary_counts, "credit": credit_counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
