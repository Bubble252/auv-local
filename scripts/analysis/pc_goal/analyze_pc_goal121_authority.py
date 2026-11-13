#!/usr/bin/env python3
"""Aggregate Goal121 residual action-authority injection profiles."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("artifacts/lowlevel/goal121_action_authority_20261111")
PROFILES = (
    "zero",
    "ppo",
    "neg-ppo",
    "ppo-2x",
    "ppo-4x",
    "rudder+",
    "stern+",
    "stern-",
    "rpm+",
    "rpm-",
)
DEV_SEEDS = tuple(range(20, 60))
REACHABLE_FAILURES = (25, 27, 31, 50, 51)


class AuthorityError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuthorityError(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AuthorityError(f"missing trace: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _final_rows(trace: dict[str, np.ndarray], seed: int) -> np.ndarray:
    rows = np.flatnonzero(np.asarray(trace["episode_seed"], dtype=np.int64) == int(seed))
    if rows.size == 0:
        raise AuthorityError(f"trace has no rows for seed {seed}")
    return rows


def _episode_metrics(trace: dict[str, np.ndarray], seed: int, episode: dict[str, Any]) -> dict[str, Any]:
    rows = _final_rows(trace, seed)
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)[rows]
    eta = np.asarray(trace["eta"], dtype=np.float64)[rows]
    goal = np.asarray(trace["goal_ned_m"], dtype=np.float64)[rows]
    applied = np.asarray(trace["actuator_residual_applied"], dtype=np.float64)[rows]
    effective = np.asarray(trace["actuator_residual_effective"], dtype=np.float64)[rows]
    gate = np.asarray(trace["frozen_residual_gate"], dtype=np.float64)[rows]
    return {
        "seed": int(seed),
        "success": bool(episode.get("success")),
        "timeout": bool(episode.get("timeout")),
        "safety": bool(episode.get("safety", False)),
        "final_distance_m": float(distance[-1]),
        "final_eta": eta[-1, :3].tolist(),
        "final_goal": goal[-1].tolist(),
        "final_eta_error_l2": float(np.linalg.norm(eta[-1, :3] - goal[-1])),
        "applied_mean": float(np.mean(np.linalg.norm(applied, axis=1))),
        "effective_mean": float(np.mean(np.linalg.norm(effective, axis=1))),
        "gate_mean": float(np.mean(gate)),
    }


def analyze_profile(root: Path, profile: str) -> dict[str, Any]:
    summary = _read_json(root / profile / "summary.json")
    trace = _load_trace(root / profile / "trace.npz")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(DEV_SEEDS):
        raise AuthorityError(f"{profile}: expected {len(DEV_SEEDS)} episodes")
    by_seed = {int(row["seed"]): row for row in episodes}
    if tuple(sorted(by_seed)) != DEV_SEEDS:
        raise AuthorityError(f"{profile}: dev seeds are not 20-59")
    metrics = [_episode_metrics(trace, seed, by_seed[seed]) for seed in DEV_SEEDS]
    return {
        "profile": profile,
        "success": sum(row["success"] for row in metrics),
        "timeout": sum(row["timeout"] for row in metrics),
        "safety": sum(row["safety"] for row in metrics),
        "episodes": metrics,
    }


def _median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else math.nan


def decide(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_profile = {run["profile"]: run for run in runs}
    zero = by_profile["zero"]
    zero_eps = {row["seed"]: row for row in zero["episodes"]}
    zero_success = {row["seed"] for row in zero["episodes"] if row["success"]}
    summary: dict[str, Any] = {}
    for profile in PROFILES:
        run = by_profile[profile]
        eps = {row["seed"]: row for row in run["episodes"]}
        success = {row["seed"] for row in run["episodes"] if row["success"]}
        reachable_delta = [
            abs(eps[seed]["final_eta_error_l2"] - zero_eps[seed]["final_eta_error_l2"])
            for seed in REACHABLE_FAILURES
        ]
        summary[profile] = {
            "success": run["success"],
            "timeout": run["timeout"],
            "safety": run["safety"],
            "gained": sorted(success - zero_success),
            "lost": sorted(zero_success - success),
            "reachable_final_eta_delta_median": _median(reachable_delta),
            "reachable_final_eta_delta_by_seed": {
                str(seed): float(
                    eps[seed]["final_eta_error_l2"] - zero_eps[seed]["final_eta_error_l2"]
                )
                for seed in REACHABLE_FAILURES
            },
            "applied_mean_median": _median([row["applied_mean"] for row in run["episodes"]]),
            "effective_mean_median": _median([row["effective_mean"] for row in run["episodes"]]),
            "gate_mean_median": _median([row["gate_mean"] for row in run["episodes"]]),
        }

    if any(summary[profile]["safety"] > 0 for profile in PROFILES):
        status = "INVALID"
        reason = "safety_termination_observed"
    elif summary["neg-ppo"]["success"] >= summary["ppo"]["success"] + 2:
        status = "NEG_PPO_BETTER"
        reason = f"neg_ppo={summary['neg-ppo']['success']}_ppo={summary['ppo']['success']}"
    else:
        max_delta = max(
            summary[profile]["reachable_final_eta_delta_median"]
            for profile in PROFILES
            if profile != "zero"
        )
        max_gain = max(len(summary[profile]["gained"]) for profile in PROFILES if profile != "zero")
        if max_delta < 0.05 and max_gain < 2:
            status = "AUTHORITY_INERT"
            reason = f"max_delta={max_delta:.4f}_max_gain={max_gain}"
        elif max_delta >= 0.25 and max_gain < 2:
            status = "AUTHORITY_WITHOUT_CAPTURE"
            reason = f"max_delta={max_delta:.4f}_max_gain={max_gain}"
        else:
            status = "AUTHORITY_PRESENT"
            reason = f"max_delta={max_delta:.4f}_max_gain={max_gain}"
    return {"status": status, "reason": reason, "profiles": summary}


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal121 residual action-authority injection",
        "",
        f"- gate: `{payload['gate']['status']}`",
        f"- reason: `{payload['gate']['reason']}`",
        "",
        "| profile | success | gained | lost | reachable Δeta median | applied | effective | gate |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        row = payload["gate"]["profiles"][profile]
        lines.append(
            f"| {profile} | {row['success']}/40 | {row['gained']} | {row['lost']} | "
            f"{_fmt(row['reachable_final_eta_delta_median'], 4)} | "
            f"{_fmt(row['applied_mean_median'], 1)} | "
            f"{_fmt(row['effective_mean_median'], 1)} | "
            f"{_fmt(row['gate_mean_median'], 3)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = [analyze_profile(args.root, profile) for profile in PROFILES]
    payload = {
        "schema_version": "auv_pc_goal121_authority.v1",
        "runs": runs,
        "gate": decide(runs),
    }
    output = args.output or (args.root / "goal121_authority_summary.json")
    markdown = args.markdown or (args.root / "goal121_authority_summary.md")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"]["status"], "profiles": payload["gate"]["profiles"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
