#!/usr/bin/env python3
"""Aggregate Goal123 constant stern/rpm bias calibration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/lowlevel/goal123_constant_bias_20261111")
PROFILES = (
    "zero",
    "stern-005",
    "stern-010",
    "stern-015",
    "stern-020",
    "stern-025",
    "stern-030",
    "rpm-010",
    "rpm-020",
    "stern-020_rpm-010",
)
DEV_SEEDS = tuple(range(20, 60))
REACHABLE_FAILURES = (25, 27, 31, 50, 51)
BASE_ANALYZER = Path("scripts/analyze_pc_goal121_authority.py")


def _load_base():
    spec = importlib.util.spec_from_file_location("goal121_authority", BASE_ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base()


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
            "lost_count": len(zero_success - success),
            "reachable_final_eta_delta_median": _BASE._median(reachable_delta),
            "reachable_final_eta_delta_by_seed": {
                str(seed): float(
                    eps[seed]["final_eta_error_l2"] - zero_eps[seed]["final_eta_error_l2"]
                )
                for seed in REACHABLE_FAILURES
            },
            "applied_mean_median": _BASE._median(
                [row["applied_mean"] for row in run["episodes"]]
            ),
            "effective_mean_median": _BASE._median(
                [row["effective_mean"] for row in run["episodes"]]
            ),
            "gate_mean_median": _BASE._median(
                [row["gate_mean"] for row in run["episodes"]]
            ),
        }

    candidates = [profile for profile in PROFILES if profile != "zero"]
    best = max(candidates, key=lambda profile: summary[profile]["success"])
    best_success = summary[best]["success"]
    best_lost = summary[best]["lost_count"]

    if any(summary[profile]["safety"] > 0 for profile in PROFILES):
        status = "INVALID"
        reason = "safety_termination_observed"
    elif summary["zero"]["success"] != 20:
        status = "INVALID"
        reason = f"zero_did_not_reproduce_20:got_{summary['zero']['success']}"
    elif best_success >= 23 and best_lost <= 1:
        status = "BIAS_IMPROVES_CAPTURE"
        reason = f"best={best}_success={best_success}_lost={best_lost}"
    elif best_success >= 21:
        status = "BIAS_WEAK"
        reason = f"best={best}_success={best_success}_lost={best_lost}"
    else:
        status = "BIAS_NONE"
        reason = f"best={best}_success={best_success}_lost={best_lost}"
    return {"status": status, "reason": reason, "best_profile": best, "profiles": summary}


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal123 constant stern/rpm bias calibration",
        "",
        f"- gate: `{payload['gate']['status']}`",
        f"- reason: `{payload['gate']['reason']}`",
        f"- best: `{payload['gate']['best_profile']}`",
        "",
        "| profile | success | gained | lost | reachable Δeta median | applied | effective |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for profile in PROFILES:
        row = payload["gate"]["profiles"][profile]
        lines.append(
            f"| {profile} | {row['success']}/40 | {row['gained']} | {row['lost']} | "
            f"{_fmt(row['reachable_final_eta_delta_median'], 4)} | "
            f"{_fmt(row['applied_mean_median'], 1)} | "
            f"{_fmt(row['effective_mean_median'], 1)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = [_BASE.analyze_profile(args.root, profile) for profile in PROFILES]
    payload = {
        "schema_version": "auv_pc_goal123_bias_calibration.v1",
        "runs": runs,
        "gate": decide(runs),
    }
    output = args.output or (args.root / "goal123_bias_summary.json")
    markdown = args.markdown or (args.root / "goal123_bias_summary.md")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"]["status"], "best": payload["gate"]["best_profile"], "profiles": payload["gate"]["profiles"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
