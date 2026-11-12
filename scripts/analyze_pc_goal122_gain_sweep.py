#!/usr/bin/env python3
"""Aggregate Goal122 PPO residual gain sweep and apply the frozen Gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/lowlevel/goal122_ppo_gain_sweep_20261111")
PROFILES = ("zero", "ppo", "ppo-4x", "ppo-8x", "ppo-16x", "ppo-32x", "ppo-64x")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args(argv)


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

    gains = [profile for profile in PROFILES if profile.startswith("ppo-") and profile not in {"ppo-4x"}]
    if any(summary[profile]["safety"] > 0 for profile in PROFILES):
        status = "INVALID"
        reason = "safety_termination_observed"
    elif any(summary[profile]["success"] < 18 for profile in PROFILES):
        status = "GAIN_HARM"
        reason = "success_below_18"
    else:
        recover = [
            profile
            for profile in gains
            if summary[profile]["success"] >= 22
            or summary[profile]["reachable_final_eta_delta_median"] >= 0.25
        ]
        moved_only = [
            profile
            for profile in gains
            if summary[profile]["success"] < 22
            and summary[profile]["reachable_final_eta_delta_median"] >= 0.25
        ]
        if recover:
            status = "GAIN_RECOVERS_CAPTURE"
            reason = f"profiles={recover}"
        elif moved_only:
            status = "GAIN_AUTHORITY_WITHOUT_CAPTURE"
            reason = f"profiles={moved_only}"
        else:
            status = "PPO_DIRECTION_UNUSABLE"
            reason = "no_gain_reached_22_or_delta_0.25"

    # Sanity: zero and ppo must reproduce the frozen references.
    if summary["zero"]["success"] != 20:
        status = "INVALID"
        reason = f"zero_did_not_reproduce_20:got_{summary['zero']['success']}"
    if summary["ppo"]["success"] != 20:
        status = "INVALID"
        reason = f"ppo_did_not_reproduce_20:got_{summary['ppo']['success']}"

    return {"status": status, "reason": reason, "profiles": summary}


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal122 PPO residual gain sweep",
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
    args = parse_args(argv)
    runs = [_BASE.analyze_profile(args.root, profile) for profile in PROFILES]
    payload = {
        "schema_version": "auv_pc_goal122_gain_sweep.v1",
        "runs": runs,
        "gate": decide(runs),
    }
    output = args.output or (args.root / "goal122_gain_sweep_summary.json")
    markdown = args.markdown or (args.root / "goal122_gain_sweep_summary.md")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"]["status"], "profiles": payload["gate"]["profiles"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
