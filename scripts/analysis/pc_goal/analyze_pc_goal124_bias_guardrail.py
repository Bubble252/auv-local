#!/usr/bin/env python3
"""Aggregate Goal124 default/strict constant-bias guardrail."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/lowlevel/goal124_bias_guardrail_20261111")
PROFILES = (
    "default_zero",
    "default_stern-025",
    "default_stern-030",
    "strict_zero",
    "strict_stern-025",
    "strict_stern-030",
)
BASE_ANALYZER = Path("scripts/analyze_pc_goal121_authority.py")


def _load_base():
    spec = importlib.util.spec_from_file_location("goal121_authority", BASE_ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base()


def _summarize(run: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    ref_success = {row["seed"] for row in reference["episodes"] if row["success"]}
    success = {row["seed"] for row in run["episodes"] if row["success"]}
    return {
        "success": run["success"],
        "timeout": run["timeout"],
        "safety": run["safety"],
        "gained": sorted(success - ref_success),
        "lost": sorted(ref_success - success),
        "lost_count": len(ref_success - success),
    }


def decide(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_profile = {run["profile"]: run for run in runs}
    default_zero = by_profile["default_zero"]
    strict_zero = by_profile["strict_zero"]
    summary: dict[str, Any] = {}
    for profile in PROFILES:
        ref = default_zero if profile.startswith("default_") else strict_zero
        item = _summarize(by_profile[profile], ref)
        if profile != ("default_zero" if profile.startswith("default_") else "strict_zero"):
            item["success_delta_vs_zero"] = item["success"] - ref["success"]
        else:
            item["success_delta_vs_zero"] = 0
        summary[profile] = item

    default_best = max(
        ("default_stern-025", "default_stern-030"),
        key=lambda profile: summary[profile]["success"],
    )
    strict_best = max(
        ("strict_stern-025", "strict_stern-030"),
        key=lambda profile: summary[profile]["success"],
    )
    default_delta = summary[default_best]["success_delta_vs_zero"]
    strict_success = summary[strict_best]["success"]

    if any(summary[profile]["safety"] > 0 for profile in PROFILES):
        status = "INVALID"
        reason = "safety_termination_observed"
    elif summary["strict_zero"]["success"] != 20:
        status = "INVALID"
        reason = f"strict_zero_not_20:got_{summary['strict_zero']['success']}"
    elif default_delta <= -2:
        status = "BIAS_DEFAULT_HARM"
        reason = f"default_best={default_best}_delta={default_delta}"
    elif strict_success < 23:
        status = "BIAS_STRICT_NOT_REPRODUCED"
        reason = f"strict_best={strict_best}_success={strict_success}"
    elif strict_success >= 23 and default_delta >= -1:
        status = "BIAS_GUARDRAIL_PASS"
        reason = f"default_delta={default_delta}_strict_success={strict_success}"
    else:
        status = "BIAS_WEAK"
        reason = f"default_delta={default_delta}_strict_success={strict_success}"
    return {
        "status": status,
        "reason": reason,
        "default_best": default_best,
        "strict_best": strict_best,
        "profiles": summary,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.2f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal124 constant bias default guardrail",
        "",
        f"- gate: `{payload['gate']['status']}`",
        f"- reason: `{payload['gate']['reason']}`",
        "",
        "| profile | success | Δ vs zero | gained | lost |",
        "|---|---:|---:|---|---|",
    ]
    for profile in PROFILES:
        row = payload["gate"]["profiles"][profile]
        lines.append(
            f"| {profile} | {row['success']}/40 | {row['success_delta_vs_zero']:+d} | "
            f"{row['gained']} | {row['lost']} |"
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
        "schema_version": "auv_pc_goal124_bias_guardrail.v1",
        "runs": runs,
        "gate": decide(runs),
    }
    output = args.output or (args.root / "goal124_guardrail_summary.json")
    markdown = args.markdown or (args.root / "goal124_guardrail_summary.md")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"]["status"], "profiles": payload["gate"]["profiles"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
