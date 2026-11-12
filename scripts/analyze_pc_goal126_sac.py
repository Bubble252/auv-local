#!/usr/bin/env python3
"""Aggregate Goal126 SAC P-A screen results and apply the preregistered gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/lowlevel/goal126_sac_pa_bias_20261111")
SEEDS = tuple(range(86201, 86209))
SCENARIOS = ("strict", "default")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _load_eval(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((path / "episodes.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    success = int(summary["success_count_pnav_tight"])
    success_seeds = sorted(int(row["seed"]) for row in rows if row.get("success_pnav_tight"))
    return {
        "success": success,
        "success_seeds": success_seeds,
        "timeout": sum(bool(row.get("timeout")) for row in rows),
        "safety": sum(bool(row.get("safety")) for row in rows),
        "finite": bool(summary.get("finite", False)),
        "action_abs_max_mean": float(summary.get("action_abs_max_mean", 0.0)),
        "residual_l2_ratio_mean": float(summary.get("residual_l2_ratio_mean", 0.0)),
        "reward_mean": float(summary.get("reward_mean", 0.0)),
        "steps_mean": float(summary.get("steps_mean", 0.0)),
        "path_three_d_rmse_m_mean": float(summary.get("path_three_d_rmse_m_mean", 0.0)),
    }


def _gate(runs: dict[int, dict[str, dict[str, Any]]], init: dict[str, dict[str, Any]]) -> dict[str, Any]:
    invalid: list[str] = []
    harm: list[str] = []
    if init["strict"]["success"] != 24:
        invalid.append(f"init_strict={init['strict']['success']} != 24")
    if init["default"]["success"] != 33:
        invalid.append(f"init_default={init['default']['success']} != 33")
    for scenario in SCENARIOS:
        if not init[scenario]["finite"]:
            invalid.append(f"init_{scenario}_nonfinite")
    for seed in SEEDS:
        for scenario in SCENARIOS:
            run = runs[seed][scenario]
            if not run["finite"]:
                invalid.append(f"{seed}_{scenario}_nonfinite")
            if run["timeout"] > 0:
                harm.append(f"{seed}_{scenario}_timeout={run['timeout']}")
            if run["safety"] > 0:
                harm.append(f"{seed}_{scenario}_safety={run['safety']}")

    strict = [runs[seed]["strict"]["success"] for seed in SEEDS]
    default = [runs[seed]["default"]["success"] for seed in SEEDS]
    strict_mean = _mean(strict)
    default_mean = _mean(default)
    if invalid:
        status = "INVALID"
        reason = "; ".join(invalid)
    elif harm:
        status = "SAC_PA_HARM"
        reason = "; ".join(harm)
    elif (
        strict_mean >= 26.0
        and sum(value >= 26 for value in strict) >= 4
        and all(value >= 31 for value in default)
    ):
        status = "SAC_PA_GAIN"
        reason = "strict_mean>=26, >=4 seeds>=26, all default>=31"
    elif strict_mean < 22.0 or any(value <= 30 for value in default):
        status = "SAC_PA_HARM"
        reason = f"strict_mean={strict_mean:.2f}, default={default}"
    elif strict_mean <= 24.5:
        status = "SAC_PA_CEILING"
        reason = "strict_mean<=24.5 and default not below control-2"
    else:
        status = "SAC_PA_MIXED"
        reason = f"strict_mean={strict_mean:.2f}, default_mean={default_mean:.2f}"
    return {
        "status": status,
        "reason": reason,
        "strict_success": dict(zip(SEEDS, strict)),
        "default_success": dict(zip(SEEDS, default)),
        "strict_mean": strict_mean,
        "default_mean": default_mean,
        "strict_seeds_at_least_26": sum(value >= 26 for value in strict),
        "default_seeds_at_least_31": sum(value >= 31 for value in default),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal126 SAC P-A screen",
        "",
        f"- gate: `{result['gate']['status']}`",
        f"- reason: `{result['gate']['reason']}`",
        f"- strict mean: `{result['gate']['strict_mean']:.2f}/40`",
        f"- default mean: `{result['gate']['default_mean']:.2f}/40`",
        "",
        "| seed | strict | default | timeout+safety | action abs max | residual/base L2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        strict = result["runs"][str(seed)]["strict"]
        default = result["runs"][str(seed)]["default"]
        lines.append(
            f"| {seed} | {strict['success']}/40 | {default['success']}/40 | "
            f"{strict['timeout'] + strict['safety'] + default['timeout'] + default['safety']} | "
            f"{strict['action_abs_max_mean']:.3f}/{default['action_abs_max_mean']:.3f} | "
            f"{strict['residual_l2_ratio_mean']:.3f}/{default['residual_l2_ratio_mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    init = {scenario: _load_eval(args.root / "init_parity_eval" / scenario) for scenario in SCENARIOS}
    runs: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        runs[seed] = {}
        for scenario in SCENARIOS:
            runs[seed][scenario] = _load_eval(args.root / "eval" / f"S_s{seed}" / scenario)
    result = {
        "schema": "auv_pc_goal126_sac_gate.v1",
        "root": str(args.root),
        "init_parity": init,
        "runs": {str(seed): runs[seed] for seed in SEEDS},
        "gate": _gate(runs, init),
    }
    output = args.output or args.root / "goal126_sac_summary.json"
    markdown = args.markdown or args.root / "goal126_sac_summary.md"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result["gate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
