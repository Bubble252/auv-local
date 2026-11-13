#!/usr/bin/env python3
"""Aggregate Goal125 bias-initialized PPO screen and apply the preregistered gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/lowlevel/goal125_bias_ppo_20261111")
ARMS = ("Z15", "B15", "B20", "BP3")
SEEDS = (86201, 86202)
SCENARIOS = ("strict", "default")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args(argv)


def _load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        rows = payload.get("episode_rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no episode rows")
    success = sum(bool(row.get("success")) for row in rows)
    timeout = sum(bool(row.get("timeout")) for row in rows)
    safety = sum(bool(row.get("safety")) for row in rows)
    final_eta = [
        float(row["final_eta_error_l2"])
        for row in rows
        if row.get("final_eta_error_l2") is not None
    ]
    residual_ratio = [
        float(row["residual_to_base_l2_ratio_mean"])
        for row in rows
        if row.get("residual_to_base_l2_ratio_mean") is not None
    ]
    return {
        "success": success,
        "timeout": timeout,
        "safety": safety,
        "success_seeds": sorted(int(row["seed"]) for row in rows if row.get("success")),
        "final_eta_error_l2_median": statistics.median(final_eta) if final_eta else None,
        "residual_to_base_l2_ratio_median": (
            statistics.median(residual_ratio) if residual_ratio else None
        ),
        "metrics": metrics,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _delta(arm: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    arm_success = set(arm["success_seeds"])
    control_success = set(control["success_seeds"])
    return {
        "success_delta": arm["success"] - control["success"],
        "gained": sorted(arm_success - control_success),
        "lost": sorted(control_success - arm_success),
        "lost_count": len(control_success - arm_success),
    }


def _gate(runs: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    control_86201 = controls[86201]
    invalid_reasons: list[str] = []
    if control_86201["strict"]["success"] != 24:
        invalid_reasons.append(
            f"control_86201_strict={control_86201['strict']['success']} != 24"
        )
    if control_86201["default"]["success"] != 33:
        invalid_reasons.append(
            f"control_86201_default={control_86201['default']['success']} != 33"
        )
    for seed in SEEDS:
        for scenario in SCENARIOS:
            run = controls[seed][scenario]
            if run["safety"] > 0:
                invalid_reasons.append(f"control_{seed}_{scenario}_safety={run['safety']}")
            if run["timeout"] > 0:
                invalid_reasons.append(f"control_{seed}_{scenario}_timeout={run['timeout']}")
    for arm in ARMS:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                run = runs[arm]["seeds"][seed][scenario]
                if run["safety"] > 0:
                    invalid_reasons.append(f"{arm}_{seed}_{scenario}_safety={run['safety']}")
                if run["timeout"] > 0:
                    invalid_reasons.append(f"{arm}_{seed}_{scenario}_timeout={run['timeout']}")

    control_mean_strict = _mean(
        [controls[seed]["strict"]["success"] for seed in SEEDS]
    )
    control_mean_default = _mean(
        [controls[seed]["default"]["success"] for seed in SEEDS]
    )

    summaries: dict[str, Any] = {}
    for arm in ARMS:
        strict = [runs[arm]["seeds"][seed]["strict"]["success"] for seed in SEEDS]
        default = [runs[arm]["seeds"][seed]["default"]["success"] for seed in SEEDS]
        strict_deltas = {}
        default_deltas = {}
        pair_details: dict[str, Any] = {}
        for seed in SEEDS:
            pair_details[str(seed)] = {}
            for scenario, target in (("strict", strict_deltas), ("default", default_deltas)):
                detail = _delta(
                    runs[arm]["seeds"][seed][scenario], controls[seed][scenario]
                )
                target[seed] = detail["success_delta"]
                pair_details[str(seed)][scenario] = detail
        mean_strict = _mean(strict)
        mean_default = _mean(default)
        if mean_strict >= control_mean_strict + 2.0 and all(
            delta >= 0 for delta in strict_deltas.values()
        ) and mean_default >= control_mean_default - 2.0:
            status = "GAIN"
        elif mean_strict < control_mean_strict - 1.0 or any(
            delta < -2 for delta in default_deltas.values()
        ):
            status = "HARM"
        elif mean_strict <= control_mean_strict + 0.5 and mean_default >= control_mean_default - 2.0:
            status = "CEILING"
        else:
            status = "MIXED"
        summaries[arm] = {
            "status": status,
            "strict_success": strict,
            "default_success": default,
            "strict_mean": mean_strict,
            "default_mean": mean_default,
            "strict_delta_mean": mean_strict - control_mean_strict,
            "default_delta_mean": mean_default - control_mean_default,
            "strict_deltas_by_seed": {str(seed): value for seed, value in strict_deltas.items()},
            "default_deltas_by_seed": {str(seed): value for seed, value in default_deltas.items()},
            "pairs": pair_details,
        }

    statuses = [summaries[arm]["status"] for arm in ARMS]
    if invalid_reasons:
        status = "INVALID"
        reason = "; ".join(invalid_reasons)
    elif "GAIN" in statuses:
        winners = [arm for arm in ARMS if summaries[arm]["status"] == "GAIN"]
        status = "PPO_BIAS_GAIN"
        reason = f"gain_arms={winners}"
    elif all(value == "CEILING" for value in statuses):
        status = "BIAS_FIXED_IS_CEILING"
        reason = "all_arms_within_0.5_of_fixed_bias"
    elif all(value == "HARM" for value in statuses):
        status = "PPO_BIAS_HARM"
        reason = "all_arms_below_fixed_bias"
    else:
        status = "PPO_BIAS_MIXED"
        reason = f"arm_statuses={dict(zip(ARMS, statuses))}"
    return {
        "status": status,
        "reason": reason,
        "control_mean_strict": control_mean_strict,
        "control_mean_default": control_mean_default,
        "arms": summaries,
        "invalid_reasons": invalid_reasons,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.2f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Goal125 bias-initialized PPO screen",
        "",
        f"- gate: `{gate['status']}`",
        f"- reason: `{gate['reason']}`",
        f"- fixed-bias control strict mean: `{gate['control_mean_strict']:.2f}/40`",
        f"- fixed-bias control default mean: `{gate['control_mean_default']:.2f}/40`",
        "",
        "| arm | status | strict mean | Δ strict | default mean | Δ default | strict seeds | default seeds |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for arm in ARMS:
        row = gate["arms"][arm]
        lines.append(
            f"| {arm} | {row['status']} | {row['strict_mean']:.2f} | "
            f"{row['strict_delta_mean']:+.2f} | {row['default_mean']:.2f} | "
            f"{row['default_delta_mean']:+.2f} | {row['strict_success']} | "
            f"{row['default_success']} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| seed | scenario | success | timeout | safety |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        for scenario in SCENARIOS:
            row = payload["controls"][str(seed)][scenario]
            lines.append(
                f"| {seed} | {scenario} | {row['success']}/40 | "
                f"{row['timeout']} | {row['safety']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs: dict[str, Any] = {}
    for arm in ARMS:
        runs[arm] = {"seeds": {}}
        for seed in SEEDS:
            runs[arm]["seeds"][seed] = {}
            for scenario in SCENARIOS:
                path = args.root / "eval" / f"{arm}_s{seed}" / scenario / "summary.json"
                runs[arm]["seeds"][seed][scenario] = _load_run(path)
    controls: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        controls[seed] = {}
        for scenario in SCENARIOS:
            path = args.root / "controls" / str(seed) / scenario / "summary.json"
            controls[seed][scenario] = _load_run(path)
    payload = {
        "schema_version": "auv_pc_goal125_bias_ppo_screen.v1",
        "runs": {arm: runs[arm] for arm in ARMS},
        "controls": {str(seed): controls[seed] for seed in SEEDS},
        "gate": _gate(runs, controls),
    }
    output = args.output or args.root / "goal125_bias_ppo_summary.json"
    markdown = args.markdown or args.root / "goal125_bias_ppo_summary.md"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"]["status"], "reason": payload["gate"]["reason"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
