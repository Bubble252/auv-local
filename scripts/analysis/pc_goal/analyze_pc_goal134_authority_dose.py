#!/usr/bin/env python3
"""Analyze Goal134's post-Gate residual-authority dose-response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCENARIOS = ("strict", "default")
ARMS = ("S_s86201", "S_s86202", "S_s86204")
SCALES = (1.0, 0.5, 0.25)
DEV_SEEDS = tuple(range(20, 60))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _load(path: Path) -> dict[str, Any]:
    directory = path
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    episodes = json.loads((directory / "episodes.json").read_text(encoding="utf-8"))
    by_seed = {int(row["seed"]): bool(row["success_pnav_tight"]) for row in episodes}
    if set(by_seed) != set(DEV_SEEDS):
        raise ValueError(f"unexpected seeds in {directory}")
    return {
        "success": int(sum(by_seed.values())),
        "success_by_seed": by_seed,
        "action_abs_max_mean": float(summary["summary"]["action_abs_max_mean"]),
        "residual_l2_ratio_mean": float(summary["summary"]["residual_l2_ratio_mean"]),
    }


def _paired(control: dict[str, Any], arm: dict[str, Any]) -> dict[str, int]:
    deltas = [
        int(arm["success_by_seed"][seed]) - int(control["success_by_seed"][seed])
        for seed in DEV_SEEDS
    ]
    return {
        "delta": int(sum(deltas)),
        "newly_won": int(sum(value > 0 for value in deltas)),
        "newly_lost": int(sum(value < 0 for value in deltas)),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal134 Residual Authority Dose",
        "",
        "**level**：`current15`；**zero**：strict/default = "
        f"{payload['zero']['strict']['success']}/{payload['zero']['default']['success']}",
        "",
        "| arm | scale | strict | d_strict | default | d_default | action max | residual L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for scale in SCALES:
            item = payload["arms"][arm]["scales"][f"{scale:.2f}"]
            lines.append(
                f"| {arm} | {scale:.2f} | {item['strict']['success']}/40 | "
                f"{item['strict']['paired_vs_zero']['delta']:+d} | "
                f"{item['default']['success']}/40 | "
                f"{item['default']['paired_vs_zero']['delta']:+d} | "
                f"{item['strict']['action_abs_max_mean']:.3f} | "
                f"{item['strict']['residual_l2_ratio_mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- 86202: `{payload['arms']['S_s86202']['status']}`",
            f"- 86204: `{payload['arms']['S_s86204']['status']}`",
            f"- 86201: `{payload['arms']['S_s86201']['status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _arm_status(arm: str, scales: dict[str, Any]) -> str:
    low = scales["0.25"]
    strict_delta = low["strict"]["paired_vs_zero"]["delta"]
    default_delta = low["default"]["paired_vs_zero"]["delta"]
    if strict_delta >= 2 and default_delta >= 0:
        return "LOW_AUTHORITY_DIRECTIONAL_GAIN"
    if strict_delta <= -2:
        return "RESIDUAL_DIRECTION_HARM"
    if strict_delta >= -1 and default_delta >= -1:
        return "AUTHORITY_DOMINANT"
    return "MIXED"


def main() -> int:
    args = _parse_args()
    root = _resolve(args.root)
    output = _resolve(args.output)
    zero = {
        scenario: _load(root / "current15" / scenario / "zero")
        for scenario in SCENARIOS
    }
    arms: dict[str, Any] = {}
    for arm in ARMS:
        scales: dict[str, Any] = {}
        for scale in SCALES:
            per_scenario: dict[str, Any] = {}
            for scenario in SCENARIOS:
                if scale == 1.0:
                    directory = root / "current15" / scenario / arm
                else:
                    directory = (
                        root
                        / "diagnostic/authority_dose"
                        / arm
                        / f"scale_{scale:g}_{scenario}"
                    )
                run = _load(directory)
                per_scenario[scenario] = {
                    **run,
                    "paired_vs_zero": _paired(zero[scenario], run),
                }
            scales[f"{scale:.2f}"] = per_scenario
        arms[arm] = {"scales": scales, "status": _arm_status(arm, scales)}
    payload = {
        "schema": "auv_pc_goal134_authority_dose.v1",
        "root": str(root),
        "level": "current15",
        "dev_seeds": list(DEV_SEEDS),
        "zero": {scenario: zero[scenario] for scenario in SCENARIOS},
        "arms": arms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({arm: arms[arm]["status"] for arm in ARMS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
