#!/usr/bin/env python3
"""Summarize the Goal85b single-factor plant-mismatch diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FACTORS = ("drag_scale", "thrust_scale", "fin_scale", "added_mass_scale")
SCENARIOS = ("default", "strict125")
ROOTS = ("baseline", "candidate")
SEEDS = tuple(range(20, 60))
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 8506


def _load(path: Path) -> dict[int, dict[str, Any]]:
    rows = json.loads((path / "episodes.json").read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in rows}


def _bootstrap(values: np.ndarray) -> dict[str, float | bool]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    mean = float(values.mean())
    return {
        "mean": mean,
        "median": float(np.median(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "standard_error": standard_error,
        "contains_zero": bool(low <= 0.0 <= high),
        "positive_seeds": int(np.count_nonzero(values > 0.0)),
        "negative_seeds": int(np.count_nonzero(values < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data/yhx/underwater-robotics/src/plan_p1/"
            "goal85b_plant_mismatch_20261110"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {}
    for factor in FACTORS:
        result[factor] = {}
        for scenario in SCENARIOS:
            result[factor][scenario] = {}
            for root in ROOTS:
                nominal = _load(
                    args.root / "b1" / "m0" / scenario / root / "zero-residual" / "s240"
                )
                treatment = _load(
                    args.root
                    / "b1_component"
                    / factor
                    / scenario
                    / root
                    / "zero-residual"
                    / "s240"
                )
                result[factor][scenario][root] = {}
                for metric in (
                    "path_three_d_rmse_m",
                    "terminal_eta_distance_m",
                ):
                    deltas = np.asarray(
                        [
                            float(treatment[seed][metric]) - float(nominal[seed][metric])
                            for seed in SEEDS
                        ],
                        dtype=np.float64,
                    )
                    result[factor][scenario][root][metric] = _bootstrap(deltas)

    payload = {"schema": "auv_pc_goal85b_components.v1", "results": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Goal85b single-factor plant mismatch",
        "",
        "Zero-residual controller, single factor `[-10%, +10%]`, paired seeds 20-59.",
        "",
        "| factor | scenario | root | path mean Δ | path median Δ | path 95% CI | terminal mean Δ |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for factor in FACTORS:
        for scenario in SCENARIOS:
            for root in ROOTS:
                path = result[factor][scenario][root]["path_three_d_rmse_m"]
                terminal = result[factor][scenario][root]["terminal_eta_distance_m"]
                lines.append(
                    f"| {factor} | {scenario} | {root} | {path['mean']:+.6f} | "
                    f"{path['median']:+.6f} | [{path['ci_low']:+.6f}, {path['ci_high']:+.6f}] | "
                    f"{terminal['mean']:+.6f} |"
                )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
