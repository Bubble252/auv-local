#!/usr/bin/env python3
"""Summarize the frozen Goal85b B1 plant-mismatch sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LEVELS = ("m0", "m05", "m10")
SCENARIOS = ("default", "strict125")
ROOTS = ("baseline", "candidate")
CHECKPOINTS = ("s240", "s752", "s880", "s1008")
ACTIONS = ("zero-residual", *CHECKPOINTS)
METRICS = (
    "path_three_d_rmse_m",
    "terminal_eta_distance_m",
    "success_eta_tight_at_done",
    "goal_plane_miss",
    "timeout",
    "safety",
    "material_saturation_steps",
)
DEV_SEEDS = tuple(range(20, 60))
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 8505
SCALE_BOUNDS = {
    "m0": (1.0, 1.0),
    "m05": (0.95, 1.05),
    "m10": (0.90, 1.10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data/yhx/underwater-robotics/src/plan_p1/"
            "goal85b_plant_mismatch_20261110/b1"
        ),
    )
    parser.add_argument(
        "--goal85-root",
        type=Path,
        default=Path(
            "/data/yhx/underwater-robotics/src/plan_p1/"
            "goal85_unbiased_nav_20261110"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lowlevel/goal85b_b1_gate.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/lowlevel/goal85b_b1_gate.md"),
    )
    return parser.parse_args()


def _run_dir(
    root: Path, level: str, scenario: str, root_name: str, action: str
) -> Path:
    action_dir, checkpoint = (
        ("zero-residual", "s240") if action == "zero-residual" else ("policy", action)
    )
    return root / level / scenario / root_name / action_dir / checkpoint


def _load_run(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    episodes_path = path / "episodes.json"
    if not summary_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"missing summary/episodes under {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    rows = {int(row["seed"]): row for row in episodes}
    if tuple(sorted(rows)) != DEV_SEEDS:
        raise ValueError(f"{path}: expected dev seeds 20-59")
    return {"path": str(path), "summary": summary, "episodes": rows}


def _load_matrix(root: Path) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for level in LEVELS:
        matrix[level] = {}
        for scenario in SCENARIOS:
            matrix[level][scenario] = {}
            for root_name in ROOTS:
                matrix[level][scenario][root_name] = {}
                for action in ACTIONS:
                    matrix[level][scenario][root_name][action] = _load_run(
                        _run_dir(root, level, scenario, root_name, action)
                    )
    return matrix


def _bootstrap(values: np.ndarray) -> dict[str, float | bool]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(DEV_SEEDS),):
        raise ValueError("paired effects must contain one value per dev seed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    return {
        "mean": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "standard_error": standard_error,
        "effect_to_seed_noise": (
            float(abs(mean) / standard_error) if standard_error > 0.0 else None
        ),
        "contains_zero": bool(low <= 0.0 <= high),
        "nonzero_seeds": int(np.count_nonzero(np.abs(values) > 1e-15)),
        "n": int(values.size),
    }


def _paired(
    baseline: dict[str, Any], treatment: dict[str, Any], metric: str
) -> np.ndarray:
    return np.asarray(
        [
            float(treatment["episodes"][seed][metric])
            - float(baseline["episodes"][seed][metric])
            for seed in DEV_SEEDS
        ],
        dtype=np.float64,
    )


def _comparisons(matrix: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for treatment_level in ("m05", "m10"):
        key = f"{treatment_level}_vs_m0"
        result[key] = {}
        for scenario in SCENARIOS:
            result[key][scenario] = {}
            for root_name in ROOTS:
                result[key][scenario][root_name] = {}
                for action in ACTIONS:
                    result[key][scenario][root_name][action] = {
                        metric: _bootstrap(
                            _paired(
                                matrix["m0"][scenario][root_name][action],
                                matrix[treatment_level][scenario][root_name][action],
                                metric,
                            )
                        )
                        for metric in METRICS
                    }
    return result


def _scale_audit(matrix: dict[str, Any]) -> dict[str, Any]:
    names = (
        "drag_scale",
        "thrust_scale",
        "fin_scale",
        "added_mass_scale",
        "actuator_time_scale",
    )
    by_level: dict[str, Any] = {}
    crn: dict[str, Any] = {}
    for level in LEVELS:
        values = np.asarray(
            [
                [
                    row["plant_scale_factors"][name]
                    for name in names
                ]
                for scenario in SCENARIOS
                for root_name in ROOTS
                for action in ACTIONS
                for row in matrix[level][scenario][root_name][action][
                    "episodes"
                ].values()
            ],
            dtype=np.float64,
        )
        lower, upper = SCALE_BOUNDS[level]
        by_level[level] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "within_frozen_bounds": bool(
                np.all(values >= lower - 1e-15)
                and np.all(values <= upper + 1e-15)
            ),
        }
        reference = None
        mismatch_count = 0
        for scenario in SCENARIOS:
            for root_name in ROOTS:
                for action in ACTIONS:
                    rows = matrix[level][scenario][root_name][action]["episodes"]
                    current = np.asarray(
                        [
                            [rows[seed]["plant_scale_factors"][name] for name in names]
                            for seed in DEV_SEEDS
                        ],
                        dtype=np.float64,
                    )
                    if reference is None:
                        reference = current
                    else:
                        mismatch_count += int(
                            np.count_nonzero(np.abs(current - reference) > 1e-15)
                        )
        crn[level] = {
            "mismatch_values": mismatch_count,
            "pass": mismatch_count == 0,
        }
    return {"levels": by_level, "common_random_numbers": crn}


def _finite_audit(matrix: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    count = 0
    for level in LEVELS:
        for scenario in SCENARIOS:
            for root_name in ROOTS:
                for action in ACTIONS:
                    run = matrix[level][scenario][root_name][action]
                    count += 1
                    if not bool(run["summary"]["summary"].get("finite", False)):
                        failures.append(run["path"])
    return {"runs": count, "failures": failures, "pass": not failures}


def _nav_bias_audit(matrix: dict[str, Any]) -> dict[str, Any]:
    means: list[np.ndarray] = []
    for level in LEVELS:
        for scenario in SCENARIOS:
            for root_name in ROOTS:
                for action in ACTIONS:
                    rows = matrix[level][scenario][root_name][action]["episodes"]
                    means.append(
                        np.asarray(
                            [
                                rows[seed]["nav_error_mean_ned_m"]
                                for seed in DEV_SEEDS
                            ],
                            dtype=np.float64,
                        ).mean(axis=0)
                    )
    stacked = np.stack(means, axis=0)
    max_abs = float(np.abs(stacked).max())
    return {
        "mean_ned_m": stacked.mean(axis=0).tolist(),
        "max_abs_component_m": max_abs,
        "threshold_m": 0.05,
        "pass": max_abs < 0.05,
    }


def _m0_goal85_equivalence(
    matrix: dict[str, Any], goal85_root: Path
) -> dict[str, Any]:
    if not goal85_root.is_dir():
        return {"available": False, "pass": None, "mismatches": []}
    compared = 0
    mismatches: list[dict[str, Any]] = []
    keys = (
        "steps",
        "success",
        "goal_plane_miss",
        "timeout",
        "safety",
        "path_three_d_rmse_m",
        "terminal_eta_distance_m",
        "terminal_p_nav_distance_m",
        "mean_reward",
        "residual_l2_ratio",
    )
    for scenario in SCENARIOS:
        for root_name in ROOTS:
            for action in ACTIONS:
                action_dir, checkpoint = (
                    ("zero-residual", "s240")
                    if action == "zero-residual"
                    else ("policy", action)
                )
                path = goal85_root / scenario / root_name / action_dir / checkpoint
                try:
                    goal85 = _load_run(path)
                except FileNotFoundError:
                    mismatches.append({"path": str(path), "reason": "missing Goal85 run"})
                    continue
                current = matrix["m0"][scenario][root_name][action]
                for seed in DEV_SEEDS:
                    compared += 1
                    for key in keys:
                        left = current["episodes"][seed][key]
                        right = goal85["episodes"][seed][key]
                        if isinstance(left, bool) or isinstance(right, bool):
                            equal = bool(left) == bool(right)
                        else:
                            equal = abs(float(left) - float(right)) <= 1e-12
                        if not equal:
                            mismatches.append(
                                {
                                    "scenario": scenario,
                                    "root": root_name,
                                    "action": action,
                                    "seed": seed,
                                    "key": key,
                                    "current": left,
                                    "goal85": right,
                                }
                            )
    return {
        "available": True,
        "episodes_compared": compared,
        "mismatch_count": len(mismatches),
        "pass": not mismatches,
        "mismatches": mismatches[:20],
    }


def _detection(comparisons: dict[str, Any]) -> dict[str, Any]:
    primary_metrics = ("path_three_d_rmse_m", "terminal_eta_distance_m")
    cells: dict[str, Any] = {}
    counts = {"m05_vs_m0": 0, "m10_vs_m0": 0}
    for comparison, scenarios in comparisons.items():
        cells[comparison] = {}
        for scenario in SCENARIOS:
            cells[comparison][scenario] = {}
            for root_name in ROOTS:
                block = scenarios[scenario][root_name]["zero-residual"]
                detected = any(
                    not block[metric]["contains_zero"] for metric in primary_metrics
                )
                cells[comparison][scenario][root_name] = {
                    "detected": detected,
                    **{
                        metric: block[metric] for metric in primary_metrics
                    },
                }
                counts[comparison] += int(detected)
    return {
        "primary_action": "zero-residual",
        "primary_metrics": list(primary_metrics),
        "cells": cells,
        "detected_cells": counts,
    }


def _status(
    finite: dict[str, Any],
    scales: dict[str, Any],
    nav: dict[str, Any],
    equivalence: dict[str, Any],
    detection: dict[str, Any],
) -> str:
    implementation_checks = (
        finite["pass"],
        all(item["within_frozen_bounds"] for item in scales["levels"].values()),
        all(item["pass"] for item in scales["common_random_numbers"].values()),
        nav["pass"],
    )
    if not all(implementation_checks):
        return "INVALID_IMPLEMENTATION"
    if equivalence["available"] and not equivalence["pass"]:
        return "INVALID_M0_MISMATCH"
    if (
        detection["detected_cells"]["m10_vs_m0"] >= 3
        or detection["detected_cells"]["m05_vs_m0"] >= 2
    ):
        return "B1_DETECTED"
    return "MISMATCH_TOO_SMALL"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal85b B1 plant-mismatch audit",
        "",
        f"- status: `{payload['status']}`",
        f"- runs: `{payload['finite_audit']['runs']}`",
        f"- episodes: `{payload['episodes']}`",
        f"- m0 vs Goal85: `{payload['m0_goal85_equivalence']}`",
        "",
        "## Scale audit",
        "",
        "| level | within bounds | CRN pass | min drag | max drag | min fin | max fin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        block = payload["scale_audit"]["levels"][level]
        crn = payload["scale_audit"]["common_random_numbers"][level]
        lines.append(
            f"| {level} | {block['within_frozen_bounds']} | {crn['pass']} | "
            f"{block['min'][0]:.6f} | {block['max'][0]:.6f} | "
            f"{block['min'][2]:.6f} | {block['max'][2]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Zero-residual detection",
            "",
            "| comparison | scenario | root | detected | path Δ | path 95% CI | terminal Δ | terminal 95% CI |",
            "|---|---|---|---:|---:|---|---:|---|",
        ]
    )
    for comparison in ("m05_vs_m0", "m10_vs_m0"):
        for scenario in SCENARIOS:
            for root_name in ROOTS:
                cell = payload["detection"]["cells"][comparison][scenario][root_name]
                path = cell["path_three_d_rmse_m"]
                terminal = cell["terminal_eta_distance_m"]
                lines.append(
                    f"| {comparison} | {scenario} | {root_name} | {cell['detected']} | "
                    f"{path['mean']:+.6f} | [{path['ci_low']:+.6f}, {path['ci_high']:+.6f}] | "
                    f"{terminal['mean']:+.6f} | [{terminal['ci_low']:+.6f}, {terminal['ci_high']:+.6f}] |"
                )
    lines.extend(
        [
            "",
            "## Navigation bias",
            "",
            f"- max abs component: `{payload['nav_bias_audit']['max_abs_component_m']:.6g} m`",
            f"- threshold: `< {payload['nav_bias_audit']['threshold_m']} m`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    matrix = _load_matrix(args.root)
    comparisons = _comparisons(matrix)
    scales = _scale_audit(matrix)
    finite = _finite_audit(matrix)
    nav = _nav_bias_audit(matrix)
    equivalence = _m0_goal85_equivalence(matrix, args.goal85_root)
    detection = _detection(comparisons)
    status = _status(finite, scales, nav, equivalence, detection)
    payload = {
        "schema": "auv_pc_goal85b_b1.v1",
        "status": status,
        "root": str(args.root),
        "episodes": 60 * len(DEV_SEEDS),
        "finite_audit": finite,
        "scale_audit": scales,
        "nav_bias_audit": nav,
        "m0_goal85_equivalence": equivalence,
        "detection": detection,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output)}, indent=2))
    return 0 if status != "INVALID_IMPLEMENTATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
