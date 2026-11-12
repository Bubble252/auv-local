#!/usr/bin/env python3
"""Summarize temporal A/B arms against a paired baseline with a non-inferiority gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TAGS = ("default", "strict125", "grid_hs")
ABLATIONS = ("none", "history-zero", "history-reverse")
SUCCESS_NONINFERIORITY = 0.02
RMSE_RATIO_LIMIT = 1.05
GUARDRAIL_DELTA = 0.02


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    center = _mean(values)
    return float(math.sqrt(sum((value - center) ** 2 for value in values) / len(values)))


def _infer_arm(path: Path, data: dict[str, Any], known_arms: set[str]) -> str | None:
    for part in path.parts:
        if part in known_arms:
            return part
    text = str(data.get("training_config", "")).lower()
    for arm in known_arms:
        if arm.lower() in text:
            return arm
    return None


def _infer_tag(path: Path, data: dict[str, Any]) -> str:
    parts = {part.lower() for part in path.parts}
    for tag in TAGS:
        if tag in parts:
            return tag
    text = str(data.get("environment_config", "")).lower()
    if "strict125" in text or "strict_125" in text:
        return "strict125"
    if "grid_high" in text or "grid_hs" in text:
        return "grid_hs"
    if "rewardscale001" in text or "c3_random_realistic" in text:
        return "default"
    raise ValueError(f"cannot infer evaluation tag from {path}")


def _episodes(data: dict[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    rows = data.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} lacks per-seed episodes")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"duplicate seed {seed} in {path}")
        result[seed] = row
    return result


def collect(root: Path, *, known_arms: set[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "auv_pc_ppo_evaluation.v1":
            continue
        arm = _infer_arm(path.relative_to(root), data, known_arms)
        if arm is None:
            continue
        tag = _infer_tag(path.relative_to(root), data)
        ablation = str(data.get("history_ablation", "none"))
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown history ablation {ablation!r} in {path}")
        key = (arm, tag, ablation)
        if key in runs:
            raise ValueError(f"duplicate evaluation group {key}: {path}")
        runs[key] = {"path": str(path), "data": data, "episodes": _episodes(data, path)}
    return runs


def _group(
    runs: dict[tuple[str, str, str], dict[str, Any]],
    arm: str,
    tag: str,
    ablation: str = "none",
) -> dict[str, Any]:
    key = (arm, tag, ablation)
    if key not in runs:
        raise ValueError(f"missing evaluation group {key}")
    return runs[key]


def _aligned(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    expected_seeds: set[int],
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    if set(baseline["episodes"]) != set(treatment["episodes"]):
        raise ValueError("baseline and treatment seeds differ")
    if set(baseline["episodes"]) != expected_seeds:
        raise ValueError(
            f"unexpected seeds: expected {sorted(expected_seeds)}, "
            f"got {sorted(baseline['episodes'])}"
        )
    seeds = sorted(expected_seeds)
    return seeds, [baseline["episodes"][s] for s in seeds], [treatment["episodes"][s] for s in seeds]


def _paired(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    key: str,
    *,
    boolean: bool = False,
) -> dict[str, Any]:
    base_values = [float(bool(row[key])) if boolean else float(row[key]) for row in baseline]
    treatment_values = [
        float(bool(row[key])) if boolean else float(row[key]) for row in treatment
    ]
    deltas = [b - a for a, b in zip(base_values, treatment_values, strict=True)]
    return {
        "baseline_mean": _mean(base_values),
        "treatment_mean": _mean(treatment_values),
        "paired_delta_mean": _mean(deltas),
        "paired_delta_std": _std(deltas),
        "paired_delta_min": min(deltas),
        "paired_delta_max": max(deltas),
    }


def compare(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    expected_seeds: set[int],
) -> dict[str, Any]:
    seeds, base_rows, treatment_rows = _aligned(baseline, treatment, expected_seeds)
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "success": _paired(base_rows, treatment_rows, "success", boolean=True),
        "strict_success": _paired(base_rows, treatment_rows, "success", boolean=True),
        "timeout": _paired(base_rows, treatment_rows, "timeout", boolean=True),
        "safety": _paired(base_rows, treatment_rows, "safety", boolean=True),
        "material_saturation_rate": _paired(
            base_rows, treatment_rows, "material_saturation_rate"
        ),
        "path_three_d_rmse_m": _paired(base_rows, treatment_rows, "path_three_d_rmse_m"),
        "path_completion_ratio": _paired(
            base_rows, treatment_rows, "path_completion_ratio"
        ),
        "mean_reward": _paired(base_rows, treatment_rows, "mean_reward"),
    }


def summarize(
    *,
    baseline_root: Path,
    baseline_arm: str,
    treatment_root: Path,
    treatment_arm: str,
    expected_seeds: set[int],
) -> dict[str, Any]:
    known = {baseline_arm, treatment_arm}
    baseline_runs = collect(baseline_root, known_arms=known)
    treatment_runs = collect(treatment_root, known_arms=known)
    comparisons: dict[str, Any] = {}
    for tag in TAGS:
        comparisons[tag] = compare(
            _group(baseline_runs, baseline_arm, tag),
            _group(treatment_runs, treatment_arm, tag),
            expected_seeds,
        )

    default = comparisons["default"]
    strict = comparisons["strict125"]
    success_drop = -min(
        default["success"]["paired_delta_mean"],
        strict["success"]["paired_delta_mean"],
    )
    strict_drop = -min(
        default["strict_success"]["paired_delta_mean"],
        strict["strict_success"]["paired_delta_mean"],
    )
    rmse_ratio = max(
        default["path_three_d_rmse_m"]["treatment_mean"]
        / max(default["path_three_d_rmse_m"]["baseline_mean"], 1e-12),
        strict["path_three_d_rmse_m"]["treatment_mean"]
        / max(strict["path_three_d_rmse_m"]["baseline_mean"], 1e-12),
    )
    timeout_increase = max(
        default["timeout"]["paired_delta_mean"],
        strict["timeout"]["paired_delta_mean"],
    )
    saturation_increase = max(
        default["material_saturation_rate"]["paired_delta_mean"],
        strict["material_saturation_rate"]["paired_delta_mean"],
    )
    safety_increase = max(
        default["safety"]["paired_delta_mean"],
        strict["safety"]["paired_delta_mean"],
    )
    passed = bool(
        success_drop <= SUCCESS_NONINFERIORITY
        and strict_drop <= SUCCESS_NONINFERIORITY
        and rmse_ratio <= RMSE_RATIO_LIMIT
        and timeout_increase <= GUARDRAIL_DELTA
        and saturation_increase <= GUARDRAIL_DELTA
        and safety_increase <= GUARDRAIL_DELTA
    )
    return {
        "schema_version": "auv_pc_abc_temporal_summary.v1",
        "baseline_root": str(baseline_root),
        "baseline_arm": baseline_arm,
        "treatment_root": str(treatment_root),
        "treatment_arm": treatment_arm,
        "expected_seeds": sorted(expected_seeds),
        "comparisons": comparisons,
        "decision": {
            "status": "pass_non_inferior" if passed else "fail_non_inferior",
            "passed": passed,
            "success_drop": success_drop,
            "strict_success_drop": strict_drop,
            "rmse_ratio": rmse_ratio,
            "timeout_increase": timeout_increase,
            "material_saturation_increase": saturation_increase,
            "safety_increase": safety_increase,
            "gate": {
                "success_drop_max": SUCCESS_NONINFERIORITY,
                "rmse_ratio_max": RMSE_RATIO_LIMIT,
                "timeout_increase_max": GUARDRAIL_DELTA,
                "saturation_increase_max": GUARDRAIL_DELTA,
                "safety_increase_max": GUARDRAIL_DELTA,
            },
        },
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# S15 temporal paired summary",
        "",
        f"- baseline: `{summary['baseline_arm']}`",
        f"- treatment: `{summary['treatment_arm']}`",
        f"- status: `{summary['decision']['status']}`",
        "",
        "| tag | success delta | strict delta | RMSE ratio | timeout delta | saturation delta | safety delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in TAGS:
        row = summary["comparisons"][tag]
        rmse_ratio = row["path_three_d_rmse_m"]["treatment_mean"] / max(
            row["path_three_d_rmse_m"]["baseline_mean"], 1e-12
        )
        lines.append(
            "| {tag} | {success:.4f} | {strict:.4f} | {rmse:.4f} | {timeout:.4f} | {sat:.4f} | {safety:.4f} |".format(
                tag=tag,
                success=row["success"]["paired_delta_mean"],
                strict=row["strict_success"]["paired_delta_mean"],
                rmse=rmse_ratio,
                timeout=row["timeout"]["paired_delta_mean"],
                sat=row["material_saturation_rate"]["paired_delta_mean"],
                safety=row["safety"]["paired_delta_mean"],
            )
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            f"- success drop: `{decision['success_drop']:.4f}`",
            f"- strict success drop: `{decision['strict_success_drop']:.4f}`",
            f"- RMSE ratio: `{decision['rmse_ratio']:.4f}`",
            f"- timeout increase: `{decision['timeout_increase']:.4f}`",
            f"- saturation increase: `{decision['material_saturation_increase']:.4f}`",
            f"- safety increase: `{decision['safety_increase']:.4f}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-arm", required=True)
    parser.add_argument("--treatment-root", type=Path, required=True)
    parser.add_argument("--treatment-arm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seed-start", type=int, required=True)
    parser.add_argument("--expected-seed-count", type=int, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    expected_seeds = set(
        range(args.expected_seed_start, args.expected_seed_start + args.expected_seed_count)
    )
    summary = summarize(
        baseline_root=args.baseline_root,
        baseline_arm=args.baseline_arm,
        treatment_root=args.treatment_root,
        treatment_arm=args.treatment_arm,
        expected_seeds=expected_seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(render(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_pass and not summary["decision"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
