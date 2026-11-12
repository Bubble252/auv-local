#!/usr/bin/env python3
"""Summarize S15 P1 path-observation screening with seed-paired gates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TAGS = ("default", "strict125")
PATH_ABLATIONS = ("none", "path-zero", "path-geom-zero", "path-context-zero")
PRIMARY_SUCCESS_DELTA = 0.05
PRIMARY_RMSE_IMPROVEMENT_M = 0.10
GUARDRAIL_SAFETY_DELTA = 0.02
GUARDRAIL_SATURATION_DELTA = 0.02
GUARDRAIL_TIMEOUT_DELTA = 0.05
PATH_DETECT_SUCCESS_DELTA = 0.05
PATH_DETECT_RMSE_DELTA_M = 0.10


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot compute standard deviation of an empty sequence")
    center = _mean(values)
    return float(math.sqrt(sum((value - center) ** 2 for value in values) / len(values)))


def _episode_map(data: dict[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(
            f"{path} lacks per-seed episodes; rerun with the updated evaluate_pc_ppo.py"
        )
    result: dict[int, dict[str, Any]] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row:
            raise ValueError(f"malformed episode row in {path}")
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"duplicate episode seed {seed} in {path}")
        result[seed] = row
    return result


def collect_runs(eval_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Collect evaluation summaries keyed by (arm, tag, path ablation)."""

    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(eval_root.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "auv_pc_ppo_evaluation.v1":
            continue
        history_ablation = data.get("history_ablation", "none")
        if history_ablation is None:
            history_ablation = "none"
        if history_ablation != "none":
            continue
        relative = path.relative_to(eval_root)
        arm = next((part for part in relative.parts if part.startswith("P1_")), None)
        if arm is None:
            arm = next(
                (
                    part
                    for part in relative.parts
                    if part.startswith("A") and "_stride" in part
                ),
                None,
            )
        if arm is None:
            recorded_arm = data.get("arm")
            arm = (
                str(recorded_arm)
                if recorded_arm not in (None, "")
                else relative.parent.name
            )
        tag = next((value for value in TAGS if value in relative.parts), None)
        if tag is None:
            environment = str(data.get("environment_config", "")).lower()
            if "strict125" in environment or "strict_125" in environment:
                tag = "strict125"
            elif "rewardscale001" in environment or "c3_random_realistic" in environment:
                tag = "default"
        if tag is None:
            raise ValueError(f"cannot infer evaluation tag from {path}")
        ablation = str(data.get("path_ablation", "none"))
        if ablation == "None":
            ablation = "none"
        if ablation not in PATH_ABLATIONS:
            raise ValueError(f"unknown path ablation {ablation!r} in {path}")
        key = (arm, tag, ablation)
        if key in runs:
            raise ValueError(f"duplicate evaluation summary for {key}: {path}")
        runs[key] = {
            "path": str(path),
            "data": data,
            "episodes": _episode_map(data, path),
        }
    if not runs:
        raise ValueError(f"no PC evaluation summaries found under {eval_root}")
    return runs


def _require_group(
    runs: dict[tuple[str, str, str], dict[str, Any]],
    arm: str,
    tag: str,
    ablation: str,
) -> dict[str, Any]:
    key = (arm, tag, ablation)
    if key not in runs:
        raise ValueError(f"missing evaluation group: {key}")
    return runs[key]


def _aligned_episodes(
    base: dict[str, Any],
    treatment: dict[str, Any],
    *,
    expected_seeds: set[int] | None,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    base_seeds = set(base["episodes"])
    treatment_seeds = set(treatment["episodes"])
    if base_seeds != treatment_seeds:
        raise ValueError(
            f"seed mismatch between {base['path']} and {treatment['path']}: "
            f"{sorted(base_seeds ^ treatment_seeds)}"
        )
    if expected_seeds is not None and base_seeds != expected_seeds:
        raise ValueError(
            f"unexpected evaluation seeds in {base['path']}: "
            f"expected {sorted(expected_seeds)}, got {sorted(base_seeds)}"
        )
    seeds = sorted(base_seeds)
    return (
        seeds,
        [base["episodes"][seed] for seed in seeds],
        [treatment["episodes"][seed] for seed in seeds],
    )


def _paired_metric(
    base_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    base_values = [float(row[key]) for row in base_rows]
    treatment_values = [float(row[key]) for row in treatment_rows]
    deltas = [
        treatment - base
        for base, treatment in zip(base_values, treatment_values, strict=True)
    ]
    return {
        "baseline_mean": _mean(base_values),
        "treatment_mean": _mean(treatment_values),
        "paired_delta_mean": _mean(deltas),
        "paired_delta_std": _std(deltas),
        "paired_delta_min": float(min(deltas)),
        "paired_delta_max": float(max(deltas)),
    }


def _binary_metric(
    base_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    base_values = [1.0 if bool(row[key]) else 0.0 for row in base_rows]
    treatment_values = [1.0 if bool(row[key]) else 0.0 for row in treatment_rows]
    deltas = [
        treatment - base
        for base, treatment in zip(base_values, treatment_values, strict=True)
    ]
    flips = [
        {
            "seed": int(base["seed"]),
            "baseline": bool(base[key]),
            "treatment": bool(treatment[key]),
        }
        for base, treatment in zip(base_rows, treatment_rows, strict=True)
        if bool(base[key]) != bool(treatment[key])
    ]
    return {
        "baseline_mean": _mean(base_values),
        "treatment_mean": _mean(treatment_values),
        "paired_delta_mean": _mean(deltas),
        "paired_delta_std": _std(deltas),
        "paired_delta_min": float(min(deltas)),
        "paired_delta_max": float(max(deltas)),
        "flips": flips,
    }


def compare_groups(
    base: dict[str, Any],
    treatment: dict[str, Any],
    *,
    expected_seeds: set[int] | None,
) -> dict[str, Any]:
    seeds, base_rows, treatment_rows = _aligned_episodes(
        base, treatment, expected_seeds=expected_seeds
    )
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "success": _binary_metric(base_rows, treatment_rows, "success"),
        "timeout": _binary_metric(base_rows, treatment_rows, "timeout"),
        "safety": _binary_metric(base_rows, treatment_rows, "safety"),
        "material_saturation_rate": _paired_metric(
            base_rows, treatment_rows, "material_saturation_rate"
        ),
        "path_three_d_rmse_m": _paired_metric(
            base_rows, treatment_rows, "path_three_d_rmse_m"
        ),
        "path_completion_ratio": _paired_metric(
            base_rows, treatment_rows, "path_completion_ratio"
        ),
        "mean_reward": _paired_metric(base_rows, treatment_rows, "mean_reward"),
    }


def _effect(
    normal: dict[str, Any],
    ablated: dict[str, Any],
    *,
    expected_seeds: set[int] | None,
) -> dict[str, Any]:
    comparison = compare_groups(normal, ablated, expected_seeds=expected_seeds)
    success_drop = -comparison["success"]["paired_delta_mean"]
    rmse_worsening = comparison["path_three_d_rmse_m"]["paired_delta_mean"]
    ablated_episodes = list(ablated["episodes"].values())
    action_l2 = [
        float(row.get("path_action_delta_l2_mean", 0.0))
        for row in ablated_episodes
    ]
    action_rpm = [
        float(row.get("path_action_delta_rpm_abs_mean_rpm", 0.0))
        for row in ablated_episodes
    ]
    detected = (
        success_drop >= PATH_DETECT_SUCCESS_DELTA
        or rmse_worsening >= PATH_DETECT_RMSE_DELTA_M
    )
    return {
        "success_drop": success_drop,
        "rmse_worsening_m": rmse_worsening,
        "timeout_change": comparison["timeout"]["paired_delta_mean"],
        "material_saturation_change": comparison["material_saturation_rate"][
            "paired_delta_mean"
        ],
        "path_action_delta_l2_mean": _mean(action_l2),
        "path_action_delta_l2_max": float(max(action_l2)),
        "path_action_delta_rpm_abs_mean_rpm": _mean(action_rpm),
        "detected": bool(detected),
        "comparison": comparison,
    }


def _decision(
    *,
    arm: str,
    normal_by_tag: dict[str, dict[str, Any]],
    path_effects_by_tag: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    default = normal_by_tag["default"]
    strict = normal_by_tag["strict125"]
    success_delta = default["success"]["paired_delta_mean"]
    strict_delta = strict["success"]["paired_delta_mean"]
    rmse_improvement = -default["path_three_d_rmse_m"]["paired_delta_mean"]
    primary_signal = (
        success_delta >= PRIMARY_SUCCESS_DELTA
        or strict_delta >= PRIMARY_SUCCESS_DELTA
        or rmse_improvement >= PRIMARY_RMSE_IMPROVEMENT_M
    )
    max_safety = max(
        default["safety"]["paired_delta_mean"],
        strict["safety"]["paired_delta_mean"],
    )
    max_saturation = max(
        default["material_saturation_rate"]["paired_delta_mean"],
        strict["material_saturation_rate"]["paired_delta_mean"],
    )
    max_timeout = max(
        default["timeout"]["paired_delta_mean"],
        strict["timeout"]["paired_delta_mean"],
    )
    guardrails_pass = (
        max_safety <= GUARDRAIL_SAFETY_DELTA
        and max_saturation <= GUARDRAIL_SATURATION_DELTA
        and max_timeout <= GUARDRAIL_TIMEOUT_DELTA
    )
    path_zero = path_effects_by_tag["default"]["path-zero"]
    path_attribution = bool(path_zero["detected"])

    if not primary_signal:
        status = "no_signal"
    elif not guardrails_pass:
        status = "fail_guardrail"
    elif not path_attribution:
        status = "no_path_attribution"
    else:
        status = "pass"

    return {
        "arm": arm,
        "status": status,
        "primary_signal": bool(primary_signal),
        "guardrails_pass": bool(guardrails_pass),
        "path_attribution": bool(path_attribution),
        "default_success_delta": success_delta,
        "strict125_success_delta": strict_delta,
        "rmse_improvement_m": rmse_improvement,
        "max_safety_delta": max_safety,
        "max_material_saturation_delta": max_saturation,
        "max_timeout_delta": max_timeout,
        "path_zero_success_drop_default": path_zero["success_drop"],
        "path_zero_rmse_worsening_default_m": path_zero["rmse_worsening_m"],
        "path_zero_action_delta_l2_mean": path_zero["path_action_delta_l2_mean"],
        "selection_key": (success_delta, strict_delta, rmse_improvement),
    }


def summarize(
    baseline_eval_root: Path,
    treatment_eval_root: Path,
    *,
    baseline_arm: str,
    treatment_arm: str,
    tags: tuple[str, ...] = TAGS,
    path_ablations: tuple[str, ...] = PATH_ABLATIONS,
    expected_seeds: set[int] | None = None,
) -> dict[str, Any]:
    baseline_runs = collect_runs(baseline_eval_root)
    treatment_runs = collect_runs(treatment_eval_root)
    normal_by_tag: dict[str, Any] = {}
    effects_by_tag: dict[str, dict[str, Any]] = {}
    for tag in tags:
        normal_by_tag[tag] = compare_groups(
            _require_group(baseline_runs, baseline_arm, tag, "none"),
            _require_group(treatment_runs, treatment_arm, tag, "none"),
            expected_seeds=expected_seeds,
        )
        effects_by_tag[tag] = {}
        for ablation in path_ablations:
            if ablation == "none":
                continue
            effects_by_tag[tag][ablation] = _effect(
                _require_group(treatment_runs, treatment_arm, tag, "none"),
                _require_group(treatment_runs, treatment_arm, tag, ablation),
                expected_seeds=expected_seeds,
            )

    decision = _decision(
        arm=treatment_arm,
        normal_by_tag=normal_by_tag,
        path_effects_by_tag=effects_by_tag,
    )
    return {
        "schema_version": "auv_pc_abc_p1_summary.v1",
        "baseline_eval_root": str(baseline_eval_root),
        "treatment_eval_root": str(treatment_eval_root),
        "baseline_arm": baseline_arm,
        "treatment_arm": treatment_arm,
        "tags": list(tags),
        "path_ablations": list(path_ablations),
        "expected_seeds": None if expected_seeds is None else sorted(expected_seeds),
        "runs": {
            "baseline": {
                f"{arm}/{tag}/{ablation}": {
                    "summary": run["path"],
                    "seed_count": len(run["episodes"]),
                }
                for (arm, tag, ablation), run in sorted(baseline_runs.items())
            },
            "treatment": {
                f"{arm}/{tag}/{ablation}": {
                    "summary": run["path"],
                    "seed_count": len(run["episodes"]),
                }
                for (arm, tag, ablation), run in sorted(treatment_runs.items())
            },
        },
        "comparisons": {
            "normal": normal_by_tag,
            "path_ablations": effects_by_tag,
        },
        "decision": decision,
        "gate_definition": {
            "primary_success_delta": PRIMARY_SUCCESS_DELTA,
            "primary_rmse_improvement_m": PRIMARY_RMSE_IMPROVEMENT_M,
            "max_safety_delta": GUARDRAIL_SAFETY_DELTA,
            "max_material_saturation_delta": GUARDRAIL_SATURATION_DELTA,
            "max_timeout_delta": GUARDRAIL_TIMEOUT_DELTA,
            "path_detection_success_delta": PATH_DETECT_SUCCESS_DELTA,
            "path_detection_rmse_delta_m": PATH_DETECT_RMSE_DELTA_M,
            "interpretation": (
                "P1 passes only when the path-enabled arm beats the A0/no-path "
                "baseline on a primary metric, all guardrails hold, and the "
                "default-tag path-zero ablation worsens success by at least "
                "0.05 or 3D RMSE by at least 0.10 m. Block-zero ablations are "
                "reported as diagnostics; the protocol did not specify a "
                "separate numeric threshold for them."
            ),
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# S15 P1 path-observation screening",
        "",
        f"- baseline: `{summary['baseline_arm']}` ({summary['baseline_eval_root']})",
        f"- treatment: `{summary['treatment_arm']}` ({summary['treatment_eval_root']})",
        f"- status: `{summary['decision']['status']}`",
        "",
        "## Paired comparisons against baseline",
        "",
        "| tag | success delta | strict125 delta | 3D RMSE improvement (m) | timeout delta | safety delta | saturation delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, comparison in summary["comparisons"]["normal"].items():
        lines.append(
            "| {tag} | {success} | {strict} | {rmse} | {timeout} | {safety} | {sat} |".format(
                tag=tag,
                success=_fmt(comparison["success"]["paired_delta_mean"]),
                strict=_fmt(comparison["success"]["paired_delta_mean"]),
                rmse=_fmt(-comparison["path_three_d_rmse_m"]["paired_delta_mean"]),
                timeout=_fmt(comparison["timeout"]["paired_delta_mean"]),
                safety=_fmt(comparison["safety"]["paired_delta_mean"]),
                sat=_fmt(comparison["material_saturation_rate"]["paired_delta_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Path ablations",
            "",
            "| tag | ablation | success drop | RMSE worsening (m) | timeout change | action L2 mean | detected |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for tag, effects in summary["comparisons"]["path_ablations"].items():
        for ablation, effect in effects.items():
            lines.append(
                "| {tag} | {ablation} | {success} | {rmse} | {timeout} | {action} | {detected} |".format(
                    tag=tag,
                    ablation=ablation,
                    success=_fmt(effect["success_drop"]),
                    rmse=_fmt(effect["rmse_worsening_m"]),
                    timeout=_fmt(effect["timeout_change"]),
                    action=_fmt(effect["path_action_delta_l2_mean"]),
                    detected=_fmt(effect["detected"]),
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- primary signal: `{_fmt(summary['decision']['primary_signal'])}`",
            f"- guardrails: `{_fmt(summary['decision']['guardrails_pass'])}`",
            f"- path attribution: `{_fmt(summary['decision']['path_attribution'])}`",
            f"- default success delta: `{_fmt(summary['decision']['default_success_delta'])}`",
            f"- strict125 success delta: `{_fmt(summary['decision']['strict125_success_delta'])}`",
            f"- 3D RMSE improvement: `{_fmt(summary['decision']['rmse_improvement_m'])} m`",
            f"- path-zero success drop (default): `{_fmt(summary['decision']['path_zero_success_drop_default'])}`",
            f"- path-zero RMSE worsening (default): `{_fmt(summary['decision']['path_zero_rmse_worsening_default_m'])} m`",
            "",
            "## Gate definition",
            "",
            "```json",
            json.dumps(summary["gate_definition"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-eval-root", required=True, type=Path)
    parser.add_argument("--treatment-eval-root", required=True, type=Path)
    parser.add_argument("--baseline-arm", required=True)
    parser.add_argument("--treatment-arm", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", action="append", default=None)
    parser.add_argument("--path-ablation", action="append", default=None)
    parser.add_argument("--expected-seed-start", type=int, default=None)
    parser.add_argument("--expected-seed-count", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tags = tuple(args.tag or TAGS)
    path_ablations = tuple(args.path_ablation or PATH_ABLATIONS)
    expected_seeds = None
    if (args.expected_seed_start is None) != (args.expected_seed_count is None):
        raise ValueError(
            "--expected-seed-start and --expected-seed-count must be used together"
        )
    if args.expected_seed_count is not None:
        expected_seeds = set(
            range(
                args.expected_seed_start,
                args.expected_seed_start + args.expected_seed_count,
            )
        )
    summary = summarize(
        args.baseline_eval_root,
        args.treatment_eval_root,
        baseline_arm=args.baseline_arm,
        treatment_arm=args.treatment_arm,
        tags=tags,
        path_ablations=path_ablations,
        expected_seeds=expected_seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
