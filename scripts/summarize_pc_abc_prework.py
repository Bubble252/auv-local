#!/usr/bin/env python3
"""Summarize S15 P0/P1/P2/P3 screening results with seed-paired gates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ARMS = ("A0_stride1", "A1_stride5", "A2_stride10")
TAGS = ("default", "strict125", "grid_hs")
ABLATIONS = ("none", "history-zero", "history-reverse")
PRIMARY_SUCCESS_DELTA = 0.05
PRIMARY_RMSE_IMPROVEMENT_M = 0.10
GUARDRAIL_SAFETY_DELTA = 0.02
GUARDRAIL_SATURATION_DELTA = 0.02
GUARDRAIL_TIMEOUT_DELTA = 0.05
HISTORY_DETECT_SUCCESS_DELTA = 0.01
HISTORY_DETECT_RMSE_DELTA_M = 0.02


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


def _arm_from_path(path: Path, data: dict[str, Any]) -> str:
    for part in path.parts:
        if part in ARMS:
            return part
    training_config = str(data.get("training_config", ""))
    lowered = training_config.lower()
    for arm, token in (
        ("A0_stride1", "a0_stride1"),
        ("A1_stride5", "a1_stride5"),
        ("A2_stride10", "a2_stride10"),
    ):
        if token in lowered:
            return arm
    raise ValueError(f"cannot infer ABC arm from {path}")


def _tag_from_path(path: Path, data: dict[str, Any]) -> str:
    parts = {part.lower() for part in path.parts}
    for tag in TAGS:
        if tag in parts:
            return tag
    environment = str(data.get("environment_config", data.get("environment_config_override", "")))
    lowered = environment.lower()
    if "strict125" in lowered or "strict_125" in lowered:
        return "strict125"
    if "rewardscale001" in lowered or "c3_random_realistic" in lowered:
        return "default"
    raise ValueError(f"cannot infer evaluation tag from {path}")


def _ablation_from_data(path: Path, data: dict[str, Any]) -> str:
    value = str(data.get("history_ablation", "none"))
    if value not in ABLATIONS:
        raise ValueError(f"unknown history ablation {value!r} in {path}")
    return value


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
    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(eval_root.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "auv_pc_ppo_evaluation.v1":
            continue
        arm = _arm_from_path(path.relative_to(eval_root), data)
        tag = _tag_from_path(path.relative_to(eval_root), data)
        ablation = _ablation_from_data(path, data)
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


def _success_metric(
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
        "success": _success_metric(base_rows, treatment_rows, "success"),
        "strict_success": _success_metric(base_rows, treatment_rows, "success"),
        "timeout": _success_metric(base_rows, treatment_rows, "timeout"),
        "safety": _success_metric(base_rows, treatment_rows, "safety"),
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


def _history_effect(
    normal: dict[str, Any],
    ablated: dict[str, Any],
    *,
    expected_seeds: set[int] | None,
) -> dict[str, Any]:
    comparison = compare_groups(normal, ablated, expected_seeds=expected_seeds)
    success_drop = -comparison["success"]["paired_delta_mean"]
    rmse_worsening = comparison["path_three_d_rmse_m"]["paired_delta_mean"]
    detected = (
        abs(success_drop) >= HISTORY_DETECT_SUCCESS_DELTA
        or abs(rmse_worsening) >= HISTORY_DETECT_RMSE_DELTA_M
    )
    return {
        "success_drop": success_drop,
        "rmse_worsening_m": rmse_worsening,
        "timeout_change": comparison["timeout"]["paired_delta_mean"],
        "material_saturation_change": comparison["material_saturation_rate"][
            "paired_delta_mean"
        ],
        "detected": bool(detected),
        "comparison": comparison,
    }


def _treatment_decision(
    arm: str,
    normal_by_tag: dict[str, dict[str, Any]],
    *,
    expected_seeds: set[int] | None,
) -> dict[str, Any]:
    default = normal_by_tag.get("default")
    strict = normal_by_tag.get("strict125")
    if default is None or strict is None:
        raise ValueError(f"{arm} requires both default and strict125 comparisons")

    success_delta = default["success"]["paired_delta_mean"]
    strict_delta = strict["success"]["paired_delta_mean"]
    rmse_improvement = -default["path_three_d_rmse_m"]["paired_delta_mean"]
    primary_signal = (
        success_delta >= PRIMARY_SUCCESS_DELTA
        or strict_delta >= PRIMARY_SUCCESS_DELTA
        or rmse_improvement >= PRIMARY_RMSE_IMPROVEMENT_M
    )
    max_safety_delta = max(
        default["safety"]["paired_delta_mean"],
        strict["safety"]["paired_delta_mean"],
    )
    max_saturation_delta = max(
        default["material_saturation_rate"]["paired_delta_mean"],
        strict["material_saturation_rate"]["paired_delta_mean"],
    )
    max_timeout_delta = max(
        default["timeout"]["paired_delta_mean"],
        strict["timeout"]["paired_delta_mean"],
    )
    guardrails_pass = (
        max_safety_delta <= GUARDRAIL_SAFETY_DELTA
        and max_saturation_delta <= GUARDRAIL_SATURATION_DELTA
        and max_timeout_delta <= GUARDRAIL_TIMEOUT_DELTA
    )

    if not primary_signal:
        status = "no_signal"
    elif not guardrails_pass:
        status = "fail_guardrail"
    else:
        status = "pass_pending_history_check"

    return {
        "arm": arm,
        "primary_signal": bool(primary_signal),
        "guardrails_pass": bool(guardrails_pass),
        "status_before_history": status,
        "default_success_delta": success_delta,
        "strict125_success_delta": strict_delta,
        "rmse_improvement_m": rmse_improvement,
        "max_safety_delta": max_safety_delta,
        "max_material_saturation_delta": max_saturation_delta,
        "max_timeout_delta": max_timeout_delta,
        "selection_key": (
            success_delta,
            strict_delta,
            rmse_improvement,
        ),
    }


def summarize(
    eval_root: Path,
    *,
    baseline_arm: str = "A0_stride1",
    treatments: tuple[str, ...] = ("A1_stride5", "A2_stride10"),
    tags: tuple[str, ...] = TAGS,
    ablations: tuple[str, ...] = ABLATIONS,
    expected_seeds: set[int] | None = None,
) -> dict[str, Any]:
    runs = collect_runs(eval_root)
    comparisons: dict[str, Any] = {}
    decisions: dict[str, Any] = {}

    for treatment in treatments:
        per_tag: dict[str, Any] = {}
        for tag in tags:
            per_tag[tag] = compare_groups(
                _require_group(runs, baseline_arm, tag, "none"),
                _require_group(runs, treatment, tag, "none"),
                expected_seeds=expected_seeds,
            )

        history: dict[str, Any] = {}
        for ablation in ablations:
            if ablation == "none":
                continue
            normal = _require_group(runs, treatment, "default", "none")
            ablated = _require_group(runs, treatment, "default", ablation)
            history[ablation] = _history_effect(
                normal, ablated, expected_seeds=expected_seeds
            )

        decision = _treatment_decision(
            treatment,
            per_tag,
            expected_seeds=expected_seeds,
        )
        history_detected = any(item["detected"] for item in history.values())
        if decision["status_before_history"] == "pass_pending_history_check":
            decision["status"] = (
                "pass" if history_detected else "pass_without_history_attribution"
            )
        else:
            decision["status"] = decision["status_before_history"]
        decision["history_detected"] = bool(history_detected)
        decision["history"] = history
        comparisons[treatment] = {"normal": per_tag, "history": history}
        decisions[treatment] = decision

    eligible = [
        decision
        for decision in decisions.values()
        if decision["status"] == "pass"
    ]
    if eligible:
        selected = max(eligible, key=lambda item: item["selection_key"])
        selected_arm = selected["arm"]
        selection_status = "treatment"
    else:
        selected_arm = baseline_arm
        selection_status = "baseline_no_signal"

    return {
        "schema_version": "auv_pc_abc_prework_summary.v1",
        "eval_root": str(eval_root),
        "baseline_arm": baseline_arm,
        "treatments": list(treatments),
        "tags": list(tags),
        "history_ablations": list(ablations),
        "expected_seeds": None if expected_seeds is None else sorted(expected_seeds),
        "runs": {
            f"{arm}/{tag}/{ablation}": {
                "summary": run["path"],
                "seed_count": len(run["episodes"]),
            }
            for (arm, tag, ablation), run in sorted(runs.items())
        },
        "comparisons": comparisons,
        "decisions": decisions,
        "selected_arm": selected_arm,
        "selection_status": selection_status,
        "gate_definition": {
            "primary_success_delta": PRIMARY_SUCCESS_DELTA,
            "primary_rmse_improvement_m": PRIMARY_RMSE_IMPROVEMENT_M,
            "max_safety_delta": GUARDRAIL_SAFETY_DELTA,
            "max_material_saturation_delta": GUARDRAIL_SATURATION_DELTA,
            "max_timeout_delta": GUARDRAIL_TIMEOUT_DELTA,
            "history_detection_success_delta": HISTORY_DETECT_SUCCESS_DELTA,
            "history_detection_rmse_delta_m": HISTORY_DETECT_RMSE_DELTA_M,
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
        "# S15 P0 A-stride screening",
        "",
        f"- eval root: `{summary['eval_root']}`",
        f"- baseline: `{summary['baseline_arm']}`",
        f"- selected arm: `{summary['selected_arm']}`",
        f"- selection status: `{summary['selection_status']}`",
        "",
        "## Paired normal comparisons",
        "",
        "| arm | tag | success delta | strict125 delta | 3D RMSE improvement (m) | timeout delta | safety delta | saturation delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, payload in summary["comparisons"].items():
        for tag, comparison in payload["normal"].items():
            lines.append(
                "| {arm} | {tag} | {success} | {strict} | {rmse} | {timeout} | {safety} | {sat} |".format(
                    arm=arm,
                    tag=tag,
                    success=_fmt(comparison["success"]["paired_delta_mean"]),
                    strict=_fmt(comparison["strict_success"]["paired_delta_mean"]),
                    rmse=_fmt(-comparison["path_three_d_rmse_m"]["paired_delta_mean"]),
                    timeout=_fmt(comparison["timeout"]["paired_delta_mean"]),
                    safety=_fmt(comparison["safety"]["paired_delta_mean"]),
                    sat=_fmt(
                        comparison["material_saturation_rate"]["paired_delta_mean"]
                    ),
                )
            )
    lines.extend(
        [
            "",
            "## History ablation diagnostics",
            "",
            "| arm | ablation | success drop | RMSE worsening (m) | timeout change | detected |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for arm, payload in summary["comparisons"].items():
        for ablation, effect in payload["history"].items():
            lines.append(
                "| {arm} | {ablation} | {success} | {rmse} | {timeout} | {detected} |".format(
                    arm=arm,
                    ablation=ablation,
                    success=_fmt(effect["success_drop"]),
                    rmse=_fmt(effect["rmse_worsening_m"]),
                    timeout=_fmt(effect["timeout_change"]),
                    detected=_fmt(effect["detected"]),
                )
            )
    lines.extend(["", "## Decisions", ""])
    for arm, decision in summary["decisions"].items():
        lines.extend(
            [
                f"### {arm}",
                "",
                f"- status: `{decision['status']}`",
                f"- primary signal: `{_fmt(decision['primary_signal'])}`",
                f"- guardrails: `{_fmt(decision['guardrails_pass'])}`",
                f"- history attribution: `{_fmt(decision['history_detected'])}`",
                f"- default success delta: `{_fmt(decision['default_success_delta'])}`",
                f"- strict125 success delta: `{_fmt(decision['strict125_success_delta'])}`",
                f"- 3D RMSE improvement: `{_fmt(decision['rmse_improvement_m'])} m`",
                "",
            ]
        )
    lines.extend(
        [
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
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-arm", default="A0_stride1")
    parser.add_argument("--treatment-arm", action="append", default=None)
    parser.add_argument("--tag", action="append", default=None)
    parser.add_argument("--history-ablation", action="append", default=None)
    parser.add_argument("--expected-seed-start", type=int, default=None)
    parser.add_argument("--expected-seed-count", type=int, default=None)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    treatments = tuple(args.treatment_arm or ("A1_stride5", "A2_stride10"))
    tags = tuple(args.tag or TAGS)
    ablations = tuple(args.history_ablation or ABLATIONS)
    expected_seeds = None
    if (args.expected_seed_start is None) != (args.expected_seed_count is None):
        raise ValueError("--expected-seed-start and --expected-seed-count must be used together")
    if args.expected_seed_count is not None:
        expected_seeds = set(
            range(args.expected_seed_start, args.expected_seed_start + args.expected_seed_count)
        )
    summary = summarize(
        args.eval_root,
        baseline_arm=args.baseline_arm,
        treatments=treatments,
        tags=tags,
        ablations=ablations,
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
    if args.require_pass and summary["selection_status"] != "treatment":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
