#!/usr/bin/env python3
"""Summarize S15 P2 raw-vs-encoded history DADE with seed-paired gates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TAGS = ("default", "strict125")
HISTORY_ABLATIONS = ("none", "history-zero")
PRIMARY_SUCCESS_DELTA = 0.05
PRIMARY_RMSE_IMPROVEMENT_M = 0.10
GUARDRAIL_SAFETY_DELTA = 0.02
GUARDRAIL_SATURATION_DELTA = 0.02
GUARDRAIL_TIMEOUT_DELTA = 0.05
HISTORY_DETECT_SUCCESS_DELTA = 0.05
HISTORY_DETECT_RMSE_DELTA_M = 0.10
LATENCY_GATE_MS = 20.0


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
        raise ValueError(f"{path} lacks per-seed episodes")
    result: dict[int, dict[str, Any]] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row:
            raise ValueError(f"malformed episode row in {path}")
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"duplicate episode seed {seed} in {path}")
        result[seed] = row
    return result


def _infer_arm(relative: Path, data: dict[str, Any]) -> str:
    for part in relative.parts:
        if part.startswith(("A0_", "A1_", "A2_", "P1_", "P2")):
            return part
    arm = data.get("arm")
    return str(arm) if arm else "unknown"


def collect_runs(eval_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Collect evaluation summaries keyed by (arm, tag, history ablation)."""

    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(eval_root.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "auv_pc_ppo_evaluation.v1":
            continue
        path_ablation = data.get("path_ablation", "none")
        if path_ablation not in (None, "None", "none"):
            continue
        history_ablation = data.get("history_ablation", "none")
        if history_ablation is None:
            history_ablation = "none"
        if history_ablation not in HISTORY_ABLATIONS:
            continue
        relative = path.relative_to(eval_root)
        tag = next((value for value in TAGS if value in relative.parts), None)
        if tag is None:
            environment = str(data.get("environment_config", "")).lower()
            if "strict125" in environment or "strict_125" in environment:
                tag = "strict125"
            elif "rewardscale001" in environment or "c3_random_realistic" in environment:
                tag = "default"
        if tag is None:
            raise ValueError(f"cannot infer evaluation tag from {path}")
        arm = _infer_arm(relative, data)
        key = (arm, tag, history_ablation)
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


def _history_effect(
    normal: dict[str, Any],
    ablated: dict[str, Any],
    *,
    expected_seeds: set[int] | None,
) -> dict[str, Any]:
    comparison = compare_groups(normal, ablated, expected_seeds=expected_seeds)
    success_drop = -comparison["success"]["paired_delta_mean"]
    rmse_worsening = comparison["path_three_d_rmse_m"]["paired_delta_mean"]
    action_l2 = [
        float(row.get("history_action_delta_l2_mean", 0.0))
        for row in ablated["episodes"].values()
    ]
    action_rpm = [
        float(row.get("history_action_delta_rpm_abs_mean_rpm", 0.0))
        for row in ablated["episodes"].values()
    ]
    detected = (
        success_drop >= HISTORY_DETECT_SUCCESS_DELTA
        or rmse_worsening >= HISTORY_DETECT_RMSE_DELTA_M
    )
    return {
        "success_drop": success_drop,
        "rmse_worsening_m": rmse_worsening,
        "timeout_change": comparison["timeout"]["paired_delta_mean"],
        "material_saturation_change": comparison["material_saturation_rate"][
            "paired_delta_mean"
        ],
        "history_action_delta_l2_mean": _mean(action_l2),
        "history_action_delta_l2_max": float(max(action_l2)),
        "history_action_delta_rpm_abs_mean_rpm": _mean(action_rpm),
        "detected": bool(detected),
        "comparison": comparison,
    }


def load_latency(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    latency = data.get("latency_ms")
    if not isinstance(latency, dict) or not _finite(latency.get("p95_ms")):
        raise ValueError(f"{path} lacks a finite latency_ms.p95_ms")
    p95 = float(latency["p95_ms"])
    gate = bool(data.get("gate_passed", p95 < LATENCY_GATE_MS))
    return {
        "path": str(path),
        "p50_ms": float(latency["p50_ms"]),
        "p95_ms": p95,
        "gate_ms": float(data.get("gate_ms_p95", LATENCY_GATE_MS)),
        "gate_passed": gate,
    }


def _decision(
    *,
    arm: str,
    normal_by_tag: dict[str, dict[str, Any]],
    history_effects_by_tag: dict[str, dict[str, dict[str, Any]]],
    latency: dict[str, Any] | None,
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
    history_zero = history_effects_by_tag["default"]["history-zero"]
    history_attribution = bool(history_zero["detected"])
    latency_gate_passed = None if latency is None else bool(latency["gate_passed"])

    if not primary_signal:
        status = "no_signal"
    elif not guardrails_pass:
        status = "fail_guardrail"
    elif latency_gate_passed is not True:
        status = "fail_latency"
    elif not history_attribution:
        status = "no_history_attribution"
    else:
        status = "pass"

    return {
        "arm": arm,
        "status": status,
        "primary_signal": bool(primary_signal),
        "guardrails_pass": bool(guardrails_pass),
        "latency_gate_passed": latency_gate_passed,
        "history_attribution": bool(history_attribution),
        "default_success_delta": success_delta,
        "strict125_success_delta": strict_delta,
        "rmse_improvement_m": rmse_improvement,
        "max_safety_delta": max_safety,
        "max_material_saturation_delta": max_saturation,
        "max_timeout_delta": max_timeout,
        "history_zero_success_drop_default": history_zero["success_drop"],
        "history_zero_rmse_worsening_default_m": history_zero["rmse_worsening_m"],
        "history_zero_action_delta_l2_mean": history_zero[
            "history_action_delta_l2_mean"
        ],
        "selection_key": (success_delta, strict_delta, rmse_improvement),
    }


def summarize(
    baseline_eval_root: Path,
    treatment_eval_root: Path,
    *,
    baseline_arm: str,
    treatment_arm: str,
    tags: tuple[str, ...] = TAGS,
    history_ablations: tuple[str, ...] = HISTORY_ABLATIONS,
    expected_seeds: set[int] | None = None,
    latency_summary: Path | None = None,
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
        for ablation in history_ablations:
            if ablation == "none":
                continue
            effects_by_tag[tag][ablation] = _history_effect(
                _require_group(treatment_runs, treatment_arm, tag, "none"),
                _require_group(treatment_runs, treatment_arm, tag, ablation),
                expected_seeds=expected_seeds,
            )

    latency = load_latency(latency_summary)
    decision = _decision(
        arm=treatment_arm,
        normal_by_tag=normal_by_tag,
        history_effects_by_tag=effects_by_tag,
        latency=latency,
    )
    return {
        "schema_version": "auv_pc_abc_p2_summary.v1",
        "baseline_eval_root": str(baseline_eval_root),
        "treatment_eval_root": str(treatment_eval_root),
        "baseline_arm": baseline_arm,
        "treatment_arm": treatment_arm,
        "tags": list(tags),
        "history_ablations": list(history_ablations),
        "expected_seeds": None if expected_seeds is None else sorted(expected_seeds),
        "latency": latency,
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
            "history_ablations": effects_by_tag,
        },
        "decision": decision,
        "gate_definition": {
            "primary_success_delta": PRIMARY_SUCCESS_DELTA,
            "primary_rmse_improvement_m": PRIMARY_RMSE_IMPROVEMENT_M,
            "max_safety_delta": GUARDRAIL_SAFETY_DELTA,
            "max_material_saturation_delta": GUARDRAIL_SATURATION_DELTA,
            "max_timeout_delta": GUARDRAIL_TIMEOUT_DELTA,
            "history_detection_success_delta": HISTORY_DETECT_SUCCESS_DELTA,
            "history_detection_rmse_delta_m": HISTORY_DETECT_RMSE_DELTA_M,
            "latency_gate_ms": LATENCY_GATE_MS,
        },
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# S15 P2 encoded-history DADE screening",
        "",
        f"- baseline: `{summary['baseline_arm']}`",
        f"- treatment: `{summary['treatment_arm']}`",
        f"- status: `{summary['decision']['status']}`",
        "",
        "## Paired comparisons against baseline",
        "",
        "| tag | success delta | 3D RMSE improvement (m) | timeout delta | safety delta | saturation delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag, comparison in summary["comparisons"]["normal"].items():
        lines.append(
            "| {tag} | {success} | {rmse} | {timeout} | {safety} | {sat} |".format(
                tag=tag,
                success=_fmt(comparison["success"]["paired_delta_mean"]),
                rmse=_fmt(-comparison["path_three_d_rmse_m"]["paired_delta_mean"]),
                timeout=_fmt(comparison["timeout"]["paired_delta_mean"]),
                safety=_fmt(comparison["safety"]["paired_delta_mean"]),
                sat=_fmt(comparison["material_saturation_rate"]["paired_delta_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## History-zero ablations",
            "",
            "| tag | ablation | success drop | RMSE worsening (m) | timeout change | action L2 mean | RPM abs mean | detected |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for tag, effects in summary["comparisons"]["history_ablations"].items():
        for ablation, effect in effects.items():
            lines.append(
                "| {tag} | {ablation} | {success} | {rmse} | {timeout} | {action} | {rpm} | {detected} |".format(
                    tag=tag,
                    ablation=ablation,
                    success=_fmt(effect["success_drop"]),
                    rmse=_fmt(effect["rmse_worsening_m"]),
                    timeout=_fmt(effect["timeout_change"]),
                    action=_fmt(effect["history_action_delta_l2_mean"]),
                    rpm=_fmt(effect["history_action_delta_rpm_abs_mean_rpm"]),
                    detected=_fmt(effect["detected"]),
                )
            )
    latency = summary.get("latency")
    lines.extend(
        [
            "",
            "## Latency",
            "",
            (
                "not measured"
                if latency is None
                else "p50 `{p50}` ms, p95 `{p95}` ms, gate `{gate}`".format(
                    p50=_fmt(latency["p50_ms"]),
                    p95=_fmt(latency["p95_ms"]),
                    gate=_fmt(latency["gate_passed"]),
                )
            ),
            "",
            "## Decision",
            "",
            f"- primary signal: `{_fmt(summary['decision']['primary_signal'])}`",
            f"- guardrails: `{_fmt(summary['decision']['guardrails_pass'])}`",
            f"- latency gate: `{_fmt(summary['decision']['latency_gate_passed'])}`",
            f"- history attribution: `{_fmt(summary['decision']['history_attribution'])}`",
            f"- default success delta: `{_fmt(summary['decision']['default_success_delta'])}`",
            f"- strict125 success delta: `{_fmt(summary['decision']['strict125_success_delta'])}`",
            f"- 3D RMSE improvement: `{_fmt(summary['decision']['rmse_improvement_m'])} m`",
            f"- history-zero success drop (default): `{_fmt(summary['decision']['history_zero_success_drop_default'])}`",
            f"- history-zero RMSE worsening (default): `{_fmt(summary['decision']['history_zero_rmse_worsening_default_m'])} m`",
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
    parser.add_argument("--baseline-arm", default="A0_stride1")
    parser.add_argument("--treatment-arm", default="P2b_encoded")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", action="append", default=None)
    parser.add_argument("--history-ablation", action="append", default=None)
    parser.add_argument("--latency-summary", type=Path, default=None)
    parser.add_argument("--expected-seed-start", type=int, default=None)
    parser.add_argument("--expected-seed-count", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tags = tuple(args.tag or TAGS)
    history_ablations = tuple(args.history_ablation or HISTORY_ABLATIONS)
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
        history_ablations=history_ablations,
        expected_seeds=expected_seeds,
        latency_summary=args.latency_summary,
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
