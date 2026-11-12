#!/usr/bin/env python3
"""Summarize S16 V0/V1 closed-loop runs with the pre-registered V gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from summarize_pc_abc_temporal import TAGS, _aligned, _group, _mean, _std, collect  # noqa: E402

V0_SUCCESS_GAIN = 0.05
V0_RMSE_IMPROVEMENT = 0.10
V0_TIME_RATIO_MAX = 1.05
V0_TIMEOUT_INCREASE_MAX = 0.05
V_SPEED_ENVELOPE_INCREASE_MAX = 0.02
V_SAFETY_INCREASE_MAX = 0.02
V_SATURATION_REPORT_THRESHOLD = 0.02
V1_SUCCESS_DROP_MAX = 0.02
V1_RMSE_RATIO_MAX = 1.05
V1_TIME_RATIO_MAX = 1.05
V1_GUARDRAIL_INCREASE_MAX = 0.02


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _summary_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = _values(rows, key)
    if not values:
        return {"count": 0, "mean": None, "std": None}
    return {"count": len(values), "mean": _mean(values), "std": _std(values)}


def _paired_delta(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    key: str,
    *,
    boolean: bool = False,
) -> dict[str, Any]:
    base_values = [
        float(bool(row[key])) if boolean else float(row[key]) for row in baseline
    ]
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


def _time_ratio(
    baseline: list[dict[str, Any]], treatment: list[dict[str, Any]]
) -> dict[str, Any]:
    base_success = [row for row in baseline if bool(row.get("success"))]
    treatment_success = [row for row in treatment if bool(row.get("success"))]
    base_times = _values(base_success, "time_to_goal_s")
    treatment_times = _values(treatment_success, "time_to_goal_s")
    if not base_times or not treatment_times:
        return {
            "baseline_success_count": len(base_times),
            "treatment_success_count": len(treatment_times),
            "baseline_mean_s": None,
            "treatment_mean_s": None,
            "ratio": None,
        }
    base_mean = _mean(base_times)
    treatment_mean = _mean(treatment_times)
    return {
        "baseline_success_count": len(base_times),
        "treatment_success_count": len(treatment_times),
        "baseline_mean_s": base_mean,
        "treatment_mean_s": treatment_mean,
        "ratio": treatment_mean / max(base_mean, 1e-12),
    }


def compare(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    expected_seeds: set[int],
) -> dict[str, Any]:
    seeds, base_rows, treatment_rows = _aligned(baseline, treatment, expected_seeds)
    success = _paired_delta(base_rows, treatment_rows, "success", boolean=True)
    timeout = _paired_delta(base_rows, treatment_rows, "timeout", boolean=True)
    safety = _paired_delta(base_rows, treatment_rows, "safety", boolean=True)
    saturation = _paired_delta(
        base_rows, treatment_rows, "material_saturation_rate"
    )
    speed_below_min = _paired_delta(
        base_rows, treatment_rows, "ground_speed_below_min_rate"
    )
    speed_above_max = _paired_delta(
        base_rows, treatment_rows, "ground_speed_above_max_rate"
    )
    rmse = _paired_delta(base_rows, treatment_rows, "path_three_d_rmse_m")
    horizontal = _paired_delta(base_rows, treatment_rows, "path_horizontal_rmse_m")
    vertical = _paired_delta(base_rows, treatment_rows, "path_vertical_rmse_m")
    time = _time_ratio(base_rows, treatment_rows)
    secondary = {}
    for key in (
        "ground_speed_mean_mps",
        "ground_speed_std_mps",
        "ground_speed_min_mps",
        "ground_speed_max_mps",
        "ground_surge_std_mps",
        "ground_surge_delta_abs_p95_mps",
        "ground_speed_below_min_rate",
        "ground_speed_above_max_rate",
        "initial_speed_mps",
        "initial_propeller_rpm",
        "initial_speed_ratio",
        "initial_propeller_rpm_ratio",
        "rpm_sq_mean_rpm2",
        "rpm_sq_integral_proxy_rpm2_s",
        "action_delta_abs_mean",
        "action_delta_rpm_abs_mean_rpm",
        "turn_speed_mean_mps",
        "straight_speed_mean_mps",
        "turn_straight_speed_ratio",
        "path_overshoot_m",
    ):
        secondary[key] = {
            "baseline": _summary_stats(base_rows, key),
            "treatment": _summary_stats(treatment_rows, key),
        }
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "success": success,
        "timeout": timeout,
        "safety": safety,
        "material_saturation_rate": saturation,
        "ground_speed_below_min_rate": speed_below_min,
        "ground_speed_above_max_rate": speed_above_max,
        "path_three_d_rmse_m": rmse,
        "path_horizontal_rmse_m": horizontal,
        "path_vertical_rmse_m": vertical,
        "time_to_goal": time,
        "secondary": secondary,
    }


def _worst_case(comparisons: dict[str, Any], key: str) -> float:
    return max(
        float(comparisons[tag][key]["paired_delta_mean"]) for tag in ("default", "strict125")
    )


def _best_success_gain(comparisons: dict[str, Any]) -> float:
    return max(
        float(comparisons[tag]["success"]["paired_delta_mean"])
        for tag in ("default", "strict125")
    )


def _best_rmse_improvement(comparisons: dict[str, Any]) -> float:
    return max(
        float(comparisons[tag]["path_three_d_rmse_m"]["baseline_mean"])
        - float(comparisons[tag]["path_three_d_rmse_m"]["treatment_mean"])
        for tag in ("default", "strict125")
    )


def _worst_time_ratio(comparisons: dict[str, Any]) -> float | None:
    ratios = [
        comparisons[tag]["time_to_goal"]["ratio"]
        for tag in ("default", "strict125")
        if comparisons[tag]["time_to_goal"]["ratio"] is not None
    ]
    return max(float(value) for value in ratios) if ratios else None


def decide_v0(comparisons: dict[str, Any]) -> dict[str, Any]:
    success_gain = _best_success_gain(comparisons)
    rmse_improvement = _best_rmse_improvement(comparisons)
    time_ratio = _worst_time_ratio(comparisons)
    timeout_increase = _worst_case(comparisons, "timeout")
    saturation_increase = _worst_case(comparisons, "material_saturation_rate")
    safety_increase = _worst_case(comparisons, "safety")
    speed_below_min_increase = _worst_case(
        comparisons, "ground_speed_below_min_rate"
    )
    speed_above_max_increase = _worst_case(
        comparisons, "ground_speed_above_max_rate"
    )
    performance_gate = bool(
        success_gain >= V0_SUCCESS_GAIN or rmse_improvement >= V0_RMSE_IMPROVEMENT
    )
    time_gate = time_ratio is not None and time_ratio <= V0_TIME_RATIO_MAX
    operational_guardrail = bool(
        timeout_increase <= V0_TIMEOUT_INCREASE_MAX
        and safety_increase <= V_SAFETY_INCREASE_MAX
        and speed_below_min_increase <= V_SPEED_ENVELOPE_INCREASE_MAX
        and speed_above_max_increase <= V_SPEED_ENVELOPE_INCREASE_MAX
    )
    passed = bool(performance_gate and time_gate and operational_guardrail)
    return {
        "status": "pass_v0" if passed else "fail_v0",
        "passed": passed,
        "best_success_gain": success_gain,
        "best_rmse_improvement_m": rmse_improvement,
        "worst_time_ratio": time_ratio,
        "timeout_increase": timeout_increase,
        "material_saturation": {
            "increase": saturation_increase,
            "report_threshold": V_SATURATION_REPORT_THRESHOLD,
            "exceeds_threshold": saturation_increase
            > V_SATURATION_REPORT_THRESHOLD,
            "policy": "reported_as_actuator_usage_risk_not_a_pass_fail_gate",
        },
        "safety_increase": safety_increase,
        "ground_speed_below_min_increase": speed_below_min_increase,
        "ground_speed_above_max_increase": speed_above_max_increase,
        "gates": {
            "performance_gate": performance_gate,
            "time_gate": time_gate,
            "operational_guardrail_gate": operational_guardrail,
            "success_gain_min": V0_SUCCESS_GAIN,
            "rmse_improvement_min_m": V0_RMSE_IMPROVEMENT,
            "time_ratio_max": V0_TIME_RATIO_MAX,
            "timeout_increase_max": V0_TIMEOUT_INCREASE_MAX,
            "safety_increase_max": V_SAFETY_INCREASE_MAX,
            "speed_envelope_increase_max": V_SPEED_ENVELOPE_INCREASE_MAX,
        },
    }


def decide_v1(comparisons: dict[str, Any]) -> dict[str, Any]:
    success_drop = -min(
        float(comparisons[tag]["success"]["paired_delta_mean"])
        for tag in ("default", "strict125")
    )
    rmse_ratio = max(
        float(comparisons[tag]["path_three_d_rmse_m"]["treatment_mean"])
        / max(float(comparisons[tag]["path_three_d_rmse_m"]["baseline_mean"]), 1e-12)
        for tag in ("default", "strict125")
    )
    time_ratio = _worst_time_ratio(comparisons)
    timeout_increase = _worst_case(comparisons, "timeout")
    saturation_increase = _worst_case(comparisons, "material_saturation_rate")
    safety_increase = _worst_case(comparisons, "safety")
    speed_below_min_increase = _worst_case(
        comparisons, "ground_speed_below_min_rate"
    )
    speed_above_max_increase = _worst_case(
        comparisons, "ground_speed_above_max_rate"
    )
    passed = bool(
        success_drop <= V1_SUCCESS_DROP_MAX
        and rmse_ratio <= V1_RMSE_RATIO_MAX
        and time_ratio is not None
        and time_ratio <= V1_TIME_RATIO_MAX
        and timeout_increase <= V1_GUARDRAIL_INCREASE_MAX
        and safety_increase <= V1_GUARDRAIL_INCREASE_MAX
        and speed_below_min_increase <= V_SPEED_ENVELOPE_INCREASE_MAX
        and speed_above_max_increase <= V_SPEED_ENVELOPE_INCREASE_MAX
    )
    return {
        "status": "pass_v1" if passed else "fail_v1",
        "passed": passed,
        "success_drop": success_drop,
        "rmse_ratio": rmse_ratio,
        "worst_time_ratio": time_ratio,
        "timeout_increase": timeout_increase,
        "material_saturation": {
            "increase": saturation_increase,
            "report_threshold": V_SATURATION_REPORT_THRESHOLD,
            "exceeds_threshold": saturation_increase
            > V_SATURATION_REPORT_THRESHOLD,
            "policy": "reported_as_actuator_usage_risk_not_a_pass_fail_gate",
        },
        "safety_increase": safety_increase,
        "ground_speed_below_min_increase": speed_below_min_increase,
        "ground_speed_above_max_increase": speed_above_max_increase,
        "gates": {
            "success_drop_max": V1_SUCCESS_DROP_MAX,
            "rmse_ratio_max": V1_RMSE_RATIO_MAX,
            "time_ratio_max": V1_TIME_RATIO_MAX,
            "timeout_increase_max": V1_GUARDRAIL_INCREASE_MAX,
            "safety_increase_max": V1_GUARDRAIL_INCREASE_MAX,
            "speed_envelope_increase_max": V_SPEED_ENVELOPE_INCREASE_MAX,
        },
    }


def summarize(
    *,
    stage: str,
    mode: str = "screen",
    baseline_root: Path,
    baseline_arm: str,
    treatment_root: Path,
    treatment_arm: str,
    expected_seeds: set[int],
) -> dict[str, Any]:
    if mode not in {"smoke", "screen"}:
        raise ValueError("mode must be 'smoke' or 'screen'")
    known = {baseline_arm, treatment_arm}
    baseline_runs = collect(baseline_root, known_arms=known)
    treatment_runs = collect(treatment_root, known_arms=known)
    comparisons = {
        tag: compare(
            _group(baseline_runs, baseline_arm, tag),
            _group(treatment_runs, treatment_arm, tag),
            expected_seeds,
        )
        for tag in TAGS
    }
    scientific_decision = (
        decide_v0(comparisons) if stage == "V0" else decide_v1(comparisons)
    )
    if mode == "smoke":
        decision = {
            "status": "smoke_contract_only",
            "passed": True,
            "scientific_gate_preview": scientific_decision,
            "note": "A one-update smoke is not a scientific performance comparison.",
        }
    else:
        decision = scientific_decision
    return {
        "schema_version": "auv_pc_alos_free_v_summary.v1",
        "stage": stage,
        "mode": mode,
        "baseline_root": str(baseline_root),
        "baseline_arm": baseline_arm,
        "treatment_root": str(treatment_root),
        "treatment_arm": treatment_arm,
        "expected_seeds": sorted(expected_seeds),
        "comparisons": comparisons,
        "decision": decision,
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / max(denominator, 1e-12)


def render(summary: dict[str, Any]) -> str:
    lines = [
        f"# S16 {summary['stage']} paired summary",
        "",
        f"- mode: `{summary['mode']}`",
        f"- baseline: `{summary['baseline_arm']}`",
        f"- treatment: `{summary['treatment_arm']}`",
        f"- status: `{summary['decision']['status']}`",
        "",
        "| tag | success delta | RMSE ratio | time ratio | timeout delta | saturation delta | safety delta | below-u_min delta | above-u_max delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in TAGS:
        row = summary["comparisons"][tag]
        lines.append(
            "| {tag} | {success:+.4f} | {rmse:.4f} | {time} | {timeout:+.4f} | "
            "{sat:+.4f} | {safety:+.4f} | {below_min:+.4f} | {above_max:+.4f} |".format(
                tag=tag,
                success=row["success"]["paired_delta_mean"],
                rmse=_ratio(
                    row["path_three_d_rmse_m"]["treatment_mean"],
                    row["path_three_d_rmse_m"]["baseline_mean"],
                )
                or 0.0,
                time=(
                    "n/a"
                    if row["time_to_goal"]["ratio"] is None
                    else f"{row['time_to_goal']['ratio']:.4f}"
                ),
                timeout=row["timeout"]["paired_delta_mean"],
                sat=row["material_saturation_rate"]["paired_delta_mean"],
                safety=row["safety"]["paired_delta_mean"],
                below_min=row["ground_speed_below_min_rate"]["paired_delta_mean"],
                above_max=row["ground_speed_above_max_rate"]["paired_delta_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Secondary metrics",
            "",
            "| metric | default baseline | default treatment | strict baseline | strict treatment |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in (
        "ground_speed_mean_mps",
        "ground_speed_std_mps",
        "ground_speed_min_mps",
        "ground_speed_max_mps",
        "ground_speed_below_min_rate",
        "ground_speed_above_max_rate",
        "initial_speed_mps",
        "initial_propeller_rpm",
        "ground_surge_delta_abs_p95_mps",
        "rpm_sq_mean_rpm2",
        "turn_straight_speed_ratio",
        "path_overshoot_m",
    ):
        default = summary["comparisons"]["default"]["secondary"][metric]
        strict = summary["comparisons"]["strict125"]["secondary"][metric]
        lines.append(
            f"| {metric} | {default['baseline']['mean']} | {default['treatment']['mean']} | "
            f"{strict['baseline']['mean']} | {strict['treatment']['mean']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{summary['decision']['status']}`",
            "",
            "Material saturation is reported as an actuator-usage risk only; it is not "
            "used to decide V0/V1 pass or failure.",
            "",
            json.dumps(summary["decision"], indent=2, sort_keys=True),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("V0", "V1"), required=True)
    parser.add_argument("--mode", choices=("smoke", "screen"), default="screen")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-arm", required=True)
    parser.add_argument("--treatment-root", type=Path, required=True)
    parser.add_argument("--treatment-arm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seed-start", type=int, default=1000)
    parser.add_argument("--expected-seed-count", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_seed_start < 0 or args.expected_seed_count <= 0:
        raise ValueError("expected seed range must be non-negative and non-empty")
    expected_seeds = set(
        range(args.expected_seed_start, args.expected_seed_start + args.expected_seed_count)
    )
    summary = summarize(
        stage=args.stage,
        mode=args.mode,
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
    (args.output_dir / "summary.md").write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
