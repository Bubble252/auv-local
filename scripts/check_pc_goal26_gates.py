#!/usr/bin/env python3
"""Evaluate the pre-registered Goal26 success-first PPO screen gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "auv_pc_goal26_gates.v1"
EXPECTED_SEEDS = list(range(20, 60))
SCENARIOS = ("default", "strict125")
SUCCESS_FLOOR = {"default": 0.695, "strict125": 0.470}
SUCCESS_DROP_LIMIT = 0.03
SAFETY_DELTA_LIMIT = 0.02
TERMINATION_DELTA_LIMIT = 0.05
PRE_GOAL_IMPROVEMENT_M = 0.10
SUCCESS_IMPROVEMENT = 0.03
OTHER_SCENARIO_WORSEN_LIMIT = 0.03


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-comparison", type=Path, required=True)
    parser.add_argument("--default-normal", type=Path, required=True)
    parser.add_argument("--strict125-normal", type=Path, required=True)
    parser.add_argument("--default-path-zero", type=Path, required=True)
    parser.add_argument("--strict125-path-zero", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluation input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_cell(comparison: dict[str, Any], *, scenario: str) -> dict[str, Any]:
    if comparison.get("schema_version") != "auv_pc_eval_matrix_comparison.v1":
        raise ValueError("unexpected baseline comparison schema")
    for cell in comparison.get("cells", []):
        if cell.get("scenario") == scenario and cell.get("path_mode") == "normal":
            metrics = cell.get("student")
            if isinstance(metrics, dict):
                return metrics
    raise ValueError(f"missing baseline normal cell for {scenario}")


def _candidate_metrics(payload: dict[str, Any], *, name: str) -> dict[str, Any]:
    if payload.get("schema_version") != "auv_pc_ppo_evaluation.v1":
        raise ValueError(f"unexpected PPO evaluation schema for {name}")
    if payload.get("seeds") != EXPECTED_SEEDS:
        raise ValueError(
            f"{name} must evaluate exactly dev seeds {EXPECTED_SEEDS[0]}-{EXPECTED_SEEDS[-1]}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"missing metrics for {name}")
    return metrics


def _finite_float(
    metrics: dict[str, Any], key: str, *, name: str, optional: bool = False
) -> float | None:
    value = metrics.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name}.{key} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}.{key} must be finite, got {value!r}")
    return result


def _safety_rate(metrics: dict[str, Any], *, name: str) -> float:
    key = "safety_rate" if "safety_rate" in metrics else "safety_termination_rate"
    value = _finite_float(metrics, key, name=name)
    assert value is not None
    return value


def _full_rmse(metrics: dict[str, Any], *, name: str) -> float | None:
    key = "legacy_full_3d_rmse_m" if "legacy_full_3d_rmse_m" in metrics else "path_three_d_rmse_m"
    return _finite_float(metrics, key, name=name, optional=True)


def evaluate_goal26_screen(
    *,
    baseline: dict[str, Any],
    normal: dict[str, dict[str, Any]],
    path_zero: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for scenario in SCENARIOS:
        reference = _baseline_cell(baseline, scenario=scenario)
        current = normal[scenario]
        zero = path_zero[scenario]

        success = _finite_float(current, "success_rate", name=f"{scenario}/normal")
        pre_goal = _finite_float(current, "pre_goal_3d_rmse_m", name=f"{scenario}/normal")
        safety = _safety_rate(current, name=f"{scenario}/normal")
        plane_miss = _finite_float(current, "goal_plane_miss_rate", name=f"{scenario}/normal")
        timeout = _finite_float(current, "timeout_rate", name=f"{scenario}/normal")
        ref_success = _finite_float(reference, "success_rate", name=f"{scenario}/baseline")
        ref_pre_goal = _finite_float(reference, "pre_goal_3d_rmse_m", name=f"{scenario}/baseline")
        ref_safety = _safety_rate(reference, name=f"{scenario}/baseline")
        ref_plane_miss = _finite_float(
            reference, "goal_plane_miss_rate", name=f"{scenario}/baseline"
        )
        ref_timeout = _finite_float(reference, "timeout_rate", name=f"{scenario}/baseline")
        zero_success = _finite_float(zero, "success_rate", name=f"{scenario}/path-zero")
        zero_pre_goal = _finite_float(zero, "pre_goal_3d_rmse_m", name=f"{scenario}/path-zero")

        success_delta = success - ref_success
        pre_goal_delta = pre_goal - ref_pre_goal
        safety_delta = safety - ref_safety
        plane_miss_delta = plane_miss - ref_plane_miss
        timeout_delta = timeout - ref_timeout
        pre_goal_ratio = None if ref_pre_goal == 0.0 else pre_goal / ref_pre_goal - 1.0

        scenario_checks = {
            "success_ge_floor": success >= SUCCESS_FLOOR[scenario],
            "success_drop_le_limit": success_delta >= -SUCCESS_DROP_LIMIT,
            "safety_delta_le_limit": safety_delta <= SAFETY_DELTA_LIMIT,
            "plane_miss_delta_le_limit": plane_miss_delta <= TERMINATION_DELTA_LIMIT,
            "timeout_delta_le_limit": timeout_delta <= TERMINATION_DELTA_LIMIT,
        }
        checks.update({f"{scenario}_{key}": value for key, value in scenario_checks.items()})
        scenarios[scenario] = {
            "checks": scenario_checks,
            "guardrails_pass": all(scenario_checks.values()),
            "candidate": {
                "success_rate": success,
                "pre_goal_3d_rmse_m": pre_goal,
                "full_3d_rmse_m": _full_rmse(current, name=f"{scenario}/normal"),
                "safety_rate": safety,
                "goal_plane_miss_rate": plane_miss,
                "timeout_rate": timeout,
                "material_saturation_rate": _finite_float(
                    current,
                    "material_saturation_rate",
                    name=f"{scenario}/normal",
                    optional=True,
                ),
            },
            "baseline": {
                "success_rate": ref_success,
                "pre_goal_3d_rmse_m": ref_pre_goal,
                "safety_rate": ref_safety,
                "goal_plane_miss_rate": ref_plane_miss,
                "timeout_rate": ref_timeout,
            },
            "delta": {
                "success_rate": success_delta,
                "pre_goal_3d_rmse_m": pre_goal_delta,
                "pre_goal_ratio": pre_goal_ratio,
                "safety_rate": safety_delta,
                "goal_plane_miss_rate": plane_miss_delta,
                "timeout_rate": timeout_delta,
            },
            "effect_improved": (
                pre_goal_delta <= -PRE_GOAL_IMPROVEMENT_M or success_delta >= SUCCESS_IMPROVEMENT
            ),
            "path_zero": {
                "success_rate": zero_success,
                "pre_goal_3d_rmse_m": zero_pre_goal,
                "success_delta_vs_normal": zero_success - success,
                "pre_goal_delta_vs_normal_m": zero_pre_goal - pre_goal,
            },
        }

    default = scenarios["default"]
    strict = scenarios["strict125"]
    effect_checks = {
        "default_improved_with_strict_worsen_le_3pct": (
            default["effect_improved"]
            and strict["delta"]["pre_goal_ratio"] is not None
            and strict["delta"]["pre_goal_ratio"] <= OTHER_SCENARIO_WORSEN_LIMIT
        ),
        "strict_improved_with_default_worsen_le_3pct": (
            strict["effect_improved"]
            and default["delta"]["pre_goal_ratio"] is not None
            and default["delta"]["pre_goal_ratio"] <= OTHER_SCENARIO_WORSEN_LIMIT
        ),
    }
    checks["effect_gate"] = any(effect_checks.values())
    checks["all_guardrails"] = all(scenarios[scenario]["guardrails_pass"] for scenario in SCENARIOS)
    overall = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "effect_checks": effect_checks,
        "scenarios": scenarios,
        "overall": overall,
        "decision": "PASS" if overall else "FAIL",
        "path_zero_note": (
            "Path-zero is diagnostic-only here because the frozen D3 BC also has zero "
            "path-zero success. Report it, but do not use it to rewrite the pre-registered gate."
        ),
    }


def _fmt(value: float | None, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal26 PPO screen gates",
        "",
        f"- decision: **{result['decision']}**",
        "",
        "| scenario | success | delta | pre-goal RMSE | delta | ratio | safety delta | plane-miss delta | timeout delta | guardrails |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario in SCENARIOS:
        payload = result["scenarios"][scenario]
        candidate = payload["candidate"]
        delta = payload["delta"]
        lines.append(
            f"| {scenario} | {_fmt(candidate['success_rate'])} | "
            f"{delta['success_rate']:+.4f} | {_fmt(candidate['pre_goal_3d_rmse_m'], 6)} | "
            f"{delta['pre_goal_3d_rmse_m']:+.6f} | {delta['pre_goal_ratio']:+.2%} | "
            f"{delta['safety_rate']:+.4f} | {delta['goal_plane_miss_rate']:+.4f} | "
            f"{delta['timeout_rate']:+.4f} | "
            f"{'PASS' if payload['guardrails_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"- effect gate: {'PASS' if result['checks']['effect_gate'] else 'FAIL'}",
            f"- all guardrails: {'PASS' if result['checks']['all_guardrails'] else 'FAIL'}",
            "- path-zero note: " + result["path_zero_note"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    baseline = _load_json(args.baseline_comparison)
    normal = {
        "default": _candidate_metrics(_load_json(args.default_normal), name="default/normal"),
        "strict125": _candidate_metrics(_load_json(args.strict125_normal), name="strict125/normal"),
    }
    path_zero = {
        "default": _candidate_metrics(_load_json(args.default_path_zero), name="default/path-zero"),
        "strict125": _candidate_metrics(
            _load_json(args.strict125_path_zero), name="strict125/path-zero"
        ),
    }
    result = evaluate_goal26_screen(baseline=baseline, normal=normal, path_zero=path_zero)
    result["baseline_comparison"] = str(args.baseline_comparison.resolve())
    result["inputs"] = {
        "default_normal": str(args.default_normal.resolve()),
        "strict125_normal": str(args.strict125_normal.resolve()),
        "default_path_zero": str(args.default_path_zero.resolve()),
        "strict125_path_zero": str(args.strict125_path_zero.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(result, args.output_md)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
