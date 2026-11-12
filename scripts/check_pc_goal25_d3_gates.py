#!/usr/bin/env python3
"""Evaluate the pre-registered Goal25 D3 BC gates from saved comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "auv_pc_goal25_d3_gates.v1"
SCENARIOS = ("default", "strict125")
NORMAL = "normal"
PATH_ZERO = "path_zero"
PRE_GOAL_LIMIT_M = {"default": 2.65, "strict125": 2.70}
SUCCESS_FLOOR = {"default": 0.70, "strict125": 0.40}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3", type=Path, required=True)
    parser.add_argument("--d2", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _load_comparison(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluation comparison: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "auv_pc_eval_matrix_comparison.v1":
        raise ValueError(f"unexpected comparison schema: {path}")
    return payload


def _cell(comparison: dict[str, Any], *, scenario: str, path_mode: str) -> dict[str, Any]:
    for cell in comparison.get("cells", []):
        if cell.get("scenario") == scenario and cell.get("path_mode") == path_mode:
            metrics = cell.get("student")
            if not isinstance(metrics, dict):
                raise ValueError(f"missing student metrics for {scenario}/{path_mode}")
            return metrics
    raise ValueError(f"missing comparison cell for {scenario}/{path_mode}")


def _finite_float(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def evaluate_d3_gates(
    *,
    d3: dict[str, Any],
    d2: dict[str, Any],
    manifest_present: bool,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        current = _cell(d3, scenario=scenario, path_mode=NORMAL)
        reference = _cell(d2, scenario=scenario, path_mode=NORMAL)
        current_zero = _cell(d3, scenario=scenario, path_mode=PATH_ZERO)

        current_rmse = _finite_float(
            current.get("pre_goal_3d_rmse_m"),
            name=f"{scenario} D3 pre-goal RMSE",
        )
        reference_rmse = _finite_float(
            reference.get("pre_goal_3d_rmse_m"),
            name=f"{scenario} D2 pre-goal RMSE",
        )
        current_success = _finite_float(
            current.get("success_rate"), name=f"{scenario} D3 success"
        )
        reference_success = _finite_float(
            reference.get("success_rate"), name=f"{scenario} D2 success"
        )
        current_safety = _finite_float(
            current.get("safety_rate"), name=f"{scenario} D3 safety"
        )
        reference_safety = _finite_float(
            reference.get("safety_rate"), name=f"{scenario} D2 safety"
        )
        zero_rmse = _finite_float(
            current_zero.get("pre_goal_3d_rmse_m"),
            name=f"{scenario} D3 path-zero RMSE",
        )

        success_delta = current_success - reference_success
        safety_delta = current_safety - reference_safety
        path_zero_degradation = (
            None if current_rmse == 0.0 else zero_rmse / current_rmse - 1.0
        )
        scenario_checks = {
            "pre_goal_le_limit": current_rmse <= PRE_GOAL_LIMIT_M[scenario],
            "success_ge_floor": current_success >= SUCCESS_FLOOR[scenario],
            "success_drop_le_0.03": success_delta >= -0.03,
            "safety_delta_le_0.02": safety_delta <= 0.02,
            "path_zero_degradation_ge_5pct": (
                path_zero_degradation is not None and path_zero_degradation >= 0.05
            ),
        }
        checks.update(
            {f"{scenario}_{name}": value for name, value in scenario_checks.items()}
        )
        scenarios[scenario] = {
            "checks": scenario_checks,
            "overall": all(scenario_checks.values()),
            "d3": {
                "success_rate": current_success,
                "pre_goal_3d_rmse_m": current_rmse,
                "safety_rate": current_safety,
            },
            "d2_reference": {
                "success_rate": reference_success,
                "pre_goal_3d_rmse_m": reference_rmse,
                "safety_rate": reference_safety,
            },
            "success_delta": success_delta,
            "pre_goal_rmse_delta_m": current_rmse - reference_rmse,
            "pre_goal_rmse_ratio": (
                None if reference_rmse == 0.0 else current_rmse / reference_rmse - 1.0
            ),
            "safety_delta": safety_delta,
            "path_zero_pre_goal_rmse_m": zero_rmse,
            "path_zero_degradation": path_zero_degradation,
        }

    checks["finite_and_manifest"] = manifest_present
    return {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "scenarios": scenarios,
        "overall": all(checks.values()),
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    args = _parse_args()
    d3 = _load_comparison(args.d3)
    d2 = _load_comparison(args.d2)
    result = evaluate_d3_gates(
        d3=d3,
        d2=d2,
        manifest_present=args.manifest.is_file(),
    )
    result["d3_comparison"] = str(args.d3.resolve())
    result["d2_reference"] = str(args.d2.resolve())
    result["manifest"] = str(args.manifest.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_md is not None:
        lines = [
            "# Goal25 D3 BC gates",
            "",
            f"- decision: **{result['decision']}**",
            "",
            "| scenario | success | success delta | pre-goal RMSE | RMSE ratio | "
            "path-zero degradation | scenario gate |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for scenario in SCENARIOS:
            payload = result["scenarios"][scenario]
            lines.append(
                f"| {scenario} | {payload['d3']['success_rate']:.4f} | "
                f"{payload['success_delta']:+.4f} | "
                f"{payload['d3']['pre_goal_3d_rmse_m']:.6f} | "
                f"{payload['pre_goal_rmse_ratio']:+.2%} | "
                f"{payload['path_zero_degradation']:+.2%} | "
                f"{'PASS' if payload['overall'] else 'FAIL'} |"
            )
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
