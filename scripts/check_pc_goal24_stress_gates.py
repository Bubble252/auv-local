#!/usr/bin/env python3
"""Evaluate Goal24 Phase 3 stress gates from saved evaluation matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "auv_pc_goal24_stress_gates.v1"
SCENARIOS = ("default", "strict125")
ARMS = ("s10", "s20")
NORMAL = "normal"
PATH_ZERO = "path_zero"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/dev/shm/pc_v1_stress_mix_20261108"),
    )
    parser.add_argument(
        "--nominal-reference",
        type=Path,
        default=Path("/dev/shm/pc_v1_data_recipe_20261108/eval_matrix_d2"),
    )
    parser.add_argument(
        "--stress-reference",
        type=Path,
        default=Path("/dev/shm/pc_v1_stress_mix_20261108/eval_stress_d2"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
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
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def evaluate_scenario(
    *,
    nominal: dict[str, Any],
    stress: dict[str, Any],
    nominal_reference: dict[str, Any],
    stress_reference: dict[str, Any],
    scenario: str,
) -> dict[str, Any]:
    """Evaluate one arm/scenario against the pre-registered Phase 3 gates."""

    nominal_metrics = _cell(nominal, scenario=scenario, path_mode=NORMAL)
    stress_metrics = _cell(stress, scenario=scenario, path_mode=NORMAL)
    nominal_ref = _cell(nominal_reference, scenario=scenario, path_mode=NORMAL)
    stress_ref = _cell(stress_reference, scenario=scenario, path_mode=NORMAL)
    nominal_zero = _cell(nominal, scenario=scenario, path_mode=PATH_ZERO)
    stress_zero = _cell(stress, scenario=scenario, path_mode=PATH_ZERO)

    nominal_rmse = _finite_float(nominal_metrics.get("pre_goal_3d_rmse_m"), name="nominal pre-goal RMSE")
    nominal_ref_rmse = _finite_float(
        nominal_ref.get("pre_goal_3d_rmse_m"), name="nominal reference pre-goal RMSE"
    )
    nominal_success = _finite_float(nominal_metrics.get("success_rate"), name="nominal success")
    nominal_ref_success = _finite_float(
        nominal_ref.get("success_rate"), name="nominal reference success"
    )
    nominal_safety = _finite_float(nominal_metrics.get("safety_rate"), name="nominal safety")
    nominal_ref_safety = _finite_float(
        nominal_ref.get("safety_rate"), name="nominal reference safety"
    )

    stress_rmse = _finite_float(stress_metrics.get("pre_goal_3d_rmse_m"), name="stress pre-goal RMSE")
    stress_ref_rmse = _finite_float(
        stress_ref.get("pre_goal_3d_rmse_m"), name="stress reference pre-goal RMSE"
    )
    stress_success = _finite_float(stress_metrics.get("success_rate"), name="stress success")
    stress_ref_success = _finite_float(
        stress_ref.get("success_rate"), name="stress reference success"
    )
    stress_safety = _finite_float(stress_metrics.get("safety_rate"), name="stress safety")
    stress_ref_safety = _finite_float(
        stress_ref.get("safety_rate"), name="stress reference safety"
    )

    nominal_rmse_ratio = nominal_rmse / nominal_ref_rmse - 1.0 if nominal_ref_rmse else None
    nominal_success_delta = nominal_success - nominal_ref_success
    nominal_safety_delta = nominal_safety - nominal_ref_safety
    stress_success_gain = stress_success - stress_ref_success
    stress_rmse_improvement = stress_ref_rmse - stress_rmse
    stress_safety_delta = stress_safety - stress_ref_safety

    nominal_rmse_check = nominal_rmse_ratio is not None and nominal_rmse_ratio <= 0.02
    nominal_success_check = nominal_success_delta >= -0.03
    nominal_safety_check = nominal_safety_delta <= 0.02
    stress_success_check = stress_success_gain >= 0.05
    stress_rmse_check = stress_rmse_improvement >= 0.10
    stress_gain_check = stress_success_check or stress_rmse_check
    stress_safety_check = stress_safety_delta <= 0.02

    checks = {
        "nominal_pre_goal_within_2pct": nominal_rmse_check,
        "nominal_success_drop_le_0.03": nominal_success_check,
        "nominal_safety_delta_le_0.02": nominal_safety_check,
        "stress_success_gain_ge_0.05": stress_success_check,
        "stress_pre_goal_improvement_ge_0.10": stress_rmse_check,
        "stress_gain_either": stress_gain_check,
        "stress_safety_delta_le_0.02": stress_safety_check,
    }
    required_checks = (
        nominal_rmse_check,
        nominal_success_check,
        nominal_safety_check,
        stress_gain_check,
        stress_safety_check,
    )
    return {
        "scenario": scenario,
        "metrics": {
            "nominal_success": nominal_success,
            "nominal_reference_success": nominal_ref_success,
            "nominal_success_delta": nominal_success_delta,
            "nominal_pre_goal_rmse_m": nominal_rmse,
            "nominal_reference_pre_goal_rmse_m": nominal_ref_rmse,
            "nominal_pre_goal_rmse_ratio": nominal_rmse_ratio,
            "nominal_safety_delta": nominal_safety_delta,
            "stress_success": stress_success,
            "stress_reference_success": stress_ref_success,
            "stress_success_gain": stress_success_gain,
            "stress_pre_goal_rmse_m": stress_rmse,
            "stress_reference_pre_goal_rmse_m": stress_ref_rmse,
            "stress_pre_goal_improvement_m": stress_rmse_improvement,
            "stress_safety_delta": stress_safety_delta,
            "nominal_path_zero_pre_goal_rmse_m": nominal_zero.get("pre_goal_3d_rmse_m"),
            "stress_path_zero_pre_goal_rmse_m": stress_zero.get("pre_goal_3d_rmse_m"),
        },
        "checks": checks,
        "overall": all(required_checks),
    }


def evaluate_arm(
    *,
    nominal: dict[str, Any],
    stress: dict[str, Any],
    nominal_reference: dict[str, Any],
    stress_reference: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    scenarios = {
        scenario: evaluate_scenario(
            nominal=nominal,
            stress=stress,
            nominal_reference=nominal_reference,
            stress_reference=stress_reference,
            scenario=scenario,
        )
        for scenario in SCENARIOS
    }
    return {
        "arm": arm,
        "scenarios": scenarios,
        "overall": all(item["overall"] for item in scenarios.values()),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal24 Phase 3 stress gate",
        "",
        "| arm | scenario | nominal success delta | nominal RMSE ratio | stress success gain | stress RMSE improvement | overall |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        item = payload["arms"][arm]
        for scenario in SCENARIOS:
            result = item["scenarios"][scenario]
            metrics = result["metrics"]
            lines.append(
                "| {arm} | {scenario} | {success_delta:+.4f} | {ratio:+.2%} | "
                "{stress_gain:+.4f} | {rmse_improvement:+.4f} m | {overall} |".format(
                    arm=arm,
                    scenario=scenario,
                    success_delta=metrics["nominal_success_delta"],
                    ratio=metrics["nominal_pre_goal_rmse_ratio"],
                    stress_gain=metrics["stress_success_gain"],
                    rmse_improvement=metrics["stress_pre_goal_improvement_m"],
                    overall="PASS" if result["overall"] else "FAIL",
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    nominal_reference = _load_comparison(args.nominal_reference / "comparison.json")
    stress_reference = _load_comparison(args.stress_reference / "comparison.json")
    arms = {}
    for arm in ARMS:
        nominal = _load_comparison(args.root / f"eval_nominal_{arm}" / "comparison.json")
        stress = _load_comparison(args.root / f"eval_stress_{arm}" / "comparison.json")
        arms[arm] = evaluate_arm(
            nominal=nominal,
            stress=stress,
            nominal_reference=nominal_reference,
            stress_reference=stress_reference,
            arm=arm,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "nominal_reference": str((args.nominal_reference / "comparison.json").resolve()),
        "stress_reference": str((args.stress_reference / "comparison.json").resolve()),
        "arms": arms,
        "overall": any(item["overall"] for item in arms.values()),
    }
    output_json = args.output_json or (args.root / "goal24_stress_gate.json")
    output_md = args.output_md or (args.root / "goal24_stress_gate.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["overall"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
