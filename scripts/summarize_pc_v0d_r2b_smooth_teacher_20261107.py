#!/usr/bin/env python3
"""Summarize and gate the V-0D R2B smooth-teacher screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107")
BASELINE_ROOT = Path("artifacts/lowlevel/pc_v0d_teacher_r2b_revalidation_20261106")
TAGS = ("h15", "h20", "rate", "gov")
MODES = ("default", "strict125")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 1:
        raise ValueError(f"{path} must contain exactly one variant")
    metrics = variants[0].get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} has no metrics")
    return metrics


def _expected_keys(metric: dict[str, Any]) -> None:
    required = {
        "path_three_d_rmse_m",
        "success_rate",
        "timeout_rate",
        "safety_termination_rate",
        "action_channel_saturation_rate",
    }
    missing = sorted(required - set(metric))
    if missing:
        raise ValueError(f"smooth-teacher summary is missing metrics: {missing}")


def _variant_checks(
    default: dict[str, Any],
    strict: dict[str, Any],
    baseline_default: dict[str, Any],
    baseline_strict: dict[str, Any],
) -> dict[str, bool]:
    default_sat = default["action_channel_saturation_rate"]
    return {
        "default_rmse_le_6p0": float(default["path_three_d_rmse_m"]) <= 6.0,
        "default_success_ge_0p70": float(default["success_rate"]) >= 0.70,
        "default_rudder_sat_le_0p40": float(default_sat[0]) <= 0.40,
        "default_stern_sat_le_0p20": float(default_sat[1]) <= 0.20,
        "strict125_rmse_le_10p5": float(strict["path_three_d_rmse_m"]) <= 10.5,
        "strict125_success_ge_0p10": float(strict["success_rate"]) >= 0.10,
        "default_timeout_le_baseline_plus_0p10": (
            float(default["timeout_rate"]) <= float(baseline_default["timeout_rate"]) + 0.10
        ),
        "strict125_timeout_le_baseline_plus_0p10": (
            float(strict["timeout_rate"]) <= float(baseline_strict["timeout_rate"]) + 0.10
        ),
        "default_safety_le_baseline_plus_0p02": (
            float(default["safety_termination_rate"])
            <= float(baseline_default["safety_termination_rate"]) + 0.02
        ),
        "strict125_safety_le_baseline_plus_0p02": (
            float(strict["safety_termination_rate"])
            <= float(baseline_strict["safety_termination_rate"]) + 0.02
        ),
    }


def main() -> int:
    args = _parse_args()
    baseline_default = _read_metrics(
        args.baseline_root / "R2B_default" / "teacher_summary.json"
    )
    baseline_strict = _read_metrics(
        args.baseline_root / "R2B_strict125" / "teacher_summary.json"
    )
    variants: dict[str, Any] = {}
    for tag in TAGS:
        metrics_by_mode: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            path = args.root / "teacher_smoothing" / f"{tag}_{mode}" / "teacher_summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing smooth-teacher summary: {path}")
            metrics = _read_metrics(path)
            _expected_keys(metrics)
            metrics_by_mode[mode] = metrics
        checks = _variant_checks(
            metrics_by_mode["default"],
            metrics_by_mode["strict125"],
            baseline_default,
            baseline_strict,
        )
        default_sat = metrics_by_mode["default"]["action_channel_saturation_rate"]
        variants[tag] = {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "default": metrics_by_mode["default"],
            "strict125": metrics_by_mode["strict125"],
            "selection_key": [
                float(metrics_by_mode["default"]["path_three_d_rmse_m"]),
                float(default_sat[0]) + float(default_sat[1]),
            ],
        }
    passing = [tag for tag, result in variants.items() if result["status"] == "pass"]
    selected = min(passing, key=lambda tag: variants[tag]["selection_key"]) if passing else None
    decision = {
        "schema_version": "auv_pc_v0d_r2b_smooth_teacher_decision.v1",
        "baseline": {
            "default": baseline_default,
            "strict125": baseline_strict,
        },
        "variants": variants,
        "passing_variants": passing,
        "selected_variant": selected,
        "decision": "proceed_to_bc" if selected else "skip_to_stage2",
    }
    output = args.output or args.root / "teacher_smoothing" / "stage1_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Smooth teacher stage-1 decision",
        "",
        "| variant | status | default RMSE | success | rudder sat | stern sat | strict RMSE | strict success |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, result in variants.items():
        default = result["default"]
        strict = result["strict125"]
        saturation = default["action_channel_saturation_rate"]
        lines.append(
            f"| {tag} | {result['status']} | {default['path_three_d_rmse_m']:.4f} | "
            f"{default['success_rate']:.4f} | {saturation[0]:.4f} | {saturation[1]:.4f} | "
            f"{strict['path_three_d_rmse_m']:.4f} | {strict['success_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Selected variant: `{selected}`",
            f"Decision: `{decision['decision']}`",
        ]
    )
    (output.parent / "stage1_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_variant": selected, "decision": decision["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
