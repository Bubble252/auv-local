#!/usr/bin/env python3
"""Gate the Goal18 continuous-versus-sign/magnitude BC comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/lowlevel/pc_v0d_sign_magnitude_bc_20261107")
BASELINE_ROOT = Path("artifacts/lowlevel/pc_v0d_r2b_bc_20261106/eval")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _metrics(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    student = payload.get("student")
    if not isinstance(student, dict):
        raise ValueError(f"{path} has no student metrics")
    required = {
        "path_three_d_rmse_m",
        "success_rate",
        "timeout_rate",
        "safety_termination_rate",
    }
    missing = sorted(required - set(student))
    if missing:
        raise ValueError(f"{path} is missing metrics: {missing}")
    return student


def _ordered_rate(metrics: dict[str, Any]) -> float:
    return float(metrics["any_channel_reversal_rate_per_s"])


def _evaluate_gate(
    s1_default: dict[str, Any],
    s1_strict: dict[str, Any],
    s1_pathzero: dict[str, Any],
    audit: dict[str, Any],
    baseline_default: dict[str, Any],
    baseline_strict: dict[str, Any],
) -> dict[str, Any]:
    baseline_default_rmse = float(baseline_default["path_three_d_rmse_m"])
    rmse_checks = {
        "default_rmse_le_7p0": float(s1_default["path_three_d_rmse_m"]) <= 7.0,
        "default_rmse_le_0p80_baseline": (
            float(s1_default["path_three_d_rmse_m"]) <= 0.80 * baseline_default_rmse
        ),
        "strict125_rmse_le_10p5": float(s1_strict["path_three_d_rmse_m"]) <= 10.5,
    }
    guardrail_checks = {
        "default_success_ge_0p60": float(s1_default["success_rate"]) >= 0.60,
        "strict125_success_ge_0p10": float(s1_strict["success_rate"]) >= 0.10,
        "default_safety_le_baseline_plus_0p02": (
            float(s1_default["safety_termination_rate"])
            <= float(baseline_default["safety_termination_rate"]) + 0.02
        ),
        "strict125_safety_le_baseline_plus_0p02": (
            float(s1_strict["safety_termination_rate"])
            <= float(baseline_strict["safety_termination_rate"]) + 0.02
        ),
        "path_zero_rmse_worse_by_at_least_5pct": (
            float(s1_pathzero["path_three_d_rmse_m"])
            >= 1.05 * float(s1_default["path_three_d_rmse_m"])
        ),
    }
    sign_accuracy = audit["sign_accuracy_high_action"]
    saturation_ratio = audit["predicted_teacher_saturation_ratio"]
    representation_checks = {
        "rudder_sign_accuracy_ge_0p90": (
            sign_accuracy[0] is not None and float(sign_accuracy[0]) >= 0.90
        ),
        "stern_sign_accuracy_ge_0p90": (
            sign_accuracy[1] is not None and float(sign_accuracy[1]) >= 0.90
        ),
        "rudder_saturation_ratio_ge_0p65": (
            saturation_ratio[0] is not None and float(saturation_ratio[0]) >= 0.65
        ),
        "stern_saturation_ratio_ge_0p65": (
            saturation_ratio[1] is not None and float(saturation_ratio[1]) >= 0.65
        ),
    }
    predicted_reversal = audit["reversal"]["predicted"]
    teacher_reversal = audit["reversal"]["teacher"]
    rate_limit = audit.get("rate_limit") or {}
    stability_checks = {
        "rudder_reversal_le_teacher_plus_0p05": (
            predicted_reversal["channel_reversal_rate_per_s"][0]
            <= teacher_reversal["channel_reversal_rate_per_s"][0] + 0.05
        ),
        "stern_reversal_le_teacher_plus_0p05": (
            predicted_reversal["channel_reversal_rate_per_s"][1]
            <= teacher_reversal["channel_reversal_rate_per_s"][1] + 0.05
        ),
        "any_reversal_le_teacher_plus_0p05": (
            _ordered_rate(predicted_reversal) <= _ordered_rate(teacher_reversal) + 0.05
        ),
        "rate_limit_le_teacher_plus_0p05": (
            float(rate_limit.get("predicted_rate_limit_any_channel_fraction", 1.0))
            <= float(rate_limit.get("teacher_rate_limit_any_channel_fraction", 0.0)) + 0.05
        ),
    }
    if not all(rmse_checks.values()):
        status = "FAIL_RMSE"
    elif not all(guardrail_checks.values()):
        status = "FAIL_GUARDRAIL"
    elif not all(representation_checks.values()):
        status = "FAIL_REPRESENTATION"
    elif not all(stability_checks.values()):
        status = "FAIL_STABILITY"
    else:
        status = "PASS"
    return {
        "status": status,
        "rmse_checks": rmse_checks,
        "guardrail_checks": guardrail_checks,
        "representation_checks": representation_checks,
        "stability_checks": stability_checks,
    }


def _row(prefix: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {prefix} | {float(metrics['path_three_d_rmse_m']):.4f} | "
        f"{float(metrics['success_rate']):.4f} | "
        f"{float(metrics['timeout_rate']):.4f} | "
        f"{float(metrics['safety_termination_rate']):.4f} |"
    )


def main() -> int:
    args = _parse_args()
    modes = ("default", "strict125", "pathzero")
    results = {
        run: {
            mode: _metrics(args.root / "eval" / f"{run}_{mode}" / "summary.json")
            for mode in modes
        }
        for run in ("s0", "s1")
    }
    baseline_default = _metrics(args.baseline_root / "old_default" / "summary.json")
    baseline_strict = _metrics(args.baseline_root / "old_strict125" / "summary.json")
    same_code_default = _metrics(args.baseline_root / "new_default" / "summary.json")
    same_code_strict = _metrics(args.baseline_root / "new_strict125" / "summary.json")
    audit_s0 = _read_json(args.root / "audit" / "s0_action_saturation.json")
    audit_s1 = _read_json(args.root / "audit" / "s1_action_saturation.json")
    gate = _evaluate_gate(
        results["s1"]["default"],
        results["s1"]["strict125"],
        results["s1"]["pathzero"],
        audit_s1,
        baseline_default,
        baseline_strict,
    )
    s0_same_code_ratio = (
        float(results["s0"]["default"]["path_three_d_rmse_m"])
        / float(same_code_default["path_three_d_rmse_m"])
    )
    decision = {
        "schema_version": "auv_pc_v0d_sign_magnitude_bc_decision.v1",
        "baseline": {
            "old_bc_default": baseline_default,
            "old_bc_strict125": baseline_strict,
            "same_code_bc_default": same_code_default,
            "same_code_bc_strict125": same_code_strict,
        },
        "s0": {
            "metrics": results["s0"],
            "audit": audit_s0,
            "default_rmse_over_same_code_reference": s0_same_code_ratio,
            "reproduces_same_code_reference_within_10pct": (
                0.90 <= s0_same_code_ratio <= 1.10
            ),
        },
        "s1": {"metrics": results["s1"], "audit": audit_s1},
        "gate": gate,
        "decision": (
            "proceed_to_residual_ppo_screen"
            if gate["status"] == "PASS"
            else "stop_goal18_bc_gate_failed"
        ),
    }
    output = args.output or args.root / "stage_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Goal18 sign/magnitude BC decision",
        "",
        "| run | default RMSE | success | timeout | safety |",
        "|---|---:|---:|---:|---:|",
        _row("S0 default", results["s0"]["default"]),
        _row("S1 default", results["s1"]["default"]),
        _row("S0 strict125", results["s0"]["strict125"]),
        _row("S1 strict125", results["s1"]["strict125"]),
        _row("S0 path-zero", results["s0"]["pathzero"]),
        _row("S1 path-zero", results["s1"]["pathzero"]),
        "",
        f"S0/default over same-code BC artifact: `{s0_same_code_ratio:.4f}`",
        f"Status: `{gate['status']}`",
        f"Decision: `{decision['decision']}`",
        "",
        "## Checks",
        "",
    ]
    for group in (
        "rmse_checks",
        "guardrail_checks",
        "representation_checks",
        "stability_checks",
    ):
        lines.append(f"### {group}")
        for name, passed in gate[group].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
        lines.append("")
    (output.parent / "stage_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "decision": decision["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
