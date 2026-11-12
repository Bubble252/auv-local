#!/usr/bin/env python3
"""Gate training-time action-smoothness arms against the grid residual baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

BASELINE_ARM = "frozen_bc_residual_nodade_grid_lr3e-4_scale1p0"
TREATMENT_ARMS = (
    "frozen_bc_residual_nodade_grid_smooth1e-3_lr3e-4_scale1p0",
    "frozen_bc_residual_nodade_grid_smooth1e-3_l2_1e-2_lr3e-4_scale1p0",
)
HELDOUT_TAG = "grid_hs"
IN_DOMAIN_TAGS = ("grid_lw", "grid_ls", "grid_hw")
RETENTION_TAGS = ("default", "strict125")
PRESSURE_TAGS = ("c4", "h4")
GATED_TAGS = (*IN_DOMAIN_TAGS, HELDOUT_TAG, *RETENTION_TAGS)
METRICS = (
    "success_rate",
    "path_three_d_rmse_m",
    "material_saturation_rate",
    "safety_termination_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-arm", default=BASELINE_ARM)
    parser.add_argument("--treatment-arm", action="append", default=list(TREATMENT_ARMS))
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _canonical_arm(arm: str) -> str:
    return arm.removeprefix("supervised_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_means(aggregate: dict[str, Any] | None) -> dict[str, float]:
    if aggregate is None:
        return {}
    result: dict[str, float] = {}
    for key, value in (aggregate.get("raw") or {}).items():
        if not isinstance(value, dict) or value.get("mean") is None:
            continue
        number = float(value["mean"])
        if math.isfinite(number):
            result[key] = number
    return result


def _index(
    summary: dict[str, Any], checkpoint: str
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for aggregate in summary.get("aggregates", []):
        if aggregate.get("checkpoint") != checkpoint:
            continue
        if aggregate.get("history_ablation", "none") != "none":
            continue
        key = (_canonical_arm(str(aggregate.get("arm"))), str(aggregate.get("tag")))
        if key in result:
            raise ValueError(f"duplicate aggregate for {key[0]} / {key[1]}")
        result[key] = aggregate
    return result


def _delta(treatment: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        "success_delta": treatment["success_rate"] - reference["success_rate"],
        "path_three_d_rmse_delta_m": (
            treatment["path_three_d_rmse_m"] - reference["path_three_d_rmse_m"]
        ),
        "path_three_d_rmse_ratio": (
            treatment["path_three_d_rmse_m"]
            / max(reference["path_three_d_rmse_m"], 1e-12)
        ),
        "material_saturation_delta": (
            treatment["material_saturation_rate"]
            - reference["material_saturation_rate"]
        ),
        "safety_termination_delta": (
            treatment["safety_termination_rate"]
            - reference["safety_termination_rate"]
        ),
    }


def _gate(
    treatment: dict[str, float], baseline: dict[str, float], *, heldout: bool
) -> dict[str, Any]:
    delta = _delta(treatment, baseline)
    checks = {
        "success_drop_le_0p02": -delta["success_delta"] <= 0.02 + 1e-12,
        "rmse_ratio_le_1p05": delta["path_three_d_rmse_ratio"] <= 1.05 + 1e-12,
        "saturation_not_worse": delta["material_saturation_delta"] <= 0.01 + 1e-12,
        "safety_not_worse": delta["safety_termination_delta"] <= 1e-12,
    }
    if heldout:
        checks["saturation_reduction_ge_0p02"] = (
            delta["material_saturation_delta"] <= -0.02 + 1e-12
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        **delta,
    }


def build_report(
    *,
    summary_path: Path,
    baseline_arm: str,
    treatment_arms: tuple[str, ...],
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    finals = _index(summary, "final")
    initials = _index(summary, "initial")
    baseline_arm = _canonical_arm(baseline_arm)
    treatments = tuple(_canonical_arm(arm) for arm in treatment_arms)
    stages: dict[str, Any] = {}
    for tag in (*IN_DOMAIN_TAGS, HELDOUT_TAG, *RETENTION_TAGS, *PRESSURE_TAGS):
        baseline_final_aggregate = finals.get((baseline_arm, tag))
        baseline_initial_aggregate = initials.get((baseline_arm, tag))
        baseline_final = _metric_means(baseline_final_aggregate)
        baseline_initial = _metric_means(baseline_initial_aggregate)
        entry: dict[str, Any] = {
            "tag": tag,
            "role": (
                "heldout"
                if tag == HELDOUT_TAG
                else "in_domain"
                if tag in IN_DOMAIN_TAGS
                else "retention"
                if tag in RETENTION_TAGS
                else "pressure_only"
            ),
            "baseline_final": baseline_final or None,
            "baseline_initial": baseline_initial or None,
            "treatments": {},
        }
        for arm in treatments:
            final_aggregate = finals.get((arm, tag))
            final = _metric_means(final_aggregate)
            if not final or not baseline_final:
                entry["treatments"][arm] = {"status": "missing"}
                continue
            missing = [name for name in METRICS if name not in final or name not in baseline_final]
            payload: dict[str, Any] = {
                "status": "complete" if not missing else "incomplete_metrics",
                "train_seeds": final_aggregate.get("train_seeds", []),
                "final": final,
                "delta_vs_baseline_final": _delta(final, baseline_final),
                "missing_metrics": missing,
            }
            if baseline_initial:
                payload["delta_vs_baseline_initial"] = _delta(final, baseline_initial)
            if tag in PRESSURE_TAGS:
                payload["gate"] = {"status": "pressure_only"}
            elif not missing:
                payload["gate"] = _gate(final, baseline_final, heldout=tag == HELDOUT_TAG)
            else:
                payload["gate"] = {"status": "incomplete"}
            entry["treatments"][arm] = payload
        stages[tag] = entry

    arms: dict[str, Any] = {}
    for arm in treatments:
        failed_tags: list[str] = []
        incomplete_tags: list[str] = []
        for tag in GATED_TAGS:
            item = stages[tag]["treatments"].get(arm) or {}
            gate = item.get("gate") or {}
            if gate.get("status") == "fail":
                failed_tags.append(tag)
            elif gate.get("status") != "pass":
                incomplete_tags.append(tag)
        arms[arm] = {
            "status": "fail" if failed_tags else "incomplete" if incomplete_tags else "pass",
            "failed_tags": failed_tags,
            "incomplete_tags": incomplete_tags,
        }
    if any(item["status"] == "pass" for item in arms.values()):
        overall = "pass"
    elif any(item["status"] == "incomplete" for item in arms.values()):
        overall = "incomplete"
    else:
        overall = "fail"
    return {
        "schema_version": "auv_pc_grid_smoothness_report.v1",
        "summary_path": str(summary_path),
        "baseline_arm": baseline_arm,
        "treatment_arms": list(treatments),
        "heldout_tag": HELDOUT_TAG,
        "in_domain_tags": list(IN_DOMAIN_TAGS),
        "retention_tags": list(RETENTION_TAGS),
        "pressure_only_tags": list(PRESSURE_TAGS),
        "arms": arms,
        "overall": overall,
        "stages": stages,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PC 残差时序平滑实验",
        "",
        f"- baseline: `{report['baseline_arm']}`",
        f"- overall: `{report['overall']}`",
        "- held-out gate 要求 `grid_hs` 成功率下降 ≤0.02、RMSE ≤1.05×、"
        "material saturation 至少下降 0.02。",
        "",
        "| Tag | Treatment | Success | RMSE | Saturation | Δsuccess | RMSE ratio | Δsaturation | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for tag, stage in report["stages"].items():
        for arm, item in stage["treatments"].items():
            if item.get("status") == "missing":
                lines.append(f"| {tag} | `{arm}` | missing | missing | missing | n/a | n/a | n/a | missing |")
                continue
            final = item["final"]
            delta = item["delta_vs_baseline_final"]
            gate = item.get("gate", {}).get("status", "missing")
            lines.append(
                f"| {tag} | `{arm}` | {final['success_rate']:.4f} | "
                f"{final['path_three_d_rmse_m']:.2f} m | "
                f"{final['material_saturation_rate']:.4f} | "
                f"{delta['success_delta']:+.4f} | "
                f"{delta['path_three_d_rmse_ratio']:.2f}× | "
                f"{delta['material_saturation_delta']:+.4f} | {gate} |"
            )
    lines.extend(["", "## 判定", ""])
    for arm, item in report["arms"].items():
        lines.append(
            f"- `{arm}`: `{item['status']}`; failed={item['failed_tags']}; "
            f"incomplete={item['incomplete_tags']}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    report = build_report(
        summary_path=args.summary,
        baseline_arm=args.baseline_arm,
        treatment_arms=tuple(args.treatment_arm),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(args.output_dir / "report.json")
    return 0 if report["overall"] == "pass" or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
