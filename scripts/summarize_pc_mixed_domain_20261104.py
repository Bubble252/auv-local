#!/usr/bin/env python3
"""Evaluate decoupled speed-current generalization against the multienv baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

BASELINE_ARM = "frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0"
TREATMENT_ARM = "frozen_bc_residual_nodade_mixed_domain_lr3e-4_scale1p0"
IN_DOMAIN_TAGS = ("default", "h2", "h3")
HELDOUT_TAGS = ("h0", "h1", "c4")
PRESSURE_TAGS = ("h4",)
REQUIRED_METRICS = (
    "success_rate",
    "path_three_d_rmse_m",
    "material_saturation_rate",
    "safety_termination_rate",
)
TRACKING_METRICS = (
    "mean_abs_speed_error_mps",
    "mean_abs_yaw_error_rad",
    "mean_abs_pitch_error_rad",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--h0-bc-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-arm", default=BASELINE_ARM)
    parser.add_argument("--treatment-arm", default=TREATMENT_ARM)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_means(aggregate: dict[str, Any] | None) -> dict[str, float]:
    if aggregate is None:
        return {}
    result: dict[str, float] = {}
    for key, described in (aggregate.get("raw") or {}).items():
        if not isinstance(described, dict) or described.get("mean") is None:
            continue
        value = float(described["mean"])
        if math.isfinite(value):
            result[key] = value
    return result


def _canonical_arm(arm: str) -> str:
    prefix = "supervised_"
    return arm[len(prefix) :] if arm.startswith(prefix) else arm


def _index_aggregates(
    summary: dict[str, Any], *, checkpoint: str
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for aggregate in summary.get("aggregates", []):
        if aggregate.get("checkpoint") != checkpoint:
            continue
        if aggregate.get("history_ablation", "none") != "none":
            continue
        key = (_canonical_arm(str(aggregate.get("arm"))), str(aggregate.get("tag")))
        if key in result:
            raise ValueError(f"duplicate aggregate for arm={key[0]} tag={key[1]}")
        result[key] = aggregate
    return result


def _delta(treatment: dict[str, float], reference: dict[str, float]) -> dict[str, float | None]:
    return {
        "success_delta": treatment["success_rate"] - reference["success_rate"],
        "path_three_d_rmse_delta_m": (
            treatment["path_three_d_rmse_m"] - reference["path_three_d_rmse_m"]
        ),
        "material_saturation_delta": (
            treatment["material_saturation_rate"] - reference["material_saturation_rate"]
        ),
        "path_three_d_rmse_ratio": (
            treatment["path_three_d_rmse_m"]
            / max(reference["path_three_d_rmse_m"], 1e-12)
        ),
    }


def _tracking_ratio(treatment: dict[str, float], reference: dict[str, float]) -> float | None:
    keys = [key for key in TRACKING_METRICS if key in treatment and key in reference]
    if not keys:
        return None
    return max(treatment[key] / max(reference[key], 1e-12) for key in keys)


def _retention_gate(
    treatment: dict[str, float],
    baseline: dict[str, float],
    *,
    success_drop_limit: float,
    rmse_ratio_limit: float,
    saturation_delta_limit: float,
) -> dict[str, Any]:
    delta = _delta(treatment, baseline)
    checks = {
        "success_drop": -delta["success_delta"] <= success_drop_limit + 1e-12,
        "path_rmse_ratio": delta["path_three_d_rmse_ratio"] <= rmse_ratio_limit + 1e-12,
        "material_saturation": (
            delta["material_saturation_delta"] <= saturation_delta_limit + 1e-12
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        **delta,
    }


def _h0_gate(treatment: dict[str, float], h0_bc: dict[str, float]) -> dict[str, Any]:
    delta = _delta(treatment, h0_bc)
    tracking_ratio = _tracking_ratio(treatment, h0_bc)
    checks = {
        "success_drop_le_0p05": -delta["success_delta"] <= 0.05 + 1e-12,
        "path_rmse_le_1p25x_h0_bc": delta["path_three_d_rmse_ratio"] <= 1.25 + 1e-12,
        "tracking_errors_le_1p2x_h0_bc": (
            tracking_ratio is not None and tracking_ratio <= 1.2 + 1e-12
        ),
        "material_saturation_not_worse_by_0p05": (
            delta["material_saturation_delta"] <= 0.05 + 1e-12
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tracking_error_ratio": tracking_ratio,
        **delta,
    }


def build_report(
    *,
    summary_path: Path,
    h0_bc_summary_path: Path,
    baseline_arm: str,
    treatment_arm: str,
    seed_count: int,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    h0_summary = _load_json(h0_bc_summary_path)
    h0_bc = {key: float(h0_summary["student"][key]) for key in REQUIRED_METRICS}
    for key in TRACKING_METRICS:
        if key in h0_summary["student"]:
            h0_bc[key] = float(h0_summary["student"][key])
    indexed = _index_aggregates(summary, checkpoint="final")

    stages: dict[str, Any] = {}
    for tag in (*IN_DOMAIN_TAGS, "strict125", *HELDOUT_TAGS, *PRESSURE_TAGS):
        aggregate = indexed.get((_canonical_arm(treatment_arm), tag))
        if aggregate is None:
            stages[tag] = {"status": "missing", "tag": tag}
            continue
        metrics = _metric_means(aggregate)
        missing = [key for key in REQUIRED_METRICS if key not in metrics]
        baseline_aggregate = indexed.get((_canonical_arm(baseline_arm), tag))
        baseline = _metric_means(baseline_aggregate)
        entry: dict[str, Any] = {
            "status": "complete" if not missing else "incomplete_metrics",
            "tag": tag,
            "train_seeds": aggregate.get("train_seeds", []),
            "metrics": metrics,
            "missing_metrics": missing,
            "baseline_tag": tag,
            "baseline_available": baseline_aggregate is not None,
        }
        if baseline_aggregate is not None:
            entry["baseline"] = baseline
            entry["vs_multienv"] = _retention_gate(
                metrics,
                baseline,
                success_drop_limit=(0.02 if tag == "strict125" else 0.05),
                rmse_ratio_limit=(1.10 if tag != "strict125" else 1.05),
                saturation_delta_limit=0.05,
            )
        else:
            entry["baseline"] = None
            entry["vs_multienv"] = None

        if tag in IN_DOMAIN_TAGS or tag == "strict125":
            entry["role"] = "in_domain"
        elif tag in HELDOUT_TAGS:
            entry["role"] = "heldout"
        else:
            entry["role"] = "pressure_only"

        if tag == "h0":
            entry["gate"] = _h0_gate(metrics, h0_bc)
        elif tag == "h1":
            h0_gate = _h0_gate(metrics, h0_bc)
            retention_gate = entry["vs_multienv"]
            entry["gate"] = {
                "status": (
                    "pass"
                    if h0_gate["status"] == "pass"
                    and retention_gate is not None
                    and all(retention_gate["checks"].values())
                    else "fail"
                ),
                "h0_gate": h0_gate,
                "multienv_retention": retention_gate,
            }
        elif tag == "strict125":
            entry["gate"] = {
                "status": "pass"
                if entry["vs_multienv"] is not None
                and all(entry["vs_multienv"]["checks"].values())
                else "fail",
                "basis": "strict125_retention_vs_multienv",
            }
        elif tag in IN_DOMAIN_TAGS:
            entry["gate"] = entry["vs_multienv"]
        else:
            entry["gate"] = {"status": "pressure_only"}
        stages[tag] = entry

    gated_tags = (*IN_DOMAIN_TAGS, "strict125", "h0", "h1")
    complete = all(stages[tag].get("status") == "complete" for tag in gated_tags)
    passed = all((stages[tag].get("gate") or {}).get("status") == "pass" for tag in gated_tags)
    overall_status = "pass" if complete and passed else ("incomplete" if not complete else "fail")
    return {
        "schema_version": "auv_pc_mixed_domain_report.v1",
        "summary_path": str(summary_path),
        "h0_bc_summary_path": str(h0_bc_summary_path),
        "baseline_arm": baseline_arm,
        "treatment_arm": treatment_arm,
        "seed_count": seed_count,
        "train_seed_count": max(
            (len(stages[tag].get("train_seeds", [])) for tag in stages),
            default=0,
        ),
        "stages": stages,
        "overall_status": overall_status,
        "in_domain_tags": list(IN_DOMAIN_TAGS),
        "heldout_tags": list(HELDOUT_TAGS),
        "pressure_only_tags": list(PRESSURE_TAGS),
        "notes": [
            "C4 is excluded from mixed-domain training and remains pressure-only evidence.",
            "H0/H1 strong-current cells are held out; H1 is the primary zero-shot composition test.",
            "This is a 5-training-seed screen; formal 0-49/50-99 evaluation is still required after a pass.",
        ],
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# PC mixed-domain 速度-海流解耦报告", ""]
    lines.append(
        f"`{report['treatment_arm']}` vs `{report['baseline_arm']}`；"
        f"训练 seed 数 `{report['train_seed_count']}`，每格评估 seed 数 `{report['seed_count']}`。"
    )
    lines.extend(["", "## 分域结果", ""])
    lines.append(
        "| Tag | Role | Success | 3D RMSE | Saturation | vs multienv success | "
        "RMSE ratio | Gate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for tag in (*IN_DOMAIN_TAGS, "strict125", *HELDOUT_TAGS, *PRESSURE_TAGS):
        entry = report["stages"][tag]
        if entry.get("status") != "complete":
            lines.append(f"| {tag} | {entry.get('role', '-')} | missing | missing | missing | n/a | n/a | `{entry['status']}` |")
            continue
        metrics = entry["metrics"]
        comparison = entry.get("vs_multienv") or {}
        lines.append(
            f"| {tag} | {entry['role']} | {_fmt(metrics['success_rate'])} | "
            f"{_fmt(metrics['path_three_d_rmse_m'], 2)} m | "
            f"{_fmt(metrics['material_saturation_rate'])} | "
            f"{_fmt(comparison.get('success_delta'), 4)} | "
            f"{_fmt(comparison.get('path_three_d_rmse_ratio'), 2)}× | "
            f"`{entry['gate']['status']}` |"
        )
    lines.extend(["", "## 判定", ""])
    lines.append(f"- overall: `{report['overall_status']}`")
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    report = build_report(
        summary_path=args.summary,
        h0_bc_summary_path=args.h0_bc_summary,
        baseline_arm=args.baseline_arm,
        treatment_arm=args.treatment_arm,
        seed_count=args.seed_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not args.require_pass or report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
