#!/usr/bin/env python3
"""Gate DADE against a matched no-DADE arm on the speed-current grid."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

TREATMENT_ARM = "frozen_bc_residual_dade_grid_gpu_pose_lr3e-4_scale1p0"
REFERENCE_ARM = "frozen_bc_residual_nodade_grid_gpu_lr3e-4_scale1p0"
TAGS = (
    "grid_lw",
    "grid_ls",
    "grid_hw",
    "grid_hs",
    "default",
    "strict125",
    "h1",
    "h2",
    "h3",
    "h4",
    "c4",
)
RETENTION_TAGS = ("grid_lw", "grid_ls", "grid_hw", "default", "strict125")
PRESSURE_TAGS = ("h4", "c4")
METRICS = (
    "success_rate",
    "timeout_rate",
    "path_three_d_rmse_m",
    "material_saturation_rate",
    "mean_abs_speed_error_mps",
    "mean_abs_yaw_error_rad",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--treatment-arm", default=TREATMENT_ARM)
    parser.add_argument("--reference-arm", default=REFERENCE_ARM)
    parser.add_argument("--train-seed-count", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _canonical_arm(arm: str) -> str:
    return str(arm).removeprefix("supervised_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {key}: {value}")
    return value


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _index_rows(summary: dict[str, Any]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    result: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in summary.get("rows", []):
        if row.get("history_ablation", "none") != "none":
            continue
        key = (
            _canonical_arm(str(row["arm"])),
            str(row["tag"]),
            int(row["train_seed"]),
            str(row["checkpoint"]),
        )
        if key in result:
            raise ValueError(f"duplicate evaluation row for {key}")
        result[key] = row
    return result


def _paired(
    index: dict[tuple[str, str, int, str], dict[str, Any]],
    *,
    treatment_arm: str,
    reference_arm: str,
    tag: str,
    treatment_checkpoint: str = "final",
    reference_checkpoint: str = "final",
) -> dict[str, Any]:
    seeds = sorted(
        {
            seed
            for arm, row_tag, seed, checkpoint in index
            if row_tag == tag
            and (
                (arm == treatment_arm and checkpoint == treatment_checkpoint)
                or (arm == reference_arm and checkpoint == reference_checkpoint)
            )
        }
    )
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in seeds:
        treatment = index.get((treatment_arm, tag, seed, treatment_checkpoint))
        reference = index.get((reference_arm, tag, seed, reference_checkpoint))
        if treatment is None or reference is None:
            missing.append(seed)
            continue
        treatment_values = {key: _metric(treatment, key) for key in METRICS}
        reference_values = {key: _metric(reference, key) for key in METRICS}
        rows.append(
            {
                "train_seed": seed,
                "eval_seeds": int(treatment["eval_seeds"]),
                "treatment": treatment_values,
                "reference": reference_values,
                "delta_success": treatment_values["success_rate"]
                - reference_values["success_rate"],
                "delta_timeout": treatment_values["timeout_rate"]
                - reference_values["timeout_rate"],
                "delta_saturation": treatment_values["material_saturation_rate"]
                - reference_values["material_saturation_rate"],
                "rmse_ratio": treatment_values["path_three_d_rmse_m"]
                / max(reference_values["path_three_d_rmse_m"], 1e-12),
                "speed_tracking_ratio": treatment_values["mean_abs_speed_error_mps"]
                / max(reference_values["mean_abs_speed_error_mps"], 1e-12),
                "yaw_tracking_ratio": treatment_values["mean_abs_yaw_error_rad"]
                / max(reference_values["mean_abs_yaw_error_rad"], 1e-12),
            }
        )
    keys = (
        "delta_success",
        "delta_timeout",
        "delta_saturation",
        "rmse_ratio",
        "speed_tracking_ratio",
        "yaw_tracking_ratio",
    )
    return {
        "seeds": seeds,
        "paired_seeds": [row["train_seed"] for row in rows],
        "missing_seeds": missing,
        "eval_seed_counts": sorted({int(row["eval_seeds"]) for row in rows}),
        "deltas": {key: _describe([float(row[key]) for row in rows]) for key in keys},
        "treatment_means": {
            key: _describe([float(row["treatment"][key]) for row in rows])
            for key in METRICS
        },
        "reference_means": {
            key: _describe([float(row["reference"][key]) for row in rows])
            for key in METRICS
        },
    }


def _gate(checks: dict[str, bool]) -> dict[str, Any]:
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def _retention_gate(
    comparison: dict[str, Any], *, success_drop: float, rmse_ratio: float
) -> dict[str, Any]:
    means = comparison["deltas"]
    if means["delta_success"]["mean"] is None:
        return {"status": "incomplete", "checks": {}}
    checks = {
        "success_drop": -float(means["delta_success"]["mean"]) <= success_drop + 1e-12,
        "rmse_ratio": float(means["rmse_ratio"]["mean"]) <= rmse_ratio + 1e-12,
        "saturation_delta": float(means["delta_saturation"]["mean"]) <= 0.05 + 1e-12,
    }
    return _gate(checks)


def build_report(
    *,
    summary_path: Path,
    treatment_arm: str,
    reference_arm: str,
    train_seed_count: int,
    seed_count: int,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    index = _index_rows(summary)
    treatment_arm = _canonical_arm(treatment_arm)
    reference_arm = _canonical_arm(reference_arm)
    stages: dict[str, Any] = {}
    for tag in TAGS:
        vs_reference = _paired(
            index,
            treatment_arm=treatment_arm,
            reference_arm=reference_arm,
            tag=tag,
        )
        vs_initial = _paired(
            index,
            treatment_arm=treatment_arm,
            reference_arm=treatment_arm,
            tag=tag,
            treatment_checkpoint="final",
            reference_checkpoint="initial",
        )
        n = len(vs_reference["paired_seeds"])
        complete = (
            n == train_seed_count
            and not vs_reference["missing_seeds"]
            and vs_reference["eval_seed_counts"] == [seed_count]
        )
        stage: dict[str, Any] = {
            "tag": tag,
            "status": "complete" if complete else "incomplete",
            "paired_seed_count": n,
            "vs_reference": vs_reference,
            "vs_initial": vs_initial,
        }
        if not complete:
            stage["gate"] = {"status": "incomplete", "checks": {}}
            stages[tag] = stage
            continue
        treatment_means = vs_reference["treatment_means"]
        treatment = {
            key: float(treatment_means[key]["mean"])
            for key in METRICS
        }
        if tag in ("grid_lw", "grid_ls", "grid_hw"):
            stage["role"] = "in_domain"
            stage["gate"] = _retention_gate(
                vs_reference, success_drop=0.02, rmse_ratio=1.10
            )
        elif tag == "grid_hs":
            stage["role"] = "heldout"
            delta = vs_reference["deltas"]
            transfer = (
                (float(delta["delta_success"]["mean"]) >= 0.02 - 1e-12)
                or (float(delta["rmse_ratio"]["mean"]) <= 0.95 + 1e-12)
            )
            tracking_ok = all(
                float(delta[key]["mean"]) <= 1.20 + 1e-12
                for key in ("speed_tracking_ratio", "yaw_tracking_ratio")
            )
            stage["gate"] = _gate(
                {
                    "transfer_success_or_rmse": transfer,
                    "saturation_delta": float(delta["delta_saturation"]["mean"])
                    <= 0.02 + 1e-12,
                    "tracking_ratio": tracking_ok,
                    "absolute_success": treatment["success_rate"] >= 0.90 - 1e-12,
                    "absolute_rmse": treatment["path_three_d_rmse_m"] <= 5.0 + 1e-12,
                    "absolute_saturation": treatment["material_saturation_rate"]
                    <= 0.45 + 1e-12,
                }
            )
        elif tag == "default":
            stage["role"] = "retention"
            retention = _retention_gate(
                vs_reference, success_drop=0.05, rmse_ratio=1.10
            )
            absolute = {
                "success": treatment["success_rate"] >= 0.70 - 1e-12,
                "rmse": treatment["path_three_d_rmse_m"] <= 5.0 + 1e-12,
                "timeout_not_worse": float(
                    vs_reference["deltas"]["delta_timeout"]["mean"]
                )
                <= 1e-12,
            }
            stage["gate"] = _gate({**retention["checks"], **absolute})
        elif tag == "strict125":
            stage["role"] = "retention"
            retention = _retention_gate(
                vs_reference, success_drop=0.02, rmse_ratio=1.05
            )
            absolute = {
                "success": treatment["success_rate"] >= 0.50 - 1e-12,
                "rmse": treatment["path_three_d_rmse_m"] <= 6.5 + 1e-12,
                "timeout": treatment["timeout_rate"] <= 0.531 + 1e-12,
            }
            stage["gate"] = _gate({**retention["checks"], **absolute})
        elif tag == "h1":
            stage["role"] = "robustness"
            stage["gate"] = _gate(
                {
                    "success": treatment["success_rate"] >= 0.95 - 1e-12,
                    "rmse": treatment["path_three_d_rmse_m"] <= 5.0 + 1e-12,
                    "saturation": treatment["material_saturation_rate"]
                    <= 0.40 + 1e-12,
                }
            )
        elif tag == "h3":
            stage["role"] = "robustness"
            stage["gate"] = _gate(
                {
                    "success": treatment["success_rate"] >= 0.95 - 1e-12,
                    "rmse": treatment["path_three_d_rmse_m"] <= 3.4 + 1e-12,
                    "saturation": treatment["material_saturation_rate"]
                    <= 0.45 + 1e-12,
                }
            )
        elif tag == "h2":
            stage["role"] = "diagnostic"
            stage["gate"] = {"status": "diagnostic", "checks": {}}
        else:
            stage["role"] = "pressure_only"
            stage["gate"] = {"status": "pressure_only", "checks": {}}
        stages[tag] = stage

    primary = (
        "grid_lw",
        "grid_ls",
        "grid_hw",
        "grid_hs",
        "default",
        "strict125",
        "h1",
        "h3",
    )
    if any(stages[tag]["status"] != "complete" for tag in primary):
        overall = "incomplete"
    elif all(stages[tag]["gate"]["status"] == "pass" for tag in primary):
        overall = "pass"
    else:
        overall = "fail"
    return {
        "schema_version": "auv_pc_dade_grid_report.v1",
        "summary_path": str(summary_path),
        "treatment_arm": treatment_arm,
        "reference_arm": reference_arm,
        "train_seed_count": train_seed_count,
        "seed_count": seed_count,
        "primary_tags": list(primary),
        "pressure_only_tags": list(PRESSURE_TAGS),
        "stages": stages,
        "overall_status": overall,
        "notes": [
            "All comparisons pair the same training seed and the same closed-loop evaluation seeds.",
            "grid_hs is held out from both residual arms' training data.",
            "H4 and C4 are pressure-only and do not enter the primary DADE gate.",
            "A failed grid_hs or retention gate leaves the overall status at fail.",
        ],
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PC DADE × 速度/海流网格配对报告",
        "",
        f"Treatment: `{report['treatment_arm']}`",
        f"Reference: `{report['reference_arm']}`",
        f"训练 seed 数: `{report['train_seed_count']}`；每 tag 评估 seed 数: `{report['seed_count']}`",
        "",
        "| Tag | Role | Treatment success | Reference success | Treatment RMSE | Reference RMSE | Treatment timeout | Treatment sat | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for tag in TAGS:
        stage = report["stages"][tag]
        if stage["status"] != "complete":
            lines.append(
                f"| {tag} | missing | n/a | n/a | n/a | n/a | n/a | n/a | `incomplete` |"
            )
            continue
        treatment = stage["vs_reference"]["treatment_means"]
        reference = stage["vs_reference"]["reference_means"]
        lines.append(
            "| {tag} | {role} | {ts} | {rs} | {tr} m | {rr} m | {tt} | {sat} | `{gate}` |".format(
                tag=tag,
                role=stage.get("role", "n/a"),
                ts=_fmt(treatment["success_rate"]["mean"]),
                rs=_fmt(reference["success_rate"]["mean"]),
                tr=_fmt(treatment["path_three_d_rmse_m"]["mean"], 2),
                rr=_fmt(reference["path_three_d_rmse_m"]["mean"], 2),
                tt=_fmt(treatment["timeout_rate"]["mean"]),
                sat=_fmt(treatment["material_saturation_rate"]["mean"]),
                gate=stage["gate"]["status"],
            )
        )
    lines.extend(["", "## 判定", "", f"- overall: `{report['overall_status']}`"])
    for note in report["notes"]:
        lines.append(f"- {note}")
    heldout = report["stages"]["grid_hs"]
    if heldout["status"] == "complete":
        delta = heldout["vs_reference"]["deltas"]
        lines.append(
            "- grid_hs paired delta: success `{}`, RMSE ratio `{}`, saturation `{}`".format(
                _fmt(delta["delta_success"]["mean"]),
                _fmt(delta["rmse_ratio"]["mean"]),
                _fmt(delta["delta_saturation"]["mean"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    report = build_report(
        summary_path=args.summary,
        treatment_arm=args.treatment_arm,
        reference_arm=args.reference_arm,
        train_seed_count=args.train_seed_count,
        seed_count=args.seed_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(markdown_path)
    if args.require_pass and report["overall_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
