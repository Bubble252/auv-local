#!/usr/bin/env python3
"""Gate RPM-only residual authority against the matched full-action baseline."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TREATMENT_ARM = "frozen_bc_residual_nodade_grid_rpm_only_lr3e-4_scale1p0"
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
IN_DOMAIN_TAGS = ("grid_lw", "grid_ls", "grid_hw")
RETENTION_TAGS = ("default", "strict125")
ROBUSTNESS_TAGS = ("h1", "h3")
PRESSURE_TAGS = ("h4", "c4")
METRICS = (
    "success_rate",
    "timeout_rate",
    "path_three_d_rmse_m",
    "material_saturation_rate",
    "mean_abs_speed_error_mps",
    "mean_abs_yaw_error_rad",
    "ground_route_speed_ratio_min",
    "ground_route_speed_ratio_episode_min_mean",
    "ground_route_speed_ratio_mean",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--treatment-arm", default=TREATMENT_ARM)
    parser.add_argument("--reference-arm", default=REFERENCE_ARM)
    parser.add_argument("--train-seed-count", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--tags", nargs="+", choices=TAGS, default=list(TAGS))
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _canonical_arm(arm: str) -> str:
    return str(arm).removeprefix("supervised_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


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
) -> dict[str, Any]:
    seeds = sorted(
        {
            seed
            for arm, row_tag, seed, checkpoint in index
            if checkpoint == "final"
            and row_tag == tag
            and arm in {treatment_arm, reference_arm}
        }
    )
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    incomplete_metrics: list[str] = []
    for seed in seeds:
        treatment = index.get((treatment_arm, tag, seed, "final"))
        reference = index.get((reference_arm, tag, seed, "final"))
        if treatment is None or reference is None:
            missing.append(seed)
            continue
        treatment_values = {name: _metric(treatment, name) for name in METRICS}
        reference_values = {name: _metric(reference, name) for name in METRICS}
        missing_metrics = [
            name
            for name, value in (*treatment_values.items(), *reference_values.items())
            if value is None
        ]
        if missing_metrics:
            incomplete_metrics.extend(missing_metrics)
            continue
        assert all(value is not None for value in treatment_values.values())
        assert all(value is not None for value in reference_values.values())
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
                "ground_route_speed_ratio_delta": treatment_values[
                    "ground_route_speed_ratio_min"
                ]
                - reference_values["ground_route_speed_ratio_min"],
                "ground_route_episode_min_delta": treatment_values[
                    "ground_route_speed_ratio_episode_min_mean"
                ]
                - reference_values["ground_route_speed_ratio_episode_min_mean"],
            }
        )
    keys = (
        "delta_success",
        "delta_timeout",
        "delta_saturation",
        "rmse_ratio",
        "ground_route_speed_ratio_delta",
        "ground_route_episode_min_delta",
    )
    means = {
        "treatment": {
            name: _describe([float(row["treatment"][name]) for row in rows])
            for name in METRICS
        },
        "reference": {
            name: _describe([float(row["reference"][name]) for row in rows])
            for name in METRICS
        },
        "deltas": {
            name: _describe([float(row[name]) for row in rows]) for name in keys
        },
    }
    return {
        "status": "complete" if rows else "missing",
        "paired_seeds": [row["train_seed"] for row in rows],
        "missing_train_seeds": missing,
        "incomplete_metrics": sorted(set(incomplete_metrics)),
        "rows": rows,
        "means": means,
    }


def _gate(
    *,
    tag: str,
    treatment: dict[str, float],
    reference: dict[str, float],
) -> dict[str, Any]:
    if tag in PRESSURE_TAGS:
        return {"status": "pressure_only", "checks": {}}

    checks: dict[str, bool] = {}
    if tag in IN_DOMAIN_TAGS:
        checks["success_drop_le_0.02"] = (
            treatment["success_rate"] >= reference["success_rate"] - 0.02 - 1e-12
        )
        checks["rmse_ratio_le_1.05"] = (
            treatment["path_three_d_rmse_m"]
            <= 1.05 * reference["path_three_d_rmse_m"] + 1e-12
        )
        checks["saturation_delta_le_0.05"] = (
            treatment["material_saturation_rate"]
            <= reference["material_saturation_rate"] + 0.05 + 1e-12
        )
    elif tag == "grid_hs":
        checks["success_ge_0.85"] = treatment["success_rate"] >= 0.85 - 1e-12
        checks["rmse_le_6.0"] = treatment["path_three_d_rmse_m"] <= 6.0 + 1e-12
        checks["success_not_below_reference"] = (
            treatment["success_rate"] >= reference["success_rate"] - 1e-12
        )
        checks["rmse_not_above_reference"] = (
            treatment["path_three_d_rmse_m"]
            <= reference["path_three_d_rmse_m"] + 1e-12
        )
    elif tag == "default":
        checks["success_ge_0.70"] = treatment["success_rate"] >= 0.70 - 1e-12
        checks["rmse_le_5.0"] = treatment["path_three_d_rmse_m"] <= 5.0 + 1e-12
    elif tag == "strict125":
        checks["success_ge_0.50"] = treatment["success_rate"] >= 0.50 - 1e-12
        checks["rmse_le_6.5"] = treatment["path_three_d_rmse_m"] <= 6.5 + 1e-12
        checks["timeout_le_0.531"] = treatment["timeout_rate"] <= 0.531 + 1e-12
    elif tag == "h1":
        checks["success_ge_0.95"] = treatment["success_rate"] >= 0.95 - 1e-12
        checks["rmse_le_5.0"] = treatment["path_three_d_rmse_m"] <= 5.0 + 1e-12
        checks["saturation_le_0.40"] = (
            treatment["material_saturation_rate"] <= 0.40 + 1e-12
        )
    elif tag == "h3":
        checks["success_ge_0.95"] = treatment["success_rate"] >= 0.95 - 1e-12
        checks["rmse_le_3.4"] = treatment["path_three_d_rmse_m"] <= 3.4 + 1e-12
        checks["saturation_le_0.45"] = (
            treatment["material_saturation_rate"] <= 0.45 + 1e-12
        )
    elif tag == "h2":
        return {"status": "diagnostic", "checks": {}}

    if tag in {"default", "strict125"}:
        checks["ground_route_speed_min_ge_0.75"] = (
            treatment["ground_route_speed_ratio_episode_min_mean"] >= 0.75 - 1e-12
        )
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def build_report(
    *,
    summary_path: Path,
    treatment_arm: str,
    reference_arm: str,
    train_seed_count: int,
    seed_count: int,
    tags: Sequence[str] | None = None,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    index = _index_rows(summary)
    treatment_arm = _canonical_arm(treatment_arm)
    reference_arm = _canonical_arm(reference_arm)
    requested_tags = tuple(tags) if tags is not None else TAGS
    if len(set(requested_tags)) != len(requested_tags):
        raise ValueError("report tags must be unique")
    unknown_tags = sorted(set(requested_tags) - set(TAGS))
    if unknown_tags:
        raise ValueError(f"unknown report tags: {unknown_tags}")
    stages: dict[str, Any] = {}
    for tag in requested_tags:
        paired = _paired(
            index,
            treatment_arm=treatment_arm,
            reference_arm=reference_arm,
            tag=tag,
        )
        stage: dict[str, Any] = {"tag": tag, **paired}
        if paired["status"] != "complete":
            stage["gate"] = {"status": "incomplete", "checks": {}}
        else:
            seeds = paired["paired_seeds"]
            eval_seed_counts = sorted({int(row["eval_seeds"]) for row in paired["rows"]})
            complete = (
                len(seeds) == train_seed_count
                and eval_seed_counts == [seed_count]
                and not paired["incomplete_metrics"]
            )
            if not complete:
                stage["status"] = "incomplete"
                stage["gate"] = {"status": "incomplete", "checks": {}}
            else:
                stage["gate"] = _gate(
                    tag=tag,
                    treatment={
                        name: float(paired["means"]["treatment"][name]["mean"])
                        for name in METRICS
                    },
                    reference={
                        name: float(paired["means"]["reference"][name]["mean"])
                        for name in METRICS
                    },
                )
        stages[tag] = stage

    primary = tuple(
        tag
        for tag in (*IN_DOMAIN_TAGS, "grid_hs", *RETENTION_TAGS, *ROBUSTNESS_TAGS)
        if tag in stages
    )
    if not primary:
        overall = "pressure_only"
    elif any(stages[tag]["status"] != "complete" for tag in primary):
        overall = "incomplete"
    elif all(stages[tag]["gate"]["status"] == "pass" for tag in primary):
        overall = "pass"
    else:
        overall = "fail"
    return {
        "schema_version": "auv_pc_grid_rpm_only_report.v1",
        "summary_path": str(summary_path),
        "treatment_arm": treatment_arm,
        "reference_arm": reference_arm,
        "train_seed_count": train_seed_count,
        "seed_count": seed_count,
        "requested_tags": list(requested_tags),
        "primary_tags": list(primary),
        "pressure_only_tags": list(PRESSURE_TAGS),
        "stages": stages,
        "overall_status": overall,
        "notes": [
            "RPM-only and full-action arms use the same teacher demos, frozen BC, training seeds, and evaluation seeds.",
            "The residual action mask is [0, 0, 1]; rudder and stern-plane residuals must be exactly zero.",
            "grid_hs is held out from both arms' training data.",
            "The 0.75 ground-route-speed gate uses the mean of per-episode minima; the global minimum is retained as a diagnostic.",
            "Ground-route speed is projected from closed-loop NED ground velocity, not from the commanded speed.",
            "H4 and C4 are pressure-only and do not enter the primary gate.",
        ],
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PC RPM-only 残差权限报告",
        "",
        f"Treatment: `{report['treatment_arm']}`",
        f"Reference: `{report['reference_arm']}`",
        f"训练 seed 数: `{report['train_seed_count']}`；每 tag 评估 seed 数: `{report['seed_count']}`",
        "",
        "| Tag | Role | RPM success | Full success | RPM RMSE | Full RMSE | RPM timeout | RPM sat | Route min (ep mean) | Route min (global) | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for tag in report["requested_tags"]:
        stage = report["stages"][tag]
        if stage["status"] != "complete":
            lines.append(
                f"| {tag} | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `incomplete` |"
            )
            continue
        treatment = stage["means"]["treatment"]
        reference = stage["means"]["reference"]
        lines.append(
            "| {tag} | {role} | {ts} | {rs} | {tr} m | {rr} m | {tt} | {sat} | {ground_episode} | {ground_global} | `{gate}` |".format(
                tag=tag,
                role=(
                    "heldout"
                    if tag == "grid_hs"
                    else "in_domain"
                    if tag in IN_DOMAIN_TAGS
                    else "retention"
                    if tag in RETENTION_TAGS
                    else "robustness"
                    if tag in ROBUSTNESS_TAGS
                    else "diagnostic"
                    if tag == "h2"
                    else "pressure_only"
                ),
                ts=_fmt(treatment["success_rate"]["mean"]),
                rs=_fmt(reference["success_rate"]["mean"]),
                tr=_fmt(treatment["path_three_d_rmse_m"]["mean"], 2),
                rr=_fmt(reference["path_three_d_rmse_m"]["mean"], 2),
                tt=_fmt(treatment["timeout_rate"]["mean"]),
                sat=_fmt(treatment["material_saturation_rate"]["mean"]),
                ground_episode=_fmt(
                    treatment["ground_route_speed_ratio_episode_min_mean"]["mean"]
                ),
                ground_global=_fmt(
                    treatment["ground_route_speed_ratio_min"]["mean"]
                ),
                gate=stage["gate"]["status"],
            )
        )
    lines.extend(["", "## 判定", "", f"- overall: `{report['overall_status']}`"])
    for note in report["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    report = build_report(
        summary_path=args.summary,
        treatment_arm=args.treatment_arm,
        reference_arm=args.reference_arm,
        train_seed_count=args.train_seed_count,
        seed_count=args.seed_count,
        tags=args.tags,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = args.output_dir / "report.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(markdown_path)
    if args.require_pass and report["overall_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
