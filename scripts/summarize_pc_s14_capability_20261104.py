#!/usr/bin/env python3
"""Summarize S14 route-speed reward arms and their paired S13 references."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_pc_server_evals_basics_20261103 import (  # noqa: E402
    _arm_from_summary,
    _checkpoint,
    _describe,
    _finite_metrics,
    _tag,
    _training_seed,
)

S14_NODADE_ARM = "frozen_bc_residual_nodade_s14_route_reward_lr3e-4_scale1p0"
S14_DADE_ARM = "frozen_bc_residual_dade_pose_s14_route_reward_lr3e-4_scale1p0"
S13_NODADE_ARM = "supervised_frozen_bc_residual_nodade_grid_gpu_lr3e-4_scale1p0"
S13_DADE_ARM = "supervised_frozen_bc_residual_dade_grid_gpu_pose_lr3e-4_scale1p0"

REPORT_METRICS = (
    "success_rate",
    "timeout_rate",
    "safety_termination_rate",
    "material_saturation_rate",
    "path_three_d_rmse_m",
    "path_completion_ratio_mean",
    "ground_route_speed_ratio_episode_min_mean",
    "mean_abs_yaw_error_rad",
    "mean_abs_speed_error_mps",
    "guidance_current_valid_rate",
    "guidance_current_age_s_mean",
    "guidance_crab_deg_mean",
    "terminal_speed_ratio_mean",
)

GATES: dict[str, dict[str, tuple[str, float]]] = {
    "default": {
        "success_rate": (">=", 0.70),
        "path_three_d_rmse_m": ("<=", 5.0),
    },
    "strict125": {
        "success_rate": (">=", 0.50),
        "path_three_d_rmse_m": ("<=", 6.5),
        "timeout_rate": ("<=", 0.531),
    },
    "h1": {
        "success_rate": (">=", 0.95),
        "path_three_d_rmse_m": ("<=", 5.0),
        "material_saturation_rate": ("<=", 0.40),
    },
    "h3": {
        "success_rate": (">=", 0.95),
        "path_three_d_rmse_m": ("<=", 3.4),
        "material_saturation_rate": ("<=", 0.45),
    },
    "h4": {
        "path_three_d_rmse_m": ("<=", 3.8),
        "material_saturation_rate": ("<=", 0.45),
    },
    "grid_hs": {
        "success_rate": (">=", 0.85),
        "path_three_d_rmse_m": ("<=", 6.0),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s14-eval-root", required=True, type=Path)
    parser.add_argument("--s13-nodade-eval-root", type=Path)
    parser.add_argument("--s13-dade-eval-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tags", default="grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4")
    parser.add_argument("--eval-seed-count", type=int, default=32)
    parser.add_argument("--s14-nodade-arm", default=S14_NODADE_ARM)
    parser.add_argument("--s14-dade-arm", default=S14_DADE_ARM)
    parser.add_argument("--s13-nodade-arm", default=S13_NODADE_ARM)
    parser.add_argument("--s13-dade-arm", default=S13_DADE_ARM)
    return parser.parse_args()


def _load_rows(root: Path, tags: set[str], eval_seed_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("**/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema_version") != "auv_pc_ppo_evaluation.v1":
            continue
        if _checkpoint(summary) != "final":
            continue
        if len(summary.get("seeds", [])) != eval_seed_count:
            continue
        tag = _tag(root, path)
        if tag not in tags:
            continue
        rows.append(
            {
                "arm": _arm_from_summary(summary),
                "train_seed": _training_seed(path),
                "tag": tag,
                "path": str(path),
                **_finite_metrics(summary),
            }
        )
    return rows


def _metric(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "train_seeds": sorted(row["train_seed"] for row in rows),
        **{
            name: _describe(
                [value for row in rows if (value := _metric(row, name)) is not None]
            )
            for name in REPORT_METRICS
        },
    }


def _paired(
    treatment_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    tags: set[str],
) -> dict[str, Any]:
    baseline = {
        (row["train_seed"], row["tag"]): row for row in baseline_rows
    }
    result: dict[str, Any] = {}
    for tag in sorted(tags):
        pairs: list[dict[str, float]] = []
        for row in treatment_rows:
            if row["tag"] != tag:
                continue
            reference = baseline.get((row["train_seed"], tag))
            if reference is None:
                continue
            deltas: dict[str, float] = {}
            for name in REPORT_METRICS:
                left = _metric(row, name)
                right = _metric(reference, name)
                if left is not None and right is not None:
                    deltas[f"delta_{name}"] = left - right
            pairs.append(deltas)
        result[tag] = {
            "n": len(pairs),
            **{
                f"delta_{name}": _describe(
                    [
                        pair[f"delta_{name}"]
                        for pair in pairs
                        if f"delta_{name}" in pair
                    ]
                )
                for name in REPORT_METRICS
            },
        }
    return result


def _comparison(
    treatment_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    tags: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tag in sorted(tags):
        treatment = [row for row in treatment_rows if row["tag"] == tag]
        baseline = [row for row in baseline_rows if row["tag"] == tag]
        result[tag] = {
            "treatment": _aggregate(treatment),
            "baseline": _aggregate(baseline),
            "paired": _paired(treatment, baseline, {tag})[tag],
        }
    return result


def _gate_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tag.setdefault(row["tag"], []).append(row)
    status: dict[str, Any] = {}
    for tag, gates in GATES.items():
        tag_rows = by_tag.get(tag, [])
        aggregate = _aggregate(tag_rows)
        checks: list[dict[str, Any]] = []
        for metric, (operator, threshold) in gates.items():
            mean = (aggregate.get(metric) or {}).get("mean")
            passed = (
                mean is not None
                and (mean >= threshold if operator == ">=" else mean <= threshold)
            )
            checks.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "mean": mean,
                    "passed": passed,
                }
            )
        status[tag] = {
            "status": "pass" if checks and all(check["passed"] for check in checks) else "fail",
            "checks": checks,
        }
    return status


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _mean(table: dict[str, Any] | None, metric: str) -> float | None:
    if not table:
        return None
    return (table.get(metric) or {}).get("mean")


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# PC S14 capability summary", ""]
    lines.extend(
        [
            f"- S14 eval root: `{report['s14_eval_root']}`",
            f"- S14 rows: `{report['row_counts']['s14']}`",
            f"- S13 no-DADE rows: `{report['row_counts']['s13_nodade']}`",
            f"- S13 DADE rows: `{report['row_counts']['s13_dade']}`",
            "",
        ]
    )
    for name, comparison in report["comparisons"].items():
        lines.extend(
            [
                f"## `{name}`",
                "",
                "| tag | n | success | timeout | 3D RMSE | sat | ground route min | Δsuccess | ΔRMSE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tag, entry in comparison.items():
            raw = entry["treatment"]
            paired = entry["paired"]
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    tag,
                    raw["n"],
                    _fmt(_mean(raw, "success_rate")),
                    _fmt(_mean(raw, "timeout_rate")),
                    _fmt(_mean(raw, "path_three_d_rmse_m"), 3),
                    _fmt(_mean(raw, "material_saturation_rate")),
                    _fmt(_mean(raw, "ground_route_speed_ratio_episode_min_mean")),
                    _fmt(_mean(paired, "delta_success_rate")),
                    _fmt(_mean(paired, "delta_path_three_d_rmse_m"), 3),
                )
            )
        lines.append("")
    lines.extend(["## Gate status", "", "| arm | tag | status | failed checks |", "|---|---|---|---|"])
    for arm, tags in report["gates"].items():
        for tag, status in tags.items():
            failed = [
                f"{check['metric']}={_fmt(check['mean'])} (need {check['operator']}{check['threshold']})"
                for check in status["checks"]
                if not check["passed"]
            ]
            lines.append(
                f"| `{arm}` | `{tag}` | `{status['status']}` | {'; '.join(failed) or '-'} |"
            )
    lines.extend(
        [
            "",
            "Notes: `ground route min` is the episode-min mean of the true NED route-tangent",
            "ground-speed ratio. `terminal_speed_ratio`, `current_valid_rate` and `crab_deg`",
            "remain in `summary.json` for diagnosis.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    tags = set(args.tags.split())
    s14_rows = _load_rows(args.s14_eval_root, tags, args.eval_seed_count)
    s13_nodade_rows = (
        []
        if args.s13_nodade_eval_root is None
        else _load_rows(args.s13_nodade_eval_root, tags, args.eval_seed_count)
    )
    s13_dade_rows = (
        []
        if args.s13_dade_eval_root is None
        else _load_rows(args.s13_dade_eval_root, tags, args.eval_seed_count)
    )

    treatment_nodade = [row for row in s14_rows if row["arm"] == args.s14_nodade_arm]
    treatment_dade = [row for row in s14_rows if row["arm"] == args.s14_dade_arm]
    baseline_nodade = [row for row in s13_nodade_rows if row["arm"] == args.s13_nodade_arm]
    baseline_dade = [row for row in s13_dade_rows if row["arm"] == args.s13_dade_arm]

    comparisons: dict[str, Any] = {}
    if treatment_nodade:
        comparisons["S14-A vs S13 no-DADE"] = _comparison(
            treatment_nodade, baseline_nodade, tags
        )
    if treatment_dade:
        comparisons["S14-B vs S14-A"] = _comparison(
            treatment_dade, treatment_nodade, tags
        )
        comparisons["S14-B vs S13 DADE-pose"] = _comparison(
            treatment_dade, baseline_dade, tags
        )

    report = {
        "schema_version": "auv_pc_s14_capability_summary.v1",
        "s14_eval_root": str(args.s14_eval_root),
        "s13_nodade_eval_root": (
            None if args.s13_nodade_eval_root is None else str(args.s13_nodade_eval_root)
        ),
        "s13_dade_eval_root": (
            None if args.s13_dade_eval_root is None else str(args.s13_dade_eval_root)
        ),
        "eval_seed_count": args.eval_seed_count,
        "tags": sorted(tags),
        "row_counts": {
            "s14": len(s14_rows),
            "s13_nodade": len(s13_nodade_rows),
            "s13_dade": len(s13_dade_rows),
        },
        "comparisons": comparisons,
        "gates": {
            "S14-A": _gate_status(treatment_nodade),
            "S14-B": _gate_status(treatment_dade),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "s14_rows": len(s14_rows),
                "comparisons": list(comparisons),
                "output": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
