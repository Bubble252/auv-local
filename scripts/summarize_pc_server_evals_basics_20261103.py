#!/usr/bin/env python3
"""Summarize fixed-seed PC server evaluations across arms and training seeds."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "success_rate",
    "timeout_rate",
    "safety_termination_rate",
    "material_saturation_rate",
    "path_three_d_rmse_m",
    "speed_rmse_mps",
    "yaw_rmse_rad",
    "command_hold_success_rate",
    "final_hold_fraction_mean",
    "path_completion_ratio_mean",
)
OPTIONAL_METRICS = (
    "pitch_rmse_rad",
    "mean_abs_speed_error_mps",
    "mean_abs_yaw_error_rad",
    "mean_abs_pitch_error_rad",
    "nav_position_error_rmse_m",
    "residual_action_abs_mean",
    "residual_action_abs_p95",
    "residual_action_abs_max",
    "residual_action_l2_mean",
    "base_action_l2_mean",
    "residual_to_base_l2_ratio_mean",
    "residual_rudder_abs_mean_rad",
    "residual_stern_abs_mean_rad",
    "residual_rpm_abs_mean_rpm",
    "guidance_route_speed_ratio_min",
    "ground_speed_ratio_min",
    "ground_speed_ratio_mean",
    "ground_route_speed_ratio_min",
    "ground_route_speed_ratio_episode_min_mean",
    "ground_route_speed_ratio_mean",
    "current_valid_rate",
    "terminal_speed_ratio",
    "guidance_current_valid_rate",
    "guidance_current_age_s_mean",
    "guidance_crab_deg_mean",
    "guidance_crab_deg_abs_mean",
    "guidance_speed_scale_mean",
    "guidance_terminal_phase_rate",
    "terminal_speed_ratio_mean",
    "guidance_terminal_speed_ratio_mean",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-arm", default="nodade")
    parser.add_argument(
        "--baseline-by-tag",
        action="append",
        default=[],
        metavar="TAG=ARM",
        help=(
            "override the paired baseline for one evaluation tag; may be repeated. "
            "Rows without an override use --baseline-arm"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="checkpoint token to include: integer, 'initial', or 'final'",
    )
    parser.add_argument(
        "--checkpoint-baseline",
        action="append",
        default=[],
        metavar="TARGET=SOURCE",
        help=(
            "pair a checkpoint with another checkpoint of the same arm/tag/seed; "
            "may be repeated, for example 99=initial"
        ),
    )
    parser.add_argument(
        "--history-ablation",
        action="append",
        choices=("none", "current-only", "current-zero", "history-zero", "history-reverse"),
        default=None,
        help=(
            "history-ablation evaluation to include; may be repeated. "
            "Defaults to normal evaluations only ('none')"
        ),
    )
    parser.add_argument(
        "--eval-seed-count",
        type=int,
        default=None,
        help="only include evaluations with exactly this many closed-loop seeds",
    )
    return parser.parse_args()


def _parse_baseline_by_tag(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        tag, separator, arm = value.partition("=")
        if not separator or not tag or not arm:
            raise ValueError(f"invalid --baseline-by-tag value {value!r}; expected TAG=ARM")
        if tag in result:
            raise ValueError(f"duplicate --baseline-by-tag value for tag {tag!r}")
        result[tag] = arm
    return result


def _parse_checkpoint_baseline(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        target, separator, source = value.partition("=")
        if not separator or not target or not source:
            raise ValueError(
                f"invalid --checkpoint-baseline value {value!r}; expected TARGET=SOURCE"
            )
        if target in result:
            raise ValueError(f"duplicate --checkpoint-baseline value for {target!r}")
        result[target] = source
    return result


def _arm_from_config(path: str) -> str:
    name = Path(path).name
    if "grid_gpu_frozen_bc_residual_nodade_s14_route_reward_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_s14_route_reward_lr3e-4_scale1p0"
    if "grid_gpu_frozen_bc_residual_dade_pose_s14_route_reward_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_dade_pose_s14_route_reward_lr3e-4_scale1p0"
    if "grid_dade_dagger_frozen_bc_residual_dade_pose_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_dade_grid_dagger_pose_lr3e-4_scale1p0"
    if "grid_gpu_frozen_bc_residual_dade_pose_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_dade_grid_gpu_pose_lr3e-4_scale1p0"
    if "grid_gpu_frozen_bc_residual_nodade_rpm_only_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_grid_rpm_only_lr3e-4_scale1p0"
    if "grid_gpu_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_grid_gpu_lr3e-4_scale1p0"
    if "grid_frozen_bc_residual_nodade_smooth1e-3_l2_1e-2_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_grid_smooth1e-3_l2_1e-2_lr3e-4_scale1p0"
    if "grid_frozen_bc_residual_nodade_smooth1e-3_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_grid_smooth1e-3_lr3e-4_scale1p0"
    if "grid_dagger_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_grid_dagger_lr3e-4_scale1p0"
    if "grid_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_grid_lr3e-4_scale1p0"
    if "mixed_domain_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_mixed_domain_lr3e-4_scale1p0"
    if "multienv_frozen_bc_residual_dade_pose_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_dade_multienv_pose_lr3e-4_scale1p0"
    if "multienv_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0"
    if "c2_alos_nodade_stable" in name:
        return "c2_nodade"
    if "c2_alos_dade_stable" in name:
        return "c2_dade"
    if "h3_speed_2p25_2p4_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0"
    if "h1_speed_1p8_2p0_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_h1_lr3e-4_scale1p0"
    if "h2_speed_2p0_2p25_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_h2_lr3e-4_scale1p0"
    if "c4_stress_frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_c4_lr3e-4_scale1p0"
    if "frozen_bc_residual_dade_head_isolated_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_dade_head_isolated_lr3e-4_scale1p0"
    if "frozen_bc_residual_dade_history_only_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_dade_history_only_lr3e-4_scale1p0"
    if "frozen_bc_residual_nodade_lr3e-4_scale1p0" in name:
        return "frozen_bc_residual_nodade_lr3e-4_scale1p0"
    if "h1_speed_1p8_2p0_frozen_bc_residual_nodade" in name:
        return "frozen_bc_residual_nodade_h1"
    if "h2_speed_2p0_2p25_frozen_bc_residual_nodade" in name:
        return "frozen_bc_residual_nodade_h2"
    if "c4_stress_frozen_bc_residual_nodade" in name:
        return "frozen_bc_residual_nodade_c4"
    if "frozen_bc_residual_dade_history_only" in name:
        return "frozen_bc_residual_dade_history_only"
    if "frozen_bc_residual_dade_history_required" in name:
        return "frozen_bc_residual_dade_history_required"
    if "frozen_bc_residual_dade_sidechannel10" in name:
        return "frozen_bc_residual_dade_sidechannel10"
    if "frozen_bc_residual_dade_head" in name:
        return "frozen_bc_residual_dade_head"
    if "frozen_bc_residual_dade" in name:
        return "frozen_bc_residual_dade"
    if "frozen_bc_residual_nodade" in name:
        return "frozen_bc_residual_nodade"
    if "dade_head_isolated_lr5e-4" in name:
        return "dade_head_isolated_lr5e-4"
    if "dade_history_only" in name:
        return "dade_history_only"
    if "dade_head_sidechannel10" in name:
        return "dade_head_sidechannel10"
    if "dade_head_dropout10" in name:
        return "dade_head_dropout10"
    if "dade_head_dropout05" in name:
        return "dade_head_dropout05"
    if "dade_head" in name:
        return "dade_head"
    if "_nodade_" in name:
        return "nodade"
    if "dade_aux_current" in name or "_dade_" in name:
        return "dade"
    raise ValueError(f"cannot infer arm from training config: {path}")


def _arm_from_summary(summary: dict[str, Any]) -> str:
    arm = _arm_from_config(summary["training_config"])
    metadata = summary.get("checkpoint_metadata") or {}
    if metadata.get("policy_kind") == "pc_supervised_residual":
        return f"supervised_{arm}"
    return arm


def _training_seed(summary_path: Path) -> int:
    for part in reversed(summary_path.parent.parts):
        if re.fullmatch(r"ck\w+_seed\d+_\d+(?:_[a-z0-9-]+)?", part):
            continue
        match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", part)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"cannot infer training seed from path: {summary_path}")


def _history_ablation(summary_path: Path) -> str:
    suffix = summary_path.parent.name.rsplit("_", 1)[-1]
    if suffix in {"current-only", "current-zero", "history-zero", "history-reverse"}:
        return suffix
    return "none"


def _checkpoint(summary: dict[str, Any]) -> str:
    name = Path(summary["checkpoint"]).name
    if name == "model_initial.pt":
        return "initial"
    if name == "final.pt":
        return "final"
    match = re.fullmatch(r"model_(\d+)\.pt", name)
    if match is None:
        raise ValueError(f"cannot infer checkpoint from summary: {name}")
    return match.group(1)


def _invalid_run_marker(summary: dict[str, Any]) -> Path | None:
    """Return the invalidation marker attached to a run, if present."""

    checkpoint = Path(summary["checkpoint"])
    marker = checkpoint.parent / "INVALID_FOR_DADE_CONCLUSIONS.json"
    return marker if marker.is_file() else None


def _tag(root: Path, summary_path: Path) -> str:
    parts = [part.lower() for part in summary_path.relative_to(root).parts[:-1]]
    for part in parts:
        if part in {"grid_lw", "grid_ls", "grid_hw", "grid_hs"}:
            return part
    if any(part == "c2_clean" for part in parts):
        return "c2_clean"
    if any(part == "c2" for part in parts):
        return "c2"
    if any("strict125" in part or "strict_125" in part for part in parts):
        return "strict125"
    if any(part == "c4" or part.startswith("c4_") or "c4_stress" in part for part in parts):
        return "c4"
    for part in parts:
        if re.fullmatch(r"h[0-4]", part):
            return part
    return "default"


def _finite_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary["metrics"]
    missing = [name for name in METRICS if name not in metrics]
    if missing:
        raise ValueError(f"evaluation summary is missing metrics: {missing}")
    result = {name: float(metrics[name]) for name in METRICS}
    for name in OPTIONAL_METRICS:
        if name in metrics:
            result[name] = float(metrics[name])
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("evaluation summary contains non-finite metrics")
    return result


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


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> int:
    args = _parse_args()
    baseline_by_tag = _parse_baseline_by_tag(args.baseline_by_tag)
    checkpoint_baseline = _parse_checkpoint_baseline(args.checkpoint_baseline)
    history_ablations = set(args.history_ablation or ("none",))
    rows: list[dict[str, Any]] = []
    skipped_invalid: list[dict[str, str]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for root in args.eval_root:
        for summary_path in sorted(root.glob("**/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("schema_version") != "auv_pc_ppo_evaluation.v1":
                continue
            history_ablation = _history_ablation(summary_path)
            if history_ablation not in history_ablations:
                continue
            if (
                args.eval_seed_count is not None
                and len(summary.get("seeds", [])) != args.eval_seed_count
            ):
                continue
            invalid_marker = _invalid_run_marker(summary)
            if invalid_marker is not None:
                skipped_invalid.append(
                    {
                        "summary": str(summary_path),
                        "marker": str(invalid_marker),
                        "reason": "INVALID_FOR_DADE_CONCLUSIONS",
                    }
                )
                continue
            arm = _arm_from_summary(summary)
            train_seed = _training_seed(summary_path)
            checkpoint = _checkpoint(summary)
            if args.checkpoint and checkpoint not in args.checkpoint:
                continue
            tag = _tag(root, summary_path)
            key = (arm, train_seed, checkpoint, tag, history_ablation)
            if key in seen:
                raise ValueError(f"duplicate evaluation for {key}: {summary_path}")
            seen.add(key)
            rows.append(
                {
                    "arm": arm,
                    "train_seed": train_seed,
                    "checkpoint": checkpoint,
                    "tag": tag,
                    "history_ablation": history_ablation,
                    "eval_seeds": len(summary["seeds"]),
                    "path": str(summary_path),
                    **{name: value for name, value in _finite_metrics(summary).items()},
                }
            )

    indexed = {
        (
            row["arm"],
            row["train_seed"],
            row["checkpoint"],
            row["tag"],
            row["history_ablation"],
        ): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    for row in rows:
        baseline_arm = baseline_by_tag.get(row["tag"], args.baseline_arm)
        if row["arm"] == baseline_arm:
            continue
        baseline = indexed.get(
            (
                baseline_arm,
                row["train_seed"],
                row["checkpoint"],
                row["tag"],
                row["history_ablation"],
            )
        )
        if baseline is None:
            continue
        paired.append(
            {
                "arm": row["arm"],
                "baseline_arm": baseline_arm,
                "train_seed": row["train_seed"],
                "checkpoint": row["checkpoint"],
                "tag": row["tag"],
                "history_ablation": row["history_ablation"],
                **{
                    f"delta_{name}": row[name] - baseline[name]
                    for name in (*METRICS, *OPTIONAL_METRICS)
                    if name in row and name in baseline
                },
            }
        )

    checkpoint_paired: list[dict[str, Any]] = []
    for row in rows:
        checkpoint_source = checkpoint_baseline.get(row["checkpoint"])
        if checkpoint_source is None:
            continue
        effective_checkpoint_source = checkpoint_source
        baseline = indexed.get(
            (
                row["arm"],
                row["train_seed"],
                checkpoint_source,
                row["tag"],
                row["history_ablation"],
            )
        )
        if (
            baseline is None
            and checkpoint_source == "initial"
            and row["arm"].startswith("c2_")
        ):
            effective_checkpoint_source = "0"
            baseline = indexed.get(
                (
                    row["arm"],
                    row["train_seed"],
                    "0",
                    row["tag"],
                    row["history_ablation"],
                )
            )
        if baseline is None:
            continue
        if effective_checkpoint_source == row["checkpoint"]:
            continue
        checkpoint_paired.append(
            {
                "arm": row["arm"],
                "checkpoint_baseline": effective_checkpoint_source,
                "checkpoint_baseline_requested": checkpoint_source,
                "train_seed": row["train_seed"],
                "checkpoint": row["checkpoint"],
                "tag": row["tag"],
                "history_ablation": row["history_ablation"],
                **{
                    f"delta_{name}": row[name] - baseline[name]
                    for name in (*METRICS, *OPTIONAL_METRICS)
                    if name in row and name in baseline
                },
            }
        )

    grouped_raw: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_paired: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_checkpoint_paired: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped_raw[
            (row["arm"], row["checkpoint"], row["tag"], row["history_ablation"])
        ].append(row)
    for row in paired:
        grouped_paired[
            (row["arm"], row["checkpoint"], row["tag"], row["history_ablation"])
        ].append(row)
    for row in checkpoint_paired:
        grouped_checkpoint_paired[
            (row["arm"], row["checkpoint"], row["tag"], row["history_ablation"])
        ].append(row)

    aggregates = []
    for key, group in sorted(grouped_raw.items()):
        paired_group = grouped_paired.get(key, [])
        checkpoint_paired_group = grouped_checkpoint_paired.get(key, [])
        aggregates.append(
            {
                "arm": key[0],
                "checkpoint": key[1],
                "tag": key[2],
                "history_ablation": key[3],
                "train_seeds": sorted(row["train_seed"] for row in group),
                "baseline_arm": baseline_by_tag.get(key[2], args.baseline_arm),
                "checkpoint_baseline": (
                    checkpoint_paired_group[0]["checkpoint_baseline"]
                    if checkpoint_paired_group
                    else checkpoint_baseline.get(key[1])
                ),
                "raw": {
                    name: _describe([row[name] for row in group if name in row])
                    for name in (*METRICS, *OPTIONAL_METRICS)
                },
                "paired_delta_vs_baseline": {
                    name: _describe(
                        [
                            row[f"delta_{name}"]
                            for row in paired_group
                            if f"delta_{name}" in row
                        ]
                    )
                    for name in (*METRICS, *OPTIONAL_METRICS)
                },
                "paired_delta_vs_checkpoint_baseline": {
                    name: _describe(
                        [
                            row[f"delta_{name}"]
                            for row in checkpoint_paired_group
                            if f"delta_{name}" in row
                        ]
                    )
                    for name in (*METRICS, *OPTIONAL_METRICS)
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "auv_pc_server_eval_summary.v1",
        "eval_roots": [str(path) for path in args.eval_root],
        "baseline_arm": args.baseline_arm,
        "baseline_by_tag": baseline_by_tag,
        "checkpoint_baseline": checkpoint_baseline,
        "history_ablations": sorted(history_ablations),
        "invalid_runs_skipped": sorted(
            skipped_invalid, key=lambda row: row["summary"]
        ),
        "rows": sorted(rows, key=lambda row: (row["tag"], row["checkpoint"], row["arm"], row["train_seed"])),
        "paired": sorted(
            paired,
            key=lambda row: (row["tag"], row["checkpoint"], row["arm"], row["train_seed"]),
        ),
        "checkpoint_paired": sorted(
            checkpoint_paired,
            key=lambda row: (row["tag"], row["checkpoint"], row["arm"], row["train_seed"]),
        ),
        "aggregates": aggregates,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = ["# PC server multi-seed summary", "", f"- default baseline: `{args.baseline_arm}`"]
    if skipped_invalid:
        lines.append(f"- invalid runs skipped: `{len(skipped_invalid)}`")
    lines.append("")
    if baseline_by_tag:
        lines.extend(
            f"- `{tag}` baseline: `{arm}`" for tag, arm in sorted(baseline_by_tag.items())
        )
        lines.append("")
    for aggregate in aggregates:
        raw = aggregate["raw"]
        delta = aggregate["paired_delta_vs_baseline"]
        lines.extend(
            [
                f"## `{aggregate['arm']}` / `{aggregate['tag']}` / ck`{aggregate['checkpoint']}`",
                "",
                f"- training seeds: `{aggregate['train_seeds']}`",
                f"- success: `{_fmt(raw['success_rate']['mean'])}` +/- `{_fmt(raw['success_rate']['std'])}`",
                f"- 3D RMSE: `{_fmt(raw['path_three_d_rmse_m']['mean'])}` +/- `{_fmt(raw['path_three_d_rmse_m']['std'])}` m",
                f"- timeout: `{_fmt(raw['timeout_rate']['mean'])}`",
                f"- safety: `{_fmt(raw['safety_termination_rate']['mean'])}`",
                f"- saturation: `{_fmt(raw['material_saturation_rate']['mean'])}`",
            ]
        )
        if raw["residual_action_abs_mean"]["n"]:
            lines.extend(
                [
                    f"- residual action abs mean: `{_fmt(raw['residual_action_abs_mean']['mean'])}`",
                    f"- residual/base action L2: `{_fmt(raw['residual_to_base_l2_ratio_mean']['mean'])}`",
                    f"- residual rudder/stern/RPM abs mean: "
                    f"`{_fmt(raw['residual_rudder_abs_mean_rad']['mean'])}` / "
                    f"`{_fmt(raw['residual_stern_abs_mean_rad']['mean'])}` rad / "
                    f"`{_fmt(raw['residual_rpm_abs_mean_rpm']['mean'])}` RPM",
                ]
            )
        if aggregate["arm"] != aggregate["baseline_arm"] and delta["success_rate"]["n"]:
            lines.extend(
                [
                    f"- paired baseline: `{aggregate['baseline_arm']}`",
                    f"- paired success delta: `{_fmt(delta['success_rate']['mean'])}` "
                    f"(min `{_fmt(delta['success_rate']['min'])}`, max `{_fmt(delta['success_rate']['max'])}`)",
                    f"- paired 3D RMSE delta: `{_fmt(delta['path_three_d_rmse_m']['mean'])}` m "
                    f"(min `{_fmt(delta['path_three_d_rmse_m']['min'])}`, max `{_fmt(delta['path_three_d_rmse_m']['max'])}`)",
                ]
            )
        checkpoint_delta = aggregate["paired_delta_vs_checkpoint_baseline"]
        if checkpoint_delta["success_rate"]["n"]:
            lines.extend(
                [
                    f"- checkpoint baseline: `{aggregate['checkpoint_baseline']}`",
                    f"- success delta vs checkpoint baseline: "
                    f"`{_fmt(checkpoint_delta['success_rate']['mean'])}` "
                    f"(min `{_fmt(checkpoint_delta['success_rate']['min'])}`, "
                    f"max `{_fmt(checkpoint_delta['success_rate']['max'])}`)",
                    f"- 3D RMSE delta vs checkpoint baseline: "
                    f"`{_fmt(checkpoint_delta['path_three_d_rmse_m']['mean'])}` m",
                ]
            )
        lines.append("")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired": len(paired), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
