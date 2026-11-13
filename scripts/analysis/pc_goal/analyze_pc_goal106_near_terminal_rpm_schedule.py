#!/usr/bin/env python3
"""Audit Goal106's pre-registered near-terminal rpm schedule matrix."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from analyze_pc_goal104_channel_parameterization import (
    ARM,
    EVAL_SCALE,
    EXPECTED_CHECKPOINT_SHA,
    EXPECTED_SEEDS,
    TRAINING_SCALE,
    AuditError,
    _fmt,
    _mcnemar_exact_p,
    _read_json,
    _success_by_seed,
    _validate_trace,
)
from run_pc_goal106_near_terminal_rpm_schedule import PROFILES, Profile

SCHEMA_VERSION = "auv_pc_goal106_near_terminal_rpm.v1"
SCENARIOS = ("default", "strict125")
BASELINE_PROFILE = "C0"
EXPECTED_SUCCESS = {"default": 33, "strict125": 20}
PATH_RATIO_LIMIT = 1.05
GOAL104_U050_GAINED = {25, 51, 56, 59}
EXPECTED_RUNS = {
    f"{ARM}_{scenario}_{profile}"
    for scenario in SCENARIOS
    for profile in PROFILES
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal106_near_terminal_rpm_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal106_near_terminal_rpm_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal106_near_terminal_rpm_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _float_list(summary: dict[str, Any], key: str) -> list[float] | None:
    value = summary.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise AuditError(f"{key} must be a three-value list")
    return [float(item) for item in value]


def _validate_profile_summary(
    summary_path: Path, summary: dict[str, Any], spec: Profile
) -> None:
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise AuditError(f"{summary_path}: checkpoint SHA mismatch")
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - EVAL_SCALE) > 1e-12:
        raise AuditError(
            f"{summary_path}: scale override {recorded_scale!r} != {EVAL_SCALE}"
        )
    far = _float_list(summary, "actuator_residual_channel_gains")
    if far is None or any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(far, spec.far_gains, strict=True)
    ):
        raise AuditError(f"{summary_path}: far channel gains mismatch")
    near = _float_list(summary, "actuator_residual_channel_gains_near")
    if near is None and spec.near_gains is not None:
        raise AuditError(f"{summary_path}: missing near channel gains")
    if near is not None and spec.near_gains is None:
        raise AuditError(f"{summary_path}: unexpected near channel gains")
    if near is not None and spec.near_gains is not None:
        if any(
            abs(actual - expected) > 1e-12
            for actual, expected in zip(near, spec.near_gains, strict=True)
        ):
            raise AuditError(f"{summary_path}: near channel gains mismatch")
        recorded_near_m = summary.get(
            "actuator_residual_channel_gain_near_distance_m"
        )
        recorded_far_m = summary.get("actuator_residual_channel_gain_far_distance_m")
        if (
            recorded_near_m is None
            or recorded_far_m is None
            or abs(float(recorded_near_m) - float(spec.near_distance_m)) > 1e-12
            or abs(float(recorded_far_m) - float(spec.far_distance_m)) > 1e-12
        ):
            raise AuditError(f"{summary_path}: channel-gain schedule distances mismatch")


def _run_record(run_dir: Path, *, scenario: str, profile: str) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    spec = PROFILES[profile]
    _validate_profile_summary(summary_path, summary, spec)
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise AuditError(f"{summary_path}: expected 40 dev episodes")
    _validate_trace(run_dir / "trace.npz")
    by_seed = _success_by_seed(episodes, summary_path)
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise AuditError(f"{summary_path}: missing metrics object")
    path_rmse = metrics.get("path_three_d_rmse_m")
    if not isinstance(path_rmse, (int, float)) or not math.isfinite(float(path_rmse)):
        raise AuditError(f"{summary_path}: invalid path_three_d_rmse_m")

    final_distance: dict[int, float] = {}
    pre_goal_rmse: dict[int, float] = {}
    for row in episodes:
        seed = int(row["seed"])
        value = row.get("final_goal_distance_m")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            final_distance[seed] = float(value)
        value = row.get("pre_goal_3d_rmse_m")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            pre_goal_rmse[seed] = float(value)
    if set(final_distance) != EXPECTED_SEEDS:
        raise AuditError(f"{summary_path}: missing finite final_goal_distance_m")
    if set(pre_goal_rmse) != EXPECTED_SEEDS:
        raise AuditError(f"{summary_path}: missing finite pre_goal_3d_rmse_m")

    return {
        "run": run_dir.name,
        "scenario": scenario,
        "profile": profile,
        "far_gains": list(spec.far_gains),
        "near_gains": None if spec.near_gains is None else list(spec.near_gains),
        "near_distance_m": spec.near_distance_m,
        "far_distance_m": spec.far_distance_m,
        "episodes": len(episodes),
        "success": sum(by_seed.values()),
        "success_by_seed": by_seed,
        "success_pnav": sum(bool(row.get("success_pnav_tight", False)) for row in episodes),
        "success_eta_ever": sum(
            bool(row.get("success_eta_tight_ever", False)) for row in episodes
        ),
        "success_eta_done": sum(
            bool(row.get("success_eta_tight_at_done", False)) for row in episodes
        ),
        "timeout": sum(bool(row.get("timeout", False)) for row in episodes),
        "safety": sum(bool(row.get("safety", False)) for row in episodes),
        "path_three_d_rmse_m": float(path_rmse),
        "residual_magnitude_clip_rate": float(
            metrics.get("actuator_residual_magnitude_clip_rate", math.nan)
        ),
        "residual_rate_clip_rate": float(
            metrics.get("actuator_residual_rate_clip_rate", math.nan)
        ),
        "final_goal_distance_by_seed": final_distance,
        "pre_goal_3d_rmse_by_seed": pre_goal_rmse,
    }


def _scan(root: Path) -> dict[str, dict[str, Path]]:
    if not root.is_dir():
        raise AuditError(f"{root}: missing near-terminal rpm root")
    groups: dict[str, dict[str, Path]] = {
        scenario: {profile: Path() for profile in PROFILES}
        for scenario in SCENARIOS
    }
    found: set[str] = set()
    prefix = f"{ARM}_"
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not run_dir.name.startswith(prefix):
            continue
        parts = run_dir.name[len(prefix) :].split("_", 1)
        if len(parts) != 2:
            continue
        scenario, profile = parts
        if scenario not in SCENARIOS or profile not in PROFILES:
            continue
        groups[scenario][profile] = run_dir
        found.add(run_dir.name)
    missing = sorted(EXPECTED_RUNS - found)
    if missing:
        raise AuditError(f"missing runs: {missing}")
    return groups


def _transition(
    baseline: dict[str, Any], other: dict[str, Any]
) -> dict[str, Any]:
    baseline_success = baseline["success_by_seed"]
    other_success = other["success_by_seed"]
    gained = sorted(
        seed
        for seed, value in other_success.items()
        if value and not baseline_success[seed]
    )
    lost = sorted(
        seed
        for seed in baseline_success
        if baseline_success[seed] and not other_success[seed]
    )
    return {
        "gained": gained,
        "lost": lost,
        "net": len(gained) - len(lost),
        "mcnemar_exact_p": _mcnemar_exact_p(len(gained), len(lost)),
        "goal104_u050_gain_overlap": sorted(set(gained) & GOAL104_U050_GAINED),
        "gained_terminal_distance_m": {
            seed: other["final_goal_distance_by_seed"][seed] for seed in gained
        },
        "gained_c0_terminal_distance_m": {
            seed: baseline["final_goal_distance_by_seed"][seed] for seed in gained
        },
    }


def _summarize(
    groups: dict[str, dict[str, Path]]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        records[scenario] = {
            profile: _run_record(
                groups[scenario][profile],
                scenario=scenario,
                profile=profile,
            )
            for profile in PROFILES
        }
    baselines = {
        scenario: records[scenario][BASELINE_PROFILE] for scenario in SCENARIOS
    }
    summary: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        baseline = baselines[scenario]
        for profile in PROFILES:
            record = records[scenario][profile]
            path_ratio = (
                record["path_three_d_rmse_m"] / baseline["path_three_d_rmse_m"]
                if baseline["path_three_d_rmse_m"] > 0.0
                else None
            )
            summary[scenario][profile] = {
                "profile": profile,
                "far_gains": record["far_gains"],
                "near_gains": record["near_gains"],
                "near_distance_m": record["near_distance_m"],
                "far_distance_m": record["far_distance_m"],
                "success": record["success"],
                "success_pnav": record["success_pnav"],
                "success_eta_ever": record["success_eta_ever"],
                "success_eta_done": record["success_eta_done"],
                "timeout": record["timeout"],
                "safety": record["safety"],
                "path_three_d_rmse_m": record["path_three_d_rmse_m"],
                "path_ratio_vs_C0": path_ratio,
                "median_final_goal_distance_m": float(
                    np.median(list(record["final_goal_distance_by_seed"].values()))
                ),
                "median_pre_goal_3d_rmse_m": float(
                    np.median(list(record["pre_goal_3d_rmse_by_seed"].values()))
                ),
                "residual_magnitude_clip_rate": record["residual_magnitude_clip_rate"],
                "residual_rate_clip_rate": record["residual_rate_clip_rate"],
                "transition_vs_C0": _transition(baseline, record),
            }
    return summary


def _profile_gate(
    default: dict[str, Any],
    strict: dict[str, Any],
    *,
    default_success: int,
    strict_success: int,
) -> tuple[bool, dict[str, Any]]:
    path_ratios = [default["path_ratio_vs_C0"], strict["path_ratio_vs_C0"]]
    success_ok = (
        default["success"] >= default_success
        and strict["success"] >= strict_success
    )
    path_ok = all(
        ratio is not None and ratio <= PATH_RATIO_LIMIT for ratio in path_ratios
    )
    risk_ok = (
        default["timeout"] == 0
        and strict["timeout"] == 0
        and default["safety"] == 0
        and strict["safety"] == 0
    )
    record = {
        "profile": default["profile"],
        "far_gains": default["far_gains"],
        "near_gains": default["near_gains"],
        "near_distance_m": default["near_distance_m"],
        "far_distance_m": default["far_distance_m"],
        "default_success": default["success"],
        "strict125_success": strict["success"],
        "default_path_ratio_vs_C0": default["path_ratio_vs_C0"],
        "strict125_path_ratio_vs_C0": strict["path_ratio_vs_C0"],
        "strict_gained": strict["transition_vs_C0"]["gained"],
        "strict_lost": strict["transition_vs_C0"]["lost"],
        "goal104_u050_gain_overlap": strict["transition_vs_C0"][
            "goal104_u050_gain_overlap"
        ],
        "success_ok": success_ok,
        "path_ok": path_ok,
        "risk_ok": risk_ok,
    }
    return success_ok and path_ok and risk_ok, record


def _gate(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_failures: list[str] = []
    for scenario in SCENARIOS:
        actual = summary[scenario][BASELINE_PROFILE]["success"]
        expected = EXPECTED_SUCCESS[scenario]
        if actual != expected:
            baseline_failures.append(
                f"C0 {scenario}: {actual}/40 != expected {expected}/40"
            )
    if baseline_failures:
        return {
            "status": "INVALID",
            "reason": "C0_does_not_reproduce_expected_success",
            "baseline_failures": baseline_failures,
            "qualifying_profiles": [],
            "partial_profiles": [],
            "profiles": {},
        }

    qualifying: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    all_records: dict[str, dict[str, Any]] = {}
    for profile in PROFILES:
        if profile == BASELINE_PROFILE:
            continue
        passed, record = _profile_gate(
            summary["default"][profile],
            summary["strict125"][profile],
            default_success=33,
            strict_success=23,
        )
        all_records[profile] = record
        if passed:
            qualifying.append(record)
        elif record["path_ok"] and record["risk_ok"] and (
            record["default_success"] >= 33 or record["strict125_success"] >= 23
        ):
            partial.append(record)

    if qualifying:
        status = "DISTANCE_GATED_RPM_SIGNAL"
        reason = "profile_meets_default33_strict23_path_and_risk_contract"
    elif partial and max(
        len(record["goal104_u050_gain_overlap"]) for record in all_records.values()
    ) >= 2:
        status = "PARTIAL_DISTANCE_SIGNAL"
        reason = "profile_meets_one_success_threshold_with_path_and_risk_contract"
    else:
        status = "NO_SIGNAL"
        reason = (
            "no_profile_meets_distance_gated_rpm_gate"
            if not partial
            else "goal104_u050_gain_overlap_below_2"
        )
    return {
        "status": status,
        "reason": reason,
        "baseline_failures": [],
        "qualifying_profiles": qualifying,
        "partial_profiles": partial,
        "profiles": all_records,
    }


def _profile_label(row: dict[str, Any]) -> str:
    if row["near_gains"] is None:
        return "disabled"
    return f"{row['near_gains'][2]:.2f}@{row['near_distance_m']:g}-{row['far_distance_m']:g}m"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    gate = payload["gate"]
    lines = [
        "# Goal106 Phase 0：近终端 rpm 距离门控",
        "",
        "本文件由 `scripts/analyze_pc_goal106_near_terminal_rpm_schedule.py` 生成；只读 dev40 summary/trace，不训练。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- reason: `{gate['reason']}`",
        "",
    ]
    if gate.get("baseline_failures"):
        lines.extend(["## INVALID 原因", ""])
        lines.extend(f"- {failure}" for failure in gate["baseline_failures"])
        lines.append("")
    for title, key in (
        ("DISTANCE_GATED_RPM_SIGNAL profiles", "qualifying_profiles"),
        ("PARTIAL_DISTANCE_SIGNAL profiles", "partial_profiles"),
    ):
        if not gate.get(key):
            continue
        lines.extend([f"## {title}", ""])
        for row in gate[key]:
            lines.append(
                f"- {row['profile']} ({_profile_label(row)}): default "
                f"{row['default_success']}/40, strict {row['strict125_success']}/40, "
                f"gained {row['strict_gained']}, lost {row['strict_lost']}, "
                f"Goal104 overlap {row['goal104_u050_gain_overlap']}"
            )
        lines.append("")
    for scenario in SCENARIOS:
        lines.extend(
            [
                f"## {scenario}",
                "",
                "| profile | rpm near schedule | success | path ratio | final median | pre-goal median | timeout | safety | gained | lost | p |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
            ]
        )
        for profile in PROFILES:
            row = payload["summary"][scenario][profile]
            lines.append(
                f"| {profile} | {_profile_label(row)} | {row['success']}/40 | "
                f"{_fmt(row['path_ratio_vs_C0'])} | "
                f"{row['median_final_goal_distance_m']:.3f} | "
                f"{row['median_pre_goal_3d_rmse_m']:.3f} | "
                f"{row['timeout']} | {row['safety']} | "
                f"{row['transition_vs_C0']['gained']} | "
                f"{row['transition_vs_C0']['lost']} | "
                f"{row['transition_vs_C0']['mcnemar_exact_p']:.4f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profiles = {
        name: {
            "far_gains": list(spec.far_gains),
            "near_gains": None if spec.near_gains is None else list(spec.near_gains),
            "near_distance_m": spec.near_distance_m,
            "far_distance_m": spec.far_distance_m,
        }
        for name, spec in PROFILES.items()
    }
    try:
        summary = _summarize(_scan(args.root))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "arm": ARM,
            "profiles": profiles,
            "scenarios": list(SCENARIOS),
            "eval_scale": EVAL_SCALE,
            "training_scale": TRAINING_SCALE,
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "summary": summary,
            "gate": _gate(summary),
        }
    except AuditError as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "arm": ARM,
            "profiles": profiles,
            "scenarios": list(SCENARIOS),
            "eval_scale": EVAL_SCALE,
            "training_scale": TRAINING_SCALE,
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "summary": {},
            "gate": {
                "status": "INVALID",
                "reason": str(exc),
                "baseline_failures": [],
                "qualifying_profiles": [],
                "partial_profiles": [],
                "profiles": {},
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(payload, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print(f"gate={payload['gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
