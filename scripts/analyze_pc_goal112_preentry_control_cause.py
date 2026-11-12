#!/usr/bin/env python3
"""Audit whether pre-entry geometry is reference-, tracking-, or offset-limited.

Goal112 is read-only.  It reuses the frozen Goal111 pre-entry classifier and
Goal107 traces, starts no rollout, and never accesses held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from analyze_pc_goal111_preentry_closest_approach import (
    DEV_SEEDS,
    EXPECTED_CHECKPOINT_SHA,
    EXPECTED_EVAL_SCALE,
    EXPECTED_GAINS,
    SHELLS_M,
    AuditError,
    _episode_metrics as _goal111_episode_metrics,
    _first_entry,
    _fmt,
    _load_episodes,
    _read_json,
    _trace_by_seed,
)

SCHEMA_VERSION = "auv_pc_goal112_preentry_control_cause.v1"
ARM = "P_D3"
SCENARIOS = ("default", "strict125")
PROFILES = ("C0", "U050")
PRIMARY_SHELL_M = 2.0
ONSET_SHELLS_M = (3.0, 2.5, 2.0)
WINDOW_STEPS = 5
HEADING_THRESHOLD_RAD = 0.35
PITCH_THRESHOLD_RAD = 0.45
MAJORITY_FRACTION = 0.50
EXPECTED_GOAL111_COUNTS = {
    "APPROACH_DEFICIT": 7,
    "PREENTRY_GEOMETRY_BAD": 11,
    "ENTRY_GEOMETRY_OK_LATE_FAILURE": 2,
}
EXPECTED_PRIMARY_N = 11
CAUSES = (
    "REFERENCE_BAD_ONLY",
    "TRACKING_BAD_ONLY",
    "MIXED_REFERENCE_AND_TRACKING",
    "ALIGNED_CONTROL_POSITION_OFFSET",
)
REQUIRED_CONTROL_FIELDS = (
    "time_s",
    "goal_distance_eta_m",
    "eta",
    "goal_ned_m",
    "command",
    "ground_speed_mps",
    "heading_error_rad",
    "pitch_error_rad",
    "actuator_residual_effective",
    "material_saturated",
    "guidance_terminal_phase",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal107_terminal_reward_credit_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal112_preentry_control_cause_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal112_preentry_control_cause_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def _circular_mean(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return math.nan
    return float(math.atan2(float(np.mean(np.sin(finite))), float(np.mean(np.cos(finite)))))


def _window_circular_mean(values: np.ndarray, index: int, steps: int = WINDOW_STEPS) -> float:
    start = max(0, index - steps + 1)
    return _circular_mean(values[start : index + 1])


def _window_median(values: np.ndarray, index: int, steps: int = WINDOW_STEPS) -> float:
    start = max(0, index - steps + 1)
    selected = np.asarray(values[start : index + 1], dtype=np.float64)
    finite = selected[np.isfinite(selected)]
    return float(np.median(finite)) if finite.size else math.nan


def _los_angles(eta: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(goal, dtype=np.float64)[:, :3] - np.asarray(eta, dtype=np.float64)[:, :3]
    horizontal = np.hypot(delta[:, 0], delta[:, 1])
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    pitch = np.arctan2(-delta[:, 2], np.maximum(horizontal, 1e-12))
    return heading, pitch


def _classify_control(
    *,
    command_los_heading_rad: float,
    command_los_pitch_rad: float,
    tracking_heading_rad: float,
    tracking_pitch_rad: float,
) -> dict[str, Any]:
    reference_bad = bool(
        abs(command_los_heading_rad) >= HEADING_THRESHOLD_RAD
        or abs(command_los_pitch_rad) >= PITCH_THRESHOLD_RAD
    )
    tracking_bad = bool(
        abs(tracking_heading_rad) >= HEADING_THRESHOLD_RAD
        or abs(tracking_pitch_rad) >= PITCH_THRESHOLD_RAD
    )
    if reference_bad and tracking_bad:
        cause = "MIXED_REFERENCE_AND_TRACKING"
    elif reference_bad:
        cause = "REFERENCE_BAD_ONLY"
    elif tracking_bad:
        cause = "TRACKING_BAD_ONLY"
    else:
        cause = "ALIGNED_CONTROL_POSITION_OFFSET"
    return {
        "reference_bad": reference_bad,
        "tracking_bad": tracking_bad,
        "cause": cause,
    }


def _validate_arrays(trace: dict[str, np.ndarray], *, seed: int) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_CONTROL_FIELDS if name not in trace]
    if missing:
        raise AuditError(f"seed={seed}: trace is missing control fields {missing}")
    arrays = {name: np.asarray(trace[name]) for name in REQUIRED_CONTROL_FIELDS}
    time_s = arrays["time_s"].astype(np.float64, copy=False)
    frames = time_s.size
    if frames < 2:
        raise AuditError(f"seed={seed}: trace is too short")
    scalar_names = (
        "goal_distance_eta_m",
        "ground_speed_mps",
        "heading_error_rad",
        "pitch_error_rad",
        "material_saturated",
        "guidance_terminal_phase",
    )
    for name in scalar_names:
        if arrays[name].shape != (frames,):
            raise AuditError(f"seed={seed}: {name} must have shape ({frames},)")
    if arrays["eta"].shape != (frames, 6) or arrays["goal_ned_m"].shape != (frames, 3):
        raise AuditError(f"seed={seed}: eta/goal shapes are invalid")
    if arrays["command"].shape != (frames, 3):
        raise AuditError(f"seed={seed}: command must have shape ({frames}, 3)")
    if arrays["actuator_residual_effective"].shape != (frames, 3):
        raise AuditError(f"seed={seed}: residual must have shape ({frames}, 3)")
    for name in (
        "goal_distance_eta_m",
        "ground_speed_mps",
        "heading_error_rad",
        "pitch_error_rad",
        "eta",
        "goal_ned_m",
        "command",
        "actuator_residual_effective",
    ):
        if not np.all(np.isfinite(arrays[name])):
            raise AuditError(f"seed={seed}: {name} contains non-finite values")
    return arrays


def _control_at_shell(
    arrays: dict[str, np.ndarray],
    *,
    shell_m: float,
    seed: int,
) -> dict[str, Any]:
    distance = arrays["goal_distance_eta_m"].astype(np.float64, copy=False)
    entry = _first_entry(distance, shell_m)
    if entry is None:
        return {
            "entered": False,
            "entry_index": None,
            "entry_time_s": None,
            "command_los_heading_rad": None,
            "command_los_pitch_rad": None,
            "actual_los_heading_rad": None,
            "actual_los_pitch_rad": None,
            "tracking_heading_rad": None,
            "tracking_pitch_rad": None,
            "trace_tracking_heading_rad": None,
            "trace_tracking_pitch_rad": None,
            "heading_consistency_max_abs_diff_rad": None,
            "pitch_consistency_max_abs_diff_rad": None,
            "guidance_terminal_phase_fraction": None,
            "speed_tracking_mps": None,
            "residual_l2": None,
            "material_saturation_fraction": None,
            "reference_bad": None,
            "tracking_bad": None,
            "cause": None,
        }

    time_s = arrays["time_s"].astype(np.float64, copy=False)
    eta = arrays["eta"].astype(np.float64, copy=False)
    goal = arrays["goal_ned_m"].astype(np.float64, copy=False)
    command = arrays["command"].astype(np.float64, copy=False)
    los_heading, los_pitch = _los_angles(eta, goal)

    command_heading = command[:, 0]
    command_pitch = command[:, 1]
    actual_heading = eta[:, 5]
    actual_pitch = eta[:, 4]
    command_los_heading = _wrap_angle(command_heading - los_heading)
    command_los_pitch = command_pitch - los_pitch
    actual_los_heading = _wrap_angle(actual_heading - los_heading)
    actual_los_pitch = actual_pitch - los_pitch
    tracking_heading = _wrap_angle(command_heading - actual_heading)
    tracking_pitch = command_pitch - actual_pitch

    start = max(0, entry - WINDOW_STEPS + 1)
    stop = entry + 1
    heading_diff = np.abs(
        _wrap_angle(
            tracking_heading[start:stop]
            - arrays["heading_error_rad"].astype(np.float64, copy=False)[start:stop]
        )
    )
    pitch_diff = np.abs(
        tracking_pitch[start:stop]
        - arrays["pitch_error_rad"].astype(np.float64, copy=False)[start:stop]
    )
    heading_consistency = float(np.max(heading_diff)) if heading_diff.size else math.nan
    pitch_consistency = float(np.max(pitch_diff)) if pitch_diff.size else math.nan
    if heading_consistency > 1e-5 or pitch_consistency > 1e-5:
        raise AuditError(
            f"seed={seed}, shell={shell_m}: trace tracking errors disagree with command/eta "
            f"(heading={heading_consistency}, pitch={pitch_consistency})"
        )

    command_los_heading_value = _window_circular_mean(command_los_heading, entry)
    command_los_pitch_value = _window_median(command_los_pitch, entry)
    actual_los_heading_value = _window_circular_mean(actual_los_heading, entry)
    actual_los_pitch_value = _window_median(actual_los_pitch, entry)
    tracking_heading_value = _window_circular_mean(tracking_heading, entry)
    tracking_pitch_value = _window_median(tracking_pitch, entry)
    classification = _classify_control(
        command_los_heading_rad=command_los_heading_value,
        command_los_pitch_rad=command_los_pitch_value,
        tracking_heading_rad=tracking_heading_value,
        tracking_pitch_rad=tracking_pitch_value,
    )
    residual_norm = np.linalg.norm(
        arrays["actuator_residual_effective"].astype(np.float64, copy=False), axis=1
    )
    speed_tracking = command[:, 2] - arrays["ground_speed_mps"].astype(np.float64, copy=False)
    terminal_phase = arrays["guidance_terminal_phase"].astype(np.float64, copy=False)
    saturation = arrays["material_saturated"].astype(bool, copy=False)

    return {
        "entered": True,
        "entry_index": int(entry),
        "entry_time_s": float(time_s[entry]),
        "command_los_heading_rad": command_los_heading_value,
        "command_los_pitch_rad": command_los_pitch_value,
        "actual_los_heading_rad": actual_los_heading_value,
        "actual_los_pitch_rad": actual_los_pitch_value,
        "tracking_heading_rad": tracking_heading_value,
        "tracking_pitch_rad": tracking_pitch_value,
        "trace_tracking_heading_rad": _window_circular_mean(
            arrays["heading_error_rad"].astype(np.float64, copy=False), entry
        ),
        "trace_tracking_pitch_rad": _window_median(
            arrays["pitch_error_rad"].astype(np.float64, copy=False), entry
        ),
        "heading_consistency_max_abs_diff_rad": heading_consistency,
        "pitch_consistency_max_abs_diff_rad": pitch_consistency,
        "guidance_terminal_phase_fraction": float(np.mean(terminal_phase[start:stop] > 0.5)),
        "speed_tracking_mps": _window_median(speed_tracking, entry),
        "residual_l2": _window_median(residual_norm, entry),
        "material_saturation_fraction": float(np.mean(saturation[start:stop])),
        **classification,
    }


def _episode_control_metrics(trace: dict[str, np.ndarray], *, seed: int) -> dict[str, Any]:
    arrays = _validate_arrays(trace, seed=seed)
    shells = {
        f"{shell:.1f}": _control_at_shell(arrays, shell_m=shell, seed=seed)
        for shell in SHELLS_M
    }
    bad_shells = [
        shell
        for shell in ONSET_SHELLS_M
        if shells[f"{shell:.1f}"]["cause"] not in (None, "ALIGNED_CONTROL_POSITION_OFFSET")
    ]
    return {
        "seed": seed,
        "shells": shells,
        "earliest_bad_shell_m": bad_shells[0] if bad_shells else None,
    }


def _cause_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    counts = {
        cause: sum(1 for row in records if row["primary"]["cause"] == cause)
        for cause in CAUSES
    }
    if sum(counts.values()) != count:
        raise AuditError("control-cause classes do not partition the primary subset")
    fractions = {cause: counts[cause] / count if count else math.nan for cause in CAUSES}
    reference_involved = counts["REFERENCE_BAD_ONLY"] + counts["MIXED_REFERENCE_AND_TRACKING"]
    tracking_involved = counts["TRACKING_BAD_ONLY"] + counts["MIXED_REFERENCE_AND_TRACKING"]
    return {
        "n": count,
        "counts": counts,
        "fractions": fractions,
        "reference_involved_n": reference_involved,
        "reference_involved_fraction": reference_involved / count if count else math.nan,
        "tracking_involved_n": tracking_involved,
        "tracking_involved_fraction": tracking_involved / count if count else math.nan,
        "earliest_bad_shell_counts": {
            f"{shell:.1f}": sum(
                1 for row in records if row["earliest_bad_shell_m"] == shell
            )
            for shell in ONSET_SHELLS_M
        },
        "never_bad_shell_n": sum(
            1 for row in records if row["earliest_bad_shell_m"] is None
        ),
    }


def _classify_gate(primary_summary: dict[str, Any]) -> dict[str, Any]:
    n = int(primary_summary["n"])
    if n != EXPECTED_PRIMARY_N:
        raise AuditError(f"primary subset must contain {EXPECTED_PRIMARY_N} failures, got {n}")
    reference_fraction = float(primary_summary["reference_involved_fraction"])
    tracking_fraction = float(primary_summary["tracking_involved_fraction"])
    h112_a = reference_fraction >= MAJORITY_FRACTION
    h112_b = tracking_fraction >= MAJORITY_FRACTION
    counts = primary_summary["counts"]
    aligned_count = int(counts["ALIGNED_CONTROL_POSITION_OFFSET"])
    max_other = max(
        int(counts[cause]) for cause in CAUSES if cause != "ALIGNED_CONTROL_POSITION_OFFSET"
    )
    aligned_is_unique_max = aligned_count > max_other
    if h112_a and h112_b:
        status = "JOINT_REFERENCE_AND_TRACKING"
    elif h112_a:
        status = "GUIDANCE_REFERENCE_LIMITED"
    elif h112_b:
        status = "TRACKING_LIMITED"
    elif aligned_is_unique_max:
        status = "ALIGNED_CONTROL_POSITION_OFFSET"
    else:
        status = "AMBIGUOUS_HANDOFF"
    return {
        "status": status,
        "reason": "frozen_50pct_reference_tracking_gate",
        "primary_scenario": "strict125",
        "primary_profile": "C0",
        "primary_shell_m": PRIMARY_SHELL_M,
        "primary_n": n,
        "heading_threshold_rad": HEADING_THRESHOLD_RAD,
        "pitch_threshold_rad": PITCH_THRESHOLD_RAD,
        "majority_fraction": MAJORITY_FRACTION,
        "counts": counts,
        "fractions": primary_summary["fractions"],
        "reference_involved_n": primary_summary["reference_involved_n"],
        "reference_involved_fraction": reference_fraction,
        "tracking_involved_n": primary_summary["tracking_involved_n"],
        "tracking_involved_fraction": tracking_fraction,
        "aligned_control_unique_max": aligned_is_unique_max,
        "H112_A_supported": h112_a,
        "H112_B_supported": h112_b,
        "H112_C_supported": status == "ALIGNED_CONTROL_POSITION_OFFSET",
        "H112_D_supported": status == "JOINT_REFERENCE_AND_TRACKING",
    }


def _profile_summary(run_dir: Path, *, scenario: str, profile: str) -> dict[str, Any]:
    episodes = _load_episodes(run_dir, scenario=scenario, profile=profile)
    traces = _trace_by_seed(run_dir)
    if tuple(sorted(traces)) != DEV_SEEDS:
        raise AuditError(f"{run_dir}: trace seeds must be exactly {DEV_SEEDS}")
    records: list[dict[str, Any]] = []
    for seed in DEV_SEEDS:
        success = bool(episodes[seed]["success"])
        goal111 = _goal111_episode_metrics(traces[seed], seed=seed, success=success)
        control = _episode_control_metrics(traces[seed], seed=seed)
        records.append(
            {
                "seed": seed,
                "success": success,
                "goal111_classification": goal111["classification"],
                "primary": control["shells"][f"{PRIMARY_SHELL_M:.1f}"],
                "shells": control["shells"],
                "earliest_bad_shell_m": control["earliest_bad_shell_m"],
            }
        )
    failures = [row for row in records if not row["success"]]
    goal111_counts = {
        name: sum(1 for row in failures if row["goal111_classification"] == name)
        for name in EXPECTED_GOAL111_COUNTS
    }
    if scenario == "strict125" and profile == "C0":
        if goal111_counts != EXPECTED_GOAL111_COUNTS:
            raise AuditError(
                f"strict C0 Goal111 counts changed: {goal111_counts} != {EXPECTED_GOAL111_COUNTS}"
            )
    primary_records = [
        row for row in failures if row["goal111_classification"] == "PREENTRY_GEOMETRY_BAD"
    ]
    primary_summary = _cause_summary(primary_records)
    return {
        "failure_n": len(failures),
        "goal111_counts": goal111_counts,
        "primary_subset_n": len(primary_records),
        "primary_summary": primary_summary,
        "records": records,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    primary = payload["summary"]["strict125"]["C0"]
    summary = primary["primary_summary"]
    lines = [
        "# Goal112 Phase 0：pre-entry 几何的 control-cause 审计",
        "",
        "本文件由 `scripts/analyze_pc_goal112_preentry_control_cause.py` 生成；",
        "只读 Goal107 dev40 trace，不训练、不改 reward/terminal/success、不打开 held-out。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- primary n: `{gate['primary_n']}`（Goal111 `PREENTRY_GEOMETRY_BAD` failures）",
        f"- reference involved: `{gate['reference_involved_n']}/{gate['primary_n']}` = `{_fmt(gate['reference_involved_fraction'])}`",
        f"- tracking involved: `{gate['tracking_involved_n']}/{gate['primary_n']}` = `{_fmt(gate['tracking_involved_fraction'])}`",
        "",
        "## 互斥分类",
        "",
        "| cause | count | fraction |",
        "|---|---:|---:|",
    ]
    for cause in CAUSES:
        lines.append(
            f"| `{cause}` | {gate['counts'][cause]} | {_fmt(gate['fractions'][cause])} |"
        )
    lines.extend(
        [
            "",
            "## entry 2 m 的 command/actual 指标",
            "",
            "| group | command-LOS heading | command-LOS pitch | tracking heading | tracking pitch | speed tracking |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    records = primary["records"]
    primary_records = [
        row for row in records if not row["success"] and row["goal111_classification"] == "PREENTRY_GEOMETRY_BAD"
    ]
    for cause in CAUSES:
        subset = [row for row in primary_records if row["primary"]["cause"] == cause]
        if not subset:
            continue
        values = {
            "command_los_heading_rad": [row["primary"]["command_los_heading_rad"] for row in subset],
            "command_los_pitch_rad": [row["primary"]["command_los_pitch_rad"] for row in subset],
            "tracking_heading_rad": [row["primary"]["tracking_heading_rad"] for row in subset],
            "tracking_pitch_rad": [row["primary"]["tracking_pitch_rad"] for row in subset],
            "speed_tracking_mps": [row["primary"]["speed_tracking_mps"] for row in subset],
        }
        lines.append(
            f"| `{cause}` | {_fmt(np.median(values['command_los_heading_rad']))} | "
            f"{_fmt(np.median(values['command_los_pitch_rad']))} | "
            f"{_fmt(np.median(values['tracking_heading_rad']))} | "
            f"{_fmt(np.median(values['tracking_pitch_rad']))} | "
            f"{_fmt(np.median(values['speed_tracking_mps']))} |"
        )
    lines.extend(
        [
            "",
            "## onset shell",
            "",
            "| shell (m) | earliest bad count |",
            "|---|---:|",
        ]
    )
    for shell in ONSET_SHELLS_M:
        lines.append(
            f"| {shell:.1f} | {summary['earliest_bad_shell_counts'][f'{shell:.1f}']} |"
        )
    lines.extend(
        [
            f"| never | {summary['never_bad_shell_n']} |",
            "",
            "## 全部分组",
            "",
            "| scenario | profile | failure | Goal111 pre-entry | Goal112 primary n |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        for profile in PROFILES:
            group = payload["summary"][scenario][profile]
            lines.append(
                f"| {scenario} | {profile} | {group['failure_n']} | "
                f"{group['goal111_counts']['PREENTRY_GEOMETRY_BAD']} | {group['primary_subset_n']} |"
            )
    lines.extend(
        [
            "",
            "## 判读边界",
            "",
            "- 主 Gate 只分析 strict C0 的 Goal111 `PREENTRY_GEOMETRY_BAD` 子集；",
            "- `reference` 使用 command heading/pitch 对 direct goal LOS；",
            "- `tracking` 使用 command 对实际 eta yaw/pitch；",
            "- 差值为 entry 帧及前 4 帧的 circular mean / median；",
            "- 3.0/2.5/2.0 m 只作 onset 描述，不改变主分类；",
            "- 相关性不写成因果；held-out `100–149` 未打开。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for profile in PROFILES:
            run_dir = args.root / f"{ARM}_{scenario}_{profile}"
            summary[scenario][profile] = _profile_summary(
                run_dir,
                scenario=scenario,
                profile=profile,
            )
    gate = _classify_gate(summary["strict125"]["C0"]["primary_summary"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "arm": ARM,
        "scenarios": list(SCENARIOS),
        "profiles": list(PROFILES),
        "shells_m": list(SHELLS_M),
        "onset_shells_m": list(ONSET_SHELLS_M),
        "window_steps": WINDOW_STEPS,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "eval_scale": EXPECTED_EVAL_SCALE,
        "expected_channel_gains": {name: list(values) for name, values in EXPECTED_GAINS.items()},
        "gate": gate,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
