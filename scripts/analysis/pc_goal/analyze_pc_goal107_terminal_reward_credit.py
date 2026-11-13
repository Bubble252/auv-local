#!/usr/bin/env python3
"""Audit Goal107's terminal reward, geometry, and action credit."""

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
    _read_json,
    _success_by_seed,
    _validate_trace,
)

SCHEMA_VERSION = "auv_pc_goal107_terminal_reward_credit.v1"
SCENARIOS = ("default", "strict125")
PROFILES: dict[str, tuple[float, float, float]] = {
    "C0": (1.00, 1.00, 1.00),
    "U050": (1.00, 1.00, 0.50),
}
EXPECTED_SUCCESS = {"default": 33, "strict125": 20}
EXPECTED_RUNS = {
    f"{ARM}_{scenario}_{profile}"
    for scenario in SCENARIOS
    for profile in PROFILES
}
REWARD_PREFIX = "reward_"
KEY_REWARD_METRICS = (
    "reward_terminal",
    "reward_terminal_progress",
    "reward_goal_plane_miss",
)
PATH_SPEED_ACTION_METRICS = (
    "reward_path_cross",
    "reward_path_vertical",
    "reward_path_progress",
    "reward_path_heading",
    "reward_route_speed",
    "reward_speed_smooth",
    "reward_energy",
    "reward_action",
    "reward_saturation",
)
GEOMETRY_METRICS = (
    "final_goal_distance_m",
    "terminal_horizontal_miss_m",
    "terminal_vertical_miss_m",
    "terminal_abs_heading_error_rad",
    "terminal_abs_pitch_error_rad",
    "last2s_closure_mean_mps",
    "last2s_closure_last_mps",
)
EFFECT_THRESHOLD = 0.33


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
            "artifacts/lowlevel/goal107_terminal_reward_credit_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal107_terminal_reward_credit_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _finite_median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else math.nan


def _cliffs_delta(success: Sequence[float], failure: Sequence[float]) -> float:
    a = np.asarray(
        [float(value) for value in success if math.isfinite(float(value))],
        dtype=np.float64,
    )
    b = np.asarray(
        [float(value) for value in failure if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if a.size == 0 or b.size == 0:
        return math.nan
    greater = float(np.sum(a[:, None] > b[None, :]))
    less = float(np.sum(a[:, None] < b[None, :]))
    return (greater - less) / float(a.size * b.size)


def _trace_episodes(path: Path) -> dict[int, dict[str, np.ndarray]]:
    _validate_trace(path)
    grouped: dict[int, dict[str, list[np.ndarray]]] = {}
    with np.load(path) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
        fields = list(arrays)
        seeds = np.asarray(arrays["episode_seed"], dtype=np.int64)
        for index, seed in enumerate(seeds.tolist()):
            row = grouped.setdefault(int(seed), {name: [] for name in fields})
            for name in fields:
                row[name].append(arrays[name][index])
    return {
        seed: {name: np.asarray(values) for name, values in rows.items()}
        for seed, rows in grouped.items()
    }


def _window_sum(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[mask]
    if selected.size == 0:
        return math.nan
    return float(np.sum(selected))


def _reward_share(values: dict[str, float], keys: tuple[str, ...]) -> float:
    numerator = sum(abs(values.get(key, 0.0)) for key in keys)
    denominator = sum(
        abs(value) for key, value in values.items() if key.startswith(REWARD_PREFIX)
    )
    return numerator / denominator if denominator > 0.0 else math.nan


def _episode_record(
    trace: dict[str, np.ndarray], *, seed: int, success: bool
) -> dict[str, Any]:
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)
    eta = np.asarray(trace["eta"], dtype=np.float64)
    goal = np.asarray(trace["goal_ned_m"], dtype=np.float64)
    if time_s.size < 2 or distance.size != time_s.size:
        raise AuditError(f"seed={seed}: trace is too short or misaligned")
    last2s = time_s >= float(time_s[-1]) - 2.0
    last2m = distance <= 2.0
    if not bool(last2s.any()):
        raise AuditError(f"seed={seed}: empty last-2s window")

    reward_last2s: dict[str, float] = {}
    for name in trace:
        if name.startswith(REWARD_PREFIX):
            reward_last2s[name] = _window_sum(trace[name], last2s)

    closure_all = -np.diff(distance) / np.maximum(np.diff(time_s), 1e-9)
    closure_mask = last2s[:-1]
    closure_last2s = closure_all[closure_mask]
    terminal_error = goal[-1, :3] - eta[-1, :3]
    terminal_horizontal = float(np.linalg.norm(terminal_error[:2]))
    terminal_vertical = float(abs(terminal_error[2]))
    terminal_distance = float(distance[-1])

    return {
        "seed": seed,
        "success": success,
        "last2s_steps": int(last2s.sum()),
        "last2m_steps": int(last2m.sum()) if bool(last2m.any()) else 0,
        "final_goal_distance_m": terminal_distance,
        "terminal_horizontal_miss_m": terminal_horizontal,
        "terminal_vertical_miss_m": terminal_vertical,
        "terminal_abs_heading_error_rad": float(
            abs(np.asarray(trace["heading_error_rad"], dtype=np.float64)[-1])
        ),
        "terminal_abs_pitch_error_rad": float(
            abs(np.asarray(trace["pitch_error_rad"], dtype=np.float64)[-1])
        ),
        "last2s_closure_mean_mps": float(np.nanmean(closure_last2s)),
        "last2s_closure_last_mps": float(closure_last2s[-1]),
        "last2s_closure_negative_fraction": float(
            np.mean(closure_last2s < 0.0)
        ),
        "reward_last2s": reward_last2s,
        "terminal_reward_share_last2s": _reward_share(
            reward_last2s, KEY_REWARD_METRICS
        ),
        "path_speed_action_share_last2s": _reward_share(
            reward_last2s, PATH_SPEED_ACTION_METRICS
        ),
        "last2s_rpm_base_mean": float(
            np.mean(np.asarray(trace["actuator_base_physical"])[last2s, 2])
        ),
        "last2s_rpm_effective_mean": float(
            np.mean(np.asarray(trace["actuator_residual_effective"])[last2s, 2])
        ),
        "last2s_rpm_applied_abs_mean": float(
            np.mean(np.abs(np.asarray(trace["actuator_residual_applied"])[last2s, 2]))
        ),
        "last2s_applied_norm_mean": float(
            np.mean(
                np.linalg.norm(
                    np.asarray(trace["actuator_residual_applied"])[last2s], axis=1
                )
            )
        ),
    }


def _run_record(run_dir: Path, *, scenario: str, profile: str) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise AuditError(f"{summary_path}: checkpoint SHA mismatch")
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - EVAL_SCALE) > 1e-12:
        raise AuditError(f"{summary_path}: scale override mismatch")
    recorded_gains = summary.get("actuator_residual_channel_gains")
    expected_gains = PROFILES[profile]
    if (
        not isinstance(recorded_gains, list)
        or len(recorded_gains) != 3
        or any(
            abs(float(actual) - expected) > 1e-12
            for actual, expected in zip(recorded_gains, expected_gains, strict=True)
        )
    ):
        raise AuditError(f"{summary_path}: channel gains mismatch")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise AuditError(f"{summary_path}: expected 40 dev episodes")
    by_seed = _success_by_seed(episodes, summary_path)
    trace = _trace_episodes(run_dir / "trace.npz")
    if set(trace) != EXPECTED_SEEDS:
        raise AuditError(f"{summary_path}: trace seeds do not cover 20-59")

    records = [
        _episode_record(trace[seed], seed=seed, success=by_seed[seed])
        for seed in sorted(EXPECTED_SEEDS)
    ]
    if not any(px.startswith(REWARD_PREFIX) for px in next(iter(trace.values()))):
        raise AuditError(f"{run_dir}: missing reward breakdown")
    return {
        "run": run_dir.name,
        "scenario": scenario,
        "profile": profile,
        "success": sum(by_seed.values()),
        "timeout": sum(bool(row.get("timeout", False)) for row in episodes),
        "safety": sum(bool(row.get("safety", False)) for row in episodes),
        "records": records,
    }


def _scan(root: Path) -> dict[str, dict[str, Path]]:
    if not root.is_dir():
        raise AuditError(f"{root}: missing terminal-credit root")
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


def _group_stats(
    records: list[dict[str, Any]], metric: str
) -> dict[str, float]:
    success_records = [row for row in records if row["success"]]
    failure_records = [row for row in records if not row["success"]]
    success_values = [float(row[metric]) for row in success_records]
    failure_values = [float(row[metric]) for row in failure_records]
    success_median = _finite_median(success_values)
    failure_median = _finite_median(failure_values)
    delta = _cliffs_delta(success_values, failure_values)
    return {
        "success_n": len(success_values),
        "failure_n": len(failure_values),
        "success_median": success_median,
        "failure_median": failure_median,
        "median_difference": success_median - failure_median,
        "cliffs_delta": delta,
    }


def _reward_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = sorted(
        {
            name
            for row in records
            for name in row["reward_last2s"]
            if name.startswith(REWARD_PREFIX)
        }
    )
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        success_values = [
            row["reward_last2s"].get(key, math.nan)
            for row in records
            if row["success"]
        ]
        failure_values = [
            row["reward_last2s"].get(key, math.nan)
            for row in records
            if not row["success"]
        ]
        stats[key] = {
            "success_median": _finite_median(success_values),
            "failure_median": _finite_median(failure_values),
            "cliffs_delta": _cliffs_delta(success_values, failure_values),
        }
    return stats


def _summarize(
    groups: dict[str, dict[str, Path]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        result[scenario] = {}
        for profile in PROFILES:
            record = _run_record(
                groups[scenario][profile], scenario=scenario, profile=profile
            )
            records = record["records"]
            result[scenario][profile] = {
                "success": record["success"],
                "timeout": record["timeout"],
                "safety": record["safety"],
                "reward": _reward_stats(records),
                "terminal_reward_share_last2s": _group_stats(
                    records, "terminal_reward_share_last2s"
                ),
                "path_speed_action_share_last2s": _group_stats(
                    records, "path_speed_action_share_last2s"
                ),
                "geometry": {
                    metric: _group_stats(records, metric)
                    for metric in GEOMETRY_METRICS
                },
                "action": {
                    metric: _group_stats(records, metric)
                    for metric in (
                        "last2s_rpm_base_mean",
                        "last2s_rpm_effective_mean",
                        "last2s_rpm_applied_abs_mean",
                        "last2s_applied_norm_mean",
                    )
                },
                "records": records,
            }
    return result


def _gate(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    for scenario in SCENARIOS:
        actual = summary[scenario]["C0"]["success"]
        expected = EXPECTED_SUCCESS[scenario]
        if actual != expected:
            failures.append(f"C0 {scenario}: {actual}/40 != {expected}/40")
    if failures:
        return {
            "status": "INVALID",
            "reason": "C0_does_not_reproduce_expected_success",
            "baseline_failures": failures,
        }

    primary = summary["strict125"]["C0"]
    reward_deltas = {
        name: abs(float(primary["reward"].get(name, {}).get("cliffs_delta", math.nan)))
        for name in KEY_REWARD_METRICS
    }
    reward_weak = max(
        [value for value in reward_deltas.values() if math.isfinite(value)] or [0.0]
    ) < EFFECT_THRESHOLD
    geometry_deltas = {
        name: abs(float(stats["cliffs_delta"]))
        for name, stats in primary["geometry"].items()
    }
    strong_geometry = [
        name
        for name, value in geometry_deltas.items()
        if math.isfinite(value) and value >= EFFECT_THRESHOLD
    ]
    terminal_share = float(primary["terminal_reward_share_last2s"]["failure_median"])
    drain_share = float(primary["path_speed_action_share_last2s"]["failure_median"])
    drain_dominates = drain_share > terminal_share

    if reward_weak and drain_dominates:
        status = "TERMINAL_REWARD_UNDERWEIGHTED"
        reason = "terminal_reward_separation_is_weak_and_path_speed_action_dominates"
    elif reward_weak and len(strong_geometry) >= 2:
        status = "TERMINAL_GEOMETRY_LIMITED"
        reason = "reward_separation_is_weak_but_geometry_separates_success_failure"
    elif not reward_weak and len(strong_geometry) >= 2:
        status = "CREDIT_OR_OPTIMIZATION_LIMITED"
        reason = "terminal_reward_and_geometry_BOTH_separate_but_C0_U050_gap_persists"
    else:
        status = "INCONCLUSIVE"
        reason = "neither_reward_nor_geometry_reaches_prespecified_separation_rule"
    return {
        "status": status,
        "reason": reason,
        "baseline_failures": [],
        "reward_weak": reward_weak,
        "geometry_strong_features": strong_geometry,
        "reward_cliffs_delta": reward_deltas,
        "geometry_cliffs_delta": geometry_deltas,
        "failure_terminal_reward_share": terminal_share,
        "failure_path_speed_action_share": drain_share,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    gate = payload["gate"]
    lines = [
        "# Goal107 Phase 0：terminal reward / geometry / action 审计",
        "",
        "本文件由 `scripts/analyze_pc_goal107_terminal_reward_credit.py` 生成；只读 dev40，不训练。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- reason: `{gate['reason']}`",
        "",
    ]
    if gate.get("baseline_failures"):
        lines.extend(["## INVALID 原因", ""])
        lines.extend(f"- {failure}" for failure in gate["baseline_failures"])
        lines.append("")
    else:
        lines.extend(
            [
                "## 判读摘要",
                "",
                f"- reward weak: `{gate['reward_weak']}`",
                f"- geometry strong features: `{gate['geometry_strong_features']}`",
                f"- failure terminal reward share: `{_fmt(gate['failure_terminal_reward_share'])}`",
                f"- failure path/speed/action share: `{_fmt(gate['failure_path_speed_action_share'])}`",
                "",
            ]
        )
    for scenario in SCENARIOS:
        lines.extend([f"## {scenario}", ""])
        lines.append("| profile | success | timeout | safety |")
        lines.append("|---|---:|---:|---:|")
        for profile in PROFILES:
            row = payload["summary"][scenario][profile]
            lines.append(
                f"| {profile} | {row['success']}/40 | {row['timeout']} | {row['safety']} |"
            )
        lines.extend(
            [
                "",
                "| reward (last 2 s) | success median | failure median | Cliff delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, stats in payload["summary"][scenario]["C0"]["reward"].items():
            lines.append(
                f"| {name} | {_fmt(stats['success_median'])} | "
                f"{_fmt(stats['failure_median'])} | {_fmt(stats['cliffs_delta'])} |"
            )
        lines.extend(
            [
                "",
                "| geometry | success median | failure median | Cliff delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, stats in payload["summary"][scenario]["C0"]["geometry"].items():
            lines.append(
                f"| {name} | {_fmt(stats['success_median'])} | "
                f"{_fmt(stats['failure_median'])} | {_fmt(stats['cliffs_delta'])} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = _summarize(_scan(args.root))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "arm": ARM,
            "profiles": {name: list(gains) for name, gains in PROFILES.items()},
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
            "profiles": {name: list(gains) for name, gains in PROFILES.items()},
            "scenarios": list(SCENARIOS),
            "eval_scale": EVAL_SCALE,
            "training_scale": TRAINING_SCALE,
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "summary": {},
            "gate": {
                "status": "INVALID",
                "reason": str(exc),
                "baseline_failures": [],
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
