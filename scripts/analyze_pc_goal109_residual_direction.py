#!/usr/bin/env python3
"""Audit Goal109 residual direction, magnitude, and terminal capture.

The script reads only the frozen Goal107 dev40 traces.  It does not load a
policy, train, modify reward/terminal/success, or read held-out seeds.
"""

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
)
from analyze_pc_goal107_terminal_reward_credit import (
    _cliffs_delta,
    _finite_median,
    _trace_episodes,
)

SCHEMA_VERSION = "auv_pc_goal109_residual_direction_terminal_capture.v1"
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
DIRECTION_METRICS = (
    "applied_base_cosine_mean",
    "applied_base_cosine_p05",
    "rudder_heading_sign_alignment",
    "stern_pitch_sign_alignment",
    "rpm_speed_sign_alignment",
)
MAGNITUDE_METRICS = (
    "residual_norm_mean",
    "residual_norm_p95",
    "material_saturation_fraction",
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
SHELL_THRESHOLDS = (2.0, 1.5, 1.25, 1.0)
EFFECT_THRESHOLD = 0.33
REQUIRED_TRACE_FIELDS = (
    "time_s",
    "episode_seed",
    "goal_distance_eta_m",
    "eta",
    "goal_ned_m",
    "heading_error_rad",
    "pitch_error_rad",
    "speed_error_mps",
    "actuator_residual_applied",
    "actuator_base_physical",
    "material_saturated",
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
            "artifacts/lowlevel/goal109_residual_direction_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal109_residual_direction_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _finite_values(values: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )


def _finite_quantile(values: Sequence[float], quantile: float) -> float:
    finite = _finite_values(values)
    return float(np.quantile(finite, quantile)) if finite.size else math.nan


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    return {
        f"p{int(round(quantile * 100)):02d}": _finite_quantile(values, quantile)
        for quantile in (0.05, 0.25, 0.50, 0.75, 0.95)
    }


def _trace_episodes_goal109(path: Path) -> dict[int, dict[str, np.ndarray]]:
    episodes = _trace_episodes(path)
    with np.load(path) as data:
        missing = [name for name in REQUIRED_TRACE_FIELDS if name not in data]
        if missing:
            raise AuditError(f"{path}: missing Goal109 trace fields {missing}")
        applied = np.asarray(data["actuator_residual_applied"])
        base = np.asarray(data["actuator_base_physical"])
        saturated = np.asarray(data["material_saturated"])
        speed_error = np.asarray(data["speed_error_mps"])
        if applied.ndim != 2 or applied.shape[1] != 3:
            raise AuditError(f"{path}: actuator_residual_applied must have shape (N, 3)")
        if base.shape != applied.shape:
            raise AuditError(
                f"{path}: actuator_base_physical must match residual shape"
            )
        if saturated.shape != (applied.shape[0],):
            raise AuditError(f"{path}: material_saturated must have shape (N,)")
        if speed_error.shape != (applied.shape[0],):
            raise AuditError(f"{path}: speed_error_mps must have shape (N,)")
        if not np.all(np.isfinite(speed_error)):
            raise AuditError(f"{path}: speed_error_mps contains non-finite values")
    return episodes


def _cosine_rows(applied: np.ndarray, base: np.ndarray) -> np.ndarray:
    applied_norm = np.linalg.norm(applied, axis=1)
    base_norm = np.linalg.norm(base, axis=1)
    denominator = applied_norm * base_norm
    cosine = np.full(applied.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    cosine[valid] = np.sum(applied[valid] * base[valid], axis=1) / denominator[valid]
    return np.clip(cosine, -1.0, 1.0)


def _sign_alignment(residual: np.ndarray, error: np.ndarray) -> float:
    values = np.sign(np.asarray(residual, dtype=np.float64)) * np.sign(
        np.asarray(error, dtype=np.float64)
    )
    return float(np.mean(values)) if values.size else math.nan


def _episode_record(
    trace: dict[str, np.ndarray], *, seed: int, success: bool
) -> dict[str, Any]:
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)
    eta = np.asarray(trace["eta"], dtype=np.float64)
    goal = np.asarray(trace["goal_ned_m"], dtype=np.float64)
    applied = np.asarray(trace["actuator_residual_applied"], dtype=np.float64)
    base = np.asarray(trace["actuator_base_physical"], dtype=np.float64)
    saturated = np.asarray(trace["material_saturated"], dtype=bool)
    heading_error = np.asarray(trace["heading_error_rad"], dtype=np.float64)
    pitch_error = np.asarray(trace["pitch_error_rad"], dtype=np.float64)
    speed_error = np.asarray(trace["speed_error_mps"], dtype=np.float64)
    if time_s.size < 2 or distance.size != time_s.size:
        raise AuditError(f"seed={seed}: trace is too short or misaligned")
    if applied.shape != (time_s.size, 3) or base.shape != applied.shape:
        raise AuditError(f"seed={seed}: residual/base trace shape mismatch")
    if saturated.shape != (time_s.size,):
        raise AuditError(f"seed={seed}: saturation trace shape mismatch")
    last2s = time_s >= float(time_s[-1]) - 2.0
    if not bool(last2s.any()):
        raise AuditError(f"seed={seed}: empty last-2s window")

    cosine = _cosine_rows(applied, base)
    cosine_last2s = _finite_values(cosine[last2s])
    closure_all = -np.diff(distance) / np.maximum(np.diff(time_s), 1e-9)
    closure_last2s = closure_all[last2s[:-1]]
    terminal_error = goal[-1, :3] - eta[-1, :3]
    terminal_horizontal = float(np.linalg.norm(terminal_error[:2]))
    terminal_vertical = float(abs(terminal_error[2]))
    residual_norms = np.linalg.norm(applied[last2s], axis=1)

    shell_stats = {
        f"last2s_fraction_within_{threshold:g}m": float(
            np.mean(distance[last2s] <= threshold)
        )
        for threshold in SHELL_THRESHOLDS
    }
    return {
        "seed": seed,
        "success": success,
        "last2s_steps": int(last2s.sum()),
        "applied_base_cosine_mean": float(np.mean(cosine_last2s))
        if cosine_last2s.size
        else math.nan,
        "applied_base_cosine_p05": float(np.quantile(cosine_last2s, 0.05))
        if cosine_last2s.size
        else math.nan,
        "rudder_heading_sign_alignment": _sign_alignment(
            applied[last2s, 0], heading_error[last2s]
        ),
        "stern_pitch_sign_alignment": _sign_alignment(
            applied[last2s, 1], pitch_error[last2s]
        ),
        "rpm_speed_sign_alignment": _sign_alignment(
            applied[last2s, 2], speed_error[last2s]
        ),
        "residual_norm_mean": float(np.mean(residual_norms)),
        "residual_norm_p95": float(np.quantile(residual_norms, 0.95)),
        "material_saturation_fraction": float(np.mean(saturated[last2s])),
        "final_goal_distance_m": float(distance[-1]),
        "terminal_horizontal_miss_m": terminal_horizontal,
        "terminal_vertical_miss_m": terminal_vertical,
        "terminal_abs_heading_error_rad": float(abs(heading_error[-1])),
        "terminal_abs_pitch_error_rad": float(abs(pitch_error[-1])),
        "last2s_closure_mean_mps": float(np.nanmean(closure_last2s)),
        "last2s_closure_last_mps": float(closure_last2s[-1]),
        "last2s_closure_negative_fraction": float(np.mean(closure_last2s < 0.0)),
        **shell_stats,
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
    trace = _trace_episodes_goal109(run_dir / "trace.npz")
    if set(trace) != EXPECTED_SEEDS:
        raise AuditError(f"{summary_path}: trace seeds do not cover 20-59")

    records = [
        _episode_record(trace[seed], seed=seed, success=by_seed[seed])
        for seed in sorted(EXPECTED_SEEDS)
    ]
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
        raise AuditError(f"{root}: missing Goal107 trace root")
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


def _group_stats(records: list[dict[str, Any]], metric: str) -> dict[str, float]:
    success_records = [row for row in records if row["success"]]
    failure_records = [row for row in records if not row["success"]]
    success_values = [float(row[metric]) for row in success_records]
    failure_values = [float(row[metric]) for row in failure_records]
    success_median = _finite_median(success_values)
    failure_median = _finite_median(failure_values)
    return {
        "success_n": len(success_values),
        "failure_n": len(failure_values),
        "success_median": success_median,
        "failure_median": failure_median,
        "median_difference": success_median - failure_median,
        "cliffs_delta": _cliffs_delta(success_values, failure_values),
    }


def _summarize(groups: dict[str, dict[str, Path]]) -> dict[str, dict[str, Any]]:
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
                "direction": {
                    metric: _group_stats(records, metric)
                    for metric in DIRECTION_METRICS
                },
                "magnitude": {
                    metric: _group_stats(records, metric)
                    for metric in MAGNITUDE_METRICS
                },
                "geometry": {
                    metric: _group_stats(records, metric)
                    for metric in GEOMETRY_METRICS
                },
                "shells": {
                    f"last2s_fraction_within_{threshold:g}m": _group_stats(
                        records, f"last2s_fraction_within_{threshold:g}m"
                    )
                    for threshold in SHELL_THRESHOLDS
                },
                "final_goal_distance_quantiles": _quantiles(
                    [row["final_goal_distance_m"] for row in records]
                ),
                "final_goal_distance_quantiles_success": _quantiles(
                    [row["final_goal_distance_m"] for row in records if row["success"]]
                ),
                "final_goal_distance_quantiles_failure": _quantiles(
                    [row["final_goal_distance_m"] for row in records if not row["success"]]
                ),
                "records": records,
            }
    return result


def _metric_effects(
    profile_summary: dict[str, Any], section: str, metrics: Sequence[str]
) -> dict[str, float]:
    return {
        metric: abs(float(profile_summary[section][metric]["cliffs_delta"]))
        for metric in metrics
    }


def _max_finite(values: dict[str, float]) -> float:
    finite = [value for value in values.values() if math.isfinite(value)]
    return max(finite) if finite else 0.0


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
    direction = _metric_effects(primary, "direction", DIRECTION_METRICS)
    magnitude = _metric_effects(primary, "magnitude", MAGNITUDE_METRICS)
    geometry = _metric_effects(primary, "geometry", GEOMETRY_METRICS)
    direction_max = _max_finite(direction)
    magnitude_max = _max_finite(magnitude)
    direction_strong = direction_max >= EFFECT_THRESHOLD
    magnitude_strong = magnitude_max >= EFFECT_THRESHOLD
    strong_geometry = [
        metric
        for metric, value in geometry.items()
        if math.isfinite(value) and value >= EFFECT_THRESHOLD
    ]

    if direction_strong and magnitude_strong:
        status = "RESIDUAL_DIRECTION_MISMATCH"
        reason = "direction_and_magnitude_both_separate_success_failure"
    elif magnitude_strong and not direction_strong:
        status = "RESIDUAL_MAGNITUDE_LIMITED"
        reason = "direction_does_not_separate_but_magnitude_or_saturation_does"
    elif not direction_strong and not magnitude_strong and len(strong_geometry) >= 2:
        status = "TERMINAL_GEOMETRY_LIMITED"
        reason = "direction_and_magnitude_do_not_separate_but_geometry_does"
    else:
        status = "NO_DIRECTION_SIGNAL"
        if direction_strong and not magnitude_strong:
            reason = "direction_separates_but_magnitude_does_not_no_preregistered_branch"
        else:
            reason = "no_preregistered_direction_magnitude_or_geometry_category_reached"

    return {
        "status": status,
        "reason": reason,
        "baseline_failures": [],
        "effect_threshold": EFFECT_THRESHOLD,
        "direction_cliffs_delta": direction,
        "direction_max_abs_cliffs_delta": direction_max,
        "direction_strong": direction_strong,
        "magnitude_cliffs_delta": magnitude,
        "magnitude_max_abs_cliffs_delta": magnitude_max,
        "magnitude_strong": magnitude_strong,
        "geometry_cliffs_delta": geometry,
        "geometry_strong_features": strong_geometry,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    gate = payload["gate"]
    lines = [
        "# Goal109 Phase 0：residual direction / magnitude / terminal capture 审计",
        "",
        "本文件由 `scripts/analyze_pc_goal109_residual_direction.py` 生成；",
        "只读 Goal107 dev40 trace，不训练、不改 reward/terminal/success、不打开 held-out。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- reason: `{gate['reason']}`",
        "",
    ]
    if gate["status"] == "INVALID":
        lines.extend(["## INVALID 原因", ""])
        lines.append(f"- `{gate.get('reason', '')}`")
        lines.extend(f"- {failure}" for failure in gate.get("baseline_failures", []))
        lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    else:
        lines.extend(
            [
                "## 判读摘要",
                "",
                f"- direction max |Cliff δ|: `{_fmt(gate['direction_max_abs_cliffs_delta'])}`",
                f"- magnitude max |Cliff δ|: `{_fmt(gate['magnitude_max_abs_cliffs_delta'])}`",
                f"- strong geometry features: `{gate['geometry_strong_features']}`",
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
        lines.append("")
        lines.extend(
            [
                "### C0 residual direction（last 2 s）",
                "",
                "| metric | success median | failure median | Cliff δ |",
                "|---|---:|---:|---:|",
            ]
        )
        c0 = payload["summary"][scenario]["C0"]
        for name, stats in c0["direction"].items():
            lines.append(
                f"| {name} | {_fmt(stats['success_median'])} | "
                f"{_fmt(stats['failure_median'])} | {_fmt(stats['cliffs_delta'])} |"
            )
        lines.extend(
            [
                "",
                "### C0 residual magnitude（last 2 s）",
                "",
                "| metric | success median | failure median | Cliff δ |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, stats in c0["magnitude"].items():
            lines.append(
                f"| {name} | {_fmt(stats['success_median'])} | "
                f"{_fmt(stats['failure_median'])} | {_fmt(stats['cliffs_delta'])} |"
            )
        lines.extend(
            [
                "",
                "### C0 terminal geometry / closure",
                "",
                "| metric | success median | failure median | Cliff δ |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, stats in c0["geometry"].items():
            lines.append(
                f"| {name} | {_fmt(stats['success_median'])} | "
                f"{_fmt(stats['failure_median'])} | {_fmt(stats['cliffs_delta'])} |"
            )
        lines.extend(
            [
                "",
                "### C0 shell transition（last 2 s fraction）",
                "",
                "| shell | success median | failure median | Cliff δ |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, stats in c0["shells"].items():
            lines.append(
                f"| {name} | {_fmt(stats['success_median'])} | "
                f"{_fmt(stats['failure_median'])} | {_fmt(stats['cliffs_delta'])} |"
            )
        quantiles = c0["final_goal_distance_quantiles"]
        lines.extend(
            [
                "",
                "Final goal distance quantiles (all C0 episodes):",
                "",
                "| p05 | p25 | p50 | p75 | p95 |",
                "|---:|---:|---:|---:|---:|",
                "| "
                + " | ".join(_fmt(quantiles[f"p{int(q * 100):02d}"]) for q in (0.05, 0.25, 0.50, 0.75, 0.95))
                + " |",
                "",
            ]
        )
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
            "shell_thresholds_m": list(SHELL_THRESHOLDS),
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
            "shell_thresholds_m": list(SHELL_THRESHOLDS),
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
