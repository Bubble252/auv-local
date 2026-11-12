#!/usr/bin/env python3
"""Analyze Goal172 full-matrix residual attribution diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_goal170_route_transfer import analyze_run  # noqa: E402
from check_goal172_protocol import audit_input, audit_output  # noqa: E402
from goal172_matrix_contract import (  # noqa: E402
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_FORMAL_ROOT,
    LEVELS,
    RADII,
    diagnostic_jobs,
    formal_baseline_path,
)


def _finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(array.mean())


def _fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _relative(delta: float | None, baseline: float | None) -> float | None:
    if delta is None or baseline is None or abs(baseline) < 1e-12:
        return None
    return delta / baseline


def _scale_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_by_seed = baseline["by_seed"]
    candidate_by_seed = candidate["by_seed"]
    common = sorted(set(baseline_by_seed) & set(candidate_by_seed))
    deltas = [
        float(candidate_by_seed[seed]["route_e3d_t_rmse_m"])
        - float(baseline_by_seed[seed]["route_e3d_t_rmse_m"])
        for seed in common
    ]
    base_mean = _finite_mean(
        float(baseline_by_seed[seed]["route_e3d_t_rmse_m"]) for seed in common
    )
    candidate_mean = _finite_mean(
        float(candidate_by_seed[seed]["route_e3d_t_rmse_m"]) for seed in common
    )
    delta_mean = _finite_mean(deltas)
    improved_segments = 0
    segment_deltas: dict[str, float | None] = {}
    for segment in ("0-20", "20-40", "40-60", "60-80", "80-100"):
        base_value = baseline["segments"][segment]["e3d_rmse_m"]
        candidate_value = candidate["segments"][segment]["e3d_rmse_m"]
        delta = (
            None
            if base_value is None or candidate_value is None
            else float(candidate_value) - float(base_value)
        )
        segment_deltas[segment] = delta
        if delta is not None and delta < 0.0:
            improved_segments += 1
    return {
        "seed_count": len(common),
        "baseline_mean_m": base_mean,
        "candidate_mean_m": candidate_mean,
        "delta_mean_m": delta_mean,
        "relative_delta": _relative(delta_mean, base_mean),
        "improved_segments": improved_segments,
        "segment_deltas_m": segment_deltas,
        "baseline_cross_p90_m": baseline.get("route_cross_abs_p90_m"),
        "candidate_cross_p90_m": candidate.get("route_cross_abs_p90_m"),
        "baseline_vert_p90_m": baseline.get("route_vert_abs_p90_m"),
        "candidate_vert_p90_m": candidate.get("route_vert_abs_p90_m"),
        "candidate_speed_max_mps": candidate.get("speed_max_mps"),
        "candidate_overspeed_steps": candidate.get("overspeed_steps"),
        "candidate_timeout_count": candidate.get("timeout_count"),
        "candidate_safety_count": candidate.get("safety_count"),
        "candidate_completion_fraction": candidate.get("completion_fraction"),
    }


def _aggregate(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "OK":
            continue
        grouped[tuple(row[key] for key in group_keys)].append(row)
    result: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        baseline = _finite_mean(
            row["baseline_mean_m"]
            for row in members
            if row.get("baseline_mean_m") is not None
        )
        candidate = _finite_mean(
            row["candidate_mean_m"]
            for row in members
            if row.get("candidate_mean_m") is not None
        )
        delta = _finite_mean(
            row["delta_mean_m"]
            for row in members
            if row.get("delta_mean_m") is not None
        )
        row: dict[str, Any] = {key_name: value for key_name, value in zip(group_keys, key)}
        row.update(
            {
                "status": "OK",
                "level": "all",
                "level_count": len(members),
                "improved_levels": sum(
                    1
                    for member in members
                    if member.get("delta_mean_m") is not None
                    and float(member["delta_mean_m"]) < 0.0
                ),
                "levels": [member["level"] for member in members],
                "seed_count_per_level": sorted({int(member["seed_count"]) for member in members}),
                "baseline_mean_m": baseline,
                "candidate_mean_m": candidate,
                "delta_mean_m": delta,
                "relative_delta": _relative(delta, baseline),
                "improved_segments": sum(int(member["improved_segments"]) for member in members),
                "segment_cells": 5 * len(members),
                "candidate_speed_max_mps": max(
                    (float(member["candidate_speed_max_mps"]) for member in members),
                    default=None,
                ),
                "candidate_overspeed_steps": sum(
                    int(member.get("candidate_overspeed_steps") or 0) for member in members
                ),
                "candidate_timeout_count": sum(
                    int(member.get("candidate_timeout_count") or 0) for member in members
                ),
                "candidate_safety_count": sum(
                    int(member.get("candidate_safety_count") or 0) for member in members
                ),
                "candidate_completion_fraction": _finite_mean(
                    member["candidate_completion_fraction"]
                    for member in members
                    if member.get("candidate_completion_fraction") is not None
                ),
            }
        )
        result.append(row)
    return result


def _trace_reward_means(run_dir: Path) -> dict[str, float]:
    trace_path = run_dir / "trace.npz"
    if not trace_path.is_file():
        trace_path = run_dir.parent / f"{run_dir.name}.npz"
    if not trace_path.is_file():
        return {}
    with np.load(trace_path, allow_pickle=False) as trace:
        result: dict[str, float] = {}
        for key in trace.files:
            if not key.startswith("reward_"):
                continue
            values = np.asarray(trace[key], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                result[key] = float(values.mean())
        return result


def _reward_attribution(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for radius in RADII:
        for level in LEVELS:
            old_dir = (
                root
                / "reward_attribution"
                / "old_route"
                / radius
                / level
                / "zero_seed86201"
            )
            route_dir = (
                root
                / "reward_attribution"
                / "route_health"
                / radius
                / level
                / "zero_seed86201"
            )
            if not (old_dir / "summary.json").is_file() or not (
                route_dir / "summary.json"
            ).is_file():
                rows.append({"radius": radius, "level": level, "status": "MISSING"})
                continue
            old = analyze_run(old_dir)
            route = analyze_run(route_dir)
            common = sorted(set(old["by_seed"]) & set(route["by_seed"]))
            route_diffs = [
                abs(
                    float(old["by_seed"][seed]["route_e3d_t_rmse_m"])
                    - float(route["by_seed"][seed]["route_e3d_t_rmse_m"])
                )
                for seed in common
            ]
            old_rewards = _trace_reward_means(old_dir)
            route_rewards = _trace_reward_means(route_dir)
            all_keys = sorted(set(old_rewards) | set(route_rewards))
            reward_diffs = {
                key: abs(old_rewards.get(key, 0.0) - route_rewards.get(key, 0.0))
                for key in all_keys
            }
            rows.append(
                {
                    "radius": radius,
                    "level": level,
                    "status": "OK",
                    "common_seeds": len(common),
                    "max_abs_route_delta_m": max(route_diffs) if route_diffs else None,
                    "old_route_keys": sorted(old_rewards),
                    "route_health_keys": sorted(route_rewards),
                    "reward_component_abs_diff": reward_diffs,
                }
            )
    return rows


def _comparison_rows(root: Path, formal_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scale_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    rudder_rows: list[dict[str, Any]] = []
    for job in diagnostic_jobs():
        if job.kind not in {"ppo", "sac", "channel", "rudder_scale"}:
            continue
        baseline_dir = formal_baseline_path(formal_root, job.arm, job.radius, job.level)
        candidate_dir = job.output_path(root)
        if not (baseline_dir / "summary.json").is_file():
            row = {
                "arm": job.arm,
                "radius": job.radius,
                "level": job.level,
                "status": "MISSING_BASELINE",
            }
        elif not (candidate_dir / "summary.json").is_file():
            row = {
                "arm": job.arm,
                "radius": job.radius,
                "level": job.level,
                "status": "MISSING",
            }
        else:
            baseline = analyze_run(baseline_dir)
            candidate = analyze_run(candidate_dir)
            row = {
                "arm": job.arm,
                "radius": job.radius,
                "level": job.level,
                "status": "OK",
                **_scale_delta(baseline, candidate),
            }
        if job.kind in {"ppo", "sac"}:
            row.update(
                {
                    "algorithm": "PPO" if job.kind == "ppo" else "SAC",
                    "scale": f"{job.value}x" if job.kind == "sac" else job.value,
                }
            )
            scale_rows.append(row)
        elif job.kind == "channel":
            row["channel"] = job.tag
            channel_rows.append(row)
        else:
            row["scale"] = job.value
            rudder_rows.append(row)
    return scale_rows, channel_rows, rudder_rows


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal172 residual attribution — full matrix",
        "",
        "`DIAGNOSTIC_NOT_GATE`；不训练，不打开 held-out。",
        "",
        f"- matrix: `{len(LEVELS)} levels × {len(RADII)} radii`",
        f"- diagnostic cells: `{payload['matrix']['diagnostic_cells']}`",
        f"- seed range: `{payload['matrix']['seed_start']}-{payload['matrix']['seed_end']}`",
        "",
        "## Scale amplification (aggregate over levels)",
        "",
        "| algorithm | arm | radius | scale | Δ3D RMSE | relative | improved levels | improved segments | speed max | overspeed | timeout | safety |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["scale_aggregate"]:
        lines.append(
            "| {algorithm} | {arm} | {radius} | {scale} | {delta} | {relative} | "
            "{levels}/{level_count} | {segments}/{segment_cells} | {speed} | {overspeed} | "
            "{timeout} | {safety} |".format(
                algorithm=row["algorithm"],
                arm=row["arm"],
                radius=row["radius"],
                scale=row["scale"],
                delta=_fmt(row.get("delta_mean_m"), signed=True),
                relative=(
                    "—"
                    if row.get("relative_delta") is None
                    else f"{100.0 * row['relative_delta']:+.3f}%"
                ),
                levels=row.get("improved_levels", "—"),
                level_count=row.get("level_count", "—"),
                segments=row.get("improved_segments", "—"),
                segment_cells=row.get("segment_cells", "—"),
                speed=_fmt(row.get("candidate_speed_max_mps")),
                overspeed=row.get("candidate_overspeed_steps", "—"),
                timeout=row.get("candidate_timeout_count", "—"),
                safety=row.get("candidate_safety_count", "—"),
            )
        )
    lines.extend(
        [
            "",
            "## Scale amplification by level",
            "",
            "| algorithm | arm | radius | level | scale | Δ3D RMSE | relative | improved segments |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["scale_runs"]:
        lines.append(
            "| {algorithm} | {arm} | {radius} | {level} | {scale} | {delta} | {relative} | "
            "{segments}/5 |".format(
                algorithm=row.get("algorithm", "—"),
                arm=row.get("arm", "—"),
                radius=row.get("radius", "—"),
                level=row.get("level", "—"),
                scale=row.get("scale", "—"),
                delta=_fmt(row.get("delta_mean_m"), signed=True),
                relative=(
                    "—"
                    if row.get("relative_delta") is None
                    else f"{100.0 * row['relative_delta']:+.3f}%"
                ),
                segments=row.get("improved_segments", "—"),
            )
        )
    lines.extend(
        [
            "",
            "## PPO channel attribution at effective scale 0.50",
            "",
            "| arm | radius | channel | Δ3D RMSE | relative | improved segments | speed max | overspeed |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["channel_aggregate"]:
        lines.append(
            "| {arm} | {radius} | {channel} | {delta} | {relative} | {segments}/{segment_cells} | "
            "{speed} | {overspeed} |".format(
                arm=row["arm"],
                radius=row["radius"],
                channel=row["channel"],
                delta=_fmt(row.get("delta_mean_m"), signed=True),
                relative=(
                    "—"
                    if row.get("relative_delta") is None
                    else f"{100.0 * row['relative_delta']:+.3f}%"
                ),
                segments=row.get("improved_segments", "—"),
                segment_cells=row.get("segment_cells", "—"),
                speed=_fmt(row.get("candidate_speed_max_mps")),
                overspeed=row.get("candidate_overspeed_steps", "—"),
            )
        )
    lines.extend(
        [
            "",
            "## R1_BC_PPO rudder-only scale sweep (aggregate over levels)",
            "",
            "| radius | effective scale | Δ3D RMSE | relative | improved levels | improved segments | speed max | overspeed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rudder_aggregate"]:
        lines.append(
            "| {radius} | {scale} | {delta} | {relative} | {levels}/{level_count} | "
            "{segments}/{segment_cells} | {speed} | {overspeed} |".format(
                radius=row["radius"],
                scale=row["scale"],
                delta=_fmt(row.get("delta_mean_m"), signed=True),
                relative=(
                    "—"
                    if row.get("relative_delta") is None
                    else f"{100.0 * row['relative_delta']:+.3f}%"
                ),
                levels=row.get("improved_levels", "—"),
                level_count=row.get("level_count", "—"),
                segments=row.get("improved_segments", "—"),
                segment_cells=row.get("segment_cells", "—"),
                speed=_fmt(row.get("candidate_speed_max_mps")),
                overspeed=row.get("candidate_overspeed_steps", "—"),
            )
        )
    lines.extend(
        [
            "",
            "## Reward attribution by level",
            "",
            "| radius | level | common seeds | max abs route delta | nonzero reward components |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in payload["reward_attribution"]:
        if row.get("status") != "OK":
            lines.append(
                f"| {row.get('radius', '—')} | {row.get('level', '—')} | — | — | "
                f"`{row.get('status')}` |"
            )
            continue
        nonzero = {
            key: value
            for key, value in row["reward_component_abs_diff"].items()
            if value > 1e-12
        }
        lines.append(
            "| {radius} | {level} | {seeds} | {delta} | `{nonzero}` |".format(
                radius=row["radius"],
                level=row["level"],
                seeds=row["common_seeds"],
                delta=_fmt(row["max_abs_route_delta_m"]),
                nonzero=nonzero,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_errors = audit_input(args.formal_root)
    output_errors = audit_output(args.root)
    if input_errors or output_errors:
        for error in [*input_errors, *output_errors]:
            print(f"INVALID_INCOMPLETE_EVAL: {error}", file=sys.stderr)
        return 1

    scale_rows, channel_rows, rudder_rows = _comparison_rows(args.root, args.formal_root)
    payload = {
        "schema": "goal172_residual_attribution.v2",
        "status": "DIAGNOSTIC_NOT_GATE",
        "matrix": {
            "levels": list(LEVELS),
            "radii": list(RADII),
            "seed_start": 20,
            "seed_end": 59,
            "diagnostic_cells": len(diagnostic_jobs()),
        },
        "scale_runs": scale_rows,
        "scale_aggregate": _aggregate(
            scale_rows, ("algorithm", "arm", "radius", "scale")
        ),
        "channel_runs": channel_rows,
        "channel_aggregate": _aggregate(channel_rows, ("arm", "radius", "channel")),
        "rudder_scale_runs": rudder_rows,
        "rudder_aggregate": _aggregate(rudder_rows, ("radius", "scale")),
        "reward_attribution": _reward_attribution(args.root),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "goal172_residual_attribution.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(payload, args.output / "goal172_residual_attribution.md")
    print(json.dumps(payload["matrix"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

