#!/usr/bin/env python3
"""Analyze the Goal169 zero-controller terminal alignment screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

RADII = ("default", "strict125")
LEVELS = ("nominal", "current07", "current10", "current15")
VARIANTS = (20, 30, 40, 50)
FINAL_BIN_EDGES_M = (0.0, 1.25, 1.5, 2.0, 2.5, 3.0, math.inf)
FINAL_BIN_LABELS = ("<=1.25", "1.25-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", ">3.0")


def _histogram(values: list[float]) -> dict[str, int]:
    counts, _edges = np.histogram(np.asarray(values, dtype=np.float64), bins=FINAL_BIN_EDGES_M)
    return {label: int(count) for label, count in zip(FINAL_BIN_LABELS, counts.tolist())}


def _trace_metrics(trace: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return speed, vertical capture and uncensored closest-approach diagnostics."""

    active = np.asarray(trace["active"], dtype=bool)
    seeds = np.asarray(trace["seed"], dtype=np.int64)[active]
    steps = np.asarray(trace["step"], dtype=np.int64)[active]
    distances = np.asarray(trace["goal_distance_eta"], dtype=np.float64)[active]
    vertical = np.abs(
        np.asarray(trace["eta_ned"], dtype=np.float64)[active, 2]
        - np.asarray(trace["goal_ned"], dtype=np.float64)[active, 2]
    )
    speed = np.linalg.norm(
        np.asarray(trace["ground_velocity_body_mps"], dtype=np.float64)[active], axis=1
    )
    min_distance: dict[int, float] = {}
    final_vertical: dict[int, float] = {}
    last_step: dict[int, int] = {}
    for seed, step, distance, vertical_error in zip(seeds, steps, distances, vertical):
        seed_value = int(seed)
        if seed_value not in min_distance or distance < min_distance[seed_value]:
            min_distance[seed_value] = float(distance)
        if seed_value not in last_step or step >= last_step[seed_value]:
            last_step[seed_value] = int(step)
            final_vertical[seed_value] = float(vertical_error)
    expected_seeds = {int(row["seed"]) for row in episodes}
    if expected_seeds != set(min_distance):
        raise ValueError(
            f"trace/episode seed mismatch: missing={sorted(expected_seeds - set(min_distance))}, "
            f"extra={sorted(set(min_distance) - expected_seeds)}"
        )
    return {
        "speed_mean_mps": float(np.mean(speed)),
        "speed_p50_mps": float(np.median(speed)),
        "speed_p95_mps": float(np.percentile(speed, 95.0)),
        "speed_max_mps": float(np.max(speed)),
        "overspeed_steps": int(np.sum(speed > 2.5 + 1e-12)),
        "eta_min_distance_m": [float(min_distance[int(row["seed"])]) for row in episodes],
        "final_vertical_abs_m": [float(final_vertical[int(row["seed"])]) for row in episodes],
    }


def _run(root: Path, variant: int, radius: str, level: str) -> dict[str, Any]:
    run_dir = root / f"alignment{variant}" / radius / level / "zero_seed86201"
    episodes = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trace = np.load(run_dir / "trace.npz", allow_pickle=False)
    success = sum(bool(row.get("success", False)) for row in episodes)
    timeout = sum(bool(row.get("timeout", False)) for row in episodes)
    safety = sum(bool(row.get("safety", False)) for row in episodes)
    plane_miss = sum(int(row.get("terminal_kind", -1)) == 9 for row in episodes)
    final_distance = [float(row.get("terminal_eta_distance_m", math.nan)) for row in episodes]
    failure_distance = [
        distance
        for distance, row in zip(final_distance, episodes)
        if not bool(row.get("success", False))
    ]
    trace_metrics = _trace_metrics(trace, episodes)
    success_mask = np.asarray([bool(row.get("success", False)) for row in episodes])
    min_distance = np.asarray(trace_metrics["eta_min_distance_m"], dtype=np.float64)
    min_bin_entered: dict[str, int] = {}
    min_bin_success_rate: dict[str, float | None] = {}
    for low, high, label in zip(
        FINAL_BIN_EDGES_M[:-1], FINAL_BIN_EDGES_M[1:], FINAL_BIN_LABELS
    ):
        mask = (min_distance >= low) & (min_distance < high)
        min_bin_entered[label] = int(mask.sum())
        min_bin_success_rate[label] = (
            None if not mask.any() else float(success_mask[mask].mean())
        )
    return {
        "variant": variant,
        "radius": radius,
        "level": level,
        "episodes": len(episodes),
        "success": success,
        "timeout": timeout,
        "safety": safety,
        "plane_miss": plane_miss,
        "path_three_d_rmse_m": float(summary["summary"]["path_three_d_rmse_m_mean"]),
        "final_distance_mean_m": float(np.mean(final_distance)),
        "final_vertical_abs_mean_m": float(np.mean(trace_metrics["final_vertical_abs_m"])),
        "final_distance_bins_all": _histogram(final_distance),
        "final_distance_bins_failures": _histogram(failure_distance),
        "min_distance_bins_entered": min_bin_entered,
        "min_distance_bin_success_rate_censored": min_bin_success_rate,
        "speed_mean_mps": trace_metrics["speed_mean_mps"],
        "speed_p50_mps": trace_metrics["speed_p50_mps"],
        "speed_p95_mps": trace_metrics["speed_p95_mps"],
        "speed_max_mps": trace_metrics["speed_max_mps"],
        "overspeed_steps": trace_metrics["overspeed_steps"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for radius in RADII:
            for level in LEVELS:
                rows.append(_run(args.eval_root, variant, radius, level))
    by_key = {(row["variant"], row["radius"], row["level"]): row for row in rows}
    baseline = {
        (row["radius"], row["level"]): row
        for row in rows
        if row["variant"] == 20
    }
    for row in rows:
        base = baseline[(row["radius"], row["level"])]
        row["delta_success"] = row["success"] - base["success"]
        row["delta_path_m"] = row["path_three_d_rmse_m"] - base["path_three_d_rmse_m"]
        row["delta_final_m"] = row["final_distance_mean_m"] - base["final_distance_mean_m"]
        row["delta_final_vertical_m"] = (
            row["final_vertical_abs_mean_m"] - base["final_vertical_abs_mean_m"]
        )
        row["delta_speed_mean_mps"] = row["speed_mean_mps"] - base["speed_mean_mps"]
        row["delta_timeout"] = row["timeout"] - base["timeout"]
        row["delta_safety"] = row["safety"] - base["safety"]
        row["delta_plane_miss"] = row["plane_miss"] - base["plane_miss"]

    gate: dict[str, Any] = {}
    for variant in VARIANTS:
        checks = {}
        for radius in RADII:
            primary = by_key[(variant, radius, "current10")]
            threshold = 4 if radius == "default" else 2
            guards_ok = all(
                by_key[(variant, radius, level)]["delta_success"] >= -2
                for level in ("nominal", "current07", "current15")
            )
            path_ok = all(
                by_key[(variant, radius, level)]["delta_path_m"] <= 0.05 + 1e-12
                for level in LEVELS
            )
            final_ok = all(
                by_key[(variant, radius, level)]["delta_final_m"] <= 0.05 + 1e-12
                for level in LEVELS
            )
            vertical_ok = all(
                by_key[(variant, radius, level)]["delta_final_vertical_m"]
                <= 0.05 + 1e-12
                for level in LEVELS
            )
            timeout_ok = all(
                by_key[(variant, radius, level)]["timeout"]
                <= by_key[(20, radius, level)]["timeout"]
                for level in LEVELS
            )
            safety_ok = all(
                by_key[(variant, radius, level)]["safety"]
                <= by_key[(20, radius, level)]["safety"]
                for level in LEVELS
            )
            # A material slowdown would make success gains non-comparable.  The
            # 0.05 m/s allowance is below the pre-registered 0.9 m/s eval axis spacing.
            speed_retention_ok = all(
                by_key[(variant, radius, level)]["delta_speed_mean_mps"] >= -0.05
                for level in LEVELS
            )
            near_bins_ok = True
            for level in LEVELS:
                row = by_key[(variant, radius, level)]
                base_row = by_key[(20, radius, level)]
                near_failures = sum(
                    row["final_distance_bins_failures"][label]
                    for label in ("2.0-2.5", "2.5-3.0")
                )
                base_near_failures = sum(
                    base_row["final_distance_bins_failures"][label]
                    for label in ("2.0-2.5", "2.5-3.0")
                )
                far_failures = row["final_distance_bins_failures"][">3.0"]
                base_far_failures = base_row["final_distance_bins_failures"][">3.0"]
                if near_failures > base_near_failures + 1:
                    near_bins_ok = False
                if far_failures > base_far_failures + 1:
                    near_bins_ok = False
            speed_ok = all(
                by_key[(variant, radius, level)]["overspeed_steps"] == 0
                for level in LEVELS
            )
            checks[radius] = {
                "delta_success": primary["delta_success"],
                "threshold": threshold,
                "guards_ok": guards_ok,
                "path_ok": path_ok,
                "final_ok": final_ok,
                "vertical_ok": vertical_ok,
                "timeout_ok": timeout_ok,
                "safety_ok": safety_ok,
                "near_bins_ok": near_bins_ok,
                "speed_retention_ok": speed_retention_ok,
                "speed_ok": speed_ok,
                "pass": (
                    primary["delta_success"] >= threshold
                    and guards_ok
                    and path_ok
                    and final_ok
                    and vertical_ok
                    and timeout_ok
                    and safety_ok
                    and near_bins_ok
                    and speed_retention_ok
                    and speed_ok
                ),
            }
        gate[str(variant)] = checks

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "goal169_homing_screen.v2",
        "final_distance_bin_edges_m": list(FINAL_BIN_EDGES_M[:-1]) + ["inf"],
        "rows": rows,
        "gate": gate,
    }
    (args.output / "goal169_homing_screen.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Goal169 terminal homing timing screen",
        "",
        "| alignment | radius | level | success | Δ | timeout | safety | plane miss | Δmiss | path3d | Δpath | final d | Δfinal | vert | Δvert | speed mean | Δspeed | speed max | over |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {radius} | {level} | {success}/{episodes} | {ds:+d} | "
            "{timeout} | {safety} | {plane_miss} | {dm:+d} | {path:.3f} | {dp:+.3f} | "
            "{final:.3f} | {df:+.3f} | {vertical:.3f} | {dv:+.3f} | {speed_mean:.3f} | "
            "{dspeed:+.3f} | {speed:.3f} | {over} |".format(
                variant=row["variant"],
                radius=row["radius"],
                level=row["level"],
                success=row["success"],
                episodes=row["episodes"],
                ds=row["delta_success"],
                timeout=row["timeout"],
                safety=row["safety"],
                plane_miss=row["plane_miss"],
                dm=row["delta_plane_miss"],
                path=row["path_three_d_rmse_m"],
                dp=row["delta_path_m"],
                final=row["final_distance_mean_m"],
                df=row["delta_final_m"],
                vertical=row["final_vertical_abs_mean_m"],
                dv=row["delta_final_vertical_m"],
                speed_mean=row["speed_mean_mps"],
                dspeed=row["delta_speed_mean_mps"],
                speed=row["speed_max_mps"],
                over=row["overspeed_steps"],
            )
        )
    lines.extend(
        [
            "",
            "## Final-distance bins (all episodes)",
            "",
            "| alignment | radius | level | " + " | ".join(FINAL_BIN_LABELS) + " |",
            "|---:|---|---|" + "---:|" * len(FINAL_BIN_LABELS),
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {radius} | {level} | {bins} |".format(
                variant=row["variant"],
                radius=row["radius"],
                level=row["level"],
                bins=" | ".join(
                    str(row["final_distance_bins_all"][label])
                    for label in FINAL_BIN_LABELS
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Failure-only near-miss bins",
            "",
            "| alignment | radius | level | " + " | ".join(FINAL_BIN_LABELS) + " |",
            "|---:|---|---|" + "---:|" * len(FINAL_BIN_LABELS),
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {radius} | {level} | {bins} |".format(
                variant=row["variant"],
                radius=row["radius"],
                level=row["level"],
                bins=" | ".join(
                    str(row["final_distance_bins_failures"][label])
                    for label in FINAL_BIN_LABELS
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Closest-approach bins (all episodes; success censored at endpoint stop)",
            "",
            "| alignment | radius | level | entered | success rate |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        entered = " / ".join(
            str(row["min_distance_bins_entered"][label]) for label in FINAL_BIN_LABELS
        )
        rates = " / ".join(
            "NA"
            if row["min_distance_bin_success_rate_censored"][label] is None
            else f"{row['min_distance_bin_success_rate_censored'][label]:.2f}"
            for label in FINAL_BIN_LABELS
        )
        lines.append(
            f"| {row['variant']} | {row['radius']} | {row['level']} | {entered} | {rates} |"
        )
    lines.extend(["", "## Gate", ""])
    for variant, checks in gate.items():
        lines.append(f"- alignment {variant}: `{checks}`")
    (args.output / "goal169_homing_screen.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.output / "goal169_homing_screen.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

