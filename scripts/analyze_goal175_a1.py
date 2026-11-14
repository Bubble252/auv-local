#!/usr/bin/env python3
"""Compare Goal175 A1 same-checkpoint ALOS-COMP versus ALOS-NONE route health."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_goal170_route_transfer import _mean, analyze_run, paired_delta  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "artifacts/lowlevel/goal175_alos_comp_mlp_20261114"
SEGMENTS = ("0-20", "20-40", "40-60", "60-80", "80-100")
KEYPOINTS = tuple(f"{value:02d}" for value in range(20, 91, 10))


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _terminal_vertical_mean(run_dir: Path) -> float:
    trace_path = run_dir / "trace.npz"
    trace = np.load(trace_path, allow_pickle=False)
    active = np.asarray(trace["active"], dtype=bool)
    seeds = np.asarray(trace["seed"], dtype=np.int64)
    vertical = np.asarray(trace["path_vertical_error"], dtype=np.float64)
    values: list[float] = []
    for seed in np.unique(seeds[active]):
        rows = np.flatnonzero(active & (seeds == seed))
        if rows.size:
            values.append(abs(float(vertical[rows[-1]])))
    return float(np.mean(values)) if values else math.nan


def _cell(root: Path, scenario: dict[str, Any], *, bootstrap_samples: int) -> dict[str, Any]:
    scenario_id = str(scenario["scenario"])
    comp_dir = root / "eval/P_RUD_075_COMP" / scenario_id
    none_dir = root / "eval/P_RUD_075_NONE" / scenario_id
    comp = analyze_run(comp_dir, default_length_m=35.0)
    none = analyze_run(none_dir, default_length_m=35.0)
    delta = paired_delta(comp, none, bootstrap_samples=bootstrap_samples)
    segments_improved = sum(
        1
        for label in SEGMENTS
        if delta["segments"][label]["mean"] is not None
        and float(delta["segments"][label]["mean"]) < 0.0
    )
    keypoints_improved = sum(
        1
        for key in KEYPOINTS
        if delta["keypoints"][key]["mean"] is not None
        and float(delta["keypoints"][key]["mean"]) < 0.0
    )
    route_delta = float(delta["route_e3d_t_rmse_m"]["mean"])
    cross_delta = float(delta["route_cross_abs_p90_m"]["mean"])
    vertical_delta = float(delta["route_vert_abs_p90_m"]["mean"])
    none_route = float(none["route_e3d_t_rmse_m"])
    none_cross = float(none["route_cross_abs_p90_m"])
    comp_terminal_vertical = _terminal_vertical_mean(comp_dir)
    none_terminal_vertical = _terminal_vertical_mean(none_dir)
    checks = {
        "vertical_p90_guard": vertical_delta <= 0.02,
        "terminal_vertical_guard": comp_terminal_vertical - none_terminal_vertical <= 0.02,
        "success_guard": comp["success_count"] - none["success_count"] >= -2,
        "timeout_guard": comp["timeout_count"] <= none["timeout_count"],
        "safety_guard": comp["safety_count"] <= none["safety_count"],
        "speed_guard": comp["speed_max_mps"] <= 2.5 + 1e-12 and comp["overspeed_steps"] == 0,
    }
    return {
        "scenario": scenario_id,
        "pressure": scenario["pressure"],
        "radius": scenario["radius"],
        "comp": {
            "route_e3d_rmse_m": comp["route_e3d_t_rmse_m"],
            "cross_p90_m": comp["route_cross_abs_p90_m"],
            "vertical_p90_m": comp["route_vert_abs_p90_m"],
            "terminal_vertical_abs_m": comp_terminal_vertical,
            "success_count": comp["success_count"],
            "timeout_count": comp["timeout_count"],
            "safety_count": comp["safety_count"],
            "speed_max_mps": comp["speed_max_mps"],
            "overspeed_steps": comp["overspeed_steps"],
        },
        "none": {
            "route_e3d_rmse_m": none["route_e3d_t_rmse_m"],
            "cross_p90_m": none["route_cross_abs_p90_m"],
            "vertical_p90_m": none["route_vert_abs_p90_m"],
            "terminal_vertical_abs_m": none_terminal_vertical,
            "success_count": none["success_count"],
            "timeout_count": none["timeout_count"],
            "safety_count": none["safety_count"],
            "speed_max_mps": none["speed_max_mps"],
            "overspeed_steps": none["overspeed_steps"],
        },
        "delta": {
            "route_e3d_rmse_m": route_delta,
            "route_relative": route_delta / none_route if none_route > 0.0 else None,
            "cross_p90_m": cross_delta,
            "cross_relative": cross_delta / none_cross if none_cross > 0.0 else None,
            "vertical_p90_m": vertical_delta,
            "terminal_vertical_abs_m": comp_terminal_vertical - none_terminal_vertical,
            "segments_improved": segments_improved,
            "keypoints_improved": keypoints_improved,
            "success": comp["success_count"] - none["success_count"],
            "timeout": comp["timeout_count"] - none["timeout_count"],
            "safety": comp["safety_count"] - none["safety_count"],
        },
        "checks": checks,
        "segment_deltas": {
            label: delta["segments"][label]["mean"] for label in SEGMENTS
        },
        "keypoint_deltas": {
            key: delta["keypoints"][key]["mean"] for key in KEYPOINTS
        },
    }


def _pre_gate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    current = [cell for cell in cells if cell["pressure"] == "current10"]
    route_delta = _mean(cell["delta"]["route_e3d_rmse_m"] for cell in current)
    none_route = _mean(cell["none"]["route_e3d_rmse_m"] for cell in current)
    cross_delta = _mean(cell["delta"]["cross_p90_m"] for cell in current)
    none_cross = _mean(cell["none"]["cross_p90_m"] for cell in current)
    route_relative = (
        float(route_delta) / float(none_route)
        if route_delta is not None and none_route not in (None, 0.0)
        else None
    )
    cross_relative = (
        float(cross_delta) / float(none_cross)
        if cross_delta is not None and none_cross not in (None, 0.0)
        else None
    )
    segment_deltas = {
        label: _mean(cell["segment_deltas"][label] for cell in current)
        for label in SEGMENTS
    }
    keypoint_deltas = {
        key: _mean(cell["keypoint_deltas"][key] for cell in current)
        for key in KEYPOINTS
    }
    segments_improved = sum(
        1 for value in segment_deltas.values() if value is not None and float(value) < 0.0
    )
    keypoints_improved = sum(
        1 for value in keypoint_deltas.values() if value is not None and float(value) < 0.0
    )
    guard_cells = [cell for cell in cells if not all(cell["checks"].values())]
    checks = {
        "cross_absolute_0p08": cross_delta is not None and float(cross_delta) <= -0.08,
        "cross_relative_4pct": cross_relative is not None and float(cross_relative) <= -0.04,
        "route_absolute_0p04": route_delta is not None and float(route_delta) <= -0.04,
        "route_relative_2pct": route_relative is not None and float(route_relative) <= -0.02,
        "segments_3_of_5": segments_improved >= 3,
        "keypoints_5_of_8": keypoints_improved >= 5,
        "all_guards": not guard_cells,
    }
    if all(checks.values()):
        label = "A1_ROUTE_SIGNAL_PASS_PENDING_A2"
    elif not checks["cross_absolute_0p08"] or not checks["cross_relative_4pct"]:
        label = "A1_CROSS_TRACK_NO_GAIN"
    elif not checks["route_absolute_0p04"] or not checks["route_relative_2pct"]:
        label = "A1_ROUTE_HEALTH_NO_GAIN"
    elif not checks["segments_3_of_5"] or not checks["keypoints_5_of_8"]:
        label = "A1_SEGMENT_OR_KEYPOINT_NO_GAIN"
    else:
        label = "A1_GUARD_FAIL"
    return {
        "scope": "current10 x both radii",
        "route_delta_m": route_delta,
        "route_relative": route_relative,
        "cross_delta_m": cross_delta,
        "cross_relative": cross_relative,
        "segments_improved": segments_improved,
        "keypoints_improved": keypoints_improved,
        "guard_failure_cells": [cell["scenario"] for cell in guard_cells],
        "segment_deltas_m": segment_deltas,
        "keypoint_deltas_m": keypoint_deltas,
        "checks": checks,
        "label": label,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal175 A1 — same-checkpoint ALOS-COMP vs ALOS-NONE",
        "",
        f"- status: `{payload['status']}`",
        f"- pre-gate label: `{payload['pre_gate']['label']}`",
        "- scope: eval-only, Goal173 `P_RUD_075/seed86201`, dev seeds 20-59, held-out closed.",
        "- negative deltas mean COMP is better than NONE.",
        "",
        "## Per-cell route health",
        "",
        "| pressure | radius | Δ3D RMSE | Δ3D % | Δcross p90 | Δcross % | Δvert p90 | Δterminal vert | segs | keypoints 20-90 | success Δ | timeout/safety Δ | guards |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in payload["cells"]:
        delta = cell["delta"]
        lines.append(
            "| {pressure} | {radius} | {route} | {route_pct} | {cross} | {cross_pct} | {vertical} | {terminal} | {segments}/5 | {keypoints}/8 | {success:+d} | {timeout:+d}/{safety:+d} | {guards} |".format(
                pressure=cell["pressure"],
                radius=cell["radius"],
                route=_fmt(delta["route_e3d_rmse_m"], signed=True),
                route_pct=_fmt(None if delta["route_relative"] is None else 100.0 * delta["route_relative"], digits=1, signed=True),
                cross=_fmt(delta["cross_p90_m"], signed=True),
                cross_pct=_fmt(None if delta["cross_relative"] is None else 100.0 * delta["cross_relative"], digits=1, signed=True),
                vertical=_fmt(delta["vertical_p90_m"], signed=True),
                terminal=_fmt(delta["terminal_vertical_abs_m"], signed=True),
                segments=delta["segments_improved"],
                keypoints=delta["keypoints_improved"],
                success=delta["success"],
                timeout=delta["timeout"],
                safety=delta["safety"],
                guards="PASS" if all(cell["checks"].values()) else "FAIL",
            )
        )
    pre = payload["pre_gate"]
    lines.extend(
        [
            "",
            "## Current10 pre-gate",
            "",
            f"- Δ3D route RMSE: `{_fmt(pre['route_delta_m'], signed=True)} m` (`{_fmt(None if pre['route_relative'] is None else 100.0 * pre['route_relative'], digits=1, signed=True)}%`).",
            f"- Δcross p90: `{_fmt(pre['cross_delta_m'], signed=True)} m` (`{_fmt(None if pre['cross_relative'] is None else 100.0 * pre['cross_relative'], digits=1, signed=True)}%`).",
            f"- segments improved: `{pre['segments_improved']}/5`; keypoints improved: `{pre['keypoints_improved']}/8`.",
            f"- guard failures: `{pre['guard_failure_cells'] or 'none'}`.",
            "",
            "## Checks",
            "",
        ]
    )
    for key, value in pre["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "A1 is eval-only and cannot by itself prove the latent probe or RL trainability; it only tests whether the existing MLP/action interface responds to the ALOS COMP/NONE change.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    manifest = json.loads((root / "configs/a1_manifest.json").read_text(encoding="utf-8"))
    cells = [
        _cell(root, scenario, bootstrap_samples=args.bootstrap_samples)
        for scenario in manifest["scenarios"]
    ]
    pre_gate = _pre_gate(cells)
    payload = {
        "schema_version": "auv_goal175_a1_analysis.v1",
        "goal_id": "Goal175",
        "phase": "A1",
        "status": "A1_COMPLETE_PENDING_USER_RESUME_FOR_A2",
        "root": str(root),
        "cells": cells,
        "pre_gate": pre_gate,
    }
    output_json = args.output_json or root / "a1_summary.json"
    output_md = args.output_md or root / "a1_summary.md"
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "pre_gate": pre_gate}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
