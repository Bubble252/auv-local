#!/usr/bin/env python3
"""Analyze Goal175 A2 COMP/NONE route health and residual-head contribution."""

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
DEFAULT_ROOT = REPO_ROOT / "artifacts/lowlevel/goal175_alos_comp_mlp_a2_20261114"
SEGMENTS = ("0-20", "20-40", "40-60", "60-80", "80-100")
KEYPOINTS = tuple(f"{value:02d}" for value in range(20, 91, 10))
PAIRS = {
    "P_RUD_075": ("P_RUD_075_COMP", "P_RUD_075_NONE"),
    "P_RUD_025": ("P_RUD_025_COMP", "P_RUD_025_NONE"),
    "P_ALL_075": ("P_ALL_075_COMP", "P_ALL_075_NONE"),
}
ZERO_BC = "zero_bc50"


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


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _terminal_vertical_mean(run_dir: Path) -> float:
    trace = np.load(run_dir / "trace.npz", allow_pickle=False)
    active = (
        np.asarray(trace["active"], dtype=bool)
        if "active" in trace.files
        else np.ones(np.asarray(trace["seed"]).shape, dtype=bool)
    )
    seeds = np.asarray(trace["seed"], dtype=np.int64)
    vertical = np.asarray(trace["path_vertical_error"], dtype=np.float64)
    values: list[float] = []
    for seed in np.unique(seeds[active]):
        rows = np.flatnonzero(active & (seeds == seed))
        if rows.size:
            values.append(abs(float(vertical[rows[-1]])))
    return float(np.mean(values)) if values else math.nan


def _arm_metrics(root: Path, arm: str, scenario_id: str) -> dict[str, Any]:
    run_dir = root / "eval" / arm / scenario_id
    metrics = analyze_run(run_dir, default_length_m=35.0)
    summary_metrics = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get(
        "metrics", {}
    )
    return {
        "route_e3d_rmse_m": metrics["route_e3d_t_rmse_m"],
        "cross_p90_m": metrics["route_cross_abs_p90_m"],
        "vertical_p90_m": metrics["route_vert_abs_p90_m"],
        "terminal_vertical_abs_m": _terminal_vertical_mean(run_dir),
        "success_count": metrics["success_count"],
        "timeout_count": metrics["timeout_count"],
        "safety_count": metrics["safety_count"],
        "speed_max_mps": metrics["speed_max_mps"],
        "overspeed_steps": metrics["overspeed_steps"],
        "residual_action_abs_mean": summary_metrics.get("residual_action_abs_mean"),
        "residual_to_base_l2_ratio_mean": summary_metrics.get("residual_to_base_l2_ratio_mean"),
    }


def _pair_cell(
    root: Path,
    comp_arm: str,
    none_arm: str,
    scenario: dict[str, Any],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    scenario_id = str(scenario["scenario"])
    comp_metrics = analyze_run(root / "eval" / comp_arm / scenario_id, default_length_m=35.0)
    none_metrics = analyze_run(root / "eval" / none_arm / scenario_id, default_length_m=35.0)
    delta = paired_delta(comp_metrics, none_metrics, bootstrap_samples=bootstrap_samples)
    route_delta = float(delta["route_e3d_t_rmse_m"]["mean"])
    cross_delta = float(delta["route_cross_abs_p90_m"]["mean"])
    vertical_delta = float(delta["route_vert_abs_p90_m"]["mean"])
    none_route = float(none_metrics["route_e3d_t_rmse_m"])
    none_cross = float(none_metrics["route_cross_abs_p90_m"])
    comp_terminal_vertical = _terminal_vertical_mean(root / "eval" / comp_arm / scenario_id)
    none_terminal_vertical = _terminal_vertical_mean(root / "eval" / none_arm / scenario_id)
    segment_deltas = {label: delta["segments"][label]["mean"] for label in SEGMENTS}
    keypoint_deltas = {key: delta["keypoints"][key]["mean"] for key in KEYPOINTS}
    segments_improved = sum(
        1 for value in segment_deltas.values() if value is not None and float(value) < 0.0
    )
    keypoints_improved = sum(
        1 for value in keypoint_deltas.values() if value is not None and float(value) < 0.0
    )
    checks = {
        "vertical_p90_guard": vertical_delta <= 0.02,
        "terminal_vertical_guard": comp_terminal_vertical - none_terminal_vertical <= 0.02,
        "success_guard": comp_metrics["success_count"] - none_metrics["success_count"] >= -2,
        "timeout_guard": comp_metrics["timeout_count"] <= none_metrics["timeout_count"],
        "safety_guard": comp_metrics["safety_count"] <= none_metrics["safety_count"],
        "speed_guard": (
            comp_metrics["speed_max_mps"] <= 2.5 + 1e-12 and comp_metrics["overspeed_steps"] == 0
        ),
    }
    return {
        "scenario": scenario_id,
        "pressure": scenario["pressure"],
        "radius": scenario["radius"],
        "comp_arm": comp_arm,
        "none_arm": none_arm,
        "comp": _arm_metrics(root, comp_arm, scenario_id),
        "none": _arm_metrics(root, none_arm, scenario_id),
        "delta": {
            "route_e3d_rmse_m": route_delta,
            "route_relative": route_delta / none_route if none_route > 0.0 else None,
            "cross_p90_m": cross_delta,
            "cross_relative": cross_delta / none_cross if none_cross > 0.0 else None,
            "vertical_p90_m": vertical_delta,
            "terminal_vertical_abs_m": comp_terminal_vertical - none_terminal_vertical,
            "segments_improved": segments_improved,
            "keypoints_improved": keypoints_improved,
            "success": comp_metrics["success_count"] - none_metrics["success_count"],
            "timeout": comp_metrics["timeout_count"] - none_metrics["timeout_count"],
            "safety": comp_metrics["safety_count"] - none_metrics["safety_count"],
        },
        "checks": checks,
        "segment_deltas": segment_deltas,
        "keypoint_deltas": keypoint_deltas,
    }


def _pair_gate(cells: list[dict[str, Any]], *, base_arm: str) -> dict[str, Any]:
    current = [cell for cell in cells if cell["pressure"] == "current10"]
    route_delta = _mean(cell["delta"]["route_e3d_rmse_m"] for cell in current)
    none_route = _mean(cell["none"]["route_e3d_rmse_m"] for cell in current)
    cross_delta = _mean(cell["delta"]["cross_p90_m"] for cell in current)
    none_cross = _mean(cell["none"]["cross_p90_m"] for cell in current)
    vertical_delta = _mean(cell["delta"]["vertical_p90_m"] for cell in current)
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
        label: _mean(cell["segment_deltas"][label] for cell in current) for label in SEGMENTS
    }
    keypoint_deltas = {
        key: _mean(cell["keypoint_deltas"][key] for cell in current) for key in KEYPOINTS
    }
    segments_improved = sum(
        1 for value in segment_deltas.values() if value is not None and float(value) < 0.0
    )
    keypoints_improved = sum(
        1 for value in keypoint_deltas.values() if value is not None and float(value) < 0.0
    )
    guard_cells = [cell for cell in cells if not all(cell["checks"].values())]
    vertical_guard = all(
        cell["checks"]["vertical_p90_guard"] and cell["checks"]["terminal_vertical_guard"]
        for cell in cells
    )
    success_guard = all(
        cell["checks"]["success_guard"]
        and cell["checks"]["timeout_guard"]
        and cell["checks"]["safety_guard"]
        for cell in cells
    )
    speed_guard = all(cell["checks"]["speed_guard"] for cell in cells)
    checks = {
        "cross_absolute_0p08": cross_delta is not None and float(cross_delta) <= -0.08,
        "cross_relative_4pct": cross_relative is not None and float(cross_relative) <= -0.04,
        "route_absolute_0p04": route_delta is not None and float(route_delta) <= -0.04,
        "route_relative_2pct": route_relative is not None and float(route_relative) <= -0.02,
        "segments_3_of_5": segments_improved >= 3,
        "keypoints_5_of_8": keypoints_improved >= 5,
        "vertical_guard": vertical_guard,
        "success_guard": success_guard,
        "speed_guard": speed_guard,
    }
    if all(checks.values()):
        label = "COMP_ROUTE_GAIN"
    elif not checks["cross_absolute_0p08"] or not checks["cross_relative_4pct"]:
        label = "CROSS_TRACK_NO_GAIN"
    elif not checks["route_absolute_0p04"] or not checks["route_relative_2pct"]:
        label = "ROUTE_HEALTH_NO_GAIN"
    elif not checks["segments_3_of_5"] or not checks["keypoints_5_of_8"]:
        label = "SEGMENT_OR_KEYPOINT_NO_GAIN"
    elif not checks["vertical_guard"]:
        label = "VERTICAL_GUARD_FAIL"
    elif not checks["success_guard"]:
        label = "SUCCESS_GUARD_FAIL"
    else:
        label = "SPEED_OR_STABILITY_FAIL"
    return {
        "base_arm": base_arm,
        "scope": "current10 x both radii",
        "route_delta_m": route_delta,
        "route_relative": route_relative,
        "cross_delta_m": cross_delta,
        "cross_relative": cross_relative,
        "vertical_p90_m": vertical_delta,
        "segments_improved": segments_improved,
        "keypoints_improved": keypoints_improved,
        "guard_failure_cells": [cell["scenario"] for cell in guard_cells],
        "segment_deltas_m": segment_deltas,
        "keypoint_deltas_m": keypoint_deltas,
        "checks": checks,
        "label": label,
    }


def _residual_delta(root: Path, scenario: dict[str, Any], policy_arm: str) -> dict[str, Any]:
    scenario_id = str(scenario["scenario"])
    policy = _arm_metrics(root, policy_arm, scenario_id)
    zero = _arm_metrics(root, ZERO_BC, scenario_id)
    return {
        "policy_arm": policy_arm,
        "scenario": scenario_id,
        "radius": scenario["radius"],
        "route_delta_vs_zero_bc_m": policy["route_e3d_rmse_m"] - zero["route_e3d_rmse_m"],
        "cross_delta_vs_zero_bc_m": policy["cross_p90_m"] - zero["cross_p90_m"],
        "vertical_delta_vs_zero_bc_m": policy["vertical_p90_m"] - zero["vertical_p90_m"],
        "success_delta_vs_zero_bc": policy["success_count"] - zero["success_count"],
        "residual_action_abs_mean": policy["residual_action_abs_mean"],
        "residual_to_base_l2_ratio_mean": policy["residual_to_base_l2_ratio_mean"],
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal175 A2 — trained MLP COMP/NONE route matrix",
        "",
        f"- status: `{payload['status']}`",
        f"- gate label: `{payload['gate']['label']}`",
        "- scope: 6 trained MLP arms + 3 zero baselines, dev seeds 20-59, held-out closed.",
        "- negative deltas mean COMP is better than NONE.",
        "",
        "## Pair gates (current10 × both radii)",
        "",
        "| base arm | Δ3D RMSE | Δ3D % | Δcross p90 | Δcross % | Δvert p90 | segs | keypoints | guards | label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for pair in payload["pair_gates"]:
        lines.append(
            "| {base} | {route} | {route_pct} | {cross} | {cross_pct} | {vertical} | {segments}/5 | {keypoints}/8 | {guards} | `{label}` |".format(
                base=pair["base_arm"],
                route=_fmt(pair["route_delta_m"], signed=True),
                route_pct=_fmt(
                    None if pair["route_relative"] is None else 100.0 * pair["route_relative"],
                    digits=1,
                    signed=True,
                ),
                cross=_fmt(pair["cross_delta_m"], signed=True),
                cross_pct=_fmt(
                    None if pair["cross_relative"] is None else 100.0 * pair["cross_relative"],
                    digits=1,
                    signed=True,
                ),
                vertical=_fmt(pair["vertical_p90_m"], signed=True),
                segments=pair["segments_improved"],
                keypoints=pair["keypoints_improved"],
                guards="PASS" if all(pair["checks"].values()) else "FAIL",
                label=pair["label"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-cell current10 route health",
            "",
            "| base arm | radius | Δ3D RMSE | Δ3D % | Δcross p90 | Δcross % | Δvert p90 | Δterminal vert | segs | keypoints | success Δ | timeout/safety Δ | guards |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for pair in payload["pair_cells"]:
        if pair["pressure"] != "current10":
            continue
        delta = pair["delta"]
        lines.append(
            "| {base} | {radius} | {route} | {route_pct} | {cross} | {cross_pct} | {vertical} | {terminal} | {segments}/5 | {keypoints}/8 | {success:+d} | {timeout:+d}/{safety:+d} | {guards} |".format(
                base=pair["base_arm"],
                radius=pair["radius"],
                route=_fmt(delta["route_e3d_rmse_m"], signed=True),
                route_pct=_fmt(
                    None if delta["route_relative"] is None else 100.0 * delta["route_relative"],
                    digits=1,
                    signed=True,
                ),
                cross=_fmt(delta["cross_p90_m"], signed=True),
                cross_pct=_fmt(
                    None if delta["cross_relative"] is None else 100.0 * delta["cross_relative"],
                    digits=1,
                    signed=True,
                ),
                vertical=_fmt(delta["vertical_p90_m"], signed=True),
                terminal=_fmt(delta["terminal_vertical_abs_m"], signed=True),
                segments=delta["segments_improved"],
                keypoints=delta["keypoints_improved"],
                success=delta["success"],
                timeout=delta["timeout"],
                safety=delta["safety"],
                guards="PASS" if all(pair["checks"].values()) else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## ALOS base-only check (zero residual, current10)",
            "",
            "| radius | Δ3D RMSE | Δcross p90 | Δvert p90 | Δterminal vert | success Δ | guards |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cell in payload["zero_alos_cells"]:
        if cell["pressure"] != "current10":
            continue
        delta = cell["delta"]
        lines.append(
            "| {radius} | {route} | {cross} | {vertical} | {terminal} | {success:+d} | {guards} |".format(
                radius=cell["radius"],
                route=_fmt(delta["route_e3d_rmse_m"], signed=True),
                cross=_fmt(delta["cross_p90_m"], signed=True),
                vertical=_fmt(delta["vertical_p90_m"], signed=True),
                terminal=_fmt(delta["terminal_vertical_abs_m"], signed=True),
                success=delta["success"],
                guards="PASS" if all(cell["checks"].values()) else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Residual-head attribution (current10)",
            "",
            "| policy arm | radius | Δ3D vs zero_bc50 | Δcross vs zero_bc50 | Δvert vs zero_bc50 | success Δ | residual abs mean | residual/base L2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["residual_vs_zero_bc"]:
        lines.append(
            "| {arm} | {radius} | {route} | {cross} | {vertical} | {success:+d} | {residual} | {ratio} |".format(
                arm=row["policy_arm"],
                radius=row["radius"],
                route=_fmt(row["route_delta_vs_zero_bc_m"], signed=True),
                cross=_fmt(row["cross_delta_vs_zero_bc_m"], signed=True),
                vertical=_fmt(row["vertical_delta_vs_zero_bc_m"], signed=True),
                success=row["success_delta_vs_zero_bc"],
                residual=_fmt(row["residual_action_abs_mean"], digits=4),
                ratio=_fmt(
                    None
                    if row["residual_to_base_l2_ratio_mean"] is None
                    else 100.0 * row["residual_to_base_l2_ratio_mean"],
                    digits=3,
                )
                + "%"
                if row["residual_to_base_l2_ratio_mean"] is not None
                else "NA",
            )
        )
    lines.extend(
        [
            "",
            "## Gate checks",
            "",
        ]
    )
    for key, value in payload["gate"]["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            f"- `h175c_supported`: `{payload['h175c_supported']}`",
            "",
            "The gate is evaluated on the pre-registered current10 × both-radii slice;",
            "vertical and success are guardrails, not alternate objectives.",
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
    manifest = json.loads((root / "configs/a2_manifest.json").read_text(encoding="utf-8"))
    pair_cells: list[dict[str, Any]] = []
    pair_gates: list[dict[str, Any]] = []
    zero_alos_cells = [
        _pair_cell(
            root,
            "zero_alos_comp",
            "zero_alos_none",
            scenario,
            bootstrap_samples=args.bootstrap_samples,
        )
        for scenario in manifest["eval_scenarios"]
    ]
    for cell in zero_alos_cells:
        cell["base_arm"] = "zero_alos"
    zero_alos_gate = _pair_gate(zero_alos_cells, base_arm="zero_alos")
    for base_arm, (comp_arm, none_arm) in PAIRS.items():
        cells = [
            _pair_cell(
                root,
                comp_arm,
                none_arm,
                scenario,
                bootstrap_samples=args.bootstrap_samples,
            )
            for scenario in manifest["eval_scenarios"]
        ]
        for cell in cells:
            cell["base_arm"] = base_arm
        pair_cells.extend(cells)
        pair_gates.append(_pair_gate(cells, base_arm=base_arm))
    pair_by_base = {row["base_arm"]: row for row in pair_gates}
    gate = dict(pair_by_base["P_RUD_075"])
    gate["label"] = gate["label"]
    h175c_supported = (
        gate["label"] == "COMP_ROUTE_GAIN"
        and pair_by_base["P_ALL_075"]["label"] != "COMP_ROUTE_GAIN"
    )
    residual_vs_zero_bc = [
        _residual_delta(root, scenario, comp_arm)
        for base_arm, (comp_arm, _none_arm) in PAIRS.items()
        for scenario in manifest["eval_scenarios"]
        if scenario["pressure"] == "current10"
    ]
    payload = {
        "schema_version": "auv_goal175_a2_analysis.v1",
        "goal_id": "Goal175",
        "phase": "A2",
        "status": "PASS" if gate["label"] == "COMP_ROUTE_GAIN" else "FAIL",
        "root": str(root),
        "pair_cells": pair_cells,
        "pair_gates": pair_gates,
        "zero_alos_cells": zero_alos_cells,
        "zero_alos_gate": zero_alos_gate,
        "gate": gate,
        "h175c_supported": h175c_supported,
        "residual_vs_zero_bc": residual_vs_zero_bc,
    }
    output_json = args.output_json or root / "a2_summary.json"
    output_md = args.output_md or root / "a2_summary.md"
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gate": gate,
                "h175c_supported": h175c_supported,
                "pair_gates": pair_gates,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
