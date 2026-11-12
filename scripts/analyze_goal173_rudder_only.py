#!/usr/bin/env python3
"""Analyze Goal173 with the complete 88-cell evaluation contract."""

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
from audit_goal173_artifacts import _load_audit  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "lowlevel" / "goal173" / "goal173_manifest.json"
SEGMENTS = ("0-20", "20-40", "40-60", "60-80", "80-100")
KEYPOINTS_20_90 = tuple(f"{value:02d}" for value in range(20, 91, 10))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--zero-arm", default="zero_bc50")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _policy_dir(root: Path, arm: str, radius: str, level: str, seed: int) -> Path:
    return root / "eval" / arm / radius / level / f"policy_seed{seed}"


def _zero_dir(root: Path, zero_arm: str, radius: str, level: str) -> Path:
    return root / "eval" / zero_arm / radius / level / "zero_seed86201"


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


def _cell_row(
    policy: dict[str, Any],
    delta: dict[str, Any],
    zero: dict[str, Any],
) -> dict[str, Any]:
    route = delta["route_e3d_t_rmse_m"]
    cross = delta["route_cross_abs_p90_m"]
    vertical = delta["route_vert_abs_p90_m"]
    preterminal = delta["route_preterminal_e3d_rmse_m"]
    completion = delta["completion_fraction"]
    keypoint_improved = sum(
        1
        for key in KEYPOINTS_20_90
        if delta["keypoints"][key]["mean"] is not None
        and float(delta["keypoints"][key]["mean"]) < 0.0
    )
    segment_improved = sum(
        1
        for label in SEGMENTS
        if delta["segments"][label]["mean"] is not None
        and float(delta["segments"][label]["mean"]) < 0.0
    )
    worst_segment = max(
        (
            float(delta["segments"][label]["mean"])
            for label in SEGMENTS
            if delta["segments"][label]["mean"] is not None
        ),
        default=None,
    )
    zero_route = _mean([zero["route_e3d_t_rmse_m"]])
    relative = (
        None
        if route.get("mean") is None or zero_route in (None, 0.0)
        else float(route["mean"]) / float(zero_route)
    )
    checks = {
        "route_absolute_0p05": route.get("mean") is not None
        and float(route["mean"]) <= -0.05,
        "route_relative_5pct": relative is not None and relative <= -0.05,
        "segments_3_of_5": segment_improved >= 3,
        "segments_worst_le_0p15": worst_segment is not None and worst_segment <= 0.15,
        "keypoints_5_of_8": keypoint_improved >= 5,
        "preterminal_0p03": preterminal.get("mean") is not None
        and float(preterminal["mean"]) <= -0.03,
        "completion_ge_minus_0p01": completion.get("mean") is not None
        and float(completion["mean"]) >= -0.01,
        "vertical_non_degradation_0p05": vertical.get("mean") is not None
        and float(vertical["mean"]) <= 0.05,
        "speed_max_2p5": policy.get("speed_max_mps") is not None
        and float(policy["speed_max_mps"]) <= 2.5 + 1e-12,
        "overspeed_zero": int(policy.get("overspeed_steps", 0)) == 0,
        "timeout_zero": int(policy.get("timeout_count", 0)) == 0,
        "safety_zero": int(policy.get("safety_count", 0)) == 0,
    }
    return {
        "status": "OK",
        "route_e3d_rmse_m": route.get("mean"),
        "route_relative_delta": relative,
        "route_ci95": [route.get("ci95_low"), route.get("ci95_high")],
        "cross_p90_delta_m": cross.get("mean"),
        "vertical_p90_delta_m": vertical.get("mean"),
        "preterminal_e3d_delta_m": preterminal.get("mean"),
        "completion_delta": completion.get("mean"),
        "segment_improved": segment_improved,
        "segment_worst_delta_m": worst_segment,
        "keypoint_20_90_improved": keypoint_improved,
        "zero_route_e3d_rmse_m": zero_route,
        "candidate_route_e3d_rmse_m": _mean([policy["route_e3d_t_rmse_m"]]),
        "candidate_cross_p90_m": policy.get("route_cross_abs_p90_m"),
        "candidate_vertical_p90_m": policy.get("route_vert_abs_p90_m"),
        "candidate_speed_max_mps": policy.get("speed_max_mps"),
        "candidate_speed_mean_mps": policy.get("speed_mean_mps"),
        "candidate_overspeed_steps": int(policy.get("overspeed_steps", 0)),
        "candidate_timeout_count": int(policy.get("timeout_count", 0)),
        "candidate_safety_count": int(policy.get("safety_count", 0)),
        "candidate_completion_fraction": policy.get("completion_fraction"),
        "candidate_progress_efficiency": policy.get("progress_efficiency"),
        "success_rate_secondary": policy.get("success_rate"),
        "zero_success_rate_secondary": zero.get("success_rate"),
        "checks": checks,
        "cell_pass": all(checks.values()),
        "segment_deltas_m": {
            label: delta["segments"][label]["mean"] for label in SEGMENTS
        },
    }


def _aggregate(cells: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in cells:
        if row.get("status") != "OK":
            continue
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        route = _mean(row["route_e3d_rmse_m"] for row in members)
        zero_route = _mean(row["zero_route_e3d_rmse_m"] for row in members)
        relative = (
            None
            if route is None or zero_route in (None, 0.0)
            else float(route) / float(zero_route)
        )
        result.append(
            {
                **dict(zip(keys, key)),
                "cells": len(members),
                "cells_pass": sum(bool(row["cell_pass"]) for row in members),
                "route_e3d_rmse_m": route,
                "route_relative_delta": relative,
                "cross_p90_delta_m": _mean(
                    row["cross_p90_delta_m"] for row in members
                ),
                "vertical_p90_delta_m": _mean(
                    row["vertical_p90_delta_m"] for row in members
                ),
                "preterminal_e3d_delta_m": _mean(
                    row["preterminal_e3d_delta_m"] for row in members
                ),
                "completion_delta": _mean(row["completion_delta"] for row in members),
                "segment_cells_improved": sum(
                    int(row["segment_improved"]) for row in members
                ),
                "segment_cells": 5 * len(members),
                "keypoint_cells_improved": sum(
                    int(row["keypoint_20_90_improved"]) for row in members
                ),
                "keypoint_cells": 8 * len(members),
                "speed_max_mps": max(
                    (float(row["candidate_speed_max_mps"]) for row in members),
                    default=None,
                ),
                "overspeed_steps": sum(
                    int(row["candidate_overspeed_steps"]) for row in members
                ),
                "timeout_count": sum(
                    int(row["candidate_timeout_count"]) for row in members
                ),
                "safety_count": sum(
                    int(row["candidate_safety_count"]) for row in members
                ),
            }
        )
    return result


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal173 rudder-only residual — full 88-cell analysis",
        "",
        f"- status: `{payload['status']}`",
        f"- eval cells: `{payload['eval_audit']['present']}/{payload['eval_audit']['expected']}`",
        f"- baseline: `{payload['zero_arm']}`",
        "- 主指标：整条路线 3D/cross/vertical、五段、20–90% keypoints、completion/progress；success 仅二级诊断。",
        "",
        "## Policy cells",
        "",
        "| arm | radius | level | Δ3D RMSE | relative | cross Δp90 | vert Δp90 | pre-term Δ | segs | keypoints 20–90 | speed max | over | timeout | completion Δ | success* | cell pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["cells"]:
        if row.get("status") != "OK":
            continue
        lines.append(
            "| {arm} | {radius} | {level} | {route} | {rel} | {cross} | {vert} | "
            "{pre} | {segs}/5 | {keys}/8 | {speed} | {over} | {timeout} | {comp} | "
            "{succ} | {passed} |".format(
                arm=row["arm"],
                radius=row["radius"],
                level=row["level"],
                route=_fmt(row["route_e3d_rmse_m"], signed=True),
                rel=(
                    "NA"
                    if row["route_relative_delta"] is None
                    else f"{100.0 * row['route_relative_delta']:+.2f}%"
                ),
                cross=_fmt(row["cross_p90_delta_m"], signed=True),
                vert=_fmt(row["vertical_p90_delta_m"], signed=True),
                pre=_fmt(row["preterminal_e3d_delta_m"], signed=True),
                segs=row["segment_improved"],
                keys=row["keypoint_20_90_improved"],
                speed=_fmt(row["candidate_speed_max_mps"]),
                over=row["candidate_overspeed_steps"],
                timeout=row["candidate_timeout_count"],
                comp=_fmt(row["completion_delta"], signed=True),
                succ=_fmt(row["success_rate_secondary"]),
                passed="PASS" if row["cell_pass"] else "FAIL",
            )
        )
    lines += [
        "",
        "## Arm aggregate",
        "",
        "| arm | radius | cells pass | Δ3D RMSE | relative | cross Δp90 | vert Δp90 | seg cells | keypoint cells | speed max | over | timeout |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["arm_radius_aggregate"]:
        lines.append(
            "| {arm} | {radius} | {passed}/{cells} | {route} | {rel} | {cross} | "
            "{vert} | {segs}/{segcells} | {keys}/{keycells} | {speed} | {over} | {timeout} |".format(
                arm=row["arm"],
                radius=row["radius"],
                passed=row["cells_pass"],
                cells=row["cells"],
                route=_fmt(row["route_e3d_rmse_m"], signed=True),
                rel=(
                    "NA"
                    if row["route_relative_delta"] is None
                    else f"{100.0 * row['route_relative_delta']:+.2f}%"
                ),
                cross=_fmt(row["cross_p90_delta_m"], signed=True),
                vert=_fmt(row["vertical_p90_delta_m"], signed=True),
                segs=row["segment_cells_improved"],
                segcells=row["segment_cells"],
                keys=row["keypoint_cells_improved"],
                keycells=row["keypoint_cells"],
                speed=_fmt(row["speed_max_mps"]),
                over=row["overspeed_steps"],
                timeout=row["timeout_count"],
            )
        )
    lines += ["", "## Gate173 checks", ""]
    for arm, checks in payload["gate173"].items():
        lines.append(f"### {arm}")
        for name, value in checks.items():
            lines.append(f"- `{name}`: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    manifest = _manifest()
    protocol = manifest["protocol"]
    arms = tuple(str(item["arm"]) for item in manifest["arms"])
    levels = tuple(str(value) for value in protocol["levels"])
    radii = tuple(str(value) for value in protocol["radii"])
    seeds = (int(manifest["train_seed"]),)

    audit = _load_audit()
    errors = audit.audit_eval(
        root=root,
        seeds=seeds,
        seed_start=int(protocol["dev_seed_start"]),
        seed_count=int(protocol["dev_seed_count"]),
    )
    expected = int(protocol["expected_total_eval_runs"])
    present = expected - len([error for error in errors if "missing" in error.lower()])
    if errors:
        payload = {
            "status": "INVALID_INCOMPLETE_EVAL",
            "zero_arm": args.zero_arm,
            "eval_audit": {"expected": expected, "present": present, "errors": errors},
        }
        output = args.output or (root / "analysis" / "goal173_rudder_only.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    cells: list[dict[str, Any]] = []
    for arm in arms:
        for radius in radii:
            for level in levels:
                for seed in seeds:
                    policy_dir = _policy_dir(root, arm, radius, level, seed)
                    zero_dir = _zero_dir(root, args.zero_arm, radius, level)
                    policy = analyze_run(policy_dir)
                    zero = analyze_run(zero_dir)
                    delta = paired_delta(
                        policy, zero, bootstrap_samples=args.bootstrap_samples
                    )
                    row = _cell_row(policy, delta, zero)
                    row.update(
                        {
                            "arm": arm,
                            "radius": radius,
                            "level": level,
                            "checkpoint_seed": seed,
                        }
                    )
                    cells.append(row)

    arm_radius = _aggregate(cells, ("arm", "radius"))
    arm_level = _aggregate(cells, ("arm", "level"))
    arm_all = _aggregate(cells, ("arm",))
    by_key = {(row["arm"], row["radius"], row["level"]): row for row in cells}
    gate173: dict[str, Any] = {}
    for arm in arms:
        current10 = [
            by_key[(arm, radius, "current10")] for radius in radii
        ]
        screen_pass = any(row["cell_pass"] for row in current10)
        route_cells = [
            bool(by_key[(arm, radius, level)]["checks"]["route_absolute_0p05"])
            for radius in radii
            for level in levels
        ]
        gate173[arm] = {
            "screen_current10_any_radius": screen_pass,
            "screen_current10_default": current10[0]["cell_pass"],
            "screen_current10_strict125": current10[1]["cell_pass"],
            "route_positive_cells_6_of_8": sum(route_cells) >= 6,
            "no_overspeed_any_cell": all(
                row["candidate_overspeed_steps"] == 0 for row in cells if row["arm"] == arm
            ),
            "no_timeout_any_cell": all(
                row["candidate_timeout_count"] == 0 for row in cells if row["arm"] == arm
            ),
            "vertical_guard_any_cell": all(
                row["checks"]["vertical_non_degradation_0p05"]
                for row in cells
                if row["arm"] == arm
            ),
        }

    candidate = "P_RUD_075"
    status = (
        "SCREEN_CANDIDATE_PASS_NOT_FORMAL"
        if gate173.get(candidate, {}).get("screen_current10_any_radius")
        else "SCREEN_NO_CANDIDATE"
    )
    payload = {
        "status": status,
        "diagnostic_only": True,
        "zero_arm": args.zero_arm,
        "eval_audit": {"expected": expected, "present": expected, "errors": []},
        "cells": cells,
        "arm_radius_aggregate": arm_radius,
        "arm_level_aggregate": arm_level,
        "arm_aggregate": arm_all,
        "gate173": gate173,
    }
    output = args.output or (root / "analysis" / "goal173_rudder_only.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(payload, output.with_suffix(".md"))
    print(json.dumps({key: payload[key] for key in ("status", "eval_audit", "gate173")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
