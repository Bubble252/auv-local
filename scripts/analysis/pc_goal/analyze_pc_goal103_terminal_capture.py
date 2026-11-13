#!/usr/bin/env python3
"""Read-only audit of Goal100 B20 strict terminal-capture failures."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal103_terminal_capture.v1"
ARMS = ("P_D3", "P_BC15")
EXPECTED_SEEDS = set(range(20, 60))
EXPECTED_FAILURES = 20
SHELL_MARGIN_M = 0.25
AUTHORITY_CLIP_THRESHOLD = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal100_terminal_prior_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal103_terminal_capture_20261111/phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal103_terminal_capture_20261111/phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _goal_radius(summary: dict[str, Any], summary_path: Path) -> float:
    environment = summary.get("environment_config")
    if not isinstance(environment, str):
        raise ValueError(f"{summary_path}: missing environment_config")
    path = Path(environment)
    if not path.is_absolute():
        path = summary_path.parents[4] / path
    config = _read_json(path)
    radius = config.get("plant", {}).get("goal_radius_m")
    if not isinstance(radius, (int, float)) or not math.isfinite(float(radius)):
        raise ValueError(f"{path}: invalid plant.goal_radius_m")
    return float(radius)


def _episode_geometry(
    data: dict[str, np.ndarray], *, seed: int, goal_radius_m: float
) -> dict[str, Any]:
    required = (
        "episode_seed",
        "step",
        "eta",
        "goal_ned_m",
        "goal_tangent_ned",
        "signed_goal_tangent_offset_eta_m",
        "goal_distance_eta_m",
        "goal_distance_pnav_m",
        "endpoint_entry_reached",
        "material_saturated",
        "time_s",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"trace missing fields {missing}")
    episode_seed = np.asarray(data["episode_seed"], dtype=np.int64)
    mask = episode_seed == seed
    if not bool(mask.any()):
        raise ValueError(f"trace has no steps for seed {seed}")
    index = np.flatnonzero(mask)
    offset = np.asarray(
        data["signed_goal_tangent_offset_eta_m"], dtype=np.float64
    )[index]
    crossing_positions = np.flatnonzero(offset >= 0.0)
    crossing = int(index[crossing_positions[0]]) if crossing_positions.size else int(index[-1])
    eta = np.asarray(data["eta"], dtype=np.float64)[crossing, :3]
    goal = np.asarray(data["goal_ned_m"], dtype=np.float64)[crossing]
    tangent = np.asarray(data["goal_tangent_ned"], dtype=np.float64)[crossing]
    radius_vector = eta - goal
    distance = float(np.linalg.norm(radius_vector))
    signed = float(np.dot(radius_vector, tangent))
    perpendicular_sq = max(distance * distance - signed * signed, 0.0)
    perpendicular = math.sqrt(perpendicular_sq)
    perpendicular_vector = radius_vector - signed * tangent
    vertical = abs(float(perpendicular_vector[2]))
    horizontal = float(np.linalg.norm(perpendicular_vector[:2]))
    pnav_distance = float(
        np.asarray(data["goal_distance_pnav_m"], dtype=np.float64)[crossing]
    )
    terminal_mask = mask & (
        np.asarray(data["goal_distance_eta_m"], dtype=np.float64) <= 3.0
    )
    if not bool(terminal_mask.any()):
        terminal_mask = mask
    saturated = np.asarray(data["material_saturated"], dtype=bool)[terminal_mask]
    residual = np.asarray(data["actuator_residual_applied"], dtype=np.float64)[
        terminal_mask
    ]
    residual_l2 = np.linalg.norm(residual, axis=1)
    return {
        "seed": seed,
        "plane_crossing_step": int(np.asarray(data["step"], dtype=np.int64)[crossing]),
        "plane_crossing_time_s": float(np.asarray(data["time_s"])[crossing]),
        "endpoint_entry_reached": bool(
            np.asarray(data["endpoint_entry_reached"], dtype=bool)[crossing]
        ),
        "miss_distance_at_plane_m": distance,
        "radial_excess_m": distance - goal_radius_m,
        "signed_tangent_offset_m": signed,
        "perpendicular_miss_m": perpendicular,
        "vertical_miss_m": vertical,
        "horizontal_miss_m": horizontal,
        "pnav_plane_distance_m": pnav_distance,
        "pnav_eta_gap_m": pnav_distance - distance,
        "terminal_material_saturated_fraction": float(np.mean(saturated)),
        "terminal_residual_l2_mean": float(np.mean(residual_l2)),
        "terminal_residual_l2_max": float(np.max(residual_l2)),
        "terminal_steps": int(terminal_mask.sum()),
    }


def _classify(failures: list[dict[str, Any]]) -> dict[str, Any]:
    shell = [
        row
        for row in failures
        if row["endpoint_entry_reached"] and row["radial_excess_m"] <= SHELL_MARGIN_M
    ]
    authority = [
        row
        for row in failures
        if max(
            float(row["magnitude_clip_rate"]), float(row["rate_clip_rate"])
        )
        >= AUTHORITY_CLIP_THRESHOLD
        and row["radial_excess_m"] > SHELL_MARGIN_M
    ]
    nav_eta = [
        row
        for row in failures
        if bool(row["goal_capture_pnav"]) and not bool(row["goal_capture_eta"])
    ]
    if len(shell) >= 10:
        status = "TERMINAL_SHELL_DOMINANT"
    elif len(authority) >= 10:
        status = "CONTROL_AUTHORITY_LIMITED"
    elif len(nav_eta) >= 10:
        status = "NAV_ETA_DISAGREEMENT"
    else:
        status = "MIXED"
    return {
        "status": status,
        "shell_failures": len(shell),
        "authority_failures": len(authority),
        "nav_eta_failures": len(nav_eta),
        "shell_seeds": sorted(row["seed"] for row in shell),
        "authority_seeds": sorted(row["seed"] for row in authority),
        "nav_eta_seeds": sorted(row["seed"] for row in nav_eta),
    }


def _audit_arm(root: Path, arm: str) -> dict[str, Any]:
    run = root / f"{arm}_strict125_B20"
    summary_path = run / "summary.json"
    trace_path = run / "trace.npz"
    if not summary_path.is_file() or not trace_path.is_file():
        raise ValueError(f"{run}: missing summary.json or trace.npz")
    summary = _read_json(summary_path)
    if summary.get("action_mode") != "frozen-base":
        raise ValueError(f"{summary_path}: action_mode must be frozen-base")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise ValueError(f"{summary_path}: expected 40 dev episodes")
    seeds = {int(row["seed"]) for row in episodes}
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{summary_path}: seeds do not cover 20-59")
    failures = [row for row in episodes if not bool(row["success"])]
    if len(failures) != EXPECTED_FAILURES:
        raise ValueError(
            f"{summary_path}: expected {EXPECTED_FAILURES} failures, got {len(failures)}"
        )
    goal_radius_m = _goal_radius(summary, summary_path)
    with np.load(trace_path) as payload:
        data = {name: payload[name] for name in payload.files}
    rows: list[dict[str, Any]] = []
    for episode in failures:
        seed = int(episode["seed"])
        row = _episode_geometry(data, seed=seed, goal_radius_m=goal_radius_m)
        row.update(
            {
                "final_goal_distance_m": float(episode.get("final_goal_distance_m", math.nan)),
                "p_nav_done_distance_m": float(
                    episode.get("p_nav_done_distance_m", math.nan)
                ),
                "goal_capture_pnav": bool(episode.get("goal_capture_pnav", False)),
                "goal_capture_eta": bool(episode.get("goal_capture_eta", False)),
                "goal_plane_miss": bool(episode.get("goal_plane_miss", False)),
                "magnitude_clip_rate": float(
                    episode.get("actuator_residual_magnitude_clip_rate", math.nan)
                ),
                "rate_clip_rate": float(
                    episode.get("actuator_residual_rate_clip_rate", math.nan)
                ),
                "guidance_saturation_emergency_rate": float(
                    episode.get("guidance_saturation_emergency_rate", math.nan)
                ),
                "mean_abs_yaw_error_rad": float(
                    episode.get("mean_abs_yaw_error_rad", math.nan)
                ),
                "mean_abs_pitch_error_rad": float(
                    episode.get("mean_abs_pitch_error_rad", math.nan)
                ),
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: row["seed"])
    classification = _classify(rows)
    return {
        "arm": arm,
        "goal_radius_m": goal_radius_m,
        "failure_count": len(rows),
        "failures": rows,
        "classification": classification,
        "counterfactual_success": {
            "actual": 20,
            "goal_radius_1_50": 20 + sum(row["miss_distance_at_plane_m"] <= 1.50 for row in rows),
            "goal_radius_1_75": 20 + sum(row["miss_distance_at_plane_m"] <= 1.75 for row in rows),
            "goal_radius_2_00": 20 + sum(row["miss_distance_at_plane_m"] <= 2.00 for row in rows),
        },
        "excess_quantiles_m": {
            "min": float(np.min([row["radial_excess_m"] for row in rows])),
            "median": float(np.median([row["radial_excess_m"] for row in rows])),
            "max": float(np.max([row["radial_excess_m"] for row in rows])),
            "p90": float(np.quantile([row["radial_excess_m"] for row in rows], 0.9)),
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Goal103 B20 strict terminal-capture audit", ""]
    for arm in ARMS:
        result = payload["arms"][arm]
        classification = result["classification"]
        lines.extend(
            [
                f"## {arm}",
                "",
                f"- classification: `{classification['status']}`",
                f"- goal radius: `{result['goal_radius_m']:.3f} m`",
                f"- shell / authority / nav-eta: "
                f"`{classification['shell_failures']}` / "
                f"`{classification['authority_failures']}` / "
                f"`{classification['nav_eta_failures']}`",
                f"- radial excess min/median/p90/max: "
                f"`{result['excess_quantiles_m']['min']:.3f}` / "
                f"`{result['excess_quantiles_m']['median']:.3f}` / "
                f"`{result['excess_quantiles_m']['p90']:.3f}` / "
                f"`{result['excess_quantiles_m']['max']:.3f}` m",
                f"- counterfactual success: `{result['counterfactual_success']}`",
                "",
                "| seed | miss | excess | vert | horiz | clip mag/rate | pnav gap |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["failures"]:
            lines.append(
                f"| {row['seed']} | {row['miss_distance_at_plane_m']:.3f} | "
                f"{row['radial_excess_m']:.3f} | {row['vertical_miss_m']:.3f} | "
                f"{row['horizontal_miss_m']:.3f} | "
                f"{row['magnitude_clip_rate']:.2f}/{row['rate_clip_rate']:.2f} | "
                f"{row['pnav_eta_gap_m']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    arms = {arm: _audit_arm(args.root, arm) for arm in ARMS}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seeds": sorted(EXPECTED_SEEDS),
        "goal_radius_m": next(iter(arms.values()))["goal_radius_m"],
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(
        {
            "status": {arm: arms[arm]["classification"]["status"] for arm in ARMS},
            "counterfactual_success": {
                arm: arms[arm]["counterfactual_success"] for arm in ARMS
            },
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
