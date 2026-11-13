#!/usr/bin/env python3
"""Read-only terminal-window analysis for Goal28 traces.

This script consumes ``evaluate_pc_ppo.py --trace-output`` artifacts.  It does
not load a policy or step the environment.  Each trace is grouped by its
episode seed, and the terminal state is decomposed into longitudinal,
horizontal-normal and vertical components relative to the goal tangent.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = "auv_pc_goal28_trace_analysis.v1"
EXPECTED_SEEDS = tuple(range(20, 60))
ARMS = ("d3", "g26_p2")
SCENARIOS = ("default", "strict125")
TERMINAL_WINDOW_S = 2.0
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}


@dataclass(frozen=True)
class TerminalRecord:
    arm: str
    scenario: str
    seed: int
    outcome: str
    success_protocol: bool
    success_eta: bool
    pnav_only_success: bool
    terminal_kind: int
    final_time_s: float
    pnav_distance_m: float
    eta_distance_m: float
    longitudinal_pnav_m: float
    horizontal_normal_pnav_m: float
    vertical_pnav_m: float
    longitudinal_eta_m: float
    horizontal_normal_eta_m: float
    vertical_eta_m: float
    nav_error_m: float
    terminal_speed_ratio: float
    terminal_yaw_error_abs_mean_rad: float
    terminal_pitch_error_abs_mean_rad: float
    terminal_material_saturation_rate: float
    terminal_rudder_abs_mean_rad: float
    terminal_stern_abs_mean_rad: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="ARM:SCENARIO=DIR",
        help="repeat for each trace directory containing trace.npz and summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_inputs(values: Sequence[str]) -> dict[tuple[str, str], Path]:
    inputs: dict[tuple[str, str], Path] = {}
    for value in values:
        try:
            key, path_text = value.split("=", 1)
            arm, scenario = key.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid --trace {value!r}; expected ARM:SCENARIO=DIR"
            ) from exc
        if arm not in ARMS or scenario not in SCENARIOS:
            raise ValueError(f"unsupported trace key {arm}:{scenario}")
        if (arm, scenario) in inputs:
            raise ValueError(f"duplicate trace for {arm}:{scenario}")
        inputs[(arm, scenario)] = Path(path_text)
    missing = [
        f"{arm}:{scenario}"
        for arm in ARMS
        for scenario in SCENARIOS
        if (arm, scenario) not in inputs
    ]
    if missing:
        raise ValueError(f"missing traces: {', '.join(missing)}")
    return inputs


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "auv_pc_ppo_evaluation.v1":
        raise ValueError(f"{path}: unexpected summary schema")
    if tuple(payload.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError(f"{path}: expected dev seeds 20-59")
    if payload.get("endpoint_stop_enabled") is not True:
        raise ValueError(f"{path}: endpoint_stop is required")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(EXPECTED_SEEDS):
        raise ValueError(f"{path}: expected 40 episode records")
    if tuple(int(item.get("seed", -1)) for item in episodes) != EXPECTED_SEEDS:
        raise ValueError(f"{path}: episode seeds do not match 20-59")
    return payload


def _outcome(item: dict[str, Any]) -> str:
    flags = {
        "success": bool(item["success"]),
        "plane_miss": bool(item["goal_plane_miss"]),
        "timeout": bool(item["timeout"]),
        "safety": bool(item["safety"]),
    }
    active = [name for name, value in flags.items() if value]
    if len(active) != 1:
        raise ValueError(f"seed {item.get('seed')}: non-exclusive outcomes {active}")
    return active[0]


def _decompose(offset: np.ndarray, tangent: np.ndarray) -> tuple[float, float, float]:
    longitudinal = float(np.dot(offset, tangent))
    normal = offset - longitudinal * tangent
    horizontal = float(math.hypot(float(normal[0]), float(normal[1])))
    vertical = float(normal[2])
    return longitudinal, horizontal, vertical


def _term_stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _records_for(
    arm: str, scenario: str, directory: Path
) -> list[TerminalRecord]:
    trace_path = directory / "trace.npz"
    summary_path = directory / "summary.json"
    summary = _load_summary(summary_path)
    data = np.load(trace_path, allow_pickle=False)
    required = {
        "time_s",
        "episode_seed",
        "step",
        "eta",
        "p_nav",
        "goal_ned_m",
        "goal_tangent_ned",
        "goal_distance_pnav_m",
        "goal_distance_eta_m",
        "signed_goal_tangent_offset_pnav_m",
        "signed_goal_tangent_offset_eta_m",
        "terminal_kind",
        "guidance_terminal_speed_ratio",
        "heading_error_rad",
        "pitch_error_rad",
        "material_saturated",
        "actual_actuator",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{trace_path}: missing trace fields {missing}")

    summary_by_seed = {int(item["seed"]): item for item in summary["episodes"]}
    seeds = data["episode_seed"].astype(np.int64)
    records: list[TerminalRecord] = []
    for seed in EXPECTED_SEEDS:
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{trace_path}: no trace rows for seed {seed}")
        last = int(indices[-1])
        item = summary_by_seed[seed]
        outcome = _outcome(item)
        terminal_kind = int(data["terminal_kind"][last])
        expected_kind = {
            "success": 1,
            "timeout": 2,
            "plane_miss": 9,
        }.get(outcome)
        if expected_kind is not None and terminal_kind != expected_kind:
            raise ValueError(
                f"{trace_path}: seed {seed} terminal kind {terminal_kind} "
                f"does not match outcome {outcome}"
            )

        goal = np.asarray(data["goal_ned_m"][last], dtype=np.float64)
        tangent = np.asarray(data["goal_tangent_ned"][last], dtype=np.float64)
        p_nav = np.asarray(data["p_nav"][last], dtype=np.float64)
        eta = np.asarray(data["eta"][last, :3], dtype=np.float64)
        if not (np.all(np.isfinite(goal)) and np.all(np.isfinite(tangent))):
            raise ValueError(f"{trace_path}: seed {seed} has non-finite goal geometry")
        if not (np.all(np.isfinite(p_nav)) and np.all(np.isfinite(eta))):
            raise ValueError(f"{trace_path}: seed {seed} has non-finite terminal state")
        longitudinal_pnav, horizontal_pnav, vertical_pnav = _decompose(
            p_nav - goal, tangent
        )
        longitudinal_eta, horizontal_eta, vertical_eta = _decompose(eta - goal, tangent)

        last_time = float(data["time_s"][last])
        window = indices[data["time_s"][indices] >= last_time - TERMINAL_WINDOW_S]
        if window.size == 0:
            raise ValueError(f"{trace_path}: seed {seed} has an empty terminal window")
        yaw = np.abs(np.asarray(data["heading_error_rad"][window], dtype=np.float64))
        pitch = np.abs(np.asarray(data["pitch_error_rad"][window], dtype=np.float64))
        saturation = np.asarray(
            data["material_saturated"][window], dtype=np.float64
        )
        speed_ratio = np.asarray(
            data["guidance_terminal_speed_ratio"][window], dtype=np.float64
        )
        actuator = np.asarray(data["actual_actuator"][window], dtype=np.float64)
        if not (
            np.all(np.isfinite(yaw))
            and np.all(np.isfinite(pitch))
            and np.all(np.isfinite(saturation))
            and np.all(np.isfinite(speed_ratio))
            and np.all(np.isfinite(actuator))
        ):
            raise ValueError(f"{trace_path}: seed {seed} has non-finite terminal window")

        records.append(
            TerminalRecord(
                arm=arm,
                scenario=scenario,
                seed=seed,
                outcome=outcome,
                success_protocol=bool(item["success"]),
                success_eta=bool(item.get("goal_capture_eta", False)),
                pnav_only_success=bool(item["success"])
                and not bool(item.get("goal_capture_eta", False)),
                terminal_kind=terminal_kind,
                final_time_s=last_time,
                pnav_distance_m=float(
                    np.linalg.norm(p_nav - goal)
                ),
                eta_distance_m=float(np.linalg.norm(eta - goal)),
                longitudinal_pnav_m=longitudinal_pnav,
                horizontal_normal_pnav_m=horizontal_pnav,
                vertical_pnav_m=vertical_pnav,
                longitudinal_eta_m=longitudinal_eta,
                horizontal_normal_eta_m=horizontal_eta,
                vertical_eta_m=vertical_eta,
                nav_error_m=float(np.linalg.norm(p_nav - eta)),
                terminal_speed_ratio=float(np.mean(speed_ratio)),
                terminal_yaw_error_abs_mean_rad=float(np.mean(yaw)),
                terminal_pitch_error_abs_mean_rad=float(np.mean(pitch)),
                terminal_material_saturation_rate=float(np.mean(saturation > 0.5)),
                terminal_rudder_abs_mean_rad=float(np.mean(np.abs(actuator[:, 0]))),
                terminal_stern_abs_mean_rad=float(np.mean(np.abs(actuator[:, 1]))),
            )
        )
    return records


def _aggregate(records: list[TerminalRecord], goal_radius_m: float) -> dict[str, Any]:
    groups: dict[str, list[TerminalRecord]] = {}
    for record in records:
        groups.setdefault(record.outcome, []).append(record)
    success_eta = [record for record in records if record.success_eta]
    pnav_only = [record for record in records if record.pnav_only_success]
    plane_miss = groups.get("plane_miss", [])

    def group(rows: list[TerminalRecord]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "success_protocol_rate": (
                None if not rows else float(np.mean([row.success_protocol for row in rows]))
            ),
            "success_eta_rate": (
                None if not rows else float(np.mean([row.success_eta for row in rows]))
            ),
            "pnav_distance_m": _term_stats(row.pnav_distance_m for row in rows),
            "eta_distance_m": _term_stats(row.eta_distance_m for row in rows),
            "longitudinal_pnav_m": _term_stats(row.longitudinal_pnav_m for row in rows),
            "horizontal_normal_pnav_m": _term_stats(
                row.horizontal_normal_pnav_m for row in rows
            ),
            "vertical_pnav_m": _term_stats(row.vertical_pnav_m for row in rows),
            "longitudinal_eta_m": _term_stats(row.longitudinal_eta_m for row in rows),
            "horizontal_normal_eta_m": _term_stats(
                row.horizontal_normal_eta_m for row in rows
            ),
            "vertical_eta_m": _term_stats(row.vertical_eta_m for row in rows),
            "nav_error_m": _term_stats(row.nav_error_m for row in rows),
            "terminal_speed_ratio": _term_stats(
                row.terminal_speed_ratio for row in rows
            ),
            "terminal_yaw_error_abs_mean_rad": _term_stats(
                row.terminal_yaw_error_abs_mean_rad for row in rows
            ),
            "terminal_pitch_error_abs_mean_rad": _term_stats(
                row.terminal_pitch_error_abs_mean_rad for row in rows
            ),
            "terminal_material_saturation_rate": _term_stats(
                row.terminal_material_saturation_rate for row in rows
            ),
            "terminal_rudder_abs_mean_rad": _term_stats(
                row.terminal_rudder_abs_mean_rad for row in rows
            ),
            "terminal_stern_abs_mean_rad": _term_stats(
                row.terminal_stern_abs_mean_rad for row in rows
            ),
        }

    miss_horizontal_share = [
        row.horizontal_normal_pnav_m
        / max(row.horizontal_normal_pnav_m + abs(row.vertical_pnav_m), 1e-12)
        for row in plane_miss
    ]
    miss_dist_ratio = [
        row.pnav_distance_m / goal_radius_m for row in plane_miss if goal_radius_m > 0
    ]
    miss_speed_median = (
        float(np.median([row.terminal_speed_ratio for row in plane_miss]))
        if plane_miss
        else None
    )
    success_speed_median = (
        float(np.median([row.terminal_speed_ratio for row in success_eta]))
        if success_eta
        else None
    )
    miss_sat_median = (
        float(
            np.median(
                [row.terminal_material_saturation_rate for row in plane_miss]
            )
        )
        if plane_miss
        else None
    )
    success_sat_median = (
        float(
            np.median(
                [row.terminal_material_saturation_rate for row in success_eta]
            )
        )
        if success_eta
        else None
    )
    return {
        "goal_radius_m": goal_radius_m,
        "groups": {name: group(rows) for name, rows in sorted(groups.items())},
        "pnav_only": group(pnav_only),
        "success_eta": group(success_eta),
        "plane_miss": group(plane_miss),
        "plane_miss_horizontal_share_median": (
            None if not miss_horizontal_share else float(np.median(miss_horizontal_share))
        ),
        "plane_miss_distance_to_radius_median": (
            None if not miss_dist_ratio else float(np.median(miss_dist_ratio))
        ),
        "plane_miss_minus_success_terminal_speed_ratio_median": (
            None
            if miss_speed_median is None or success_speed_median is None
            else miss_speed_median - success_speed_median
        ),
        "plane_miss_minus_success_terminal_saturation_median": (
            None
            if miss_sat_median is None or success_sat_median is None
            else miss_sat_median - success_sat_median
        ),
    }


def analyze(inputs: dict[tuple[str, str], Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "expected_seeds": list(EXPECTED_SEEDS),
        "terminal_window_s": TERMINAL_WINDOW_S,
        "arms": {arm: {} for arm in ARMS},
    }
    for (arm, scenario), directory in sorted(inputs.items()):
        summary = _load_summary(directory / "summary.json")
        goal_radius = GOAL_RADIUS_M[scenario]
        records = _records_for(arm, scenario, directory)
        result["arms"][arm][scenario] = {
            "source": str(directory),
            "summary_sha256": summary.get("summary_sha256"),
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "aggregate": _aggregate(records, goal_radius),
            "records": [record.__dict__ for record in records],
        }
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal28 terminal trace analysis",
        "",
        "Read-only trace decomposition; no training.",
        "",
        "| arm | scenario | outcome | n | pnav dist med | eta dist med | longitudinal pnav med | horizontal-normal med | vertical med | terminal speed ratio med | terminal sat. med |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            aggregate = result["arms"][arm][scenario]["aggregate"]
            for outcome in ("success", "plane_miss", "timeout", "safety"):
                stats = aggregate["groups"].get(outcome)
                if stats is None:
                    continue
                lines.append(
                    "| {arm} | {scenario} | {outcome} | {n} | {pnav} | {eta} | "
                    "{long} | {horiz} | {vert} | {speed} | {sat} |".format(
                        arm=arm,
                        scenario=scenario,
                        outcome=outcome,
                        n=stats["count"],
                        pnav=_fmt(stats["pnav_distance_m"]["median"]),
                        eta=_fmt(stats["eta_distance_m"]["median"]),
                        long=_fmt(stats["longitudinal_pnav_m"]["median"]),
                        horiz=_fmt(stats["horizontal_normal_pnav_m"]["median"]),
                        vert=_fmt(stats["vertical_pnav_m"]["median"]),
                        speed=_fmt(stats["terminal_speed_ratio"]["median"]),
                        sat=_fmt(stats["terminal_material_saturation_rate"]["median"]),
                    )
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    inputs = _parse_inputs(args.trace)
    result = analyze(inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trace_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "trace_analysis.md").write_text(_markdown(result), encoding="utf-8")
    for arm in ARMS:
        for scenario in SCENARIOS:
            aggregate = result["arms"][arm][scenario]["aggregate"]
            print(
                f"{arm}/{scenario}: "
                f"miss_dist_ratio_med={_fmt(aggregate['plane_miss_distance_to_radius_median'])}, "
                f"horizontal_share_med={_fmt(aggregate['plane_miss_horizontal_share_median'])}, "
                f"speed_delta={_fmt(aggregate['plane_miss_minus_success_terminal_speed_ratio_median'])}, "
                f"sat_delta={_fmt(aggregate['plane_miss_minus_success_terminal_saturation_median'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
