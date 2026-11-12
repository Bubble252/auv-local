#!/usr/bin/env python3
"""Read-only Goal73 command-demand / curvature audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal73_command_demand.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
WINDOW_DELTAS_S = (1.0, 2.0)
BASELINE_FIELDS = ("success", "terminal_eta_capture")

THRESHOLDS = {
    "default": {
        "heading_p90_gap_rad_s": 0.10,
        "pitch_p90_gap_rad_s": 0.05,
        "material_saturation_gap": 0.05,
        "curvature_gap_1_per_m": 0.02,
    },
    "strict125": {
        "heading_p90_gap_rad_s": 0.05,
        "pitch_p90_gap_rad_s": 0.025,
        "material_saturation_gap": 0.025,
        "curvature_gap_1_per_m": 0.01,
    },
}


def _load_module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G71 = _load_module("goal71_analysis_for_goal73", "analyze_pc_goal71_command_rate_limit.py")
G72 = _load_module("goal72_analysis_for_goal73", "analyze_pc_goal72_terminal_geometry.py")

DiagnosticError = G71.DiagnosticError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/traces"))
    parser.add_argument(
        "--limiter-off-root",
        type=Path,
        default=Path("artifacts/goal67_terminal_reachability"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/goal71_command_rate_limit"),
    )
    parser.add_argument("--limiter-off-suffix", default="_dwell50_m025")
    parser.add_argument("--candidate-suffix", default="_dwell50_m025_cmdlimit")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal73_command_demand/analysis/goal73.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal73_command_demand/analysis/goal73.md"),
    )
    return parser.parse_args(argv)


def _read_episodes(
    run_dir: Path, required_fields: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    if required_fields is None:
        return G72._read_episodes(run_dir)
    return G71._read_episodes(run_dir, required_fields)


def _true(row: dict[str, Any]) -> bool:
    return G72._true(row)


def _run_name(scenario: str, key: str, suffix: str) -> str:
    return G72._run_name(scenario, key, suffix)


def _seed_rows(trace: Any, seed: int) -> dict[str, np.ndarray]:
    rows = G71._seed_rows(trace, seed)
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    for field in ("guidance_speed_scale", "guidance_crab_angle_rad", "p_nav_ned"):
        if field not in trace:
            raise DiagnosticError(f"seed {seed}: trace missing field {field}")
        values = np.asarray(trace[field])[selected]
        if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
            raise DiagnosticError(f"seed {seed}: non-finite trace field {field}")
        rows[field] = values
    return rows


def _p90(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.percentile(np.abs(finite), 90))


def _window_metrics(rows: dict[str, np.ndarray], delta_s: float) -> dict[str, float | None]:
    window, _ = G71._window_slice(rows, delta_s)
    times = np.asarray(rows["time_s"][window], dtype=np.float64)
    command = np.asarray(rows["command_physical"][window], dtype=np.float64)
    nu = np.asarray(rows["nu"][window], dtype=np.float64)
    material = np.asarray(rows["material_saturated"][window], dtype=bool)
    speed_scale = np.asarray(rows["guidance_speed_scale"][window], dtype=np.float64)
    crab = np.asarray(rows["guidance_crab_angle_rad"][window], dtype=np.float64)
    if command.shape != (times.size, 3) or nu.shape != (times.size, 6):
        raise DiagnosticError("command/nu window shape mismatch")
    heading_rate = G71._derivative(times, command[:, 0])
    pitch_rate = G71._derivative(times, command[:, 1])
    actual_speed = np.abs(nu[:, 0])
    yaw_rate = np.abs(nu[:, 5])
    curvature = yaw_rate / np.maximum(actual_speed, 0.1)
    return {
        "heading_rate_p90_rad_s": _p90(heading_rate),
        "pitch_rate_p90_rad_s": _p90(pitch_rate),
        "material_saturation_fraction": float(np.mean(material)) if material.size else None,
        "mean_command_speed_mps": float(np.mean(command[:, 2])) if command.size else None,
        "mean_guidance_speed_scale": float(np.mean(speed_scale)) if speed_scale.size else None,
        "mean_abs_curvature_1_per_m": float(np.mean(curvature)) if curvature.size else None,
        "mean_abs_crab_angle_rad": float(np.mean(np.abs(crab))) if crab.size else None,
    }


def _seed_record(
    *,
    scenario: str,
    key: str,
    action_mode: str,
    seed: int,
    group: str,
    on_rows: dict[str, np.ndarray],
    off_rows: dict[str, np.ndarray],
    on_episode: dict[str, Any],
    baseline_episode: dict[str, Any],
    protocol_radius_m: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "scenario": scenario,
        "run_key": key,
        "action_mode": action_mode,
        "seed": seed,
        "group": group,
        "baseline_true": _true(baseline_episode),
        "limiter_on_true": _true(on_episode),
        "limiter_on_success": bool(on_episode["success"]),
        "limiter_on_terminal_eta_capture": bool(on_episode["terminal_eta_capture"]),
        "limiter_on_terminal_kind": int(on_episode["terminal_kind"]),
        "limiter_on_goal_plane_miss": bool(on_episode["goal_plane_miss"]),
        "limiter_on_timeout": bool(on_episode["timeout"]),
        "limiter_on_safety": bool(on_episode["safety"]),
        "terminal_p_nav_distance_m": float(on_episode["terminal_p_nav_distance_m"]),
        "terminal_eta_distance_m": float(on_episode["terminal_eta_distance_m"]),
        "terminal_abs_signed_offset_m": abs(float(on_episode["terminal_signed_goal_offset_m"])),
        "terminal_signed_offset_m": float(on_episode["terminal_signed_goal_offset_m"]),
    }
    eta = np.asarray(on_rows["eta_ned"][-1], dtype=np.float64)
    p_nav = np.asarray(on_rows["p_nav_ned"][-1], dtype=np.float64)
    goal = np.asarray(on_rows["goal_ned"][-1], dtype=np.float64)
    record["terminal_p_nav_eta_error_m"] = float(np.linalg.norm(p_nav - eta))
    record["terminal_gate_distance_m"] = float(np.linalg.norm(p_nav - goal))
    signed = float(on_episode["terminal_signed_goal_offset_m"])
    p_nav_dist = float(on_episode["terminal_p_nav_distance_m"])
    if p_nav_dist <= protocol_radius_m:
        record["terminal_capture_mode"] = "sphere"
    elif signed >= 0.0:
        record["terminal_capture_mode"] = "plane"
    else:
        record["terminal_capture_mode"] = "unknown"
    for delta in WINDOW_DELTAS_S:
        on_metrics = _window_metrics(on_rows, delta)
        off_metrics = _window_metrics(off_rows, delta)
        tag = f"{delta:g}s"
        for name, value in on_metrics.items():
            record[f"limiter_on_{tag}_{name}"] = value
        for name, value in off_metrics.items():
            record[f"limiter_off_{tag}_{name}"] = value
    return record


def _task(
    task: tuple[str, str, str, str, str, str],
) -> list[dict[str, Any]]:
    scenario, key, baseline_name, off_name, on_name, action_mode = task
    baseline_dir = Path(baseline_name)
    off_dir = Path(off_name)
    on_dir = Path(on_name)
    baseline_rows = {int(row["seed"]): row for row in _read_episodes(baseline_dir, BASELINE_FIELDS)}
    off_rows = {int(row["seed"]): row for row in _read_episodes(off_dir)}
    on_rows = {int(row["seed"]): row for row in _read_episodes(on_dir)}
    summary = json.loads((on_dir / "summary.json").read_text(encoding="utf-8"))
    protocol_radius_m = float(summary["eval_goal_radius_m"])
    records: list[dict[str, Any]] = []
    with (
        np.load(off_dir / "trace.npz", allow_pickle=False) as off_trace,
        np.load(on_dir / "trace.npz", allow_pickle=False) as on_trace,
    ):
        for seed in DEV_SEEDS:
            baseline = baseline_rows[seed]
            off = off_rows[seed]
            on = on_rows[seed]
            baseline_true = _true(baseline)
            off_true = _true(off)
            on_true = _true(on)
            if off_true and not on_true:
                group = "newly_lost"
            elif baseline_true and off_true and on_true:
                group = "both_true"
            elif baseline_true and not off_true:
                group = "original_lost"
            else:
                group = "other"
            records.append(
                _seed_record(
                    scenario=scenario,
                    key=key,
                    action_mode=action_mode,
                    seed=seed,
                    group=group,
                    on_rows=_seed_rows(on_trace, seed),
                    off_rows=_seed_rows(off_trace, seed),
                    on_episode=on,
                    baseline_episode=baseline,
                    protocol_radius_m=protocol_radius_m,
                )
            )
    return records


def _tasks(args: argparse.Namespace) -> list[tuple[str, str, str, str, str, str]]:
    tasks: list[tuple[str, str, str, str, str, str]] = []
    for scenario in SCENARIOS:
        for checkpoint in POLICY_SEEDS:
            key = str(checkpoint)
            tasks.append(
                (
                    scenario,
                    key,
                    str(args.baseline_root / f"{scenario}_policy_s{checkpoint}"),
                    str(args.limiter_off_root / _run_name(scenario, key, args.limiter_off_suffix)),
                    str(args.candidate_root / _run_name(scenario, key, args.candidate_suffix)),
                    "policy",
                )
            )
        tasks.append(
            (
                scenario,
                "zero",
                str(args.baseline_root / f"{scenario}_zero"),
                str(args.limiter_off_root / _run_name(scenario, "zero", args.limiter_off_suffix)),
                str(args.candidate_root / _run_name(scenario, "zero", args.candidate_suffix)),
                "zero-residual",
            )
        )
    return tasks


def _metric_median(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None and math.isfinite(float(record[field]))
    ]
    return None if not values else float(np.median(values))


def _metric_gap(
    newly: list[dict[str, Any]], both: list[dict[str, Any]], field: str
) -> float | None:
    a = _metric_median(newly, field)
    b = _metric_median(both, field)
    if a is None or b is None:
        return None
    return a - b


def _decide_h73a(
    scenario: str,
    gaps: dict[str, float | None],
    n_newly: int,
) -> dict[str, Any]:
    threshold = THRESHOLDS[scenario]
    checks = {
        "heading": gaps.get("heading_rate_p90_rad_s") is not None
        and float(gaps["heading_rate_p90_rad_s"]) >= threshold["heading_p90_gap_rad_s"],
        "pitch": gaps.get("pitch_rate_p90_rad_s") is not None
        and float(gaps["pitch_rate_p90_rad_s"]) >= threshold["pitch_p90_gap_rad_s"],
        "saturation": gaps.get("material_saturation_fraction") is not None
        and float(gaps["material_saturation_fraction"]) >= threshold["material_saturation_gap"],
        "curvature": gaps.get("mean_abs_curvature_1_per_m") is not None
        and float(gaps["mean_abs_curvature_1_per_m"]) >= threshold["curvature_gap_1_per_m"],
    }
    supported = n_newly >= 3 and any(checks.values())
    return {
        "n_newly_lost": n_newly,
        "checks": checks,
        "supported": supported,
        "thresholds": threshold,
    }


def _aggregate(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    groups = {
        name: [record for record in records if record["group"] == name]
        for name in ("newly_lost", "both_true", "original_lost", "other")
    }
    fields: list[str] = []
    fields.extend(
        [
            "terminal_p_nav_eta_error_m",
            "terminal_p_nav_distance_m",
            "terminal_eta_distance_m",
            "terminal_abs_signed_offset_m",
            "terminal_gate_distance_m",
        ]
    )
    for delta in WINDOW_DELTAS_S:
        tag = f"{delta:g}s"
        fields.extend(
            [
                f"limiter_on_{tag}_heading_rate_p90_rad_s",
                f"limiter_on_{tag}_pitch_rate_p90_rad_s",
                f"limiter_on_{tag}_material_saturation_fraction",
                f"limiter_on_{tag}_mean_command_speed_mps",
                f"limiter_on_{tag}_mean_guidance_speed_scale",
                f"limiter_on_{tag}_mean_abs_curvature_1_per_m",
                f"limiter_on_{tag}_mean_abs_crab_angle_rad",
            ]
        )
    aggregate = {
        name: {
            "n": len(group),
            "metrics": {field: _metric_median(group, field) for field in fields},
        }
        for name, group in groups.items()
    }
    gaps: dict[str, float | None] = {}
    for metric in (
        "terminal_p_nav_eta_error_m",
        "terminal_p_nav_distance_m",
        "terminal_eta_distance_m",
        "terminal_abs_signed_offset_m",
        "terminal_gate_distance_m",
    ):
        gaps[metric] = _metric_gap(groups["newly_lost"], groups["both_true"], metric)
    for delta in WINDOW_DELTAS_S:
        tag = f"{delta:g}s"
        for metric in (
            "heading_rate_p90_rad_s",
            "pitch_rate_p90_rad_s",
            "material_saturation_fraction",
            "mean_command_speed_mps",
            "mean_guidance_speed_scale",
            "mean_abs_curvature_1_per_m",
            "mean_abs_crab_angle_rad",
        ):
            field = f"limiter_on_{tag}_{metric}"
            gaps[f"limiter_on_{tag}_{metric}"] = _metric_gap(
                groups["newly_lost"], groups["both_true"], field
            )
    primary_gaps = {
        "heading_rate_p90_rad_s": gaps["limiter_on_1s_heading_rate_p90_rad_s"],
        "pitch_rate_p90_rad_s": gaps["limiter_on_1s_pitch_rate_p90_rad_s"],
        "material_saturation_fraction": gaps["limiter_on_1s_material_saturation_fraction"],
        "mean_abs_curvature_1_per_m": gaps["limiter_on_1s_mean_abs_curvature_1_per_m"],
    }
    return {
        "groups": aggregate,
        "gaps_newly_lost_minus_both_true": gaps,
        "h73a": _decide_h73a(scenario, primary_gaps, len(groups["newly_lost"])),
        "false_success_cases": [
            {
                "run_key": record["run_key"],
                "action_mode": record["action_mode"],
                "seed": record["seed"],
                "group": record["group"],
                "terminal_capture_mode": record.get("terminal_capture_mode"),
                "terminal_p_nav_distance_m": record.get("terminal_p_nav_distance_m"),
                "terminal_eta_distance_m": record.get("terminal_eta_distance_m"),
                "terminal_abs_signed_offset_m": record.get("terminal_abs_signed_offset_m"),
                "terminal_p_nav_eta_error_m": record.get("terminal_p_nav_eta_error_m"),
                "terminal_gate_distance_m": record.get("terminal_gate_distance_m"),
            }
            for record in records
            if record.get("limiter_on_success")
            and not record.get("limiter_on_terminal_eta_capture")
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal73 Step 1：command-demand / 曲率只读审计",
        "",
        f"状态：`{payload['overall']['status']}`",
        "",
        "## 分组中位数（limiter-on，1 s pre-terminal）",
        "",
        "| scenario | group | n | heading rate P90 (rad/s) | pitch rate P90 (rad/s) | saturation | curvature (1/m) | speed (m/s) | terminal p_nav-eta error (m) | terminal eta dist (m) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, result in payload["scenarios"].items():
        for group in ("newly_lost", "both_true", "original_lost"):
            metrics = result["groups"][group]["metrics"]
            lines.append(
                "| {scenario} | {group} | {n} | {heading} | {pitch} | {sat} | {curv} | {speed} | {err} | {eta} |".format(
                    scenario=scenario,
                    group=group,
                    n=result["groups"][group]["n"],
                    heading=_fmt(metrics.get("limiter_on_1s_heading_rate_p90_rad_s")),
                    pitch=_fmt(metrics.get("limiter_on_1s_pitch_rate_p90_rad_s")),
                    sat=_fmt(metrics.get("limiter_on_1s_material_saturation_fraction")),
                    curv=_fmt(metrics.get("limiter_on_1s_mean_abs_curvature_1_per_m")),
                    speed=_fmt(metrics.get("limiter_on_1s_mean_command_speed_mps")),
                    err=_fmt(metrics.get("terminal_p_nav_eta_error_m")),
                    eta=_fmt(metrics.get("terminal_eta_distance_m")),
                )
            )
    lines.extend(
        [
            "",
            "## False-success 终端明细",
            "",
            "| scenario | run | seed | mode | p_nav dist (m) | eta dist (m) | |signed offset| (m) | p_nav-eta error (m) | gate dist (m) |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, result in payload["scenarios"].items():
        for case in result["false_success_cases"]:
            lines.append(
                "| {scenario} | {run} | {seed} | {mode} | {pnav} | {eta} | {offset} | {err} | {gate} |".format(
                    scenario=scenario,
                    run=case["run_key"],
                    seed=case["seed"],
                    mode=case["terminal_capture_mode"],
                    pnav=_fmt(case["terminal_p_nav_distance_m"]),
                    eta=_fmt(case["terminal_eta_distance_m"]),
                    offset=_fmt(case["terminal_abs_signed_offset_m"]),
                    err=_fmt(case["terminal_p_nav_eta_error_m"]),
                    gate=_fmt(case["terminal_gate_distance_m"]),
                )
            )
    lines.extend(["", "## H73-A 判读", ""])
    for scenario, result in payload["scenarios"].items():
        h = result["h73a"]
        lines.append(
            f"- `{scenario}`：`supported={_fmt(h['supported'])}`，"
            f"newly_lost n={h['n_newly_lost']}，checks={h['checks']}"
        )
    lines.extend(
        [
            "",
            "## 隔离",
            "",
            f"- held-out opened: `{_fmt(payload['overall']['heldout_opened'])}`",
            f"- training started: `{_fmt(payload['overall']['training_started'])}`",
            f"- new rollouts: `{_fmt(payload['overall']['new_rollouts'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    try:
        tasks = _tasks(args)
        if args.jobs == 1:
            task_results = [_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
                task_results = list(pool.map(_task, tasks))
        records = [record for rows in task_results for record in rows]
        scenarios = {
            scenario: _aggregate(
                [record for record in records if record["scenario"] == scenario],
                scenario,
            )
            for scenario in SCENARIOS
        }
        supported = all(result["h73a"]["supported"] for result in scenarios.values())
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": "COMMAND_DEMAND_SUPPORTED"
                if supported
                else "COMMAND_DEMAND_NOT_SUPPORTED",
                "h73a_supported": supported,
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": False,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        args.markdown.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
        return 0
    except (DiagnosticError, FileNotFoundError, KeyError, ValueError) as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": {},
            "overall": {
                "status": "INVALID",
                "error": str(error),
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": False,
            },
        }
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Goal73 command-demand audit: INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
