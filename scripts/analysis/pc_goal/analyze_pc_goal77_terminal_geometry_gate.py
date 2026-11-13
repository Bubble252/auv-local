#!/usr/bin/env python3
"""Read-only Goal77 terminal geometry and entry-gate audit."""

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

SCHEMA_VERSION = "auv_pc_goal77_terminal_geometry_gate.v1"
LATERAL_FRACTION = 0.75
VERTICAL_FRACTION = 0.50
H77A_FALSE_REDUCTION_MIN = 0.75
H77A_TRUE_RETENTION_MIN = 0.95
H77B_FALSE_REDUCTION_MIN = 0.90
H77B_TRUE_RETENTION_MIN = 0.95
H77B_GAIN_MIN = 0.10
H77B_RETENTION_LOSS_MAX = 0.02
IDENTITY_TOL = 1e-8


def _load_module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G76 = _load_module("goal76_analysis_for_goal77", "analyze_pc_goal76_label_semantics.py")
G73 = G76.G73
DiagnosticError = G76.DiagnosticError


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
        default=Path("artifacts/goal77_terminal_geometry_gate/analysis/goal77.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal77_terminal_geometry_gate/analysis/goal77.md"),
    )
    return parser.parse_args(argv)


def _goal_frame_offsets(delta: np.ndarray, tangent: np.ndarray) -> dict[str, float]:
    delta = np.asarray(delta, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    if delta.shape != (3,) or tangent.shape != (3,):
        raise DiagnosticError("delta and tangent must have shape (3,)")
    tangent_norm = float(np.linalg.norm(tangent))
    if not math.isfinite(tangent_norm) or tangent_norm <= 1e-12:
        raise DiagnosticError("goal tangent is zero or non-finite")
    t_axis = tangent / tangent_norm
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    v_axis = up - float(np.dot(up, t_axis)) * t_axis
    v_norm = float(np.linalg.norm(v_axis))
    if not math.isfinite(v_norm) or v_norm <= 1e-8:
        raise DiagnosticError("goal tangent is parallel to world vertical")
    v_axis = v_axis / v_norm
    h_axis = np.cross(t_axis, v_axis)
    h_norm = float(np.linalg.norm(h_axis))
    if not math.isfinite(h_norm) or h_norm <= 1e-12:
        raise DiagnosticError("failed to construct goal-frame lateral axis")
    h_axis = h_axis / h_norm
    signed = float(np.dot(delta, t_axis))
    vertical = float(np.dot(delta, v_axis))
    lateral = float(np.dot(delta, h_axis))
    reconstruction = signed**2 + vertical**2 + lateral**2
    identity_error = abs(float(np.dot(delta, delta)) - reconstruction)
    if identity_error > IDENTITY_TOL:
        raise DiagnosticError(f"goal-frame norm identity error {identity_error}")
    return {
        "signed_m": signed,
        "vertical_m": vertical,
        "lateral_m": lateral,
        "identity_error": identity_error,
    }


def _gate_flags(
    *, signed: float, vertical: float, lateral: float, tight_radius: float
) -> dict[str, bool]:
    plane = bool(signed >= 0.0)
    lateral_gate = bool(abs(lateral) <= LATERAL_FRACTION * tight_radius + 1e-12)
    vertical_gate = bool(abs(vertical) <= VERTICAL_FRACTION * tight_radius + 1e-12)
    return {
        "G1": plane,
        "G2": lateral_gate,
        "G3": vertical_gate,
        "G4": bool(plane and lateral_gate and vertical_gate),
    }


def _last_trace_vector(trace: Any, seed: int, name: str) -> np.ndarray:
    if name not in trace:
        raise DiagnosticError(f"trace missing field {name}")
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    values = np.asarray(trace[name])[selected]
    if not np.all(np.isfinite(values)):
        raise DiagnosticError(f"seed {seed}: non-finite trace field {name}")
    return np.asarray(values[-1], dtype=np.float64)


def _task(task: tuple[str, str, str, str, str, str]) -> list[dict[str, Any]]:
    scenario, key, _, _, on_name, action_mode = task
    on_dir = Path(on_name)
    episodes = {int(row["seed"]): row for row in G73._read_episodes(on_dir)}
    summary = json.loads((on_dir / "summary.json").read_text(encoding="utf-8"))
    tight_radius, protocol_radius = G76.G75.G74._radii(on_dir, summary)
    records: list[dict[str, Any]] = []
    with np.load(on_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in G73.DEV_SEEDS:
            episode = episodes[seed]
            rows = G73._seed_rows(trace, seed)
            eta = np.asarray(rows["eta_ned"][-1], dtype=np.float64)
            p_nav = np.asarray(rows["p_nav_ned"][-1], dtype=np.float64)
            goal = np.asarray(rows["goal_ned"][-1], dtype=np.float64)
            tangent = _last_trace_vector(trace, seed, "goal_tangent_ned")
            delta = p_nav - goal
            frame = _goal_frame_offsets(delta, tangent)
            signed_trace = float(np.asarray(rows["signed_offset"])[-1])
            if not math.isclose(frame["signed_m"], signed_trace, abs_tol=1e-7):
                raise DiagnosticError(
                    f"{on_dir} seed={seed}: signed offset mismatch "
                    f"{frame['signed_m']} != {signed_trace}"
                )
            d_p = float(np.linalg.norm(delta))
            d_e = float(np.linalg.norm(eta - goal))
            if not math.isclose(
                d_p, float(episode["terminal_p_nav_distance_m"]), abs_tol=1e-6
            ):
                raise DiagnosticError(f"{on_dir} seed={seed}: p_nav distance mismatch")
            if not math.isclose(
                d_e, float(episode["terminal_eta_distance_m"]), abs_tol=1e-6
            ):
                raise DiagnosticError(f"{on_dir} seed={seed}: eta distance mismatch")
            protocol_capture = bool(d_e <= protocol_radius + 1e-12)
            if protocol_capture != bool(episode["terminal_eta_capture"]):
                raise DiagnosticError(
                    f"{on_dir} seed={seed}: terminal_eta_capture mismatch"
                )
            tight_capture = bool(d_e <= tight_radius + 1e-12)
            success_none = bool(d_p <= tight_radius + 1e-12)
            flags = _gate_flags(
                signed=frame["signed_m"],
                vertical=frame["vertical_m"],
                lateral=frame["lateral_m"],
                tight_radius=tight_radius,
            )
            record: dict[str, Any] = {
                "scenario": scenario,
                "run_key": key,
                "action_mode": action_mode,
                "seed": seed,
                "tight_radius_m": tight_radius,
                "protocol_radius_m": protocol_radius,
                "terminal_p_nav_distance_m": d_p,
                "terminal_eta_distance_m": d_e,
                "signed_m": frame["signed_m"],
                "vertical_m": frame["vertical_m"],
                "lateral_m": frame["lateral_m"],
                "identity_error": frame["identity_error"],
                "actual_success": bool(episode["success"]),
                "success_none": success_none,
                "tight_capture": tight_capture,
                "protocol_capture": protocol_capture,
                "mismatch": bool(success_none) != bool(episode["success"]),
            }
            for arm, gate in flags.items():
                success = bool(success_none and gate)
                record[f"success_{arm}"] = success
                record[f"tight_true_{arm}"] = bool(success and tight_capture)
                record[f"tight_false_{arm}"] = bool(success and not tight_capture)
                record[f"protocol_true_{arm}"] = bool(success and protocol_capture)
            records.append(record)
    return records


def _count(records: list[dict[str, Any]], field: str) -> int:
    return int(sum(bool(record[field]) for record in records))


def _arm_stats(records: list[dict[str, Any]], arm: str, base: dict[str, int]) -> dict[str, Any]:
    tight_true = _count(records, f"tight_true_{arm}")
    tight_false = _count(records, f"tight_false_{arm}")
    protocol_true = _count(records, f"protocol_true_{arm}")
    false_reduction = (
        None
        if base["tight_false"] <= 0
        else (base["tight_false"] - tight_false) / base["tight_false"]
    )
    true_retention = (
        None
        if base["tight_true"] <= 0
        else tight_true / base["tight_true"]
    )
    protocol_retention = (
        None
        if base["protocol_true"] <= 0
        else protocol_true / base["protocol_true"]
    )
    return {
        "success": _count(records, f"success_{arm}"),
        "tight_true": tight_true,
        "tight_false": tight_false,
        "protocol_true": protocol_true,
        "tight_false_reduction_fraction": false_reduction,
        "tight_true_retention": true_retention,
        "protocol_true_retention": protocol_retention,
    }


def _median(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None and math.isfinite(float(record[field]))
    ]
    return None if not values else float(np.median(values))


def _h77b_gain_supported(scenarios: dict[str, Any]) -> bool:
    if not scenarios:
        return False
    gains: list[float] = []
    retention_losses: list[float] = []
    for result in scenarios.values():
        g1 = result["arms"]["G1"]
        g4 = result["arms"]["G4"]
        if (
            g1["tight_false_reduction_fraction"] is None
            or g4["tight_false_reduction_fraction"] is None
            or g1["tight_true_retention"] is None
            or g4["tight_true_retention"] is None
        ):
            return False
        gains.append(
            g4["tight_false_reduction_fraction"]
            - g1["tight_false_reduction_fraction"]
        )
        retention_losses.append(g1["tight_true_retention"] - g4["tight_true_retention"])
    return bool(
        any(gain >= H77B_GAIN_MIN for gain in gains)
        and all(
            loss <= H77B_RETENTION_LOSS_MAX + 1e-12 for loss in retention_losses
        )
    )


def _aggregate(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    base = {
        "success": _count(records, "success_none"),
        "tight_true": int(sum(r["success_none"] and r["tight_capture"] for r in records)),
        "tight_false": int(sum(r["success_none"] and not r["tight_capture"] for r in records)),
        "protocol_true": int(
            sum(r["success_none"] and r["protocol_capture"] for r in records)
        ),
    }
    arms = {arm: _arm_stats(records, arm, base) for arm in ("G1", "G2", "G3", "G4")}
    g1 = arms["G1"]
    g4 = arms["G4"]
    h77a_supported = bool(
        g1["tight_false_reduction_fraction"] is not None
        and g1["tight_true_retention"] is not None
        and g1["tight_false_reduction_fraction"] >= H77A_FALSE_REDUCTION_MIN
        and g1["tight_true_retention"] >= H77A_TRUE_RETENTION_MIN
    )
    h77b_supported = bool(
        g4["tight_false_reduction_fraction"] is not None
        and g4["tight_true_retention"] is not None
        and g4["tight_false_reduction_fraction"] >= H77B_FALSE_REDUCTION_MIN
        and g4["tight_true_retention"] >= H77B_TRUE_RETENTION_MIN
    )
    tight_false_records = [r for r in records if r["success_none"] and not r["tight_capture"]]
    return {
        "n": len(records),
        "base": base,
        "arms": arms,
        "gates": {
            "H77-A": {"supported": h77a_supported},
            "H77-B": {"supported": h77b_supported},
        },
        "mismatch": _count(records, "mismatch"),
        "max_identity_error": max(float(r["identity_error"]) for r in records),
        "remaining_false_geometry": {
            "n": len(tight_false_records),
            "signed_m_median": _median(tight_false_records, "signed_m"),
            "lateral_m_median": _median(tight_false_records, "lateral_m"),
            "vertical_m_median": _median(tight_false_records, "vertical_m"),
        },
        "by_checkpoint": {
            key: {
                "n": len([r for r in records if r["run_key"] == key]),
                "tight_false_none": int(
                    sum(r["run_key"] == key and r["success_none"] and not r["tight_capture"] for r in records)
                ),
                "tight_false_g1": int(
                    sum(r["run_key"] == key and r["tight_false_G1"] for r in records)
                ),
                "tight_false_g4": int(
                    sum(r["run_key"] == key and r["tight_false_G4"] for r in records)
                ),
            }
            for key in ("240", "752", "880", "1008", "zero")
        },
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
        "# Goal77：terminal lateral/vertical geometry 与 entry gate",
        "",
        f"状态：`{payload['overall']['status']}`",
        "",
        "## 固定候选 gate",
        "",
        "| scenario | arm | success | tight true | tight false | tight true 保留率 | tight false 降幅 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, result in payload["scenarios"].items():
        for arm in ("G1", "G2", "G3", "G4"):
            stats = result["arms"][arm]
            lines.append(
                f"| {scenario} | {arm} | {stats['success']} | {stats['tight_true']} | "
                f"{stats['tight_false']} | {_fmt(stats['tight_true_retention'])} | "
                f"{_fmt(stats['tight_false_reduction_fraction'])} |"
            )
    lines.extend(
        [
            "",
            "## Gate 判读",
            "",
        ]
    )
    for scenario, result in payload["scenarios"].items():
        lines.append(
            f"- `{scenario}`：H77-A={_fmt(result['gates']['H77-A']['supported'])}，"
            f"H77-B={_fmt(result['gates']['H77-B']['supported'])}，"
            f"mismatch={result['mismatch']}，"
            f"max identity error={result['max_identity_error']:.2e}"
        )
    lines.extend(
        [
            "",
            "## 剩余 tight false 几何（G0）",
            "",
            "| scenario | n | signed | lateral | vertical |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario, result in payload["scenarios"].items():
        geometry = result["remaining_false_geometry"]
        lines.append(
            f"| {scenario} | {geometry['n']} | {_fmt(geometry['signed_m_median'])} | "
            f"{_fmt(geometry['lateral_m_median'])} | {_fmt(geometry['vertical_m_median'])} |"
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
        tasks = G73._tasks(args)
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
            for scenario in G73.SCENARIOS
        }
        h77a = all(result["gates"]["H77-A"]["supported"] for result in scenarios.values())
        h77b = all(result["gates"]["H77-B"]["supported"] for result in scenarios.values())
        h77b_gain = _h77b_gain_supported(scenarios)
        mismatch = sum(result["mismatch"] for result in scenarios.values())
        max_identity_error = max(result["max_identity_error"] for result in scenarios.values())
        if mismatch > 0 or max_identity_error > IDENTITY_TOL:
            status = "INVALID"
        elif h77a:
            status = "PLANE_GATE_SUPPORTED"
        elif h77b and h77b_gain:
            status = "ANISOTROPIC_GATE_SUPPORTED"
        else:
            status = "GEOMETRY_GATE_NOT_SUPPORTED"
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": status,
                "H77-A supported": h77a,
                "H77-B supported": h77b,
                "H77-B gain supported": h77b_gain,
                "mismatch": mismatch,
                "max_identity_error": max_identity_error,
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
        return 0 if status != "INVALID" else 2
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
        print(f"Goal77 terminal-geometry audit: INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
