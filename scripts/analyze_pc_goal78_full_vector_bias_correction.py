#!/usr/bin/env python3
"""Read-only Goal78 full-vector terminal bias correction audit."""

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

SCHEMA_VERSION = "auv_pc_goal78_full_vector_bias_correction.v1"
DEADBAND_M = 0.25
FALSE_REDUCTION_MIN = 0.50
TRUE_RETENTION_MIN = 0.95
GAIN_MIN = 0.25
RETENTION_LOSS_MAX = 0.05


def _load_module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G77 = _load_module(
    "goal77_analysis_for_goal78", "analyze_pc_goal77_terminal_geometry_gate.py"
)
G76 = G77.G76
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
        default=Path("artifacts/goal78_full_vector_bias/analysis/goal78.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal78_full_vector_bias/analysis/goal78.md"),
    )
    return parser.parse_args(argv)


def _correction_arms(
    *,
    eta: np.ndarray,
    p_nav: np.ndarray,
    goal: np.ndarray,
) -> dict[str, np.ndarray | float]:
    eta = np.asarray(eta, dtype=np.float64)
    p_nav = np.asarray(p_nav, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if eta.shape != (3,) or p_nav.shape != (3,) or goal.shape != (3,):
        raise DiagnosticError("eta, p_nav and goal must have shape (3,)")
    error = p_nav - eta
    error_norm = float(np.linalg.norm(error))
    direction = goal - eta
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        u_goal = np.zeros(3, dtype=np.float64)
        along = 0.0
    else:
        u_goal = direction / direction_norm
        along = float(np.dot(error, u_goal))
    along_correction = max(along - DEADBAND_M, 0.0)
    full_correction = max(error_norm - DEADBAND_M, 0.0)
    if error_norm <= 1e-12:
        u_error = np.zeros(3, dtype=np.float64)
    else:
        u_error = error / error_norm
    p_along = p_nav - along_correction * u_goal
    p_full = p_nav - full_correction * u_error
    return {
        "error_vector": error,
        "error_norm": error_norm,
        "along_goal": along,
        "along_correction": float(along_correction),
        "full_correction": float(full_correction),
        "p_nav_along": p_along,
        "p_nav_full": p_full,
    }


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
            if not math.isclose(
                float(np.linalg.norm(p_nav - goal)),
                float(episode["terminal_p_nav_distance_m"]),
                abs_tol=1e-6,
            ):
                raise DiagnosticError(f"{on_dir} seed={seed}: p_nav distance mismatch")
            if not math.isclose(
                float(np.linalg.norm(eta - goal)),
                float(episode["terminal_eta_distance_m"]),
                abs_tol=1e-6,
            ):
                raise DiagnosticError(f"{on_dir} seed={seed}: eta distance mismatch")
            arms = _correction_arms(eta=eta, p_nav=p_nav, goal=goal)
            d_e = float(np.linalg.norm(eta - goal))
            tight_capture = bool(d_e <= tight_radius + 1e-12)
            protocol_capture = bool(d_e <= protocol_radius + 1e-12)
            if protocol_capture != bool(episode["terminal_eta_capture"]):
                raise DiagnosticError(
                    f"{on_dir} seed={seed}: terminal_eta_capture mismatch"
                )
            record: dict[str, Any] = {
                "scenario": scenario,
                "run_key": key,
                "action_mode": action_mode,
                "seed": seed,
                "tight_radius_m": tight_radius,
                "protocol_radius_m": protocol_radius,
                "d_e": d_e,
                "tight_capture": tight_capture,
                "protocol_capture": protocol_capture,
                "actual_success": bool(episode["success"]),
                "error_norm": float(arms["error_norm"]),
                "along_goal": float(arms["along_goal"]),
                "along_correction": float(arms["along_correction"]),
                "full_correction": float(arms["full_correction"]),
            }
            successes = {
                "A0": bool(np.linalg.norm(p_nav - goal) <= tight_radius + 1e-12),
                "A1": bool(
                    np.linalg.norm(np.asarray(arms["p_nav_along"]) - goal)
                    <= tight_radius + 1e-12
                ),
                "A2": bool(
                    np.linalg.norm(np.asarray(arms["p_nav_full"]) - goal)
                    <= tight_radius + 1e-12
                ),
            }
            record["mismatch"] = bool(successes["A0"]) != bool(episode["success"])
            for arm, success in successes.items():
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
    return {
        "success": _count(records, f"success_{arm}"),
        "tight_true": tight_true,
        "tight_false": tight_false,
        "protocol_true": _count(records, f"protocol_true_{arm}"),
        "tight_false_reduction_fraction": false_reduction,
        "tight_true_retention": true_retention,
    }


def _median(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None and math.isfinite(float(record[field]))
    ]
    return None if not values else float(np.median(values))


def _h78b_supported(scenarios: dict[str, Any]) -> bool:
    gains: list[float] = []
    losses: list[float] = []
    for result in scenarios.values():
        a1 = result["arms"]["A1"]
        a2 = result["arms"]["A2"]
        if (
            a1["tight_false_reduction_fraction"] is None
            or a2["tight_false_reduction_fraction"] is None
            or a1["tight_true_retention"] is None
            or a2["tight_true_retention"] is None
        ):
            return False
        gains.append(
            a2["tight_false_reduction_fraction"]
            - a1["tight_false_reduction_fraction"]
        )
        losses.append(a1["tight_true_retention"] - a2["tight_true_retention"])
    return bool(
        any(gain >= GAIN_MIN for gain in gains)
        and all(loss <= RETENTION_LOSS_MAX + 1e-12 for loss in losses)
    )


def _aggregate(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    base = {
        "success": _count(records, "success_A0"),
        "tight_true": int(sum(r["success_A0"] and r["tight_capture"] for r in records)),
        "tight_false": int(
            sum(r["success_A0"] and not r["tight_capture"] for r in records)
        ),
        "protocol_true": int(
            sum(r["success_A0"] and r["protocol_capture"] for r in records)
        ),
    }
    arms = {arm: _arm_stats(records, arm, base) for arm in ("A0", "A1", "A2")}
    a2 = arms["A2"]
    h78a_supported = bool(
        a2["tight_false_reduction_fraction"] is not None
        and a2["tight_true_retention"] is not None
        and a2["tight_false_reduction_fraction"] >= FALSE_REDUCTION_MIN
        and a2["tight_true_retention"] >= TRUE_RETENTION_MIN
    )
    return {
        "n": len(records),
        "base": base,
        "arms": arms,
        "gates": {
            "H78-A": {"supported": h78a_supported},
            "H78-B": {"supported": False},
        },
        "mismatch": _count(records, "mismatch"),
        "correction_medians": {
            "error_norm": _median(records, "error_norm"),
            "along_goal": _median(records, "along_goal"),
            "along_correction": _median(records, "along_correction"),
            "full_correction": _median(records, "full_correction"),
        },
        "by_checkpoint": {
            key: {
                "n": sum(r["run_key"] == key for r in records),
                "tight_false_A0": sum(
                    r["run_key"] == key and r["success_A0"] and not r["tight_capture"]
                    for r in records
                ),
                "tight_false_A1": sum(
                    r["run_key"] == key and r["tight_false_A1"] for r in records
                ),
                "tight_false_A2": sum(
                    r["run_key"] == key and r["tight_false_A2"] for r in records
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
        "# Goal78：full-vector terminal bias correction",
        "",
        f"状态：`{payload['overall']['status']}`",
        "",
        "| scenario | arm | success | tight true | tight false | tight true 保留率 | tight false 降幅 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, result in payload["scenarios"].items():
        for arm in ("A0", "A1", "A2"):
            stats = result["arms"][arm]
            lines.append(
                f"| {scenario} | {arm} | {stats['success']} | {stats['tight_true']} | "
                f"{stats['tight_false']} | {_fmt(stats['tight_true_retention'])} | "
                f"{_fmt(stats['tight_false_reduction_fraction'])} |"
            )
    lines.extend(["", "## Gate 判读", ""])
    for scenario, result in payload["scenarios"].items():
        lines.append(
            f"- `{scenario}`：H78-A={_fmt(result['gates']['H78-A']['supported'])}，"
            f"H78-B={_fmt(result['gates']['H78-B']['supported'])}，"
            f"mismatch={result['mismatch']}"
        )
    lines.extend(
        [
            "",
            "## correction 中位数",
            "",
            "| scenario | error norm | along goal | along correction | full correction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario, result in payload["scenarios"].items():
        med = result["correction_medians"]
        lines.append(
            f"| {scenario} | {_fmt(med['error_norm'])} | {_fmt(med['along_goal'])} | "
            f"{_fmt(med['along_correction'])} | {_fmt(med['full_correction'])} |"
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
        h78a = all(result["gates"]["H78-A"]["supported"] for result in scenarios.values())
        h78b = _h78b_supported(scenarios)
        for result in scenarios.values():
            result["gates"]["H78-B"]["supported"] = h78b
        mismatch = sum(result["mismatch"] for result in scenarios.values())
        if mismatch > 0:
            status = "INVALID"
        elif h78a:
            status = "FULL_VECTOR_DEADBAND_SUPPORTED"
        elif h78b:
            status = "FULL_VECTOR_DIRECTION_SUPPORTED_BUT_GATE_FAIL"
        else:
            status = "FULL_VECTOR_DEADBAND_NOT_SUPPORTED"
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": status,
                "H78-A supported": h78a,
                "H78-B supported": h78b,
                "mismatch": mismatch,
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
        print(f"Goal78 full-vector audit: INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
