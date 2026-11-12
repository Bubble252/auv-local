#!/usr/bin/env python3
"""Read-only Goal75 oracle terminal p_nav bias-correction audit."""

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

SCHEMA_VERSION = "auv_pc_goal75_oracle_bias_correction.v1"
DEADBAND_M = 0.25
FULL_FALSE_REDUCTION_MIN = 0.50
FULL_TRUE_RETENTION_MIN = 0.80
DEADBAND_TRUE_RETENTION_MIN = 0.95
MIN_FALSE_SUCCESS = 5
MIN_TRUE_SUCCESS = 5
ARMS = ("none", "full", "deadband")


def _load_module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G74 = _load_module("goal74_analysis_for_goal75", "analyze_pc_goal74_capture_margin.py")
G73 = G74.G73
DiagnosticError = G74.DiagnosticError


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
        default=Path("artifacts/goal75_oracle_bias/analysis/goal75.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal75_oracle_bias/analysis/goal75.md"),
    )
    return parser.parse_args(argv)


def _correction_fields(
    *,
    eta: np.ndarray,
    p_nav: np.ndarray,
    goal: np.ndarray,
    tight_radius: float,
    protocol_radius: float,
) -> dict[str, Any]:
    eta = np.asarray(eta, dtype=np.float64)
    p_nav = np.asarray(p_nav, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if eta.shape != (3,) or p_nav.shape != (3,) or goal.shape != (3,):
        raise DiagnosticError("eta, p_nav and goal must have shape (3,)")

    delta_goal = goal - eta
    direction_norm = float(np.linalg.norm(delta_goal))
    error = p_nav - eta
    if direction_norm <= 1e-12:
        direction = np.zeros(3, dtype=np.float64)
        bias_projection = 0.0
    else:
        direction = delta_goal / direction_norm
        bias_projection = float(np.dot(error, direction))

    full_correction = max(bias_projection, 0.0)
    deadband_correction = max(bias_projection - DEADBAND_M, 0.0)
    p_nav_full = p_nav - full_correction * direction
    p_nav_deadband = p_nav - deadband_correction * direction

    d_p = float(np.linalg.norm(p_nav - goal))
    d_full = float(np.linalg.norm(p_nav_full - goal))
    d_deadband = float(np.linalg.norm(p_nav_deadband - goal))
    d_e = float(np.linalg.norm(eta - goal))
    true_capture = bool(d_e <= protocol_radius + 1e-12)
    successes = {
        "none": bool(d_p <= tight_radius + 1e-12),
        "full": bool(d_full <= tight_radius + 1e-12),
        "deadband": bool(d_deadband <= tight_radius + 1e-12),
    }
    fields: dict[str, Any] = {
        "bias_projection_m": bias_projection,
        "p_nav_eta_error_m": float(np.linalg.norm(error)),
        "p_nav_none_distance_m": d_p,
        "p_nav_full_distance_m": d_full,
        "p_nav_deadband_distance_m": d_deadband,
        "terminal_eta_distance_m": d_e,
        "p_nav_margin_to_tight_m": float(tight_radius - d_p),
        "eta_outside_protocol_m": float(d_e - protocol_radius),
        "true_capture": true_capture,
    }
    for arm in ARMS:
        success = successes[arm]
        fields[f"success_{arm}"] = success
        fields[f"true_success_{arm}"] = bool(success and true_capture)
        fields[f"false_success_{arm}"] = bool(success and not true_capture)
    return fields


def _task(task: tuple[str, str, str, str, str, str]) -> list[dict[str, Any]]:
    scenario, key, _, _, on_name, action_mode = task
    on_dir = Path(on_name)
    episodes = {int(row["seed"]): row for row in G73._read_episodes(on_dir)}
    summary = json.loads((on_dir / "summary.json").read_text(encoding="utf-8"))
    tight_radius, protocol_radius = G74._radii(on_dir, summary)
    records: list[dict[str, Any]] = []
    with np.load(on_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in G73.DEV_SEEDS:
            episode = episodes[seed]
            rows = G73._seed_rows(trace, seed)
            eta = np.asarray(rows["eta_ned"][-1], dtype=np.float64)
            p_nav = np.asarray(rows["p_nav_ned"][-1], dtype=np.float64)
            goal = np.asarray(rows["goal_ned"][-1], dtype=np.float64)
            fields = _correction_fields(
                eta=eta,
                p_nav=p_nav,
                goal=goal,
                tight_radius=tight_radius,
                protocol_radius=protocol_radius,
            )
            actual_true_capture = bool(episode["terminal_eta_capture"])
            if actual_true_capture != fields["true_capture"]:
                raise DiagnosticError(
                    f"{on_dir} seed={seed}: terminal_eta_capture does not match "
                    "protocol-radius recomputation"
                )
            record: dict[str, Any] = {
                "scenario": scenario,
                "run_key": key,
                "action_mode": action_mode,
                "seed": seed,
                "tight_radius_m": tight_radius,
                "protocol_radius_m": protocol_radius,
                "actual_success": bool(episode["success"]),
                "actual_true_capture": actual_true_capture,
            }
            record.update(fields)
            records.append(record)
    return records


def _median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.median(finite))


def _count(records: list[dict[str, Any]], field: str) -> int:
    return int(sum(bool(record[field]) for record in records))


def _group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "bias_projection_median_m": _median(
            [record["bias_projection_m"] for record in records]
        ),
        "p_nav_eta_error_median_m": _median(
            [record["p_nav_eta_error_m"] for record in records]
        ),
        "p_nav_margin_to_tight_median_m": _median(
            [record["p_nav_margin_to_tight_m"] for record in records]
        ),
        "eta_outside_protocol_median_m": _median(
            [record["eta_outside_protocol_m"] for record in records]
        ),
    }


def _arm_counts(records: list[dict[str, Any]], arm: str) -> dict[str, int]:
    return {
        "success": _count(records, f"success_{arm}"),
        "true_success": _count(records, f"true_success_{arm}"),
        "false_success": _count(records, f"false_success_{arm}"),
    }


def _retention(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _reduction(base: int, corrected: int) -> int:
    return int(base - corrected)


def _aggregate(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    base = _arm_counts(records, "none")
    arms = {arm: _arm_counts(records, arm) for arm in ARMS}
    full_reduction = _reduction(base["false_success"], arms["full"]["false_success"])
    deadband_reduction = _reduction(
        base["false_success"], arms["deadband"]["false_success"]
    )
    full_false_reduction_fraction = _retention(
        full_reduction, base["false_success"]
    )
    deadband_false_reduction_fraction = _retention(
        deadband_reduction, base["false_success"]
    )
    full_true_retention = _retention(arms["full"]["true_success"], base["true_success"])
    deadband_true_retention = _retention(
        arms["deadband"]["true_success"], base["true_success"]
    )
    enough_samples = (
        base["false_success"] >= MIN_FALSE_SUCCESS
        and base["true_success"] >= MIN_TRUE_SUCCESS
    )
    full_supported = bool(
        enough_samples
        and full_false_reduction_fraction is not None
        and full_false_reduction_fraction >= FULL_FALSE_REDUCTION_MIN
        and full_true_retention is not None
        and full_true_retention >= FULL_TRUE_RETENTION_MIN
    )
    deadband_supported = bool(
        enough_samples
        and deadband_false_reduction_fraction is not None
        and deadband_false_reduction_fraction >= FULL_FALSE_REDUCTION_MIN
        and deadband_true_retention is not None
        and deadband_true_retention >= DEADBAND_TRUE_RETENTION_MIN
    )
    false_group = [record for record in records if record["false_success_none"]]
    true_group = [record for record in records if record["true_success_none"]]
    mismatch = sum(
        bool(record["success_none"]) != bool(record["actual_success"])
        for record in records
    )
    return {
        "n": len(records),
        "base": base,
        "arms": arms,
        "gates": {
            "H75-A": {
                "supported": full_supported,
                "false_success_reduction": full_reduction,
                "false_success_reduction_fraction": full_false_reduction_fraction,
                "true_success_retention": full_true_retention,
                "true_success_lost": _reduction(
                    base["true_success"], arms["full"]["true_success"]
                ),
            },
            "H75-B": {
                "supported": deadband_supported,
                "false_success_reduction": deadband_reduction,
                "false_success_reduction_fraction": deadband_false_reduction_fraction,
                "true_success_retention": deadband_true_retention,
                "true_success_lost": _reduction(
                    base["true_success"], arms["deadband"]["true_success"]
                ),
            },
        },
        "false_success_group": _group_stats(false_group),
        "true_success_group": _group_stats(true_group),
        "reclassification_mismatch": mismatch,
        "reclassification_mismatch_rate": mismatch / max(len(records), 1),
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
        "# Goal75：oracle terminal p_nav bias correction",
        "",
        f"状态：`{payload['overall']['status']}`",
        "",
        "## 修正结果",
        "",
        "| scenario | arm | success | true success | false-success | false 降幅 | true 保留率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, result in payload["scenarios"].items():
        scenario_key = scenario
        for arm in ARMS:
            counts = result["arms"][arm]
            if arm == "none":
                reduction = "—"
                retention = "—"
            elif arm == "full":
                reduction = result["gates"]["H75-A"][
                    "false_success_reduction_fraction"
                ]
                retention = result["gates"]["H75-A"]["true_success_retention"]
            else:
                reduction = result["gates"]["H75-B"][
                    "false_success_reduction_fraction"
                ]
                retention = result["gates"]["H75-B"]["true_success_retention"]
            lines.append(
                f"| {scenario_key} | {arm} | {counts['success']} | "
                f"{counts['true_success']} | {counts['false_success']} | "
                f"{_fmt(reduction)} | {_fmt(retention)} |"
            )
    lines.extend(
        [
            "",
            "## Bias 分组（no correction）",
            "",
            "| scenario | group | n | bias projection (m) | p_nav-eta error (m) | tight margin (m) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for scenario, result in payload["scenarios"].items():
        for group in ("false_success_group", "true_success_group"):
            stats = result[group]
            lines.append(
                f"| {scenario} | {group} | {stats['n']} | "
                f"{_fmt(stats['bias_projection_median_m'])} | "
                f"{_fmt(stats['p_nav_eta_error_median_m'])} | "
                f"{_fmt(stats['p_nav_margin_to_tight_median_m'])} |"
            )
    lines.extend(["", "## Gate 判读", ""])
    for scenario, result in payload["scenarios"].items():
        lines.append(
            f"- `{scenario}`：H75-A={_fmt(result['gates']['H75-A']['supported'])}，"
            f"H75-B={_fmt(result['gates']['H75-B']['supported'])}，"
            f"mismatch={result['reclassification_mismatch']}/{result['n']}"
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
        mismatch_rate = max(
            result["reclassification_mismatch_rate"] for result in scenarios.values()
        )
        full_supported = all(
            bool(result["gates"]["H75-A"]["supported"])
            for result in scenarios.values()
        )
        deadband_supported = all(
            bool(result["gates"]["H75-B"]["supported"])
            for result in scenarios.values()
        )
        any_supported = any(
            bool(result["gates"]["H75-A"]["supported"])
            or bool(result["gates"]["H75-B"]["supported"])
            for result in scenarios.values()
        )
        if mismatch_rate > 0.02:
            status = "INVALID"
        elif full_supported and deadband_supported:
            status = "ORACLE_AND_DEADBAND_SUPPORTED"
        elif deadband_supported:
            status = "DEADBAND_ORACLE_SUPPORTED"
        elif full_supported:
            status = "FULL_ORACLE_SUPPORTED"
        elif any_supported:
            status = "PARTIAL_SUPPORT"
        else:
            status = "ORACLE_BIAS_CORRECTION_NOT_SUPPORTED"
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": status,
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
        print(f"Goal75 oracle bias audit: INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
