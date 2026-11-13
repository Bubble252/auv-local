#!/usr/bin/env python3
"""Read-only Goal74 terminal-capture margin and p_nav-eta bias audit."""

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

SCHEMA_VERSION = "auv_pc_goal74_capture_margin.v2"
MARGINS_M = (0.0, 0.10, 0.25, 0.50)
BIAS_MEDIAN_MIN_M = 0.10
BIAS_GAP_MIN_M = 0.05
FALSE_SUCCESS_REDUCTION_MIN = 2
MARGIN_TRUE_RETENTION_MIN = 0.50
MIN_SAMPLES = 3


def _load_module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G73 = _load_module("goal73_analysis_for_goal74", "analyze_pc_goal73_command_demand.py")
DiagnosticError = G73.DiagnosticError


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
        default=Path("artifacts/goal74_capture_margin/analysis/goal74.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal74_capture_margin/analysis/goal74.md"),
    )
    return parser.parse_args(argv)


def _margin_tag(margin: float) -> str:
    return f"{margin:.2f}".replace(".", "p")


def _radii(run_dir: Path, summary: dict[str, Any]) -> tuple[float, float]:
    tight_radius = float(summary["eval_goal_radius_m"])
    protocol_radius = float(
        G73.G71._protocol_config(run_dir, summary)["plant"]["goal_radius_m"]
    )
    guard_radius = float(summary["eval_goal_plane_miss_guard_radius_m"])
    if not math.isfinite(tight_radius) or tight_radius <= 0.0:
        raise DiagnosticError(f"{run_dir}: invalid tight radius {tight_radius}")
    if not math.isfinite(protocol_radius) or protocol_radius <= 0.0:
        raise DiagnosticError(f"{run_dir}: invalid protocol radius {protocol_radius}")
    if tight_radius > protocol_radius + 1e-12:
        raise DiagnosticError(f"{run_dir}: tight radius exceeds protocol radius")
    if not math.isclose(guard_radius, protocol_radius, rel_tol=0.0, abs_tol=1e-12):
        raise DiagnosticError(
            f"{run_dir}: plane-miss guard radius does not match protocol radius"
        )
    return tight_radius, protocol_radius


def _margin_fields(
    *,
    d_p: float,
    d_e: float,
    tight_radius: float,
    protocol_radius: float,
    signed: float,
) -> dict[str, bool]:
    """Reclassify one terminal snapshot against the actual eval protocol.

    ``success`` comes from the tight-radius GOAL sphere.  ``terminal_eta_capture``
    is measured separately at the plant/protocol radius, so the two radii must
    never be conflated.
    """

    true_capture = bool(d_e <= protocol_radius + 1e-12)
    fields: dict[str, bool] = {"true_capture": true_capture}
    for margin in MARGINS_M:
        tag = _margin_tag(margin)
        sphere = bool(d_p <= tight_radius - margin + 1e-12)
        success = sphere
        fields[f"sphere_{tag}"] = sphere
        fields[f"plane_{tag}"] = bool(signed >= 0.0)
        fields[f"success_{tag}"] = success
        fields[f"true_success_{tag}"] = bool(success and true_capture)
        fields[f"false_success_{tag}"] = bool(success and not true_capture)
    return fields


def _task(task: tuple[str, str, str, str, str, str]) -> list[dict[str, Any]]:
    scenario, key, _, _, on_name, action_mode = task
    on_dir = Path(on_name)
    episodes = {int(row["seed"]): row for row in G73._read_episodes(on_dir)}
    summary = json.loads((on_dir / "summary.json").read_text(encoding="utf-8"))
    tight_radius, protocol_radius = _radii(on_dir, summary)
    records: list[dict[str, Any]] = []
    with np.load(on_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in G73.DEV_SEEDS:
            episode = episodes[seed]
            rows = G73._seed_rows(trace, seed)
            eta = np.asarray(rows["eta_ned"][-1], dtype=np.float64)
            p_nav = np.asarray(rows["p_nav_ned"][-1], dtype=np.float64)
            goal = np.asarray(rows["goal_ned"][-1], dtype=np.float64)
            signed = float(np.asarray(rows["signed_offset"])[-1])
            d_p = float(np.linalg.norm(p_nav - goal))
            d_e = float(np.linalg.norm(eta - goal))
            if not math.isclose(d_p, float(episode["terminal_p_nav_distance_m"]), abs_tol=1e-6):
                raise DiagnosticError(f"{on_dir} seed={seed}: p_nav distance mismatch")
            if not math.isclose(d_e, float(episode["terminal_eta_distance_m"]), abs_tol=1e-6):
                raise DiagnosticError(f"{on_dir} seed={seed}: eta distance mismatch")
            direction = goal - eta
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                bias_projection = 0.0
            else:
                bias_projection = float(np.dot(p_nav - eta, direction / norm))
            record: dict[str, Any] = {
                "scenario": scenario,
                "run_key": key,
                "action_mode": action_mode,
                "seed": seed,
                "tight_radius_m": tight_radius,
                "protocol_radius_m": protocol_radius,
                "terminal_p_nav_distance_m": d_p,
                "terminal_eta_distance_m": d_e,
                "terminal_signed_offset_m": signed,
                "p_nav_eta_error_m": float(np.linalg.norm(p_nav - eta)),
                "bias_projection_m": bias_projection,
                "actual_success": bool(episode["success"]),
                "actual_true_capture": bool(episode["terminal_eta_capture"]),
            }
            fields = _margin_fields(
                d_p=d_p,
                d_e=d_e,
                tight_radius=tight_radius,
                protocol_radius=protocol_radius,
                signed=signed,
            )
            if record["actual_true_capture"] != fields["true_capture"]:
                raise DiagnosticError(
                    f"{on_dir} seed={seed}: terminal_eta_capture does not match "
                    "protocol-radius recomputation"
                )
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
        "bias_projection_median_m": _median([record["bias_projection_m"] for record in records]),
        "p_nav_eta_error_median_m": _median([record["p_nav_eta_error_m"] for record in records]),
        "p_nav_margin_to_tight_median_m": _median(
            [
                record["tight_radius_m"] - record["terminal_p_nav_distance_m"]
                for record in records
            ]
        ),
        "eta_outside_protocol_median_m": _median(
            [record["terminal_eta_distance_m"] - record["protocol_radius_m"] for record in records]
        ),
    }


def _aggregate(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    false_group = [record for record in records if record["false_success_0p00"]]
    true_group = [record for record in records if record["true_success_0p00"]]
    margin_counts = {
        f"{margin:.2f}": {
            "success": _count(records, f"success_{_margin_tag(margin)}"),
            "true_success": _count(records, f"true_success_{_margin_tag(margin)}"),
            "false_success": _count(records, f"false_success_{_margin_tag(margin)}"),
        }
        for margin in MARGINS_M
    }
    false_counts = [margin_counts[f"{margin:.2f}"]["false_success"] for margin in MARGINS_M]
    monotonic = all(
        later <= earlier for earlier, later in zip(false_counts, false_counts[1:], strict=False)
    )
    reduction = false_counts[0] - margin_counts["0.25"]["false_success"]
    true_success_at_zero = margin_counts["0.00"]["true_success"]
    true_success_at_margin = margin_counts["0.25"]["true_success"]
    true_success_retention = (
        None
        if true_success_at_zero == 0
        else true_success_at_margin / true_success_at_zero
    )
    margin_usable = bool(
        true_success_retention is not None
        and true_success_retention >= MARGIN_TRUE_RETENTION_MIN
    )
    false_stats = _group_stats(false_group)
    true_stats = _group_stats(true_group)
    bias_gap = (
        None
        if false_stats["bias_projection_median_m"] is None
        or true_stats["bias_projection_median_m"] is None
        else float(false_stats["bias_projection_median_m"])
        - float(true_stats["bias_projection_median_m"])
    )
    h74a = {
        "supported": bool(
            monotonic and reduction >= FALSE_SUCCESS_REDUCTION_MIN and margin_usable
        ),
        "monotonic": monotonic,
        "false_success_reduction_at_0p25": reduction,
        "false_success_counts": false_counts,
        "true_success_loss_at_0p25": true_success_at_zero - true_success_at_margin,
        "true_success_retention_at_0p25": true_success_retention,
        "margin_usable": margin_usable,
    }
    h74b = {
        "supported": bool(
            len(false_group) >= MIN_SAMPLES
            and len(true_group) >= MIN_SAMPLES
            and false_stats["bias_projection_median_m"] is not None
            and float(false_stats["bias_projection_median_m"]) >= BIAS_MEDIAN_MIN_M
            and bias_gap is not None
            and float(bias_gap) >= BIAS_GAP_MIN_M
        ),
        "bias_gap_m": bias_gap,
    }
    mismatch = sum(
        bool(record["success_0p00"]) != bool(record["actual_success"]) for record in records
    )
    return {
        "n": len(records),
        "margin_counts": margin_counts,
        "false_success_group": false_stats,
        "true_success_group": true_stats,
        "h74a": h74a,
        "h74b": h74b,
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
        "# Goal74：terminal capture margin 与 p_nav-eta 方向偏差",
        "",
        f"状态：`{payload['overall']['status']}`",
        "",
        "## Margin 重分类",
        "",
        "| scenario | margin (m) | success | true success | false-success | true retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, result in payload["scenarios"].items():
        for margin in MARGINS_M:
            row = result["margin_counts"][f"{margin:.2f}"]
            zero_true = result["margin_counts"]["0.00"]["true_success"]
            retention = None if zero_true == 0 else row["true_success"] / zero_true
            lines.append(
                f"| {scenario} | {margin:.2f} | {row['success']} | "
                f"{row['true_success']} | {row['false_success']} | {_fmt(retention)} |"
            )
    lines.extend(
        [
            "",
            "## Bias 与边界余量（margin=0）",
            "",
            "| scenario | group | n | bias projection (m) | p_nav-eta error (m) | p_nav margin to tight (m) | eta outside protocol (m) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, result in payload["scenarios"].items():
        for group in ("false_success_group", "true_success_group"):
            stats = result[group]
            lines.append(
                f"| {scenario} | {group} | {stats['n']} | "
                f"{_fmt(stats['bias_projection_median_m'])} | "
                f"{_fmt(stats['p_nav_eta_error_median_m'])} | "
                f"{_fmt(stats['p_nav_margin_to_tight_median_m'])} | "
                f"{_fmt(stats['eta_outside_protocol_median_m'])} |"
            )
    lines.extend(["", "## Gate 判读", ""])
    for scenario, result in payload["scenarios"].items():
        lines.append(
            f"- `{scenario}`：H74-A={_fmt(result['h74a']['supported'])}，"
            f"H74-B={_fmt(result['h74b']['supported'])}，"
            f"margin_usable={_fmt(result['h74a']['margin_usable'])}，"
            f"true_retention={_fmt(result['h74a']['true_success_retention_at_0p25'])}，"
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
        margin_usable = all(
            bool(result["h74a"]["margin_usable"]) for result in scenarios.values()
        )
        margin_supported = all(
            bool(result["h74a"]["supported"]) for result in scenarios.values()
        )
        bias_supported = all(
            bool(result["h74b"]["supported"]) for result in scenarios.values()
        )
        any_supported = any(
            bool(result["h74a"]["supported"]) or bool(result["h74b"]["supported"])
            for result in scenarios.values()
        )
        if mismatch_rate > 0.02:
            status = "INVALID"
        elif margin_supported:
            status = "MARGIN_SUPPORTED"
        elif bias_supported and not margin_usable:
            status = "MARGIN_UNUSABLE_BIAS_SUPPORTED"
        elif bias_supported:
            status = "BIAS_SUPPORTED"
        elif not margin_usable:
            status = "MARGIN_UNUSABLE"
        elif any_supported:
            status = "PARTIAL_SUPPORT"
        else:
            status = "CAPTURE_NOT_EXPLANATORY"
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "overall": {
                "status": status,
                "margin_usable": margin_usable,
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
        print(f"Goal74 capture-margin audit: INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
