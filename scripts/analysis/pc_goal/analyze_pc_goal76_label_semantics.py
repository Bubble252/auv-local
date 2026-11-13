#!/usr/bin/env python3
"""Read-only Goal76 terminal-label semantics and deadband re-evaluation."""

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

SCHEMA_VERSION = "auv_pc_goal76_label_semantics.v1"
TIGHT_FALSE_REDUCTION_MIN = 0.50
TIGHT_TRUE_RETENTION_MIN = 0.95
TIGHT_TO_PROTOCOL_MAX_RATIO = 0.70
MIN_PROTOCOL_ONLY = 10
MIN_GROUP_SAMPLES = 5


def _load_module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G75 = _load_module("goal75_analysis_for_goal76", "analyze_pc_goal75_oracle_bias_correction.py")
G73 = G75.G73
DiagnosticError = G75.DiagnosticError


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
        default=Path("artifacts/goal76_label_semantics/analysis/goal76.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal76_label_semantics/analysis/goal76.md"),
    )
    return parser.parse_args(argv)


def _seed_last(trace: Any, seed: int, name: str) -> np.ndarray:
    if name not in trace:
        raise DiagnosticError(f"trace missing field {name}")
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    values = np.asarray(trace[name])[selected]
    if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
        raise DiagnosticError(f"seed {seed}: non-finite trace field {name}")
    return values[-1]


def _last_vector(trace: Any, seed: int, name: str) -> np.ndarray:
    return np.asarray(_seed_last(trace, seed, name), dtype=np.float64)


def _last_scalar(trace: Any, seed: int, name: str) -> float:
    return float(_seed_last(trace, seed, name))


def _label_fields(fields: dict[str, Any]) -> dict[str, Any]:
    tight = float(fields["tight_radius_m"])
    d_e = float(fields["terminal_eta_distance_m"])
    tight_capture = bool(d_e <= tight + 1e-12)
    success_none = bool(fields["success_none"])
    success_full = bool(fields["success_full"])
    success_deadband = bool(fields["success_deadband"])
    labels: dict[str, Any] = {
        "tight_capture": tight_capture,
        "protocol_capture": bool(fields["true_capture"]),
    }
    for arm, success in (
        ("none", success_none),
        ("full", success_full),
        ("deadband", success_deadband),
    ):
        labels[f"protocol_true_{arm}"] = bool(success and labels["protocol_capture"])
        labels[f"tight_true_{arm}"] = bool(success and tight_capture)
        labels[f"tight_false_{arm}"] = bool(success and not tight_capture)
    return labels


def _task(task: tuple[str, str, str, str, str, str]) -> list[dict[str, Any]]:
    scenario, key, _, _, on_name, action_mode = task
    on_dir = Path(on_name)
    episodes = {int(row["seed"]): row for row in G73._read_episodes(on_dir)}
    summary = json.loads((on_dir / "summary.json").read_text(encoding="utf-8"))
    tight_radius, protocol_radius = G75.G74._radii(on_dir, summary)
    records: list[dict[str, Any]] = []
    with np.load(on_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in G73.DEV_SEEDS:
            episode = episodes[seed]
            rows = G73._seed_rows(trace, seed)
            eta = np.asarray(rows["eta_ned"][-1], dtype=np.float64)
            p_nav = np.asarray(rows["p_nav_ned"][-1], dtype=np.float64)
            goal = np.asarray(rows["goal_ned"][-1], dtype=np.float64)
            fields = G75._correction_fields(
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
                "signed_offset_m": _last_scalar(trace, seed, "signed_offset"),
                "path_error_3d_m": _last_scalar(trace, seed, "path_error_3d"),
                "path_horizontal_error_m": _last_scalar(
                    trace, seed, "path_horizontal_error"
                ),
                "path_vertical_error_m": _last_scalar(
                    trace, seed, "path_vertical_error"
                ),
                "goal_distance_m": _last_scalar(trace, seed, "goal_distance"),
                "base_command_l2": float(
                    np.linalg.norm(_last_vector(trace, seed, "base_command"))
                ),
                "residual_l2": float(
                    np.linalg.norm(_last_vector(trace, seed, "residual"))
                ),
                "action_l2": float(
                    np.linalg.norm(_last_vector(trace, seed, "action"))
                ),
                "residual_gate": _last_scalar(trace, seed, "gate"),
                "scale_clipped": bool(np.any(_seed_last(trace, seed, "scale_clipped"))),
                "rate_clipped": bool(np.any(_seed_last(trace, seed, "rate_clipped"))),
            }
            record.update(fields)
            record.update(_label_fields(record))
            records.append(record)
    return records


def _median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.median(finite))


def _count(records: list[dict[str, Any]], field: str) -> int:
    return int(sum(bool(record[field]) for record in records))


def _feature_medians(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "bias_projection_m",
        "p_nav_eta_error_m",
        "signed_offset_m",
        "path_error_3d_m",
        "goal_distance_m",
        "base_command_l2",
        "residual_l2",
        "action_l2",
        "residual_gate",
    )
    return {
        "n": len(records),
        **{f"{name}_median": _median([record[name] for record in records]) for name in fields},
    }


def _aggregate(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    base = {
        "success": _count(records, "success_none"),
        "protocol_true": _count(records, "protocol_true_none"),
        "tight_true": _count(records, "tight_true_none"),
        "tight_false": _count(records, "tight_false_none"),
    }
    deadband = {
        "success": _count(records, "success_deadband"),
        "protocol_true": _count(records, "protocol_true_deadband"),
        "tight_true": _count(records, "tight_true_deadband"),
        "tight_false": _count(records, "tight_false_deadband"),
    }
    full = {
        "success": _count(records, "success_full"),
        "protocol_true": _count(records, "protocol_true_full"),
        "tight_true": _count(records, "tight_true_full"),
        "tight_false": _count(records, "tight_false_full"),
    }
    tight_false_reduction = base["tight_false"] - deadband["tight_false"]
    tight_false_reduction_fraction = (
        None
        if base["tight_false"] <= 0
        else tight_false_reduction / base["tight_false"]
    )
    tight_true_retention = (
        None
        if base["tight_true"] <= 0
        else deadband["tight_true"] / base["tight_true"]
    )
    tight_protocol_ratio = (
        None
        if base["protocol_true"] <= 0
        else base["tight_true"] / base["protocol_true"]
    )
    protocol_only = base["protocol_true"] - base["tight_true"]
    enough_samples = (
        base["protocol_true"] >= MIN_GROUP_SAMPLES
        and base["tight_true"] >= MIN_GROUP_SAMPLES
        and base["tight_false"] >= MIN_GROUP_SAMPLES
    )
    h76a_supported = bool(
        enough_samples
        and tight_false_reduction_fraction is not None
        and tight_false_reduction_fraction >= TIGHT_FALSE_REDUCTION_MIN
        and tight_true_retention is not None
        and tight_true_retention >= TIGHT_TRUE_RETENTION_MIN
    )
    h76b_supported = bool(
        tight_protocol_ratio is not None
        and tight_protocol_ratio <= TIGHT_TO_PROTOCOL_MAX_RATIO
        and protocol_only >= MIN_PROTOCOL_ONLY
    )
    high_b = [record for record in records if record["bias_projection_m"] > G75.DEADBAND_M]
    high_b_tight_true = [record for record in high_b if record["tight_true_none"]]
    high_b_tight_false = [record for record in high_b if record["tight_false_none"]]
    mismatch = sum(
        bool(record["success_none"]) != bool(record["actual_success"])
        for record in records
    )
    return {
        "n": len(records),
        "base": base,
        "deadband": deadband,
        "full": full,
        "protocol_only": protocol_only,
        "gates": {
            "H76-A": {
                "supported": h76a_supported,
                "tight_false_reduction": tight_false_reduction,
                "tight_false_reduction_fraction": tight_false_reduction_fraction,
                "tight_true_retention": tight_true_retention,
                "tight_true_lost": base["tight_true"] - deadband["tight_true"],
            },
            "H76-B": {
                "supported": h76b_supported,
                "tight_to_protocol_ratio": tight_protocol_ratio,
                "protocol_only": protocol_only,
            },
        },
        "high_b_descriptive": {
            "n": len(high_b),
            "tight_true": _feature_medians(high_b_tight_true),
            "tight_false": _feature_medians(high_b_tight_false),
        },
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
        "# Goal76：terminal success label semantics",
        "",
        f"状态：`{payload['overall']['status']}`",
        "",
        "## protocol vs tight 标签",
        "",
        "| scenario | arm | success | protocol true | tight true | tight false | tight true 保留率 | tight false 降幅 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, result in payload["scenarios"].items():
        for arm in ("none", "deadband", "full"):
            counts = result["base"] if arm == "none" else result[arm]
            if arm == "none":
                true_ret = "—"
                false_red = "—"
            elif arm == "deadband":
                true_ret = result["gates"]["H76-A"]["tight_true_retention"]
                false_red = result["gates"]["H76-A"]["tight_false_reduction_fraction"]
            else:
                true_ret = (
                    None
                    if result["base"]["tight_true"] <= 0
                    else counts["tight_true"] / result["base"]["tight_true"]
                )
                false_red = (
                    None
                    if result["base"]["tight_false"] <= 0
                    else (result["base"]["tight_false"] - counts["tight_false"])
                    / result["base"]["tight_false"]
                )
            lines.append(
                f"| {scenario} | {arm} | {counts['success']} | {counts['protocol_true']} | "
                f"{counts['tight_true']} | {counts['tight_false']} | "
                f"{_fmt(true_ret)} | {_fmt(false_red)} |"
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
            f"- `{scenario}`：H76-A={_fmt(result['gates']['H76-A']['supported'])}，"
            f"H76-B={_fmt(result['gates']['H76-B']['supported'])}，"
            f"protocol-only={result['protocol_only']}，"
            f"tight/protocol={_fmt(result['gates']['H76-B']['tight_to_protocol_ratio'])}"
        )
    lines.extend(
        [
            "",
            "## 高 b 样本描述性对比（不作 Gate）",
            "",
            "| scenario | group | n | bias | p_nav-eta | signed offset | path 3d | residual L2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, result in payload["scenarios"].items():
        for group in ("tight_true", "tight_false"):
            stats = result["high_b_descriptive"][group]
            lines.append(
                f"| {scenario} | {group} | {stats['n']} | "
                f"{_fmt(stats['bias_projection_m_median'])} | "
                f"{_fmt(stats['p_nav_eta_error_m_median'])} | "
                f"{_fmt(stats['signed_offset_m_median'])} | "
                f"{_fmt(stats['path_error_3d_m_median'])} | "
                f"{_fmt(stats['residual_l2_median'])} |"
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
        h76a = all(
            bool(result["gates"]["H76-A"]["supported"])
            for result in scenarios.values()
        )
        h76b = all(
            bool(result["gates"]["H76-B"]["supported"])
            for result in scenarios.values()
        )
        if mismatch_rate > 0.02:
            status = "INVALID"
        elif h76a and h76b:
            status = "TIGHT_LABEL_AND_DEADBAND_SUPPORTED"
        elif h76a:
            status = "TIGHT_LABEL_DEADBAND_SUPPORTED"
        elif h76b:
            status = "LABEL_CONFLATION_CONFIRMED_BUT_DEADBAND_FAIL"
        else:
            status = "LABEL_SEMANTICS_NOT_EXPLANATORY"
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
        print(f"Goal76 label audit: INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
