#!/usr/bin/env python3
"""Goal65 paired analysis for the 50-step terminal plane-miss dwell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal65_plane_miss_dwell.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
DWELL_STEPS = 50
MARGIN_M = 0.25
TRUE_RETENTION_MIN = {"default": 0.90, "strict125": 0.85}
FALSE_REMOVAL_MIN = 0.50
FALSE_SUCCESS_SLACK = 2
PRECISION_MIN = 0.85
TERMINAL_DELTA_MAX = 0.05
CHECKPOINT_RETENTION_MIN = 0.80
CHECKPOINT_PASS_COUNT_MIN = 3
ZERO_RETENTION_MIN = 0.80


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/traces"))
    parser.add_argument(
        "--immediate-root",
        type=Path,
        default=Path("artifacts/goal63_terminal_suppression"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/goal65_plane_miss_dwell"),
    )
    parser.add_argument("--candidate-suffix", default="_dwell50_m025")
    parser.add_argument("--immediate-suffix", default="_m025")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal65_plane_miss_dwell/analysis/goal65.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal65_plane_miss_dwell/analysis/goal65.md"),
    )
    return parser.parse_args(argv)


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _trace_terminal_p_nav(run_dir: Path, rows: list[dict[str, Any]]) -> dict[int, float]:
    trace_path = run_dir / "trace.npz"
    if not trace_path.is_file():
        return {}
    result: dict[int, float] = {}
    with np.load(trace_path, allow_pickle=False) as trace:
        for row in rows:
            seed = int(row["seed"])
            terminal = np.flatnonzero((trace["seed"] == seed) & trace["done"].astype(bool))
            if terminal.size != 1:
                raise ValueError(f"{run_dir}: seed {seed} must have one terminal trace row")
            index = int(terminal[0])
            p_nav = np.asarray(trace["p_nav_ned"][index], dtype=np.float64)
            goal = np.asarray(trace["goal_ned"][index], dtype=np.float64)
            result[seed] = _finite(
                np.linalg.norm(p_nav - goal),
                name=f"{run_dir}:terminal_p_nav_distance_m",
            )
    return result


def _episodes(run_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{run_dir}: episodes must be a list")
    if [int(row["seed"]) for row in rows] != list(DEV_SEEDS):
        raise ValueError(f"{run_dir}: episode seeds must be exactly 20-59")
    traced_p_nav = _trace_terminal_p_nav(run_dir, rows)
    for row in rows:
        for key in (
            "steps",
            "success",
            "goal_plane_miss",
            "timeout",
            "safety",
            "terminal_eta_capture",
            "terminal_eta_distance_m",
        ):
            if key not in row:
                raise ValueError(f"{run_dir}: missing episode field {key}")
        _finite(row["terminal_eta_distance_m"], name=f"{run_dir}:terminal_eta_distance_m")
        if "terminal_p_nav_distance_m" not in row:
            row["terminal_p_nav_distance_m"] = traced_p_nav.get(int(row["seed"]))
        if row["terminal_p_nav_distance_m"] is None:
            raise ValueError(f"{run_dir}: missing terminal_p_nav_distance_m")
        _finite(
            row["terminal_p_nav_distance_m"],
            name=f"{run_dir}:terminal_p_nav_distance_m",
        )
    if any(seed in HELDOUT_SEEDS for seed in DEV_SEEDS):
        raise ValueError("held-out seed in contract")
    return rows


def _protocol_radius(run_dir: Path, summary: dict[str, Any]) -> float:
    config_path = Path(str(summary["environment_config"]))
    if not config_path.is_absolute():
        config_path = run_dir / config_path
        if not config_path.is_file():
            config_path = Path(__file__).resolve().parents[3] / str(summary["environment_config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return float(config["plant"]["goal_radius_m"])


def _validate_run(
    baseline_dir: Path,
    immediate_dir: Path,
    candidate_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_summary = _summary(baseline_dir)
    immediate_summary = _summary(immediate_dir)
    candidate_summary = _summary(candidate_dir)
    base_rows = _episodes(baseline_dir)
    immediate_rows = _episodes(immediate_dir)
    candidate_rows = _episodes(candidate_dir)
    for label, summary in (
        ("baseline", base_summary),
        ("immediate", immediate_summary),
        ("candidate", candidate_summary),
    ):
        if summary.get("termination_mode") != "endpoint_stop":
            raise ValueError(f"{label}: expected endpoint_stop")
        if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
            raise ValueError(f"{label}: expected CPU/float64")
        if summary.get("summary", {}).get("finite") is not True:
            raise ValueError(f"{label}: summary is not finite")
        if summary.get("summary", {}).get("action_mode") != expected_action_mode:
            raise ValueError(f"{label}: action mode mismatch")
    shas = {
        base_summary.get("checkpoint_sha256"),
        immediate_summary.get("checkpoint_sha256"),
        candidate_summary.get("checkpoint_sha256"),
    }
    if len(shas) != 1:
        raise ValueError("baseline/immediate/candidate checkpoint SHA mismatch")
    if expected_action_mode == "policy" and None in shas:
        raise ValueError("policy checkpoint SHA is missing")

    radius = _protocol_radius(baseline_dir, base_summary)
    for label, run_dir, summary in (
        ("immediate", immediate_dir, immediate_summary),
        ("candidate", candidate_dir, candidate_summary),
    ):
        if not math.isclose(_protocol_radius(run_dir, summary), radius, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label}: protocol radius mismatch")
    expected_candidate_radius = max(radius - MARGIN_M, 0.0)
    for label, summary in (
        ("immediate", immediate_summary),
        ("candidate", candidate_summary),
    ):
        if not math.isclose(
            _finite(summary.get("eval_goal_radius_m"), name=f"{label} radius"),
            expected_candidate_radius,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label}: eval_goal_radius_m mismatch")
        if not math.isclose(
            _finite(
                summary.get("eval_goal_plane_miss_guard_radius_m"),
                name=f"{label} plane guard",
            ),
            radius,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label}: plane-miss guard must equal protocol radius")
    immediate_dwell = int(immediate_summary.get("eval_goal_plane_miss_dwell_steps", 0))
    candidate_dwell = int(candidate_summary.get("eval_goal_plane_miss_dwell_steps", -1))
    if immediate_dwell != 0:
        raise ValueError("immediate reference must use dwell=0")
    if candidate_dwell != DWELL_STEPS:
        raise ValueError(f"candidate must use dwell={DWELL_STEPS}")
    return (
        base_rows,
        immediate_rows,
        candidate_rows,
        {
            "protocol_radius_m": radius,
            "candidate_radius_m": expected_candidate_radius,
            "checkpoint_sha256": candidate_summary.get("checkpoint_sha256"),
            "immediate_dwell_steps": immediate_dwell,
            "candidate_dwell_steps": candidate_dwell,
        },
    )


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    count = int(np.sum(denominator))
    return None if count == 0 else float(np.sum(numerator) / count)


def _median(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.median(values))


def _paired_metrics(
    base_rows: list[dict[str, Any]],
    immediate_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base_success = np.asarray([row["success"] for row in base_rows], dtype=bool)
    base_capture = np.asarray([row["terminal_eta_capture"] for row in base_rows], dtype=bool)
    immediate_success = np.asarray([row["success"] for row in immediate_rows], dtype=bool)
    immediate_capture = np.asarray(
        [row["terminal_eta_capture"] for row in immediate_rows], dtype=bool
    )
    candidate_success = np.asarray([row["success"] for row in candidate_rows], dtype=bool)
    candidate_capture = np.asarray(
        [row["terminal_eta_capture"] for row in candidate_rows], dtype=bool
    )
    base_false = base_success & ~base_capture
    base_true = base_success & base_capture
    immediate_false = immediate_success & ~immediate_capture
    candidate_false = candidate_success & ~candidate_capture
    candidate_true = candidate_success & candidate_capture
    removed = base_false & ~candidate_false
    retained = base_true & candidate_true
    false_to_true = base_false & candidate_true
    false_to_none = base_false & ~candidate_success

    base_steps = np.asarray([row["steps"] for row in base_rows], dtype=np.float64)
    candidate_steps = np.asarray([row["steps"] for row in candidate_rows], dtype=np.float64)
    base_p_nav = np.asarray(
        [row["terminal_p_nav_distance_m"] for row in base_rows], dtype=np.float64
    )
    candidate_p_nav = np.asarray(
        [row["terminal_p_nav_distance_m"] for row in candidate_rows], dtype=np.float64
    )
    base_eta = np.asarray([row["terminal_eta_distance_m"] for row in base_rows], dtype=np.float64)
    candidate_eta = np.asarray(
        [row["terminal_eta_distance_m"] for row in candidate_rows], dtype=np.float64
    )
    base_timeout = float(np.mean([row["timeout"] for row in base_rows]))
    candidate_timeout = float(np.mean([row["timeout"] for row in candidate_rows]))
    base_safety = float(np.mean([row["safety"] for row in base_rows]))
    candidate_safety = float(np.mean([row["safety"] for row in candidate_rows]))
    return {
        "n": len(base_rows),
        "baseline_success_count": int(np.sum(base_success)),
        "baseline_true_success_count": int(np.sum(base_true)),
        "baseline_false_success_count": int(np.sum(base_false)),
        "immediate_true_success_count": int(np.sum(immediate_success & immediate_capture)),
        "immediate_false_success_count": int(np.sum(immediate_false)),
        "candidate_success_count": int(np.sum(candidate_success)),
        "candidate_true_success_count": int(np.sum(candidate_true)),
        "candidate_false_success_count": int(np.sum(candidate_false)),
        "true_success_retained_count": int(np.sum(retained)),
        "false_success_removed_count": int(np.sum(removed)),
        "false_to_true_count": int(np.sum(false_to_true)),
        "false_to_none_count": int(np.sum(false_to_none)),
        "true_success_retention": _ratio(retained, base_true),
        "false_success_removed_fraction": _ratio(removed, base_false),
        "false_to_true_fraction": _ratio(false_to_true, base_false),
        "candidate_precision": _ratio(candidate_true, candidate_success),
        "baseline_success_rate": float(np.mean(base_success)),
        "candidate_success_rate": float(np.mean(candidate_success)),
        "baseline_capture_rate": float(np.mean(base_capture)),
        "candidate_capture_rate": float(np.mean(candidate_capture)),
        "baseline_timeout_rate": base_timeout,
        "candidate_timeout_rate": candidate_timeout,
        "timeout_delta": candidate_timeout - base_timeout,
        "baseline_safety_rate": base_safety,
        "candidate_safety_rate": candidate_safety,
        "safety_delta": candidate_safety - base_safety,
        "extra_steps_mean": float(np.mean(candidate_steps - base_steps)),
        "extra_steps_median": float(np.median(candidate_steps - base_steps)),
        "baseline_terminal_p_nav_median_m": _median(base_p_nav),
        "candidate_terminal_p_nav_median_m": _median(candidate_p_nav),
        "baseline_terminal_eta_median_m": _median(base_eta),
        "candidate_terminal_eta_median_m": _median(candidate_eta),
    }


def _h65_a(stats: dict[str, Any], scenario: str) -> bool:
    retention = stats["true_success_retention"]
    removed = stats["false_success_removed_fraction"]
    return bool(
        stats["baseline_false_success_count"] > 0
        and retention is not None
        and retention >= TRUE_RETENTION_MIN[scenario]
        and removed is not None
        and removed >= FALSE_REMOVAL_MIN
    )


def _h65_b(stats: dict[str, Any], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    false_guard = bool(
        stats["candidate_false_success_count"]
        <= stats["immediate_false_success_count"] + FALSE_SUCCESS_SLACK
    )
    precision_guard = bool(
        stats["candidate_precision"] is not None and stats["candidate_precision"] >= PRECISION_MIN
    )
    terminal_guard = bool(
        stats["timeout_delta"] <= TERMINAL_DELTA_MAX and stats["safety_delta"] <= TERMINAL_DELTA_MAX
    )
    checkpoint_pass_count = sum(
        runs[str(checkpoint)]["true_success_retention"] is not None
        and runs[str(checkpoint)]["true_success_retention"] >= CHECKPOINT_RETENTION_MIN
        for checkpoint in POLICY_SEEDS
    )
    checkpoint_guard = checkpoint_pass_count >= CHECKPOINT_PASS_COUNT_MIN
    return {
        "H65_B_supported": bool(
            false_guard and precision_guard and terminal_guard and checkpoint_guard
        ),
        "false_success_guard_passed": false_guard,
        "precision_guard_passed": precision_guard,
        "terminal_guard_passed": terminal_guard,
        "checkpoint_pass_count": int(checkpoint_pass_count),
        "checkpoint_guard_passed": checkpoint_guard,
    }


def _h65_c(stats: dict[str, Any]) -> bool:
    retention = stats["true_success_retention"]
    return bool(retention is not None and retention >= ZERO_RETENTION_MIN)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal65 terminal plane-miss dwell",
        "",
        "日期：2026-11-10<br>",
        "baseline 复用 Goal60 trace；immediate 复用 Goal63 dwell=0；"
        "candidate 只新增 eval-only 50-step dwell。",
        "dev 20–59；CPU/float64；不训练、不打开 held-out。",
        "",
    ]
    for scenario, result in payload["scenarios"].items():
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"protocol radius `{result['protocol_radius_m']:.2f} m`；candidate radius "
                f"`{result['candidate_radius_m']:.2f} m`。",
                "",
                "| run | base false | imm false | cand false | retained | precision | timeout Δ | safety Δ | extra steps |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in (*map(str, POLICY_SEEDS), "zero", "pooled"):
            stats = result["runs"][name] if name != "pooled" else result["pooled"]
            lines.append(
                f"| {name} | {stats['baseline_false_success_count']} | "
                f"{stats['immediate_false_success_count']} | "
                f"{stats['candidate_false_success_count']} | "
                f"{_fmt(stats['true_success_retention'])} | "
                f"{_fmt(stats['candidate_precision'])} | "
                f"{stats['timeout_delta']:+.3f} | {stats['safety_delta']:+.3f} | "
                f"{stats['extra_steps_median']:.1f} |"
            )
        h = result["hypotheses"]
        lines.extend(
            [
                "",
                f"- H65-A `{h['H65_A_supported']}`；H65-B `{h['H65_B_supported']}`；"
                f"H65-C `{h['H65_C_supported']}`。",
                f"- false guard `{h['false_success_guard_passed']}`；precision guard "
                f"`{h['precision_guard_passed']}`；terminal guard "
                f"`{h['terminal_guard_passed']}`；checkpoint pass "
                f"`{h['checkpoint_pass_count']}/4`。",
                "",
            ]
        )
    overall = payload["overall"]
    lines.extend(
        [
            "## Overall",
            "",
            f"- status: `{overall['status']}`",
            f"- H65-A scenarios: {overall['H65_A_scenarios']}",
            f"- H65-B scenarios: {overall['H65_B_scenarios']}",
            f"- H65-C scenarios: {overall['H65_C_scenarios']}",
            f"- held-out opened: {overall['heldout_opened']}；training started: "
            f"{overall['training_started']}；new_rollouts: {overall['new_rollouts']}",
            "",
        ]
    )
    return "\n".join(lines)


def _overall_status(scenarios: dict[str, Any]) -> str:
    if all(
        result["hypotheses"][key]
        for result in scenarios.values()
        for key in ("H65_A_supported", "H65_B_supported", "H65_C_supported")
    ):
        return "DWELL_PASS_DIAGNOSTIC"
    if any(
        not result["hypotheses"]["false_success_guard_passed"]
        or not result["hypotheses"]["terminal_guard_passed"]
        for result in scenarios.values()
    ):
        return "DWELL_INVALID_FALSE_SUCCESS"
    if any(not result["hypotheses"]["H65_A_supported"] for result in scenarios.values()):
        return "DWELL_NOT_SUFFICIENT"
    return "DWELL_PARTIAL_DIAGNOSTIC"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        runs: dict[str, dict[str, Any]] = {}
        pooled_base: list[dict[str, Any]] = []
        pooled_immediate: list[dict[str, Any]] = []
        pooled_candidate: list[dict[str, Any]] = []
        protocol_radius: float | None = None
        candidate_radius: float | None = None
        for checkpoint in POLICY_SEEDS:
            name = f"{scenario}_policy_s{checkpoint}"
            base_rows, immediate_rows, candidate_rows, meta = _validate_run(
                args.baseline_root / name,
                args.immediate_root / f"{name}{args.immediate_suffix}",
                args.candidate_root / f"{name}{args.candidate_suffix}",
                expected_action_mode="policy",
            )
            if protocol_radius is None:
                protocol_radius = float(meta["protocol_radius_m"])
                candidate_radius = float(meta["candidate_radius_m"])
            runs[str(checkpoint)] = _paired_metrics(base_rows, immediate_rows, candidate_rows)
            pooled_base.extend(base_rows)
            pooled_immediate.extend(immediate_rows)
            pooled_candidate.extend(candidate_rows)

        zero_name = f"{scenario}_zero"
        zero_base, zero_immediate, zero_candidate, _ = _validate_run(
            args.baseline_root / zero_name,
            args.immediate_root / f"{zero_name}{args.immediate_suffix}",
            args.candidate_root / f"{zero_name}{args.candidate_suffix}",
            expected_action_mode="zero-residual",
        )
        runs["zero"] = _paired_metrics(zero_base, zero_immediate, zero_candidate)
        pooled = _paired_metrics(pooled_base, pooled_immediate, pooled_candidate)
        hypotheses = {
            "H65_A_supported": _h65_a(pooled, scenario),
            "H65_C_supported": _h65_c(runs["zero"]),
            **_h65_b(pooled, runs),
            "true_retention_min": TRUE_RETENTION_MIN[scenario],
            "false_removal_min": FALSE_REMOVAL_MIN,
            "precision_min": PRECISION_MIN,
            "checkpoint_retention_min": CHECKPOINT_RETENTION_MIN,
            "zero_retention_min": ZERO_RETENTION_MIN,
        }
        scenarios[scenario] = {
            "protocol_radius_m": protocol_radius,
            "candidate_radius_m": candidate_radius,
            "runs": runs,
            "pooled": pooled,
            "hypotheses": hypotheses,
        }

    overall = {
        "status": _overall_status(scenarios),
        "H65_A_scenarios": [
            name for name, result in scenarios.items() if result["hypotheses"]["H65_A_supported"]
        ],
        "H65_B_scenarios": [
            name for name, result in scenarios.items() if result["hypotheses"]["H65_B_supported"]
        ],
        "H65_C_scenarios": [
            name for name, result in scenarios.items() if result["hypotheses"]["H65_C_supported"]
        ],
        "heldout_opened": False,
        "training_started": False,
        "new_rollouts": True,
    }
    payload = {"schema": SCHEMA_VERSION, "scenarios": scenarios, "overall": overall}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
