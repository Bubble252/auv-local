#!/usr/bin/env python3
"""Goal63 paired analysis for frozen-checkpoint terminal suppression."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal63_terminal_suppression.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
MARGIN_M = 0.25
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_261_110
FALSE_REMOVAL_MIN = 0.50
TRUE_RETENTION_MIN = 0.90
CHECKPOINT_FALSE_REMOVAL_MIN = 0.40
CHECKPOINT_TRUE_RETENTION_MIN = 0.85
CHECKPOINT_PASS_COUNT_MIN = 3
ZERO_TRUE_RETENTION_MIN = 0.80
SAFETY_DELTA_MAX = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/traces"))
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/goal63_terminal_suppression"),
    )
    parser.add_argument(
        "--candidate-suffix",
        default="_m025",
        help="Suffix appended to scenario/checkpoint run directory names",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal63_terminal_suppression/analysis/goal63.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal63_terminal_suppression/analysis/goal63.md"),
    )
    return parser.parse_args(argv)


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _summary(trace_dir: Path) -> dict[str, Any]:
    return json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))


def _episodes(trace_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads((trace_dir / "episodes.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{trace_dir}: episodes must be a list")
    if [int(row["seed"]) for row in rows] != list(DEV_SEEDS):
        raise ValueError(f"{trace_dir}: episode seeds must be exactly 20-59")
    traced_p_nav = _trace_terminal_p_nav(trace_dir, rows)
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
                raise ValueError(f"{trace_dir}: missing episode field {key}")
        _finite(row["terminal_eta_distance_m"], name=f"{trace_dir}:terminal_eta_distance_m")
        if "terminal_p_nav_distance_m" not in row:
            row["terminal_p_nav_distance_m"] = traced_p_nav.get(int(row["seed"]))
        if row["terminal_p_nav_distance_m"] is not None:
            _finite(
                row["terminal_p_nav_distance_m"],
                name=f"{trace_dir}:terminal_p_nav_distance_m",
            )
    if any(seed in HELDOUT_SEEDS for seed in DEV_SEEDS):
        raise ValueError("held-out seed in contract")
    return rows


def _trace_terminal_p_nav(
    trace_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[int, float]:
    trace_path = trace_dir / "trace.npz"
    if not trace_path.is_file():
        return {}
    result: dict[int, float] = {}
    with np.load(trace_path, allow_pickle=False) as trace:
        for row in rows:
            seed = int(row["seed"])
            terminal = np.flatnonzero(
                (trace["seed"] == seed) & trace["done"].astype(bool)
            )
            if terminal.size != 1:
                raise ValueError(
                    f"{trace_dir}: seed {seed} must have one terminal trace row"
                )
            index = int(terminal[0])
            p_nav = np.asarray(trace["p_nav_ned"][index], dtype=np.float64)
            goal = np.asarray(trace["goal_ned"][index], dtype=np.float64)
            result[seed] = _finite(
                np.linalg.norm(p_nav - goal),
                name=f"{trace_dir}:terminal_p_nav_distance_m",
            )
    return result


def _protocol_radius(trace_dir: Path, summary: dict[str, Any]) -> float:
    config_path = Path(str(summary["environment_config"]))
    if not config_path.is_absolute():
        config_path = trace_dir / config_path
        if not config_path.is_file():
            config_path = Path(__file__).resolve().parents[3] / str(summary["environment_config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return float(config["plant"]["goal_radius_m"])


def _validate_pair(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    expected_action_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_summary = _summary(baseline_dir)
    cand_summary = _summary(candidate_dir)
    base_rows = _episodes(baseline_dir)
    cand_rows = _episodes(candidate_dir)
    for label, summary in (("baseline", base_summary), ("candidate", cand_summary)):
        if summary.get("termination_mode") != "endpoint_stop":
            raise ValueError(f"{label}: expected endpoint_stop")
        if summary.get("device") != "cpu" or summary.get("dtype") != "float64":
            raise ValueError(f"{label}: expected CPU/float64")
        if summary.get("summary", {}).get("finite") is not True:
            raise ValueError(f"{label}: summary is not finite")
        if summary.get("summary", {}).get("action_mode") != expected_action_mode:
            raise ValueError(f"{label}: action mode mismatch")
    if base_summary.get("checkpoint_sha256") != cand_summary.get("checkpoint_sha256"):
        raise ValueError("baseline/candidate checkpoint SHA mismatch")
    radius = _protocol_radius(baseline_dir, base_summary)
    candidate_radius = _protocol_radius(candidate_dir, cand_summary)
    if not math.isclose(radius, candidate_radius, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("baseline/candidate protocol radius mismatch")
    expected_candidate_radius = max(radius - MARGIN_M, 0.0)
    if not math.isclose(
        _finite(cand_summary.get("eval_goal_radius_m"), name="candidate radius"),
        expected_candidate_radius,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate eval_goal_radius_m mismatch")
    if not math.isclose(
        _finite(
            cand_summary.get("eval_goal_plane_miss_guard_radius_m"),
            name="candidate plane guard",
        ),
        radius,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate plane-miss guard must equal the protocol radius")
    return base_rows, cand_rows, {
        "protocol_radius_m": radius,
        "candidate_radius_m": expected_candidate_radius,
        "checkpoint_sha256": cand_summary.get("checkpoint_sha256"),
    }


def _bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
) -> dict[str, float] | None:
    if int(np.sum(denominator)) == 0:
        return None
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, denominator.size, denominator.size)
        draw_denominator = int(np.sum(denominator[indices]))
        if draw_denominator == 0:
            continue
        draws.append(float(np.sum(numerator[indices]) / draw_denominator))
    if not draws:
        return None
    return {
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    count = int(np.sum(denominator))
    return None if count == 0 else float(np.sum(numerator) / count)


def _paired_metrics(
    base_rows: list[dict[str, Any]],
    cand_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    base_success = np.asarray([row["success"] for row in base_rows], dtype=bool)
    base_capture = np.asarray([row["terminal_eta_capture"] for row in base_rows], dtype=bool)
    cand_success = np.asarray([row["success"] for row in cand_rows], dtype=bool)
    cand_capture = np.asarray(
        [row["terminal_eta_capture"] for row in cand_rows], dtype=bool
    )
    base_false = base_success & ~base_capture
    base_true = base_success & base_capture
    cand_false = cand_success & ~cand_capture
    cand_true = cand_success & cand_capture
    removed = base_false & ~cand_false
    retained = base_true & cand_true
    false_to_true = base_false & cand_true
    false_to_none = base_false & ~cand_success
    base_steps = np.asarray([row["steps"] for row in base_rows], dtype=np.float64)
    cand_steps = np.asarray([row["steps"] for row in cand_rows], dtype=np.float64)
    extra_steps = cand_steps - base_steps
    base_p_nav = _finite_values(
        [row.get("terminal_p_nav_distance_m") for row in base_rows]
    )
    cand_p_nav = _finite_values(
        [row.get("terminal_p_nav_distance_m") for row in cand_rows]
    )
    base_eta = np.asarray(
        [row["terminal_eta_distance_m"] for row in base_rows], dtype=np.float64
    )
    cand_eta = np.asarray(
        [row["terminal_eta_distance_m"] for row in cand_rows], dtype=np.float64
    )
    return {
        "n": len(base_rows),
        "baseline_success_count": int(np.sum(base_success)),
        "baseline_capture_count": int(np.sum(base_capture)),
        "baseline_false_success_count": int(np.sum(base_false)),
        "baseline_true_success_count": int(np.sum(base_true)),
        "candidate_success_count": int(np.sum(cand_success)),
        "candidate_capture_count": int(np.sum(cand_capture)),
        "candidate_false_success_count": int(np.sum(cand_false)),
        "candidate_true_success_count": int(np.sum(cand_true)),
        "false_success_removed_count": int(np.sum(removed)),
        "true_success_retained_count": int(np.sum(retained)),
        "false_to_true_count": int(np.sum(false_to_true)),
        "false_to_none_count": int(np.sum(false_to_none)),
        "false_success_removed_fraction": _ratio(removed, base_false),
        "false_success_removed_ci": _bootstrap_ratio(removed, base_false, seed=seed),
        "true_success_retention": _ratio(retained, base_true),
        "true_success_retention_ci": _bootstrap_ratio(retained, base_true, seed=seed + 1),
        "false_to_true_fraction": _ratio(false_to_true, base_false),
        "false_to_true_ci": _bootstrap_ratio(false_to_true, base_false, seed=seed + 2),
        "candidate_precision": _ratio(cand_true, cand_success),
        "candidate_precision_ci": _bootstrap_ratio(cand_true, cand_success, seed=seed + 3),
        "baseline_success_rate": float(np.mean(base_success)),
        "candidate_success_rate": float(np.mean(cand_success)),
        "baseline_capture_rate": float(np.mean(base_capture)),
        "candidate_capture_rate": float(np.mean(cand_capture)),
        "baseline_plane_miss_rate": float(
            np.mean([row["goal_plane_miss"] for row in base_rows])
        ),
        "candidate_plane_miss_rate": float(
            np.mean([row["goal_plane_miss"] for row in cand_rows])
        ),
        "baseline_timeout_rate": float(np.mean([row["timeout"] for row in base_rows])),
        "candidate_timeout_rate": float(np.mean([row["timeout"] for row in cand_rows])),
        "baseline_safety_rate": float(np.mean([row["safety"] for row in base_rows])),
        "candidate_safety_rate": float(np.mean([row["safety"] for row in cand_rows])),
        "extra_steps_mean": float(np.mean(extra_steps)),
        "extra_steps_median": float(np.median(extra_steps)),
        "baseline_terminal_p_nav_distance_median_m": _median_or_none(base_p_nav),
        "candidate_terminal_p_nav_distance_median_m": _median_or_none(cand_p_nav),
        "baseline_terminal_eta_distance_median_m": float(np.median(base_eta)),
        "candidate_terminal_eta_distance_median_m": float(np.median(cand_eta)),
    }


def _finite_values(values: list[Any]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None],
        dtype=np.float64,
    )


def _median_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.median(values))


def _rule_pass(stats: dict[str, Any]) -> bool:
    removed = stats["false_success_removed_fraction"]
    retention = stats["true_success_retention"]
    return bool(
        stats["baseline_false_success_count"] > 0
        and removed is not None
        and removed >= FALSE_REMOVAL_MIN
        and retention is not None
        and retention >= TRUE_RETENTION_MIN
        and stats["baseline_safety_rate"] + SAFETY_DELTA_MAX
        >= stats["candidate_safety_rate"]
    )


def _checkpoint_pass(stats: dict[str, Any]) -> bool:
    removed = stats["false_success_removed_fraction"]
    retention = stats["true_success_retention"]
    return bool(
        removed is not None
        and removed >= CHECKPOINT_FALSE_REMOVAL_MIN
        and retention is not None
        and retention >= CHECKPOINT_TRUE_RETENTION_MIN
    )


def _zero_pass(stats: dict[str, Any]) -> bool:
    removed = stats["false_success_removed_fraction"]
    retention = stats["true_success_retention"]
    return bool(
        removed is not None
        and removed > 0.0
        and retention is not None
        and retention >= ZERO_TRUE_RETENTION_MIN
    )


def _hypotheses(
    pooled: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    zero: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_count = sum(
        _checkpoint_pass(runs[str(checkpoint)]) for checkpoint in POLICY_SEEDS
    )
    h_a = _rule_pass(pooled)
    h_b = bool(h_a and checkpoint_count >= CHECKPOINT_PASS_COUNT_MIN)
    h_c = _zero_pass(zero)
    return {
        "H63_A_supported": h_a,
        "H63_B_supported": h_b,
        "H63_C_supported": h_c,
        "checkpoint_pass_count": checkpoint_count,
        "safety_guard_passed": bool(
            pooled["baseline_safety_rate"] + SAFETY_DELTA_MAX
            >= pooled["candidate_safety_rate"]
        ),
    }


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
        "# Goal63 frozen-checkpoint terminal suppression",
        "",
        "日期：2026-11-10<br>",
        "baseline 复用 Goal60 trace；candidate 仅改 eval-only GOAL radius 与 plane-miss guard。",
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
                "| run | base false | cand false | removed | true retained | false→true | timeout Δ | plane Δ | extra steps |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in (*map(str, POLICY_SEEDS), "zero", "pooled"):
            stats = result["runs"][name] if name != "pooled" else result["pooled"]
            lines.append(
                f"| {name} | {stats['baseline_false_success_count']} | "
                f"{stats['candidate_false_success_count']} | "
                f"{_fmt(stats['false_success_removed_fraction'])} | "
                f"{_fmt(stats['true_success_retention'])} | "
                f"{_fmt(stats['false_to_true_fraction'])} | "
                f"{stats['candidate_timeout_rate'] - stats['baseline_timeout_rate']:+.3f} | "
                f"{stats['candidate_plane_miss_rate'] - stats['baseline_plane_miss_rate']:+.3f} | "
                f"{stats['extra_steps_median']:.1f} |"
            )
        h = result["hypotheses"]
        lines.extend(
            [
                "",
                f"- H63-A `{h['H63_A_supported']}`；H63-B `{h['H63_B_supported']}`；"
                f"H63-C `{h['H63_C_supported']}`。",
                f"- checkpoint pass `{h['checkpoint_pass_count']}/4`；"
                f"safety guard `{h['safety_guard_passed']}`。",
                "",
            ]
        )
    overall = payload["overall"]
    lines.extend(
        [
            "## Overall",
            "",
            f"- status: `{overall['status']}`",
            f"- H63-A scenarios: {overall['H63_A_scenarios']}",
            f"- H63-B scenarios: {overall['H63_B_scenarios']}",
            f"- H63-C scenarios: {overall['H63_C_scenarios']}",
            f"- held-out opened: {overall['heldout_opened']}；training started: "
            f"{overall['training_started']}；new rollouts: {overall['new_rollouts']}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios: dict[str, Any] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        runs: dict[str, dict[str, Any]] = {}
        policy_base_rows: list[dict[str, Any]] = []
        policy_cand_rows: list[dict[str, Any]] = []
        protocol_radius: float | None = None
        candidate_radius: float | None = None
        for checkpoint_index, checkpoint in enumerate(POLICY_SEEDS):
            name = f"{scenario}_policy_s{checkpoint}"
            base_dir = args.baseline_root / name
            cand_dir = args.candidate_root / f"{name}{args.candidate_suffix}"
            base_rows, cand_rows, meta = _validate_pair(
                base_dir,
                cand_dir,
                expected_action_mode="policy",
            )
            if protocol_radius is None:
                protocol_radius = float(meta["protocol_radius_m"])
                candidate_radius = float(meta["candidate_radius_m"])
            elif not math.isclose(
                protocol_radius, float(meta["protocol_radius_m"]), abs_tol=1e-12
            ):
                raise ValueError(f"{scenario}: protocol radius mismatch across checkpoints")
            policy_base_rows.extend(base_rows)
            policy_cand_rows.extend(cand_rows)
            runs[str(checkpoint)] = _paired_metrics(
                base_rows,
                cand_rows,
                seed=BOOTSTRAP_SEED
                + scenario_index * 1000
                + checkpoint_index * 100,
            )
        zero_name = f"{scenario}_zero"
        zero_base_rows, zero_cand_rows, zero_meta = _validate_pair(
            args.baseline_root / zero_name,
            args.candidate_root / f"{zero_name}{args.candidate_suffix}",
            expected_action_mode="zero-residual",
        )
        runs["zero"] = _paired_metrics(
            zero_base_rows,
            zero_cand_rows,
            seed=BOOTSTRAP_SEED + scenario_index * 1000 + 500,
        )
        pooled = _paired_metrics(
            policy_base_rows,
            policy_cand_rows,
            seed=BOOTSTRAP_SEED + scenario_index * 1000 + 700,
        )
        scenarios[scenario] = {
            "protocol_radius_m": protocol_radius,
            "candidate_radius_m": candidate_radius,
            "runs": runs,
            "pooled": pooled,
            "hypotheses": _hypotheses(pooled, runs, runs["zero"]),
        }
    status = "FAIL_TERMINAL_SUPPRESSION"
    if all(
        value["hypotheses"]["H63_A_supported"]
        and value["hypotheses"]["H63_B_supported"]
        and value["hypotheses"]["H63_C_supported"]
        for value in scenarios.values()
    ):
        status = "PASS_TERMINAL_SUPPRESSION"
    elif any(value["hypotheses"]["H63_A_supported"] for value in scenarios.values()):
        status = "PARTIAL_TERMINAL_SUPPRESSION"
    payload = {
        "schema": SCHEMA_VERSION,
        "scenarios": scenarios,
        "overall": {
            "status": status,
            "H63_A_scenarios": [
                name
                for name, value in scenarios.items()
                if value["hypotheses"]["H63_A_supported"]
            ],
            "H63_B_scenarios": [
                name
                for name, value in scenarios.items()
                if value["hypotheses"]["H63_B_supported"]
            ],
            "H63_C_scenarios": [
                name
                for name, value in scenarios.items()
                if value["hypotheses"]["H63_C_supported"]
            ],
            "heldout_opened": False,
            "training_started": False,
            "new_rollouts": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
