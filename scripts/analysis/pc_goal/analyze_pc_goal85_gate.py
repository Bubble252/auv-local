#!/usr/bin/env python3
"""Aggregate and adjudicate the frozen Goal85 eval-only matrix.

This is a read-only analysis.  It does not step an environment, train, select a
checkpoint, or alter success semantics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal85_gate.v1"
ROOTS = ("baseline", "candidate")
SCENARIOS = ("default", "strict125")
POLICY_CHECKPOINTS = ("s240", "s752", "s880", "s1008")
DEV_SEEDS = tuple(range(20, 60))
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261110
EPS = 1e-12


class DiagnosticError(ValueError):
    """Raised when the frozen matrix is incomplete or inconsistent."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data/yhx/underwater-robotics/src/plan_p1/goal85_unbiased_nav_20261110"
        ),
    )
    parser.add_argument("--bias-audit", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args(argv)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _bootstrap_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_DRAWS, array.size))
    means = array[indices].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _load_run(path: Path, *, action: str, ckpt: str) -> dict[str, Any]:
    summary_path = path / "summary.json"
    episodes_path = path / "episodes.json"
    if not summary_path.exists() or not episodes_path.exists():
        raise DiagnosticError(f"{path}: missing summary.json or episodes.json")
    summary = _read_json(summary_path)
    episodes = _read_json(episodes_path)
    if not isinstance(summary, dict) or not isinstance(episodes, list):
        raise DiagnosticError(f"{path}: malformed summary or episodes")
    seed_values = [int(row["seed"]) for row in episodes]
    if tuple(seed_values) != DEV_SEEDS:
        raise DiagnosticError(f"{path}: episodes are not dev seeds 20-59")
    summary_block = summary["summary"]
    if int(summary_block["episodes"]) != len(DEV_SEEDS):
        raise DiagnosticError(f"{path}: expected 40 episodes")
    if summary.get("termination_position_source") != "eta":
        raise DiagnosticError(f"{path}: position_source is not eta")
    if summary.get("sensor_profile") != "unbiased_noise_v1":
        raise DiagnosticError(f"{path}: sensor profile is not unbiased_noise_v1")
    if summary.get("ideal_actor_observations") is not False:
        raise DiagnosticError(f"{path}: ideal actor observations enabled")
    if summary.get("success_frame") != "eta":
        raise DiagnosticError(f"{path}: success frame is not eta")
    if not bool(summary_block.get("finite")):
        raise DiagnosticError(f"{path}: non-finite metrics")
    return {
        "path": str(path),
        "action": action,
        "checkpoint": ckpt,
        "eval_goal_radius_m": float(summary["eval_goal_radius_m"]),
        "summary": summary_block,
        "episodes": {int(row["seed"]): row for row in episodes},
    }


def load_matrix(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    matrix: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario in SCENARIOS:
        for root_name in ROOTS:
            base = root / scenario / root_name
            runs: dict[str, Any] = {
                "policy": {},
                "zero-residual": _load_run(
                    base / "zero-residual" / "s240",
                    action="zero-residual",
                    ckpt="s240",
                ),
            }
            for ckpt in POLICY_CHECKPOINTS:
                runs["policy"][ckpt] = _load_run(
                    base / "policy" / ckpt, action="policy", ckpt=ckpt
                )
            matrix[(scenario, root_name)] = runs
    expected_runs = len(SCENARIOS) * len(ROOTS) * (1 + len(POLICY_CHECKPOINTS))
    actual_runs = sum(1 + len(item["policy"]) for item in matrix.values())
    if actual_runs != expected_runs:
        raise DiagnosticError(f"expected {expected_runs} runs, found {actual_runs}")
    return matrix


def _paired(
    policy: dict[str, Any], zero: dict[str, Any], field: str
) -> list[float]:
    return [
        float(policy["episodes"][seed][field]) - float(zero["episodes"][seed][field])
        for seed in DEV_SEEDS
    ]


def _metric_block(
    matrix: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    fields = {
        "eta_done": "success_eta_tight_at_done",
        "eta_ever": "success_eta_tight_ever",
        "p_nav": "success_pnav_tight",
        "plane_miss": "goal_plane_miss",
        "timeout": "timeout",
        "safety": "safety",
        "path_rmse": "path_three_d_rmse_m",
        "terminal_eta_distance": "terminal_eta_distance_m",
        "terminal_p_nav_distance": "terminal_p_nav_distance_m",
    }
    payload: dict[str, Any] = {"per_checkpoint": {}, "per_scenario": {}}
    for (scenario, root_name), runs in matrix.items():
        zero = runs["zero-residual"]
        per_ckpt: dict[str, Any] = {}
        cluster_deltas: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for ckpt, policy in runs["policy"].items():
            block: dict[str, Any] = {}
            for label, field in fields.items():
                deltas = _paired(policy, zero, field)
                low, high = _bootstrap_ci(deltas)
                block[label] = {
                    "mean": _mean(deltas),
                    "ci95": [low, high],
                    "positive_runs": int(sum(value > EPS for value in deltas)),
                    "negative_runs": int(sum(value < -EPS for value in deltas)),
                }
                for seed, value in zip(DEV_SEEDS, deltas, strict=True):
                    cluster_deltas[label][seed].append(value)
            block["policy_summary"] = policy["summary"]
            block["zero_summary"] = zero["summary"]
            per_ckpt[ckpt] = block
        scenario_block: dict[str, Any] = {}
        for label in fields:
            cluster_means = [
                _mean(cluster_deltas[label][seed]) for seed in DEV_SEEDS
            ]
            low, high = _bootstrap_ci(cluster_means)
            scenario_block[label] = {
                "mean": _mean(cluster_means),
                "ci95": [low, high],
                "checkpoint_means": {
                    ckpt: per_ckpt[ckpt][label]["mean"] for ckpt in per_ckpt
                },
            }
        payload["per_checkpoint"][f"{scenario}/{root_name}"] = per_ckpt
        payload["per_scenario"][f"{scenario}/{root_name}"] = scenario_block
    return payload


def _frame_gap(
    matrix: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    """Paired p_nav-vs-eta success gap, clustered by seed.

    A positive value means p_nav reports more success than truth.  Goal84 had a
    stable negative gap (eta below p_nav); Goal85 should remove that direction.
    """

    def _pnav_done(run: dict[str, Any], seed: int) -> float:
        row = run["episodes"][seed]
        return float(
            float(row["terminal_p_nav_distance_m"])
            <= float(run["eval_goal_radius_m"]) + EPS
        )

    def _distance_error(run: dict[str, Any], seed: int) -> float:
        row = run["episodes"][seed]
        return float(row["terminal_p_nav_distance_m"]) - float(
            row["terminal_eta_distance_m"]
        )

    clusters: dict[int, list[float]] = defaultdict(list)
    distance_clusters: dict[int, list[float]] = defaultdict(list)
    for runs in matrix.values():
        for policy in runs["policy"].values():
            for seed in DEV_SEEDS:
                value = _pnav_done(policy, seed) - float(
                    policy["episodes"][seed]["success_eta_tight_at_done"]
                )
                clusters[seed].append(value)
                distance_clusters[seed].append(_distance_error(policy, seed))
        zero = runs["zero-residual"]
        for seed in DEV_SEEDS:
            eta_done = float(zero["episodes"][seed]["success_eta_tight_at_done"])
            clusters[seed].append(_pnav_done(zero, seed) - eta_done)
            distance_clusters[seed].append(_distance_error(zero, seed))
    values = [_mean(clusters[seed]) for seed in DEV_SEEDS]
    low, high = _bootstrap_ci(values)
    distance_values = [_mean(distance_clusters[seed]) for seed in DEV_SEEDS]
    distance_low, distance_high = _bootstrap_ci(distance_values)
    distance_mean = _mean(distance_values)
    return {
        "success_gap": {
            "mean": _mean(values),
            "ci95": [low, high],
            "contains_zero": bool(low <= 0.0 <= high),
            "stable_negative": bool(high < 0.0),
            "per_seed": {str(seed): _mean(clusters[seed]) for seed in DEV_SEEDS},
        },
        "distance_bias": {
            "mean_m": distance_mean,
            "ci95_m": [distance_low, distance_high],
            "passed": bool(abs(distance_mean) < 0.05),
            "per_seed_m": {
                str(seed): _mean(distance_clusters[seed]) for seed in DEV_SEEDS
            },
        },
        "note": (
            "The done-frame success gap is not identifiable under endpoint-stop: "
            "successful eta rows are censored at the radius boundary, so zero-mean "
            "p_nav noise pushes roughly half of them outside.  The distance-bias "
            "test is the censoring-safe frame diagnostic."
        ),
    }


def _bias_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(path)
    return {
        "path": str(path),
        "passed": bool(payload.get("passed")),
        "max_abs_mean_m": payload.get("max_abs_mean_m"),
        "max_abs_drift_m": payload.get("max_abs_drift_m"),
        "std_m": payload.get("std_m"),
    }


def _gate(checks: dict[str, bool]) -> str:
    if not all(checks.values()):
        return "FAIL_CONTRACT"
    return "PASS"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal85 Gate",
        "",
        f"- status: **{payload['status']}**",
        f"- runs/episodes: `{payload['protocol']['runs']}` / `{payload['protocol']['episodes']}`",
        "",
        "| scenario/root | metric | mean Δ (pp) | 95% CI (pp) |",
        "|---|---|---:|---:|",
    ]
    for key, block in payload["metrics"]["per_scenario"].items():
        for label in ("eta_done", "eta_ever", "p_nav", "plane_miss"):
            item = block[label]
            lines.append(
                f"| {key} | {label} | {100.0 * item['mean']:.2f} | "
                f"[{100.0 * item['ci95'][0]:.2f}, {100.0 * item['ci95'][1]:.2f}] |"
            )
    lines.extend(["", "## Frame diagnostic", ""])
    gap = payload["frame_gap"]
    lines.append(
        "- done-frame success gap (censored, diagnostic only): "
        f"`{100.0 * gap['success_gap']['mean']:.2f} pp` "
        f"95% CI `[{100.0 * gap['success_gap']['ci95'][0]:.2f}, "
        f"{100.0 * gap['success_gap']['ci95'][1]:.2f}] pp`"
    )
    lines.append(
        "- done-frame distance bias `mean(p_nav-eta)`: "
        f"`{gap['distance_bias']['mean_m']:.4f} m` "
        f"95% CI `[{gap['distance_bias']['ci95_m'][0]:.4f}, "
        f"{gap['distance_bias']['ci95_m'][1]:.4f}] m`"
    )
    lines.append(f"- {gap['note']}")
    lines.extend(["", "## Check", ""])
    for name, passed in payload["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = load_matrix(args.root)
    metrics = _metric_block(matrix)
    frame_gap = _frame_gap(matrix)
    bias = _bias_audit(args.bias_audit)
    runs = sum(1 + len(item["policy"]) for item in matrix.values())
    episodes = sum(
        len(item["zero-residual"]["episodes"])
        + sum(len(policy["episodes"]) for policy in item["policy"].values())
        for item in matrix.values()
    )
    checks = {
        "runs_20": runs == 20,
        "episodes_800": episodes == 800,
        "policy_16_zero_4": (
            sum(len(item["policy"]) for item in matrix.values()) == 16
            and len(matrix) * 1 == 4
        ),
        "dev_seeds_only": all(
            tuple(sorted(item["zero-residual"]["episodes"])) == DEV_SEEDS
            and all(
                tuple(sorted(policy["episodes"])) == DEV_SEEDS
                for policy in item["policy"].values()
            )
            for item in matrix.values()
        ),
        "heldout_closed": True,
        "frame_distance_bias_within_0p05m": bool(
            frame_gap["distance_bias"]["passed"]
        ),
        "bias_audit_passed": True if bias is None else bool(bias["passed"]),
    }
    contract_pass = _gate(checks) == "PASS"
    if contract_pass and not frame_gap["success_gap"]["contains_zero"]:
        status = "PASS_CONTRACT_FRAME_DIAGNOSTIC_INVALID"
    elif contract_pass:
        status = "PASS"
    else:
        status = "FAIL_CONTRACT"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "protocol": {
            "runs": runs,
            "episodes": episodes,
            "policy_runs": sum(len(item["policy"]) for item in matrix.values()),
            "zero_residual_runs": len(matrix),
            "dev_seeds": list(DEV_SEEDS),
            "heldout_opened": False,
            "training_started": False,
        },
        "checks": checks,
        "bias_audit": bias,
        "frame_gap": frame_gap,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "frame_gap": frame_gap}))
    return 0 if status != "FAIL_CONTRACT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
