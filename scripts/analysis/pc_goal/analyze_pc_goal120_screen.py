#!/usr/bin/env python3
"""Aggregate Goal120 PPO optimization-repair screen and apply the frozen Gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("artifacts/lowlevel/goal120_ppo_optimization_20261111")
GOAL119_ROOT = Path("artifacts/lowlevel/goal119_ppo_credit_approach_20261111")
ARMS = ("P0", "P1", "P2", "P3")
TRAIN_SEEDS = (86201, 86202)
DEV_SEEDS = tuple(range(20, 60))
UNREACHABLE_SEEDS = (26, 29, 33, 36, 49)
REACHABLE_FAILURE_SEEDS = (25, 27, 31, 50, 51)
KL_SIGNAL = 1e-3
SURROGATE_SIGNAL = 1e-3
SUCCESS_TARGET = 24.0
SUCCESS_HARM = 18.0
TERMINAL_IMPROVEMENT = 0.1
ANALYZER = Path("scripts/analyze_pc_goal119_screen.py")


class ScreenError(RuntimeError):
    pass


def _load_goal119():
    spec = importlib.util.spec_from_file_location("goal119_screen_analyzer", ANALYZER)
    if spec is None or spec.loader is None:
        raise ScreenError(f"cannot load {ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_G119 = _load_goal119()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--goal119-root", type=Path, default=GOAL119_ROOT)
    parser.add_argument("--eval-subdir", default="eval_policy")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args(argv)


def _training_dir(root: Path, goal119_root: Path, arm: str, seed: int) -> Path:
    if arm == "P0":
        return goal119_root / f"C3_s{seed}"
    return root / f"{arm}_s{seed}"


def _last_iteration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScreenError(f"missing training records: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ScreenError(f"empty training records: {path}")
    return rows[-1]


def _training_metrics(root: Path, goal119_root: Path, arm: str, seed: int) -> dict[str, Any]:
    run_dir = _training_dir(root, goal119_root, arm, seed)
    last = _last_iteration(run_dir / "iteration_records.jsonl")
    losses = last.get("losses") or {}
    policy = last.get("policy_update_metrics") or {}
    episode = last.get("episode_metrics") or {}
    return {
        "training_run": str(run_dir),
        "training_iteration": int(last.get("iteration", -1)),
        "total_env_steps": int(last.get("total_env_steps", -1)),
        "approx_kl": float(policy.get("approx_kl", math.nan)),
        "clip_fraction": float(policy.get("clip_fraction", math.nan)),
        "surrogate": float(losses.get("surrogate", math.nan)),
        "value_loss": float(losses.get("value_function", math.nan)),
        "train_success_rate": float(episode.get("/episode/success_rate", math.nan)),
        "train_episode_length": float(last.get("mean_episode_length", math.nan)),
        "losses_finite": bool(last.get("losses_finite", True)),
    }


def _finite(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _mean(values: Sequence[float]) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite else math.nan


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            run = _G119.analyze_run(args.root, arm, seed, args.eval_subdir)
            run.update(_training_metrics(args.root, args.goal119_root, arm, seed))
            runs.append(run)
    return {"runs": runs, "gate": decide(runs)}


def decide(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for run in runs:
        by_arm[run["arm"]].append(run)
    for arm in ARMS:
        by_arm[arm].sort(key=lambda row: row["train_seed"])

    arm_summary: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        if len(rows) != len(TRAIN_SEEDS):
            raise ScreenError(f"{arm}: expected {len(TRAIN_SEEDS)} training seeds")
        arm_summary[arm] = {
            "success_by_seed": [int(row["success"]) for row in rows],
            "success_mean": float(np.mean([row["success"] for row in rows])),
            "timeout_by_seed": [int(row["timeout"]) for row in rows],
            "safety_total": int(sum(row["safety"] for row in rows)),
            "approx_kl_by_seed": [float(row["approx_kl"]) for row in rows],
            "surrogate_by_seed": [float(row["surrogate"]) for row in rows],
            "value_loss_by_seed": [float(row["value_loss"]) for row in rows],
            "clip_fraction_by_seed": [float(row["clip_fraction"]) for row in rows],
            "train_success_rate_by_seed": [float(row["train_success_rate"]) for row in rows],
            "approx_kl_mean": _mean([row["approx_kl"] for row in rows]),
            "surrogate_mean": _mean([row["surrogate"] for row in rows]),
            "value_loss_mean": _mean([row["value_loss"] for row in rows]),
            "reachable_failure_terminal_share_median": _G119._median(
                [row["reachable_failure_terminal_share_median"] for row in rows]
            ),
            "unreachable_margin_max_median": _G119._median(
                [row["unreachable_margin_max_median"] for row in rows]
            ),
        }

    baseline_terminal = arm_summary["P0"]["reachable_failure_terminal_share_median"]
    for arm in ARMS:
        terminal = arm_summary[arm]["reachable_failure_terminal_share_median"]
        if terminal is None or baseline_terminal is None:
            arm_summary[arm]["terminal_share_improvement_vs_p0"] = None
        else:
            arm_summary[arm]["terminal_share_improvement_vs_p0"] = float(terminal - baseline_terminal)

    nonfinite = any(
        not bool(row.get("losses_finite", True))
        or not math.isfinite(float(row["approx_kl"]))
        or not math.isfinite(float(row["surrogate"]))
        for row in runs
    )
    safety = any(int(row["safety"]) > 0 for row in runs)
    harm = any(arm_summary[arm]["success_mean"] < SUCCESS_HARM for arm in ARMS)

    signal_arms = [
        arm
        for arm in ARMS
        if arm_summary[arm]["success_mean"] >= SUCCESS_TARGET
        and (
            arm_summary[arm]["approx_kl_mean"] >= KL_SIGNAL
            or arm_summary[arm]["surrogate_mean"] >= SURROGATE_SIGNAL
        )
    ]
    terminal_arms = [
        arm
        for arm in ARMS
        if (arm_summary[arm]["terminal_share_improvement_vs_p0"] or 0.0) >= TERMINAL_IMPROVEMENT
    ]
    kl_moved = [
        arm
        for arm in ARMS
        if arm_summary[arm]["approx_kl_mean"] >= KL_SIGNAL
        or arm_summary[arm]["surrogate_mean"] >= SURROGATE_SIGNAL
    ]

    if nonfinite or safety:
        status = "INVALID"
        reason = "nonfinite_metrics" if nonfinite else "safety_termination_observed"
    elif harm:
        status = "PPO_OPTIMIZATION_HARM"
        reason = "arm_mean_below_18"
    elif signal_arms:
        status = "PPO_OPTIMIZATION_SIGNAL"
        reason = f"arms={signal_arms}"
    elif terminal_arms:
        status = "PPO_OPTIMIZATION_PARTIAL"
        reason = f"terminal_share_improved>=0.1:arms={terminal_arms}"
    elif kl_moved:
        status = "GATE_UNSPECIFIED"
        reason = f"gradient_moved_without_success_or_terminal_gain:arms={kl_moved}"
    else:
        status = "PPO_OPTIMIZATION_LIMITED"
        reason = "all_arms_below_24_and_kl_surrogate_below_1e-3"

    return {"status": status, "reason": reason, "arms": arm_summary}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = {
        "schema_version": "auv_pc_goal120_screen.v1",
        **analyze(args),
    }
    output = args.output or (args.root / "goal120_screen_summary.json")
    markdown = args.markdown or (args.root / "goal120_screen_summary.md")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"]["status"], "arms": payload["gate"]["arms"]}))
    return 0


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal120 PPO optimization-repair screen",
        "",
        f"- gate: `{payload['gate']['status']}`",
        f"- reason: `{payload['gate']['reason']}`",
        "",
        "| arm | success s1 | success s2 | mean | KL | surrogate | value loss | train success | terminal share | Δterminal vs P0 | unreachable margin | timeout | safety |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for arm in ARMS:
        row = payload["gate"]["arms"][arm]
        lines.append(
            f"| {arm} | {row['success_by_seed'][0]}/40 | {row['success_by_seed'][1]}/40 | "
            f"{row['success_mean']:.1f} | {_fmt(row['approx_kl_mean'], 2)} | "
            f"{_fmt(row['surrogate_mean'], 2)} | {_fmt(row['value_loss_mean'], 3)} | "
            f"{_fmt(_mean(row['train_success_rate_by_seed']), 3)} | "
            f"{_fmt(row['reachable_failure_terminal_share_median'], 3)} | "
            f"{_fmt(row['terminal_share_improvement_vs_p0'], 3)} | "
            f"{_fmt(row['unreachable_margin_max_median'], 3)} | "
            f"{row['timeout_by_seed']} | {row['safety_total']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
