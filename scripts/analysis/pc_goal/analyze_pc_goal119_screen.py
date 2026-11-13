#!/usr/bin/env python3
"""Aggregate Goal119 2x2 PPO screen results and apply the frozen Gate."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("artifacts/lowlevel/goal119_ppo_credit_approach_20261111")
ARMS = ("C0", "C1", "C2", "C3")
TRAIN_SEEDS = (86201, 86202)
DEV_SEEDS = tuple(range(20, 60))
UNREACHABLE_SEEDS = (26, 29, 33, 36, 49)
REACHABLE_FAILURE_SEEDS = (25, 27, 31, 50, 51)
SHELL_M = 2.0
VERTICAL_P95_MPS = 1.746415301801522
PITCH_P95_RAD_S = 0.7597548116623882
LAST_WINDOW_S = 2.0
TERMINAL_REWARD_FIELDS = (
    "reward_terminal",
    "reward_terminal_progress",
    "reward_goal_plane_miss",
)
REWARD_EXCLUDE = {"reward_total", "reward_ppo_scaled"}


class ScreenError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--eval-subdir", default="eval")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScreenError(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ScreenError(f"missing trace: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _reward_summary(
    trace: dict[str, np.ndarray], rows: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    reward_fields = sorted(
        name
        for name in trace
        if name.startswith("reward_") and name not in REWARD_EXCLUDE
    )
    if not reward_fields:
        raise ScreenError("trace has no reward_* fields")
    sums: dict[str, float] = {}
    for name in reward_fields:
        values = np.asarray(trace[name], dtype=np.float64)[rows][mask]
        sums[name] = float(np.sum(values[np.isfinite(values)]))
    terminal_terms = [name for name in TERMINAL_REWARD_FIELDS if name in sums]
    if not terminal_terms:
        raise ScreenError("no terminal reward fields found")
    terminal = float(sum(sums[name] for name in terminal_terms))
    dense = float(
        sum(abs(sums[name]) for name in reward_fields if name not in terminal_terms)
    )
    absolute_total = float(sum(abs(sums[name]) for name in reward_fields))
    return {
        "terminal_sum": terminal,
        "dense_abs_sum": dense,
        "terminal_share": terminal / absolute_total if absolute_total > 0.0 else math.nan,
        "dense_terminal_ratio": dense / (abs(terminal) + 1e-9),
    }


def _episode_metrics(
    trace: dict[str, np.ndarray], seed: int, episode: dict[str, Any]
) -> dict[str, Any]:
    rows = np.flatnonzero(np.asarray(trace["episode_seed"], dtype=np.int64) == int(seed))
    if rows.size == 0:
        raise ScreenError(f"trace has no rows for seed {seed}")
    time_s = np.asarray(trace["time_s"], dtype=np.float64)[rows]
    distance = np.asarray(trace["goal_distance_eta_m"], dtype=np.float64)[rows]
    eta = np.asarray(trace["eta"], dtype=np.float64)[rows]
    goal = np.asarray(trace["goal_ned_m"], dtype=np.float64)[rows]
    pitch_error = np.asarray(trace["pitch_error_rad"], dtype=np.float64)[rows]
    entry_hits = np.flatnonzero(distance <= SHELL_M)
    entry = int(entry_hits[0]) if entry_hits.size else None
    closest = int(np.argmin(distance))
    last2 = time_s >= float(time_s[-1]) - LAST_WINDOW_S
    rewards = _reward_summary(trace, rows, last2)
    result: dict[str, Any] = {
        "seed": int(seed),
        "success": bool(episode.get("success")),
        "timeout": bool(episode.get("timeout")),
        "safety": bool(episode.get("safety", False)),
        "closest_distance_m": float(distance[closest]),
        "terminal_share_last2s": rewards["terminal_share"],
        "dense_terminal_ratio_last2s": rewards["dense_terminal_ratio"],
        "entry2": entry is not None,
    }
    if entry is None:
        result.update(
            {
                "vertical_m": None,
                "pitch_rad": None,
                "margin_max": None,
            }
        )
        return result
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else math.nan
    horizon = float(time_s[closest] - time_s[entry])
    if not math.isfinite(horizon) or horizon <= 0.0:
        horizon = dt
    vertical = float(abs(eta[entry, 2] - goal[entry, 2]))
    pitch = float(abs(pitch_error[entry]))
    margin_v = (vertical / max(horizon, 1e-9)) / VERTICAL_P95_MPS
    margin_p = (pitch / max(horizon, 1e-9)) / PITCH_P95_RAD_S
    result.update(
        {
            "vertical_m": vertical,
            "pitch_rad": pitch,
            "T_to_closest_s": horizon,
            "margin_vertical": margin_v,
            "margin_pitch": margin_p,
            "margin_max": max(margin_v, margin_p),
        }
    )
    return result


def _median(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def analyze_run(
    root: Path, arm: str, train_seed: int, eval_subdir: str = "eval"
) -> dict[str, Any]:
    job = f"{arm}_s{train_seed}"
    run_dir = root / eval_subdir / job
    summary = _read_json(run_dir / "summary.json")
    trace = _load_trace(run_dir / "trace.npz")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(DEV_SEEDS):
        raise ScreenError(f"{job}: expected {len(DEV_SEEDS)} episodes")
    by_seed = {int(row["seed"]): row for row in episodes}
    if tuple(sorted(by_seed)) != DEV_SEEDS:
        raise ScreenError(f"{job}: dev seeds are not 20-59")
    metrics = [_episode_metrics(trace, seed, by_seed[seed]) for seed in DEV_SEEDS]
    by_metric = {row["seed"]: row for row in metrics}
    success_metrics = [row for row in metrics if row["success"]]
    unreachable = [by_metric[seed] for seed in UNREACHABLE_SEEDS]
    reachable = [by_metric[seed] for seed in REACHABLE_FAILURE_SEEDS]
    return {
        "arm": arm,
        "train_seed": int(train_seed),
        "job": job,
        "success": sum(row["success"] for row in metrics),
        "timeout": sum(row["timeout"] for row in metrics),
        "safety": sum(row["safety"] for row in metrics),
        "success_terminal_share_median": _median(
            [row["terminal_share_last2s"] for row in success_metrics]
        ),
        "success_dense_terminal_median": _median(
            [row["dense_terminal_ratio_last2s"] for row in success_metrics]
        ),
        "reachable_failure_terminal_share_median": _median(
            [row["terminal_share_last2s"] for row in reachable]
        ),
        "unreachable_margin_max_median": _median(
            [row["margin_max"] for row in unreachable]
        ),
        "unreachable_success": sum(by_metric[seed]["success"] for seed in UNREACHABLE_SEEDS),
        "reachable_success": sum(by_metric[seed]["success"] for seed in REACHABLE_FAILURE_SEEDS),
        "episodes": metrics,
    }


def decide(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for run in runs:
        by_arm[run["arm"]].append(run)
    for arm in ARMS:
        by_arm[arm].sort(key=lambda row: row["train_seed"])
    arm_summary: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        successes = [row["success"] for row in rows]
        arm_summary[arm] = {
            "success_by_seed": successes,
            "success_mean": float(np.mean(successes)),
            "timeout_by_seed": [row["timeout"] for row in rows],
            "safety_total": sum(row["safety"] for row in rows),
            "reachable_failure_terminal_share_median": _median(
                [row["reachable_failure_terminal_share_median"] for row in rows]
            ),
            "unreachable_margin_max_median": _median(
                [row["unreachable_margin_max_median"] for row in rows]
            ),
        }
    if any(row["safety"] > 0 for row in runs):
        status = "INVALID"
        reason = "safety_termination_observed"
    else:
        strong = [
            arm
            for arm in ARMS
            if arm_summary[arm]["success_mean"] >= 28.0
            and min(arm_summary[arm]["success_by_seed"]) >= 24
            and max(arm_summary[arm]["timeout_by_seed"]) <= 1
        ]
        directional = [
            arm
            for arm in ARMS
            if arm_summary[arm]["success_mean"] >= 24.0
            and min(arm_summary[arm]["success_by_seed"]) >= 21
        ]
        if strong:
            status = "STRONG_SCREEN"
            reason = f"arms={strong}"
        elif directional:
            status = "DIRECTIONAL_SCREEN"
            reason = f"arms={directional}"
        elif arm_summary["C0"]["success_mean"] >= 24.0:
            status = "TRAINING_DOSE_PRIMARY"
            reason = "only_C0_reached_24"
        else:
            status = "NO_SCREEN_MOVEMENT"
            reason = "no_arm_reached_24"
    return {"status": status, "reason": reason, "arms": arm_summary}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal119 PPO 2x2 screen",
        "",
        f"- gate: `{payload['gate']['status']}`",
        f"- reason: `{payload['gate']['reason']}`",
        "",
        "| arm | success s1 | success s2 | mean | timeout | safety | reachable fail terminal | unreachable margin |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["gate"]["arms"][arm]
        lines.append(
            f"| {arm} | {row['success_by_seed'][0]}/40 | {row['success_by_seed'][1]}/40 | "
            f"{row['success_mean']:.1f} | {row['timeout_by_seed']} | {row['safety_total']} | "
            f"{row['reachable_failure_terminal_share_median']:.3f} | "
            f"{row['unreachable_margin_max_median']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = [
        analyze_run(args.root, arm, seed, args.eval_subdir)
        for arm in ARMS
        for seed in TRAIN_SEEDS
    ]
    payload = {
        "schema_version": "auv_pc_goal119_screen.v1",
        "runs": runs,
        "gate": decide(runs),
    }
    output = args.output or (args.root / "goal119_screen_summary.json")
    markdown = args.markdown or (args.root / "goal119_screen_summary.md")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"]["status"], "arms": payload["gate"]["arms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
