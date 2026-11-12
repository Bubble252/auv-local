#!/usr/bin/env python3
"""Analyze Goal129 learner-authority gate/cap screen against Goal128 controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.batch.pc import PCTerminalKind

from analyze_pc_goal128_credit import (
    _aggregate,
    _bootstrap_ci,
    _episode_metrics,
    _load_trace,
    _trace_meta,
)

ARMS = ("zero", "m0p125", "m1p0", "G1p0", "C1p0", "GC1p0", "G0p125")
NEW_ARMS = ("G1p0", "C1p0", "GC1p0", "G0p125")
SCENARIOS = ("strict", "default")
TRAIN_SEEDS = tuple(range(86201, 86209))
DEV_SEEDS = tuple(range(20, 60))
ZERO_SEED = 86201
ZERO_STRICT = 24
ZERO_DEFAULT = 33
MULTIPLIERS = {
    "zero": 0.0,
    "m0p125": 0.125,
    "m1p0": 1.0,
    "G1p0": 1.0,
    "C1p0": 1.0,
    "GC1p0": 1.0,
    "G0p125": 0.125,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal128-root", type=Path, required=True)
    parser.add_argument("--goal129-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _trace_path(goal128_root: Path, goal129_root: Path, arm: str, scenario: str, seed: int) -> Path:
    if arm in {"zero", "m0p125", "m1p0"}:
        if arm == "zero":
            return goal128_root / "zero" / scenario / "trace.npz"
        return goal128_root / f"dose_{arm}" / f"S_s{seed}" / scenario / "trace.npz"
    return goal129_root / arm / f"S_s{seed}" / scenario / "trace.npz"


def _gate_from_eta(eta: np.ndarray, enter_m: float = 10.0, full_m: float = 3.0) -> np.ndarray:
    progress = np.clip((enter_m - eta) / (enter_m - full_m), 0.0, 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def _load(
    goal128_root: Path,
    goal129_root: Path,
) -> tuple[dict[tuple[str, str, int], dict[int, dict[str, Any]]], dict[tuple[str, str, int], dict[str, Any]]]:
    traces: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = {}
    metas: dict[tuple[str, str, int], dict[str, Any]] = {}
    for arm in ARMS:
        train_seeds = (ZERO_SEED,) if arm == "zero" else TRAIN_SEEDS
        for seed in train_seeds:
            for scenario in SCENARIOS:
                path = _trace_path(goal128_root, goal129_root, arm, scenario, seed)
                trace = _load_trace(path)
                meta = _trace_meta(path)
                if not bool(meta.get("finite", False)):
                    raise ValueError(f"non-finite trace: {path}")
                episodes = {
                    int(dev_seed): _episode_metrics(
                        trace,
                        seed=int(dev_seed),
                        dt_s=float(meta["control_dt_s"]),
                        radius_m=float(meta["eval_goal_radius_m"]),
                    )
                    for dev_seed in DEV_SEEDS
                }
                traces[(arm, scenario, seed)] = episodes
                metas[(arm, scenario, seed)] = meta
    return traces, metas


def _success_by_train_seed(
    traces: dict[tuple[str, str, int], dict[int, dict[str, Any]]],
    arm: str,
    scenario: str,
) -> list[int]:
    if arm == "zero":
        episodes = traces[("zero", scenario, ZERO_SEED)]
        return [int(sum(bool(item["success"]) for item in episodes.values()))] * len(TRAIN_SEEDS)
    return [
        int(sum(bool(item["success"]) for item in traces[(arm, scenario, seed)].values()))
        for seed in TRAIN_SEEDS
    ]


def _per_seed_pairs(
    traces: dict[tuple[str, str, int], dict[int, dict[str, Any]]],
    arm: str,
    scenario: str,
) -> list[int]:
    if arm == "zero":
        return [0] * len(TRAIN_SEEDS)
    zero = traces[("zero", scenario, ZERO_SEED)]
    return [
        int(
            sum(
                bool(episode["success"]) - bool(zero[dev_seed]["success"])
                for dev_seed, episode in traces[(arm, scenario, seed)].items()
            )
        )
        for seed in TRAIN_SEEDS
    ]


def _learner_stats(
    goal128_root: Path,
    goal129_root: Path,
    arm: str,
) -> dict[str, float]:
    if arm == "zero":
        return {"near_goal_learner_to_bc": float("nan"), "near_goal_saturation": float("nan")}
    ratios: list[float] = []
    saturations: list[float] = []
    multiplier = MULTIPLIERS[arm]
    for seed in TRAIN_SEEDS:
        trace = _load_trace(_trace_path(goal128_root, goal129_root, arm, "strict", seed))
        gate = _gate_from_eta(trace["goal_distance_eta"])
        if "learner_gated_action" in trace:
            learner = trace["learner_gated_action"]
        else:
            learner = trace["action"] * gate[:, None]
        bc = trace["bc_residual_action"]
        near = gate > 0.5
        if not bool(near.any()):
            continue
        learner_norm = np.linalg.norm(learner[near], axis=1)
        bc_norm = np.linalg.norm(bc[near], axis=1)
        valid = bc_norm > 1e-12
        ratios.append(float(np.mean(learner_norm[valid] / bc_norm[valid])))
        if multiplier > 0.0:
            raw = np.abs(trace["action"][near]) / multiplier
            saturations.append(float(np.mean(np.any(raw > 0.95, axis=1))))
    return {
        "near_goal_learner_to_bc": float(np.mean(ratios)) if ratios else float("nan"),
        "near_goal_saturation": float(np.mean(saturations)) if saturations else float("nan"),
    }


def _arm_gate(strict: list[int], default: list[int]) -> dict[str, Any]:
    strict_mean = float(np.mean(strict))
    default_mean = float(np.mean(default))
    strict_delta = [value - ZERO_STRICT for value in strict]
    default_delta = [value - ZERO_DEFAULT for value in default]
    gain = (
        strict_mean >= 25.5
        and default_mean >= 33.0
        and sum(value > 0 for value in strict_delta) >= 4
        and min(strict_delta) >= -2
    )
    safe = strict_mean >= 24.0 and default_mean >= 33.0 and min(strict_delta) >= -1
    status = (
        "ACTION_PARAM_GAIN"
        if gain
        else "ACTION_PARAM_SAFE_PARITY"
        if safe
        else "ACTION_PARAM_NO_EFFECT"
    )
    rng = np.random.default_rng(20261111)
    low, high = _bootstrap_ci([float(v) for v in strict_delta], rng, 2000)
    return {
        "status": status,
        "strict_mean": strict_mean,
        "default_mean": default_mean,
        "strict_deltas": strict_delta,
        "default_deltas": default_delta,
        "strict_improved_seeds": int(sum(value > 0 for value in strict_delta)),
        "strict_worst_delta": int(min(strict_delta)),
        "strict_delta_ci95": [low, high],
        "gain_thresholds": {
            "strict_mean": 25.5,
            "default_mean": 33.0,
            "improved_seeds": 4,
            "worst_delta": -2,
        },
        "safe_thresholds": {
            "strict_mean": 24.0,
            "default_mean": 33.0,
            "worst_delta": -1,
        },
    }


def _episode_done_rows(trace: dict[str, np.ndarray]) -> np.ndarray:
    episode_ids = np.unique(trace["episode_id"])
    rows: list[int] = []
    for episode_id in episode_ids:
        episode_rows = np.flatnonzero(trace["episode_id"] == episode_id)
        episode_rows = episode_rows[np.argsort(trace["step"][episode_rows], kind="stable")]
        done_rows = episode_rows[trace["done"][episode_rows].astype(bool)]
        rows.append(int(done_rows[0] if done_rows.size else episode_rows[-1]))
    return np.asarray(rows, dtype=np.int64)


def _terminal_kind_counts(trace: dict[str, np.ndarray]) -> dict[str, int]:
    rows = _episode_done_rows(trace)
    counts: dict[str, int] = {}
    for kind in trace["terminal_kind"][rows].astype(np.int64):
        name = PCTerminalKind(int(kind)).name
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _aligned_trace_pair(
    new_trace: dict[str, np.ndarray], old_trace: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str] | None:
    for key in ("episode_id", "env_index"):
        if key not in new_trace or key not in old_trace:
            continue
        new_order = np.lexsort((new_trace["step"], new_trace[key]))
        old_order = np.lexsort((old_trace["step"], old_trace[key]))
        if new_order.size != old_order.size:
            return None
        if not np.array_equal(new_trace[key][new_order], old_trace[key][old_order]):
            continue
        if not np.array_equal(new_trace["step"][new_order], old_trace["step"][old_order]):
            continue
        return (
            {name: values[new_order] for name, values in new_trace.items()},
            {name: values[old_order] for name, values in old_trace.items()},
            key,
        )
    return None


def _parity(goal128_root: Path, goal129_root: Path) -> dict[str, Any] | None:
    new_path = goal129_root / "parity_m1p0_full" / "summary.json"
    old_path = goal128_root / "dose_m1p0" / "S_s86201" / "strict" / "summary.json"
    if not new_path.is_file() or not old_path.is_file():
        return None
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new_summary = new["summary"]
    old_summary = old["summary"]
    new_trace = _load_trace(new_path.with_name("trace.npz"))
    old_trace = _load_trace(old_path.with_name("trace.npz"))
    new_counts = _terminal_kind_counts(new_trace)
    old_counts = _terminal_kind_counts(old_trace)
    keys = ("episode_count", "success_count_eta_tight_at_done")
    summary_match = all(new_summary.get(key) == old_summary.get(key) for key in keys)
    terminal_match = new_counts == old_counts

    pair = _aligned_trace_pair(new_trace, old_trace)
    maxdiff: dict[str, float | None] = {
        name: None for name in ("action", "learned_residual_action", "residual", "bc_residual_action")
    }
    aligned_by: str | None = None
    action_match = False
    if pair is not None:
        new_aligned, old_aligned, aligned_by = pair
        for name in maxdiff:
            if name in new_aligned and name in old_aligned:
                maxdiff[name] = float(
                    np.max(np.abs(new_aligned[name].astype(np.float64) - old_aligned[name].astype(np.float64)))
                )
        action_match = all(
            value is not None and value <= 1e-12 for value in maxdiff.values()
        )
    return {
        "new": {key: new_summary.get(key) for key in keys},
        "old": {key: old_summary.get(key) for key in keys},
        "new_terminal_kind_counts": new_counts,
        "old_terminal_kind_counts": old_counts,
        "summary_match": summary_match,
        "terminal_match": terminal_match,
        "action_maxdiff": maxdiff,
        "aligned_by": aligned_by,
        "action_match": action_match,
        "match": summary_match and terminal_match and action_match,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float) and np.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal129 Action-Parameterization Gate Screen",
        "",
        "**状态**：`EVAL_ONLY / HELDOUT_CLOSED`",
        "",
        "## Aggregate",
        "",
        "| arm | scenario | success | timeout | plane-miss | dense/step | pre64 eta | learner/BC | near-goal sat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            learner = payload["learner_stats"][arm]
            lines.append(
                "| {arm} | {scenario} | {success}/{episodes} | {timeout} | {miss} | {dense} | {eta} | {ratio} | {sat} |".format(
                    arm=arm,
                    scenario=scenario,
                    success=item["success"],
                    episodes=item["episodes"],
                    timeout=item["timeout"],
                    miss=item["goal_plane_miss"],
                    dense=_fmt(item.get("dense_per_step_mean")),
                    eta=_fmt(item.get("pre64_eta_mean_mean")),
                    ratio=_fmt(learner["near_goal_learner_to_bc"]),
                    sat=_fmt(learner["near_goal_saturation"]),
                )
            )
    lines.extend(
        [
            "",
            "## Per-seed success（strict / default）",
            "",
            "| arm | " + " | ".join(str(seed) for seed in TRAIN_SEEDS) + " |",
            "|---|" + "---:|" * len(TRAIN_SEEDS),
        ]
    )
    for arm in ("m0p125", "m1p0", *NEW_ARMS):
        strict = payload["success_by_train_seed"][arm]["strict"]
        default = payload["success_by_train_seed"][arm]["default"]
        cells = [f"{s}/{d}" for s, d in zip(strict, default)]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| arm | strict mean | default mean | strict Δ seeds | worst Δ | Δ CI95 | status |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for arm in NEW_ARMS:
        gate = payload["gate"][arm]
        lines.append(
            "| {arm} | {strict:.3f} | {default:.3f} | {improved}/8 | {worst:+d} | [{low:+.3f},{high:+.3f}] | {status} |".format(
                arm=arm,
                strict=gate["strict_mean"],
                default=gate["default_mean"],
                improved=gate["strict_improved_seeds"],
                worst=gate["strict_worst_delta"],
                low=gate["strict_delta_ci95"][0],
                high=gate["strict_delta_ci95"][1],
                status=gate["status"],
            )
        )
    if payload.get("parity") is not None:
        parity = payload["parity"]
        lines.extend(
            [
                "",
                "## Parity",
                "",
                f"`none + cap=0` 复现 M1.0：`{parity['match']}`",
                "",
                f"- summary counts: `{parity['summary_match']}`；terminal kinds: `{parity['terminal_match']}`",
                f"- aligned by `{parity['aligned_by']}`；action match: `{parity['action_match']}`",
                f"- maxdiff: `{parity['action_maxdiff']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    goal128_root = _resolve(args.goal128_root)
    goal129_root = _resolve(args.goal129_root)
    output = _resolve(args.output)
    traces, metas = _load(goal128_root, goal129_root)

    by_arm_scenario: dict[str, dict[str, Any]] = {}
    success_by_train_seed: dict[str, dict[str, list[int]]] = {}
    gate: dict[str, Any] = {}
    learner_stats: dict[str, Any] = {}
    for arm in ARMS:
        by_arm_scenario[arm] = {}
        success_by_train_seed[arm] = {}
        train_seeds = (ZERO_SEED,) if arm == "zero" else TRAIN_SEEDS
        for scenario in SCENARIOS:
            episodes = [
                episode
                for seed in train_seeds
                for episode in traces[(arm, scenario, seed)].values()
            ]
            by_arm_scenario[arm][scenario] = _aggregate(episodes)
            success_by_train_seed[arm][scenario] = _success_by_train_seed(
                traces, arm, scenario
            )
        if arm in NEW_ARMS or arm in {"m0p125", "m1p0"}:
            gate[arm] = _arm_gate(
                success_by_train_seed[arm]["strict"],
                success_by_train_seed[arm]["default"],
            )
        learner_stats[arm] = _learner_stats(goal128_root, goal129_root, arm)

    payload = {
        "schema": "auv_goal129_action_parameterization_gate.v1",
        "goal128_root": str(goal128_root),
        "goal129_root": str(goal129_root),
        "arms": list(ARMS),
        "new_arms": list(NEW_ARMS),
        "scenarios": list(SCENARIOS),
        "train_seeds": list(TRAIN_SEEDS),
        "by_arm_scenario": by_arm_scenario,
        "success_by_train_seed": success_by_train_seed,
        "paired_success_vs_zero": {
            arm: {
                scenario: _per_seed_pairs(traces, arm, scenario)
                for scenario in SCENARIOS
            }
            for arm in ("m0p125", "m1p0", *NEW_ARMS)
        },
        "gate": gate,
        "learner_stats": learner_stats,
        "parity": _parity(goal128_root, goal129_root),
        "trace_meta": {
            f"{arm}/{scenario}/{seed}": metas[(arm, scenario, seed)]
            for arm in ARMS
            for scenario in SCENARIOS
            for seed in ((ZERO_SEED,) if arm == "zero" else TRAIN_SEEDS)
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "gate": {arm: gate[arm]["status"] for arm in NEW_ARMS},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
