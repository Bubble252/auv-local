#!/usr/bin/env python3
"""Analyze Goal130 channel-aware cap and direction-gate traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pc_goal128_credit import (
    _aggregate,
    _bootstrap_ci,
    _episode_metrics,
    _load_trace,
    _trace_meta,
)
from analyze_pc_goal129_gate import _aligned_trace_pair, _terminal_kind_counts

CONTROL_ARMS = ("zero", "m1p0")
NEW_ARMS = ("R0p10", "R0p25", "R0p50", "F0p25", "P0p25", "D0p0", "D0p5", "R0p25D0p0")
ARMS = (*CONTROL_ARMS, *NEW_ARMS)
SCENARIOS = ("strict", "default")
TRAIN_SEEDS = tuple(range(86201, 86209))
DEV_SEEDS = tuple(range(20, 60))
ZERO_SEED = 86201
ZERO_STRICT = 24
ZERO_DEFAULT = 33
CHANNEL_NAMES = ("rudder", "stern_plane", "propeller_rpm")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal128-root", type=Path, required=True)
    parser.add_argument("--goal129-root", type=Path, required=True)
    parser.add_argument("--goal130-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _trace_path(
    goal128_root: Path,
    goal130_root: Path,
    arm: str,
    scenario: str,
    seed: int,
) -> Path:
    if arm == "zero":
        return goal128_root / "zero" / scenario / "trace.npz"
    if arm == "m1p0":
        return goal128_root / "dose_m1p0" / f"S_s{seed}" / scenario / "trace.npz"
    return goal130_root / arm / f"S_s{seed}" / scenario / "trace.npz"


def _gate_from_eta(eta: np.ndarray, enter_m: float = 10.0, full_m: float = 3.0) -> np.ndarray:
    progress = np.clip((enter_m - eta) / (enter_m - full_m), 0.0, 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def _load(goal128_root: Path, goal130_root: Path):
    traces: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = {}
    metas: dict[tuple[str, str, int], dict[str, Any]] = {}
    for arm in ARMS:
        seeds = (ZERO_SEED,) if arm == "zero" else TRAIN_SEEDS
        for seed in seeds:
            for scenario in SCENARIOS:
                path = _trace_path(goal128_root, goal130_root, arm, scenario, seed)
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


def _success_by_train_seed(traces, arm: str, scenario: str) -> list[int]:
    if arm == "zero":
        episodes = traces[("zero", scenario, ZERO_SEED)]
        count = int(sum(bool(item["success"]) for item in episodes.values()))
        return [count] * len(TRAIN_SEEDS)
    return [
        int(sum(bool(item["success"]) for item in traces[(arm, scenario, seed)].values()))
        for seed in TRAIN_SEEDS
    ]


def _gate(strict: list[int], default: list[int]) -> dict[str, Any]:
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
    low, high = _bootstrap_ci([float(value) for value in strict_delta], rng, 2000)
    return {
        "status": status,
        "strict_mean": strict_mean,
        "default_mean": default_mean,
        "strict_deltas": strict_delta,
        "default_deltas": default_delta,
        "strict_improved_seeds": int(sum(value > 0 for value in strict_delta)),
        "strict_worst_delta": int(min(strict_delta)),
        "strict_delta_ci95": [low, high],
    }


def _learner_stats(goal128_root: Path, goal130_root: Path, arm: str) -> dict[str, Any]:
    if arm == "zero":
        return {"channel_ratio": [float("nan")] * 3, "direction_cos_mean": float("nan"), "direction_block_fraction": float("nan")}
    ratios: list[np.ndarray] = []
    cos_values: list[np.ndarray] = []
    blocked: list[np.ndarray] = []
    for seed in TRAIN_SEEDS:
        trace = _load_trace(
            _trace_path(goal128_root, goal130_root, arm, "strict", seed)
        )
        gate = _gate_from_eta(trace["goal_distance_eta"])
        near = gate > 0.5
        if not bool(near.any()):
            continue
        if "learner_gated_action" in trace:
            learner = trace["learner_gated_action"][near]
        else:
            learner = trace["action"][near] * gate[near, None]
        base = trace["bc_residual_action"][near]
        base_abs_mean = np.mean(np.abs(base), axis=0)
        learner_abs_mean = np.mean(np.abs(learner), axis=0)
        ratios.append(learner_abs_mean / np.maximum(base_abs_mean, 1e-12))
        if "learner_direction_cos" in trace:
            cos = np.asarray(trace["learner_direction_cos"])[near, 0]
        else:
            denom = np.linalg.norm(learner, axis=1) * np.linalg.norm(base, axis=1)
            cos = np.zeros(learner.shape[0], dtype=np.float64)
            valid = denom > 1e-12
            cos[valid] = np.sum(learner[valid] * base[valid], axis=1) / denom[valid]
        cos_values.append(cos)
        blocked.append(cos < 0.0)
    return {
        "channel_ratio": (
            np.mean(np.asarray(ratios), axis=0).tolist() if ratios else [float("nan")] * 3
        ),
        "direction_cos_mean": float(np.mean(np.concatenate(cos_values))) if cos_values else float("nan"),
        "direction_block_fraction": float(np.mean(np.concatenate(blocked))) if blocked else float("nan"),
    }


def _parity(goal129_root: Path, goal130_root: Path) -> dict[str, Any] | None:
    new_dir = goal130_root / "parity_m1p0_full"
    old_dir = goal129_root / "parity_m1p0_full"
    if not (new_dir / "summary.json").is_file() or not (old_dir / "summary.json").is_file():
        return None
    new_summary = json.loads((new_dir / "summary.json").read_text(encoding="utf-8"))["summary"]
    old_summary = json.loads((old_dir / "summary.json").read_text(encoding="utf-8"))["summary"]
    new_trace = _load_trace(new_dir / "trace.npz")
    old_trace = _load_trace(old_dir / "trace.npz")
    counts_match = new_summary.get("episode_count") == old_summary.get("episode_count")
    successes_match = new_summary.get("success_count_eta_tight_at_done") == old_summary.get("success_count_eta_tight_at_done")
    kinds_match = _terminal_kind_counts(new_trace) == _terminal_kind_counts(old_trace)
    pair = _aligned_trace_pair(new_trace, old_trace)
    maxdiff: dict[str, float | None] = {"action": None, "learned_residual_action": None}
    aligned_by = None
    action_match = False
    if pair is not None:
        new_aligned, old_aligned, aligned_by = pair
        for name in maxdiff:
            if name in new_aligned and name in old_aligned:
                maxdiff[name] = float(
                    np.max(
                        np.abs(
                            new_aligned[name].astype(np.float64)
                            - old_aligned[name].astype(np.float64)
                        )
                    )
                )
        action_match = all(value is not None and value <= 1e-12 for value in maxdiff.values())
    return {
        "episode_count_match": counts_match,
        "success_count_match": successes_match,
        "terminal_kind_match": kinds_match,
        "aligned_by": aligned_by,
        "action_maxdiff": maxdiff,
        "action_match": action_match,
        "match": bool(counts_match and successes_match and kinds_match and action_match),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float) and np.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal130 Channel-Aware Cap / Direction Gate",
        "",
        "**状态**：`EVAL_ONLY / HELDOUT_CLOSED`",
        "",
        "## Aggregate",
        "",
        "| arm | scenario | success | timeout | plane-miss | dense/step | pre64 eta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            lines.append(
                "| {arm} | {scenario} | {success}/{episodes} | {timeout} | {miss} | {dense} | {eta} |".format(
                    arm=arm,
                    scenario=scenario,
                    success=item["success"],
                    episodes=item["episodes"],
                    timeout=item["timeout"],
                    miss=item["goal_plane_miss"],
                    dense=_fmt(item.get("dense_per_step_mean")),
                    eta=_fmt(item.get("pre64_eta_mean_mean")),
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
    for arm in NEW_ARMS:
        strict = payload["success_by_train_seed"][arm]["strict"]
        default = payload["success_by_train_seed"][arm]["default"]
        lines.append(
            f"| {arm} | " + " | ".join(f"{s}/{d}" for s, d in zip(strict, default)) + " |"
        )
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
    lines.extend(
        [
            "",
            "## Near-goal learner/BC by channel and direction",
            "",
            "| arm | " + " | ".join(CHANNEL_NAMES) + " | cos(L,B) | cos<0 fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        stats = payload["learner_stats"][arm]
        ratios = stats["channel_ratio"]
        lines.append(
            "| {arm} | {r0} | {r1} | {r2} | {cos} | {blocked} |".format(
                arm=arm,
                r0=_fmt(ratios[0]),
                r1=_fmt(ratios[1]),
                r2=_fmt(ratios[2]),
                cos=_fmt(stats["direction_cos_mean"]),
                blocked=_fmt(stats["direction_block_fraction"]),
            )
        )
    if payload.get("parity") is not None:
        parity = payload["parity"]
        lines.extend(
            [
                "",
                "## Parity",
                "",
                f"`all switches off` 复现 Goal129 M1.0：`{parity['match']}`",
                "",
                f"- counts: `{parity['episode_count_match']}`；success: `{parity['success_count_match']}`；terminal: `{parity['terminal_kind_match']}`",
                f"- aligned by `{parity['aligned_by']}`；action match: `{parity['action_match']}`",
                f"- maxdiff: `{parity['action_maxdiff']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    goal128_root = _resolve(args.goal128_root)
    goal129_root = _resolve(args.goal129_root)
    goal130_root = _resolve(args.goal130_root)
    output = _resolve(args.output)
    traces, metas = _load(goal128_root, goal130_root)

    by_arm_scenario: dict[str, dict[str, Any]] = {}
    success_by_train_seed: dict[str, dict[str, list[int]]] = {}
    gate: dict[str, Any] = {}
    learner_stats: dict[str, Any] = {}
    for arm in ARMS:
        by_arm_scenario[arm] = {}
        success_by_train_seed[arm] = {}
        seeds = (ZERO_SEED,) if arm == "zero" else TRAIN_SEEDS
        for scenario in SCENARIOS:
            episodes = [
                episode
                for seed in seeds
                for episode in traces[(arm, scenario, seed)].values()
            ]
            by_arm_scenario[arm][scenario] = _aggregate(episodes)
            success_by_train_seed[arm][scenario] = _success_by_train_seed(
                traces, arm, scenario
            )
        if arm in NEW_ARMS:
            gate[arm] = _gate(
                success_by_train_seed[arm]["strict"],
                success_by_train_seed[arm]["default"],
            )
        learner_stats[arm] = _learner_stats(goal128_root, goal130_root, arm)

    payload = {
        "schema": "auv_goal130_channel_cap_direction.v1",
        "goal128_root": str(goal128_root),
        "goal129_root": str(goal129_root),
        "goal130_root": str(goal130_root),
        "arms": list(ARMS),
        "new_arms": list(NEW_ARMS),
        "scenarios": list(SCENARIOS),
        "train_seeds": list(TRAIN_SEEDS),
        "by_arm_scenario": by_arm_scenario,
        "success_by_train_seed": success_by_train_seed,
        "gate": gate,
        "learner_stats": learner_stats,
        "parity": _parity(goal129_root, goal130_root),
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
