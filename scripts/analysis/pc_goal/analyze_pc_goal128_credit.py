#!/usr/bin/env python3
"""Analyze Goal128 SAC reward/terminal traces without new rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.batch.pc import PC_SAFETY_TERMINAL_KINDS, PCTerminalKind

ARMS = ("zero", "m0p125", "m1p0")
SCENARIOS = ("strict", "default")
TRAIN_SEEDS = tuple(range(86201, 86209))
WINDOWS = (32, 64)
EPS = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zero-train-seed", type=int, default=86201)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _trace_path(root: Path, arm: str, scenario: str, train_seed: int) -> Path:
    if arm == "zero":
        return root / "zero" / scenario / "trace.npz"
    return root / f"dose_{arm}" / f"S_s{train_seed}" / scenario / "trace.npz"


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        trace = {name: np.asarray(payload[name]) for name in payload.files}
    required = {
        "seed",
        "step",
        "done",
        "terminal_kind",
        "goal_distance_eta",
        "goal_distance_pnav",
        "reward_total",
        "reward_terminal",
        "reward_terminal_progress",
    }
    missing = sorted(required - set(trace))
    if missing:
        raise ValueError(f"{path}: missing trace fields {missing}")
    return trace


def _trace_meta(path: Path) -> dict[str, Any]:
    meta_path = path.with_name("trace_meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    return float(value)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _bootstrap_ci(values: list[float], rng: np.random.Generator, count: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1 or count <= 0:
        return float(array[0]), float(array[0])
    draws = rng.choice(array, size=(count, array.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _window_stats(trace: dict[str, np.ndarray], rows: np.ndarray, dt_s: float) -> dict[str, float]:
    if rows.size == 0:
        return {}
    eta = trace["goal_distance_eta"][rows]
    pnav = trace["goal_distance_pnav"][rows]
    path = trace["path_error_3d"][rows]
    horizontal = trace["path_horizontal_error"][rows]
    vertical = trace["path_vertical_error"][rows]
    residual = np.linalg.norm(trace["residual"][rows], axis=1)
    action = np.max(np.abs(trace["action"][rows]), axis=1)
    gate = trace["gate"][rows]
    speed = trace.get("ground_velocity_body_mps")
    if speed is None:
        speed = trace.get("nu")
    pitch = trace.get("eta_rpy")
    result = {
        "eta_mean": _safe_float(np.mean(eta)),
        "eta_min": _safe_float(np.min(eta)),
        "eta_start": _safe_float(eta[0]),
        "eta_end": _safe_float(eta[-1]),
        "pnav_mean": _safe_float(np.mean(pnav)),
        "path_3d_mean": _safe_float(np.mean(path)),
        "path_horizontal_mean": _safe_float(np.mean(horizontal)),
        "path_vertical_mean": _safe_float(np.mean(vertical)),
        "residual_norm_mean": _safe_float(np.mean(residual)),
        "action_absmax_mean": _safe_float(np.mean(action)),
        "gate_mean": _safe_float(np.mean(gate)),
    }
    if speed is not None:
        result["speed_surge_mean"] = _safe_float(np.mean(speed[rows, 0]))
    if pitch is not None:
        result["pitch_mean"] = _safe_float(np.mean(pitch[rows, 1]))
    if rows.size > 1 and dt_s > 0.0:
        result["eta_closure_mps"] = _safe_float((eta[0] - eta[-1]) / ((rows.size - 1) * dt_s))
    return result


def _episode_metrics(
    trace: dict[str, np.ndarray],
    *,
    seed: int,
    dt_s: float,
    radius_m: float,
) -> dict[str, Any]:
    rows = np.flatnonzero(trace["seed"] == seed)
    if rows.size == 0:
        raise ValueError(f"trace has no rows for seed {seed}")
    rows = rows[np.argsort(trace["step"][rows], kind="stable")]
    done_rows = rows[trace["done"][rows].astype(bool)]
    terminal_row = int(done_rows[0]) if done_rows.size else int(rows[-1])
    terminal_pos = int(np.flatnonzero(rows == terminal_row)[0])
    prefix = rows[: terminal_pos + 1]
    pre = rows[:terminal_pos]
    term_kind = int(trace["terminal_kind"][terminal_row])
    total = trace["reward_total"][prefix]
    terminal = trace["reward_terminal"][prefix]
    terminal_progress = trace["reward_terminal_progress"][prefix]
    dense = total - terminal - terminal_progress
    steps = int(prefix.size)
    terminal_sum = float(np.sum(terminal))
    terminal_progress_sum = float(np.sum(terminal_progress))
    dense_sum = float(np.sum(dense))
    total_sum = float(np.sum(total))
    n2s = max(1, int(round(2.0 / dt_s))) if dt_s > 0.0 else 1
    terminal_2s = float(np.sum(terminal[-n2s:]))
    total_2s = float(np.sum(total[-n2s:]))
    terminal_share_2s = terminal_2s / max(abs(total_2s), EPS)
    terminal_to_dense = terminal_sum / max(abs(dense_sum), EPS)
    the_kind = PCTerminalKind(term_kind)
    result: dict[str, Any] = {
        "seed": int(seed),
        "steps": steps,
        "terminal_kind": term_kind,
        "terminal_name": the_kind.name,
        "success": the_kind is PCTerminalKind.GOAL,
        "timeout": the_kind is PCTerminalKind.TIME_LIMIT,
        "goal_plane_miss": the_kind is PCTerminalKind.GOAL_PLANE_MISS,
        "safety": the_kind in PC_SAFETY_TERMINAL_KINDS,
        "terminal_eta_m": float(trace["goal_distance_eta"][terminal_row]),
        "terminal_pnav_m": float(trace["goal_distance_pnav"][terminal_row]),
        "terminal_eta_tight_ever": bool(np.any(trace["goal_distance_eta"][prefix] <= radius_m + EPS)),
        "terminal_pnav_tight_ever": bool(np.any(trace["goal_distance_pnav"][prefix] <= radius_m + EPS)),
        "reward_total_sum": total_sum,
        "reward_terminal_sum": terminal_sum,
        "reward_terminal_progress_sum": terminal_progress_sum,
        "dense_nonterminal_sum": dense_sum,
        "dense_per_step": dense_sum / max(float(steps), 1.0),
        "terminal_progress_per_step": terminal_progress_sum / max(float(steps), 1.0),
        "terminal_share_last2s": terminal_share_2s,
        "terminal_to_dense_ratio": terminal_to_dense,
        "goal_plane_miss_sum": float(
            np.sum(trace.get("reward_goal_plane_miss", np.zeros_like(total))[prefix])
        ),
        "safety_sum": float(np.sum(trace.get("reward_safety", np.zeros_like(total))[prefix])),
    }
    entry_rows = np.flatnonzero(trace["goal_distance_eta"][prefix] <= 2.0)
    if entry_rows.size:
        entry_abs = int(entry_rows[0])
        entry_row = int(prefix[entry_abs])
        result["first_2m_step"] = int(trace["step"][entry_row])
        result["first_2m_eta_m"] = float(trace["goal_distance_eta"][entry_row])
        result["first_2m_vertical_m"] = float(trace["path_vertical_error"][entry_row])
        result["first_2m_horizontal_m"] = float(trace["path_horizontal_error"][entry_row])
    else:
        result["first_2m_step"] = None
        result["first_2m_eta_m"] = float("nan")
        result["first_2m_vertical_m"] = float("nan")
        result["first_2m_horizontal_m"] = float("nan")
    for window in WINDOWS:
        start = max(0, terminal_pos - window)
        window_rows = pre[start:]
        for key, value in _window_stats(trace, window_rows, dt_s).items():
            result[f"pre{window}_{key}"] = value
    return result


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        key
        for key, value in episodes[0].items()
        if _is_number(value) and key != "seed"
    ]
    aggregate: dict[str, Any] = {
        "episodes": len(episodes),
        "success": int(sum(bool(item["success"]) for item in episodes)),
        "timeout": int(sum(bool(item["timeout"]) for item in episodes)),
        "goal_plane_miss": int(sum(bool(item["goal_plane_miss"]) for item in episodes)),
        "safety": int(sum(bool(item["safety"]) for item in episodes)),
        "terminal_names": {
            name: int(sum(item["terminal_name"] == name for item in episodes))
            for name in sorted({item["terminal_name"] for item in episodes})
        },
    }
    for key in numeric_keys:
        values = [
            float(item[key])
            for item in episodes
            if _is_number(item.get(key)) and np.isfinite(float(item[key]))
        ]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        aggregate[f"{key}_mean"] = float(array.mean())
        aggregate[f"{key}_median"] = float(np.median(array))
        aggregate[f"{key}_min"] = float(array.min())
        aggregate[f"{key}_max"] = float(array.max())
    return aggregate


def _paired(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    keys: list[str],
    *,
    rng: np.random.Generator,
    bootstrap: int,
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    result: dict[str, Any] = {"count": len(common), "seeds": common}
    for key in keys:
        values = [float(left[seed][key] - right[seed][key]) for seed in common]
        finite = [value for value in values if np.isfinite(value)]
        low, high = _bootstrap_ci(finite, rng, bootstrap)
        result[key] = {
            "mean": _mean(finite),
            "median": float(np.median(finite)) if finite else float("nan"),
            "ci95": [low, high],
            "positive": int(sum(value > 0.0 for value in finite)),
            "negative": int(sum(value < 0.0 for value in finite)),
            "values": values,
        }
    return result


def _group_signal(episodes: list[dict[str, Any]], key: str, success: bool) -> float:
    values = [
        float(item[key])
        for item in episodes
        if bool(item["success"]) is success
        and _is_number(item.get(key))
        and np.isfinite(float(item[key]))
    ]
    return _mean(values)


def _signals(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [item for item in episodes if not bool(item["success"])]
    successes = [item for item in episodes if bool(item["success"])]
    return {
        "failure_count": len(failures),
        "success_count": len(successes),
        "dense_per_step_failure": _group_signal(episodes, "dense_per_step", False),
        "dense_per_step_success": _group_signal(episodes, "dense_per_step", True),
        "terminal_share_failure": _group_signal(episodes, "terminal_share_last2s", False),
        "terminal_share_success": _group_signal(episodes, "terminal_share_last2s", True),
        "terminal_to_dense_failure": _group_signal(episodes, "terminal_to_dense_ratio", False),
        "terminal_to_dense_success": _group_signal(episodes, "terminal_to_dense_ratio", True),
        "pre64_eta_failure": _group_signal(episodes, "pre64_eta_mean", False),
        "pre64_eta_success": _group_signal(episodes, "pre64_eta_mean", True),
        "pre64_vertical_failure": _group_signal(episodes, "pre64_path_vertical_mean", False),
        "pre64_vertical_success": _group_signal(episodes, "pre64_path_vertical_mean", True),
        "pre64_residual_failure": _group_signal(episodes, "pre64_residual_norm_mean", False),
        "pre64_residual_success": _group_signal(episodes, "pre64_residual_norm_mean", True),
    }


def _provisional(signals: dict[str, Any]) -> str:
    if signals["failure_count"] == 0 or signals["success_count"] == 0:
        return "INSUFFICIENT_CLASSES"
    dense_gap = signals["dense_per_step_failure"] - signals["dense_per_step_success"]
    share_gap = signals["terminal_share_failure"] - signals["terminal_share_success"]
    geometry_gap = signals["pre64_eta_failure"] - signals["pre64_eta_success"]
    if np.isfinite(dense_gap) and np.isfinite(share_gap) and dense_gap >= 0.0 and share_gap <= 0.0:
        return "DENSE_CREDIT_DOMINANT"
    if np.isfinite(share_gap) and share_gap <= 0.0:
        return "TERMINAL_CREDIT_MISSING"
    if np.isfinite(geometry_gap) and geometry_gap >= 0.25:
        return "GEOMETRY_LIMITED"
    return "NO_STABLE_REWARD_TERMINAL_DEFECT"


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float) and np.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_delta(entry: dict[str, Any], digits: int = 3) -> str:
    mean = float(entry["mean"])
    if not np.isfinite(mean):
        return "n/a"
    low, high = entry["ci95"]
    return (
        f"{mean:+.{digits}f} [{low:+.{digits}f},{high:+.{digits}f}]"
        f" (+{entry['positive']}/-{entry['negative']})"
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal128 Reward / Terminal Credit Audit",
        "",
        "**状态**：`EVAL_ONLY / HELDOUT_CLOSED`",
        "",
        "## Aggregate",
        "",
        "| arm | scenario | success | timeout | plane-miss | dense/step | terminal share 2s | terminal/dense | pre64 eta | pre64 vertical |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            item = payload["by_arm_scenario"][arm][scenario]
            lines.append(
                "| {arm} | {scenario} | {success}/{episodes} | {timeout} | {miss} | {dense} | {share} | {ratio} | {eta} | {vertical} |".format(
                    arm=arm,
                    scenario=scenario,
                    success=item["success"],
                    episodes=item["episodes"],
                    timeout=item["timeout"],
                    miss=item["goal_plane_miss"],
                    dense=_fmt(item.get("dense_per_step_mean")),
                    share=_fmt(item.get("terminal_share_last2s_mean")),
                    ratio=_fmt(item.get("terminal_to_dense_ratio_mean")),
                    eta=_fmt(item.get("pre64_eta_mean_mean")),
                    vertical=_fmt(item.get("pre64_path_vertical_mean_mean")),
                )
            )
    lines.extend(
        [
            "",
            "## Hypothesis Signals",
            "",
            "| arm | failure dense/step | success dense/step | failure term share | success term share | failure pre64 eta | success pre64 eta | provisional |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for arm in ("m0p125", "m1p0"):
        signal = payload["signals"][arm]
        lines.append(
            "| {arm} | {fd} | {sd} | {ft} | {st} | {fe} | {se} | {verdict} |".format(
                arm=arm,
                fd=_fmt(signal["dense_per_step_failure"]),
                sd=_fmt(signal["dense_per_step_success"]),
                ft=_fmt(signal["terminal_share_failure"]),
                st=_fmt(signal["terminal_share_success"]),
                fe=_fmt(signal["pre64_eta_failure"]),
                se=_fmt(signal["pre64_eta_success"]),
                verdict=_provisional(signal),
            )
        )
    lines.extend(
        [
            "",
            "## Paired vs zero（同 seed 配对，差值 = arm - zero）",
            "",
            "| arm | scenario | Δsuccess | Δdense/step | Δterminal share 2s | Δterminal/dense | Δpre64 eta | Δpre64 residual |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    paired_keys = (
        "success",
        "dense_per_step",
        "terminal_share_last2s",
        "terminal_to_dense_ratio",
        "pre64_eta_mean",
        "pre64_residual_norm_mean",
    )
    for arm in ("m0p125", "m1p0"):
        for scenario in SCENARIOS:
            aggregate: dict[str, dict[str, Any]] = {}
            for key in paired_keys:
                means = [
                    float(payload["paired_vs_zero"][arm][scenario][seed][key]["mean"])
                    for seed in TRAIN_SEEDS
                ]
                finite = [value for value in means if np.isfinite(value)]
                rng = np.random.default_rng(20261111)
                low, high = _bootstrap_ci(finite, rng, 2000)
                aggregate[key] = {
                    "mean": _mean(finite),
                    "ci95": [low, high],
                    "positive": int(sum(value > 0.0 for value in finite)),
                    "negative": int(sum(value < 0.0 for value in finite)),
                }
            lines.append(
                "| {arm} | {scenario} | {s} | {d} | {t} | {r} | {e} | {v} |".format(
                    arm=arm,
                    scenario=scenario,
                    s=_fmt_delta(aggregate["success"]),
                    d=_fmt_delta(aggregate["dense_per_step"]),
                    t=_fmt_delta(aggregate["terminal_share_last2s"]),
                    r=_fmt_delta(aggregate["terminal_to_dense_ratio"]),
                    e=_fmt_delta(aggregate["pre64_eta_mean"]),
                    v=_fmt_delta(aggregate["pre64_residual_norm_mean"]),
                )
            )
    lines.extend(["", "## Per-seed 86201 vs others", ""])
    for arm in ("m0p125", "m1p0"):
        lines.extend([f"### {arm}", "", "| group | success | dense/step | terminal share | pre64 eta | pre64 vertical |", "|---|---:|---:|---:|---:|---:|"])
        for group in ("86201", "others"):
            item = payload["by_arm_scenario_trainseed"][arm]["strict"][group]
            lines.append(
                "| {group} | {success}/{episodes} | {dense} | {share} | {eta} | {vertical} |".format(
                    group=group,
                    success=item["success"],
                    episodes=item["episodes"],
                    dense=_fmt(item.get("dense_per_step_mean")),
                    share=_fmt(item.get("terminal_share_last2s_mean")),
                    eta=_fmt(item.get("pre64_eta_mean_mean")),
                    vertical=_fmt(item.get("pre64_path_vertical_mean_mean")),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    root = _resolve(args.root)
    output = _resolve(args.output)
    rng = np.random.default_rng(20261111)
    traces: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = {}
    metas: dict[tuple[str, str, int], dict[str, Any]] = {}
    for arm in ARMS:
        train_seeds = (args.zero_train_seed,) if arm == "zero" else TRAIN_SEEDS
        for train_seed in train_seeds:
            for scenario in SCENARIOS:
                path = _trace_path(root, arm, scenario, train_seed)
                trace = _load_trace(path)
                meta = _trace_meta(path)
                if not bool(meta.get("finite", False)):
                    raise ValueError(f"non-finite trace: {path}")
                episodes: dict[int, dict[str, Any]] = {}
                for seed in sorted(set(int(value) for value in trace["seed"].tolist())):
                    episodes[seed] = _episode_metrics(
                        trace,
                        seed=seed,
                        dt_s=float(meta["control_dt_s"]),
                        radius_m=float(meta["eval_goal_radius_m"]),
                    )
                traces[(arm, scenario, train_seed)] = episodes
                metas[(arm, scenario, train_seed)] = meta

    by_arm_scenario: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        by_arm_scenario[arm] = {}
        train_seeds = (args.zero_train_seed,) if arm == "zero" else TRAIN_SEEDS
        for scenario in SCENARIOS:
            episodes = [
                episode
                for train_seed in train_seeds
                for episode in traces[(arm, scenario, train_seed)].values()
            ]
            by_arm_scenario[arm][scenario] = _aggregate(episodes)

    by_arm_scenario_trainseed: dict[str, dict[str, Any]] = {}
    for arm in ("m0p125", "m1p0"):
        by_arm_scenario_trainseed[arm] = {}
        for scenario in SCENARIOS:
            by_arm_scenario_trainseed[arm][scenario] = {}
            for train_seed in TRAIN_SEEDS:
                group = "86201" if train_seed == args.zero_train_seed else "others"
                episodes = list(traces[(arm, scenario, train_seed)].values())
                current = by_arm_scenario_trainseed[arm][scenario].setdefault(group, [])
                current.extend(episodes)
    for arm in ("m0p125", "m1p0"):
        for scenario in SCENARIOS:
            for group in ("86201", "others"):
                by_arm_scenario_trainseed[arm][scenario][group] = _aggregate(
                    by_arm_scenario_trainseed[arm][scenario][group]
                )

    signals: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    compare_keys = [
        "success",
        "terminal_eta_m",
        "dense_per_step",
        "terminal_share_last2s",
        "terminal_to_dense_ratio",
        "pre32_eta_mean",
        "pre64_eta_mean",
        "pre64_path_vertical_mean",
        "pre64_path_horizontal_mean",
        "pre64_residual_norm_mean",
        "pre64_action_absmax_mean",
    ]
    for arm in ("m0p125", "m1p0"):
        signals[arm] = _signals(
            [
                episode
                for scenario in SCENARIOS
                for train_seed in TRAIN_SEEDS
                for episode in traces[(arm, scenario, train_seed)].values()
            ]
        )
        paired[arm] = {}
        for scenario in SCENARIOS:
            paired[arm][scenario] = {}
            for train_seed in TRAIN_SEEDS:
                paired[arm][scenario][train_seed] = _paired(
                    traces[(arm, scenario, train_seed)],
                    traces[("zero", scenario, args.zero_train_seed)],
                    compare_keys,
                    rng=rng,
                    bootstrap=args.bootstrap,
                )

    payload = {
        "schema": "auv_goal128_reward_terminal_credit.v1",
        "root": str(root),
        "arms": list(ARMS),
        "scenarios": list(SCENARIOS),
        "train_seeds": list(TRAIN_SEEDS),
        "by_arm_scenario": by_arm_scenario,
        "by_arm_scenario_trainseed": by_arm_scenario_trainseed,
        "signals": signals,
        "paired_vs_zero": paired,
        "provisional_verdict": {
            arm: _provisional(signals[arm]) for arm in ("m0p125", "m1p0")
        },
        "trace_meta": {
            f"{arm}/{scenario}/{train_seed}": metas[(arm, scenario, train_seed)]
            for arm in ARMS
            for scenario in SCENARIOS
            for train_seed in ((args.zero_train_seed,) if arm == "zero" else TRAIN_SEEDS)
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "verdict": payload["provisional_verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
