#!/usr/bin/env python3
"""Read-only Goal145 audit of terminal capture, path error and residual authority."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal145_terminal_capture_vertical_audit_20261111"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="default: <root>/audit_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="default: <root>/audit_summary.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=145)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _episodes(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "episodes.json"
    if path.is_file():
        payload = _read_json(path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
            return payload["episodes"]
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        payload = _read_json(summary_path)
        episodes = payload.get("episodes")
        if isinstance(episodes, list):
            return episodes
    return []


def _success(episode: dict[str, Any]) -> bool:
    value = episode.get("success_eta_tight_at_done")
    if value is None:
        value = episode.get("success")
    return bool(value)


def _finite_mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(array.mean())


def _finite_median(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.median(array))


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or samples <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _summarize_delta(
    name: str,
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    low, high = _bootstrap_ci(array, samples=samples, seed=seed)
    return {
        "name": name,
        "n": int(array.size),
        "mean": None if array.size == 0 else float(array.mean()),
        "median": None if array.size == 0 else float(np.median(array)),
        "ci95_low": low,
        "ci95_high": high,
        "positive": int(np.sum(array > 1e-12)),
        "negative": int(np.sum(array < -1e-12)),
        "zero": int(np.sum(np.abs(array) <= 1e-12)),
    }


def _trace_episode_metrics(
    trace_path: Path,
    *,
    interface: str,
) -> dict[int, dict[str, float]]:
    if not trace_path.is_file():
        return {}
    trace = np.load(trace_path, allow_pickle=False)
    seed_key = "seed" if "seed" in trace.files else "episode_seed"
    if seed_key not in trace.files:
        return {}
    seeds = np.unique(trace[seed_key])
    result: dict[int, dict[str, float]] = {}
    for seed in seeds:
        index = np.flatnonzero(trace[seed_key] == seed)
        if index.size == 0:
            continue
        metrics: dict[str, float] = {}
        if "path_vertical_error" in trace.files:
            vertical = np.asarray(trace["path_vertical_error"][index], dtype=np.float64)
            horizontal = np.asarray(
                trace["path_horizontal_error"][index], dtype=np.float64
            )
            metrics["path_vertical_rmse_m"] = float(np.sqrt(np.mean(vertical**2)))
            metrics["path_horizontal_rmse_m"] = float(np.sqrt(np.mean(horizontal**2)))
            metrics["path_three_d_rmse_m"] = float(
                np.sqrt(np.mean(vertical**2 + horizontal**2))
            )
            metrics["path_vertical_mse_m2"] = float(np.mean(vertical**2))
            metrics["path_horizontal_mse_m2"] = float(np.mean(horizontal**2))
        eta_key = None
        for candidate in ("eta", "eta_ned"):
            if candidate in trace.files:
                eta_key = candidate
                break
        if eta_key is not None:
            eta = np.asarray(trace[eta_key][index, :3], dtype=np.float64)
            key = "goal_ned_m" if "goal_ned_m" in trace.files else "goal_ned"
            goal = np.asarray(trace[key][index, :3], dtype=np.float64)
            miss = eta - goal
            metrics["final_miss_n_m"] = float(miss[-1, 0])
            metrics["final_miss_e_m"] = float(miss[-1, 1])
            metrics["final_miss_d_m"] = float(miss[-1, 2])
            metrics["final_vertical_abs_m"] = float(abs(miss[-1, 2]))
            metrics["final_horizontal_m"] = float(
                math.hypot(miss[-1, 0], miss[-1, 1])
            )
            distance_key = None
            for candidate in ("goal_distance_eta_m", "goal_distance_eta"):
                if candidate in trace.files:
                    distance_key = candidate
                    break
            if distance_key in trace.files:
                distance = np.asarray(trace[distance_key][index], dtype=np.float64)
                metrics["final_goal_distance_m"] = float(distance[-1])
                near = distance <= 3.0
                if bool(near.any()):
                    metrics["near_goal_steps"] = float(near.sum())
                    metrics["near_goal_vertical_abs_mean_m"] = float(
                        np.mean(np.abs(miss[near, 2]))
                    )
                    metrics["near_goal_horizontal_mean_m"] = float(
                        np.mean(np.hypot(miss[near, 0], miss[near, 1]))
                    )
        if "guidance_terminal_phase" in trace.files:
            metrics["terminal_phase_rate"] = float(
                np.mean(
                    np.asarray(
                        trace["guidance_terminal_phase"][index], dtype=np.float64
                    )
                    >= 0.5
                )
            )
        residual_key = None
        for candidate in (
            "learned_residual_action",
            "residual_action",
            "residual",
        ):
            if candidate in trace.files:
                residual_key = candidate
                break
        if residual_key is not None:
            residual = np.asarray(trace[residual_key][index], dtype=np.float64)
            metrics["residual_channel_rms"] = [
                float(value) for value in np.sqrt(np.mean(residual**2, axis=0))
            ]
            metrics["residual_abs_mean"] = float(np.mean(np.abs(residual)))
        if "actual_actuator" in trace.files:
            actuator = np.asarray(trace["actual_actuator"][index], dtype=np.float64)
            metrics["actuator_saturation_fraction"] = float(
                np.mean(np.abs(actuator) >= 0.98)
            )
            metrics["actuator_channel_rms"] = [
                float(value) for value in np.sqrt(np.mean(actuator**2, axis=0))
            ]
        if "command_residual_applied" in trace.files:
            command_residual = np.asarray(
                trace["command_residual_applied"][index], dtype=np.float64
            )
            metrics["command_residual_channel_abs_mean"] = [
                float(value) for value in np.mean(np.abs(command_residual), axis=0)
            ]
            metrics["command_residual_abs_mean"] = float(np.mean(np.abs(command_residual)))
        if "material_saturated" in trace.files:
            metrics["material_saturation_fraction"] = float(
                np.mean(np.asarray(trace["material_saturated"][index], dtype=bool))
            )
        result[int(seed)] = metrics
    return result


def _load_run(run_dir: Path, *, interface: str) -> dict[str, Any]:
    episodes = {int(item["seed"]): item for item in _episodes(run_dir)}
    trace_metrics = _trace_episode_metrics(run_dir / "trace.npz", interface=interface)
    seeds = sorted(set(episodes) & set(trace_metrics)) or sorted(episodes)
    return {
        "dir": str(run_dir),
        "n_episodes": len(seeds),
        "episodes": {seed: episodes[seed] for seed in seeds if seed in episodes},
        "trace": {seed: trace_metrics[seed] for seed in seeds if seed in trace_metrics},
    }


def _paired(
    policy: dict[str, Any],
    zero: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    policy_seeds = set(policy["episodes"]) & set(zero["episodes"])
    policy_seeds &= set(policy["trace"]) & set(zero["trace"])
    seeds = sorted(policy_seeds)
    rows: list[dict[str, Any]] = []
    for episode_seed in seeds:
        policy_ep = policy["episodes"][episode_seed]
        zero_ep = zero["episodes"][episode_seed]
        policy_trace = policy["trace"][episode_seed]
        zero_trace = zero["trace"][episode_seed]
        row: dict[str, Any] = {
            "seed": episode_seed,
            "policy_success": _success(policy_ep),
            "zero_success": _success(zero_ep),
            "policy_plane_miss": bool(policy_ep.get("goal_plane_miss", False)),
            "zero_plane_miss": bool(zero_ep.get("goal_plane_miss", False)),
            "policy_timeout": bool(policy_ep.get("timeout", False)),
            "zero_timeout": bool(zero_ep.get("timeout", False)),
            "policy_safety": bool(policy_ep.get("safety", False)),
            "zero_safety": bool(zero_ep.get("safety", False)),
        }
        for key in (
            "path_vertical_rmse_m",
            "path_horizontal_rmse_m",
            "path_three_d_rmse_m",
            "path_vertical_mse_m2",
            "path_horizontal_mse_m2",
            "final_goal_distance_m",
            "final_vertical_abs_m",
            "final_horizontal_m",
            "final_miss_d_m",
            "near_goal_vertical_abs_mean_m",
            "near_goal_horizontal_mean_m",
            "terminal_phase_rate",
            "actuator_saturation_fraction",
            "material_saturation_fraction",
            "command_residual_abs_mean",
            "residual_abs_mean",
        ):
            policy_value = policy_trace.get(key)
            zero_value = zero_trace.get(key)
            if policy_value is not None and zero_value is not None:
                row[f"policy_{key}"] = float(policy_value)
                row[f"zero_{key}"] = float(zero_value)
                row[f"delta_{key}"] = float(policy_value) - float(zero_value)
        rows.append(row)

    summary: dict[str, Any] = {
        "n_pairs": len(rows),
        "policy_success_count": sum(row["policy_success"] for row in rows),
        "zero_success_count": sum(row["zero_success"] for row in rows),
        "policy_success_rate": (
            None if not rows else sum(row["policy_success"] for row in rows) / len(rows)
        ),
        "zero_success_rate": (
            None if not rows else sum(row["zero_success"] for row in rows) / len(rows)
        ),
        "policy_plane_miss_count": sum(row["policy_plane_miss"] for row in rows),
        "zero_plane_miss_count": sum(row["zero_plane_miss"] for row in rows),
        "policy_timeout_count": sum(row["policy_timeout"] for row in rows),
        "zero_timeout_count": sum(row["zero_timeout"] for row in rows),
        "policy_safety_count": sum(row["policy_safety"] for row in rows),
        "zero_safety_count": sum(row["zero_safety"] for row in rows),
        "flips": {
            "policy_only": [row["seed"] for row in rows if row["policy_success"] and not row["zero_success"]],
            "zero_only": [row["seed"] for row in rows if row["zero_success"] and not row["policy_success"]],
            "both": [row["seed"] for row in rows if row["policy_success"] and row["zero_success"]],
            "neither": [row["seed"] for row in rows if not row["policy_success"] and not row["zero_success"]],
        },
        "deltas": {},
        "flip_conditions": {},
    }
    for key in (
        "success",
        "plane_miss",
        "path_vertical_rmse_m",
        "path_horizontal_rmse_m",
        "path_three_d_rmse_m",
        "path_vertical_mse_m2",
        "path_horizontal_mse_m2",
        "final_goal_distance_m",
        "final_vertical_abs_m",
        "final_horizontal_m",
        "near_goal_vertical_abs_mean_m",
        "near_goal_horizontal_mean_m",
        "terminal_phase_rate",
        "actuator_saturation_fraction",
        "material_saturation_fraction",
        "command_residual_abs_mean",
        "residual_abs_mean",
    ):
        if key == "success":
            values = [
                float(row["policy_success"]) - float(row["zero_success"])
                for row in rows
            ]
        elif key == "plane_miss":
            values = [
                float(row["policy_plane_miss"]) - float(row["zero_plane_miss"])
                for row in rows
            ]
        else:
            values = [row[f"delta_{key}"] for row in rows if f"delta_{key}" in row]
        summary["deltas"][key] = _summarize_delta(
            key, values, samples=samples, seed=seed
        )

    delta_vertical_mse = summary["deltas"]["path_vertical_mse_m2"]["mean"]
    delta_horizontal_mse = summary["deltas"]["path_horizontal_mse_m2"]["mean"]
    positive_vertical = max(0.0, float(delta_vertical_mse or 0.0))
    positive_horizontal = max(0.0, float(delta_horizontal_mse or 0.0))
    denominator = positive_vertical + positive_horizontal
    summary["vertical_mse_share"] = (
        None if denominator <= 0.0 else positive_vertical / denominator
    )

    for label, key in (
        ("policy_only", "policy_only"),
        ("zero_only", "zero_only"),
        ("both", "both"),
        ("neither", "neither"),
    ):
        subset = [row for row in rows if row["seed"] in set(summary["flips"][key])]
        summary["flip_conditions"][label] = {
            "n": len(subset),
            "delta_vertical_rmse_m": _finite_mean(
                [row.get("delta_path_vertical_rmse_m", math.nan) for row in subset]
            ),
            "delta_horizontal_rmse_m": _finite_mean(
                [row.get("delta_path_horizontal_rmse_m", math.nan) for row in subset]
            ),
            "delta_final_goal_distance_m": _finite_mean(
                [row.get("delta_final_goal_distance_m", math.nan) for row in subset]
            ),
            "delta_final_vertical_abs_m": _finite_mean(
                [row.get("delta_final_vertical_abs_m", math.nan) for row in subset]
            ),
            "delta_final_horizontal_m": _finite_mean(
                [row.get("delta_final_horizontal_m", math.nan) for row in subset]
            ),
        }
    summary["rows"] = rows
    return summary


def _existing_ppo_context(t2_repo: Path) -> dict[str, Any]:
    root = t2_repo / "artifacts/lowlevel/goal86/screen"
    result: dict[str, Any] = {}
    for name in ("t2_ppo_zero_dev40", "t2_ppo_zero_strict125_dev40"):
        path = root / name / "summary.json"
        if not path.is_file():
            continue
        episodes = _read_json(path).get("episodes", [])
        for group, selected in (
            ("success", [item for item in episodes if _success(item)]),
            ("failure", [item for item in episodes if not _success(item)]),
        ):
            result[f"{name}/{group}"] = {
                "n": len(selected),
                "path_vertical_rmse_m": _finite_mean(
                    [float(item.get("path_vertical_rmse_m", math.nan)) for item in selected]
                ),
                "path_horizontal_rmse_m": _finite_mean(
                    [float(item.get("path_horizontal_rmse_m", math.nan)) for item in selected]
                ),
                "path_three_d_rmse_m": _finite_mean(
                    [float(item.get("path_three_d_rmse_m", math.nan)) for item in selected]
                ),
                "final_goal_distance_m": _finite_mean(
                    [float(item.get("final_goal_distance_m", math.nan)) for item in selected]
                ),
            }
    return result


def _outcome_geometry(run: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[int]] = {"success": [], "failure": []}
    for seed, episode in run["episodes"].items():
        groups["success" if _success(episode) else "failure"].append(seed)
    result: dict[str, Any] = {}
    for name, seeds in groups.items():
        traces = [run["trace"][seed] for seed in seeds if seed in run["trace"]]
        distances = [
            trace["final_goal_distance_m"]
            for trace in traces
            if "final_goal_distance_m" in trace
        ]
        result[name] = {
            "n": len(seeds),
            "final_goal_distance_m": _finite_mean(distances),
            "final_vertical_abs_m": _finite_mean(
                [
                    trace["final_vertical_abs_m"]
                    for trace in traces
                    if "final_vertical_abs_m" in trace
                ]
            ),
            "final_horizontal_m": _finite_mean(
                [
                    trace["final_horizontal_m"]
                    for trace in traces
                    if "final_horizontal_m" in trace
                ]
            ),
            "near_goal_vertical_abs_mean_m": _finite_mean(
                [
                    trace["near_goal_vertical_abs_mean_m"]
                    for trace in traces
                    if "near_goal_vertical_abs_mean_m" in trace
                ]
            ),
            "near_goal_horizontal_mean_m": _finite_mean(
                [
                    trace["near_goal_horizontal_mean_m"]
                    for trace in traces
                    if "near_goal_horizontal_mean_m" in trace
                ]
            ),
            "terminal_phase_rate": _finite_mean(
                [
                    trace["terminal_phase_rate"]
                    for trace in traces
                    if "terminal_phase_rate" in trace
                ]
            ),
        }
    failure_distances = [
        run["trace"][seed]["final_goal_distance_m"]
        for seed in groups["failure"]
        if seed in run["trace"] and "final_goal_distance_m" in run["trace"][seed]
    ]
    result["failure_distance_bins"] = {
        "1.25-1.5": sum(1 for value in failure_distances if 1.25 <= value < 1.5),
        "1.5-2.0": sum(1 for value in failure_distances if 1.5 <= value < 2.0),
        ">=2.0": sum(1 for value in failure_distances if value >= 2.0),
    }
    return result


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal145 trace 审计摘要",
        "",
        "> 由 `scripts/analyze_pc_goal145_capture_path_audit.py` 生成；只读 dev40 trace，"
        "不改 reward/terminal/success，不打开 held-out。",
        "",
        "## T2-SAC",
        "",
    ]
    for scenario, block in summary["t2"].items():
        if not block:
            continue
        lines += [
            f"### {scenario}",
            "",
            f"- 配对数：`{block['paired']['n_pairs']}`",
            f"- success：policy `{block['paired']['policy_success_count']}` / zero `{block['paired']['zero_success_count']}`",
            f"- plane miss：policy `{block['paired']['policy_plane_miss_count']}` / zero `{block['paired']['zero_plane_miss_count']}`",
            f"- vertical MSE share：`{block['paired']['vertical_mse_share']}`",
            "",
            "| metric | mean Δ (policy - zero) | 95% CI | + / - / 0 |",
            "|---|---:|---:|---:|",
        ]
        for key in (
            "success",
            "plane_miss",
            "path_vertical_rmse_m",
            "path_horizontal_rmse_m",
            "path_three_d_rmse_m",
            "final_goal_distance_m",
            "final_vertical_abs_m",
            "final_horizontal_m",
        ):
            item = block["paired"]["deltas"][key]
            lines.append(
                f"| `{key}` | {_fmt(item['mean'])} | "
                f"[{_fmt(item['ci95_low'])}, {_fmt(item['ci95_high'])}] | "
                f"{item['positive']} / {item['negative']} / {item['zero']} |"
            )
        lines += [
            "",
            f"- success flips policy-only：`{block['paired']['flips']['policy_only']}`",
            f"- success flips zero-only：`{block['paired']['flips']['zero_only']}`",
            "",
        ]

    lines += ["## C1-PPO", ""]
    for name, block in summary["c1"].items():
        if not block:
            continue
        lines += [
            f"### {name}",
            "",
            f"- 配对数：`{block['paired']['n_pairs']}`",
            f"- success：policy `{block['paired']['policy_success_count']}` / zero `{block['paired']['zero_success_count']}`",
            f"- plane miss：policy `{block['paired']['policy_plane_miss_count']}` / zero `{block['paired']['zero_plane_miss_count']}`",
            "",
            "| metric | mean Δ (policy - zero) | 95% CI | + / - / 0 |",
            "|---|---:|---:|---:|",
        ]
        for key in (
            "success",
            "plane_miss",
            "final_goal_distance_m",
            "final_vertical_abs_m",
            "final_horizontal_m",
            "near_goal_vertical_abs_mean_m",
            "near_goal_horizontal_mean_m",
            "terminal_phase_rate",
            "actuator_saturation_fraction",
        ):
            item = block["paired"]["deltas"][key]
            lines.append(
                f"| `{key}` | {_fmt(item['mean'])} | "
                f"[{_fmt(item['ci95_low'])}, {_fmt(item['ci95_high'])}] | "
                f"{item['positive']} / {item['negative']} / {item['zero']} |"
            )
        lines += [
            "",
            f"- success flips policy-only：`{block['paired']['flips']['policy_only']}`",
            f"- success flips zero-only：`{block['paired']['flips']['zero_only']}`",
            "",
            "| group | n | final distance | final vertical | final horizontal | near-goal vertical | near-goal horizontal |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for group in ("success", "failure"):
            item = block["policy_outcomes"][group]
            lines.append(
                f"| {group} | {item['n']} | {_fmt(item['final_goal_distance_m'])} | "
                f"{_fmt(item['final_vertical_abs_m'])} | {_fmt(item['final_horizontal_m'])} | "
                f"{_fmt(item['near_goal_vertical_abs_mean_m'])} | "
                f"{_fmt(item['near_goal_horizontal_mean_m'])} |"
            )
        lines += [
            "",
            f"- failure distance bins：`{block['policy_outcomes']['failure_distance_bins']}`",
            "",
        ]

    if summary.get("existing_ppo_context"):
        lines += ["## 已有 T2-PPO 零残差上下文", ""]
        lines += ["| group | n | vertical RMSE | horizontal RMSE | 3D RMSE | final distance |", "|---|---:|---:|---:|---:|---:|"]
        for key, value in summary["existing_ppo_context"].items():
            lines.append(
                f"| `{key}` | {value['n']} | {_fmt(value['path_vertical_rmse_m'])} | "
                f"{_fmt(value['path_horizontal_rmse_m'])} | {_fmt(value['path_three_d_rmse_m'])} | "
                f"{_fmt(value['final_goal_distance_m'])} |"
            )
        lines.append("")

    lines += ["## 裁决", "", f"- `{summary['verdict']}`", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.5f}"


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output_json = args.output_json or (root / "audit_summary.json")
    output_md = args.output_md or (root / "audit_summary.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    t2_repo = Path("/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式")
    summary: dict[str, Any] = {
        "root": str(root),
        "t2": {},
        "c1": {},
        "existing_ppo_context": _existing_ppo_context(t2_repo),
    }

    policy_dirs = {
        "default": [
            root / "t2/policy_t2_sac/default",
            root / "t2/policy_t2_sac_seed86202/default",
            root / "t2/policy_t2_sac_seed86203/default",
        ],
        "strict125": [
            root / "t2/policy_t2_sac/strict125",
            root / "t2/policy_t2_sac_seed86202/strict125",
            root / "t2/policy_t2_sac_seed86203/strict125",
        ],
    }
    zero_dirs = {
        "default": root / "t2/zero/default",
        "strict125": root / "t2/zero/strict125",
    }
    for scenario, dirs in policy_dirs.items():
        zero_dir = zero_dirs[scenario]
        if not zero_dir.is_dir():
            continue
        zero = _load_run(zero_dir, interface="t2")
        policy_runs = [_load_run(path, interface="t2") for path in dirs if path.is_dir()]
        if not policy_runs:
            continue
        scenario_block: dict[str, Any] = {"zero": zero, "policy_runs": []}
        pooled_rows: list[dict[str, Any]] = []
        for policy in policy_runs:
            paired = _paired(
                policy,
                zero,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            scenario_block["policy_runs"].append(
                {"dir": policy["dir"], "paired": paired}
            )
            pooled_rows.extend(paired["rows"])
        pooled = _pooled_rows(pooled_rows, samples=args.bootstrap_samples, seed=args.seed)
        scenario_block["paired"] = pooled
        summary["t2"][scenario] = scenario_block

    c1_specs = {
        "command_300k": (
            root / "c1/command_000300000_policy/default",
            root / "c1/command_000300000_zero/default",
        ),
        "actuator_300k": (
            root / "c1/actuator_000300000_policy/default",
            root / "c1/actuator_000300000_zero/default",
        ),
        "command_400k": (
            root / "c1/command_000400000_policy/default",
            None,
        ),
        "actuator_400k": (
            root / "c1/actuator_000400000_policy/default",
            None,
        ),
    }
    for name, (policy_dir, zero_dir) in c1_specs.items():
        if not policy_dir.is_dir() or zero_dir is None or not zero_dir.is_dir():
            continue
        policy = _load_run(policy_dir, interface="c1")
        zero = _load_run(zero_dir, interface="c1")
        if policy["n_episodes"] == 0 or zero["n_episodes"] == 0:
            continue
        summary["c1"][name] = {
            "policy": policy,
            "zero": zero,
            "policy_outcomes": _outcome_geometry(policy),
            "zero_outcomes": _outcome_geometry(zero),
            "paired": _paired(
                policy,
                zero,
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        }

    t2_verdicts: list[str] = []
    for scenario, block in summary["t2"].items():
        paired = block["paired"]
        delta = paired["deltas"]["path_vertical_rmse_m"]
        share = paired["vertical_mse_share"]
        vertical_positive = delta["mean"] is not None and delta["mean"] > 0.0
        ci_positive = (
            delta["ci95_low"] is not None and delta["ci95_low"] > 0.0
        )
        if vertical_positive and ci_positive and share is not None and share >= 0.50:
            t2_verdicts.append("VERTICAL_DOMINANT_CONFIRMED")
        elif (
            paired["deltas"]["path_horizontal_rmse_m"]["mean"] is not None
            and paired["deltas"]["path_horizontal_rmse_m"]["mean"] > 0.0
        ):
            t2_verdicts.append("HORIZONTAL_OR_MIXED_PATH_DEGRADATION")
        else:
            t2_verdicts.append("VERTICAL_NOT_CONFIRMED")

    if len(set(t2_verdicts)) == 1 and t2_verdicts:
        verdict = t2_verdicts[0]
    elif t2_verdicts:
        verdict = "MIXED_OR_NO_VERTICAL_DOMINANCE: " + ",".join(sorted(set(t2_verdicts)))
    else:
        verdict = "INCOMPLETE_TRACE_DATA"

    if summary["c1"]:
        c1_command = summary["c1"].get("command_300k")
        c1_actuator = summary["c1"].get("actuator_300k")
        c1_verdicts: list[str] = []
        for block in summary["c1"].values():
            outcomes = block["policy_outcomes"]
            success = outcomes["success"]
            failure = outcomes["failure"]
            if (
                success["n"] > 0
                and failure["n"] > 0
                and success["final_vertical_abs_m"] is not None
                and failure["final_vertical_abs_m"] is not None
                and failure["final_vertical_abs_m"]
                >= success["final_vertical_abs_m"] + 0.25
                and failure["final_goal_distance_m"] is not None
                and failure["final_goal_distance_m"] <= 2.5
            ):
                c1_verdicts.append("C1_TERMINAL_VERTICAL_DOMINANCE")
            if (
                block["paired"]["deltas"]["success"]["mean"] == 0.0
                and not block["paired"]["flips"]["policy_only"]
                and not block["paired"]["flips"]["zero_only"]
            ):
                c1_verdicts.append("C1_RESIDUAL_NO_CLOSED_LOOP_EFFECT")
        if c1_verdicts:
            verdict += " / " + " / ".join(sorted(set(c1_verdicts)))
        elif c1_command or c1_actuator:
            verdict += " / C1_TERMINAL_AUTHORITY_AUDIT_AVAILABLE"
    summary["verdict"] = verdict

    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, output_md)
    print(json.dumps({"json": str(output_json), "md": str(output_md), "verdict": verdict}, indent=2))
    return 0


def _pooled_rows(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_pairs": len(rows),
        "policy_success_count": sum(row["policy_success"] for row in rows),
        "zero_success_count": sum(row["zero_success"] for row in rows),
        "policy_plane_miss_count": sum(row["policy_plane_miss"] for row in rows),
        "zero_plane_miss_count": sum(row["zero_plane_miss"] for row in rows),
        "policy_timeout_count": sum(row["policy_timeout"] for row in rows),
        "zero_timeout_count": sum(row["zero_timeout"] for row in rows),
        "policy_safety_count": sum(row["policy_safety"] for row in rows),
        "zero_safety_count": sum(row["zero_safety"] for row in rows),
        "flips": {
            "policy_only": [row["seed"] for row in rows if row["policy_success"] and not row["zero_success"]],
            "zero_only": [row["seed"] for row in rows if row["zero_success"] and not row["policy_success"]],
            "both": [row["seed"] for row in rows if row["policy_success"] and row["zero_success"]],
            "neither": [row["seed"] for row in rows if not row["policy_success"] and not row["zero_success"]],
        },
        "deltas": {},
        "flip_conditions": {},
        "rows": rows,
    }
    for key in (
        "success",
        "plane_miss",
        "path_vertical_rmse_m",
        "path_horizontal_rmse_m",
        "path_three_d_rmse_m",
        "path_vertical_mse_m2",
        "path_horizontal_mse_m2",
        "final_goal_distance_m",
        "final_vertical_abs_m",
        "final_horizontal_m",
        "near_goal_vertical_abs_mean_m",
        "near_goal_horizontal_mean_m",
        "terminal_phase_rate",
        "actuator_saturation_fraction",
        "material_saturation_fraction",
        "command_residual_abs_mean",
        "residual_abs_mean",
    ):
        if key == "success":
            values = [
                float(row["policy_success"]) - float(row["zero_success"])
                for row in rows
            ]
        elif key == "plane_miss":
            values = [
                float(row["policy_plane_miss"]) - float(row["zero_plane_miss"])
                for row in rows
            ]
        else:
            values = [row[f"delta_{key}"] for row in rows if f"delta_{key}" in row]
        summary["deltas"][key] = _summarize_delta(
            key, values, samples=samples, seed=seed
        )
    delta_vertical_mse = summary["deltas"]["path_vertical_mse_m2"]["mean"]
    delta_horizontal_mse = summary["deltas"]["path_horizontal_mse_m2"]["mean"]
    positive_vertical = max(0.0, float(delta_vertical_mse or 0.0))
    positive_horizontal = max(0.0, float(delta_horizontal_mse or 0.0))
    denominator = positive_vertical + positive_horizontal
    summary["vertical_mse_share"] = (
        None if denominator <= 0.0 else positive_vertical / denominator
    )
    for label in ("policy_only", "zero_only", "both", "neither"):
        subset = [row for row in rows if row["seed"] in set(summary["flips"][label])]
        summary["flip_conditions"][label] = {
            "n": len(subset),
            "delta_vertical_rmse_m": _finite_mean(
                [row.get("delta_path_vertical_rmse_m", math.nan) for row in subset]
            ),
            "delta_horizontal_rmse_m": _finite_mean(
                [row.get("delta_path_horizontal_rmse_m", math.nan) for row in subset]
            ),
            "delta_final_goal_distance_m": _finite_mean(
                [row.get("delta_final_goal_distance_m", math.nan) for row in subset]
            ),
            "delta_final_vertical_abs_m": _finite_mean(
                [row.get("delta_final_vertical_abs_m", math.nan) for row in subset]
            ),
            "delta_final_horizontal_m": _finite_mean(
                [row.get("delta_final_horizontal_m", math.nan) for row in subset]
            ),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
