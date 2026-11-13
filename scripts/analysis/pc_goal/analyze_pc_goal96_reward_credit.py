#!/usr/bin/env python3
"""Goal96 Phase B read-only reward / credit audit.

The analyzer consumes evaluation traces produced with
``--trace-reward-breakdown``. It never loads a policy, steps an environment,
trains, changes success semantics, or reads held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal96_reward_credit_audit.v1"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20_261_112
EPS = 1e-12
WINDOWS_M = (
    ("10-5", 5.0, 10.0),
    ("5-3", 3.0, 5.0),
    ("3-2", 2.0, 3.0),
    ("2-1.25", 1.25, 2.0),
)
PRETERMINAL_WINDOWS = ("5-3", "3-2")
REWARD_TO_GO_THRESHOLDS_M = (5.0, 3.0, 2.0)
TERMINAL_COMPONENTS = ("reward_terminal", "reward_goal_plane_miss")
EXCLUDED_COMPONENTS = {"reward_total", "reward_ppo_scaled"}
TERMINAL_POSITIVE_SHARE_LIMIT = 0.80
DISCOUNT_GAMMA = 0.995
DIRECT_REWARD_FIELDS = (
    "reward_command_yaw",
    "reward_command_pitch",
    "reward_command_speed",
    "reward_hold",
    "reward_roll_angle",
    "reward_roll_rate",
    "reward_action",
    "reward_saturation",
    "reward_route_speed",
    "reward_material_saturation",
    "reward_safety",
    "reward_goal_plane_miss",
    "reward_terminal_progress",
    "reward_terminal",
)


class AuditError(ValueError):
    """Raised when a trace cannot support the reward audit."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/lowlevel/goal96_terminal_credit_audit_20261111/traces_reward")
        ],
        help="One or more directories scanned recursively for trace.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseB.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseB.md"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def _first_field(data: np.lib.npyio.NpzFile, names: Iterable[str], label: str) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    raise AuditError(f"{label}: missing trace fields {list(names)}")


def _episode_rows(
    seeds: np.ndarray, steps: np.ndarray, terminal_kind: np.ndarray, seed: int
) -> np.ndarray:
    rows = np.flatnonzero(seeds == seed)
    if rows.size == 0:
        raise AuditError(f"seed {seed}: no trace rows")
    rows = rows[np.argsort(steps[rows], kind="stable")]
    ended = np.flatnonzero(terminal_kind[rows] != 0)
    if ended.size:
        rows = rows[: int(ended[0]) + 1]
    return rows


def _closure(distance: np.ndarray, dt_s: float) -> np.ndarray:
    closure = np.zeros_like(distance, dtype=np.float64)
    if distance.size < 2:
        return closure
    closure[:-1] = -(distance[1:] - distance[:-1]) / dt_s
    closure[-1] = closure[-2]
    return closure


def _mean(values: Sequence[float]) -> float | None:
    if len(values) == 0:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _bootstrap_mean_diff(
    success: np.ndarray,
    failure: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    success = np.asarray(success, dtype=np.float64)
    failure = np.asarray(failure, dtype=np.float64)
    result: dict[str, Any] = {
        "success_mean": _mean(success),
        "failure_mean": _mean(failure),
        "diff": None,
        "ci_low": None,
        "ci_high": None,
        "success_episodes": int(success.size),
        "failure_episodes": int(failure.size),
        "valid_draws": 0,
    }
    if success.size == 0 or failure.size == 0:
        return result
    result["diff"] = float(np.mean(success) - np.mean(failure))
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        left = rng.choice(success, size=success.size, replace=True)
        right = rng.choice(failure, size=failure.size, replace=True)
        differences[index] = float(np.mean(left) - np.mean(right))
    low, high = np.quantile(differences, [0.025, 0.975])
    result.update(
        {
            "ci_low": float(low),
            "ci_high": float(high),
            "valid_draws": int(draws),
        }
    )
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size < 2:
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None
    left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(
        np.float64
    )
    right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(
        np.float64
    )
    left_rank -= float(np.mean(left_rank))
    right_rank -= float(np.mean(right_rank))
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator <= EPS:
        return None
    return float(np.dot(left_rank, right_rank) / denominator)


def _terminal_proximity(run_dir: Path) -> tuple[dict[int, bool], int]:
    path = run_dir / "terminal_proximity.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise AuditError(f"{run_dir}: missing terminal_proximity.json and summary.json")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        raise AuditError(f"{path}: episodes must be a list")
    success = {int(row["seed"]): bool(row["success"]) for row in episodes}
    if not success:
        raise AuditError(f"{path}: no episode rows")
    return success, len(episodes)


def _load_reward_weights(summary: dict[str, Any]) -> dict[str, float | str]:
    value = summary.get("environment_config")
    if value is None:
        return {
            "pc_control_mode": "alos",
            "w_path_3d": 0.0,
            "w_yaw": 0.0,
            "w_prog": 0.0,
            "w_path_speed_smooth": 0.0,
            "w_path_energy": 0.0,
            "w_u": 0.0,
        }
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    config = json.loads(path.read_text(encoding="utf-8"))
    reward = config.get("reward", {})
    mode = str(config.get("experiment", {}).get("pc_control_mode", "alos"))
    return {
        "pc_control_mode": mode,
        "w_path_3d": float(reward.get("w_path_3d", 0.0) if mode == "alos" else reward.get("w_3d", 0.0)),
        "w_yaw": float(reward.get("w_yaw", 0.0)),
        "w_prog": float(reward.get("w_prog", 0.0)),
        "w_path_speed_smooth": float(reward.get("w_path_speed_smooth", 0.0)),
        "w_path_energy": float(reward.get("w_path_energy", 0.0)),
        "w_u": float(reward.get("w_u", 0.0)),
    }


def _reward_contributions(
    data: np.lib.npyio.NpzFile, weights: dict[str, float | str]
) -> dict[str, np.ndarray]:
    if "reward_total" not in data.files:
        raise AuditError("trace is missing reward_total")
    contributions = {
        name: np.asarray(data[name], dtype=np.float64)
        for name in DIRECT_REWARD_FIELDS
        if name in data.files
    }
    if "reward_path_cross" in data.files and "reward_path_vertical" in data.files:
        cross = np.asarray(data["reward_path_cross"], dtype=np.float64)
        vertical = np.asarray(data["reward_path_vertical"], dtype=np.float64)
        contributions["reward_path_track_3d"] = (
            -float(weights["w_path_3d"]) * np.sqrt(np.square(cross) + np.square(vertical)) / 5.0
        )
    if "reward_path_heading" in data.files:
        contributions["reward_path_heading_reward"] = (
            -float(weights["w_yaw"])
            * np.abs(np.asarray(data["reward_path_heading"], dtype=np.float64))
            / math.pi
        )
    if "reward_path_progress" in data.files:
        contributions["reward_path_progress_reward"] = float(weights["w_prog"]) * np.asarray(
            data["reward_path_progress"], dtype=np.float64
        )
    if "reward_speed_smooth" in data.files:
        contributions["reward_speed_smooth_reward"] = -float(
            weights["w_path_speed_smooth"]
        ) * np.asarray(data["reward_speed_smooth"], dtype=np.float64)
    if "reward_energy" in data.files:
        contributions["reward_energy_reward"] = -float(weights["w_path_energy"]) * np.asarray(
            data["reward_energy"], dtype=np.float64
        )
    if float(weights["w_u"]) != 0.0:
        raise AuditError(
            "trace cannot reconstruct w_u path reward because path speed error is not recorded"
        )
    reconstructed = np.sum(np.stack(tuple(contributions.values()), axis=0), axis=0)
    total = np.asarray(data["reward_total"], dtype=np.float64)
    max_error = float(np.max(np.abs(reconstructed - total))) if total.size else 0.0
    if max_error > 1e-6:
        raise AuditError(
            "reward breakdown does not reconstruct reward_total: "
            f"max_abs_error={max_error:.6g}"
        )
    return contributions


def _episode_record(
    *,
    data: np.lib.npyio.NpzFile,
    seed: int,
    success: bool,
    dt_s: float,
    contributions: dict[str, np.ndarray],
) -> dict[str, Any]:
    seeds = np.asarray(data["episode_seed"])
    steps = np.asarray(data["step"])
    terminal_kind = np.asarray(data["terminal_kind"])
    rows = _episode_rows(seeds, steps, terminal_kind, seed)
    distance = np.asarray(
        _first_field(data, ("goal_distance_eta_m", "goal_distance_m"), "episode"),
        dtype=np.float64,
    )[rows]
    total = np.asarray(data["reward_total"], dtype=np.float64)[rows]
    closure = _closure(distance, dt_s)
    window_closure: dict[str, float | None] = {}
    for name, lower, upper in WINDOWS_M:
        mask = (distance >= lower) & (distance < upper)
        window_closure[name] = _mean(closure[mask]) if bool(mask.any()) else None
    reward_returns = {
        name: float(np.sum(values[rows])) for name, values in contributions.items()
    }
    terminal_return = sum(reward_returns.get(name, 0.0) for name in TERMINAL_COMPONENTS)
    terminal_row = np.zeros_like(total)
    for name in TERMINAL_COMPONENTS:
        if name in contributions:
            terminal_row += contributions[name][rows]
    positive_total = float(np.sum(np.maximum(total, 0.0)))
    discount = DISCOUNT_GAMMA ** np.arange(total.size, dtype=np.float64)
    discounted_total = float(np.dot(discount, total))
    discounted_terminal = float(np.dot(discount, terminal_row))
    entry_reward_to_go: dict[str, float | None] = {}
    entry_reward_to_go_ex_terminal: dict[str, float | None] = {}
    reward_to_go = np.cumsum(total[::-1])[::-1]
    for threshold in REWARD_TO_GO_THRESHOLDS_M:
        entered = np.flatnonzero(distance <= threshold)
        if entered.size:
            entry = int(entered[0])
            entry_reward_to_go[f"{threshold:g}m"] = float(reward_to_go[entry])
            tail_total = total[entry:]
            tail_terminal = terminal_row[entry:]
            tail_discount = DISCOUNT_GAMMA ** np.arange(tail_total.size, dtype=np.float64)
            entry_reward_to_go_ex_terminal[f"{threshold:g}m"] = float(
                np.dot(tail_discount, tail_total - tail_terminal)
            )
        else:
            entry_reward_to_go[f"{threshold:g}m"] = None
            entry_reward_to_go_ex_terminal[f"{threshold:g}m"] = None
    return {
        "seed": int(seed),
        "success": bool(success),
        "steps": int(rows.size),
        "total_return": float(np.sum(total)),
        "positive_total_return": positive_total,
        "terminal_return": float(terminal_return),
        "discounted_total_return": discounted_total,
        "discounted_terminal_return": discounted_terminal,
        "discounted_terminal_abs_share": (
            abs(discounted_terminal) / max(abs(discounted_total), EPS)
        ),
        "terminal_positive_share": (
            max(float(terminal_return), 0.0) / positive_total
            if positive_total > EPS
            else None
        ),
        "component_returns": reward_returns,
        "window_closure": window_closure,
        "preterminal_closure": _mean(
            [window_closure[name] for name in PRETERMINAL_WINDOWS if window_closure[name] is not None]
        ),
        "reward_to_go_at_entry": entry_reward_to_go,
        "discounted_reward_to_go_ex_terminal_at_entry": entry_reward_to_go_ex_terminal,
    }


def analyze_run(
    run_dir: Path,
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    trace_path = run_dir / "trace.npz"
    if not trace_path.exists():
        raise AuditError(f"{trace_path}: not found")
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise AuditError(f"{summary_path}: not found")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dt_s = float(summary.get("dt_s", summary.get("metrics", {}).get("dt_s", 0.0)))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        with np.load(trace_path, allow_pickle=False) as data:
            time_s = np.asarray(data["time_s"], dtype=np.float64)
        unique = np.unique(time_s)
        if len(unique) < 2:
            raise AuditError(f"{summary_path}: cannot infer dt_s from trace")
        dt_s = float(np.median(np.diff(unique[: min(len(unique), 1000)])))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise AuditError(f"{summary_path}: invalid dt_s={dt_s!r}")
    success_by_seed, episode_count = _terminal_proximity(run_dir)
    reward_weights = _load_reward_weights(summary)
    with np.load(trace_path, allow_pickle=False) as data:
        contributions = _reward_contributions(data, reward_weights)
        records = [
            _episode_record(
                data=data,
                seed=seed,
                success=success,
                dt_s=dt_s,
                contributions=contributions,
            )
            for seed, success in sorted(success_by_seed.items())
        ]
    component_names = tuple(contributions)
    successes = [record for record in records if record["success"]]
    failures = [record for record in records if not record["success"]]

    closure_windows: dict[str, Any] = {}
    for name, _, _ in WINDOWS_M:
        success_values = np.asarray(
            [record["window_closure"][name] for record in successes if record["window_closure"][name] is not None],
            dtype=np.float64,
        )
        failure_values = np.asarray(
            [record["window_closure"][name] for record in failures if record["window_closure"][name] is not None],
            dtype=np.float64,
        )
        closure_windows[name] = _bootstrap_mean_diff(
            success_values,
            failure_values,
            draws=bootstrap_draws,
            seed=bootstrap_seed + len(closure_windows),
        )

    preterminal = _bootstrap_mean_diff(
        np.asarray(
            [record["preterminal_closure"] for record in successes if record["preterminal_closure"] is not None],
            dtype=np.float64,
        ),
        np.asarray(
            [record["preterminal_closure"] for record in failures if record["preterminal_closure"] is not None],
            dtype=np.float64,
        ),
        draws=bootstrap_draws,
        seed=bootstrap_seed + 10,
    )

    component_returns = np.asarray(
        [[record["component_returns"][name] for name in component_names] for record in records],
        dtype=np.float64,
    )
    component_variances = np.var(component_returns, axis=0)
    variance_denominator = float(np.sum(component_variances))
    component_summary = {
        name: {
            "success_mean": _mean(
                [record["component_returns"][name] for record in successes]
            ),
            "failure_mean": _mean(
                [record["component_returns"][name] for record in failures]
            ),
            "variance_share": (
                float(component_variances[index] / variance_denominator)
                if variance_denominator > EPS
                else None
            ),
        }
        for index, name in enumerate(component_names)
    }

    terminal_share_success = np.asarray(
        [record["terminal_positive_share"] for record in successes if record["terminal_positive_share"] is not None],
        dtype=np.float64,
    )
    terminal_share_failure = np.asarray(
        [record["terminal_positive_share"] for record in failures if record["terminal_positive_share"] is not None],
        dtype=np.float64,
    )
    terminal_share = _bootstrap_mean_diff(
        terminal_share_success,
        terminal_share_failure,
        draws=bootstrap_draws,
        seed=bootstrap_seed + 11,
    )
    success_terminal_share_mean = _mean(terminal_share_success.tolist())
    discounted_terminal_share = _bootstrap_mean_diff(
        np.asarray(
            [
                record["discounted_terminal_abs_share"]
                for record in successes
                if record["discounted_terminal_abs_share"] is not None
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                record["discounted_terminal_abs_share"]
                for record in failures
                if record["discounted_terminal_abs_share"] is not None
            ],
            dtype=np.float64,
        ),
        draws=bootstrap_draws,
        seed=bootstrap_seed + 12,
    )

    reward_to_go: dict[str, Any] = {}
    reward_to_go_ex_terminal: dict[str, Any] = {}
    for threshold in REWARD_TO_GO_THRESHOLDS_M:
        key = f"{threshold:g}m"
        reward_to_go[key] = _bootstrap_mean_diff(
            np.asarray(
                [record["reward_to_go_at_entry"][key] for record in successes if record["reward_to_go_at_entry"][key] is not None],
                dtype=np.float64,
            ),
            np.asarray(
                [record["reward_to_go_at_entry"][key] for record in failures if record["reward_to_go_at_entry"][key] is not None],
                dtype=np.float64,
            ),
            draws=bootstrap_draws,
            seed=bootstrap_seed + 20 + int(threshold * 10),
        )
        reward_to_go_ex_terminal[key] = _bootstrap_mean_diff(
            np.asarray(
                [
                    record["discounted_reward_to_go_ex_terminal_at_entry"][key]
                    for record in successes
                    if record["discounted_reward_to_go_ex_terminal_at_entry"][key]
                    is not None
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    record["discounted_reward_to_go_ex_terminal_at_entry"][key]
                    for record in failures
                    if record["discounted_reward_to_go_ex_terminal_at_entry"][key]
                    is not None
                ],
                dtype=np.float64,
            ),
            draws=bootstrap_draws,
            seed=bootstrap_seed + 40 + int(threshold * 10),
        )

    closure_supported = bool(
        preterminal["diff"] is not None
        and preterminal["diff"] > 0.0
        and preterminal["ci_low"] is not None
        and preterminal["ci_low"] > 0.0
    )
    terminal_share_ok = bool(
        success_terminal_share_mean is not None
        and success_terminal_share_mean < TERMINAL_POSITIVE_SHARE_LIMIT
    )
    preterminal_records = [
        record for record in records if record["preterminal_closure"] is not None
    ]
    return {
        "run": run_dir.name,
        "trace": str(trace_path),
        "successes": len(successes),
        "episodes": episode_count,
        "dt_s": dt_s,
        "window_closure": closure_windows,
        "preterminal_closure": preterminal,
        "reward_to_go_at_entry": reward_to_go,
        "discounted_reward_to_go_ex_terminal_at_entry": reward_to_go_ex_terminal,
        "terminal_positive_share": terminal_share,
        "discounted_terminal_abs_share": discounted_terminal_share,
        "success_terminal_positive_share_mean": success_terminal_share_mean,
        "component_returns": component_summary,
        "preterminal_closure_vs_success_spearman": _spearman(
            np.asarray(
                [1.0 if record["success"] else 0.0 for record in preterminal_records],
                dtype=np.float64,
            ),
            np.asarray(
                [record["preterminal_closure"] for record in preterminal_records],
                dtype=np.float64,
            ),
        ),
        "decision": {
            "preterminal_closure_positive": closure_supported,
            "terminal_positive_share_below_0.80": terminal_share_ok,
            "reward_aligned": bool(closure_supported and terminal_share_ok),
        },
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "terminal_positive_share_limit": TERMINAL_POSITIVE_SHARE_LIMIT,
        "windows_m": [
            {"name": name, "lower": lower, "upper": upper}
            for name, lower, upper in WINDOWS_M
        ],
        "runs": runs,
    }


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.4f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal96 Phase B：reward / credit 只读审计",
        "",
        "本文件由 `scripts/analyze_pc_goal96_reward_credit.py` 生成；只读 reward trace，不训练。",
        "",
        "`reward_to_go_at_entry` 是 episode 首次进入该距离壳层时的剩余 return；`terminal_positive_share` 是 terminal 正收益占所有正 total reward 的比例。",
        "",
    ]
    for run in payload["runs"]:
        lines.append(f"## {run['run']}（success {run['successes']}/{run['episodes']}）")
        lines.append("")
        lines.append("| 指标 | success − failure [95% CI] |")
        lines.append("|---|---:|")
        for key in ("5-3", "3-2", "2-1.25"):
            value = run["window_closure"][key]
            lines.append(
                f"| closure {key} | {_fmt(value['diff'])} [{_fmt(value['ci_low'])}, {_fmt(value['ci_high'])}] |"
            )
        value = run["preterminal_closure"]
        lines.append(
            f"| **preterminal closure** | {_fmt(value['diff'])} [{_fmt(value['ci_low'])}, {_fmt(value['ci_high'])}] |"
        )
        for key in ("5m", "3m", "2m"):
            value = run["reward_to_go_at_entry"][key]
            lines.append(
                f"| reward-to-go @ {key} | {_fmt(value['diff'])} [{_fmt(value['ci_low'])}, {_fmt(value['ci_high'])}] |"
            )
        for key in ("5m", "3m", "2m"):
            value = run["discounted_reward_to_go_ex_terminal_at_entry"][key]
            lines.append(
                f"| discounted ex-terminal RTG @ {key} | {_fmt(value['diff'])} [{_fmt(value['ci_low'])}, {_fmt(value['ci_high'])}] |"
            )
        lines.append("")
        lines.append(
            f"- success terminal positive share: `{_fmt(run['success_terminal_positive_share_mean'])}`"
        )
        discounted_share = run["discounted_terminal_abs_share"]
        lines.append(
            "- discounted |terminal| / |total|: "
            f"success `{_fmt(discounted_share['success_mean'])}`, "
            f"failure `{_fmt(discounted_share['failure_mean'])}`"
        )
        top_components = sorted(
            run["component_returns"].items(),
            key=lambda item: abs(item[1]["variance_share"] or 0.0),
            reverse=True,
        )[:5]
        lines.append(
            "- top variance components: "
            + ", ".join(
                f"`{name}`={_fmt(values['variance_share'])}"
                for name, values in top_components
            )
        )
        decision = run["decision"]
        lines.append(
            f"- decision: closure_positive={decision['preterminal_closure_positive']}, "
            f"terminal_share_below_0.80={decision['terminal_positive_share_below_0.80']}, "
            f"reward_aligned={decision['reward_aligned']}"
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = sorted(
        {path.parent for root in args.trace_root for path in root.rglob("trace.npz")}
    )
    if not run_dirs:
        raise AuditError(f"{args.trace_root}: no trace.npz found")
    runs = [
        analyze_run(
            path,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        for path in run_dirs
    ]
    payload = summarize(runs)
    payload["bootstrap_seed"] = args.bootstrap_seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(payload, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
