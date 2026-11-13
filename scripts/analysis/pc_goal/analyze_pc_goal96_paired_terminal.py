#!/usr/bin/env python3
"""Goal96 paired seed and terminal-window divergence audit.

This is a read-only follow-up to the Phase A/B Goal96 audits.  It aligns the
initial, final, and frozen-base evaluations by dev seed and reports:

* which seeds flip between success and failure for each policy phase;
* how the normalized PPO residual changes from initial to final inside the
  pre-registered terminal windows.

The script does not load a policy, step an environment, train, change success
semantics, or read held-out seeds.  ``success`` is taken from the existing
evaluation summary; terminal rows are only used to bound trace windows.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal96_paired_terminal.v1"
WINDOWS_M = (("3-2", 2.0, 3.0), ("2-1.25", 1.25, 2.0))
PHASES = ("initial_policy", "final_policy", "frozen-base")
GROUPS = (
    "P_D3_default",
    "P_D3_strict125",
    "P_BC15_default",
    "P_BC15_strict125",
)
PAIRS = (
    ("initial_policy", "final_policy"),
    ("initial_policy", "frozen-base"),
    ("final_policy", "frozen-base"),
)
EPS = 1e-12


class AuditError(ValueError):
    """Raised when a trace cannot support the paired audit."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal96_terminal_credit_audit_20261111/traces_reward"
        ),
        help="Directory containing <group>_<phase>/trace.npz runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseC.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseC.md"
        ),
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=GROUPS,
        dest="groups",
        help="Audit only this group; repeatable. Default: all available groups.",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _success_by_seed(summary: dict[str, Any], run_dir: Path) -> dict[int, bool]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise AuditError(f"{run_dir}: summary has no episode list")
    result: dict[int, bool] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{run_dir}: malformed episode row {row!r}")
        result[int(row["seed"])] = bool(row["success"])
    return result


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


def _window_stats(
    distance: np.ndarray,
    residual: np.ndarray,
    lower: float,
    upper: float,
    dt_s: float,
) -> dict[str, Any] | None:
    mask = (distance <= upper + EPS) & (distance >= lower - EPS)
    if not bool(np.any(mask)):
        return None
    values = residual[mask]
    closure = np.zeros(values.shape[0], dtype=np.float64)
    if values.shape[0] > 1:
        closure[:-1] = -(np.diff(distance[mask]) / dt_s)
        closure[-1] = closure[-2]
    return {
        "rows": int(values.shape[0]),
        "mean_residual": values.mean(axis=0).tolist(),
        "mean_residual_l2": float(np.linalg.norm(values, axis=1).mean()),
        "mean_closure": float(closure.mean()) if closure.size else None,
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    trace_path = run_dir / "trace.npz"
    summary_path = run_dir / "summary.json"
    if not trace_path.is_file() or not summary_path.is_file():
        raise AuditError(f"{run_dir}: missing trace.npz or summary.json")
    summary = _read_json(summary_path)
    success_by_seed = _success_by_seed(summary, run_dir)
    dt_s = float(summary.get("dt_s", 0.02))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise AuditError(f"{run_dir}: invalid dt_s={dt_s}")
    with np.load(trace_path) as data:
        seeds = np.asarray(data["episode_seed"], dtype=np.int64)
        steps = np.asarray(data["step"], dtype=np.int64)
        terminal_kind = np.asarray(data["terminal_kind"], dtype=np.int64)
        distance = np.asarray(data["goal_distance_eta_m"], dtype=np.float64)
        residual = np.asarray(data["residual_action"], dtype=np.float64)

    episodes: dict[int, dict[str, Any]] = {}
    for seed in sorted(success_by_seed):
        rows = _episode_rows(seeds, steps, terminal_kind, seed)
        episodes[seed] = {
            "success": success_by_seed[seed],
            "terminal_kind": int(terminal_kind[rows[-1]]),
            "min_goal_distance_eta_m": float(np.min(distance[rows])),
            "windows": {
                name: _window_stats(
                    distance[rows], residual[rows], lower, upper, dt_s
                )
                for name, lower, upper in WINDOWS_M
            },
        }
    return {"run": run_dir.name, "episodes": episodes}


def _transition(base: dict[int, bool], other: dict[int, bool]) -> dict[str, Any]:
    gained = sorted(seed for seed, value in other.items() if value and not base[seed])
    lost = sorted(seed for seed, value in base.items() if value and not other[seed])
    return {
        "gained": gained,
        "lost": lost,
        "base_successes": sum(base.values()),
        "other_successes": sum(other.values()),
        "net": len(gained) - len(lost),
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= EPS:
        return None
    return float(np.dot(left_array, right_array) / denominator)


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _divergence(
    initial: dict[str, Any], final: dict[str, Any], transition: str
) -> dict[str, Any] | None:
    if initial is None or final is None:
        return None
    ratio = None
    if initial["mean_residual_l2"] > EPS:
        ratio = final["mean_residual_l2"] / initial["mean_residual_l2"]
    closure_delta = None
    if initial["mean_closure"] is not None and final["mean_closure"] is not None:
        closure_delta = final["mean_closure"] - initial["mean_closure"]
    return {
        "transition": transition,
        "cos_initial_final": _cosine(initial["mean_residual"], final["mean_residual"]),
        "residual_l2_initial": initial["mean_residual_l2"],
        "residual_l2_final": final["mean_residual_l2"],
        "residual_l2_ratio": ratio,
        "closure_initial": initial["mean_closure"],
        "closure_final": final["mean_closure"],
        "closure_delta": closure_delta,
    }


def _summarize_divergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "mean_cos_initial_final": _mean(row["cos_initial_final"] for row in rows),
        "median_cos_initial_final": _median(row["cos_initial_final"] for row in rows),
        "mean_residual_l2_ratio": _mean(row["residual_l2_ratio"] for row in rows),
        "median_residual_l2_ratio": _median(row["residual_l2_ratio"] for row in rows),
        "mean_closure_delta": _mean(row["closure_delta"] for row in rows),
        "median_closure_delta": _median(row["closure_delta"] for row in rows),
    }


def analyze_group(root: Path, group: str) -> dict[str, Any]:
    runs = {
        phase: analyze_run(root / f"{group}_{phase}")
        for phase in PHASES
    }
    success = {
        phase: {seed: row["success"] for seed, row in runs[phase]["episodes"].items()}
        for phase in PHASES
    }
    transitions = {
        f"{base}->{other}": _transition(success[base], success[other])
        for base, other in PAIRS
    }

    divergence: dict[str, Any] = {}
    for window_name, _, _ in WINDOWS_M:
        rows: list[dict[str, Any]] = []
        for seed in sorted(success["initial_policy"]):
            initial = runs["initial_policy"]["episodes"][seed]["windows"][window_name]
            final = runs["final_policy"]["episodes"][seed]["windows"][window_name]
            if initial is None or final is None:
                continue
            base_success = success["initial_policy"][seed]
            final_success = success["final_policy"][seed]
            if base_success and final_success:
                transition = "stable_success"
            elif not base_success and not final_success:
                transition = "stable_failure"
            elif final_success:
                transition = "gained"
            else:
                transition = "lost"
            row = _divergence(initial, final, transition)
            if row is None:
                continue
            row["seed"] = seed
            rows.append(row)
        by_transition: dict[str, Any] = {}
        for name in ("lost", "gained", "stable_success", "stable_failure"):
            subset = [row for row in rows if row["transition"] == name]
            if subset:
                by_transition[name] = _summarize_divergence(subset)
        divergence[window_name] = {
            "all": _summarize_divergence(rows),
            "by_transition": by_transition,
            "episodes": rows,
        }

    return {
        "group": group,
        "success": {phase: sum(values.values()) for phase, values in success.items()},
        "transitions": transitions,
        "terminal_divergence_initial_to_final": divergence,
    }


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "windows_m": [
            {"name": name, "lower": lower, "upper": upper}
            for name, lower, upper in WINDOWS_M
        ],
        "groups": groups,
    }


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.4f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal96 Phase C：逐 seed 先验保持与终端分叉",
        "",
        "本文件由 `scripts/analyze_pc_goal96_paired_terminal.py` 生成；只读现有 trace，不训练。",
        "",
    ]
    for group in payload["groups"]:
        lines.append(f"## {group['group']}")
        lines.append("")
        lines.append("| phase | success |")
        lines.append("|---|---:|")
        for phase in PHASES:
            lines.append(f"| {phase} | {group['success'][phase]}/40 |")
        lines.append("")
        lines.append("| transition | gained | lost | net |")
        lines.append("|---|---|---|---:|")
        for name, row in group["transitions"].items():
            lines.append(
                f"| {name} | {row['gained']} | {row['lost']} | {row['net']:+d} |"
            )
        lines.append("")
        lines.append("| window | n | mean cos(initial,final) | median cos | mean L2 ratio | mean closure delta |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for window_name, _, _ in WINDOWS_M:
            row = group["terminal_divergence_initial_to_final"][window_name]["all"]
            lines.append(
                f"| {window_name} | {row['episodes']} | "
                f"{_fmt(row['mean_cos_initial_final'])} | "
                f"{_fmt(row['median_cos_initial_final'])} | "
                f"{_fmt(row['mean_residual_l2_ratio'])} | "
                f"{_fmt(row['mean_closure_delta'])} |"
            )
        lines.append("")
        for window_name, _, _ in WINDOWS_M:
            payload_window = group["terminal_divergence_initial_to_final"][window_name]
            lost = payload_window["by_transition"].get("lost")
            if lost is None:
                continue
            lines.append(
                f"- `{window_name}` lost-only: cos={_fmt(lost['mean_cos_initial_final'])}, "
                f"L2 ratio={_fmt(lost['mean_residual_l2_ratio'])}, "
                f"closure delta={_fmt(lost['mean_closure_delta'])}"
            )
        lines.append("")
    lines.append("## 读法")
    lines.append("")
    lines.append(
        "`cos(initial,final)` 高但 `L2 ratio` 高，表示 PPO 主要放大了已有监督残差，"
        "没有学出新的方向；`closure delta <= 0` 表示放大的残差没有换来更快的径向闭合。"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    groups = list(args.groups) if args.groups else [
        group
        for group in GROUPS
        if all((args.trace_root / f"{group}_{phase}").is_dir() for phase in PHASES)
    ]
    if not groups:
        raise AuditError(f"{args.trace_root}: no complete goal96 groups found")
    payload = summarize([analyze_group(args.trace_root, group) for group in groups])
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
