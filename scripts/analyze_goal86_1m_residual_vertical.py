#!/usr/bin/env python3
"""Audit within-episode stern-residual/vertical-error coupling in rich traces."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _corr(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 20 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def analyze_trace(trace_path: Path, episodes_path: Path) -> list[dict[str, Any]]:
    train_tag = trace_path.parents[1].name
    train_seed = int(train_tag.rsplit("_", 1)[1])
    scenario = trace_path.parent.name
    pressure = next(
        value
        for value in ("current10", "noise_n2", "plant20", "nominal")
        if f"p-{value}__" in scenario
    )
    radius = "strict125" if scenario.endswith("strict125") else "default"
    episodes = {
        int(row["seed"]): row
        for row in json.loads(episodes_path.read_text(encoding="utf-8"))
    }
    with np.load(trace_path) as trace:
        episode_id = trace["episode_id"]
        seed = trace["seed"]
        active = trace["active"]
        distance = trace["goal_distance_eta"]
        window = active & (distance >= 2.0) & (distance < 10.0)
        rows: list[dict[str, Any]] = []
        for episode in np.unique(episode_id[window]):
            mask = window & (episode_id == episode)
            eval_seed = int(np.unique(seed[mask])[0])
            stern = trace["residual"][mask, 1]
            vertical = trace["path_vertical_error"][mask]
            pitch_rate = trace["nu"][mask, 4]
            heave = trace["nu"][mask, 2]
            outcome = episodes[eval_seed]
            rows.append(
                {
                    "train_seed": train_seed,
                    "pressure": pressure,
                    "radius": radius,
                    "eval_seed": eval_seed,
                    "success": bool(outcome["success"]),
                    "steps": int(mask.sum()),
                    "corr_stern_vertical": _corr(stern, vertical),
                    "corr_stern_pitch_rate": _corr(stern, pitch_rate),
                    "corr_stern_heave": _corr(stern, heave),
                    "stern_abs_mean": float(np.mean(np.abs(stern))),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["train_seed"], row["pressure"], row["radius"])].append(row)
    result: list[dict[str, Any]] = []
    for (train_seed, pressure, radius), group in sorted(grouped.items()):
        success = [row for row in group if row["success"]]
        failure = [row for row in group if not row["success"]]

        def median_corr(selected: list[dict[str, Any]]) -> float | None:
            values = [
                row["corr_stern_vertical"]
                for row in selected
                if np.isfinite(row["corr_stern_vertical"])
            ]
            return None if not values else float(np.median(values))

        result.append(
            {
                "train_seed": train_seed,
                "pressure": pressure,
                "radius": radius,
                "episodes": len(group),
                "success_episodes": len(success),
                "failure_episodes": len(failure),
                "success_corr_median": median_corr(success),
                "failure_corr_median": median_corr(failure),
                "success_stern_abs_mean": (
                    None
                    if not success
                    else float(np.mean([row["stern_abs_mean"] for row in success]))
                ),
                "failure_stern_abs_mean": (
                    None
                    if not failure
                    else float(np.mean([row["stern_abs_mean"] for row in failure]))
                ),
            }
        )
    return result


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}f}"


def render_markdown(summary: list[dict[str, Any]], pattern: str) -> str:
    lines = [
        "# Goal86 1M residual/vertical coupling",
        "",
        f"Pattern: `{pattern}`",
        "",
        "| train seed | pressure | radius | episodes | success | failure | success corr | failure corr | success |stern| | failure |stern| |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        values = dict(row)
        values.update(
            success_corr_median=_fmt(row["success_corr_median"]),
            failure_corr_median=_fmt(row["failure_corr_median"]),
            success_stern_abs_mean=_fmt(row["success_stern_abs_mean"]),
            failure_stern_abs_mean=_fmt(row["failure_stern_abs_mean"]),
        )
        lines.append(
            "| {train_seed} | {pressure} | {radius} | {episodes} | "
            "{success_episodes} | {failure_episodes} | {success_corr_median} | "
            "{failure_corr_median} | {success_stern_abs_mean} | "
            "{failure_stern_abs_mean} |".format(**values)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        default="near10_2_*/p-*/episodes.json",
        help="glob relative to --root",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    rows: list[dict[str, Any]] = []
    for path_text in sorted(glob.glob(str(root / args.pattern))):
        episodes_path = Path(path_text)
        rows.extend(analyze_trace(episodes_path.parent / "trace.npz", episodes_path))
    if not rows:
        raise ValueError(f"no episodes matched {args.pattern!r} under {root}")
    summary = summarize(rows)
    markdown = render_markdown(summary, args.pattern)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"rows": rows, "summary": summary}, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
