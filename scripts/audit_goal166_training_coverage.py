#!/usr/bin/env python3
"""Audit whether Goal166 training covers terminal episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (86201, 86202, 86203)
ARM = "G1_gain_near10_2"


def _episode_steps(root: Path) -> list[int]:
    steps: list[int] = []
    for path in sorted(root.rglob("zero_seed*/episodes.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps.extend(int(row["steps"]) for row in payload)
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--zero-eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    zero_steps = _episode_steps(args.zero_eval_root)
    zero_stats = {
        "episodes": len(zero_steps),
        "min": int(min(zero_steps)) if zero_steps else 0,
        "p10": float(np.percentile(zero_steps, 10)) if zero_steps else float("nan"),
        "median": float(np.percentile(zero_steps, 50)) if zero_steps else float("nan"),
        "p90": float(np.percentile(zero_steps, 90)) if zero_steps else float("nan"),
        "max": int(max(zero_steps)) if zero_steps else 0,
    }
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = args.train_root / ARM / f"seed{seed}" / "training_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        num_envs = int(payload["num_envs"])
        transitions = int(payload["total_env_transitions_actual"])
        steps_per_env = transitions / num_envs
        completed = int(payload.get("completed_episodes", 0))
        success = int(payload.get("success_episodes", 0))
        rows.append(
            {
                "seed": seed,
                "num_envs": num_envs,
                "transitions": transitions,
                "steps_per_env": steps_per_env,
                "completed_episodes": completed,
                "success_episodes": success,
                "completion_gate": completed >= 256,
                "success_gate": success >= 128,
                "zero_min_le_steps_per_env": bool(
                    zero_stats["min"] <= steps_per_env
                ),
                "manifest_finite": bool(payload.get("finite", True)),
                "speed_contract": payload.get("speed_contract"),
            }
        )

    result = {
        "train_root": str(args.train_root),
        "zero_eval_root": str(args.zero_eval_root),
        "zero_episode_steps": zero_stats,
        "rows": rows,
        "pass": bool(
            all(row["completion_gate"] for row in rows)
            and all(row["success_gate"] for row in rows)
            and all(row["manifest_finite"] for row in rows)
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_coverage.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Goal166 terminal-coverage audit",
        "",
        f"Zero episodes: {zero_stats['episodes']}; min/p10/median/p90/max = "
        f"{zero_stats['min']}/{zero_stats['p10']:.0f}/{zero_stats['median']:.0f}/"
        f"{zero_stats['p90']:.0f}/{zero_stats['max']} steps.",
        "",
        "| seed | num_envs | steps/env | completed | success | completion gate | success gate |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {num_envs} | {steps_per_env:.0f} | {completed_episodes} | "
            "{success_episodes} | {completion_gate} | {success_gate} |".format(**row)
        )
    (args.output / "training_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "training_coverage.md")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
