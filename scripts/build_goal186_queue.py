#!/usr/bin/env python3
"""Write the Goal186 MOD4-MOD0 GPU-safe training queue manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_gpus(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=86201)
    parser.add_argument("--iterations", type=int, default=2630)
    parser.add_argument("--need-mb", type=int, default=22000)
    parser.add_argument("--gpus", default="0-7")
    parser.add_argument("--modules", default="mod4,mod3,mod2,mod1,mod0")
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--checkpoint-keep-last", type=int, default=20)
    parser.add_argument("--checkpoint-keep-every", type=int, default=100)
    args = parser.parse_args()

    gpus = parse_gpus(args.gpus)
    jobs = []
    modules = [item.strip() for item in args.modules.split(",") if item.strip()]
    for module in modules:
        extra_args = [
            "--curriculum-eval-stage-boundaries",
            "--curriculum-route-gate-mode",
            "enforce",
            "--curriculum-route-thresholds",
            str(args.thresholds),
        ]
        if args.save_interval is not None:
            extra_args.extend(["--save-interval", str(args.save_interval)])
        extra_args.extend(
            [
                "--checkpoint-keep-last",
                str(args.checkpoint_keep_last),
                "--checkpoint-keep-every",
                str(args.checkpoint_keep_every),
            ]
        )
        jobs.append(
            {
                "name": f"{module}_pe2e_bcres_reward_v2_seed{args.seed}",
                "config": str(
                    args.worktree
                    / "configs/lowlevel/goal186"
                    / f"train_formal_{module}_pe2e_bcres_reward_v2_seed{args.seed}.json"
                ),
                "output": str(
                    args.run_root / "full_wave" / module / f"pe2e_bcres_reward_v2_seed{args.seed}"
                ),
                "seed": args.seed,
                "iterations": args.iterations,
                "need_mb": args.need_mb,
                "gpus": gpus,
                "resume_latest": True,
                "extra_args": extra_args,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
