#!/usr/bin/env python3
"""Build a standing-eval arm manifest for the frozen Goal86 1M checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (86201, 86202, 86203)
VARIANTS = ("full", "near10_2", "mid_only", "far_off_only")
DEFAULT_CHECKPOINT_ROOT = (
    Path("/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式")
    / "artifacts/lowlevel/goal86/screen"
)
TRAINING_CONFIG = "configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json"
AUTHORITY_CONFIG = {
    "full": None,
    "near10_2": "configs/lowlevel/goal154/authority/near10_2.json",
    "mid_only": "configs/lowlevel/goal154/authority/mid_only.json",
    "far_off_only": "configs/lowlevel/goal154/authority/far_off_only.json",
}


def checkpoint_for_seed(root: Path, seed: int) -> Path:
    if seed == 86201:
        return root / "t2_sac/final.zip"
    return root / f"t2_sac_seed{seed}/final.zip"


def build_arms(
    checkpoint_root: Path,
    *,
    seeds: Sequence[int] = SEEDS,
    variants: Sequence[str] = VARIANTS,
) -> list[dict]:
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    if not variants:
        raise ValueError("at least one variant is required")
    arms = [
        {
            "tag": "zero",
            "paradigm": "T2-SAC-1M",
            "evaluator": "sac",
            "training_config": TRAINING_CONFIG,
            "action_mode": "zero-residual",
            "residual_scale": 1.0,
            "disable_curriculum_direct_command": True,
        }
    ]
    for variant in variants:
        for seed in seeds:
            checkpoint = checkpoint_for_seed(checkpoint_root, seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            config = TRAINING_CONFIG if variant == "full" else AUTHORITY_CONFIG[variant]
            assert config is not None
            gate_mode = "none"
            if variant != "full":
                payload = json.loads((REPO_ROOT / config).read_text(encoding="utf-8"))
                gate_mode = str(payload.get("learner_terminal_gate_mode", "none"))
            arms.append(
                {
                    "tag": f"{variant}_{seed}",
                    "paradigm": "T2-SAC-1M",
                    "evaluator": "sac",
                    "training_config": config,
                    "checkpoint": str(checkpoint),
                    "action_mode": "policy",
                    "residual_scale": 1.0,
                    "learner_terminal_gate_mode": gate_mode,
                }
            )
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    arms = build_arms(
        args.checkpoint_root,
        seeds=args.seeds,
        variants=args.variants,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"schema_version": "auv_pc_standing_eval_arms.v1", "arms": arms},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"arms": len(arms), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
