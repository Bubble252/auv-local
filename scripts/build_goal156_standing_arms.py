#!/usr/bin/env python3
"""Build the Goal156 standing-eval arm manifest from the full-trained checkpoints.

Goal156 trains three full-authority SAC seeds once.  Every deployment gate is an
evaluation-only arm over the same checkpoint, so the standing screen must never
silently retrain or select a checkpoint by dev success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (86201, 86202, 86203)
VARIANTS = ("full", "near10_2", "mid_only", "far_off_only")
AUTHORITY_CONFIG = {
    "full": None,
    "near10_2": "configs/lowlevel/goal154/authority/near10_2.json",
    "mid_only": "configs/lowlevel/goal154/authority/mid_only.json",
    "far_off_only": "configs/lowlevel/goal154/authority/far_off_only.json",
}


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _training_config(variant: str, seed: int) -> str:
    if variant == "full":
        return f"configs/lowlevel/goal156/full_seed{seed}.json"
    return AUTHORITY_CONFIG[variant]


def build_arms(
    root: Path,
    *,
    seeds: Sequence[int] = SEEDS,
    variants: Iterable[str] = VARIANTS,
    repo_root: Path = REPO_ROOT,
    zero_only: bool = False,
) -> list[dict]:
    """Return the frozen Goal156 standing-eval arm list.

    ``zero`` uses the seed-86201 training config so the evaluator sees the same
    observation/action contract as the policy arms.  The policy arms for every
    gate reuse ``train/full_seed<seed>/final.zip``.
    """

    root = root.resolve()
    repo_root = repo_root.resolve()
    selected_variants = tuple(variants)
    unknown = sorted(set(selected_variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    if not selected_variants:
        raise ValueError("at least one variant is required")

    zero_config = repo_root / "configs/lowlevel/goal156/full_seed86201.json"
    if not zero_config.is_file():
        raise FileNotFoundError(zero_config)
    arms: list[dict] = [
        {
            "tag": "zero",
            "paradigm": "T2-SAC",
            "evaluator": "sac",
            "training_config": _relative(zero_config, repo_root),
            "action_mode": "zero-residual",
            "residual_scale": 1.0,
            "disable_curriculum_direct_command": True,
        }
    ]
    if zero_only:
        return arms

    for variant in selected_variants:
        for seed in seeds:
            checkpoint = root / "train" / f"full_seed{seed}" / "final.zip"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            config_rel = _training_config(variant, seed)
            if config_rel is None:
                raise AssertionError(f"missing authority config for {variant}")
            config_path = repo_root / config_rel
            config = _load(config_path)
            arms.append(
                {
                    "tag": f"{variant}_{seed}",
                    "paradigm": "T2-SAC",
                    "evaluator": "sac",
                    "training_config": config_rel,
                    "checkpoint": str(checkpoint),
                    "action_mode": "policy",
                    "residual_scale": 1.0,
                    "learner_terminal_gate_mode": str(
                        config.get("learner_terminal_gate_mode", "none")
                    ),
                }
            )
    return arms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "artifacts/lowlevel/goal156_full_train_eval_gate",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--zero-only",
        action="store_true",
        help="emit only the zero-residual baseline; no checkpoint is required",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output = args.output or (root / "standing_arms.json")
    arms = build_arms(
        root,
        seeds=args.seeds,
        variants=args.variants,
        zero_only=args.zero_only,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"schema_version": "auv_pc_standing_eval_arms.v1", "arms": arms},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"arms": len(arms), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
