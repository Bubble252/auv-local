#!/usr/bin/env python3
"""Build the Goal155 standing-eval arm manifest from trained checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (86201, 86202, 86203)
ARMS = ("near10_2", "mid_only", "far_off_only")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "artifacts/lowlevel/goal155_authority_training_screen_20261111",
    )
    parser.add_argument(
        "--full-root",
        type=Path,
        default=Path(
            "/data/yhx/underwater-robotics/src/plan_p1/"
            "settled_局部规划器_反应式/artifacts/lowlevel/goal86/screen"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or (root / "standing_arms.json")
    arms: list[dict] = [
        {
            "tag": "zero",
            "paradigm": "T2-SAC",
            "evaluator": "sac",
            "training_config": "configs/lowlevel/goal155/near10_2_seed86201.json",
            "action_mode": "zero-residual",
            "residual_scale": 1.0,
            "disable_curriculum_direct_command": True,
        }
    ]
    for arm in ARMS:
        for seed in SEEDS:
            config_path = REPO_ROOT / f"configs/lowlevel/goal155/{arm}_seed{seed}.json"
            config = _load(config_path)
            checkpoint = root / "train" / f"{arm}_seed{seed}" / "final.zip"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            arms.append(
                {
                    "tag": f"{arm}_{seed}",
                    "paradigm": "T2-SAC",
                    "evaluator": "sac",
                    "training_config": str(config_path.relative_to(REPO_ROOT)),
                    "checkpoint": str(checkpoint),
                    "action_mode": "policy",
                    "residual_scale": 1.0,
                    "learner_terminal_gate_mode": config["learner_terminal_gate_mode"],
                }
            )
    for seed in SEEDS:
        full_dir = "t2_sac" if seed == 86201 else f"t2_sac_seed{seed}"
        checkpoint = args.full_root / full_dir / "final.zip"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        arms.append(
            {
                "tag": f"full_{seed}",
                "paradigm": "T2-SAC",
                "evaluator": "sac",
                "training_config": "configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json",
                "checkpoint": str(checkpoint),
                "action_mode": "policy",
                "residual_scale": 1.0,
                "learner_terminal_gate_mode": "none",
            }
        )
    payload = {"schema_version": "auv_pc_standing_eval_arms.v1", "arms": arms}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arms": len(arms), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
