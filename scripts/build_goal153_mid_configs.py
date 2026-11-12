#!/usr/bin/env python3
"""Build Goal153 mid-only (2-10 m shell) training configs for R0/R1/R2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "configs/lowlevel/goal153/train"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["R0", "R1", "R2"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"frozen_before_measurement": True, "goal": "Goal153", "variant": "mid_only", "configs": {}}
    for arm in args.arms:
        base_path = REPO_ROOT / f"configs/lowlevel/goal153/train/{arm}_C1_sac.json"
        payload = copy.deepcopy(json.loads(base_path.read_text(encoding="utf-8")))
        payload["experiment"]["name"] = f"pc_goal153_{arm}_mid_C1_sac_1m_86201_20261111"
        payload["experiment"]["output"] = (
            f"artifacts/lowlevel/goal153_reward_curriculum_20261111/pilot_bc_1m_mid/train/{arm}_mid_C1"
        )
        payload["residual_terminal_gate_enabled"] = True
        payload["residual_terminal_gate_enter_m"] = 10.0
        payload["residual_terminal_gate_full_m"] = 2.0
        payload["learner_terminal_gate_mode"] = "shell"
        payload["training"]["total_env_transitions"] = 1_000_000
        payload["training"]["screen_env_transitions"] = 1_000_000
        path = args.out / f"{arm}_mid_C1_sac.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["configs"][arm] = {
            "base": str(base_path.relative_to(REPO_ROOT)),
            "base_sha256": _sha256(base_path),
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "environment_config": payload["environment_config"],
            "gate": {"enabled": True, "enter_m": 10.0, "full_m": 2.0, "mode": "shell"},
        }
    (args.out.parent / "goal153_mid_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
