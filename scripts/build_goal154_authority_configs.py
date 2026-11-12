#!/usr/bin/env python3
"""Build Goal154 eval-only residual authority configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json"
OUT = REPO_ROOT / "configs/lowlevel/goal154/authority"
VARIANTS = {
    "full": {"gate": False, "enter": 10.0, "full": 2.0, "mode": "none", "scale": 1.0},
    "near10_2": {"gate": True, "enter": 10.0, "full": 2.0, "mode": "none", "scale": 1.0},
    "far_off_only": {"gate": True, "enter": 10.0001, "full": 10.0, "mode": "none", "scale": 1.0},
    "near_only": {"gate": True, "enter": 2.0001, "full": 2.0, "mode": "none", "scale": 1.0},
    "mid_only": {"gate": True, "enter": 10.0, "full": 2.0, "mode": "shell", "scale": 1.0},
    "full_scale_0.5": {"gate": False, "enter": 10.0, "full": 2.0, "mode": "none", "scale": 0.5},
    "full_scale_0.75": {"gate": False, "enter": 10.0, "full": 2.0, "mode": "none", "scale": 0.75},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "frozen_before_measurement": True,
        "goal": "Goal154",
        "source": str(args.base.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(args.base),
        "variants": {},
    }
    for name, spec in VARIANTS.items():
        payload = copy.deepcopy(base)
        payload["experiment"]["name"] = f"pc_goal154_authority_{name}_20261111"
        payload["experiment"]["output"] = (
            f"artifacts/lowlevel/goal154_residual_authority_20261111/eval/{name}"
        )
        payload["residual_terminal_gate_enabled"] = bool(spec["gate"])
        payload["residual_terminal_gate_enter_m"] = float(spec["enter"])
        payload["residual_terminal_gate_full_m"] = float(spec["full"])
        payload["learner_terminal_gate_mode"] = str(spec["mode"])
        payload["evaluation"]["residual_scale"] = float(spec["scale"])
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["variants"][name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            **spec,
        }
    (args.out.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
