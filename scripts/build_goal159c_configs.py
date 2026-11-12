#!/usr/bin/env python3
"""Build the corrected Goal159 C3/ALOS training configs.

The original Goal159 environment used pc_stage=C0, which activates the
training-only direct-command scaffold.  This builder keeps the H3 controller,
current range, reward, terminal contract, and SAC hyperparameters unchanged,
but moves the fixed environment to C3 so training follows the ALOS path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = ROOT / "configs/lowlevel/goal159/env_h3_current07_10_train.json"
ENV_OUTPUT = ROOT / "configs/lowlevel/goal159/env_h3_current07_10_train_c3.json"
TRAIN_SOURCE = ROOT / "configs/lowlevel/goal159/train_seed86201.json"
TRAIN_DIR = ROOT / "configs/lowlevel/goal159"
OUTPUT_ROOT = "artifacts/lowlevel/goal159c_h3_current_sac_screen_20261112"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_environment() -> dict[str, Any]:
    payload = json.loads(ENV_SOURCE.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["experiment"]["name"] = "pc_goal159c_h3_current07_10_train_c3_20261112"
    payload["experiment"]["pc_stage"] = "C3"
    payload["experiment"]["output"] = f"{OUTPUT_ROOT}/train_env"
    payload["reward"]["hold_bonus"] = False
    _write(ENV_OUTPUT, payload)
    return payload


def build_train_configs() -> list[Path]:
    outputs: list[Path] = []
    for seed in (86201, 86202, 86203):
        payload = json.loads(TRAIN_SOURCE.read_text(encoding="utf-8"))
        payload = copy.deepcopy(payload)
        payload["environment_config"] = str(ENV_OUTPUT.relative_to(ROOT))
        payload["experiment"]["name"] = f"pc_goal159c_h3_current_sac_seed{seed}_20261112"
        payload["experiment"]["output"] = f"{OUTPUT_ROOT}/train/seed{seed}"
        payload["experiment"]["train_seeds"] = [seed]
        payload["training"]["seed"] = seed
        path = TRAIN_DIR / f"train_c3_seed{seed}.json"
        _write(path, payload)
        outputs.append(path)
    return outputs


def main() -> int:
    build_environment()
    train_configs = build_train_configs()
    manifest = {
        "schema_version": "auv_goal159c_config_manifest.v1",
        "source_environment": str(ENV_SOURCE.relative_to(ROOT)),
        "source_environment_sha256": _sha256(ENV_SOURCE),
        "environment": str(ENV_OUTPUT.relative_to(ROOT)),
        "environment_sha256": _sha256(ENV_OUTPUT),
        "train_configs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in train_configs
        ],
        "correction": (
            "pc_stage C0 -> C3 and hold_bonus true -> false to disable the "
            "direct-command training scaffold. hold_bonus is changed only because "
            "it is a C0/C1 reward flag; terminal, success, goal radius, current "
            "range, and all other reward terms are unchanged."
        ),
    }
    _write(TRAIN_DIR / "goal159c_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
