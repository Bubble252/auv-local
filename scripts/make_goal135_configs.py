#!/usr/bin/env python3
"""Generate the pre-registered Goal135 authority-regularized SAC screen configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

BASE_CONFIG = Path(
    "configs/lowlevel/goal135/pc_goal135_authority_sac_base_20261111.json"
)
DEFAULT_ROOT = Path("artifacts/lowlevel/goal135_authority_regularized_sac_20261111")
SEEDS = tuple(range(86201, 86209))
SCREEN_TRANSITIONS = 1_000_192
ARMS: dict[str, dict[str, float]] = {
    "R030": {"rho": 0.30, "beta": 2.0},
    "R025": {"rho": 0.25, "beta": 2.0},
    "R020": {"rho": 0.20, "beta": 2.0},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.base.is_file():
        raise FileNotFoundError(args.base)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    if base.get("schema_version") != "auv_pc_goal135_authority_sac.v1":
        raise ValueError("unexpected Goal135 base schema")
    if int(base["training"]["total_env_transitions"]) != SCREEN_TRANSITIONS:
        raise ValueError("base config must freeze the 1M screen budget")

    training_dir = args.root / "training_configs"
    jobs: list[dict[str, Any]] = []
    for arm, parameters in ARMS.items():
        for seed in SEEDS:
            payload = copy.deepcopy(base)
            output = args.root / "screen" / arm / f"S_s{seed}"
            payload["training"]["seed"] = seed
            payload["training"]["total_env_transitions"] = SCREEN_TRANSITIONS
            payload["training"]["authority_regularization"] = {
                "enabled": True,
                "form": "hinge_rms",
                "rho": float(parameters["rho"]),
                "beta": float(parameters["beta"]),
            }
            payload["experiment"].update(
                {
                    "algorithm": "SAC",
                    "arm": arm,
                    "name": f"pc_goal135_authority_sac_{arm}_s{seed}_20261111",
                    "output": str(output),
                    "seed": seed,
                }
            )
            config_path = training_dir / arm / f"S_s{seed}.json"
            _write(config_path, payload)
            jobs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "rho": float(parameters["rho"]),
                    "beta": float(parameters["beta"]),
                    "config": str(config_path),
                    "config_sha256": _sha256(config_path),
                    "output": str(output),
                    "total_env_transitions": SCREEN_TRANSITIONS,
                }
            )

    manifest = {
        "schema_version": "auv_pc_goal135_authority_sac_manifest.v1",
        "base_config": str(args.base),
        "base_config_sha256": _sha256(args.base),
        "root": str(args.root),
        "arms": ARMS,
        "train_seeds": list(SEEDS),
        "screen_transitions": SCREEN_TRANSITIONS,
        "jobs": jobs,
        "heldout_seed_start": 100,
        "heldout_seed_count": 50,
    }
    _write(args.root / "config_manifest.json", manifest)
    print(json.dumps({"root": str(args.root), "jobs": len(jobs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
