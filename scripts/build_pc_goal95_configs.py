#!/usr/bin/env python3
"""Generate Goal95 pretrain/PPO configs from the frozen Goal94 S15 contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "configs/lowlevel/goal94_20261111/s15.json"
OUTPUT = REPO_ROOT / "configs/lowlevel/goal95_20261111"


def _write(name: str, payload: dict) -> None:
    path = OUTPUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _arm(
    source: dict,
    *,
    name: str,
    output: str,
    residual_scale: float,
    initial_std: float,
    seed: int = 86201,
) -> dict:
    payload = copy.deepcopy(source)
    payload["experiment"]["name"] = name
    payload["experiment"]["output"] = output
    payload["experiment"]["seed"] = seed
    payload["base_policy"]["residual_scale"] = float(residual_scale)
    payload["pc"]["actuator_residual_scale"] = 1.5
    payload["training"]["initial_residual_output_std"] = float(initial_std)
    return payload


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    root = "artifacts/lowlevel/goal95_supervised_residual_20261111"
    specs = {
        "s0_ppo.json": dict(
            name="pc_goal95_s0_zero_ppo_20261111",
            output=f"{root}/S0",
            residual_scale=1.0,
            initial_std=0.03,
        ),
        "p_d3_pretrain.json": dict(
            name="pc_goal95_p_d3_pretrain_20261111",
            output=f"{root}/pretrain_P_D3",
            residual_scale=1.0,
            initial_std=0.0,
        ),
        "p_d3_ppo.json": dict(
            name="pc_goal95_p_d3_ppo_20261111",
            output=f"{root}/P_D3",
            residual_scale=1.0,
            initial_std=0.0,
        ),
        "p_bc15_pretrain.json": dict(
            name="pc_goal95_p_bc15_pretrain_20261111",
            output=f"{root}/pretrain_P_BC15",
            residual_scale=2.0 / 3.0,
            initial_std=0.0,
        ),
        "p_bc15_ppo.json": dict(
            name="pc_goal95_p_bc15_ppo_20261111",
            output=f"{root}/P_BC15",
            residual_scale=2.0 / 3.0,
            initial_std=0.0,
        ),
    }
    for filename, spec in specs.items():
        _write(filename, _arm(source, **spec))
    print(f"wrote {len(specs)} configs to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
