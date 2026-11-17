#!/usr/bin/env python3
"""Build frozen-reference rollout configs for Goal182 D13.6 calibration.

These configs are intentionally outside the formal matrix.  They start from the
frozen 5D BC checkpoints and scale the trainable residual to a negligible value
so the rollout metrics describe the reference policy, not a treatment arm.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL_CONFIG_ROOT = REPO_ROOT / "configs" / "lowlevel" / "goal182"
CALIBRATION_CONFIG_ROOT = GOAL_CONFIG_ROOT / "calibration"
ENVIRONMENT_SOURCE = "env_path_fixed_speed_formal.json"
CALIBRATION_STAGES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")

SPECS = (
    (
        "pe2e",
        "train_formal_mod4_pe2e_residual_5d_m_seed86201.json",
        "calibration_reference_pe2e_frozen_bc.json",
    ),
    (
        "alosb",
        "train_formal_mod4_alosb_residual_5d_m_seed86201.json",
        "calibration_reference_alosb_frozen_bc.json",
    ),
)

REFERENCE_RESIDUAL_SCALE = (
    1e-6  # magic-number-ok: negligible residual for frozen reference rollouts
)
REFERENCE_LEARNING_RATE = (
    1e-12  # magic-number-ok: effectively no PPO update during reference rollouts
)
CALIBRATION_ITERATIONS = 100
STAGE_CALIBRATION_ITERATIONS = 100


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_config(control: str, source: Path, destination: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["base_policy"]["residual_scale"] = REFERENCE_RESIDUAL_SCALE
    payload["algorithm"]["learning_rate"] = REFERENCE_LEARNING_RATE
    payload["algorithm"]["schedule"] = "fixed"
    for key in ("learning_rate_actor", "learning_rate_critic", "learning_rate_dade_head"):
        payload["algorithm"].pop(key, None)
    payload["training"]["iterations"] = CALIBRATION_ITERATIONS
    payload["runner"]["save_interval"] = CALIBRATION_ITERATIONS
    payload["experiment"].update(
        {
            "name": f"goal182_route_gate_calibration_reference_{control}",
            "output": (
                "artifacts/lowlevel/goal182_ppo_stack_governance_20261114/"
                f"calibration/reference_{control}"
            ),
            "seed": 86201,
        }
    )
    _write(destination, payload)
    return payload


def build_stage_environment(stage: str) -> Path:
    source = GOAL_CONFIG_ROOT / ENVIRONMENT_SOURCE
    payload = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    payload["curriculum"]["initial_stage"] = stage
    payload["curriculum"]["promotion_enabled"] = False
    payload["curriculum"]["stage_mix"] = [{"stage": stage, "weight": 1.0}]
    payload["experiment"].update(
        {
            "name": f"goal182_route_gate_calibration_env_{stage}",
            "output": (
                f"artifacts/lowlevel/goal182_ppo_stack_governance_20261114/calibration/env_{stage}"
            ),
        }
    )
    destination = CALIBRATION_CONFIG_ROOT / f"env_{stage}.json"
    _write(destination, payload)
    return destination


def build_stage_config(
    control: str,
    source: Path,
    stage: str,
    environment_relative_path: str,
) -> Path:
    payload = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    payload["environment"]["config"] = environment_relative_path
    payload["training"]["iterations"] = STAGE_CALIBRATION_ITERATIONS
    payload["runner"]["save_interval"] = STAGE_CALIBRATION_ITERATIONS
    payload["experiment"].update(
        {
            "name": f"goal182_route_gate_calibration_reference_{control}_{stage}",
            "output": (
                "artifacts/lowlevel/goal182_ppo_stack_governance_20261114/"
                f"calibration/reference_{control}_{stage}"
            ),
            "seed": 86201,
        }
    )
    destination = CALIBRATION_CONFIG_ROOT / f"reference_{control}_{stage}.json"
    _write(destination, payload)
    return destination


def main() -> int:
    CALIBRATION_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    reference_configs: dict[str, Path] = {}
    for control, source_name, destination_name in SPECS:
        source = GOAL_CONFIG_ROOT / source_name
        destination = GOAL_CONFIG_ROOT / destination_name
        build_config(control, source, destination)
        reference_configs[control] = destination
        print(destination)
    environments = {stage: build_stage_environment(stage) for stage in CALIBRATION_STAGES}
    for path in environments.values():
        print(path)
    for stage, path in environments.items():
        relative = path.relative_to(REPO_ROOT).as_posix()
        for control, source in reference_configs.items():
            destination = build_stage_config(control, source, stage, relative)
            print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
