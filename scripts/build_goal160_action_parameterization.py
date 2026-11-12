#!/usr/bin/env python3
"""Build the preregistered Goal160 action-parameterization configs.

This script only writes configs and a manifest.  It never starts training.
The three arms differ only in the residual action interface and, for the
terminal arm, whether the command residual is gated during training.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = ROOT / "configs/lowlevel/goal159/env_h3_current07_10_train_c3.json"
BASE_TRAIN = ROOT / "configs/lowlevel/goal159/train_c3_seed86201.json"
BASE_EVAL_POLICY = ROOT / "configs/lowlevel/goal154/authority/near10_2.json"
BASE_EVAL_ENV = ROOT / "configs/lowlevel/goal157/env_h3.json"
OUT = ROOT / "configs/lowlevel/goal160"
TRAIN_SEEDS = (86201, 86202, 86203)
MAX_SPEED_MPS = 2.5
TRAIN_SPEED_MPS = 1.2

ARMS = {
    "A_actuator_near10_2": {
        "action_interface": "actuator",
        "terminal_train_gate": False,
        "description": "actuator residual baseline; train full, deploy near10_2",
    },
    "B_command_near10_2": {
        "action_interface": "command",
        "terminal_train_gate": False,
        "description": "3-D command residual; train full, deploy near10_2",
    },
    "C_command_terminal_train": {
        "action_interface": "command",
        "terminal_train_gate": True,
        "description": "3-D command residual trained only inside 10 m -> 2 m",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _env_payload(
    base: dict,
    *,
    arm: str,
    action_interface: str,
    terminal_train_gate: bool,
) -> dict:
    payload = deepcopy(base)
    reference = payload["reference"]
    reference["speed_mps"] = TRAIN_SPEED_MPS
    if "command_speed_range_mps" in reference:
        reference["command_speed_range_mps"] = [TRAIN_SPEED_MPS, TRAIN_SPEED_MPS]
    payload["experiment"]["name"] = f"pc_goal160_{arm.lower()}_train_20261112"
    payload["experiment"]["pc_stage"] = "C3"
    payload["experiment"]["pc_control_mode"] = "alos"
    command = payload["command_residual"]
    command["enabled"] = action_interface == "command"
    command["terminal_gate_enabled"] = bool(terminal_train_gate)
    if terminal_train_gate:
        command["terminal_gate_enter_m"] = 10.0
        command["terminal_gate_full_m"] = 2.0
    else:
        command["terminal_gate_enter_m"] = 20.0
        command["terminal_gate_full_m"] = 5.0
    return payload


def _train_payload(base: dict, *, arm: str, action_interface: str, seed: int, env_path: str) -> dict:
    payload = deepcopy(base)
    payload["action_interface"] = action_interface
    payload["environment_config"] = env_path
    payload["experiment"]["arm"] = arm
    payload["experiment"]["name"] = f"pc_goal160_{arm.lower()}_seed{seed}_20261112"
    payload["experiment"]["output"] = (
        f"artifacts/lowlevel/goal160_action_parameterization_20261112/train/{arm}/seed{seed}"
    )
    payload["experiment"]["train_seeds"] = [seed]
    payload["experiment"]["train_seed_start"] = seed
    payload["experiment"]["train_seed_count"] = 1
    payload["training"]["seed"] = seed
    payload["training"]["total_env_transitions"] = 1_000_000
    payload["training"]["screen_env_transitions"] = 1_000_000
    return payload


def _eval_env_payload(base: dict, *, command_near10_2: bool) -> dict:
    """Build an H3 eval environment with a 2.0 m termination radius.

    Scoring radius is supplied per run via ``--eval-goal-radius-m``.  The
    base env must terminate at the larger radius so the default 2.0 m score
    is valid, while strict125 can still score at 1.25 m on the same traces.
    """

    payload = deepcopy(base)
    payload["plant"]["goal_radius_m"] = 2.0
    payload["experiment"]["name"] = (
        "pc_goal160_h3_command_near10_2_eval_20261112"
        if command_near10_2
        else "pc_goal160_h3_actuator_default_eval_20261112"
    )
    payload["experiment"]["pc_stage"] = "C3"
    payload["experiment"]["pc_control_mode"] = "alos"
    command = payload["command_residual"]
    command["enabled"] = True
    command["terminal_gate_enabled"] = True
    command["terminal_gate_enter_m"] = 10.0 if command_near10_2 else 20.0
    command["terminal_gate_full_m"] = 2.0 if command_near10_2 else 5.0
    return payload


def main() -> None:
    base_env = json.loads(BASE_ENV.read_text(encoding="utf-8"))
    base_train = json.loads(BASE_TRAIN.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "schema_version": "auv_goal160_action_parameterization.v1",
        "goal": "Goal160",
        "date": "2026-11-12",
        "train_speed_mps": TRAIN_SPEED_MPS,
        "max_speed_mps": MAX_SPEED_MPS,
        "speed_contract": "v_cmd + v_current + v_variation <= 2.5; 2.5 only with zero/weak current and negligible variation",
        "heldout": "closed",
        "arms": {},
        "train_configs": [],
    }

    env_paths: dict[str, str] = {}
    for arm, spec in ARMS.items():
        env_name = f"env_{arm.lower()}.json"
        env_path = OUT / env_name
        env_payload = _env_payload(
            base_env,
            arm=arm,
            action_interface=spec["action_interface"],
            terminal_train_gate=spec["terminal_train_gate"],
        )
        _write(env_path, env_payload)
        rel_env = f"configs/lowlevel/goal160/{env_name}"
        env_paths[arm] = rel_env
        manifest["arms"][arm] = {
            "action_interface": spec["action_interface"],
            "terminal_train_gate": spec["terminal_train_gate"],
            "description": spec["description"],
            "environment_config": rel_env,
            "environment_sha256": _sha256(env_path),
            "train_speed_mps": TRAIN_SPEED_MPS,
            "command_residual": {
                key: env_payload["command_residual"][key]
                for key in (
                    "enabled",
                    "terminal_gate_enabled",
                    "terminal_gate_enter_m",
                    "terminal_gate_full_m",
                    "scale_heading_rad",
                    "scale_pitch_rad",
                    "scale_speed_mps",
                )
            },
        }

        for seed in TRAIN_SEEDS:
            train_name = f"train_{arm.lower()}_seed{seed}.json"
            train_path = OUT / train_name
            train_payload = _train_payload(
                base_train,
                arm=arm,
                action_interface=spec["action_interface"],
                seed=seed,
                env_path=rel_env,
            )
            _write(train_path, train_payload)
            manifest["train_configs"].append(
                {
                    "arm": arm,
                    "seed": seed,
                    "path": f"configs/lowlevel/goal160/{train_name}",
                    "sha256": _sha256(train_path),
                }
            )

    # Eval-only command gate: near10_2 deployment.  This is intentionally a
    # separate env config so training-time and deployment-time gates stay
    # distinguishable in the manifest.
    base_eval_env = json.loads(BASE_EVAL_ENV.read_text(encoding="utf-8"))
    actuator_eval_name = "env_h3_actuator_default_eval.json"
    actuator_eval_path = OUT / actuator_eval_name
    _write(actuator_eval_path, _eval_env_payload(base_eval_env, command_near10_2=False))
    manifest["actuator_eval_environment"] = {
        "path": f"configs/lowlevel/goal160/{actuator_eval_name}",
        "sha256": _sha256(actuator_eval_path),
    }

    eval_env_name = "env_h3_command_near10_2_eval.json"
    eval_env_path = OUT / eval_env_name
    _write(eval_env_path, _eval_env_payload(base_eval_env, command_near10_2=True))
    manifest["command_eval_environment"] = {
        "path": f"configs/lowlevel/goal160/{eval_env_name}",
        "sha256": _sha256(eval_env_path),
    }

    # Command arms need a policy config whose action interface matches the
    # checkpoint.  The actuator terminal gate is disabled here; the command
    # residual gate lives in the eval environment above.
    eval_policy = json.loads(BASE_EVAL_POLICY.read_text(encoding="utf-8"))
    eval_policy["action_interface"] = "command"
    eval_policy["residual_terminal_gate_enabled"] = False
    eval_policy["environment_config"] = manifest["command_eval_environment"]["path"]
    eval_policy["experiment"]["name"] = "pc_goal160_command_near10_2_eval_20261112"
    eval_policy["experiment"]["output"] = (
        "artifacts/lowlevel/goal160_action_parameterization_20261112/eval_command_near10_2"
    )
    eval_policy_path = OUT / "eval_command_near10_2.json"
    _write(eval_policy_path, eval_policy)
    manifest["command_eval_policy"] = {
        "path": f"configs/lowlevel/goal160/{eval_policy_path.name}",
        "sha256": _sha256(eval_policy_path),
    }

    _write(OUT / "goal160_manifest.json", manifest)
    print(f"wrote {len(manifest['train_configs'])} train configs to {OUT}")


if __name__ == "__main__":
    main()
