#!/usr/bin/env python3
"""Build the Goal182 core composition matrices and their human index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE = REPO_ROOT / "configs" / "governance"
AXIS_PATH = GOVERNANCE / "goal182_axis_registry.json"
MATRIX_SCHEMA = "auv_core_composition_matrix.v1"
INDEX_SCHEMA = "auv_core_composition_index.v1"
TRAINING_PROFILE = "PPO-FLAT-4096x48-E5-MB4"

ACTOR_BACKENDS = ("MLP", "TA")
MODULES = (
    ("MOD0", "MS_NONE"),
    ("MOD1", "MS_TEMP"),
    ("MOD2", "MS_TEMP_PRIV"),
    ("MOD3", "MS_TEMP_CURRENT"),
    ("MOD4", "MS_FULL"),
)
LEARNING = {
    "PE2E": ("BCFT", "BCRES"),
    "ALOSB": ("BCFT", "BCRES", "TRRES_CMD", "TRRES_ACT"),
}
PROGRAMS = {
    "PE2E": {
        "matrix_id": "G182-CORE-PE2E",
        "program_group_id": "G-PATH-CONDITIONED-POLICY",
        "control_contract": "CC_PE2E",
        "path": "configs/governance/goal182_core_pe2e_matrix.json",
    },
    "ALOSB": {
        "matrix_id": "G182-CORE-ALOSB",
        "program_group_id": "G-ALOS-BODY-RL",
        "control_contract": "CC_ALOS_BODY",
        "path": "configs/governance/goal182_core_alosb_matrix.json",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actor_metadata(actor: str) -> dict[str, str]:
    if actor == "MLP":
        return {
            "actor_kind": "mlp",
            "actor_input_contract": "fused",
            "token_contract": "none",
            "pooling": "none",
        }
    return {
        "actor_kind": "transformer",
        "actor_input_contract": "token_attention",
        "token_contract": "pc_history_dual_horizon_v1",
        "pooling": "attention",
    }


def _first_wave_order(actor: str, module: str) -> int | None:
    if actor != "MLP":
        return None
    order = {"MOD4": 1, "MOD3": 2, "MOD2": 3, "MOD1": 4, "MOD0": 5}
    return order[module]


def _arm(actor: str, control: str, learn: str, module: str, module_set: str) -> dict[str, Any]:
    residual_interface = None
    if learn == "TRRES_CMD":
        residual_interface = "RI_CMD"
    elif learn in {"BCRES", "TRRES_ACT"}:
        residual_interface = "RI_ACT"
    return {
        "arm_id": f"G182_INT_STACK_{actor}_{control}_{learn}_{module}",
        "actor_backend": f"AB_{actor}_M",
        "actor": _actor_metadata(actor),
        "control_contract": PROGRAMS[control]["control_contract"],
        "program_group_id": PROGRAMS[control]["program_group_id"],
        "learning_composition": learn,
        "residual_interface": residual_interface,
        "module_set": module_set,
        "module_short_code": module,
        "first_wave_order": _first_wave_order(actor, module),
        "status": "DRAFT",
    }


def _matrix(control: str) -> dict[str, Any]:
    arms = [
        _arm(actor, control, learn, module, module_set)
        for actor in ACTOR_BACKENDS
        for learn in LEARNING[control]
        for module, module_set in MODULES
    ]
    info = PROGRAMS[control]
    return {
        "schema_version": MATRIX_SCHEMA,
        "goal_id": "Goal182",
        "matrix_family_id": "G182-CORE-COMP",
        "matrix_id": info["matrix_id"],
        "program_group_id": info["program_group_id"],
        "control_contract": info["control_contract"],
        "matrix_status": "DRAFT",
        "axis_registry": "configs/governance/goal182_axis_registry.json",
        "axis_registry_sha256": _sha256(AXIS_PATH),
        "training_profile": TRAINING_PROFILE,
        "cross_program_pooling": False,
        "arm_count": len(arms),
        "arms": arms,
    }


def main() -> None:
    pe2e = _matrix("PE2E")
    alosb = _matrix("ALOSB")
    for control, payload in (("PE2E", pe2e), ("ALOSB", alosb)):
        (REPO_ROOT / PROGRAMS[control]["path"]).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    index = {
        "schema_version": INDEX_SCHEMA,
        "goal_id": "Goal182",
        "human_id": "G182-CORE-COMP",
        "status": "DRAFT",
        "axis_registry": "configs/governance/goal182_axis_registry.json",
        "axis_registry_sha256": _sha256(AXIS_PATH),
        "training_profile": TRAINING_PROFILE,
        "cross_program_pooling": False,
        "program_matrices": [
            {
                "matrix_id": PROGRAMS[control]["matrix_id"],
                "program_group_id": PROGRAMS[control]["program_group_id"],
                "control_contract": PROGRAMS[control]["control_contract"],
                "path": PROGRAMS[control]["path"],
                "arm_count": payload["arm_count"],
                "matrix_status": payload["matrix_status"],
            }
            for control, payload in (("PE2E", pe2e), ("ALOSB", alosb))
        ],
        "evaluation_matrix": "configs/governance/goal182_eval_matrix.json",
    }
    (GOVERNANCE / "goal182_core_composition_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
