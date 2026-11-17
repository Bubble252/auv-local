#!/usr/bin/env python3
"""Validate Goal182 PE2E/ALOSB composition matrices and actor metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE = REPO_ROOT / "configs" / "governance"
AXIS_PATH = GOVERNANCE / "goal182_axis_registry.json"
INDEX_PATH = GOVERNANCE / "goal182_core_composition_index.json"
FREEZE_MANIFEST = GOVERNANCE / "goal182_matrix_freeze.json"
MATRIX_SCHEMA = "auv_core_composition_matrix.v1"
TRAINING_PROFILE = "PPO-FLAT-4096x48-E5-MB4"
FORMAL_SEEDS = (86201, 86202, 86203)
ARM_RE = re.compile(r"^G182_INT_STACK_(MLP|TA)_(PE2E|ALOSB)_(BCFT|BCRES|TRRES_CMD|TRRES_ACT)_(MOD[0-4])$")
MODULE_SETS = {
    "MOD0": "MS_NONE",
    "MOD1": "MS_TEMP",
    "MOD2": "MS_TEMP_PRIV",
    "MOD3": "MS_TEMP_CURRENT",
    "MOD4": "MS_FULL",
}
ACTOR_CONTRACTS = {
    "MLP": {
        "actor_kind": "mlp",
        "actor_input_contract": "fused",
        "pooling": "none",
        "token_contract": "none",
    },
    "TA": {
        "actor_kind": "transformer",
        "actor_input_contract": "token_attention",
        "pooling": "attention",
    },
}
MODULE_CONFIG_EXPECTATIONS = {
    "MOD4": {
        "actor_current_encoder_enabled": True,
        "critic_path_dim": 30,
        "dade_enabled": True,
    },
    "MOD3": {
        "actor_current_encoder_enabled": True,
        "critic_path_dim": 0,
        "dade_enabled": True,
    },
    "MOD2": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 30,
        "dade_enabled": True,
    },
    "MOD1": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 0,
        "dade_enabled": True,
    },
    "MOD0": {
        "actor_current_encoder_enabled": False,
        "critic_path_dim": 0,
        "dade_enabled": False,
    },
}
MOD4_BC_SHA = {
    "PE2E": "d85718a432a287f2ae255528f645bc20f4445aba875c9fb9c44f008c578094b2",
    "ALOSB": "47cda13ed97e77d372d1495096dd72259987c5466c13a5d2023f45000b402749",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--freeze-manifest", type=Path, default=FREEZE_MANIFEST)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--require-first-wave-configs", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_wave_config_paths(control: str, learn: str) -> list[str]:
    stem = {
        ("PE2E", "BCFT"): "train_formal_mod4_pe2e_finetune_5d_m_seed{seed}.json",
        ("PE2E", "BCRES"): "train_formal_mod4_pe2e_residual_5d_m_seed{seed}.json",
        ("ALOSB", "BCFT"): "train_formal_mod4_alosb_finetune_5d_m_seed{seed}.json",
        ("ALOSB", "BCRES"): "train_formal_mod4_alosb_residual_5d_m_seed{seed}.json",
        ("ALOSB", "TRRES_CMD"): (
            "train_formal_mod4_alosb_trres_cmd_5d_m_seed{seed}.json"
        ),
        ("ALOSB", "TRRES_ACT"): (
            "train_formal_mod4_alosb_trres_act_5d_m_seed{seed}.json"
        ),
    }[(control, learn)]
    return [
        f"configs/lowlevel/goal182/{stem.format(seed=seed)}"
        for seed in FORMAL_SEEDS
    ]


def _validate_formal_config(
    arm: dict[str, Any],
    control: str,
    module: str,
    relative: str,
) -> list[str]:
    errors: list[str] = []
    label = f"{arm['arm_id']}.{relative}"
    path = REPO_ROOT / relative
    try:
        payload = _load(path)
    except (OSError, ValueError) as error:
        return [f"{label}: cannot read formal config: {error}; CONFIG_COVERAGE_MISSING"]
    pc = payload.get("pc")
    if not isinstance(pc, dict):
        return [f"{label}: missing pc block; MODULE_SET_CONFIG_MISMATCH"]
    expected = MODULE_CONFIG_EXPECTATIONS[module]
    for key, value in expected.items():
        if pc.get(key) != value:
            errors.append(
                f"{label}: pc.{key} must be {value!r}, got {pc.get(key)!r}; "
                "MODULE_SET_CONFIG_MISMATCH"
            )
    expected_command_dim = 0 if control == "PE2E" else 3
    if pc.get("actor_command_dim") != expected_command_dim:
        errors.append(
            f"{label}: actor_command_dim must be {expected_command_dim} for "
            f"{control}; ACTOR_INPUT_CONTRACT_MISMATCH"
        )
    if pc.get("path_dim") != 5 or pc.get("bc_path_dim") != 5:
        errors.append(f"{label}: 5D path contract required; PATH_CONTRACT_MISMATCH")
    learn = str(arm["learning_composition"])
    if learn in {"BCFT", "BCRES"}:
        if module == "MOD4":
            checkpoint_key = (
                "warmstart_actor" if learn == "BCFT" else "base_policy"
            )
            if learn == "BCFT":
                recorded_sha = payload.get("experiment", {}).get(
                    "warmstart_actor_sha256"
                )
                checkpoint = payload.get("experiment", {}).get("warmstart_actor")
            else:
                base = payload.get("base_policy", {})
                recorded_sha = base.get("checkpoint_sha256")
                checkpoint = base.get("checkpoint")
            if recorded_sha != MOD4_BC_SHA[control]:
                errors.append(
                    f"{label}: {checkpoint_key} SHA is not the compatible MOD4 "
                    f"{control} current-encoder BC; BC_CHECKPOINT_INCOMPATIBLE"
                )
            if not isinstance(checkpoint, str) or not checkpoint:
                errors.append(
                    f"{label}: missing {checkpoint_key} checkpoint; "
                    "BC_CHECKPOINT_INCOMPATIBLE"
                )
    environment = payload.get("environment")
    if isinstance(environment, dict):
        env_path = environment.get("config")
        if isinstance(env_path, str):
            try:
                env_payload = _load(REPO_ROOT / env_path)
                steps = env_payload.get("plant", {}).get("max_episode_steps")
                speed_protocol = (
                    env_payload.get("curriculum", {}).get("speed_protocol")
                )
            except (OSError, ValueError):
                steps = None
                speed_protocol = None
            if steps != 3000:
                errors.append(
                    f"{label}: formal env must use max_episode_steps=3000; "
                    "EPISODE_PROTOCOL_MISMATCH"
                )
            if speed_protocol != "SPEED_V3_CENTER_OUT_5S":
                errors.append(
                    f"{label}: formal env must use SPEED_V3_CENTER_OUT_5S; "
                    "SPEED_CURRICULUM_PROTOCOL_INVALID"
                )
    if payload.get("training_profile") != TRAINING_PROFILE:
        errors.append(f"{label}: training_profile mismatch")
    return errors


def _validate_arm(arm: dict[str, Any], matrix: dict[str, Any], errors: list[str]) -> None:
    label = f"{matrix['matrix_id']}.{arm.get('arm_id')}"
    match = ARM_RE.match(str(arm.get("arm_id", "")))
    if match is None:
        errors.append(f"{label}: arm_id does not match canonical grammar")
        return
    actor, control, learn, module = match.groups()
    if control != ("PE2E" if matrix["control_contract"] == "CC_PE2E" else "ALOSB"):
        errors.append(f"{label}: arm control disagrees with matrix control_contract")
    if arm.get("actor_backend") != f"AB_{actor}_M":
        errors.append(f"{label}: actor_backend mismatch")
    if arm.get("control_contract") != matrix["control_contract"]:
        errors.append(f"{label}: control_contract mismatch")
    if arm.get("program_group_id") != matrix["program_group_id"]:
        errors.append(f"{label}: program_group_id mismatch")
    if arm.get("learning_composition") != learn:
        errors.append(f"{label}: learning_composition mismatch")
    if arm.get("module_set") != MODULE_SETS[module] or arm.get("module_short_code") != module:
        errors.append(f"{label}: module-set mismatch")
    if control == "PE2E" and learn not in {"BCFT", "BCRES"}:
        errors.append(f"{label}: PE2E only allows BCFT/BCRES")
    if control == "ALOSB" and learn not in {"BCFT", "BCRES", "TRRES_CMD", "TRRES_ACT"}:
        errors.append(f"{label}: ALOSB learning composition is invalid")
    actor_metadata = arm.get("actor")
    if not isinstance(actor_metadata, dict):
        errors.append(f"{label}: missing actor metadata; ACTOR_METADATA_MISSING")
    else:
        expected = ACTOR_CONTRACTS[actor]
        for key, value in expected.items():
            if actor_metadata.get(key) != value:
                errors.append(f"{label}: actor.{key} must be {value!r}; ACTOR_METADATA_CONTRACT_MISMATCH")
        if actor == "TA" and actor_metadata.get("token_contract") in {None, "", "none"}:
            errors.append(f"{label}: TA requires a token contract; TOKENIZATION_CONFOUND")
    expected_order = {"MOD4": 1, "MOD3": 2, "MOD2": 3, "MOD1": 4, "MOD0": 5}[module]
    if actor == "MLP" and arm.get("first_wave_order") != expected_order:
        errors.append(f"{label}: first_wave_order mismatch")
    if actor == "TA" and arm.get("first_wave_order") is not None:
        errors.append(f"{label}: Transformer must not enter the first wave")


def validate_matrix(matrix: dict[str, Any], axis_sha: str, errors: list[str]) -> set[str]:
    required = {
        "schema_version",
        "goal_id",
        "matrix_family_id",
        "matrix_id",
        "program_group_id",
        "control_contract",
        "matrix_status",
        "axis_registry",
        "axis_registry_sha256",
        "training_profile",
        "cross_program_pooling",
        "arm_count",
        "arms",
    }
    missing = sorted(required - set(matrix))
    if missing:
        errors.append(f"{matrix.get('matrix_id', '<unknown>')}: missing {missing}")
        return set()
    if matrix["schema_version"] != MATRIX_SCHEMA:
        errors.append(f"{matrix['matrix_id']}: wrong schema_version")
    if matrix["goal_id"] != "Goal182" or matrix["matrix_family_id"] != "G182-CORE-COMP":
        errors.append(f"{matrix['matrix_id']}: goal/family mismatch")
    if matrix["training_profile"] != TRAINING_PROFILE:
        errors.append(f"{matrix['matrix_id']}: wrong training_profile")
    if matrix["cross_program_pooling"] is not False:
        errors.append(f"{matrix['matrix_id']}: cross_program_pooling must be false")
    if matrix["axis_registry_sha256"] != axis_sha:
        errors.append(f"{matrix['matrix_id']}: axis registry SHA is stale")
    arms = matrix["arms"]
    if not isinstance(arms, list) or matrix["arm_count"] != len(arms):
        errors.append(f"{matrix['matrix_id']}: arm_count mismatch")
        return set()
    ids = [str(arm.get("arm_id")) for arm in arms if isinstance(arm, dict)]
    if len(ids) != len(set(ids)):
        errors.append(f"{matrix['matrix_id']}: duplicate arm_id")
    for arm in arms:
        if not isinstance(arm, dict):
            errors.append(f"{matrix['matrix_id']}: arm must be an object")
            continue
        _validate_arm(arm, matrix, errors)
    return set(ids)


def main() -> int:
    args = parse_args()
    if args.schema_only:
        print("PASS goal182 core matrix checker schema")
        return 0
    errors: list[str] = []
    axis_sha = _sha256(AXIS_PATH)
    manifest_entries: dict[str, dict[str, Any]] = {}
    if args.require_frozen:
        if not args.freeze_manifest.is_file():
            errors.append(f"freeze manifest not found: {args.freeze_manifest}")
        else:
            manifest = _load(args.freeze_manifest)
            if manifest.get("schema_version") != "auv_goal182_matrix_freeze.v1":
                errors.append("freeze manifest: wrong schema_version")
            if manifest.get("goal_id") != "Goal182":
                errors.append("freeze manifest: wrong goal_id")
            if manifest.get("training_profile") != TRAINING_PROFILE:
                errors.append("freeze manifest: wrong training_profile")
            manifest_entries = {
                str(entry.get("path")): entry
                for entry in manifest.get("matrices", [])
                if isinstance(entry, dict)
            }
    index = _load(args.index)
    if index.get("schema_version") != "auv_core_composition_index.v1":
        errors.append("index: wrong schema_version")
    if index.get("human_id") != "G182-CORE-COMP" or index.get("goal_id") != "Goal182":
        errors.append("index: goal/human id mismatch")
    if index.get("axis_registry_sha256") != axis_sha:
        errors.append("index: axis registry SHA is stale")
    if index.get("cross_program_pooling") is not False:
        errors.append("index: cross_program_pooling must be false")
    all_ids: set[str] = set()
    total = 0
    for ref in index.get("program_matrices", []):
        path = REPO_ROOT / ref["path"]
        if not path.is_file():
            errors.append(f"index: missing matrix file {ref['path']}")
            continue
        matrix = _load(path)
        relative = str(path.relative_to(REPO_ROOT))
        if args.require_frozen:
            if matrix.get("matrix_status") != "FROZEN":
                errors.append(f"{relative}: matrix_status must be FROZEN")
            entry = manifest_entries.get(relative)
            if entry is None:
                errors.append(f"freeze manifest: missing {relative}")
            elif entry.get("sha256") != _sha256(path):
                errors.append(f"freeze manifest: SHA mismatch for {relative}")
        for key in ("matrix_id", "program_group_id", "control_contract", "arm_count"):
            if ref.get(key) != matrix.get(key):
                errors.append(f"index: {key} mismatch for {ref['path']}")
        ids = validate_matrix(matrix, axis_sha, errors)
        if args.require_first_wave_configs:
            control = "PE2E" if matrix["control_contract"] == "CC_PE2E" else "ALOSB"
            for arm in matrix["arms"]:
                if arm.get("first_wave_order") != 1:
                    continue
                for relative in _first_wave_config_paths(
                    control, str(arm["learning_composition"])
                ):
                    if not (REPO_ROOT / relative).is_file():
                        errors.append(
                            f"{arm['arm_id']}: missing formal config {relative}; "
                            "CONFIG_COVERAGE_MISSING"
                        )
                        continue
                    match = ARM_RE.match(str(arm.get("arm_id", "")))
                    module = match.group(4) if match is not None else "MOD4"
                    errors.extend(
                        _validate_formal_config(
                            arm, control, module, relative
                        )
                    )
        overlap = all_ids & ids
        if overlap:
            errors.append(f"{matrix['matrix_id']}: arm duplicated across programs: {sorted(overlap)}")
        all_ids |= ids
        total += len(ids)
    if total != 60:
        errors.append(f"index: expected 60 total core arms, got {total}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"PASS goal182 matrices arms={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
