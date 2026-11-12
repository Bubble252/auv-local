#!/usr/bin/env python3
"""Audit Goal86 BC checkpoints and command-teacher availability (read-only).

The audit answers three questions before Goal86 spends any training budget:

1. Which BC checkpoints exist, and what action space do they actually encode?
2. Can each checkpoint initialise a PPO actor trunk without silent re-wiring?
3. Is there any valid command teacher for the ``P-C`` / ``U2`` arms?

It never creates an environment, never trains, and never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from auv_lowlevel.batch.pc_bc import (
    PCBCPolicy,
    load_pc_bc_actor_prior,
    read_pc_bc_checkpoint_metadata,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_CRITIC_STATE_DIM,
    PCActorCritic,
    PCPolicyLayout,
)

SCHEMA_VERSION = "auv_pc_goal86_bc_interface_audit.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_GLOBS = (
    "checkpoints/**/*.pt",
    "artifacts/**/pc_bc_actor.pt",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="BC checkpoint to audit; repeatable. Defaults to repository globs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-warmstart-probe",
        action="store_true",
        help="skip constructing a PPO actor and loading the BC weights",
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def _discover_checkpoints(explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        return sorted({_resolve(path) for path in explicit})
    found: set[Path] = set()
    for pattern in DEFAULT_CHECKPOINT_GLOBS:
        found.update(Path(match) for match in glob.glob(str(REPO_ROOT / pattern), recursive=True))
    return sorted(path for path in found if path.is_file())


def _normalization_from_metadata(metadata: dict[str, Any]) -> PCNormalization:
    payload = metadata["normalization"]
    return PCNormalization(
        position_scale_m=float(payload["position_scale_m"]),
        attitude_scale_rad=float(payload["attitude_scale_rad"]),
        angular_rate_scale_rad_s=float(payload["angular_rate_scale_rad_s"]),
        surge_speed_scale_mps=float(payload["surge_speed_scale_mps"]),
        sway_heave_speed_scale_mps=float(payload["sway_heave_speed_scale_mps"]),
        current_scale_mps=float(payload["current_scale_mps"]),
    )


def _warmstart_probe(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a PPO actor with the BC layout and load the BC weights."""

    try:
        command_dim = int(metadata["actor_command_dim"])
        path_dim = int(metadata["path_dim"])
        current_enabled = str(metadata.get("actor_current_encoder", "none")) != "none"
        short_window = 10
        long_window = 50
        history_path_dim = 0
        actor_layout = PCPolicyLayout(
            state_dim=PC_ACTOR_STATE_DIM,
            path_dim=path_dim,
            current_dim=3 if current_enabled else 0,
            short_window=short_window,
            long_window=long_window,
            command_dim=command_dim,
            history_path_dim=history_path_dim,
        )
        critic_layout = PCPolicyLayout(
            state_dim=PC_CRITIC_STATE_DIM,
            short_window=short_window,
            long_window=long_window,
            command_dim=command_dim,
        )
        policy = PCActorCritic(
            actor_layout.flat_dim,
            critic_layout.flat_dim,
            int(metadata["actor_output_dim"]),
            actor_state_dim=PC_ACTOR_STATE_DIM,
            critic_state_dim=PC_CRITIC_STATE_DIM,
            history_short_window=short_window,
            history_long_window=long_window,
            d_z_total=int(metadata["d_z_total"]),
            n_in_reserved=int(metadata["n_in_reserved"]),
            n_z_reserved=int(metadata["n_z_reserved"]),
            state_encoder_hidden_dims=tuple(metadata["state_encoder_hidden_dims"]),
            actor_command_dim=command_dim,
            critic_command_dim=command_dim,
            path_dim=path_dim,
            path_encoder_hidden_dims=tuple(metadata["path_encoder_hidden_dims"] or (64,)),
            path_encoder_alpha=(
                float(metadata["path_encoder_alpha"]) if path_dim else 1.0
            ),
            actor_current_encoder_enabled=current_enabled,
            actor_current_encoder_hidden_dims=tuple(
                metadata.get("actor_current_encoder_hidden_dims") or (64,)
            ),
            actor_current_encoder_alpha=float(
                metadata.get("actor_current_encoder_alpha") or 1.0
            ),
            actor_hidden_dims=tuple(metadata["actor_hidden_dims"]),
            activation=str(metadata.get("activation", "elu")),
        )
        report = load_pc_bc_actor_prior(
            policy,
            path,
            expected_d_z_total=int(metadata["d_z_total"]),
            expected_n_in_reserved=int(metadata["n_in_reserved"]),
            expected_n_z_reserved=int(metadata["n_z_reserved"]),
            expected_actor_hidden_dims=tuple(metadata["actor_hidden_dims"]),
            expected_state_encoder_hidden_dims=tuple(
                metadata["state_encoder_hidden_dims"]
            ),
            expected_normalization=_normalization_from_metadata(metadata),
        )
        return {
            "probe": "load_pc_bc_actor_prior",
            "passed": True,
            "error": None,
            "command_projection": _as_plain(report.get("command_projection")),
            "actor_dade_output_zeroed": bool(
                report.get("actor_dade_output_zeroed", False)
            ),
            "path_encoder_zero_output": bool(
                report.get("path_encoder_zero_output", False)
            ),
        }
    except Exception as error:  # noqa: BLE001 - audit must record every failure
        return {
            "probe": "load_pc_bc_actor_prior",
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _audit_checkpoint(path: Path, *, probe: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
        "metadata_read_ok": False,
        "metadata_error": None,
    }
    if not path.is_file():
        record["metadata_error"] = "missing checkpoint"
        return record
    try:
        metadata = read_pc_bc_checkpoint_metadata(path)
        PCBCPolicy(path, device="cpu")
        record["metadata_read_ok"] = True
        record["policy_kind"] = metadata.get("policy_kind")
        record["observation_schema"] = metadata.get("observation_schema")
        record["action_envelope"] = metadata.get("action_envelope")
        record["action_head"] = metadata.get("action_head")
        record["action_decode_rule"] = metadata.get("action_decode_rule")
        record["action_full_scale"] = _as_plain(metadata.get("action_full_scale"))
        record["actor_state_names"] = _as_plain(metadata.get("actor_state_names"))
        record["command_names"] = _as_plain(metadata.get("command_names"))
        record["actor_command_dim"] = int(metadata.get("actor_command_dim", -1))
        record["path_dim"] = int(metadata.get("path_dim", -1))
        record["actor_output_dim"] = int(metadata.get("actor_output_dim", -1))
        record["z_t_dim"] = metadata.get("z_t_dim")
        record["actor_top_level_input_dim"] = metadata.get(
            "actor_top_level_input_dim"
        )
        record["actor_current_encoder"] = metadata.get("actor_current_encoder")
        record["dataset"] = _as_plain(metadata.get("dataset"))
        record["training_result"] = _as_plain(metadata.get("training_result"))
        command_dim = record["actor_command_dim"]
        output_dim = record["actor_output_dim"]
        continuous = record["action_head"] == "continuous"
        pc20 = record["action_envelope"] == "pc20"
        actuator_output = (
            record["policy_kind"] == "pc_bc_actor"
            and output_dim == 3
            and pc20
            and continuous
        )
        command_output = (
            record["policy_kind"] != "pc_bc_actor"
            and command_dim > 0
            and output_dim == command_dim
            and pc20
        )
        record["label_space"] = "actuator" if actuator_output else "unknown"
        record["capabilities"] = {
            "P-A_actuator_residual_init": actuator_output,
            "P-C_command_residual_init": command_output,
            "U2_path_command_residual_init": command_output,
            "requires_command_columns": command_dim > 0,
            "requires_current_encoder": record["actor_current_encoder"] != "none",
            "ppo_actor_trunk_contract": (
                "actor/encoder/path_encoder keys match PCActorCritic layout"
            ),
        }
        record["ppo_actor_warmstart_probe"] = (
            _warmstart_probe(path, metadata)
            if probe and actuator_output
            else {
                "probe": "load_pc_bc_actor_prior",
                "passed": False,
                "skipped": probe is False,
                "reason": "not an actuator-output BC checkpoint",
            }
        )
    except Exception as error:  # noqa: BLE001 - audit must record malformed files
        record["metadata_error"] = f"{type(error).__name__}: {error}"
    return record


def _teacher_audit(checkpoints: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Classify the teacher sources that exist in this repository."""

    actuator_teacher_sources = [
        {
            "path": "src/auv_lowlevel/batch/pc_teacher.py",
            "class": "BatchedPCTeacher",
            "docstring": "Privileged PD/PI teacher adapter for PC20 demonstration collection.",
            "label_space": "actuator",
            "evidence": (
                "control() forwards a BatchedRemus100Teacher output and returns "
                "physical [rudder, stern, rpm] actions"
            ),
        },
        {
            "path": "src/auv_lowlevel/env.py",
            "symbol": "teacher_command",
            "label_space": "actuator",
            "evidence": (
                "teacher.control() returns Remus100ActuatorCommand and is passed "
                "through _command_vector into the actuator base action"
            ),
        },
    ]
    demonstration_sources: dict[str, int] = {}
    demonstration_command_dims: set[int] = set()
    for checkpoint in checkpoints:
        dataset = checkpoint.get("dataset") or {}
        inputs = dataset.get("inputs") if isinstance(dataset, dict) else None
        if not isinstance(inputs, list):
            continue
        for item in inputs:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "unknown"))
            demonstration_sources[source] = demonstration_sources.get(source, 0) + 1
            if item.get("actor_command_dim") is not None:
                demonstration_command_dims.add(int(item["actor_command_dim"]))
    teacher_checkpoints = sorted(
        str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        for pattern in ("checkpoints/**/*teacher*.pt", "artifacts/**/*teacher*.pt")
        for path in (Path(match) for match in glob.glob(str(REPO_ROOT / pattern), recursive=True))
        if path.is_file()
    )
    command_teacher_available = False
    return {
        "actuator_teacher_sources": actuator_teacher_sources,
        "command_teacher_sources": [],
        "command_teacher_checkpoints": teacher_checkpoints,
        "demonstration_sources": demonstration_sources,
        "demonstration_actor_command_dims": sorted(demonstration_command_dims),
        "command_teacher_available": command_teacher_available,
        "command_teacher_status": (
            "BC_COMMAND_TEACHER_UNAVAILABLE"
            if not command_teacher_available
            else "AVAILABLE"
        ),
        "command_teacher_reason": (
            "The repository contains only privileged PD/PI actuator teachers. "
            "No code path or artifact produces deployable-observation "
            "[psi_d, theta_d, u_d] teacher labels with closed-loop evidence, "
            "and the D3 BC checkpoint has actor_command_dim=0."
        ),
        "P-C_allowed": command_teacher_available,
        "U2_allowed": command_teacher_available,
    }


def run_audit(
    *,
    checkpoints: Sequence[Path],
    probe: bool = True,
) -> dict[str, Any]:
    paths = _discover_checkpoints(checkpoints)
    records = [_audit_checkpoint(path, probe=probe) for path in paths]
    teacher = _teacher_audit(records)
    actuator_ready = [
        record
        for record in records
        if record.get("capabilities", {}).get("P-A_actuator_residual_init")
        and record.get("ppo_actor_warmstart_probe", {}).get("passed")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(REPO_ROOT),
        "checkpoint_count": len(records),
        "checkpoints": records,
        "actuator_bc_ready": len(actuator_ready) > 0,
        "actuator_bc_ready_paths": [record["path"] for record in actuator_ready],
        "command_bc_ready": False,
        "teacher_audit": teacher,
        "arm_decision": {
            "P-A": "READY" if actuator_ready else "BLOCKED_NO_VALID_ACTUATOR_BC",
            "P-C": "NOT_AVAILABLE" if not teacher["P-C_allowed"] else "READY",
            "U2": "NOT_AVAILABLE" if not teacher["U2_allowed"] else "READY",
            "T1": "READY_SUBJECT_TO_RL_BACKEND_AUDIT",
            "T2": "READY_SUBJECT_TO_RL_BACKEND_AUDIT",
            "U1": "READY_SUBJECT_TO_RL_BACKEND_AUDIT",
        },
        "passed": bool(records) and all(record["metadata_read_ok"] for record in records),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_audit(checkpoints=args.checkpoint, probe=not args.skip_warmstart_probe)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "checkpoint_count": payload["checkpoint_count"],
        "actuator_bc_ready": payload["actuator_bc_ready"],
        "P-A": payload["arm_decision"]["P-A"],
        "P-C": payload["arm_decision"]["P-C"],
        "U2": payload["arm_decision"]["U2"],
        "command_teacher_status": payload["teacher_audit"]["command_teacher_status"],
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
