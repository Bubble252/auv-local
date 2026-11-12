from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.config import load_config
from auv_lowlevel.observation_schema import observation_schema_for_dim


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a 26D I0 checkpoint and gate artifacts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--strict-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _scan_rollouts(directory: Path, *, observation_dim: int) -> dict[str, Any]:
    paths = sorted(directory.glob("seed_*.npz"))
    if not paths:
        raise ValueError(f"no rollout files found in {directory}")
    array_count = 0
    sample_count = 0
    max_abs_value = 0.0
    for path in paths:
        with np.load(path) as payload:
            if "observation" not in payload.files:
                raise ValueError(f"{path} has no observation array")
            observation = np.asarray(payload["observation"])
            if observation.ndim != 2 or observation.shape[1] != observation_dim:
                raise ValueError(
                    f"{path} observation has shape {observation.shape}, "
                    f"expected (N, {observation_dim})"
                )
            sample_count += int(observation.shape[0])
            for name in payload.files:
                array = np.asarray(payload[name])
                array_count += 1
                if np.issubdtype(array.dtype, np.floating):
                    if not np.all(np.isfinite(array)):
                        raise ValueError(f"{path}:{name} contains NaN or Inf")
                    max_abs_value = max(max_abs_value, float(np.max(np.abs(array))))
    return {
        "episode_count": len(paths),
        "array_count": array_count,
        "sample_count": sample_count,
        "max_abs_value": max_abs_value,
        "all_finite": True,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    schema = observation_schema_for_dim(config.observation.dim)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    errors: list[str] = []

    required_checkpoint_fields = {
        "schema_version",
        "observation_dim",
        "observation_schema",
        "observation_mean",
        "observation_std",
        "action_scales",
        "training_config_sha256",
        "metadata",
    }
    missing = sorted(required_checkpoint_fields - set(checkpoint))
    if missing:
        errors.append(f"checkpoint is missing fields: {missing}")
    if checkpoint.get("schema_version") != "auv_odarrl_bc.v1":
        errors.append("unsupported BC checkpoint schema")
    if checkpoint.get("observation_dim") != schema.dim:
        errors.append("checkpoint observation_dim does not match config")
    if checkpoint.get("observation_schema") != schema.version:
        errors.append("checkpoint observation_schema does not match config")
    for field in ("observation_mean", "observation_std"):
        shape = tuple(np.asarray(checkpoint.get(field)).shape)
        if shape != (schema.dim,):
            errors.append(f"checkpoint {field} has shape {shape}")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("checkpoint metadata must be a mapping")
        metadata = {}
    for field in ("demos_sha256", "initialization"):
        if not metadata.get(field):
            errors.append(f"checkpoint metadata is missing {field}")
    if not metadata.get("teacher_input_sha256"):
        errors.append("checkpoint metadata is missing teacher_input_sha256")

    gate = _load_json(args.gate_summary)
    if gate.get("schema_version") != "auv_odarrl_i0_gate.v1":
        errors.append("unsupported I0 gate summary schema")
    if gate.get("config_sha256") != config.config_sha256:
        errors.append("gate config_sha256 does not match --config")
    if gate.get("observation_dim") != schema.dim:
        errors.append("gate observation_dim does not match config")
    if gate.get("observation_schema") != schema.version:
        errors.append("gate observation_schema does not match config")
    seeds = tuple(int(seed) for seed in gate.get("seeds", ()))
    if len(seeds) != 50 or len(set(seeds)) != 50:
        errors.append("gate must contain 50 unique seeds")
    if not bool(gate.get("gate", {}).get("passed")):
        errors.append("I0 gate did not pass")
    if gate.get("b0_comparison") is None:
        errors.append("gate summary has no B0 comparison")

    rollout_root = args.gate_summary.resolve().parent / "per_seed"
    teacher_rollouts = _scan_rollouts(rollout_root / "teacher", observation_dim=schema.dim)
    student_rollouts = _scan_rollouts(rollout_root / "student", observation_dim=schema.dim)
    for role, scan in (("teacher", teacher_rollouts), ("student", student_rollouts)):
        if scan["episode_count"] != len(seeds):
            errors.append(f"{role} rollout count does not match seed count")

    strict_summary = None
    if args.strict_summary is not None:
        strict = _load_json(args.strict_summary)
        if tuple(int(seed) for seed in strict.get("seeds", ())) != seeds:
            errors.append("strict summary seeds do not match the main gate")
        strict_summary = {
            "summary_path": str(args.strict_summary.resolve()),
            "goal_radius_override_m": strict.get("goal_radius_override_m"),
            "teacher_success_rate": strict["aggregate"]["teacher"]["success_rate"],
            "student_success_rate": strict["aggregate"]["student"]["success_rate"],
        }

    result = {
        "schema_version": "auv_odarrl_i0_audit.v1",
        "passed": not errors,
        "errors": errors,
        "config_path": str(args.config.resolve()),
        "config_sha256": config.config_sha256,
        "observation_dim": schema.dim,
        "observation_schema": schema.version,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_training_config_sha256": checkpoint.get("training_config_sha256"),
        "checkpoint_data_sha256": metadata.get("demos_sha256"),
        "checkpoint_initialization": metadata.get("initialization"),
        "checkpoint_git": metadata.get("git"),
        "checkpoint_teacher_input_sha256": metadata.get("teacher_input_sha256"),
        "seeds": list(seeds),
        "gate": gate.get("gate"),
        "b0_comparison": gate.get("b0_comparison"),
        "strict": strict_summary,
        "teacher_rollouts": teacher_rollouts,
        "student_rollouts": student_rollouts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
