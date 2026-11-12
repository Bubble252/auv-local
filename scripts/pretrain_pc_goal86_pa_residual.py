#!/usr/bin/env python3
"""Fit a Goal86 P-A residual actor to true D3-minus-ALOS actuator labels.

The resulting checkpoint uses the existing PC BC checkpoint schema so PPO
``load_pc_bc_actor_prior`` and SAC ``initialize_sac_head_from_bc`` can consume
it.  The checkpoint is deliberately command-free (``actor_command_dim=0``) and
path8-compatible; Goal86 P-A keeps its command3/endpoint4 inputs in the RL
policy, with command and endpoint columns initialized by the existing loaders.

This script does not train PPO/SAC and does not open held-out seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.pc_bc import (
    PC_ACTION_ENVELOPE,
    PC_SCHEMA_VERSION,
    PCBCTrainingConfig,
    PCDemonstrationDataset,
    pc_action_full_scale,
    read_pc_bc_checkpoint_metadata,
    train_pc_bc,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--d3-checkpoint", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-scale", type=float, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=86051)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zero_actor_head(source: Path, target: Path) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("D3 checkpoint has no state_dict")
    indices = [
        int(name.split(".")[1])
        for name in state
        if name.startswith("actor.") and name.endswith(".weight")
    ]
    if not indices:
        raise ValueError("D3 checkpoint has no actor linear layers")
    final_index = max(indices)
    weight_name = f"actor.{final_index}.weight"
    bias_name = f"actor.{final_index}.bias"
    if weight_name not in state or bias_name not in state:
        raise ValueError("D3 checkpoint final actor layer is incomplete")
    state[weight_name] = torch.zeros_like(state[weight_name])
    state[bias_name] = torch.zeros_like(state[bias_name])
    checkpoint["state_dict"] = state
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return {
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "zero_head_checkpoint": str(target.resolve()),
        "zero_head_sha256": _sha256(target),
        "final_actor_layer": final_index,
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "abs_mean": absolute.mean(axis=0).tolist(),
        "abs_p95": np.percentile(absolute, 95, axis=0).tolist(),
        "abs_max": absolute.max(axis=0).tolist(),
        "clip_fraction": (absolute > 1.0 + 1e-6).mean(axis=0).tolist(),
    }


def main() -> int:
    args = _parse_args()
    labels_path = _resolve(args.labels)
    d3_path = _resolve(args.d3_checkpoint)
    env_path = _resolve(args.environment_config)
    output = _resolve(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    if not np.isfinite(args.residual_scale) or args.residual_scale <= 0.0:
        raise ValueError("--residual-scale must be finite and positive")
    output.mkdir(parents=True, exist_ok=True)

    environment = load_config(env_path)
    normalization = PCNormalization(
        position_scale_m=environment.normalization.position_scale_m,
        attitude_scale_rad=environment.normalization.attitude_scale_rad,
        angular_rate_scale_rad_s=environment.normalization.angular_rate_scale_rad_s,
        surge_speed_scale_mps=environment.normalization.surge_speed_scale_mps,
        sway_heave_speed_scale_mps=environment.normalization.sway_heave_speed_scale_mps,
        current_scale_mps=environment.normalization.current_scale_mps,
    )
    with np.load(labels_path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    required = {
        "actor_state",
        "actor_path",
        "base_actuator",
        "d3_actuator",
        "residual_physical",
        "residual_normalized",
        "episode_id",
        "seed",
        "step",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"label npz is missing fields: {sorted(missing)}")
    count = int(arrays["actor_state"].shape[0])
    if args.max_samples is not None:
        count = min(count, int(args.max_samples))
    if count < 2:
        raise ValueError("at least two samples are required")

    scale = float(args.residual_scale)
    full_scale = torch.tensor(pc_action_full_scale(), dtype=torch.float64)
    physical = torch.as_tensor(arrays["residual_physical"][:count], dtype=torch.float64)
    expected_normalized = physical / (scale * full_scale)
    actual_normalized = torch.as_tensor(
        arrays["residual_normalized"][:count], dtype=torch.float64
    )
    if not torch.allclose(expected_normalized, actual_normalized, atol=1e-8, rtol=1e-7):
        raise ValueError(
            "--residual-scale does not match the normalized residual labels; "
            "regenerate the label audit at this scale"
        )
    target = torch.clamp(actual_normalized, -1.0, 1.0).to(dtype=torch.float32).numpy()
    dataset = PCDemonstrationDataset(
        actor_state=np.asarray(arrays["actor_state"][:count], dtype=np.float32),
        command=np.zeros((count, 0), dtype=np.float32),
        action=target,
        episode_id=np.asarray(arrays["episode_id"][:count], dtype=np.int64),
        seed=np.asarray(arrays["seed"][:count], dtype=np.int64),
        step=np.asarray(arrays["step"][:count], dtype=np.int64),
        actual_actuator=np.asarray(
            arrays["d3_actuator"][:count], dtype=np.float32
        ),
        reward=np.zeros((count,), dtype=np.float32),
        observation_schema=PC_SCHEMA_VERSION,
        action_envelope=PC_ACTION_ENVELOPE,
        actor_path=np.asarray(arrays["actor_path"][:count, :8], dtype=np.float32),
    )
    d3_metadata = read_pc_bc_checkpoint_metadata(d3_path)
    if int(d3_metadata["actor_command_dim"]) != 0:
        raise ValueError("D3 checkpoint must be command-free")
    if int(d3_metadata["path_dim"]) != 8:
        raise ValueError("D3 checkpoint must use path_dim=8")
    zero_head = _zero_actor_head(d3_path, output / "d3_zero_head_for_init.pt")
    config = PCBCTrainingConfig(
        state_encoder_hidden_dims=(128, 128),
        actor_hidden_dims=(256, 128),
        d_z_total=64,
        n_in_reserved=8,
        n_z_reserved=4,
        activation="elu",
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        learning_rate=float(args.learning_rate),
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
        device=str(args.device),
        actor_command_dim=0,
        path_dim=8,
        path_encoder_hidden_dims=(64,),
        path_encoder_alpha=1.0,
        actor_current_encoder_enabled=False,
        saturation_weight=0.0,
    )
    label_stats = _stats(actual_normalized.numpy())
    metadata = {
        "dataset_kind": "goal86_pa_d3_minus_alos_actuator_residual",
        "labels": str(labels_path.resolve()),
        "labels_sha256": _sha256(labels_path),
        "residual_scale": scale,
        "residual_denominator": (scale * full_scale).tolist(),
        "label_stats": label_stats,
        "d3_zero_head": zero_head,
        "source_d3_sha256": d3_metadata["checkpoint_sha256"],
        "command_policy": "command-free BC prior; P-A RL keeps command3 input with zero new columns",
        "path_policy": "path8 BC prior; P-A RL keeps endpoint4 zero-initialized",
    }
    result = train_pc_bc(
        dataset,
        config,
        output_dir=output,
        metadata=metadata,
        normalization=normalization,
        init_checkpoint=output / "d3_zero_head_for_init.pt",
    )

    checkpoint_path = output / "pc_bc_actor.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["policy_kind"] = "pc_goal86_pa_residual_bc"
    checkpoint["residual_semantics"] = {
        "target": "normalize(D3_actuator - ALOS_actuator)",
        "residual_scale": scale,
        "labels_sha256": metadata["labels_sha256"],
        "source_d3_sha256": d3_metadata["checkpoint_sha256"],
        "command_free_bc": True,
        "path8_bc": True,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = _sha256(checkpoint_path)
    result["checkpoint_path"] = str(checkpoint_path.resolve())
    result["checkpoint_sha256"] = checkpoint_sha
    result["residual_semantics"] = checkpoint["residual_semantics"]
    result["label_stats"] = label_stats
    result["d3_zero_head"] = zero_head
    (output / "training_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "auv_goal86_pa_residual_bc_manifest.v1",
                **checkpoint,
                "state_dict": "<stored in pc_bc_actor.pt>",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "training_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
