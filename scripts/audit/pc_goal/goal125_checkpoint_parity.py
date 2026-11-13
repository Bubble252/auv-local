#!/usr/bin/env python3
"""Compare Goal125 initialization/checkpoint structure against a prior PC run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

ARCHITECTURE_METADATA_KEYS = (
    "actor_observation_dim",
    "critic_observation_dim",
    "history_short_window",
    "history_long_window",
    "history_short_stride",
    "history_long_stride",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-bias",
        default=None,
        help="expected candidate pc.base_residual_bias JSON, e.g. '[0,-0.25,0]'",
    )
    parser.add_argument(
        "--expect-zero-residual-output",
        action="store_true",
        help="require candidate actor.4 weight and bias to be all zero",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        return {}
    metadata = infos.get("auv_batch_checkpoint")
    return metadata if isinstance(metadata, dict) else {}


def _pc(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(checkpoint).get("pc")
    return metadata if isinstance(metadata, dict) else {}


def _noisy_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return sorted(
        key
        for key in state
        if any(token in key.lower() for token in ("rho", "sigma", "noisy", "noise"))
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    reference = _load(args.reference)
    candidate = _load(args.candidate)
    ref_state = reference.get("model_state_dict")
    cand_state = candidate.get("model_state_dict")
    if not isinstance(ref_state, dict) or not isinstance(cand_state, dict):
        raise ValueError("both checkpoints must contain model_state_dict")

    missing = sorted(set(ref_state) - set(cand_state))
    extra = sorted(set(cand_state) - set(ref_state))
    mismatches: list[dict[str, Any]] = []
    for name in sorted(set(ref_state) & set(cand_state)):
        left = ref_state[name]
        right = cand_state[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            if left != right:
                mismatches.append({"name": name, "reason": "non_tensor_mismatch"})
            continue
        if tuple(left.shape) != tuple(right.shape):
            mismatches.append(
                {
                    "name": name,
                    "reason": "shape_mismatch",
                    "reference_shape": list(left.shape),
                    "candidate_shape": list(right.shape),
                }
            )
            continue
        if not torch.equal(left, right):
            max_abs = float((left.float() - right.float()).abs().max().item())
            mismatches.append(
                {
                    "name": name,
                    "reason": "value_mismatch",
                    "max_abs_diff": max_abs,
                }
            )

    ref_pc = _pc(reference)
    cand_pc = _pc(candidate)
    ref_metadata = _metadata(reference)
    cand_metadata = _metadata(candidate)
    metadata_mismatches: list[dict[str, Any]] = []
    for key in ARCHITECTURE_METADATA_KEYS:
        if ref_metadata.get(key) != cand_metadata.get(key):
            metadata_mismatches.append(
                {
                    "key": key,
                    "reference": _jsonable(ref_metadata.get(key)),
                    "candidate": _jsonable(cand_metadata.get(key)),
                }
            )
    for key in (
        "actor_layout",
        "critic_layout",
        "actor_encoder",
        "path",
        "critic_encoder",
        "dade",
        "dade_enabled",
        "actor_command_dim",
        "critic_command_dim",
        "history_path_dim",
        "actor_current_encoder_enabled",
    ):
        if ref_pc.get(key) != cand_pc.get(key):
            metadata_mismatches.append(
                {
                    "key": f"pc.{key}",
                    "reference": _jsonable(ref_pc.get(key)),
                    "candidate": _jsonable(cand_pc.get(key)),
                }
            )

    expected_bias = None if args.expected_bias is None else json.loads(args.expected_bias)
    actual_bias = cand_pc.get("base_residual_bias")
    bias_match = True
    if expected_bias is not None:
        bias_match = actual_bias is not None and all(
            math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)
            for left, right in zip(actual_bias, expected_bias)
        ) and len(actual_bias) == len(expected_bias)

    zero_residual = True
    if args.expect_zero_residual_output:
        for name in ("actor.4.weight", "actor.4.bias"):
            value = cand_state.get(name)
            if not isinstance(value, torch.Tensor) or bool(value.abs().max().item() != 0.0):
                zero_residual = False

    ref_noisy = _noisy_keys(ref_state)
    cand_noisy = _noisy_keys(cand_state)
    report = {
        "schema_version": "auv_pc_goal125_checkpoint_parity.v1",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "state_keys": {
            "reference": len(ref_state),
            "candidate": len(cand_state),
            "missing_in_candidate": missing,
            "extra_in_candidate": extra,
        },
        "tensor_mismatches": mismatches,
        "metadata_mismatches": metadata_mismatches,
        "reference_noisy_keys": ref_noisy,
        "candidate_noisy_keys": cand_noisy,
        "candidate_base_residual_bias": actual_bias,
        "expected_base_residual_bias": expected_bias,
        "bias_match": bias_match,
        "zero_residual_output": zero_residual,
        "pass": not missing
        and not extra
        and not mismatches
        and not metadata_mismatches
        and not ref_noisy
        and not cand_noisy
        and bias_match
        and zero_residual,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
