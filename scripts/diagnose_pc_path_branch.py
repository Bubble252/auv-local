#!/usr/bin/env python3
"""Audit whether a PC checkpoint's path branch is materially active."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from auv_lowlevel.batch.pc_diagnostics import policy_diagnostics
from auv_lowlevel.batch.pc_policy import (
    PC_ACTOR_STATE_DIM,
    PC_CRITIC_STATE_DIM,
    PCActorCritic,
    PCPolicyLayout,
)
from auv_lowlevel.ppo_config import load_training_config


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _load(policy: PCActorCritic, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model_state_dict: {path}")
    policy.load_state_dict(state, strict=True)
    return state


def _norm(value: torch.Tensor) -> float | None:
    if not (value.is_floating_point() or value.is_complex()):
        return None
    return float(torch.linalg.vector_norm(value.detach()).item())


def _compare_states(
    state: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    common = sorted(set(state) & set(reference))
    norms: dict[str, float | None] = {}
    deltas: dict[str, float | None] = {}
    for name in common:
        norms[name] = _norm(state[name])
        deltas[name] = (
            _norm(state[name] - reference[name])
            if state[name].is_floating_point() or state[name].is_complex()
            else None
        )
    return {"common_parameter_count": len(common), "norms": norms, "deltas": deltas}


def _build_policy(training_config: Path) -> PCActorCritic:
    training = load_training_config(training_config)
    if training.pc is None or training.history is None:
        raise ValueError("diagnosis requires a PC training config")
    actor_layout = PCPolicyLayout(
        state_dim=PC_ACTOR_STATE_DIM,
        path_dim=training.pc.path_dim,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        command_dim=training.pc.actor_command_dim,
        history_path_dim=training.pc.history_path_dim,
    )
    critic_layout = PCPolicyLayout(
        state_dim=PC_CRITIC_STATE_DIM,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        command_dim=training.pc.critic_command_dim,
    )
    policy_config = training.rsl_train_cfg()["policy"]
    return PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        **{key: value for key, value in policy_config.items() if key != "class_name"},
    )


def diagnose(
    training_config: Path,
    checkpoints: list[Path],
    *,
    bc_checkpoint: Path | None,
) -> dict[str, Any]:
    policy = _build_policy(training_config)
    states: list[dict[str, torch.Tensor]] = []
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        state = _load(policy, checkpoint)
        states.append(state)
        records.append(
            {
                "checkpoint": str(checkpoint),
                "diagnostics": policy_diagnostics(policy),
            }
        )

    bc = None
    if bc_checkpoint is not None:
        bc_state = torch.load(bc_checkpoint, map_location="cpu", weights_only=False).get(
            "state_dict"
        )
        if not isinstance(bc_state, dict):
            raise ValueError("BC checkpoint has no state_dict")
        bc = {
            "checkpoint": str(bc_checkpoint),
            "comparison_to_final": _compare_states(states[-1], bc_state),
        }

    transitions = []
    for index, (left, right) in enumerate(zip(states, states[1:], strict=False), start=1):
        transitions.append(
            {
                "from": str(checkpoints[index - 1]),
                "to": str(checkpoints[index]),
                "comparison": _compare_states(right, left),
            }
        )
    return {
        "schema_version": "auv_pc_path_branch_diagnosis.v1",
        "training_config": str(training_config),
        "checkpoints": records,
        "transitions": transitions,
        "bc_reference": bc,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = diagnose(
        _resolve(args.training_config),
        [_resolve(path) for path in args.checkpoint],
        bc_checkpoint=None if args.bc_checkpoint is None else _resolve(args.bc_checkpoint),
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for record in result["checkpoints"]:
        diag = record["diagnostics"]
        print(
            f"{record['checkpoint']}: alpha={diag['path_encoder_alpha']} "
            f"final_norm={diag['path_encoder_final_weight_norm']} "
            f"effective_scale={diag['path_encoder_effective_output_scale']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
