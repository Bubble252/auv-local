"""Measure whether a PC actor actually uses its DADE history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from auv_lowlevel._paths import resolve_from_root
from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.checkpoint import validate_pc_checkpoint_metadata
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_history_ablation import (
    PC_HISTORY_ABLATION_MODES,
    ablate_pc_history,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_policy import PCActorCritic, pc_schema_hash
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure DADE feature and action sensitivity to PC history ablations"
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _build_policy(
    training_config_path: Path,
    checkpoint_path: Path,
    *,
    num_actor_obs: int,
    num_critic_obs: int,
    device: torch.device,
) -> tuple[PCActorCritic, dict[str, Any]]:
    training = load_training_config(training_config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = validate_pc_checkpoint_metadata(
        checkpoint,
        expected_actor_observation_dim=num_actor_obs,
        expected_critic_observation_dim=num_critic_obs,
        expected_schema_hash=pc_schema_hash(),
    )
    assert metadata is not None
    if metadata["training_config_sha256"] != sha256_file(training_config_path):
        raise ValueError("checkpoint training-config hash does not match --training-config")
    policy_config = training.rsl_train_cfg()["policy"]
    policy = PCActorCritic(
        num_actor_obs,
        num_critic_obs,
        3,
        **{key: value for key, value in policy_config.items() if key != "class_name"},
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    return policy, metadata


def _distribution(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    flat = torch.cat([value.reshape(-1).to(dtype=torch.float64) for value in values])
    return {
        "mean": float(flat.mean()),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


def _run(
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    environment_config_path: Path | None,
    seeds: tuple[int, ...],
    max_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    training = load_training_config(training_config_path)
    if training.pc is None or training.history is None:
        raise ValueError("diagnostic requires a PC training config with history")
    resolved_environment_config = (
        environment_config_path
        if environment_config_path is not None
        else _resolve(Path(training.environment_config))
    )
    config = load_config(resolved_environment_config)
    if config.arm != "PC" or config.policy_mode != "pure_controller":
        raise ValueError("environment config is not a PC pure-controller config")
    inner = BatchedPCLowLevelEnv(
        config,
        num_envs=len(seeds),
        device=device.type,
        dtype=dtype,
        validate_finite=True,
    )
    env = BatchTorchPCVecEnv(
        inner,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        short_stride=training.history.short_stride,
        long_stride=training.history.long_stride,
        normalization=PCNormalization(
            position_scale_m=config.normalization.position_scale_m,
            attitude_scale_rad=config.normalization.attitude_scale_rad,
            angular_rate_scale_rad_s=config.normalization.angular_rate_scale_rad_s,
            surge_speed_scale_mps=config.normalization.surge_speed_scale_mps,
            sway_heave_speed_scale_mps=config.normalization.sway_heave_speed_scale_mps,
            current_scale_mps=config.normalization.current_scale_mps,
        ),
        current_frame_dropout_prob=0.0,
    )
    policy, metadata = _build_policy(
        training_config_path,
        checkpoint_path,
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        device=device,
    )
    observations, _ = env.reset(seeds)
    active = torch.ones(len(seeds), dtype=torch.bool, device=device)
    base_action_values: list[torch.Tensor] = []
    base_chat_norms: list[torch.Tensor] = []
    mode_values: dict[str, dict[str, list[torch.Tensor]]] = {
        mode: {"action_delta_l2": [], "z_delta_l2": [], "chat_delta_l2": []}
        for mode in PC_HISTORY_ABLATION_MODES
        if mode != "none"
    }

    for _ in range(max_steps):
        if not bool(active.any()):
            break
        active_observation = observations[active]
        with torch.inference_mode():
            base_action = policy.act_inference(active_observation)
            state, _, _, _ = policy.actor_layout.split(active_observation)
            base_z = policy.actor_encoder(policy._trainable_actor_state(state))
            if policy.actor_dade is None:
                base_chat = torch.zeros_like(base_z)
            else:
                base_chat = policy._actor_features_with_dade(active_observation)[1]

            base_action_values.append(torch.linalg.vector_norm(base_action, dim=1))
            base_chat_norms.append(torch.linalg.vector_norm(base_chat, dim=1))
            for mode in mode_values:
                ablated_observation = ablate_pc_history(
                    active_observation, policy.actor_layout, mode
                )
                state_ab, _, _, _ = policy.actor_layout.split(ablated_observation)
                z_ab = policy.actor_encoder(policy._trainable_actor_state(state_ab))
                chat_ab = (
                    torch.zeros_like(z_ab)
                    if policy.actor_dade is None
                    else policy._actor_features_with_dade(ablated_observation)[1]
                )
                action_ab = policy.act_inference(ablated_observation)
                mode_values[mode]["action_delta_l2"].append(
                    torch.linalg.vector_norm(action_ab - base_action, dim=1)
                )
                mode_values[mode]["z_delta_l2"].append(
                    torch.linalg.vector_norm(z_ab - base_z, dim=1)
                )
                mode_values[mode]["chat_delta_l2"].append(
                    torch.linalg.vector_norm(chat_ab - base_chat, dim=1)
                )

        full_action = torch.zeros_like(observations[:, :3])
        full_action[active] = base_action
        observations, _, done, _ = env.step(full_action, auto_reset=False)
        active &= ~done.to(dtype=torch.bool)

    return {
        "schema_version": "auv_pc_dade_history_diagnostic.v1",
        "training_config": str(training_config_path),
        "environment_config": str(resolved_environment_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": metadata,
        "seeds": list(seeds),
        "max_steps": max_steps,
        "samples": int(sum(value.numel() for value in base_action_values)),
        "dade_enabled": bool(policy.dade_enabled),
        "base_action_l2": _distribution(base_action_values),
        "base_dade_feature_l2": _distribution(base_chat_norms),
        "ablations": {
            mode: {name: _distribution(values) for name, values in metrics.items()}
            for mode, metrics in mode_values.items()
        },
        "notes": {
            "current-only": "removes older history; z_t and c_t unchanged",
            "current-zero": "removes current state and newest history row; z_t also changes",
            "history-zero": "zeros both windows; z_t and c_t unchanged",
            "history-reverse": "reverses temporal order in both windows; z_t and c_t unchanged",
            "interpretation": (
                "A useful history pathway should show non-trivial chat_delta_l2 for "
                "current-only/history-zero/history-reverse, not only for current-zero."
            ),
        },
    }


def main() -> int:
    args = _parse_args()
    if args.seed_start < 0 or args.seed_count <= 0 or args.max_steps <= 0:
        raise ValueError("seed count and max steps must be positive")
    training_config_path = _resolve(args.training_config)
    checkpoint_path = _resolve(args.checkpoint)
    environment_config_path = (
        None if args.environment_config is None else _resolve(args.environment_config)
    )
    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    report = _run(
        training_config_path=training_config_path,
        checkpoint_path=checkpoint_path,
        environment_config_path=environment_config_path,
        seeds=seeds,
        max_steps=args.max_steps,
        device=torch.device(args.device),
        dtype=torch.float32 if args.dtype == "float32" else torch.float64,
    )
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
