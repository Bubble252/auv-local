from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from auv_lowlevel.config import load_config
from auv_rl.ddpg.agent import DDPGAgent
from auv_rl.ddpg.networks import NoisyLinear
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv
from auv_rl.training import load_training_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether NoisyNet, rather than the deterministic policy, drives residuals"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=240)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--control-steps", type=int, default=200)
    parser.add_argument("--noise-samples", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def _sigma_summary(agent: DDPGAgent) -> dict[str, Any]:
    layers = [
        module
        for module in agent.actor.modules()
        if isinstance(module, NoisyLinear) and module.use_noisy
    ]
    if not layers:
        return {
            "noisy_layer_count": 0,
            "hidden_sigma_mean": None,
            "output_sigma_mean": None,
        }
    sigmas = [
        float(F.softplus(layer.weight_rho).mean().detach().cpu())
        for layer in layers
    ]
    hidden = sigmas[:-1]
    return {
        "noisy_layer_count": len(layers),
        "hidden_sigma_mean": float(np.mean(hidden)) if hidden else None,
        "output_sigma_mean": float(sigmas[-1]),
    }


def _history_sequence(agent: DDPGAgent, env_ids: tuple[int, ...]) -> torch.Tensor:
    histories = [agent.env_history(env_id) for env_id in env_ids]
    lengths = {len(history) for history in histories}
    if lengths != {len(histories[0])} or not histories[0]:
        raise RuntimeError("all probe envs must have the same non-empty history length")
    stacked = np.stack([np.stack(history, axis=0) for history in histories], axis=0)
    return torch.as_tensor(stacked, device=agent.device, dtype=torch.float32)


def main() -> int:
    args = _parse_args()
    if args.control_steps <= 0 or args.noise_samples <= 0:
        raise ValueError("control-steps and noise-samples must be positive")
    payload = load_training_payload(args.config)
    observation = dict(payload.get("observation", {}))
    training = dict(payload.get("training", {}))
    environment_config = load_config(args.environment_config)
    seeds = np.arange(args.seed_start, args.seed_start + args.seed_count, dtype=np.int64)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    env = PCCommandResidualOffPolicyEnv(
        environment_config,
        num_envs=len(seeds),
        device=args.device,
        dtype=dtype,
        path_dim=int(observation.get("path_dim", 8)),
        validate_finite=False,
    )
    agent = DDPGAgent.load(
        args.checkpoint,
        device=args.device,
        expected_observation_version=env.observation_version,
    )
    observations, _ = env.reset(seeds.tolist())
    env_ids = tuple(range(env.num_envs))
    for env_id in env_ids:
        agent.begin_episode_for_env(env_id, int(env.episode_ids[env_id]))

    det_abs_sum = np.zeros(env.action_dim, dtype=np.float64)
    det_l2_values: list[float] = []
    det_saturation = np.zeros(env.action_dim, dtype=np.float64)
    det_saturation_any = 0
    raw_saturation = np.zeros(env.action_dim, dtype=np.float64)
    raw_saturation_any = 0
    raw_abs_sum = 0.0
    raw_abs_max = 0.0
    raw_abs_values: list[float] = []
    noise_ratio_values: list[float] = []
    noise_larger = 0
    cosine_values: list[float] = []
    noisy_l2_values: list[float] = []
    sample_mean_diff_values: list[float] = []
    applied_abs_sum = np.zeros(env.action_dim, dtype=np.float64)
    applied_l2_sum = 0.0
    base_l2_sum = 0.0
    gate_active_applied_l2_sum = 0.0
    gate_active_base_l2_sum = 0.0
    gate_active_steps = 0
    scale_clip_steps = 0
    rate_clip_steps = 0
    gate_sum = 0.0
    step_count = 0
    action_threshold = 0.9 * float(training.get("actor_output_scale", 1.0))
    raw_threshold = (
        9.0 if str(training.get("actor_output_squash", "tanh")).lower() == "softsign" else 3.0
    )

    for _ in range(args.control_steps):
        actions = agent.act_batch(env_ids, observations, evaluate=True)
        action_tensor = torch.as_tensor(actions, device=agent.device, dtype=torch.float32)
        sequence = _history_sequence(agent, env_ids)
        with torch.no_grad():
            raw, _ = agent.actor.forward_raw(sequence, sample_noise=False)
            noisy_samples = []
            for _ in range(args.noise_samples):
                noisy, _ = agent.actor(
                    sequence,
                    sample_noise=True,
                    per_sample_noise=True,
                )
                noisy_samples.append(noisy[:, -1, :])
            noisy_tensor = torch.stack(noisy_samples, dim=0)

        det_abs_sum += np.abs(actions).sum(axis=0)
        det_l2 = torch.linalg.norm(action_tensor, dim=1)
        det_l2_values.extend(det_l2.detach().cpu().numpy().tolist())
        det_saturation += np.sum(np.abs(actions) >= action_threshold, axis=0)
        det_saturation_any += int(np.sum(np.any(np.abs(actions) >= action_threshold, axis=1)))
        raw_last = raw[:, -1, :].abs()
        raw_saturation += torch.sum(raw_last >= raw_threshold, dim=0).detach().cpu().numpy()
        raw_saturation_any += int(torch.sum(torch.any(raw_last >= raw_threshold, dim=1)).item())
        raw_abs_sum += float(torch.sum(raw_last).detach().cpu())
        raw_abs_max = max(raw_abs_max, float(torch.max(raw_last).detach().cpu()))
        raw_abs_values.extend(raw_last.detach().cpu().numpy().reshape(-1).tolist())
        noise_std = noisy_tensor.std(dim=0, unbiased=False)
        noise_std_l2 = torch.linalg.norm(noise_std, dim=1)
        ratio = noise_std_l2 / torch.clamp(det_l2, min=1e-12)
        noise_ratio_values.extend(ratio.detach().cpu().numpy().tolist())
        noise_larger += int(torch.sum(noise_std_l2 >= det_l2).item())
        cosine = torch.sum(noisy_tensor * action_tensor.unsqueeze(0), dim=-1) / torch.clamp(
            torch.linalg.norm(noisy_tensor, dim=-1) * det_l2.unsqueeze(0),
            min=1e-12,
        )
        cosine_values.extend(cosine.detach().cpu().numpy().reshape(-1).tolist())
        noisy_l2_values.extend(
            torch.linalg.norm(noisy_tensor, dim=-1).detach().cpu().numpy().reshape(-1).tolist()
        )
        sample_mean_diff_values.extend(
            (noisy_tensor - action_tensor.unsqueeze(0))
            .abs()
            .amax(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

        observations, _, terminated, truncated, _ = env.step(actions, auto_reset=False)
        applied = np.asarray(env.last_applied_residual, dtype=np.float64)
        base = np.asarray(env.last_base_command, dtype=np.float64)
        gate = np.asarray(env.last_residual_gate, dtype=np.float64)
        applied_abs_sum += np.abs(applied).sum(axis=0)
        applied_l2_sum += float(np.sum(np.linalg.norm(applied, axis=1)))
        base_l2_sum += float(np.sum(np.linalg.norm(base, axis=1)))
        gate_active = gate > 0.5
        gate_active_steps += int(np.sum(gate_active))
        gate_active_applied_l2_sum += float(
            np.sum(np.linalg.norm(applied[gate_active], axis=1))
        )
        gate_active_base_l2_sum += float(
            np.sum(np.linalg.norm(base[gate_active], axis=1))
        )
        scale_clip_steps += int(np.sum(np.any(env.last_scale_clipped, axis=1)))
        rate_clip_steps += int(np.sum(np.any(env.last_rate_clipped, axis=1)))
        gate_sum += float(np.sum(env.last_residual_gate))
        step_count += 1
        if np.any(terminated | truncated):
            break

    states = step_count * env.num_envs
    summary = {
        "schema": "auv_goal59_noisynet_probe.v1",
        "config": str(args.config),
        "environment_config": str(args.environment_config),
        "checkpoint": str(args.checkpoint),
        "seed_range": [int(seeds[0]), int(seeds[-1])],
        "envs": env.num_envs,
        "control_steps": step_count,
        "states": states,
        "noise_samples_per_state": args.noise_samples,
        "actor_output_scale": float(training.get("actor_output_scale", 1.0)),
        "actor_output_squash": str(training.get("actor_output_squash", "tanh")),
        "deterministic_abs_mean_per_dim": (det_abs_sum / max(states, 1)).tolist(),
        "deterministic_action_l2_mean": float(np.mean(det_l2_values)),
        "deterministic_action_l2_p95": float(np.quantile(det_l2_values, 0.95)),
        "deterministic_action_saturation_fraction_per_dim": (
            det_saturation / max(states, 1)
        ).tolist(),
        "deterministic_action_saturation_fraction_any_dim": float(
            det_saturation_any / max(states, 1)
        ),
        "actor_raw_abs_mean": float(raw_abs_sum / max(states * env.action_dim, 1)),
        "actor_raw_abs_max": float(raw_abs_max),
        "actor_raw_abs_p95": float(np.quantile(raw_abs_values, 0.95)),
        "raw_saturation_fraction_per_dim": (
            raw_saturation / max(states, 1)
        ).tolist(),
        "raw_saturation_fraction_any_dim": float(
            raw_saturation_any / max(states, 1)
        ),
        "noise_std_to_det_l2_ratio_mean": float(np.mean(noise_ratio_values)),
        "noise_std_to_det_l2_ratio_median": float(np.median(noise_ratio_values)),
        "noise_std_larger_than_det_fraction": float(noise_larger / max(states, 1)),
        "noisy_sample_to_mean_cosine_mean": float(np.mean(cosine_values)),
        "noisy_sample_to_mean_cosine_median": float(np.median(cosine_values)),
        "noisy_sample_l2_mean": float(np.mean(noisy_l2_values)),
        "deterministic_vs_noise_sample_mean_max_abs_diff": float(
            np.mean(sample_mean_diff_values)
        ),
        "applied_residual_abs_mean_per_dim": (applied_abs_sum / max(states, 1)).tolist(),
        "applied_residual_l2_mean": float(applied_l2_sum / max(states, 1)),
        "applied_to_base_l2_ratio": float(applied_l2_sum / max(base_l2_sum, 1e-12)),
        "gate_active_rate": float(gate_active_steps / max(states, 1)),
        "residual_l2_ratio_gate_active": float(
            gate_active_applied_l2_sum / max(gate_active_base_l2_sum, 1e-12)
        ),
        "scale_clip_rate_any_dim": float(scale_clip_steps / max(states, 1)),
        "rate_clip_rate_any_dim": float(rate_clip_steps / max(states, 1)),
        "residual_gate_mean": float(gate_sum / max(states, 1)),
        **_sigma_summary(agent),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
