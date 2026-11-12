"""Vectorized off-policy training loop for the PC command-residual adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agent import DDPGAgent
from .pc_command_residual_env import PCCommandResidualOffPolicyEnv


@dataclass(frozen=True)
class PCTrainingMetrics:
    requested_env_transitions: int
    env_transitions: int
    control_steps: int
    gradient_steps: int
    update_to_data_ratio: float
    actual_update_to_data_ratio: float
    wall_time_s: float
    transitions_per_s: float
    gradient_steps_per_s: float
    completed_episodes: int
    residual_abs_mean: float
    residual_l2_ratio: float
    residual_scale_clip_rate: float
    residual_rate_clip_rate: float
    residual_gate_mean: float
    gate_active_rate: float
    residual_l2_ratio_gate_active: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_episode_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_rows(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_pc_command_residual_ddpg(
    env: PCCommandResidualOffPolicyEnv,
    agent: DDPGAgent,
    *,
    total_env_transitions: int,
    seed: int,
    update_to_data_ratio: float = 0.125,
    output: Path | str | None = None,
    config_payload: dict[str, Any] | None = None,
) -> PCTrainingMetrics:
    if total_env_transitions <= 0:
        raise ValueError("total_env_transitions must be positive")
    if not math.isfinite(update_to_data_ratio) or update_to_data_ratio < 0.0:
        raise ValueError("update_to_data_ratio must be finite and non-negative")
    if agent.config.observation_dim != env.observation_dim:
        raise ValueError("agent observation dimension does not match environment")
    if agent.config.action_dim != env.action_dim:
        raise ValueError("agent action dimension does not match environment")
    if agent.config.privileged_dim != env.privileged_dim:
        raise ValueError("agent privileged dimension does not match environment")
    agent.set_observation_spec(
        {
            "name": env.observation_name,
            "version": env.observation_version,
            "shape": [env.observation_dim],
        }
    )
    output_path = None if output is None else Path(output)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    observations, _ = env.reset()
    env_ids = tuple(range(env.num_envs))
    for env_id in env_ids:
        agent.begin_episode_for_env(env_id, int(env.episode_ids[env_id]))

    env_transitions = 0
    control_steps = 0
    gradient_steps = 0
    episode_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, float]] = []
    residual_abs_sum = 0.0
    residual_l2_sum = 0.0
    base_l2_sum = 0.0
    residual_values = 0
    scale_clip_steps = 0
    rate_clip_steps = 0
    gate_sum = 0.0
    gate_values = 0
    gate_active_steps = 0
    gate_active_residual_l2_sum = 0.0
    gate_active_base_l2_sum = 0.0
    started = time.perf_counter()

    while env_transitions < total_env_transitions:
        privileged = env.privileged.copy()
        actions = agent.act_batch(env_ids, observations, evaluate=False)
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        agent.observe_batch(
            env_ids,
            observations,
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(info["final_observations"], dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            privileged=privileged,
            next_privileged=info["final_privileged"],
        )

        applied = env.last_applied_residual
        base = env.last_base_command
        gate = env.last_residual_gate
        residual_abs_sum += float(np.sum(np.abs(applied)))
        residual_l2_sum += float(np.sum(np.linalg.norm(applied, axis=1)))
        base_l2_sum += float(np.sum(np.linalg.norm(base, axis=1)))
        residual_values += int(applied.size)
        scale_clip_steps += int(np.any(env.last_scale_clipped, axis=1).sum())
        rate_clip_steps += int(np.any(env.last_rate_clipped, axis=1).sum())
        gate_sum += float(np.sum(gate))
        gate_values += int(gate.size)
        gate_active = gate > 0.5
        gate_active_steps += int(np.sum(gate_active))
        gate_active_residual_l2_sum += float(
            np.sum(np.linalg.norm(applied[gate_active], axis=1))
        )
        gate_active_base_l2_sum += float(
            np.sum(np.linalg.norm(base[gate_active], axis=1))
        )

        env_transitions += env.num_envs
        control_steps += 1
        observations = next_observations

        done = np.flatnonzero(terminated | truncated)
        for env_id in done.tolist():
            agent.begin_episode_for_env(int(env_id), int(env.episode_ids[int(env_id)]))

        if len(agent.replay) >= agent.config.learning_starts:
            eligible_transitions = max(env_transitions - agent.config.learning_starts, 0)
            target_gradient_steps = int(math.floor(eligible_transitions * update_to_data_ratio))
            while gradient_steps < target_gradient_steps:
                update = agent.update(gradient_steps)
                if update is None:
                    break
                gradient_steps += 1

        completed = env.drain_completed_infos()
        for row in completed:
            row["env_transitions"] = env_transitions
            row["gradient_steps"] = gradient_steps
        episode_rows.extend(completed)

        if control_steps == 1 or control_steps % 100 == 0:
            elapsed = time.perf_counter() - started
            metrics_rows.append(
                {
                    "control_step": float(control_steps),
                    "env_transitions": float(env_transitions),
                    "gradient_steps": float(gradient_steps),
                    "elapsed_s": float(elapsed),
                    "transitions_per_s": float(env_transitions / max(elapsed, 1e-12)),
                    "residual_abs_mean": float(np.mean(np.abs(applied))),
                    "residual_l2_ratio": float(
                        np.sum(np.linalg.norm(applied, axis=1))
                        / max(float(np.sum(np.linalg.norm(base, axis=1))), 1e-12)
                    ),
                    "residual_gate_mean": float(np.mean(gate)),
                }
            )
            print(
                f"control_step={control_steps} env_transitions={env_transitions} "
                f"completed_episodes={len(episode_rows)} gradients={gradient_steps} "
                f"residual_abs_mean={np.mean(np.abs(applied)):.6g}",
                flush=True,
            )

    wall_time = time.perf_counter() - started
    residual_abs_mean = residual_abs_sum / max(residual_values, 1)
    residual_l2_ratio = residual_l2_sum / max(base_l2_sum, 1e-12)
    metrics = PCTrainingMetrics(
        requested_env_transitions=int(total_env_transitions),
        env_transitions=int(env_transitions),
        control_steps=int(control_steps),
        gradient_steps=int(gradient_steps),
        update_to_data_ratio=float(update_to_data_ratio),
        actual_update_to_data_ratio=float(
            gradient_steps / max(env_transitions - agent.config.learning_starts, 1)
        ),
        wall_time_s=float(wall_time),
        transitions_per_s=float(env_transitions / max(wall_time, 1e-12)),
        gradient_steps_per_s=float(gradient_steps / max(wall_time, 1e-12)),
        completed_episodes=len(episode_rows),
        residual_abs_mean=float(residual_abs_mean),
        residual_l2_ratio=float(residual_l2_ratio),
        residual_scale_clip_rate=float(scale_clip_steps / max(control_steps * env.num_envs, 1)),
        residual_rate_clip_rate=float(rate_clip_steps / max(control_steps * env.num_envs, 1)),
        residual_gate_mean=float(gate_sum / max(gate_values, 1)),
        gate_active_rate=float(gate_active_steps / max(control_steps * env.num_envs, 1)),
        residual_l2_ratio_gate_active=float(
            gate_active_residual_l2_sum / max(gate_active_base_l2_sum, 1e-12)
        ),
    )

    if output_path is not None:
        agent.save(output_path / "final.pt")
        agent.save(output_path / "model.pt")
        _write_metrics_rows(output_path / "training_curve.csv", metrics_rows)
        _write_episode_rows(output_path / "episodes.csv", episode_rows)
        (output_path / "training_metrics.json").write_text(
            json.dumps(metrics.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest = {
            "schema": "auv_pc_command_residual_offpolicy.v1",
            "seed": int(seed),
            "num_envs": int(env.num_envs),
            "requested_env_transitions": int(total_env_transitions),
            "env_transitions": int(env_transitions),
            "gradient_steps": int(gradient_steps),
            "observation_name": env.observation_name,
            "observation_version": env.observation_version,
            "observation_dim": int(env.observation_dim),
            "privileged_dim": int(env.privileged_dim),
            "action_dim": int(env.action_dim),
            "controller_state": env.wrapper.controller_state,
            "command_residual_scale": env.wrapper.scale.detach().cpu().tolist(),
            "command_residual_rate_limit": env.wrapper.rate_limit.detach().cpu().tolist(),
            "metrics": metrics.as_dict(),
            "config_hash": None if config_payload is None else _payload_hash(config_payload),
            "git_commit": _git_commit(),
            "device": str(env.device),
        }
        (output_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return metrics


__all__ = ["PCTrainingMetrics", "train_pc_command_residual_ddpg"]
