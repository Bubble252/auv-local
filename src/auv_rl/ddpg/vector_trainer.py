from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found, unused-ignore]

from ..envs.parallel_planar import P1TensorVectorEnv
from ..training import EPISODE_FIELDS
from .agent import DDPGAgent
from .diagnostics import (
    critic_health_passes,
    diagnose_critic,
    eval_saturation_gate_passes,
    reset_tanh_gate_passes,
    saturation_gate_passes,
)

_TRAINING_STATE_SCHEMA = "auv_p1_vector_training_state.v1"


@dataclass(frozen=True)
class VectorTrainingMetrics:
    requested_env_transitions: int
    env_transitions: int
    vector_control_steps: int
    environment_substeps: int
    gradient_steps: int
    update_to_data_ratio: float
    actual_update_to_data_ratio: float
    total_update_to_data_ratio: float
    warmup_env_transitions: int
    eligible_env_transitions: int
    wall_time_s: float
    env_steps_per_s: float
    control_steps_per_s: float
    gradient_steps_per_s: float
    sensor_time_s: float
    environment_time_s: float
    replay_add_time_s: float
    replay_sample_time_s: float
    pper_update_time_s: float
    gradient_time_s: float
    evaluation_time_s: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class VectorEvaluationMetrics:
    env_transitions: int
    gradient_steps: int
    episodes: int
    successes: int
    scenario_episodes: dict[str, int]
    scenario_successes: dict[str, int]
    collisions: int
    out_of_bounds: int
    timeouts: int
    mean_u_dot_saturation: float
    mean_yaw_rate_saturation: float
    finite: bool
    critic_health_gate: bool
    reset_tanh_gate: bool
    eval_saturation_gate: bool
    saturation_gate: bool
    success_gate: bool
    gate_passed: bool
    score: tuple[int, int, int, float, int]
    health: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = list(self.score)
        return payload


def _evaluation_score(
    gate_passed: bool,
    successes: int,
    failures: int,
    saturation: float,
    timeouts: int,
) -> tuple[int, int, int, float, int]:
    return (
        int(gate_passed),
        int(successes),
        -int(failures),
        -float(saturation),
        -int(timeouts),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_vector_training_state(path: Path | str) -> dict[str, Any]:
    """Load the checkpoint that makes vector training resumable."""

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema") != _TRAINING_STATE_SCHEMA:
        raise ValueError("not an auv P1 vector training-state checkpoint")
    return state


def _write_training_state(
    path: Path,
    *,
    env: P1TensorVectorEnv,
    agent: DDPGAgent,
    observations: np.ndarray,
    env_transitions: int,
    vector_control_steps: int,
    environment_substeps: int,
    gradient_steps: int,
    metrics_rows: list[dict[str, float]],
    episode_rows: list[dict[str, Any]],
    evaluation_history: list[dict[str, Any]],
    best_score: tuple[int, int, int, float, int] | None,
    timers: dict[str, float],
    elapsed_wall_time_s: float,
    seed: int,
) -> None:
    payload = {
        "schema": _TRAINING_STATE_SCHEMA,
        "seed": int(seed),
        "num_envs": int(env.num_envs),
        "env_transitions": int(env_transitions),
        "vector_control_steps": int(vector_control_steps),
        "environment_substeps": int(environment_substeps),
        "gradient_steps": int(gradient_steps),
        "observations": np.asarray(observations, dtype=np.float32).copy(),
        "agent": agent.training_state_dict(),
        "environment": env.training_state_dict(),
        "metrics_rows": list(metrics_rows),
        "episode_rows": list(episode_rows),
        "evaluation_history": list(evaluation_history),
        "best_score": None if best_score is None else list(best_score),
        "timers": dict(timers),
        "elapsed_wall_time_s": float(elapsed_wall_time_s),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate_parallel_ddpg(
    agent: DDPGAgent,
    env: P1TensorVectorEnv,
    *,
    env_transitions: int,
    gradient_steps: int,
    scenario_keys: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    env_id_offset: int = 1_000_000,
) -> tuple[list[dict[str, Any]], VectorEvaluationMetrics]:
    """Evaluate one checkpoint on deterministic, scenario-separated episodes."""

    keys = env.config.active_scenario_keys if scenario_keys is None else scenario_keys
    if not keys:
        raise ValueError("scenario_keys must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")

    rows: list[dict[str, Any]] = []
    reset_observations: list[np.ndarray] = []
    evaluation_env = P1TensorVectorEnv(1, env.config, seed=0)
    evaluation_index = 0
    for scenario_key in keys:
        for seed in seeds:
            observation, _ = evaluation_env.reset(
                seed=seed,
                options={"scenario_key": scenario_key, "jitter": False},
            )
            reset_observations.append(np.asarray(observation[0], dtype=np.float32).copy())
            eval_env_id = env_id_offset + evaluation_index
            agent.begin_episode_for_env(eval_env_id, evaluation_index)
            for _ in range(evaluation_env.config.max_episode_steps):
                action = agent.act_batch([eval_env_id], observation, evaluate=True)
                observation, _, terminated, truncated, _ = evaluation_env.step(action)
                if bool(terminated[0]) or bool(truncated[0]):
                    completed = evaluation_env.drain_completed_infos()
                    if len(completed) != 1:
                        raise RuntimeError("evaluation episode did not produce one completion row")
                    row = completed[0]
                    row["eval_seed"] = int(seed)
                    row["eval_episode_index"] = int(evaluation_index)
                    rows.append(row)
                    break
            else:
                raise RuntimeError(
                    "evaluation did not terminate at "
                    f"{evaluation_env.config.max_episode_steps} steps"
                )
            evaluation_index += 1

    health = diagnose_critic(agent, np.stack(reset_observations, axis=0))
    scenario_episodes = {
        key: sum(1 for row in rows if str(row.get("scenario_key")) == key) for key in keys
    }
    scenario_successes = {
        key: sum(
            1 for row in rows if str(row.get("scenario_key")) == key and bool(row.get("success"))
        )
        for key in keys
    }
    u_dot_saturation = float(
        np.mean([float(row.get("action_u_dot_saturation_ratio", 0.0)) for row in rows])
    )
    yaw_saturation = float(
        np.mean([float(row.get("action_yaw_rate_saturation_ratio", 0.0)) for row in rows])
    )
    finite = all(
        math.isfinite(float(row.get(field, 0.0)))
        for row in rows
        for field in ("goal_distance_m", "reward_total", "min_clearance_m")
    )
    success_gate = bool(
        scenario_episodes.get("s1", 0) > 0
        and scenario_successes.get("s1", 0) * 3 >= scenario_episodes.get("s1", 1) * 2
        and scenario_episodes.get("s2", 0) > 0
        and scenario_successes.get("s2", 0) * 3 >= scenario_episodes.get("s2", 1)
    )
    critic_health_gate = critic_health_passes(health)
    reset_tanh_gate = reset_tanh_gate_passes(health)
    eval_saturation_gate = eval_saturation_gate_passes(rows)
    saturation_gate = saturation_gate_passes(health, rows)
    gate_passed = bool(
        finite
        and critic_health_gate
        and reset_tanh_gate
        and eval_saturation_gate
        and saturation_gate
        and success_gate
    )
    successes = sum(bool(row.get("success")) for row in rows)
    collisions = sum(bool(row.get("collision")) for row in rows)
    out_of_bounds = sum(bool(row.get("out_of_bounds")) for row in rows)
    timeouts = sum(bool(row.get("timeout")) for row in rows)
    score = _evaluation_score(
        gate_passed,
        successes,
        collisions + out_of_bounds,
        max(u_dot_saturation, yaw_saturation),
        timeouts,
    )
    summary = VectorEvaluationMetrics(
        env_transitions=int(env_transitions),
        gradient_steps=int(gradient_steps),
        episodes=len(rows),
        successes=successes,
        scenario_episodes=scenario_episodes,
        scenario_successes=scenario_successes,
        collisions=collisions,
        out_of_bounds=out_of_bounds,
        timeouts=timeouts,
        mean_u_dot_saturation=u_dot_saturation,
        mean_yaw_rate_saturation=yaw_saturation,
        finite=finite,
        critic_health_gate=critic_health_gate,
        reset_tanh_gate=reset_tanh_gate,
        eval_saturation_gate=eval_saturation_gate,
        saturation_gate=saturation_gate,
        success_gate=success_gate,
        gate_passed=gate_passed,
        score=score,
        health=health,
    )
    return rows, summary


def train_parallel_ddpg(
    env: P1TensorVectorEnv,
    agent: DDPGAgent,
    *,
    total_env_transitions: int,
    seed: int = 0,
    update_to_data_ratio: float = 1.0,
    output: Path | str | None = None,
    evaluation_every_env_transitions: int | None = None,
    evaluation_seeds: tuple[int, ...] = (0, 1, 2),
    evaluation_scenario_keys: tuple[str, ...] | None = None,
    resume_state: dict[str, Any] | None = None,
) -> VectorTrainingMetrics:
    """Run vectorized P1 rollout with optional deterministic checkpoint evaluation."""

    if total_env_transitions <= 0:
        raise ValueError("total_env_transitions must be positive")
    if not math.isfinite(update_to_data_ratio) or update_to_data_ratio < 0.0:
        raise ValueError("update_to_data_ratio must be finite and non-negative")
    if evaluation_every_env_transitions is not None and evaluation_every_env_transitions <= 0:
        raise ValueError("evaluation_every_env_transitions must be positive")
    if evaluation_every_env_transitions is not None and output is None:
        raise ValueError("periodic evaluation requires an output directory")
    if agent.config.observation_dim != env.builder.spec.shape[0]:
        raise ValueError("agent observation dimension does not match environment")
    if agent.config.action_dim != env.action_space.shape[-1]:
        raise ValueError("agent action dimension does not match environment")
    agent.set_observation_spec(
        {
            "name": env.builder.spec.name,
            "version": env.builder.spec.version,
            "shape": list(env.builder.spec.shape),
        }
    )

    output_path = None if output is None else Path(output)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    env_ids = tuple(range(env.num_envs))
    metrics_rows: list[dict[str, float]] = []
    episode_rows: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    env_transitions = 0
    vector_control_steps = 0
    environment_substeps = 0
    gradient_steps = 0
    sensor_time_s = 0.0
    environment_time_s = 0.0
    replay_add_time_s = 0.0
    replay_sample_time_s = 0.0
    pper_update_time_s = 0.0
    gradient_time_s = 0.0
    evaluation_time_s = 0.0
    best_score: tuple[int, int, int, float, int] | None = None
    previous_wall_time_s = 0.0

    if resume_state is None:
        observations, _ = env.reset(seed=seed)
        for env_id in env_ids:
            agent.begin_episode_for_env(env_id, int(env.episode_ids[env_id]))
    else:
        if resume_state.get("schema") != _TRAINING_STATE_SCHEMA:
            raise ValueError("resume state schema does not match vector trainer")
        if int(resume_state["num_envs"]) != env.num_envs:
            raise ValueError("resume state num_envs does not match environment")
        if int(resume_state["seed"]) != seed:
            raise ValueError("resume state seed does not match requested seed")
        restored_transitions = int(resume_state["env_transitions"])
        if restored_transitions > total_env_transitions:
            raise ValueError("requested total is smaller than the resumed transition count")
        env.load_training_state_dict(resume_state["environment"])
        agent.load_training_state_dict(resume_state["agent"])
        observations = np.asarray(resume_state["observations"], dtype=np.float32).copy()
        expected_shape = (env.num_envs, env.builder.spec.shape[0])
        if observations.shape != expected_shape or not np.all(np.isfinite(observations)):
            raise ValueError("resume state observations do not match the environment")
        env_transitions = restored_transitions
        vector_control_steps = int(resume_state["vector_control_steps"])
        environment_substeps = int(resume_state["environment_substeps"])
        gradient_steps = int(resume_state["gradient_steps"])
        metrics_rows = list(resume_state["metrics_rows"])
        episode_rows = list(resume_state["episode_rows"])
        evaluation_history = list(resume_state["evaluation_history"])
        raw_best_score = resume_state.get("best_score")
        best_score = None if raw_best_score is None else tuple(raw_best_score)
        timers = dict(resume_state.get("timers", {}))
        sensor_time_s = float(timers.get("sensor_time_s", 0.0))
        environment_time_s = float(timers.get("environment_time_s", 0.0))
        replay_add_time_s = float(timers.get("replay_add_time_s", 0.0))
        replay_sample_time_s = float(timers.get("replay_sample_time_s", 0.0))
        pper_update_time_s = float(timers.get("pper_update_time_s", 0.0))
        gradient_time_s = float(timers.get("gradient_time_s", 0.0))
        evaluation_time_s = float(timers.get("evaluation_time_s", 0.0))
        previous_wall_time_s = float(resume_state.get("elapsed_wall_time_s", 0.0))

    next_evaluation = evaluation_every_env_transitions
    started = time.perf_counter()

    def run_evaluation() -> None:
        nonlocal evaluation_time_s, best_score
        eval_started = time.perf_counter()
        rows, summary = evaluate_parallel_ddpg(
            agent,
            env,
            env_transitions=env_transitions,
            gradient_steps=gradient_steps,
            scenario_keys=evaluation_scenario_keys,
            seeds=evaluation_seeds,
        )
        evaluation_time_s += time.perf_counter() - eval_started
        serializable_rows = [_jsonable(row) for row in rows]
        evaluation_history.append(
            {
                "summary": summary.as_dict(),
                "episodes": serializable_rows,
            }
        )
        if output_path is None:
            return
        agent.save(output_path / "checkpoint_latest.pt")
        if best_score is None or summary.score > best_score:
            best_score = summary.score
            agent.save(output_path / "checkpoint_best.pt")
            (output_path / "best_evaluation.json").write_text(
                json.dumps(
                    {
                        "env_transitions": env_transitions,
                        "gradient_steps": gradient_steps,
                        "summary": summary.as_dict(),
                        "episodes": serializable_rows,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        (output_path / "evaluation_history.json").write_text(
            json.dumps(evaluation_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_training_state(
            output_path / "training_state.pt",
            env=env,
            agent=agent,
            observations=observations,
            env_transitions=env_transitions,
            vector_control_steps=vector_control_steps,
            environment_substeps=environment_substeps,
            gradient_steps=gradient_steps,
            metrics_rows=metrics_rows,
            episode_rows=episode_rows,
            evaluation_history=evaluation_history,
            best_score=best_score,
            timers={
                "sensor_time_s": sensor_time_s,
                "environment_time_s": environment_time_s,
                "replay_add_time_s": replay_add_time_s,
                "replay_sample_time_s": replay_sample_time_s,
                "pper_update_time_s": pper_update_time_s,
                "gradient_time_s": gradient_time_s,
                "evaluation_time_s": evaluation_time_s,
            },
            elapsed_wall_time_s=previous_wall_time_s + time.perf_counter() - started,
            seed=seed,
        )

    if resume_state is None and evaluation_every_env_transitions is not None:
        run_evaluation()
        next_evaluation = evaluation_every_env_transitions
    elif resume_state is not None and evaluation_every_env_transitions is not None:
        next_evaluation = (env_transitions // evaluation_every_env_transitions + 1) * (
            evaluation_every_env_transitions
        )

    while env_transitions < total_env_transitions:
        actions = agent.act_batch(env_ids, observations, evaluate=False)
        environment_start = time.perf_counter()
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        environment_time_s += time.perf_counter() - environment_start
        sensor_time_s += env.last_sensor_time_s
        final_observations = np.asarray(info["final_observations"], dtype=np.float32).copy()
        replay_start = time.perf_counter()
        agent.observe_batch(
            env_ids,
            observations,
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            final_observations,
            np.asarray(terminated, dtype=np.bool_),
        )
        replay_add_time_s += time.perf_counter() - replay_start

        env_transitions += env.num_envs
        vector_control_steps += 1
        environment_substeps += env.num_envs * int(
            round(env.config.dt_s / env.config.integration_step_s)
        )
        observations = next_observations

        done_indices = np.flatnonzero(terminated | truncated)
        for env_id in done_indices.tolist():
            agent.begin_episode_for_env(int(env_id), int(env.episode_ids[int(env_id)]))

        if len(agent.replay) >= agent.config.learning_starts:
            eligible_transitions = max(env_transitions - agent.config.learning_starts, 0)
            target_gradient_steps = int(math.floor(eligible_transitions * update_to_data_ratio))
            while gradient_steps < target_gradient_steps:
                update_start = time.perf_counter()
                update_metrics = agent.update(gradient_steps)
                gradient_time_s += time.perf_counter() - update_start
                if update_metrics is None:
                    break
                replay_sample_time_s += agent.last_replay_sample_time_s
                pper_update_time_s += agent.last_pper_update_time_s
                gradient_steps += 1
                if (
                    gradient_steps >= 2
                    and agent.config.cuda_graph_update
                    and agent.device.startswith("cuda")
                    and not agent.cuda_graph_active
                ):
                    raise RuntimeError("CUDA graph update was requested but could not be captured")
                metrics_rows.append(
                    {
                        "gradient_step": float(gradient_steps),
                        "env_transitions": float(env_transitions),
                        "critic_loss": update_metrics.critic_loss,
                        "actor_loss": update_metrics.actor_loss,
                        "mean_q": update_metrics.mean_q,
                        "mean_td_error": update_metrics.mean_td_error,
                        "beta": update_metrics.beta,
                    }
                )

        completed = env.drain_completed_infos()
        for row in completed:
            row["env_transitions"] = env_transitions
            row["gradient_steps"] = gradient_steps
        episode_rows.extend(completed)

        if vector_control_steps == 1 or vector_control_steps % 100 == 0:
            print(
                f"control_step={vector_control_steps} env_transitions={env_transitions} "
                f"completed_episodes={len(episode_rows)} gradients={gradient_steps}",
                flush=True,
            )

        if (
            evaluation_every_env_transitions is not None
            and next_evaluation is not None
            and env_transitions >= next_evaluation
        ):
            run_evaluation()
            while next_evaluation <= env_transitions:
                next_evaluation += evaluation_every_env_transitions

    if (
        evaluation_every_env_transitions is not None
        and evaluation_history
        and int(evaluation_history[-1]["summary"]["env_transitions"]) != env_transitions
    ):
        run_evaluation()

    wall_time = previous_wall_time_s + time.perf_counter() - started
    warmup_transitions = min(agent.config.learning_starts, env_transitions)
    eligible_transitions = max(env_transitions - warmup_transitions, 0)
    actual_ratio = gradient_steps / max(eligible_transitions, 1)
    metrics = VectorTrainingMetrics(
        requested_env_transitions=int(total_env_transitions),
        env_transitions=int(env_transitions),
        vector_control_steps=int(vector_control_steps),
        environment_substeps=int(environment_substeps),
        gradient_steps=int(gradient_steps),
        update_to_data_ratio=float(update_to_data_ratio),
        actual_update_to_data_ratio=float(actual_ratio),
        total_update_to_data_ratio=float(gradient_steps / max(env_transitions, 1)),
        warmup_env_transitions=int(warmup_transitions),
        eligible_env_transitions=int(eligible_transitions),
        wall_time_s=float(wall_time),
        env_steps_per_s=float(environment_substeps / max(wall_time, 1e-12)),
        control_steps_per_s=float(env_transitions / max(wall_time, 1e-12)),
        gradient_steps_per_s=float(gradient_steps / max(wall_time, 1e-12)),
        sensor_time_s=float(sensor_time_s),
        environment_time_s=float(environment_time_s),
        replay_add_time_s=float(replay_add_time_s),
        replay_sample_time_s=float(replay_sample_time_s),
        pper_update_time_s=float(pper_update_time_s),
        gradient_time_s=float(gradient_time_s),
        evaluation_time_s=float(evaluation_time_s),
    )
    if output_path is not None:
        _write_outputs(
            env,
            agent,
            metrics,
            metrics_rows,
            episode_rows,
            evaluation_history,
            best_score,
            output_path,
            seed,
        )
        if evaluation_every_env_transitions is None:
            _write_training_state(
                output_path / "training_state.pt",
                env=env,
                agent=agent,
                observations=observations,
                env_transitions=env_transitions,
                vector_control_steps=vector_control_steps,
                environment_substeps=environment_substeps,
                gradient_steps=gradient_steps,
                metrics_rows=metrics_rows,
                episode_rows=episode_rows,
                evaluation_history=evaluation_history,
                best_score=best_score,
                timers={
                    "sensor_time_s": sensor_time_s,
                    "environment_time_s": environment_time_s,
                    "replay_add_time_s": replay_add_time_s,
                    "replay_sample_time_s": replay_sample_time_s,
                    "pper_update_time_s": pper_update_time_s,
                    "gradient_time_s": gradient_time_s,
                    "evaluation_time_s": evaluation_time_s,
                },
                elapsed_wall_time_s=wall_time,
                seed=seed,
            )
    return metrics


def _recent_scenario_successes(
    episode_rows: list[dict[str, Any]],
) -> dict[str, int]:
    if not episode_rows:
        return {"s1": 0, "s2": 0}
    start = max(int(math.floor(len(episode_rows) * 0.8)), 0)
    recent = episode_rows[start:]
    return {
        key: sum(
            1 for row in recent if str(row.get("scenario_key")) == key and bool(row.get("success"))
        )
        for key in ("s1", "s2")
    }


def _write_outputs(
    env: P1TensorVectorEnv,
    agent: DDPGAgent,
    metrics: VectorTrainingMetrics,
    metrics_rows: list[dict[str, float]],
    episode_rows: list[dict[str, Any]],
    evaluation_history: list[dict[str, Any]],
    best_score: tuple[int, int, int, float, int] | None,
    output: Path,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    recent_successes = _recent_scenario_successes(episode_rows)
    payload = {
        "schema": "auv_p1_vector_training.v1",
        "num_envs": env.num_envs,
        "seed": seed,
        "observation_source": env.observation_source,
        "perception_mode": env.perception.perception_mode,
        "perception_config_hash": env.perception.config_hash,
        "observation_spec": {
            "name": env.builder.spec.name,
            "version": env.builder.spec.version,
            "shape": list(env.builder.spec.shape),
            "normalization": dict(env.builder.spec.normalization),
        },
        "action_spec": {
            "low": env.action_space.low[0].tolist(),
            "high": env.action_space.high[0].tolist(),
        },
        "update_to_data_ratio": metrics.update_to_data_ratio,
        "metrics": metrics.as_dict(),
        "recent_scenario_successes": recent_successes,
        "best_score": None if best_score is None else list(best_score),
        "checkpoint_best": str(output / "checkpoint_best.pt"),
        "checkpoint_latest": str(output / "checkpoint_latest.pt"),
        "checkpoint_selection_source": "analytic_13sector",
        "evaluation_source": env.observation_source,
        "periodic_evaluation_source": env.observation_source,
        "final_evaluation_source": "oceansim_simplified_pcl",
        "cuda_graph_requested": bool(agent.config.cuda_graph_update),
        "cuda_graph_active": bool(agent.cuda_graph_active),
        "note": "Training and checkpoint selection use analytic_13sector; final evaluation must use the OceanSim P1-A chain.",
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "training_metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if metrics_rows:
        with (output / "gradient_updates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(metrics_rows[0]))
            writer.writeheader()
            writer.writerows(metrics_rows)
    if episode_rows:
        fieldnames = list(EPISODE_FIELDS)
        for extra in ("env_transitions", "gradient_steps"):
            if extra not in fieldnames:
                fieldnames.append(extra)
        with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(episode_rows)
    if evaluation_history:
        (output / "evaluation_history.json").write_text(
            json.dumps(evaluation_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    agent.save(output / "model.pt")


__all__ = [
    "VectorEvaluationMetrics",
    "VectorTrainingMetrics",
    "evaluate_parallel_ddpg",
    "load_vector_training_state",
    "train_parallel_ddpg",
]
