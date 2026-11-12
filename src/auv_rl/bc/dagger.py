from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from ..envs.planar_avoidance import PlanarAvoidanceEnv
from ..privileged import privileged_critic_features
from .policy import BCActorPolicy
from .teacher import PotentialFieldTeacher, PotentialFieldTeacherConfig


@dataclass(frozen=True)
class DaggerCollectionConfig:
    scenarios: tuple[str, ...] = ("s1", "s2")
    seeds: tuple[int, ...] = (0, 1, 2)
    jitter: bool = True
    noise_sigma: tuple[float, float] = (0.02, 0.01)
    correction_threshold: float = 0.02
    near_clearance_m: float = 1.0
    sample_every_steps: int = 25
    failure_window_steps: int = 50
    max_transitions: int = 100_000
    seed_base: int = 20_000

    def __post_init__(self) -> None:
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must be non-empty and unique")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-empty and non-negative")
        if len(self.noise_sigma) != 2 or any(
            not np.isfinite(value) or value < 0.0 for value in self.noise_sigma
        ):
            raise ValueError("noise_sigma must contain two finite non-negative values")
        if not np.isfinite(self.correction_threshold) or self.correction_threshold < 0.0:
            raise ValueError("correction_threshold must be finite and non-negative")
        if not np.isfinite(self.near_clearance_m) or self.near_clearance_m < 0.0:
            raise ValueError("near_clearance_m must be finite and non-negative")
        if self.sample_every_steps <= 0 or self.failure_window_steps <= 0:
            raise ValueError("sample_every_steps and failure_window_steps must be positive")
        if self.max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        if self.seed_base < 0:
            raise ValueError("seed_base must be non-negative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_dagger_demonstrations(
    env: PlanarAvoidanceEnv,
    policy: BCActorPolicy,
    teacher: PotentialFieldTeacher,
    config: DaggerCollectionConfig,
    *,
    checkpoint_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Collect teacher corrections on states visited by the current BC policy."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed_base)
    if not isinstance(env.action_space, Box) or not isinstance(env.observation_space, Box):
        raise TypeError("DAgger requires Box observation and action spaces")
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    noise_scale = np.asarray(config.noise_sigma, dtype=np.float64)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[bool] = []
    scenario_keys: list[str] = []
    scenario_families: list[str] = []
    episode_ids: list[int] = []
    episode_steps: list[int] = []
    privileged_features: list[np.ndarray] = []
    next_privileged_features: list[np.ndarray] = []
    bc_actions: list[np.ndarray] = []
    correction_l2: list[float] = []
    statistics: dict[str, dict[str, Any]] = {}
    episode_id = 0
    selected_total = 0

    for scenario_key in config.scenarios:
        scenario_stats: dict[str, Any] = {
            "episodes": 0,
            "successes": 0,
            "collisions": 0,
            "out_of_bounds": 0,
            "timeouts": 0,
            "visited_transitions": 0,
            "selected_transitions": 0,
        }
        for _ in config.seeds:
            observation, info = env.reset(
                seed=config.seed_base + episode_id,
                options={"scenario_key": scenario_key, "jitter": config.jitter},
            )
            policy.begin_episode(episode_id)
            episode_rows: list[dict[str, Any]] = []
            terminated = False
            truncated = False
            step = 0
            while not terminated and not truncated:
                scene = env.privileged_scene_state()
                privileged = privileged_critic_features(scene)
                bc_action = np.asarray(policy.act(observation, evaluate=True), dtype=np.float64)
                noisy_action = np.clip(
                    bc_action + rng.normal(0.0, noise_scale),
                    low,
                    high,
                )
                teacher_action = np.asarray(teacher.act(scene), dtype=np.float64)
                delta = float(np.linalg.norm(teacher_action - noisy_action))
                clearance = float(info.get("min_clearance_m", float("inf")))
                selected = bool(
                    delta >= config.correction_threshold
                    or clearance <= config.near_clearance_m
                    or step % config.sample_every_steps == 0
                )
                next_observation, reward, terminated, truncated, next_info = env.step(noisy_action)
                next_privileged = privileged_critic_features(env.privileged_scene_state())
                episode_rows.append(
                    {
                        "observation": np.asarray(observation, dtype=np.float32).copy(),
                        "action": teacher_action.astype(np.float32),
                        "reward": float(reward),
                        "next_observation": np.asarray(next_observation, dtype=np.float32).copy(),
                        "done": bool(terminated or truncated),
                        "step": step,
                        "privileged": privileged,
                        "next_privileged": next_privileged,
                        "bc_action": noisy_action.astype(np.float32),
                        "correction_l2": delta,
                        "selected": selected,
                    }
                )
                observation = next_observation
                info = next_info
                step += 1
            failed = bool(
                info.get("collision", False)
                or info.get("out_of_bounds", False)
                or info.get("timeout", False)
            )
            if failed:
                for row in episode_rows[-config.failure_window_steps :]:
                    row["selected"] = True
            for row in episode_rows:
                if not row["selected"]:
                    continue
                observations.append(row["observation"])
                actions.append(row["action"])
                rewards.append(row["reward"])
                next_observations.append(row["next_observation"])
                dones.append(row["done"])
                scenario_keys.append(scenario_key)
                scenario_families.append(scenario_key)
                episode_ids.append(episode_id)
                episode_steps.append(int(row["step"]))
                privileged_features.append(row["privileged"])
                next_privileged_features.append(row["next_privileged"])
                bc_actions.append(row["bc_action"])
                correction_l2.append(float(row["correction_l2"]))
            scenario_stats["episodes"] += 1
            scenario_stats["visited_transitions"] += len(episode_rows)
            scenario_stats["selected_transitions"] += sum(
                bool(row["selected"]) for row in episode_rows
            )
            scenario_stats["successes"] += int(bool(info.get("success", False)))
            scenario_stats["collisions"] += int(bool(info.get("collision", False)))
            scenario_stats["out_of_bounds"] += int(bool(info.get("out_of_bounds", False)))
            scenario_stats["timeouts"] += int(bool(info.get("timeout", False)))
            episode_id += 1
        statistics[scenario_key] = scenario_stats
        selected_total += int(scenario_stats["selected_transitions"])

    if selected_total > config.max_transitions:
        raise RuntimeError(
            f"DAgger selected {selected_total} transitions, exceeding {config.max_transitions}"
        )
    if not observations:
        raise RuntimeError("DAgger selected no transitions")

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    if observation_shape is None or action_shape is None:
        raise ValueError("environment must expose discrete observation and action shapes")
    demo_path = output / "demos.npz"
    np.savez_compressed(
        demo_path,
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.stack(next_observations).astype(np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        scenario_keys=np.asarray(scenario_keys),
        scenario_families=np.asarray(scenario_families),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        episode_steps=np.asarray(episode_steps, dtype=np.int64),
        teacher_success=np.ones(len(observations), dtype=np.bool_),
        privileged=np.stack(privileged_features).astype(np.float32),
        next_privileged=np.stack(next_privileged_features).astype(np.float32),
        bc_action=np.stack(bc_actions).astype(np.float32),
        correction_l2=np.asarray(correction_l2, dtype=np.float32),
        data_source=np.full(len(observations), "dagger", dtype="<U16"),
    )
    manifest = {
        "schema": "auv_p1_bc_dagger_manifest.v1",
        "demo_path": str(demo_path),
        "demo_sha256": _sha256(demo_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(Path(checkpoint_path)),
        "visited_transitions": sum(item["visited_transitions"] for item in statistics.values()),
        "selected_transitions": len(observations),
        "selected_fraction": len(observations)
        / max(sum(item["visited_transitions"] for item in statistics.values()), 1),
        "mean_correction_l2": float(np.mean(correction_l2)),
        "p95_correction_l2": float(np.percentile(correction_l2, 95)),
        "scenario_statistics": statistics,
        "config": asdict(config),
        "teacher_config": asdict(teacher.config)
        if isinstance(teacher.config, PotentialFieldTeacherConfig)
        else None,
        "note": "Every stored action is a teacher label; failed BC rollouts are used only for DAgger.",
    }
    (output / "dagger_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


__all__ = ["DaggerCollectionConfig", "collect_dagger_demonstrations"]
