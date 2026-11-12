from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.planar_avoidance import PlanarAvoidanceEnv
from ..privileged import PRIVILEGED_FEATURE_DIM, privileged_critic_features
from .teacher import PotentialFieldTeacher, PotentialFieldTeacherConfig


@dataclass(frozen=True)
class DemoCollectionConfig:
    scenarios: tuple[str, ...] = ("s1", "s2")
    successes_per_scenario: int = 30
    success_targets: tuple[tuple[str, int], ...] = ()
    scenario_families: tuple[tuple[str, str], ...] = ()
    max_attempts_per_scenario: int = 100
    max_transitions: int = 300_000
    jitter: bool = True
    seed_base: int = 10_000
    scenario_pool: str = "fixed"
    success_gates: tuple[tuple[str, float], ...] = (("s1", 0.9), ("s2", 0.7))
    family_success_gates: tuple[tuple[str, float], ...] = ()
    max_mean_action_saturation: float = 0.5

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("scenarios must not be empty")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must not contain duplicates")
        if self.successes_per_scenario <= 0:
            raise ValueError("successes_per_scenario must be positive")
        requested_targets = (
            [target for _, target in self.success_targets]
            if self.success_targets
            else [self.successes_per_scenario]
        )
        if self.max_attempts_per_scenario < max(requested_targets):
            raise ValueError("max_attempts_per_scenario must cover the success target")
        if self.max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        if self.seed_base < 0:
            raise ValueError("seed_base must be non-negative")
        if not self.scenario_pool:
            raise ValueError("scenario_pool must not be empty")
        if not 0.0 < self.max_mean_action_saturation <= 1.0:
            raise ValueError("max_mean_action_saturation must be in (0, 1]")
        for key, threshold in self.success_gates:
            if not key:
                raise ValueError("success gate scenario keys must not be empty")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("success gate thresholds must be in [0, 1]")
        for key, threshold in self.family_success_gates:
            if not key:
                raise ValueError("family success gate names must not be empty")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("family success gate thresholds must be in [0, 1]")
        target_keys: set[str] = set()
        for key, target in self.success_targets:
            if key not in self.scenarios:
                raise ValueError(f"success target uses unknown scenario: {key}")
            if key in target_keys:
                raise ValueError(f"duplicate success target for scenario: {key}")
            if target <= 0:
                raise ValueError("success targets must be positive")
            if target > self.max_attempts_per_scenario:
                raise ValueError("max_attempts_per_scenario must cover every success target")
            target_keys.add(key)
        family_keys: set[str] = set()
        for key, family in self.scenario_families:
            if key not in self.scenarios:
                raise ValueError(f"scenario family uses unknown scenario: {key}")
            if key in family_keys:
                raise ValueError(f"duplicate scenario family for scenario: {key}")
            if not family:
                raise ValueError("scenario family names must not be empty")
            family_keys.add(key)

    def success_target(self, scenario_key: str) -> int:
        """Return the success quota for one scenario."""

        for key, target in self.success_targets:
            if key == scenario_key:
                return int(target)
        return int(self.successes_per_scenario)

    def family_for(self, scenario_key: str) -> str:
        """Return the scenario family, defaulting to the scenario key."""

        for key, family in self.scenario_families:
            if key == scenario_key:
                return family
        return scenario_key


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_demonstrations(
    env: PlanarAvoidanceEnv,
    teacher: PotentialFieldTeacher,
    config: DemoCollectionConfig,
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[bool] = []
    scenario_keys: list[str] = []
    scenario_family_keys: list[str] = []
    episode_ids: list[int] = []
    episode_steps: list[int] = []
    teacher_successes: list[bool] = []
    privileged_features: list[np.ndarray] = []
    next_privileged_features: list[np.ndarray] = []
    transition_count = 0
    limit_hit = False
    episode_id = 0
    statistics: dict[str, dict[str, Any]] = {}
    for scenario_key in config.scenarios:
        if limit_hit:
            break
        success_target = config.success_target(scenario_key)
        scenario_family = config.family_for(scenario_key)
        successes = 0
        attempts = 0
        collisions = 0
        out_of_bounds = 0
        timeouts = 0
        saturation_u: list[float] = []
        saturation_yaw: list[float] = []
        min_clearances: list[float] = []
        while successes < success_target and attempts < config.max_attempts_per_scenario:
            observation, _ = env.reset(
                seed=config.seed_base + episode_id,
                options={"scenario_key": scenario_key, "jitter": config.jitter},
            )
            episode_rows: list[
                tuple[
                    np.ndarray,
                    np.ndarray,
                    float,
                    np.ndarray,
                    bool,
                    int,
                    np.ndarray,
                    np.ndarray,
                ]
            ] = []
            terminated = False
            truncated = False
            step = 0
            info: dict[str, Any] = {}
            while not terminated and not truncated:
                privileged = privileged_critic_features(env.privileged_scene_state())
                action = teacher.act(env.privileged_scene_state())
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_privileged = privileged_critic_features(env.privileged_scene_state())
                episode_rows.append(
                    (
                        np.asarray(observation, dtype=np.float32).copy(),
                        np.asarray(action, dtype=np.float32).copy(),
                        float(reward),
                        np.asarray(next_observation, dtype=np.float32).copy(),
                        bool(terminated or truncated),
                        step,
                        privileged,
                        next_privileged,
                    )
                )
                observation = next_observation
                step += 1
            success = bool(info.get("success", False))
            saturation_u.append(float(info.get("action_u_dot_saturation_ratio", 0.0)))
            saturation_yaw.append(float(info.get("action_yaw_rate_saturation_ratio", 0.0)))
            min_clearances.append(float(info.get("min_clearance_m", float("inf"))))
            if success:
                if transition_count + len(episode_rows) > config.max_transitions:
                    limit_hit = True
                    break
                for row in episode_rows:
                    observations.append(row[0])
                    actions.append(row[1])
                    rewards.append(row[2])
                    next_observations.append(row[3])
                    dones.append(row[4])
                    scenario_keys.append(scenario_key)
                    scenario_family_keys.append(scenario_family)
                    episode_ids.append(episode_id)
                    episode_steps.append(row[5])
                    teacher_successes.append(True)
                    privileged_features.append(row[6])
                    next_privileged_features.append(row[7])
                successes += 1
                transition_count += len(episode_rows)
            elif bool(info.get("collision", False)):
                collisions += 1
            elif bool(info.get("out_of_bounds", False)):
                out_of_bounds += 1
            elif bool(info.get("timeout", False)):
                timeouts += 1
            attempts += 1
            episode_id += 1
        statistics[scenario_key] = {
            "family": scenario_family,
            "success_target": success_target,
            "successes": successes,
            "attempts": attempts,
            "collisions": collisions,
            "out_of_bounds": out_of_bounds,
            "timeouts": timeouts,
            "success_rate": successes / attempts if attempts else 0.0,
            "mean_u_dot_saturation": float(np.mean(saturation_u)) if saturation_u else 0.0,
            "mean_yaw_rate_saturation": (float(np.mean(saturation_yaw)) if saturation_yaw else 0.0),
            "max_u_dot_saturation": float(np.max(saturation_u)) if saturation_u else 0.0,
            "max_yaw_rate_saturation": (float(np.max(saturation_yaw)) if saturation_yaw else 0.0),
            "min_clearance_m": min(min_clearances) if min_clearances else None,
        }

    demo_path = output / "demos.npz"
    observation_shape = env.observation_space.shape
    if observation_shape is None:
        raise ValueError("environment observation space must have a shape")
    observation_dim = int(observation_shape[-1])
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("environment action space must have a shape")
    action_dim = int(action_shape[-1])
    observations_array = (
        np.stack(observations, axis=0)
        if observations
        else np.empty((0, observation_dim), dtype=np.float32)
    )
    actions_array = (
        np.stack(actions, axis=0) if actions else np.empty((0, action_dim), dtype=np.float32)
    )
    next_observations_array = (
        np.stack(next_observations, axis=0)
        if next_observations
        else np.empty_like(observations_array)
    )
    privileged_array = (
        np.stack(privileged_features, axis=0)
        if privileged_features
        else np.empty((0, PRIVILEGED_FEATURE_DIM), dtype=np.float32)
    )
    next_privileged_array = (
        np.stack(next_privileged_features, axis=0)
        if next_privileged_features
        else np.empty((0, PRIVILEGED_FEATURE_DIM), dtype=np.float32)
    )
    np.savez_compressed(
        demo_path,
        observations=observations_array,
        actions=actions_array,
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=next_observations_array,
        dones=np.asarray(dones, dtype=np.bool_),
        scenario_keys=np.asarray(scenario_keys),
        scenario_families=np.asarray(scenario_family_keys),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        episode_steps=np.asarray(episode_steps, dtype=np.int64),
        teacher_success=np.asarray(teacher_successes, dtype=np.bool_),
        privileged=privileged_array,
        next_privileged=next_privileged_array,
    )
    success_gates = {key: threshold for key, threshold in config.success_gates}
    family_success_gates = {key: threshold for key, threshold in config.family_success_gates}
    gate_results = {
        key: bool(
            key in statistics and statistics[key]["success_rate"] >= success_gates.get(key, 0.0)
        )
        for key in config.scenarios
    }
    saturation_gate = all(
        statistics.get(key, {}).get("mean_u_dot_saturation", 1.0)
        < config.max_mean_action_saturation
        and statistics.get(key, {}).get("mean_yaw_rate_saturation", 1.0)
        < config.max_mean_action_saturation
        for key in config.scenarios
    )
    count_gate = all(
        statistics.get(key, {}).get("successes", 0) >= config.success_target(key)
        for key in config.scenarios
    )
    family_statistics: dict[str, dict[str, Any]] = {}
    for family in sorted({stats["family"] for stats in statistics.values()}):
        family_scenarios = [
            stats for stats in statistics.values() if str(stats["family"]) == family
        ]
        family_successes = sum(int(stats["successes"]) for stats in family_scenarios)
        family_attempts = sum(int(stats["attempts"]) for stats in family_scenarios)
        family_statistics[family] = {
            "scenarios": len(family_scenarios),
            "successes": family_successes,
            "attempts": family_attempts,
            "success_rate": family_successes / family_attempts if family_attempts else 0.0,
        }
    family_gate_results = {
        family: bool(
            family in family_statistics and family_statistics[family]["success_rate"] >= threshold
        )
        for family, threshold in family_success_gates.items()
    }
    data_gate = bool(
        len(observations) == len(actions) == len(next_observations)
        and np.all(np.isfinite(observations_array))
        and np.all(np.isfinite(actions_array))
    )
    passed = bool(
        count_gate
        and all(gate_results.values())
        and all(family_gate_results.values())
        and saturation_gate
        and data_gate
        and not limit_hit
    )
    manifest = {
        "schema": "auv_p1_demo_manifest.v2",
        "demo_path": str(demo_path),
        "demo_sha256": _sha256(demo_path),
        "transitions": len(observations),
        "episodes_used": len(set(episode_ids)),
        "scenario_statistics": statistics,
        "scenario_family_statistics": family_statistics,
        "config": {
            "scenarios": list(config.scenarios),
            "successes_per_scenario": config.successes_per_scenario,
            "success_targets": dict(config.success_targets),
            "scenario_families": dict(config.scenario_families),
            "max_attempts_per_scenario": config.max_attempts_per_scenario,
            "max_transitions": config.max_transitions,
            "jitter": config.jitter,
            "seed_base": config.seed_base,
            "scenario_pool": config.scenario_pool,
            "success_gates": success_gates,
            "family_success_gates": family_success_gates,
            "max_mean_action_saturation": config.max_mean_action_saturation,
        },
        "teacher_config": asdict(teacher.config)
        if isinstance(teacher.config, PotentialFieldTeacherConfig)
        else None,
        "observation_spec": env.builder.spec.name,
        "observation_source": env.perception.observation_source,
        "perception_mode": env.perception.perception_mode,
        "observation_version": env.builder.spec.version,
        "sector_range_scale_m": env.builder.max_range_m,
        "action_bounds": {
            "low": [-float(env.limits.max_accel_mps2), -float(env.limits.max_yaw_rate_rps)],
            "high": [float(env.limits.max_accel_mps2), float(env.limits.max_yaw_rate_rps)],
        },
        "gates": {
            "success_count": count_gate,
            "success_rate": gate_results,
            "family_success_rate": family_gate_results,
            "action_saturation": saturation_gate,
            "data_finite": data_gate,
            "transition_limit_not_hit": not limit_hit,
            "passed": passed,
        },
        "teacher_used": True,
        "bc_pretrained": True,
        "privileged_critic": False,
        "oracle_in_actor": False,
        "note": "Only successful teacher episodes are stored for BC; failures are counted separately.",
    }
    (output / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(
            f"demo collection gates failed: {json.dumps(manifest['gates'], ensure_ascii=False)}"
        )
    return manifest


__all__ = ["DemoCollectionConfig", "collect_demonstrations"]
