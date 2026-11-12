from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "auv_ppo_isaaclab.v1"
OBSERVATION_NAME = "forward13_target_u_r"
OBSERVATION_VERSION = "2.0-120m"
KNOWN_SCENARIOS = ("s1", "s2", "c1", "c2", "c3")
ALLOWED_OVERRIDES = ("num_envs", "seed", "output", "max_iterations", "device", "resume")


class ConfigError(ValueError):
    """Raised when a PPO experiment config is missing, malformed, or inconsistent."""


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _strict_keys(data: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    keys = set(data)
    expected = set(allowed)
    unknown = sorted(keys - expected)
    missing = sorted(expected - keys)
    if unknown:
        raise ConfigError(f"{where}: unknown keys: {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"{where}: missing keys: {', '.join(missing)}")


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{where}: must be finite")
    return result


def _float(
    value: Any,
    where: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    exclusive_maximum: bool = False,
) -> float:
    result = _number(value, where)
    if minimum is not None:
        if exclusive_minimum and not result > minimum:
            raise ConfigError(f"{where}: must be > {minimum}")
        if not exclusive_minimum and result < minimum:
            raise ConfigError(f"{where}: must be >= {minimum}")
    if maximum is not None:
        if exclusive_maximum and not result < maximum:
            raise ConfigError(f"{where}: must be < {maximum}")
        if not exclusive_maximum and result > maximum:
            raise ConfigError(f"{where}: must be <= {maximum}")
    return result


def _int(value: Any, where: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: expected an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}: must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{where}: must be <= {maximum}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected a boolean, got {type(value).__name__}")
    return value


def _string(value: Any, where: str, *, non_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where}: expected a string, got {type(value).__name__}")
    if non_empty and not value.strip():
        raise ConfigError(f"{where}: must not be empty")
    return value


def _float_list(value: Any, where: str, *, length: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}: expected an array")
    if length is not None and len(value) != length:
        raise ConfigError(f"{where}: expected {length} items, got {len(value)}")
    return tuple(_number(item, f"{where}[{index}]") for index, item in enumerate(value))


def _int_list(value: Any, where: str, *, non_empty: bool = True) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}: expected an array")
    if non_empty and not value:
        raise ConfigError(f"{where}: must not be empty")
    return tuple(_int(item, f"{where}[{index}]", minimum=1) for index, item in enumerate(value))


def _string_list(value: Any, where: str, *, non_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}: expected an array")
    if non_empty and not value:
        raise ConfigError(f"{where}: must not be empty")
    return tuple(_string(item, f"{where}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class ExperimentSection:
    name: str
    seed: int
    output: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> ExperimentSection:
        data = _object(data, where)
        _strict_keys(data, ("name", "seed", "output"), where)
        return cls(
            name=_string(data["name"], f"{where}.name"),
            seed=_int(data["seed"], f"{where}.seed", minimum=0),
            output=_string(data["output"], f"{where}.output"),
        )


@dataclass(frozen=True)
class EnvSection:
    num_envs: int
    dt_s: float
    integration_step_s: float
    max_episode_steps: int
    sensor_max_range_m: float
    local_effective_range_m: float
    goal_radius_m: float
    action_max_accel_mps2: float
    action_max_yaw_rate_rps: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> EnvSection:
        data = _object(data, where)
        _strict_keys(data, tuple(field.name for field in fields(cls)), where)
        return cls(
            num_envs=_int(data["num_envs"], f"{where}.num_envs", minimum=1, maximum=8192),
            dt_s=_float(data["dt_s"], f"{where}.dt_s", minimum=0.0, exclusive_minimum=True),
            integration_step_s=_float(
                data["integration_step_s"],
                f"{where}.integration_step_s",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            max_episode_steps=_int(
                data["max_episode_steps"],
                f"{where}.max_episode_steps",
                minimum=1,
                maximum=100000,
            ),
            sensor_max_range_m=_float(
                data["sensor_max_range_m"],
                f"{where}.sensor_max_range_m",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            local_effective_range_m=_float(
                data["local_effective_range_m"],
                f"{where}.local_effective_range_m",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            goal_radius_m=_float(
                data["goal_radius_m"],
                f"{where}.goal_radius_m",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            action_max_accel_mps2=_float(
                data["action_max_accel_mps2"],
                f"{where}.action_max_accel_mps2",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            action_max_yaw_rate_rps=_float(
                data["action_max_yaw_rate_rps"],
                f"{where}.action_max_yaw_rate_rps",
                minimum=0.0,
                exclusive_minimum=True,
            ),
        )


@dataclass(frozen=True)
class ObservationSection:
    name: str
    version: str
    history_length: int
    sector_count: int
    field_of_view_rad: float
    ray_count: int
    min_range_m: float
    origin_body_m: tuple[float, float, float]
    vertical_half_width_m: float
    sector_width_deg: float
    normalize: bool
    normalization_max_range_m: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> ObservationSection:
        data = _object(data, where)
        _strict_keys(data, tuple(field.name for field in fields(cls)), where)
        origin = _float_list(data["origin_body_m"], f"{where}.origin_body_m", length=3)
        return cls(
            name=_string(data["name"], f"{where}.name"),
            version=_string(data["version"], f"{where}.version"),
            history_length=_int(
                data["history_length"], f"{where}.history_length", minimum=1, maximum=64
            ),
            sector_count=_int(data["sector_count"], f"{where}.sector_count", minimum=1),
            field_of_view_rad=_float(
                data["field_of_view_rad"],
                f"{where}.field_of_view_rad",
                minimum=0.0,
                maximum=math.pi,
                exclusive_minimum=True,
            ),
            ray_count=_int(data["ray_count"], f"{where}.ray_count", minimum=1, maximum=10000),
            min_range_m=_float(
                data["min_range_m"], f"{where}.min_range_m", minimum=0.0, exclusive_minimum=True
            ),
            origin_body_m=(origin[0], origin[1], origin[2]),
            vertical_half_width_m=_float(
                data["vertical_half_width_m"],
                f"{where}.vertical_half_width_m",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            sector_width_deg=_float(
                data["sector_width_deg"],
                f"{where}.sector_width_deg",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            normalize=_bool(data["normalize"], f"{where}.normalize"),
            normalization_max_range_m=_float(
                data["normalization_max_range_m"],
                f"{where}.normalization_max_range_m",
                minimum=0.0,
                exclusive_minimum=True,
            ),
        )


@dataclass(frozen=True)
class RewardSection:
    version: str
    distance_penalty_factor: float
    action_penalty_factor: float
    near_obstacle_penalty: float
    far_obstacle_penalty: float
    collision_penalty: float
    timeout_penalty: float
    goal_bonus: float
    reward_obstacle_range_m: float
    obstacle_influence_fraction: float
    reward_step_scale: float
    ppo_reward_scale: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> RewardSection:
        data = _object(data, where)
        _strict_keys(data, tuple(field.name for field in fields(cls)), where)
        return cls(
            version=_string(data["version"], f"{where}.version"),
            distance_penalty_factor=_float(
                data["distance_penalty_factor"],
                f"{where}.distance_penalty_factor",
                maximum=0.0,
            ),
            action_penalty_factor=_float(
                data["action_penalty_factor"],
                f"{where}.action_penalty_factor",
                maximum=0.0,
            ),
            near_obstacle_penalty=_float(
                data["near_obstacle_penalty"], f"{where}.near_obstacle_penalty", maximum=0.0
            ),
            far_obstacle_penalty=_float(
                data["far_obstacle_penalty"], f"{where}.far_obstacle_penalty", maximum=0.0
            ),
            collision_penalty=_float(
                data["collision_penalty"],
                f"{where}.collision_penalty",
                maximum=0.0,
                exclusive_maximum=True,
            ),
            timeout_penalty=_float(
                data["timeout_penalty"],
                f"{where}.timeout_penalty",
                maximum=0.0,
            ),
            goal_bonus=_float(
                data["goal_bonus"],
                f"{where}.goal_bonus",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            reward_obstacle_range_m=_float(
                data["reward_obstacle_range_m"],
                f"{where}.reward_obstacle_range_m",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            obstacle_influence_fraction=_float(
                data["obstacle_influence_fraction"],
                f"{where}.obstacle_influence_fraction",
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
                exclusive_maximum=True,
            ),
            reward_step_scale=_float(
                data["reward_step_scale"],
                f"{where}.reward_step_scale",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            ppo_reward_scale=_float(
                data["ppo_reward_scale"],
                f"{where}.ppo_reward_scale",
                minimum=0.0,
                exclusive_minimum=True,
            ),
        )


@dataclass(frozen=True)
class CurriculumSection:
    stage: str
    scenario_mix: dict[str, float]
    start_jitter_xy_m: float
    start_jitter_yaw_rad: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> CurriculumSection:
        data = _object(data, where)
        _strict_keys(data, tuple(field.name for field in fields(cls)), where)
        mix = _object(data["scenario_mix"], f"{where}.scenario_mix")
        _strict_keys(mix, KNOWN_SCENARIOS, f"{where}.scenario_mix")
        scenario_mix = {
            key: _float(mix[key], f"{where}.scenario_mix.{key}", minimum=0.0, maximum=1.0)
            for key in KNOWN_SCENARIOS
        }
        return cls(
            stage=_string(data["stage"], f"{where}.stage"),
            scenario_mix=scenario_mix,
            start_jitter_xy_m=_float(
                data["start_jitter_xy_m"],
                f"{where}.start_jitter_xy_m",
                minimum=0.0,
            ),
            start_jitter_yaw_rad=_float(
                data["start_jitter_yaw_rad"],
                f"{where}.start_jitter_yaw_rad",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True)
class PpoSection:
    num_steps_per_env: int
    max_iterations: int
    save_interval: int
    learning_rate: float
    gamma: float
    lam: float
    clip_param: float
    entropy_coef: float
    value_loss_coef: float
    num_learning_epochs: int
    num_mini_batches: int
    max_grad_norm: float
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> PpoSection:
        data = _object(data, where)
        _strict_keys(data, tuple(field.name for field in fields(cls)), where)
        return cls(
            num_steps_per_env=_int(
                data["num_steps_per_env"],
                f"{where}.num_steps_per_env",
                minimum=1,
                maximum=4096,
            ),
            max_iterations=_int(
                data["max_iterations"], f"{where}.max_iterations", minimum=1, maximum=1000000
            ),
            save_interval=_int(
                data["save_interval"], f"{where}.save_interval", minimum=1, maximum=1000000
            ),
            learning_rate=_float(
                data["learning_rate"],
                f"{where}.learning_rate",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            gamma=_float(
                data["gamma"],
                f"{where}.gamma",
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
                exclusive_maximum=True,
            ),
            lam=_float(data["lam"], f"{where}.lam", minimum=0.0, maximum=1.0),
            clip_param=_float(
                data["clip_param"],
                f"{where}.clip_param",
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
                exclusive_maximum=True,
            ),
            entropy_coef=_float(data["entropy_coef"], f"{where}.entropy_coef", minimum=0.0),
            value_loss_coef=_float(
                data["value_loss_coef"], f"{where}.value_loss_coef", minimum=0.0
            ),
            num_learning_epochs=_int(
                data["num_learning_epochs"],
                f"{where}.num_learning_epochs",
                minimum=1,
                maximum=100,
            ),
            num_mini_batches=_int(
                data["num_mini_batches"],
                f"{where}.num_mini_batches",
                minimum=1,
                maximum=1024,
            ),
            max_grad_norm=_float(
                data["max_grad_norm"],
                f"{where}.max_grad_norm",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            actor_hidden_dims=_int_list(data["actor_hidden_dims"], f"{where}.actor_hidden_dims"),
            critic_hidden_dims=_int_list(data["critic_hidden_dims"], f"{where}.critic_hidden_dims"),
        )


@dataclass(frozen=True)
class EvaluationSection:
    eval_every_iterations: int
    seeds: tuple[int, ...]
    scenario_keys: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> EvaluationSection:
        data = _object(data, where)
        _strict_keys(data, tuple(field.name for field in fields(cls)), where)
        seeds = tuple(
            _int(item, f"{where}.seeds[{index}]", minimum=0)
            for index, item in enumerate(_list(data["seeds"], f"{where}.seeds"))
        )
        if not seeds:
            raise ConfigError(f"{where}.seeds: must not be empty")
        if len(set(seeds)) != len(seeds):
            raise ConfigError(f"{where}.seeds: values must be unique")
        scenario_keys = _string_list(data["scenario_keys"], f"{where}.scenario_keys")
        unknown = sorted(set(scenario_keys) - set(KNOWN_SCENARIOS))
        if unknown:
            raise ConfigError(f"{where}.scenario_keys: unknown scenarios: {', '.join(unknown)}")
        if len(set(scenario_keys)) != len(scenario_keys):
            raise ConfigError(f"{where}.scenario_keys: values must be unique")
        return cls(
            eval_every_iterations=_int(
                data["eval_every_iterations"],
                f"{where}.eval_every_iterations",
                minimum=1,
                maximum=1000000,
            ),
            seeds=seeds,
            scenario_keys=scenario_keys,
        )


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}: expected an array")
    return list(value)


@dataclass(frozen=True)
class P1AvoidanceEnvCfg:
    schema_version: str
    experiment: ExperimentSection
    env: EnvSection
    observation: ObservationSection
    reward: RewardSection
    curriculum: CurriculumSection
    ppo: PpoSection
    evaluation: EvaluationSection

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "config") -> P1AvoidanceEnvCfg:
        data = _object(data, where)
        _strict_keys(data, tuple(field.name for field in fields(cls)), where)
        schema_version = _string(data["schema_version"], f"{where}.schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ConfigError(
                f"{where}.schema_version: expected {SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        cfg = cls(
            schema_version=schema_version,
            experiment=ExperimentSection.from_dict(data["experiment"], f"{where}.experiment"),
            env=EnvSection.from_dict(data["env"], f"{where}.env"),
            observation=ObservationSection.from_dict(data["observation"], f"{where}.observation"),
            reward=RewardSection.from_dict(data["reward"], f"{where}.reward"),
            curriculum=CurriculumSection.from_dict(data["curriculum"], f"{where}.curriculum"),
            ppo=PpoSection.from_dict(data["ppo"], f"{where}.ppo"),
            evaluation=EvaluationSection.from_dict(data["evaluation"], f"{where}.evaluation"),
        )
        cfg._validate_contract()
        return cfg

    @classmethod
    def from_json(
        cls, path: Path | str, overrides: Mapping[str, Any] | None = None
    ) -> P1AvoidanceEnvCfg:
        return load_config(path, overrides).config

    def _validate_contract(self) -> None:
        env = self.env
        obs = self.observation
        reward = self.reward
        if env.integration_step_s > env.dt_s + 1e-12:
            raise ConfigError("env.integration_step_s must not exceed env.dt_s")
        ratio = env.dt_s / env.integration_step_s
        if abs(ratio - round(ratio)) > 1e-9:
            raise ConfigError("env.dt_s must be an integer multiple of integration_step_s")
        if env.local_effective_range_m > env.sensor_max_range_m + 1e-9:
            raise ConfigError("env.local_effective_range_m must not exceed sensor range")
        if obs.name != OBSERVATION_NAME:
            raise ConfigError(f"observation.name must be {OBSERVATION_NAME!r} for schema v1")
        if obs.version != OBSERVATION_VERSION:
            raise ConfigError(f"observation.version must be {OBSERVATION_VERSION!r} for schema v1")
        if obs.sector_count != 13:
            raise ConfigError("observation.sector_count must be 13 for schema v1")
        field_of_view_deg = math.degrees(obs.field_of_view_rad)
        expected_fov = obs.sector_count * obs.sector_width_deg
        if abs(field_of_view_deg - expected_fov) > 1e-6:
            raise ConfigError(
                "observation.field_of_view_rad must equal sector_count * sector_width_deg"
            )
        if obs.normalization_max_range_m > env.sensor_max_range_m + 1e-9:
            raise ConfigError("observation normalization range must not exceed sensor range")
        if obs.min_range_m >= obs.normalization_max_range_m:
            raise ConfigError("observation.min_range_m must be below normalization range")
        if reward.version != "paper":
            raise ConfigError("reward.version must be 'paper' for schema v1")
        if reward.reward_obstacle_range_m > env.sensor_max_range_m + 1e-9:
            raise ConfigError("reward.reward_obstacle_range_m must not exceed sensor range")
        expected_step_scale = env.dt_s / 0.2
        if abs(reward.reward_step_scale - expected_step_scale) > 1e-9:
            raise ConfigError(
                "reward.reward_step_scale must equal dt_s / 0.2 for 50Hz time normalization"
            )
        mix_sum = sum(self.curriculum.scenario_mix.values())
        if abs(mix_sum - 1.0) > 1e-6:
            raise ConfigError("curriculum.scenario_mix weights must sum to 1.0")
        rollout = self.rollout_transitions
        if self.ppo.num_mini_batches > rollout:
            raise ConfigError("ppo.num_mini_batches must not exceed rollout transitions")
        if rollout % self.ppo.num_mini_batches != 0:
            raise ConfigError("rollout transitions must be divisible by num_mini_batches")

    @property
    def raw_observation_dim(self) -> int:
        return self.observation.sector_count + 4

    @property
    def policy_input_dim(self) -> int:
        return self.observation.history_length * self.raw_observation_dim

    @property
    def rollout_transitions(self) -> int:
        return self.env.num_envs * self.ppo.num_steps_per_env

    @property
    def minibatch_size(self) -> int:
        return self.rollout_transitions // self.ppo.num_mini_batches

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_sha256(self) -> str:
        return config_sha256(self.as_dict())


def config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_with_extends(
    path: Path, source_files: list[Path], stack: tuple[Path, ...]
) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"extends cycle detected: {chain}")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: top-level JSON value must be an object")
    parent = payload.get("extends")
    if parent is None:
        merged = copy.deepcopy(payload)
    else:
        parent_name = _string(parent, f"{path}: extends")
        parent_path = (path.parent / parent_name).resolve()
        base = _load_with_extends(parent_path, source_files, (*stack, path))
        merged = _deep_merge(base, payload)
        merged.pop("extends", None)
    source_files.append(path)
    return merged


@dataclass(frozen=True)
class ConfigBundle:
    config: P1AvoidanceEnvCfg
    resolved: dict[str, Any]
    source_files: tuple[Path, ...]
    cli_overrides: dict[str, Any]

    @property
    def config_sha256(self) -> str:
        return config_sha256(self.resolved)

    def write(self, output_dir: Path | str) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "resolved_config.json").write_text(
            json.dumps(self.resolved, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": "auv_ppo_isaaclab_config_manifest.v1",
            "config_sha256": self.config_sha256,
            "source_files": [str(path) for path in self.source_files],
            "cli_overrides": self.cli_overrides,
            "derived": {
                "raw_observation_dim": self.config.raw_observation_dim,
                "policy_input_dim": self.config.policy_input_dim,
                "rollout_transitions": self.config.rollout_transitions,
                "minibatch_size": self.config.minibatch_size,
            },
        }
        (root / "config_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def load_config(path: Path | str, overrides: Mapping[str, Any] | None = None) -> ConfigBundle:
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(ALLOWED_OVERRIDES))
    if unknown:
        raise ConfigError(f"unsupported overrides: {', '.join(unknown)}")
    clean = {key: value for key, value in overrides.items() if value is not None}
    source_files: list[Path] = []
    raw = _load_with_extends(Path(path), source_files, ())
    base_cfg = P1AvoidanceEnvCfg.from_dict(raw)
    resolved = base_cfg.as_dict()
    if "num_envs" in clean:
        resolved["env"]["num_envs"] = _int(clean["num_envs"], "override.num_envs", minimum=1)
    if "seed" in clean:
        resolved["experiment"]["seed"] = _int(clean["seed"], "override.seed", minimum=0)
    if "output" in clean:
        resolved["experiment"]["output"] = _string(clean["output"], "override.output")
    if "max_iterations" in clean:
        resolved["ppo"]["max_iterations"] = _int(
            clean["max_iterations"], "override.max_iterations", minimum=1
        )
    cfg = P1AvoidanceEnvCfg.from_dict(resolved)
    return ConfigBundle(
        config=cfg,
        resolved=cfg.as_dict(),
        source_files=tuple(source_files),
        cli_overrides=clean,
    )


__all__ = [
    "ALLOWED_OVERRIDES",
    "ConfigBundle",
    "ConfigError",
    "CurriculumSection",
    "EnvSection",
    "EvaluationSection",
    "ExperimentSection",
    "KNOWN_SCENARIOS",
    "ObservationSection",
    "OBSERVATION_NAME",
    "OBSERVATION_VERSION",
    "P1AvoidanceEnvCfg",
    "PpoSection",
    "RewardSection",
    "SCHEMA_VERSION",
    "config_sha256",
    "load_config",
]
