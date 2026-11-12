"""Deterministic analytic evaluation for P1 PPO checkpoints.

This module deliberately depends only on the torch batch core.  It does not
start Isaac Sim or Omniverse, so checkpoint evaluation can run in the same
IsaacLab Python environment without creating another simulation process.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from auv_isaaclab.config import KNOWN_SCENARIOS, ConfigError, P1AvoidanceEnvCfg
from auv_isaaclab.tasks.p1_avoidance.batch import P1BatchCore
from auv_isaaclab.tasks.p1_avoidance.scenarios import build_scenario_groups

EVAL_SCHEMA_VERSION = "auv_ppo_isaaclab.eval.v1"
EVAL_SUMMARY_SCHEMA = "auv_p1_ppo_eval.v1"
REQUIRED_METRICS = (
    "success",
    "collision",
    "out_of_bounds",
    "timeout",
    "min_clearance_m",
    "time_to_goal_s",
    "path_length_m",
    "action_saturation",
)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _strict_keys(data: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    missing = sorted(set(allowed) - set(data))
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


def _float(value: Any, where: str, *, minimum: float | None = None) -> float:
    result = _number(value, where)
    if minimum is not None and result < minimum:
        raise ConfigError(f"{where}: must be >= {minimum}")
    return result


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: expected an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}: must be >= {minimum}")
    return value


def _vector(value: Any, where: str, length: int = 3) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ConfigError(f"{where}: expected an array of {length} numbers")
    return tuple(_number(item, f"{where}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


@dataclass(frozen=True)
class ObstacleSpec:
    kind: str
    center_xyz_m: tuple[float, float, float]
    radius_m: float | None = None
    height_m: float | None = None
    half_extents_m: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> ObstacleSpec:
        data = _object(data, where)
        if "type" not in data:
            raise ConfigError(f"{where}: missing keys: type")
        kind = data["type"]
        if kind not in ("cylinder", "box", "sphere"):
            raise ConfigError(f"{where}.type: expected cylinder, box or sphere")
        center = _vector(data.get("center_xyz_m"), f"{where}.center_xyz_m")
        if kind == "cylinder":
            _strict_keys(data, ("type", "center_xyz_m", "radius_m", "height_m"), where)
            return cls(
                kind=kind,
                center_xyz_m=center,
                radius_m=_float(data["radius_m"], f"{where}.radius_m", minimum=0.0),
                height_m=_float(data["height_m"], f"{where}.height_m", minimum=0.0),
            )
        if kind == "sphere":
            _strict_keys(data, ("type", "center_xyz_m", "radius_m"), where)
            return cls(
                kind=kind,
                center_xyz_m=center,
                radius_m=_float(data["radius_m"], f"{where}.radius_m", minimum=0.0),
            )
        _strict_keys(data, ("type", "center_xyz_m", "half_extents_m"), where)
        return cls(
            kind=kind,
            center_xyz_m=center,
            half_extents_m=_vector(data["half_extents_m"], f"{where}.half_extents_m"),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.kind, "center_xyz_m": list(self.center_xyz_m)}
        if self.radius_m is not None:
            payload["radius_m"] = self.radius_m
        if self.height_m is not None:
            payload["height_m"] = self.height_m
        if self.half_extents_m is not None:
            payload["half_extents_m"] = list(self.half_extents_m)
        return payload


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    start_xyz_m: tuple[float, float, float]
    goal_xyz_m: tuple[float, float, float]
    obstacles: tuple[ObstacleSpec, ...]
    variant: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> ScenarioSpec:
        data = _object(data, where)
        required = ("key", "start_xyz_m", "goal_xyz_m", "obstacles")
        allowed = (*required, "variant")
        unknown = sorted(set(data) - set(allowed))
        missing = sorted(set(required) - set(data))
        if unknown:
            raise ConfigError(f"{where}: unknown keys: {', '.join(unknown)}")
        if missing:
            raise ConfigError(f"{where}: missing keys: {', '.join(missing)}")
        key = data["key"]
        if not isinstance(key, str) or key not in KNOWN_SCENARIOS:
            raise ConfigError(f"{where}.key: unknown scenario key {key!r}")
        variant = data.get("variant")
        if variant is not None and (not isinstance(variant, str) or not variant.strip()):
            raise ConfigError(f"{where}.variant: expected a non-empty string")
        raw_obstacles = data["obstacles"]
        if not isinstance(raw_obstacles, list):
            raise ConfigError(f"{where}.obstacles: expected an array")
        return cls(
            key=key,
            start_xyz_m=_vector(data["start_xyz_m"], f"{where}.start_xyz_m"),
            goal_xyz_m=_vector(data["goal_xyz_m"], f"{where}.goal_xyz_m"),
            obstacles=tuple(
                ObstacleSpec.from_dict(item, f"{where}.obstacles[{index}]")
                for index, item in enumerate(raw_obstacles)
            ),
            variant=variant,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "key": self.key,
            "start_xyz_m": list(self.start_xyz_m),
            "goal_xyz_m": list(self.goal_xyz_m),
            "obstacles": [item.as_dict() for item in self.obstacles],
        }
        if self.variant is not None:
            payload["variant"] = self.variant
        return payload


@dataclass(frozen=True)
class SuccessGates:
    episodes_per_scenario: int
    s1_min_successes: int
    s2_min_successes: int
    max_action_saturation: float
    max_both_action_saturation: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> SuccessGates:
        data = _object(data, where)
        _strict_keys(
            data,
            (
                "episodes_per_scenario",
                "s1_min_successes",
                "s2_min_successes",
                "max_action_saturation",
                "max_both_action_saturation",
            ),
            where,
        )
        return cls(
            episodes_per_scenario=_int(
                data["episodes_per_scenario"], f"{where}.episodes_per_scenario", minimum=1
            ),
            s1_min_successes=_int(data["s1_min_successes"], f"{where}.s1_min_successes", minimum=0),
            s2_min_successes=_int(data["s2_min_successes"], f"{where}.s2_min_successes", minimum=0),
            max_action_saturation=_float(
                data["max_action_saturation"],
                f"{where}.max_action_saturation",
                minimum=0.0,
            ),
            max_both_action_saturation=_float(
                data["max_both_action_saturation"],
                f"{where}.max_both_action_saturation",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True)
class EvalSpec:
    source_path: Path
    base_config: Path
    checkpoint: Path | None
    seeds: tuple[int, ...]
    scenarios: tuple[ScenarioSpec, ...]
    success_gates: SuccessGates
    metrics: tuple[str, ...]
    output: str
    start_jitter_xy_m: float
    start_jitter_yaw_rad: float

    @property
    def episode_count(self) -> int:
        return len(self.seeds) * len(self.scenarios)

    @property
    def has_jitter(self) -> bool:
        return self.start_jitter_xy_m > 0.0 or self.start_jitter_yaw_rad > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVAL_SCHEMA_VERSION,
            "base_config": str(self.base_config),
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "seeds": list(self.seeds),
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
            "success_gates": {
                "episodes_per_scenario": self.success_gates.episodes_per_scenario,
                "s1_min_successes": self.success_gates.s1_min_successes,
                "s2_min_successes": self.success_gates.s2_min_successes,
                "max_action_saturation": self.success_gates.max_action_saturation,
                "max_both_action_saturation": self.success_gates.max_both_action_saturation,
            },
            "metrics": list(self.metrics),
            "output": self.output,
            "start_jitter_xy_m": self.start_jitter_xy_m,
            "start_jitter_yaw_rad": self.start_jitter_yaw_rad,
        }


def load_eval_spec(path: Path | str) -> EvalSpec:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"eval config not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{source}: invalid JSON: {exc}") from exc
    data = _object(payload, str(source))
    _strict_keys(
        data,
        (
            "schema_version",
            "base_config",
            "checkpoint",
            "seeds",
            "scenarios",
            "success_gates",
            "metrics",
            "output",
            "start_jitter_xy_m",
            "start_jitter_yaw_rad",
        ),
        str(source),
    )
    if data["schema_version"] != EVAL_SCHEMA_VERSION:
        raise ConfigError(f"{source}.schema_version: expected {EVAL_SCHEMA_VERSION!r}")
    base_name = data["base_config"]
    if not isinstance(base_name, str) or not base_name.strip():
        raise ConfigError(f"{source}.base_config: expected a non-empty string")
    base_config = (source.parent / base_name).resolve()
    checkpoint_value = data["checkpoint"]
    if checkpoint_value is None:
        checkpoint = None
    elif isinstance(checkpoint_value, str) and checkpoint_value.strip():
        checkpoint = Path(checkpoint_value)
    else:
        raise ConfigError(f"{source}.checkpoint: expected a string or null")
    raw_seeds = data["seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ConfigError(f"{source}.seeds: expected a non-empty array")
    seeds = tuple(
        _int(item, f"{source}.seeds[{index}]", minimum=0) for index, item in enumerate(raw_seeds)
    )
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"{source}.seeds: values must be unique")
    raw_scenarios = data["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ConfigError(f"{source}.scenarios: expected a non-empty array")
    scenarios = tuple(
        ScenarioSpec.from_dict(item, f"{source}.scenarios[{index}]")
        for index, item in enumerate(raw_scenarios)
    )
    keys = [(scenario.key, scenario.variant) for scenario in scenarios]
    if len(set(keys)) != len(keys):
        raise ConfigError(f"{source}.scenarios: (key, variant) pairs must be unique")
    gates = SuccessGates.from_dict(data["success_gates"], f"{source}.success_gates")
    if gates.episodes_per_scenario != len(seeds):
        raise ConfigError(f"{source}.success_gates.episodes_per_scenario must equal len(seeds)")
    raw_metrics = data["metrics"]
    if not isinstance(raw_metrics, list) or not all(isinstance(item, str) for item in raw_metrics):
        raise ConfigError(f"{source}.metrics: expected an array of strings")
    missing_metrics = sorted(set(REQUIRED_METRICS) - set(raw_metrics))
    if missing_metrics:
        raise ConfigError(f"{source}.metrics: missing metrics: {', '.join(missing_metrics)}")
    output = data["output"]
    if not isinstance(output, str) or not output.strip():
        raise ConfigError(f"{source}.output: expected a non-empty string")
    return EvalSpec(
        source_path=source,
        base_config=base_config,
        checkpoint=checkpoint,
        seeds=seeds,
        scenarios=scenarios,
        success_gates=gates,
        metrics=tuple(raw_metrics),
        output=output,
        start_jitter_xy_m=_float(
            data["start_jitter_xy_m"], f"{source}.start_jitter_xy_m", minimum=0.0
        ),
        start_jitter_yaw_rad=_float(
            data["start_jitter_yaw_rad"], f"{source}.start_jitter_yaw_rad", minimum=0.0
        ),
    )


def validate_scenarios(spec: EvalSpec) -> None:
    """Check that declared eval geometry matches the batch-core scenarios."""

    groups = build_scenario_groups()
    for scenario in spec.scenarios:
        group = groups[scenario.key]
        if scenario.variant is None:
            if len(group) != 1:
                raise ConfigError(
                    f"eval scenario {scenario.key}: variant is required for a multi-layout group"
                )
            selected = group[0]
        else:
            matches = [case for case in group if case.key == scenario.variant]
            if not matches:
                raise ConfigError(
                    f"eval scenario {scenario.key}: unknown variant {scenario.variant!r}"
                )
            selected = matches[0]
        case = selected.scenario
        label = f"{scenario.key}/{scenario.variant}" if scenario.variant else scenario.key
        if any(
            abs(float(a) - float(b)) > 1e-6
            for a, b in zip(scenario.start_xyz_m, case.start.position_ned_m, strict=True)
        ):
            raise ConfigError(f"eval scenario {label}: start does not match core")
        if any(
            abs(float(a) - float(b)) > 1e-6
            for a, b in zip(scenario.goal_xyz_m, case.goal_ned_m, strict=True)
        ):
            raise ConfigError(f"eval scenario {label}: goal does not match core")
        core_obstacles = sorted(
            case.world.obstacle_map.obstacles,
            key=lambda item: (item.kind, tuple(item.center_ned_m)),
        )
        declared = sorted(scenario.obstacles, key=lambda item: (item.kind, item.center_xyz_m))
        if len(declared) != len(core_obstacles):
            raise ConfigError(f"eval scenario {label}: obstacle count does not match core")
        for index, (left, right) in enumerate(zip(declared, core_obstacles, strict=True)):
            if left.kind != right.kind:
                raise ConfigError(f"eval scenario {label}: obstacle {index} type mismatch")
            if any(
                abs(a - b) > 1e-6
                for a, b in zip(left.center_xyz_m, right.center_ned_m, strict=True)
            ):
                raise ConfigError(f"eval scenario {label}: obstacle {index} center mismatch")
            if left.kind == "cylinder":
                if (
                    left.radius_m is None
                    or right.radius_m is None
                    or abs(left.radius_m - float(right.radius_m)) > 1e-6
                    or left.height_m is None
                    or right.height_m is None
                    or abs(left.height_m - float(right.height_m)) > 1e-6
                ):
                    raise ConfigError(f"eval scenario {label}: obstacle {index} cylinder mismatch")
            elif left.kind == "box":
                if left.half_extents_m is None or right.half_extents_m is None:
                    raise ConfigError(f"eval scenario {label}: obstacle {index} box mismatch")
                if any(
                    abs(a - float(b)) > 1e-6
                    for a, b in zip(left.half_extents_m, right.half_extents_m, strict=True)
                ):
                    raise ConfigError(f"eval scenario {label}: obstacle {index} box mismatch")
            elif left.kind == "sphere":
                if (
                    left.radius_m is None
                    or right.radius_m is None
                    or abs(left.radius_m - float(right.radius_m)) > 1e-6
                ):
                    raise ConfigError(f"eval scenario {label}: obstacle {index} sphere mismatch")


def build_eval_cfg(base_cfg: P1AvoidanceEnvCfg, spec: EvalSpec) -> P1AvoidanceEnvCfg:
    validate_scenarios(spec)
    env = replace(base_cfg.env, num_envs=spec.episode_count)
    curriculum = replace(
        base_cfg.curriculum,
        start_jitter_xy_m=spec.start_jitter_xy_m,
        start_jitter_yaw_rad=spec.start_jitter_yaw_rad,
    )
    return replace(base_cfg, env=env, curriculum=curriculum)


def _new_episode(
    scenario: ScenarioSpec,
    seed: int,
    index: int,
) -> dict[str, Any]:
    return {
        "index": index,
        "scenario": scenario.key,
        "scenario_variant": scenario.variant,
        "seed": seed,
        "success": False,
        "collision": False,
        "out_of_bounds": False,
        "timeout": False,
        "min_clearance_m": None,
        "time_to_goal_s": None,
        "path_length_m": None,
        "final_goal_distance_m": None,
        "episode_steps": None,
        "action_saturation": 0.0,
        "all_zero_actions": True,
    }


def _finish_episode(
    record: dict[str, Any],
    info: Mapping[str, Any],
    index: int,
    saturation: float,
    saturation_u_dot: float,
    saturation_yaw: float,
    both_saturation: float,
    all_zero: bool,
) -> None:
    success = bool(info["success"][index])
    record.update(
        {
            "success": success,
            "collision": bool(info["collision"][index]),
            "out_of_bounds": bool(info["out_of_bounds"][index]),
            "timeout": bool(info["timeout"][index]),
            "min_clearance_m": _finite_or_none(float(info["min_clearance_m"][index])),
            "time_to_goal_s": float(info["simulation_time_s"][index]) if success else None,
            "path_length_m": _finite_or_none(float(info["path_length_m"][index])),
            "final_goal_distance_m": _finite_or_none(float(info["goal_distance_m"][index])),
            "episode_steps": int(info["rl_step"][index]),
            "action_saturation": saturation,
            "action_saturation_u_dot": saturation_u_dot,
            "action_saturation_yaw": saturation_yaw,
            "both_action_saturation": both_saturation,
            "all_zero_actions": all_zero,
        }
    )


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _aggregate(
    records: Sequence[Mapping[str, Any]],
    scenario_key: str,
    variant: str | None = None,
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if record["scenario"] == scenario_key
        and (variant is None or record["scenario_variant"] == variant)
    ]
    count = len(selected)
    successes = sum(bool(record["success"]) for record in selected)
    collisions = sum(bool(record["collision"]) for record in selected)
    out_of_bounds = sum(bool(record["out_of_bounds"]) for record in selected)
    timeouts = sum(bool(record["timeout"]) for record in selected)
    times = [
        float(record["time_to_goal_s"])
        for record in selected
        if record["time_to_goal_s"] is not None
    ]
    clearances = [
        float(record["min_clearance_m"])
        for record in selected
        if record["min_clearance_m"] is not None
    ]
    return {
        "scenario": scenario_key,
        "variant": variant,
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count if count else 0.0,
        "collisions": collisions,
        "out_of_bounds": out_of_bounds,
        "timeouts": timeouts,
        "mean_time_to_goal_s": sum(times) / len(times) if times else None,
        "mean_min_clearance_m": sum(clearances) / len(clearances) if clearances else None,
    }


def evaluate_policy(
    policy: Callable[[Tensor], Tensor],
    cfg: P1AvoidanceEnvCfg,
    spec: EvalSpec,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all fixed eval episodes and return a JSON-serializable summary."""

    if cfg.env.num_envs != spec.episode_count:
        raise ValueError(f"eval cfg num_envs must be {spec.episode_count}, got {cfg.env.num_envs}")
    episodes = [(scenario, seed) for scenario in spec.scenarios for seed in spec.seeds]
    core = P1BatchCore(
        cfg,
        device=device,
        dtype=dtype,
        scenario_keys=[scenario.key for scenario, _ in episodes],
        scenario_variant_keys=[
            scenario.variant if scenario.variant is not None else scenario.key
            for scenario, _ in episodes
        ],
        seed=0,
    )
    saturation_sum = torch.zeros(len(episodes), device=core.device, dtype=torch.float64)
    saturation_u_sum = torch.zeros(len(episodes), device=core.device, dtype=torch.float64)
    saturation_yaw_sum = torch.zeros(len(episodes), device=core.device, dtype=torch.float64)
    both_saturation_sum = torch.zeros(len(episodes), device=core.device, dtype=torch.float64)
    action_count = torch.zeros(len(episodes), device=core.device, dtype=torch.float64)
    nonzero_count = torch.zeros(len(episodes), device=core.device, dtype=torch.float64)
    records: list[dict[str, Any]] = []
    for index, (scenario, seed) in enumerate(episodes):
        core.reset(
            torch.tensor([index], device=core.device, dtype=torch.long),
            scenario_keys=scenario.key,
            variant_keys=scenario.variant,
            jitter=spec.has_jitter,
            seed=seed,
        )
        records.append(
            _new_episode(
                scenario,
                seed,
                index,
            )
        )

    active = torch.ones(len(episodes), device=core.device, dtype=torch.bool)
    for _ in range(cfg.env.max_episode_steps):
        if not active.any():
            break
        with torch.inference_mode():
            raw_actions = policy(core.observation)
        if raw_actions.shape != (len(episodes), 2):
            raise ValueError(
                f"policy must return ({len(episodes)}, 2), got {tuple(raw_actions.shape)}"
            )
        if not bool(torch.isfinite(raw_actions).all()):
            raise ValueError("policy returned non-finite actions")
        normalized_actions = torch.clamp(raw_actions.to(device=core.device, dtype=dtype), -1.0, 1.0)
        saturated = normalized_actions.abs() >= 0.99
        active_float = active.to(dtype=torch.float64)
        saturation_sum += saturated.to(dtype=torch.float64).sum(dim=1) * active_float
        saturation_u_sum += saturated[:, 0].to(dtype=torch.float64) * active_float
        saturation_yaw_sum += saturated[:, 1].to(dtype=torch.float64) * active_float
        both_saturation_sum += saturated.all(dim=1).to(dtype=torch.float64) * active_float
        action_count += 2.0 * active_float
        nonzero_count += (normalized_actions.abs() > 1e-8).to(dtype=torch.float64).sum(
            dim=1
        ) * active_float
        action_scale = torch.tensor(
            [cfg.env.action_max_accel_mps2, cfg.env.action_max_yaw_rate_rps],
            device=core.device,
            dtype=dtype,
        )
        step = core.step(normalized_actions * action_scale)
        done = (step.terminated | step.truncated).nonzero(as_tuple=False).flatten()
        if done.numel() == 0:
            continue
        for index in done.tolist():
            if not bool(active[index]):
                continue
            sat = 0.0
            sat_u = 0.0
            sat_yaw = 0.0
            both_sat = 0.0
            if float(action_count[index]) > 0.0:
                sat = float(saturation_sum[index] / action_count[index])
                sat_u = float(saturation_u_sum[index] / (0.5 * action_count[index]))
                sat_yaw = float(saturation_yaw_sum[index] / (0.5 * action_count[index]))
                both_sat = float(both_saturation_sum[index] / (0.5 * action_count[index]))
            all_zero = float(nonzero_count[index]) == 0.0
            _finish_episode(
                records[index], step.info, index, sat, sat_u, sat_yaw, both_sat, all_zero
            )
            active[index] = False
        core.reset(done, jitter=False)

    for index, record in enumerate(records):
        if record["episode_steps"] is None:
            raise RuntimeError(
                f"eval episode {index} ({record['scenario']}, seed {record['seed']}) "
                "did not terminate within max_episode_steps"
            )

    by_scenario = {
        key: _aggregate(records, key)
        for key in dict.fromkeys(scenario.key for scenario in spec.scenarios)
    }
    by_instance = {
        f"{scenario.key}/{scenario.variant or scenario.key}": _aggregate(
            records, scenario.key, scenario.variant
        )
        for scenario in spec.scenarios
    }
    s1 = by_scenario.get("s1", {"successes": 0})
    s2 = by_scenario.get("s2", {"successes": 0})
    gate_records = [record for record in records if record["scenario"] in {"s1", "s2"}] or records
    action_saturation = (
        sum(float(record["action_saturation"]) for record in gate_records) / len(gate_records)
        if gate_records
        else 0.0
    )
    action_saturation_u_dot = (
        sum(float(record["action_saturation_u_dot"]) for record in gate_records) / len(gate_records)
        if gate_records
        else 0.0
    )
    action_saturation_yaw = (
        sum(float(record["action_saturation_yaw"]) for record in gate_records) / len(gate_records)
        if gate_records
        else 0.0
    )
    both_action_saturation = (
        sum(float(record["both_action_saturation"]) for record in gate_records) / len(gate_records)
        if gate_records
        else 0.0
    )
    all_zero = any(bool(record["all_zero_actions"]) for record in gate_records)
    all_scenarios_all_zero = any(bool(record["all_zero_actions"]) for record in records)
    all_scenarios_action_saturation = (
        sum(float(record["action_saturation"]) for record in records) / len(records)
        if records
        else 0.0
    )
    finite = all(
        record["min_clearance_m"] is None or math.isfinite(float(record["min_clearance_m"]))
        for record in records
    ) and all(
        record["path_length_m"] is None or math.isfinite(float(record["path_length_m"]))
        for record in records
    )
    gates = {
        "s1_successes": int(s1["successes"]),
        "s2_successes": int(s2["successes"]),
        "s1_required": spec.success_gates.s1_min_successes,
        "s2_required": spec.success_gates.s2_min_successes,
        "s1_pass": int(s1["successes"]) >= spec.success_gates.s1_min_successes,
        "s2_pass": int(s2["successes"]) >= spec.success_gates.s2_min_successes,
        "action_saturation": action_saturation,
        "action_saturation_u_dot": action_saturation_u_dot,
        "action_saturation_yaw": action_saturation_yaw,
        "both_action_saturation": both_action_saturation,
        "action_saturation_pass": (
            action_saturation_yaw < spec.success_gates.max_action_saturation
            and both_action_saturation < spec.success_gates.max_both_action_saturation
        ),
        "no_all_zero_actions": not all_zero,
        "all_scenarios_action_saturation": all_scenarios_action_saturation,
        "all_scenarios_no_all_zero_actions": not all_scenarios_all_zero,
        "finite_metrics": finite,
    }
    gates["pass"] = bool(
        gates["s1_pass"]
        and gates["s2_pass"]
        and gates["action_saturation_pass"]
        and gates["no_all_zero_actions"]
        and gates["finite_metrics"]
    )
    summary: dict[str, Any] = {
        "schema": EVAL_SUMMARY_SCHEMA,
        "status": "PASS" if gates["pass"] else "FAIL_GATE",
        "eval_spec": spec.as_dict(),
        "environment": {
            "device": str(core.device),
            "dtype": str(dtype),
            "num_envs": len(episodes),
            "dt_s": cfg.env.dt_s,
            "integration_step_s": cfg.env.integration_step_s,
            "max_episode_steps": cfg.env.max_episode_steps,
            "sensor_max_range_m": cfg.env.sensor_max_range_m,
            "policy_input_dim": cfg.policy_input_dim,
            "start_jitter_xy_m": spec.start_jitter_xy_m,
            "start_jitter_yaw_rad": spec.start_jitter_yaw_rad,
        },
        "gates": gates,
        "scenarios": by_scenario,
        "scenario_instances": by_instance,
        "episodes": records,
    }
    if metadata:
        summary.update(metadata)
    return summary


def evaluation_score(summary: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Lexicographic-ish score used only for checkpoint selection."""

    gates = summary["gates"]
    scenarios = summary["scenarios"]
    total_successes = sum(int(item["successes"]) for item in scenarios.values())
    times = [
        float(record["time_to_goal_s"])
        for record in summary["episodes"]
        if record["time_to_goal_s"] is not None
    ]
    clearances = [
        float(record["min_clearance_m"])
        for record in summary["episodes"]
        if record["min_clearance_m"] is not None
    ]
    mean_time = sum(times) / len(times) if times else 1.0e9
    mean_clearance = sum(clearances) / len(clearances) if clearances else -1.0e9
    return (
        1.0 if gates["pass"] else 0.0,
        float(total_successes),
        -mean_time,
        mean_clearance,
    )


def eval_spec_sha256(spec: EvalSpec) -> str:
    return hashlib.sha256(
        json.dumps(spec.as_dict(), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


__all__ = [
    "EVAL_SCHEMA_VERSION",
    "EVAL_SUMMARY_SCHEMA",
    "EvalSpec",
    "ObstacleSpec",
    "ScenarioSpec",
    "SuccessGates",
    "build_eval_cfg",
    "eval_spec_sha256",
    "evaluate_policy",
    "evaluation_score",
    "load_eval_spec",
    "validate_scenarios",
]
