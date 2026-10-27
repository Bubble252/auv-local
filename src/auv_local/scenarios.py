from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from auv_planner.attitude import forward_quaternion
from auv_planner.contracts import (
    CapabilitySet,
    Obstacle,
    ObstacleMap,
    ScenarioConfig,
    State,
    Trajectory,
    UniformCurrent,
    VehicleSpec,
    WorldSnapshot,
)
from auv_planner.scenarios import remus100_vehicle

from .contracts import RolloutConfig

M2_NOMINAL_SPEED_MPS = 1.0
M2_WORLD_BOUNDS_NED_M = np.array(
    [[-10.0, 18.0], [-6.0, 6.0], [-12.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class M2Scenario:
    key: str
    scenario: ScenarioConfig
    nominal_speed_mps: float
    max_duration_s: float


@dataclass(frozen=True)
class M2Config:
    rollout: RolloutConfig
    seeds: tuple[int, ...]
    nominal_speed_mps: float
    max_duration_s: float
    pvs_sample_time_s: float
    oracle_ray_count: int
    oracle_min_range_m: float
    oracle_max_range_m: float
    oracle_validity_s: float
    horizontal_lookahead_m: float
    vertical_lookahead_m: float


def default_m2_config() -> M2Config:
    """Return the frozen M2 configuration.

    The REMUS 100 model declares a 1.0 m/s minimum speed. The earlier draft
    used 0.8 m/s, which the existing controller correctly rejects, so M2 uses
    the model minimum instead of silently weakening the vehicle contract.
    """

    return M2Config(
        rollout=RolloutConfig(
            horizon_s=2.0,
            step_s=0.2,
            replan_period_s=0.5,
            max_path_length_m=5.0,
            model_id="line_detour_v1",
            inherit_depth_from_route=True,
        ),
        seeds=(0, 1, 2),
        nominal_speed_mps=M2_NOMINAL_SPEED_MPS,
        max_duration_s=60.0,
        pvs_sample_time_s=0.02,
        oracle_ray_count=72,
        oracle_min_range_m=0.05,
        oracle_max_range_m=8.0,
        oracle_validity_s=0.5,
        horizontal_lookahead_m=1.5,
        vertical_lookahead_m=1.0,
    )


def load_m2_config(path: Path | str) -> M2Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("M2 config must be a JSON object")
    rollout_payload = payload.get("rollout")
    if not isinstance(rollout_payload, dict):
        raise ValueError("M2 config requires a rollout object")
    default = default_m2_config()
    seeds = tuple(int(seed) for seed in payload.get("seeds", default.seeds))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty sequence of non-negative integers")
    return M2Config(
        rollout=RolloutConfig(**rollout_payload),
        seeds=seeds,
        nominal_speed_mps=float(payload.get("nominal_speed_mps", default.nominal_speed_mps)),
        max_duration_s=float(payload.get("max_duration_s", default.max_duration_s)),
        pvs_sample_time_s=float(payload.get("pvs_sample_time_s", default.pvs_sample_time_s)),
        oracle_ray_count=int(payload.get("oracle_ray_count", default.oracle_ray_count)),
        oracle_min_range_m=float(payload.get("oracle_min_range_m", default.oracle_min_range_m)),
        oracle_max_range_m=float(payload.get("oracle_max_range_m", default.oracle_max_range_m)),
        oracle_validity_s=float(payload.get("oracle_validity_s", default.oracle_validity_s)),
        horizontal_lookahead_m=float(
            payload.get("horizontal_lookahead_m", default.horizontal_lookahead_m)
        ),
        vertical_lookahead_m=float(
            payload.get("vertical_lookahead_m", default.vertical_lookahead_m)
        ),
    )


def m2_world_bounds() -> np.ndarray:
    return M2_WORLD_BOUNDS_NED_M.copy()


def m2_capability() -> CapabilitySet:
    return CapabilitySet(available_actuators=("propeller", "yaw_fin", "pitch_fin"))


def _start_state(position_ned_m: tuple[float, float, float], speed_mps: float) -> State:
    return State(
        t_s=0.0,
        position_ned_m=np.asarray(position_ned_m, dtype=np.float64),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        linear_velocity_body_mps=np.array([speed_mps, 0.0, 0.0], dtype=np.float64),
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )


def _scenario(
    name: str,
    obstacles: tuple[Obstacle, ...],
    *,
    start_ned_m: tuple[float, float, float],
    goal_ned_m: tuple[float, float, float],
    speed_mps: float,
    vehicle: VehicleSpec | None = None,
) -> ScenarioConfig:
    start = _start_state(start_ned_m, speed_mps)
    world = WorldSnapshot(
        t_s=start.t_s,
        obstacle_map=ObstacleMap(m2_world_bounds(), obstacles),
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3, dtype=np.float64)),
    )
    return ScenarioConfig(
        name=name,
        world=world,
        vehicle=remus100_vehicle() if vehicle is None else vehicle,
        start=start,
        goal_ned_m=np.asarray(goal_ned_m, dtype=np.float64),
        goal_radius_m=1.0,
        constraints={"m2_smoke_goal_radius": 1.0},
    )


def m2_scenarios(config: M2Config | None = None) -> tuple[M2Scenario, ...]:
    active_config = default_m2_config() if config is None else config
    start = (-8.0, 0.0, -5.0)
    goal = (15.0, 0.0, -5.0)

    empty = _scenario(
        "m2_s1_empty",
        (),
        start_ned_m=start,
        goal_ned_m=goal,
        speed_mps=active_config.nominal_speed_mps,
    )

    cylinder = Obstacle(
        obstacle_id="m2-s2-cylinder",
        kind="cylinder",
        center_ned_m=np.array([4.0, 0.0, -6.0], dtype=np.float64),
        radius_m=1.0,
        height_m=12.0,
    )
    single_obstacle = _scenario(
        "m2_s2_single_cylinder",
        (cylinder,),
        start_ned_m=start,
        goal_ned_m=goal,
        speed_mps=active_config.nominal_speed_mps,
    )

    corridor = (
        Obstacle(
            obstacle_id="m2-s3-corridor-left",
            kind="box",
            center_ned_m=np.array([4.0, -2.6, -6.0], dtype=np.float64),
            half_extents_m=np.array([14.0, 0.1, 6.0], dtype=np.float64),
        ),
        Obstacle(
            obstacle_id="m2-s3-corridor-right",
            kind="box",
            center_ned_m=np.array([4.0, 2.6, -6.0], dtype=np.float64),
            half_extents_m=np.array([14.0, 0.1, 6.0], dtype=np.float64),
        ),
        Obstacle(
            obstacle_id="m2-s3-gate-left",
            kind="box",
            center_ned_m=np.array([4.5, -1.15, -6.0], dtype=np.float64),
            half_extents_m=np.array([0.1, 1.35, 6.0], dtype=np.float64),
        ),
        Obstacle(
            obstacle_id="m2-s3-gate-right",
            kind="box",
            center_ned_m=np.array([4.5, 2.35, -6.0], dtype=np.float64),
            half_extents_m=np.array([0.1, 0.15, 6.0], dtype=np.float64),
        ),
    )
    offset_gate = _scenario(
        "m2_s3_offset_gate",
        corridor,
        start_ned_m=start,
        goal_ned_m=goal,
        speed_mps=active_config.nominal_speed_mps,
    )

    return (
        M2Scenario("s1", empty, active_config.nominal_speed_mps, active_config.max_duration_s),
        M2Scenario(
            "s2",
            single_obstacle,
            active_config.nominal_speed_mps,
            active_config.max_duration_s,
        ),
        M2Scenario(
            "s3",
            offset_gate,
            active_config.nominal_speed_mps,
            active_config.max_duration_s,
        ),
    )


def make_route_window(state: State, goal_ned_m: np.ndarray, speed_mps: float) -> Trajectory:
    """Build a straight nominal route from the current state to the goal."""

    goal = np.asarray(goal_ned_m, dtype=np.float64)
    if goal.shape != (3,):
        raise ValueError("goal_ned_m must have shape (3,)")
    if not np.isfinite(speed_mps) or speed_mps <= 0.0:
        raise ValueError("speed_mps must be finite and positive")
    start_position = state.position_ned_m.copy()
    delta = goal - start_position
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        end_position = start_position + np.array([0.1, 0.0, 0.0], dtype=np.float64)
        distance = 0.1
    else:
        end_position = goal.copy()
    direction = (end_position - start_position) / distance
    quat = forward_quaternion(direction)
    duration = max(distance / float(speed_mps), 0.1)
    return Trajectory(
        trajectory_id=f"route-{state.t_s:.3f}",
        t_s=np.array([state.t_s, state.t_s + duration], dtype=np.float64),
        position_ned_m=np.vstack((start_position, end_position)),
        quat_xyzw=np.vstack((quat, quat)),
        linear_velocity_body_mps=np.array(
            [[speed_mps, 0.0, 0.0], [speed_mps, 0.0, 0.0]],
            dtype=np.float64,
        ),
        angular_velocity_body_rps=np.zeros((2, 3), dtype=np.float64),
    )


def config_to_dict(config: M2Config) -> dict[str, Any]:
    return {
        "rollout": {
            "horizon_s": config.rollout.horizon_s,
            "step_s": config.rollout.step_s,
            "replan_period_s": config.rollout.replan_period_s,
            "max_path_length_m": config.rollout.max_path_length_m,
            "model_id": config.rollout.model_id,
            "stop_on_collision": config.rollout.stop_on_collision,
            "clip_actions": config.rollout.clip_actions,
            "inherit_depth_from_route": config.rollout.inherit_depth_from_route,
        },
        "seeds": list(config.seeds),
        "nominal_speed_mps": config.nominal_speed_mps,
        "max_duration_s": config.max_duration_s,
        "pvs_sample_time_s": config.pvs_sample_time_s,
        "oracle_ray_count": config.oracle_ray_count,
        "oracle_min_range_m": config.oracle_min_range_m,
        "oracle_max_range_m": config.oracle_max_range_m,
        "oracle_validity_s": config.oracle_validity_s,
        "horizontal_lookahead_m": config.horizontal_lookahead_m,
        "vertical_lookahead_m": config.vertical_lookahead_m,
    }


__all__ = [
    "M2Config",
    "M2Scenario",
    "M2_NOMINAL_SPEED_MPS",
    "M2_WORLD_BOUNDS_NED_M",
    "config_to_dict",
    "default_m2_config",
    "load_m2_config",
    "m2_capability",
    "m2_scenarios",
    "m2_world_bounds",
    "make_route_window",
]
