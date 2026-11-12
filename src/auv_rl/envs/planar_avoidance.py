from __future__ import annotations

from dataclasses import dataclass, field
from math import degrees, isfinite, pi
from typing import Any

import gymnasium as gym
import numpy as np
from auv_core.attitude import euler_from_quaternion, quaternion_from_euler
from auv_core.contracts import FloatArray, ObstacleMap, State, VehicleSpec
from auv_core.geometry import HullSpec, hull_clearance_details, hull_from_vehicle

from auv_local.contracts import LocalObservation, SurgeYawAction
from auv_local.oracle import OracleSensor

from ..observation import ForwardSectorObservationBuilder
from ..perception import OraclePerceptionBackend, PerceptionBackend
from ..planar import PlanarKinematics, PlanarMotionLimits
from ..reward import PaperReward, PaperRewardConfig, RewardTerms
from ..scenarios import PlanarScenarioCase, default_planar_scenarios, scenario_by_key


def _positive(value: float, name: str) -> float:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class PlanarEnvConfig:
    dt_s: float = 0.2
    integration_step_s: float = 0.02
    max_episode_steps: int = 300
    sensor_max_range_m: float = 120.0
    sensor_ray_count: int = 1300
    sensor_validity_s: float = 0.5
    sensor_min_range_m: float = 0.2
    sensor_vertical_half_width_m: float = 0.5
    sensor_origin_body_m: tuple[float, float, float] = (0.8, 0.0, 0.0)
    sensor_field_of_view_rad: float = 130.0 * pi / 180.0
    local_effective_range_m: float = 20.0
    reward_obstacle_range_m: float = 20.0
    reward_step_scale: float = 1.0
    observation_version: str = "1.0"
    analytic_observation_source: str = "analytic_13sector"
    analytic_perception_mode: str = "analytic_13sector_geometry"
    front_obstacle_cone_rad: float = 30.0 * pi / 180.0
    sensor_id: str = "oceansim-pcl"
    action_max_accel_mps2: float = 0.5
    action_max_yaw_rate_rps: float = 15.0 * pi / 180.0
    start_jitter_xy_m: float = 0.0
    start_jitter_yaw_rad: float = 0.0
    active_scenario_keys: tuple[str, ...] = ("s1", "s2")
    reward: PaperRewardConfig = field(default_factory=PaperRewardConfig)

    def __post_init__(self) -> None:
        dt = _positive(self.dt_s, "dt_s")
        integration_step = _positive(self.integration_step_s, "integration_step_s")
        if integration_step > dt + 1e-12:
            raise ValueError("integration_step_s must not exceed dt_s")
        substeps = dt / integration_step
        if abs(substeps - round(substeps)) > 1e-9:
            raise ValueError("dt_s must be an integer multiple of integration_step_s")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.sensor_ray_count < 8:
            raise ValueError("sensor_ray_count must be at least 8")
        if not self.sensor_id:
            raise ValueError("sensor_id must not be empty")
        local_range = _positive(self.local_effective_range_m, "local_effective_range_m")
        sensor_range = _positive(self.sensor_max_range_m, "sensor_max_range_m")
        if local_range > sensor_range:
            raise ValueError("local_effective_range_m must not exceed sensor_max_range_m")
        reward_range = _positive(self.reward_obstacle_range_m, "reward_obstacle_range_m")
        if reward_range > sensor_range:
            raise ValueError("reward_obstacle_range_m must not exceed sensor_max_range_m")
        reward_scale = _positive(self.reward_step_scale, "reward_step_scale")
        observation_version = self.observation_version.strip()
        if not observation_version:
            raise ValueError("observation_version must not be empty")
        analytic_source = self.analytic_observation_source.strip()
        if not analytic_source:
            raise ValueError("analytic_observation_source must not be empty")
        analytic_mode = self.analytic_perception_mode.strip()
        if not analytic_mode:
            raise ValueError("analytic_perception_mode must not be empty")
        sensor_min = _positive(self.sensor_min_range_m, "sensor_min_range_m")
        if sensor_min >= sensor_range:
            raise ValueError("sensor_min_range_m must be smaller than sensor_max_range_m")
        vertical = _positive(self.sensor_vertical_half_width_m, "sensor_vertical_half_width_m")
        fov = _positive(self.sensor_field_of_view_rad, "sensor_field_of_view_rad")
        if fov > 2.0 * pi:
            raise ValueError("sensor_field_of_view_rad must not exceed 2*pi")
        front_cone = _positive(self.front_obstacle_cone_rad, "front_obstacle_cone_rad")
        if front_cone > 0.5 * fov:
            raise ValueError("front_obstacle_cone_rad must fit inside the sensor field of view")
        origin = np.asarray(self.sensor_origin_body_m, dtype=np.float64)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("sensor_origin_body_m must be a finite shape-(3,) array")
        if not self.active_scenario_keys:
            raise ValueError("active_scenario_keys must not be empty")
        if any(not key for key in self.active_scenario_keys):
            raise ValueError("active_scenario_keys must not contain empty values")
        object.__setattr__(self, "dt_s", dt)
        object.__setattr__(self, "integration_step_s", integration_step)
        object.__setattr__(self, "sensor_max_range_m", sensor_range)
        object.__setattr__(self, "local_effective_range_m", local_range)
        object.__setattr__(self, "reward_obstacle_range_m", reward_range)
        object.__setattr__(self, "reward_step_scale", reward_scale)
        object.__setattr__(self, "observation_version", observation_version)
        object.__setattr__(self, "analytic_observation_source", analytic_source)
        object.__setattr__(self, "analytic_perception_mode", analytic_mode)
        object.__setattr__(self, "sensor_min_range_m", sensor_min)
        object.__setattr__(self, "sensor_vertical_half_width_m", vertical)
        object.__setattr__(self, "sensor_field_of_view_rad", fov)
        object.__setattr__(self, "front_obstacle_cone_rad", front_cone)
        object.__setattr__(
            self,
            "sensor_origin_body_m",
            tuple(float(item) for item in origin),
        )
        object.__setattr__(
            self,
            "sensor_validity_s",
            _positive(self.sensor_validity_s, "sensor_validity_s"),
        )
        object.__setattr__(
            self,
            "action_max_accel_mps2",
            _positive(self.action_max_accel_mps2, "action_max_accel_mps2"),
        )
        object.__setattr__(
            self,
            "action_max_yaw_rate_rps",
            _positive(self.action_max_yaw_rate_rps, "action_max_yaw_rate_rps"),
        )
        for name in ("start_jitter_xy_m", "start_jitter_yaw_rad"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.reward, PaperRewardConfig):
            raise TypeError("reward must be a PaperRewardConfig")


@dataclass(frozen=True)
class PrivilegedSceneState:
    """Ground-truth scene data exposed only to the teacher/collector."""

    state: State
    goal_ned_m: FloatArray
    goal_radius_m: float
    obstacle_map: ObstacleMap
    vehicle: VehicleSpec


class PlanarAvoidanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Two-dimensional pretraining environment for the P1 curriculum.

    This environment deliberately uses a horizontal surrogate and does not
    replace PVS or OceanSim. Its purpose is to learn obstacle avoidance before
    the policy is connected to rollout, ALOS and the REMUS controller.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: PlanarEnvConfig | None = None,
        *,
        scenarios: tuple[PlanarScenarioCase, ...] | None = None,
        perception: PerceptionBackend | None = None,
    ) -> None:
        super().__init__()
        self.config = PlanarEnvConfig() if config is None else config
        if not isinstance(self.config, PlanarEnvConfig):
            raise TypeError("config must be a PlanarEnvConfig")
        self._cases = default_planar_scenarios() if scenarios is None else scenarios
        if not self._cases:
            raise ValueError("scenarios must not be empty")
        self._case_by_key = {case.key: case for case in self._cases}
        missing = [key for key in self.config.active_scenario_keys if key not in self._case_by_key]
        if missing:
            raise ValueError(f"active scenarios are unavailable: {', '.join(missing)}")
        first_vehicle = self._cases[0].scenario.vehicle
        if any(case.scenario.vehicle.name != first_vehicle.name for case in self._cases):
            raise ValueError("all P1 scenarios must use the same vehicle model")

        self.limits = PlanarMotionLimits(
            min_speed_mps=first_vehicle.min_speed_mps,
            max_speed_mps=first_vehicle.max_speed_mps,
            max_accel_mps2=self.config.action_max_accel_mps2,
            max_yaw_rate_rps=self.config.action_max_yaw_rate_rps,
        )
        self.kinematics = PlanarKinematics(self.limits)
        self.builder = ForwardSectorObservationBuilder(
            max_range_m=self.config.local_effective_range_m,
            sector_count=13,
            field_of_view_rad=self.config.sensor_field_of_view_rad,
            min_speed_mps=self.limits.min_speed_mps,
            max_speed_mps=self.limits.max_speed_mps,
            max_yaw_rate_rps=first_vehicle.max_yaw_rate_rps,
            action_max_accel_mps2=self.limits.max_accel_mps2,
            action_max_yaw_rate_rps=self.limits.max_yaw_rate_rps,
            version=self.config.observation_version,
        )
        self.reward_function = PaperReward(self.config.reward)
        self.perception = (
            OraclePerceptionBackend(
                OracleSensor(
                    ray_count=self.config.sensor_ray_count,
                    min_range_m=self.config.sensor_min_range_m,
                    max_range_m=self.config.sensor_max_range_m,
                    validity_s=self.config.sensor_validity_s,
                    sensor_id="oracle-360",
                )
            )
            if perception is None
            else perception
        )
        self._hulls: dict[str, HullSpec] = {
            case.key: hull_from_vehicle(case.scenario.vehicle) for case in self._cases
        }
        self._rng = np.random.default_rng(0)
        self._case: PlanarScenarioCase | None = None
        self._state: State | None = None
        self._initial_goal_distance_m = 1.0
        self._previous_goal_distance_m = 1.0
        self._previous_clearance_m = float("inf")
        self._episode_steps = 0
        self._episode_path_length_m = 0.0
        self._episode_min_clearance_m = float("inf")
        self._previous_action: FloatArray | None = None
        self._last_observation_health: dict[str, float] = {}
        self._episode_action_clip_count = 0
        self._episode_action_clipped_steps = 0
        self._episode_action_step_count = 0
        self._episode_action_saturation_counts = np.zeros(2, dtype=np.int64)
        self._episode_action_abs_sum = np.zeros(2, dtype=np.float64)
        self._episode_action_abs_max = np.zeros(2, dtype=np.float64)
        self._episode_start_jitter_xy_m = 0.0
        self._episode_start_jitter_yaw_rad = 0.0
        self._terminated = False

        observation_size = self.builder.spec.shape[0]
        low = np.zeros(observation_size, dtype=np.float32)
        high = np.ones(observation_size, dtype=np.float32)
        low[self.builder.sector_count] = 0.0
        high[self.builder.sector_count] = 1.0
        low[self.builder.sector_count + 1] = -1.0
        low[self.builder.sector_count + 3] = -1.0
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array(
                [-self.limits.max_accel_mps2, -self.limits.max_yaw_rate_rps],
                dtype=np.float32,
            ),
            high=np.array(
                [self.limits.max_accel_mps2, self.limits.max_yaw_rate_rps],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    @property
    def scenario_key(self) -> str:
        if self._case is None:
            raise RuntimeError("reset() must be called before reading scenario_key")
        return self._case.key

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        seed_value = 0 if seed is None else int(seed)
        self._rng = np.random.default_rng(seed_value)
        selected = None if options is None else options.get("scenario_key")
        jitter_option = None if options is None else options.get("jitter")
        if jitter_option is not None and not isinstance(jitter_option, bool):
            raise TypeError("options['jitter'] must be a bool when provided")
        if selected is None:
            keys = self.config.active_scenario_keys
            selected = keys[int(self._rng.integers(0, len(keys)))]
        if not isinstance(selected, str):
            raise TypeError("options['scenario_key'] must be a string")
        self._case = scenario_by_key(self._cases, selected)
        scenario = self._case.scenario
        self.perception.reset(scenario=scenario)
        start = scenario.start
        jitter_xy = self.config.start_jitter_xy_m if jitter_option is not False else 0.0
        jitter_yaw = self.config.start_jitter_yaw_rad if jitter_option is not False else 0.0
        self._episode_start_jitter_xy_m = float(jitter_xy)
        self._episode_start_jitter_yaw_rad = float(jitter_yaw)
        position = start.position_ned_m + np.array(
            [
                float(self._rng.uniform(-jitter_xy, jitter_xy)) if jitter_xy else 0.0,
                float(self._rng.uniform(-jitter_xy, jitter_xy)) if jitter_xy else 0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        yaw = float(euler_from_quaternion(start.quat_xyzw)[2])
        if jitter_yaw:
            yaw += float(self._rng.uniform(-jitter_yaw, jitter_yaw))
        self._state = State(
            t_s=0.0,
            position_ned_m=position,
            quat_xyzw=quaternion_from_euler(0.0, 0.0, yaw),
            linear_velocity_body_mps=start.linear_velocity_body_mps.copy(),
            angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
        )
        self._initial_goal_distance_m = max(
            float(np.linalg.norm(scenario.goal_ned_m - position)),
            1e-6,
        )
        self._previous_goal_distance_m = self._initial_goal_distance_m
        self._episode_steps = 0
        self._episode_path_length_m = 0.0
        clearance = hull_clearance_details(
            scenario.world.obstacle_map,
            self._state,
            self._hulls[self._case.key],
        )
        self._episode_min_clearance_m = clearance.clearance_m
        self._previous_clearance_m = clearance.clearance_m
        self._previous_action = np.zeros(2, dtype=np.float64)
        self._last_observation_health = {}
        self._episode_action_clip_count = 0
        self._episode_action_clipped_steps = 0
        self._episode_action_step_count = 0
        self._episode_action_saturation_counts[:] = 0
        self._episode_action_abs_sum[:] = 0.0
        self._episode_action_abs_max[:] = 0.0
        self._terminated = False
        observation = self._observation()
        info = self._info(
            goal_distance_m=self._initial_goal_distance_m,
            collision=clearance.collision,
            out_of_bounds=clearance.out_of_bounds,
            goal_reached=False,
            success=False,
            reward_terms=RewardTerms(0.0, 0.0, 0.0),
            action_clipped=False,
        )
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None or self._case is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("step() called after episode termination; call reset()")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (2,) or not np.all(np.isfinite(action_array)):
            raise ValueError("action must be a finite shape-(2,) array")
        action_low = np.array(
            [-self.limits.max_accel_mps2, -self.limits.max_yaw_rate_rps],
            dtype=np.float64,
        )
        action_high = np.array(
            [self.limits.max_accel_mps2, self.limits.max_yaw_rate_rps],
            dtype=np.float64,
        )
        clipped_action = np.clip(action_array, action_low, action_high)
        previous_action = self._previous_action
        if previous_action is None:
            previous_action = np.zeros(2, dtype=np.float64)
        previous_goal_distance = self._previous_goal_distance_m
        previous_clearance = self._previous_clearance_m
        self._episode_action_step_count += 1
        clipped_dims = np.abs(clipped_action - action_array) > 1e-12
        self._episode_action_clip_count += int(np.count_nonzero(clipped_dims))
        if bool(np.any(clipped_dims)):
            self._episode_action_clipped_steps += 1
        action_range = action_high - action_low
        saturation_tolerance = 1e-3 * action_range
        self._episode_action_saturation_counts += (
            np.abs(clipped_action) >= (action_high - saturation_tolerance)
        ).astype(np.int64)
        self._episode_action_abs_sum += np.abs(clipped_action)
        self._episode_action_abs_max = np.maximum(
            self._episode_action_abs_max,
            np.abs(clipped_action),
        )
        requested = SurgeYawAction(
            u_dot_mps2=float(clipped_action[0]),
            yaw_rate_rps=float(clipped_action[1]),
        )
        scenario = self._case.scenario
        obstacle_map = scenario.world.obstacle_map
        hull = self._hulls[self._case.key]
        state = self._state
        collision = False
        out_of_bounds = False
        goal_reached = False
        action_clipped = bool(np.any(np.abs(clipped_action - action_array) > 1e-12))
        substeps = int(round(self.config.dt_s / self.config.integration_step_s))
        min_clearance = self._episode_min_clearance_m

        for _ in range(substeps):
            current_ned = scenario.world.current_field.sample(state.position_ned_m, state.t_s)
            transition = self.kinematics.step(
                state,
                requested,
                self.config.integration_step_s,
                current_ned_mps=current_ned,
            )
            previous_position = state.position_ned_m
            state = transition.state
            action_clipped = action_clipped or transition.action_clipped
            self._episode_path_length_m += float(
                np.linalg.norm(state.position_ned_m - previous_position)
            )
            clearance = hull_clearance_details(obstacle_map, state, hull)
            min_clearance = min(min_clearance, clearance.clearance_m)
            collision = clearance.collision
            out_of_bounds = clearance.out_of_bounds
            if collision or out_of_bounds:
                break
            if float(np.linalg.norm(state.position_ned_m - scenario.goal_ned_m)) <= (
                scenario.goal_radius_m + 1e-9
            ):
                goal_reached = True
                break

        self._state = state
        self._episode_min_clearance_m = min_clearance
        self._previous_action = clipped_action.copy()
        self._episode_steps += 1
        self._terminated = bool(collision or out_of_bounds or goal_reached)
        truncated = bool(
            not self._terminated and self._episode_steps >= self.config.max_episode_steps
        )
        goal_distance = float(np.linalg.norm(state.position_ned_m - scenario.goal_ned_m))
        action_scale = np.array(
            [self.limits.max_accel_mps2, self.limits.max_yaw_rate_rps],
            dtype=np.float64,
        )
        reference_speed = (
            self.config.reward.reference_speed_mps
            if self.config.reward.reference_speed_mps is not None
            else float(scenario.start.linear_velocity_body_mps[0])
        )
        local_observation = self._local_observation()
        observation = self.builder.build(
            local_observation,
            scenario.goal_ned_m,
            initial_goal_distance_m=self._initial_goal_distance_m,
        )
        scan = local_observation.scans[0]
        front_range = self.builder.nearest_range(scan, self.config.front_obstacle_cone_rad)
        reward_terms = self.reward_function.compute(
            goal_distance_m=goal_distance,
            initial_goal_distance_m=self._initial_goal_distance_m,
            front_obstacle_range_m=front_range,
            sensor_max_range_m=self.config.local_effective_range_m,
            reward_obstacle_range_m=self.config.reward_obstacle_range_m,
            reward_step_scale=self.config.reward_step_scale,
            collision=collision,
            out_of_bounds=out_of_bounds,
            goal_reached=goal_reached,
            truncated=truncated,
            previous_goal_distance_m=previous_goal_distance,
            previous_clearance_m=previous_clearance,
            current_clearance_m=clearance.clearance_m,
            surge_speed_mps=float(state.linear_velocity_body_mps[0]),
            normalized_action=clipped_action / action_scale,
            previous_normalized_action=previous_action / action_scale,
            reference_speed_mps=reference_speed,
        )
        self._previous_goal_distance_m = goal_distance
        self._previous_clearance_m = clearance.clearance_m
        success = bool(goal_reached and not collision and not out_of_bounds and not truncated)
        info = self._info(
            goal_distance_m=goal_distance,
            collision=collision,
            out_of_bounds=out_of_bounds,
            goal_reached=goal_reached,
            success=success,
            reward_terms=reward_terms,
            action_clipped=action_clipped,
            front_obstacle_range_m=front_range,
            truncated=truncated,
        )
        return observation, reward_terms.total, self._terminated, truncated, info

    def _observation(self) -> np.ndarray:
        if self._case is None:
            raise RuntimeError("reset() must be called before building an observation")
        return self.builder.build(
            self._local_observation(),
            self._case.scenario.goal_ned_m,
            initial_goal_distance_m=self._initial_goal_distance_m,
        )

    def privileged_scene_state(self) -> PrivilegedSceneState:
        """Return oracle scene data for teacher collection, never for the actor."""

        if self._case is None or self._state is None:
            raise RuntimeError("reset() must be called before reading privileged state")
        return PrivilegedSceneState(
            state=self._state,
            goal_ned_m=self._case.scenario.goal_ned_m.copy(),
            goal_radius_m=self._case.scenario.goal_radius_m,
            obstacle_map=self._case.scenario.world.obstacle_map,
            vehicle=self._case.scenario.vehicle,
        )

    def _local_observation(self) -> LocalObservation:
        if self._state is None or self._case is None:
            raise RuntimeError("reset() must be called before building an observation")
        observation = self.perception.observe(
            self._state,
            previous_action=self._previous_action,
        )
        self._last_observation_health = dict(observation.health)
        return observation

    def _info(
        self,
        *,
        goal_distance_m: float,
        collision: bool,
        out_of_bounds: bool,
        goal_reached: bool,
        success: bool,
        reward_terms: RewardTerms,
        action_clipped: bool,
        front_obstacle_range_m: float | None = None,
        truncated: bool = False,
    ) -> dict[str, Any]:
        if self._case is None or self._state is None:
            raise RuntimeError("reset() must be called before building info")
        failure_reason: str | None = None
        if collision:
            failure_reason = "collision"
        elif out_of_bounds:
            failure_reason = "out_of_bounds"
        elif truncated:
            failure_reason = "timeout"
        elif not goal_reached:
            failure_reason = "in_progress"
        steps = max(self._episode_action_step_count, 1)
        health = self._last_observation_health
        return {
            "scenario_key": self._case.key,
            "scenario_name": self._case.name,
            "rl_step": self._episode_steps,
            "simulation_time_s": self._state.t_s,
            "success": success,
            "goal_reached": goal_reached,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "timeout": truncated,
            "failure_reason": failure_reason,
            "goal_distance_m": goal_distance_m,
            "initial_goal_distance_m": self._initial_goal_distance_m,
            "min_clearance_m": self._episode_min_clearance_m,
            "path_length_m": self._episode_path_length_m,
            "front_obstacle_range_m": front_obstacle_range_m,
            "action_clipped": action_clipped,
            "action_u_dot_mps2": float(self._previous_action[0])
            if self._previous_action is not None
            else 0.0,
            "action_yaw_rate_rps": float(self._previous_action[1])
            if self._previous_action is not None
            else 0.0,
            "action_yaw_rate_deg_s": degrees(float(self._previous_action[1]))
            if self._previous_action is not None
            else 0.0,
            "action_clip_count": self._episode_action_clip_count,
            "action_clipped_step_count": self._episode_action_clipped_steps,
            "action_u_dot_saturation_ratio": float(
                self._episode_action_saturation_counts[0] / steps
            ),
            "action_yaw_rate_saturation_ratio": float(
                self._episode_action_saturation_counts[1] / steps
            ),
            "action_u_dot_max_abs": float(self._episode_action_abs_max[0]),
            "action_yaw_rate_max_abs": float(self._episode_action_abs_max[1]),
            "action_u_dot_mean_abs": float(self._episode_action_abs_sum[0] / steps),
            "action_yaw_rate_mean_abs": float(self._episode_action_abs_sum[1] / steps),
            "start_jitter_xy_m": self._episode_start_jitter_xy_m,
            "start_jitter_yaw_rad": self._episode_start_jitter_yaw_rad,
            "observation_source": self.perception.observation_source,
            "perception_mode": self.perception.perception_mode,
            "perception_config_hash": self.perception.config_hash,
            "sensor_scan_valid": float(health.get("scan_valid", 0.0)),
            "sensor_points_total": float(health.get("points_total", 0.0)),
            "sensor_points_used": float(health.get("points_used", 0.0)),
            "sensor_empty_rays": float(health.get("empty_rays", 0.0)),
            "sensor_frame_age_s": float(health.get("frame_age_s", 0.0)),
            "sensor_scan_returned_false": float(health.get("scan_returned_false", 0.0)),
            "sensor_pose_write_ms": float(health.get("pose_write_ms", 0.0)),
            "sensor_render_update_ms": float(health.get("render_update_ms", 0.0)),
            "sensor_scan_ms": float(health.get("sensor_scan_ms", 0.0)),
            "sensor_point_transform_ms": float(health.get("point_transform_ms", 0.0)),
            "reward_total": reward_terms.total,
            "reward_distance": reward_terms.distance_reward,
            "reward_obstacle": reward_terms.obstacle_reward,
            "reward_terminal": reward_terms.terminal_reward,
            "reward_goal_progress": reward_terms.goal_progress_reward,
            "reward_reference_speed": reward_terms.reference_speed_reward,
            "reward_action_smoothness": reward_terms.action_smoothness_reward,
        }


__all__ = ["PlanarAvoidanceEnv", "PlanarEnvConfig", "PrivilegedSceneState"]
