from __future__ import annotations

import copy
import time
from math import pi
from typing import Any

import gymnasium as gym
import numpy as np
from auv_core.attitude import quaternion_from_euler
from auv_core.contracts import ObstacleMap, State
from auv_core.geometry import (
    HullSpec,
    hull_from_vehicle,
    hull_sphere_centers_from_poses,
    obstacle_signed_distance_array,
)
from numpy.typing import NDArray

from ..observation import ForwardSectorObservationBuilder
from ..perception import analytic_ray_ranges
from ..planar import PlanarMotionLimits
from ..reward import PaperRewardConfig
from ..scenarios import PlanarScenarioCase, default_planar_scenarios, scenario_by_key
from .analytic import build_analytic_perception
from .planar_avoidance import PlanarEnvConfig


def _wrap_angle(angle_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray((angle_rad + pi) % (2.0 * pi) - pi, dtype=np.float64)


_TRAINING_STATE_ARRAYS = (
    "_positions",
    "_yaws",
    "_surges",
    "_yaw_rates",
    "_time_s",
    "_episode_ids",
    "_episode_steps",
    "_initial_goal_distance",
    "_previous_goal_distance",
    "_previous_clearance",
    "_last_clearance",
    "_min_clearance",
    "_path_length",
    "_previous_action",
    "_reference_speed",
    "_scenario_keys",
    "_goals",
    "_goal_radii",
    "_action_clip_count",
    "_action_clipped_steps",
    "_action_step_count",
    "_action_saturation_counts",
    "_action_abs_sum",
    "_action_abs_max",
    "_start_jitter_xy",
    "_start_jitter_yaw",
    "_sensor_points_total",
    "_sensor_points_used",
    "_sensor_empty_rays",
    "_sensor_frame_age",
)


class P1TensorVectorEnv:
    """Batch P1 environment with analytic perception and automatic reset."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        config: PlanarEnvConfig | None = None,
        *,
        scenarios: tuple[PlanarScenarioCase, ...] | None = None,
        seed: int = 0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
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
        self.perception = build_analytic_perception(self.config)
        self._hull: HullSpec = hull_from_vehicle(first_vehicle)
        self._rng = np.random.default_rng(seed)
        self._next_episode_id = 0
        self._completed_infos: list[dict[str, Any]] = []

        self._positions = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._yaws = np.zeros(self.num_envs, dtype=np.float64)
        self._surges = np.full(self.num_envs, self.limits.min_speed_mps, dtype=np.float64)
        self._yaw_rates = np.zeros(self.num_envs, dtype=np.float64)
        self._time_s = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_ids = np.full(self.num_envs, -1, dtype=np.int64)
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._initial_goal_distance = np.ones(self.num_envs, dtype=np.float64)
        self._previous_goal_distance = np.ones(self.num_envs, dtype=np.float64)
        self._previous_clearance = np.full(self.num_envs, np.inf, dtype=np.float64)
        self._last_clearance = np.full(self.num_envs, np.inf, dtype=np.float64)
        self._min_clearance = np.full(self.num_envs, np.inf, dtype=np.float64)
        self._path_length = np.zeros(self.num_envs, dtype=np.float64)
        self._previous_action = np.zeros((self.num_envs, 2), dtype=np.float64)
        self._reference_speed = np.ones(self.num_envs, dtype=np.float64)
        self._scenario_keys = np.full(self.num_envs, "", dtype=object)
        self._goals = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._goal_radii = np.ones(self.num_envs, dtype=np.float64)
        self._action_clip_count = np.zeros(self.num_envs, dtype=np.int64)
        self._action_clipped_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._action_step_count = np.zeros(self.num_envs, dtype=np.int64)
        self._action_saturation_counts = np.zeros((self.num_envs, 2), dtype=np.int64)
        self._action_abs_sum = np.zeros((self.num_envs, 2), dtype=np.float64)
        self._action_abs_max = np.zeros((self.num_envs, 2), dtype=np.float64)
        self._start_jitter_xy = np.zeros(self.num_envs, dtype=np.float64)
        self._start_jitter_yaw = np.zeros(self.num_envs, dtype=np.float64)
        self._sensor_points_total = np.zeros(self.num_envs, dtype=np.float64)
        self._sensor_points_used = np.zeros(self.num_envs, dtype=np.float64)
        self._sensor_empty_rays = np.zeros(self.num_envs, dtype=np.float64)
        self._sensor_frame_age = np.zeros(self.num_envs, dtype=np.float64)
        self.last_sensor_time_s = 0.0

        observation_size = self.builder.spec.shape[0]
        low = np.zeros((self.num_envs, observation_size), dtype=np.float32)
        high = np.ones((self.num_envs, observation_size), dtype=np.float32)
        low[:, self.builder.sector_count + 1] = -1.0
        low[:, self.builder.sector_count + 3] = -1.0
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.tile(
                np.array([-self.limits.max_accel_mps2, -self.limits.max_yaw_rate_rps]),
                (self.num_envs, 1),
            ).astype(np.float32),
            high=np.tile(
                np.array([self.limits.max_accel_mps2, self.limits.max_yaw_rate_rps]),
                (self.num_envs, 1),
            ).astype(np.float32),
            dtype=np.float32,
        )

    @property
    def episode_ids(self) -> NDArray[np.int64]:
        return self._episode_ids.copy()

    @property
    def episode_steps(self) -> NDArray[np.int64]:
        return self._episode_steps.copy()

    @property
    def scenario_keys(self) -> NDArray[np.object_]:
        return self._scenario_keys.copy()

    @property
    def observation_source(self) -> str:
        return self.perception.observation_source

    def state(self, env_id: int) -> State:
        if not 0 <= env_id < self.num_envs:
            raise IndexError("env_id out of range")
        return State(
            t_s=float(self._time_s[env_id]),
            position_ned_m=self._positions[env_id].copy(),
            quat_xyzw=quaternion_from_euler(0.0, 0.0, float(self._yaws[env_id])),
            linear_velocity_body_mps=np.array(
                [self._surges[env_id], 0.0, 0.0],
                dtype=np.float64,
            ),
            angular_velocity_body_rps=np.array(
                [0.0, 0.0, self._yaw_rates[env_id]],
                dtype=np.float64,
            ),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        options = {} if options is None else options
        jitter_option = options.get("jitter")
        if jitter_option is not None and not isinstance(jitter_option, bool):
            raise TypeError("options['jitter'] must be a bool when provided")
        scenario_option = options.get("scenario_key")
        self._completed_infos.clear()
        self._reset_indices(
            np.arange(self.num_envs, dtype=np.int64),
            scenario_key=scenario_option,
            jitter=jitter_option,
        )
        observations, _, _, _, _ = self._batch_observation()
        info = self._reset_info()
        return observations, info

    def step(
        self,
        actions: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        action_array = np.asarray(actions, dtype=np.float64)
        if action_array.shape != (self.num_envs, 2) or not np.all(np.isfinite(action_array)):
            raise ValueError("actions must be a finite shape-(num_envs, 2) array")
        action_low = np.array(
            [-self.limits.max_accel_mps2, -self.limits.max_yaw_rate_rps],
            dtype=np.float64,
        )
        action_high = np.array(
            [self.limits.max_accel_mps2, self.limits.max_yaw_rate_rps],
            dtype=np.float64,
        )
        clipped: NDArray[np.float64] = np.asarray(
            np.clip(action_array, action_low, action_high),
            dtype=np.float64,
        )
        previous_action = self._previous_action.copy()
        previous_goal_distance = self._previous_goal_distance.copy()
        previous_clearance = self._previous_clearance.copy()
        self._action_step_count += 1
        clipped_dims = np.abs(clipped - action_array) > 1e-12
        self._action_clip_count += np.count_nonzero(clipped_dims, axis=1)
        self._action_clipped_steps += np.any(clipped_dims, axis=1).astype(np.int64)
        action_range = action_high - action_low
        saturation_tolerance = 1e-3 * action_range
        self._action_saturation_counts += (
            np.abs(clipped) >= (action_high - saturation_tolerance)
        ).astype(np.int64)
        self._action_abs_sum += np.abs(clipped)
        self._action_abs_max = np.maximum(self._action_abs_max, np.abs(clipped))

        collision = np.zeros(self.num_envs, dtype=bool)
        out_of_bounds = np.zeros(self.num_envs, dtype=bool)
        goal_reached = np.zeros(self.num_envs, dtype=bool)
        action_clipped = np.any(clipped_dims, axis=1)
        substeps = int(round(self.config.dt_s / self.config.integration_step_s))
        current_clearance = self._last_clearance.copy()

        for _ in range(substeps):
            active = np.flatnonzero(~(collision | out_of_bounds | goal_reached))
            if active.size == 0:
                break
            current = np.zeros((active.size, 3), dtype=np.float64)
            for row, env_id in enumerate(active):
                current[row] = self._case_by_key[
                    str(self._scenario_keys[env_id])
                ].scenario.world.current_field.sample(
                    self._positions[env_id],
                    float(self._time_s[env_id]),
                )
            yaw = self._yaws[active]
            next_surge = np.clip(
                self._surges[active] + clipped[active, 0] * self.config.integration_step_s,
                self.limits.min_speed_mps,
                self.limits.max_speed_mps,
            )
            next_yaw = _wrap_angle(yaw + clipped[active, 1] * self.config.integration_step_s)
            previous_positions = self._positions[active].copy()
            next_positions = previous_positions + np.column_stack(
                (
                    (next_surge * np.cos(yaw) + current[:, 0]) * self.config.integration_step_s,
                    (next_surge * np.sin(yaw) + current[:, 1]) * self.config.integration_step_s,
                    np.zeros(active.size, dtype=np.float64),
                )
            )
            self._path_length[active] += np.linalg.norm(
                next_positions - previous_positions,
                axis=1,
            )
            self._positions[active] = next_positions
            self._yaws[active] = next_yaw
            self._surges[active] = next_surge
            self._yaw_rates[active] = clipped[active, 1]
            self._time_s[active] += self.config.integration_step_s

            clearance, obstacle_clearance, boundary_clearance = self._batch_clearance(
                next_positions,
                next_yaw,
                self._scenario_keys[active],
            )
            current_clearance[active] = clearance
            self._last_clearance[active] = clearance
            self._min_clearance[active] = np.minimum(
                self._min_clearance[active],
                clearance,
            )
            collision[active] = obstacle_clearance < 0.0
            out_of_bounds[active] = boundary_clearance < 0.0
            goal_distance = np.linalg.norm(self._goals[active] - next_positions, axis=1)
            goal_reached[active] = goal_distance <= self._goal_radii[active] + 1e-9

        self._episode_steps += 1
        terminated = collision | out_of_bounds | goal_reached
        truncated = (~terminated) & (self._episode_steps >= self.config.max_episode_steps)
        goal_distance = np.linalg.norm(self._goals - self._positions, axis=1)
        observations, ranges, confidence, sector_ranges, front_ranges = self._batch_observation()
        self._sensor_points_total[:] = 0.0
        self._sensor_points_used[:] = 0.0
        self._sensor_empty_rays[:] = np.count_nonzero(confidence < 0.5, axis=1)
        self._sensor_frame_age[:] = 0.0
        action_scale = np.array(
            [self.limits.max_accel_mps2, self.limits.max_yaw_rate_rps],
            dtype=np.float64,
        )
        reward_terms = self._reward_terms(
            goal_distance=goal_distance,
            initial_goal_distance=self._initial_goal_distance,
            front_ranges=front_ranges,
            collision=collision,
            out_of_bounds=out_of_bounds,
            goal_reached=goal_reached,
            previous_goal_distance=previous_goal_distance,
            previous_clearance=previous_clearance,
            current_clearance=current_clearance,
            surge=self._surges,
            normalized_action=clipped / action_scale,
            previous_normalized_action=previous_action / action_scale,
        )
        rewards = reward_terms["total"]
        success = goal_reached & ~collision & ~out_of_bounds & ~truncated
        info = self._step_info(
            goal_distance=goal_distance,
            collision=collision,
            out_of_bounds=out_of_bounds,
            goal_reached=goal_reached,
            success=success,
            truncated=truncated,
            front_ranges=front_ranges,
            action_clipped=action_clipped,
            reward_terms=reward_terms,
            observations=observations,
        )
        done = terminated | truncated
        if np.any(done):
            done_indices = np.flatnonzero(done)
            for env_id in done_indices.tolist():
                self._completed_infos.append(self._scalar_info(info, env_id))
            final_observations = observations.copy()
            info["final_observations"] = final_observations
            info["reset_mask"] = done.copy()
            self._reset_indices(done_indices, scenario_key=None, jitter=None)
            reset_observations, _, _, _, _ = self._batch_observation()
            observations = observations.copy()
            observations[done_indices] = reset_observations[done_indices]
        else:
            info["final_observations"] = observations.copy()
            info["reset_mask"] = done.copy()

        not_done = ~done
        self._previous_goal_distance[not_done] = goal_distance[not_done]
        self._previous_clearance[not_done] = current_clearance[not_done]
        self._previous_action[not_done] = clipped[not_done]
        return observations, rewards, terminated, truncated, info

    def drain_completed_infos(self) -> list[dict[str, Any]]:
        rows = self._completed_infos
        self._completed_infos = []
        return rows

    def close(self) -> None:
        return None

    def training_state_dict(self) -> dict[str, Any]:
        """Capture all mutable environment state and the reset RNG."""

        return {
            "format": "auv_rl.p1_tensor_vector_env.v1",
            "num_envs": self.num_envs,
            "arrays": {
                name: np.asarray(getattr(self, name)).copy() for name in _TRAINING_STATE_ARRAYS
            },
            "next_episode_id": int(self._next_episode_id),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "completed_infos": copy.deepcopy(self._completed_infos),
            "last_sensor_time_s": float(self.last_sensor_time_s),
        }

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`training_state_dict`."""

        if not isinstance(state, dict) or state.get("format") != "auv_rl.p1_tensor_vector_env.v1":
            raise ValueError("not a P1 tensor-vector environment training-state checkpoint")
        if int(state["num_envs"]) != self.num_envs:
            raise ValueError("environment checkpoint num_envs does not match")
        arrays = state["arrays"]
        for name in _TRAINING_STATE_ARRAYS:
            destination = getattr(self, name)
            source = np.asarray(arrays[name], dtype=destination.dtype)
            if source.shape != destination.shape:
                raise ValueError(f"environment checkpoint {name} shape does not match")
            destination[...] = source
        self._next_episode_id = int(state["next_episode_id"])
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, dict):
            raise ValueError("environment checkpoint is missing its RNG state")
        self._rng.bit_generator.state = copy.deepcopy(rng_state)
        self._completed_infos = copy.deepcopy(list(state.get("completed_infos", ())))
        self.last_sensor_time_s = float(state.get("last_sensor_time_s", 0.0))

    def _reset_indices(
        self,
        indices: NDArray[np.int64],
        *,
        scenario_key: object,
        jitter: bool | None,
    ) -> None:
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional array")
        keys = self._resolve_scenario_keys(indices.size, scenario_key)
        jitter_xy = self.config.start_jitter_xy_m if jitter is not False else 0.0
        jitter_yaw = self.config.start_jitter_yaw_rad if jitter is not False else 0.0
        for row, env_id_value in enumerate(indices.tolist()):
            env_id = int(env_id_value)
            key = keys[row]
            case = scenario_by_key(self._cases, key)
            scenario = case.scenario
            start = scenario.start
            position = start.position_ned_m + np.array(
                [
                    float(self._rng.uniform(-jitter_xy, jitter_xy)) if jitter_xy else 0.0,
                    float(self._rng.uniform(-jitter_xy, jitter_xy)) if jitter_xy else 0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            yaw = float(2.0 * np.arctan2(start.quat_xyzw[2], start.quat_xyzw[3]))
            if jitter_yaw:
                yaw += float(self._rng.uniform(-jitter_yaw, jitter_yaw))
            self._positions[env_id] = position
            self._yaws[env_id] = float((yaw + pi) % (2.0 * pi) - pi)
            self._surges[env_id] = float(start.linear_velocity_body_mps[0])
            self._yaw_rates[env_id] = 0.0
            self._time_s[env_id] = 0.0
            self._episode_ids[env_id] = self._next_episode_id
            self._next_episode_id += 1
            self._episode_steps[env_id] = 0
            self._scenario_keys[env_id] = key
            self._goals[env_id] = scenario.goal_ned_m
            self._goal_radii[env_id] = scenario.goal_radius_m
            self._initial_goal_distance[env_id] = max(
                float(np.linalg.norm(scenario.goal_ned_m - position)),
                1e-6,
            )
            self._previous_goal_distance[env_id] = self._initial_goal_distance[env_id]
            self._path_length[env_id] = 0.0
            self._previous_action[env_id] = 0.0
            self._reference_speed[env_id] = float(start.linear_velocity_body_mps[0])
            self._action_clip_count[env_id] = 0
            self._action_clipped_steps[env_id] = 0
            self._action_step_count[env_id] = 0
            self._action_saturation_counts[env_id] = 0
            self._action_abs_sum[env_id] = 0.0
            self._action_abs_max[env_id] = 0.0
            self._start_jitter_xy[env_id] = float(jitter_xy)
            self._start_jitter_yaw[env_id] = float(jitter_yaw)

        clearance, _, _ = self._batch_clearance(
            self._positions[indices],
            self._yaws[indices],
            self._scenario_keys[indices],
        )
        self._last_clearance[indices] = clearance
        self._previous_clearance[indices] = clearance
        self._min_clearance[indices] = clearance

    def _resolve_scenario_keys(
        self,
        count: int,
        scenario_key: object,
    ) -> tuple[str, ...]:
        if scenario_key is None:
            return tuple(
                self.config.active_scenario_keys[index % len(self.config.active_scenario_keys)]
                for index in range(count)
            )
        if isinstance(scenario_key, str):
            keys = (scenario_key,) * count
        else:
            if not isinstance(scenario_key, (tuple, list)):
                raise TypeError("options['scenario_key'] must be a string or sequence")
            keys = tuple(str(item) for item in scenario_key)
            if len(keys) != count:
                raise ValueError("scenario_key sequence length must match reset size")
        missing = [key for key in keys if key not in self._case_by_key]
        if missing:
            raise ValueError(f"unknown scenario keys: {', '.join(missing)}")
        inactive = [key for key in keys if key not in self.config.active_scenario_keys]
        if inactive:
            raise ValueError(f"inactive scenario keys: {', '.join(inactive)}")
        return keys

    def _batch_clearance(
        self,
        positions: NDArray[np.float64],
        yaws: NDArray[np.float64],
        scenario_keys: NDArray[np.object_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        quats = np.zeros((positions.shape[0], 4), dtype=np.float64)
        quats[:, 2] = np.sin(0.5 * yaws)
        quats[:, 3] = np.cos(0.5 * yaws)
        centers = hull_sphere_centers_from_poses(positions, quats, self._hull)
        obstacle_clearance = np.full(positions.shape[0], np.inf, dtype=np.float64)
        boundary_clearance = np.full(positions.shape[0], np.inf, dtype=np.float64)
        for key in self.config.active_scenario_keys:
            group = np.asarray([str(item) == key for item in scenario_keys], dtype=bool)
            if not np.any(group):
                continue
            obstacle_map = self._case_by_key[key].scenario.world.obstacle_map
            group_obstacle = np.full(int(np.count_nonzero(group)), np.inf, dtype=np.float64)
            group_centers = centers[group]
            for obstacle in obstacle_map.obstacles:
                distances = obstacle_signed_distance_array(
                    group_centers.reshape(-1, 3),
                    obstacle,
                ).reshape(group_centers.shape[:2])
                group_obstacle = np.minimum(group_obstacle, np.min(distances, axis=1))
            group_obstacle -= self._hull.sphere_radius_m
            group_boundary = self._bounds_clearance(group_centers, obstacle_map)
            obstacle_clearance[group] = group_obstacle
            boundary_clearance[group] = group_boundary
        clearance = np.minimum(obstacle_clearance, boundary_clearance)
        return clearance, obstacle_clearance, boundary_clearance

    def _bounds_clearance(
        self,
        centers: NDArray[np.float64],
        obstacle_map: ObstacleMap,
    ) -> NDArray[np.float64]:
        bounds = np.asarray(obstacle_map.bounds_ned_m, dtype=np.float64)
        lower = bounds[:, 0].reshape(1, 1, 3)
        upper = bounds[:, 1].reshape(1, 1, 3)
        inside_distance = np.minimum(centers - lower, upper - centers)
        all_inside = np.all(inside_distance >= 0.0, axis=2)
        outside_offset = np.maximum(lower - centers, 0.0) + np.maximum(centers - upper, 0.0)
        outside_distance = -np.linalg.norm(outside_offset, axis=2)
        signed = np.where(all_inside, np.min(inside_distance, axis=2), outside_distance)
        return np.min(signed, axis=1) - self._hull.sphere_radius_m

    def _batch_observation(
        self,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        started = time.perf_counter()
        directions_body = self.perception.directions_body
        body_angles = np.arctan2(directions_body[:, 1], directions_body[:, 0])
        world_angles = body_angles.reshape(1, -1) + self._yaws.reshape(-1, 1)
        world_directions = np.stack(
            (
                np.cos(world_angles),
                np.sin(world_angles),
                np.zeros_like(world_angles),
            ),
            axis=-1,
        )
        origin = self._positions + np.column_stack(
            (
                self.config.sensor_origin_body_m[0] * np.cos(self._yaws),
                self.config.sensor_origin_body_m[0] * np.sin(self._yaws),
                np.zeros(self.num_envs, dtype=np.float64),
            )
        )
        ranges = np.full(
            (self.num_envs, self.config.sensor_ray_count),
            self.config.sensor_max_range_m,
            dtype=np.float64,
        )
        confidence = np.zeros_like(ranges)
        for key in self.config.active_scenario_keys:
            group = np.asarray([str(item) == key for item in self._scenario_keys], dtype=bool)
            if not np.any(group):
                continue
            obstacle_map = self._case_by_key[key].scenario.world.obstacle_map
            group_ranges, group_confidence = analytic_ray_ranges(
                origin[group],
                world_directions[group],
                tuple(obstacle_map.obstacles),
                self.perception.config,
            )
            ranges[group] = group_ranges
            confidence[group] = group_confidence
        sector_ranges = self.builder.sector_ranges_batch(ranges, directions_body)
        front_ranges = self.builder.nearest_range_batch(
            ranges,
            directions_body,
            self.config.front_obstacle_cone_rad,
        )
        observations = self.builder.build_batch(
            sector_ranges,
            self._positions,
            self._yaws,
            self._surges,
            self._yaw_rates,
            self._goals,
            self._initial_goal_distance,
        )
        self.last_sensor_time_s = time.perf_counter() - started
        return observations, ranges, confidence, sector_ranges, front_ranges

    def _reward_terms(
        self,
        *,
        goal_distance: NDArray[np.float64],
        initial_goal_distance: NDArray[np.float64],
        front_ranges: NDArray[np.float64],
        collision: NDArray[np.bool_],
        out_of_bounds: NDArray[np.bool_],
        goal_reached: NDArray[np.bool_],
        previous_goal_distance: NDArray[np.float64],
        previous_clearance: NDArray[np.float64],
        current_clearance: NDArray[np.float64],
        surge: NDArray[np.float64],
        normalized_action: NDArray[np.float64],
        previous_normalized_action: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        config = self.config.reward
        if not isinstance(config, PaperRewardConfig):
            raise TypeError("reward config must be a PaperRewardConfig")
        reward_range = self.config.reward_obstacle_range_m
        step_scale = self.config.reward_step_scale
        distance_reward: NDArray[np.float64]
        obstacle_reward: NDArray[np.float64]
        goal_progress_reward: NDArray[np.float64]
        reference_speed_reward: NDArray[np.float64]
        action_smoothness_reward: NDArray[np.float64]
        if config.reward_version == "paper":
            distance_reward = config.distance_penalty_factor * np.tanh(
                goal_distance / initial_goal_distance
            )
            threshold = config.obstacle_influence_fraction * reward_range
            if config.continuous_obstacle_penalty:
                fraction = np.minimum(front_ranges / config.continuous_penalty_range_m, 1.0)
                obstacle_reward = config.near_obstacle_penalty * (1.0 - fraction)
            else:
                obstacle_reward = np.where(
                    front_ranges < threshold,
                    config.near_obstacle_penalty,
                    np.where(
                        front_ranges < reward_range,
                        config.far_obstacle_penalty,
                        0.0,
                    ),
                )
            distance_reward *= step_scale
            obstacle_reward *= step_scale
            goal_progress_reward = np.zeros(self.num_envs, dtype=np.float64)
            reference_speed_reward = np.zeros(self.num_envs, dtype=np.float64)
            action_smoothness_reward = np.zeros(self.num_envs, dtype=np.float64)
        else:
            distance_reward = np.zeros(self.num_envs, dtype=np.float64)
            previous_potential = np.maximum(0.0, config.safety_clearance_m - previous_clearance)
            current_potential = np.maximum(0.0, config.safety_clearance_m - current_clearance)
            goal_progress_reward = config.goal_progress_weight * (
                previous_goal_distance - goal_distance
            )
            obstacle_reward = config.obstacle_potential_weight * (
                previous_potential - current_potential
            )
            obstacle_reward *= step_scale
            reference_speed = (
                np.full(self.num_envs, config.reference_speed_mps, dtype=np.float64)
                if config.reference_speed_mps is not None
                else self._reference_speed
            )
            reference_speed_reward = -config.reference_speed_weight * (surge - reference_speed) ** 2
            reference_speed_reward *= step_scale
            action_smoothness_reward = -config.action_smoothness_weight * np.sum(
                (normalized_action - previous_normalized_action) ** 2,
                axis=1,
            )
            action_smoothness_reward *= step_scale
        terminal_reward = np.where(
            collision | out_of_bounds,
            config.collision_penalty,
            np.where(goal_reached, config.goal_bonus, 0.0),
        )
        total = (
            distance_reward
            + obstacle_reward
            + terminal_reward
            + goal_progress_reward
            + reference_speed_reward
            + action_smoothness_reward
        )
        return {
            "total": total,
            "distance": distance_reward,
            "obstacle": obstacle_reward,
            "terminal": terminal_reward,
            "goal_progress": goal_progress_reward,
            "reference_speed": reference_speed_reward,
            "action_smoothness": action_smoothness_reward,
        }

    def _reset_info(self) -> dict[str, Any]:
        return {
            "scenario_key": self._scenario_keys.copy(),
            "episode_id": self._episode_ids.copy(),
            "rl_step": self._episode_steps.copy(),
            "observation_source": self.observation_source,
            "perception_mode": self.perception.perception_mode,
            "perception_config_hash": self.perception.config_hash,
            "initial_goal_distance_m": self._initial_goal_distance.copy(),
        }

    def _step_info(
        self,
        *,
        goal_distance: NDArray[np.float64],
        collision: NDArray[np.bool_],
        out_of_bounds: NDArray[np.bool_],
        goal_reached: NDArray[np.bool_],
        success: NDArray[np.bool_],
        truncated: NDArray[np.bool_],
        front_ranges: NDArray[np.float64],
        action_clipped: NDArray[np.bool_],
        reward_terms: dict[str, NDArray[np.float64]],
        observations: NDArray[np.float32],
    ) -> dict[str, Any]:
        failure_reason = np.full(self.num_envs, "in_progress", dtype=object)
        failure_reason[collision] = "collision"
        failure_reason[out_of_bounds] = "out_of_bounds"
        failure_reason[truncated] = "timeout"
        failure_reason[goal_reached] = None
        steps = np.maximum(self._action_step_count, 1)
        scenario_names = np.asarray(
            [self._case_by_key[str(key)].name for key in self._scenario_keys],
            dtype=object,
        )
        info: dict[str, Any] = {
            "scenario_key": self._scenario_keys.copy(),
            "scenario_name": scenario_names,
            "episode_id": self._episode_ids.copy(),
            "rl_step": self._episode_steps.copy(),
            "simulation_time_s": self._time_s.copy(),
            "success": success.copy(),
            "goal_reached": goal_reached.copy(),
            "collision": collision.copy(),
            "out_of_bounds": out_of_bounds.copy(),
            "timeout": truncated.copy(),
            "failure_reason": failure_reason,
            "goal_distance_m": goal_distance.copy(),
            "initial_goal_distance_m": self._initial_goal_distance.copy(),
            "min_clearance_m": self._min_clearance.copy(),
            "path_length_m": self._path_length.copy(),
            "front_obstacle_range_m": front_ranges.copy(),
            "action_clipped": action_clipped.copy(),
            "action_u_dot_mps2": self._previous_action[:, 0].copy(),
            "action_yaw_rate_rps": self._previous_action[:, 1].copy(),
            "action_yaw_rate_deg_s": np.degrees(self._previous_action[:, 1]),
            "action_clip_count": self._action_clip_count.copy(),
            "action_clipped_step_count": self._action_clipped_steps.copy(),
            "action_u_dot_saturation_ratio": self._action_saturation_counts[:, 0] / steps,
            "action_yaw_rate_saturation_ratio": self._action_saturation_counts[:, 1] / steps,
            "action_u_dot_max_abs": self._action_abs_max[:, 0].copy(),
            "action_yaw_rate_max_abs": self._action_abs_max[:, 1].copy(),
            "action_u_dot_mean_abs": self._action_abs_sum[:, 0] / steps,
            "action_yaw_rate_mean_abs": self._action_abs_sum[:, 1] / steps,
            "start_jitter_xy_m": self._start_jitter_xy.copy(),
            "start_jitter_yaw_rad": self._start_jitter_yaw.copy(),
            "observation_source": self.observation_source,
            "perception_mode": self.perception.perception_mode,
            "perception_config_hash": self.perception.config_hash,
            "sensor_scan_valid": np.ones(self.num_envs, dtype=np.float64),
            "sensor_points_total": self._sensor_points_total.copy(),
            "sensor_points_used": self._sensor_points_used.copy(),
            "sensor_empty_rays": self._sensor_empty_rays.copy(),
            "sensor_frame_age_s": self._sensor_frame_age.copy(),
            "sensor_scan_returned_false": np.zeros(self.num_envs, dtype=np.float64),
            "reward_total": reward_terms["total"].copy(),
            "reward_distance": reward_terms["distance"].copy(),
            "reward_obstacle": reward_terms["obstacle"].copy(),
            "reward_terminal": reward_terms["terminal"].copy(),
            "reward_goal_progress": reward_terms["goal_progress"].copy(),
            "reward_reference_speed": reward_terms["reference_speed"].copy(),
            "reward_action_smoothness": reward_terms["action_smoothness"].copy(),
        }
        return info

    def _scalar_info(self, info: dict[str, Any], env_id: int) -> dict[str, Any]:
        keep_arrays = {"final_observations", "reset_mask"}
        row: dict[str, Any] = {}
        for key, value in info.items():
            if key in keep_arrays:
                continue
            if (
                isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[0] == self.num_envs
            ):
                cell = value[env_id]
                copy_method = getattr(cell, "copy", None)
                row[key] = copy_method() if callable(copy_method) else cell
            elif isinstance(value, np.generic):
                row[key] = value.item()
            else:
                row[key] = value
        return row


__all__ = ["P1TensorVectorEnv"]
