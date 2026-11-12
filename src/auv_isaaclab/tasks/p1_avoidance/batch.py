"""Torch batch core for the P1 planar environment.

The core is intentionally independent of IsaacLab.  It owns the two-dimensional
state, analytic 13-sector sensor, collision/clearance logic, reward and history
buffer.  G3 will wrap it in ``DirectRLEnv``; keeping the math here makes the
``N=1`` parity test runnable on CPU without Omniverse.

S1/S2 are the fixed regression scenarios.  C1-C3 are deterministic groups of
curriculum layouts; the training path may resample a group member on reset,
while evaluation can pin a specific member through ``assign_scenarios``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from auv_core.geometry import hull_from_vehicle
from torch import Tensor

from auv_isaaclab.config import P1AvoidanceEnvCfg
from auv_isaaclab.tasks.p1_avoidance.scenarios import build_scenario_groups
from auv_rl.scenarios import PlanarScenarioCase


def _wrap_angle(value: Tensor) -> Tensor:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _repeat_keys(keys: str | Sequence[str], count: int) -> list[str]:
    if isinstance(keys, str):
        return [keys] * count
    result = list(keys)
    if len(result) == 1:
        return result * count
    if len(result) != count:
        raise ValueError(f"expected 1 or {count} scenario keys, got {len(result)}")
    return result


def _weighted_keys(num_envs: int, mix: dict[str, float], seed: int) -> list[str]:
    active = [(key, weight) for key, weight in mix.items() if weight > 0.0]
    if not active:
        raise ValueError("curriculum.scenario_mix has no active scenario")
    raw = [(key, weight * num_envs) for key, weight in active]
    counts = {key: int(math.floor(value)) for key, value in raw}
    remainder = num_envs - sum(counts.values())
    ranked = sorted(raw, key=lambda item: item[1] - math.floor(item[1]), reverse=True)
    for index in range(remainder):
        counts[ranked[index % len(ranked)][0]] += 1
    keys: list[str] = []
    for key, _ in active:
        keys.extend([key] * counts[key])
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(num_envs, generator=generator).tolist()
    return [keys[index] for index in order]


@dataclass(frozen=True)
class BatchReset:
    observation: Tensor
    raw_observation: Tensor
    info: dict[str, Any]


@dataclass(frozen=True)
class BatchStep:
    observation: Tensor
    raw_observation: Tensor
    reward: Tensor
    ppo_reward: Tensor
    terminated: Tensor
    truncated: Tensor
    info: dict[str, Any]


class P1BatchCore:
    """Vectorized P1 state, sensor, reward and history buffer."""

    def __init__(
        self,
        cfg: P1AvoidanceEnvCfg,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        scenario_keys: str | Sequence[str] | None = None,
        scenario_variant_keys: str | Sequence[str] | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(cfg, P1AvoidanceEnvCfg):
            raise TypeError("cfg must be a P1AvoidanceEnvCfg")
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_envs = cfg.env.num_envs
        self._scenario_groups = build_scenario_groups()
        self._cases = {case.key: case for group in self._scenario_groups.values() for case in group}
        self._case_data = {key: self._extract_case(case) for key, case in self._cases.items()}
        self._max_cylinders = max(
            (len(data["cylinders"]) for data in self._case_data.values()), default=0
        )
        self._max_boxes = max((len(data["boxes"]) for data in self._case_data.values()), default=0)
        self._max_spheres = max(
            (len(data["spheres"]) for data in self._case_data.values()), default=0
        )
        self._max_cylinders = max(self._max_cylinders, 1)
        self._max_boxes = max(self._max_boxes, 1)
        self._max_spheres = max(self._max_spheres, 1)

        vehicles = {
            case.scenario.vehicle.name: case.scenario.vehicle for case in self._cases.values()
        }
        if len(vehicles) != 1:
            raise ValueError("all P1 scenarios must use the same vehicle model")
        hull = hull_from_vehicle(next(iter(vehicles.values())))
        self._hull_radius = float(hull.sphere_radius_m)
        self._hull_offsets = torch.as_tensor(
            hull.offsets_body_m[:, 0],
            device=self.device,
            dtype=self.dtype,
        )
        self._sensor_angles = -0.5 * cfg.observation.field_of_view_rad + (
            torch.arange(cfg.observation.ray_count, device=self.device, dtype=self.dtype) + 0.5
        ) * (cfg.observation.field_of_view_rad / cfg.observation.ray_count)
        self._front_mask = torch.abs(self._sensor_angles) <= (30.0 * math.pi / 180.0 + 1e-12)
        if cfg.observation.ray_count % cfg.observation.sector_count != 0:
            raise ValueError("ray_count must be divisible by sector_count")

        self.positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self.yaws = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.surges = torch.ones(self.num_envs, device=self.device, dtype=self.dtype)
        self.yaw_rates = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.time_s = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.initial_goal_distance = torch.ones(self.num_envs, device=self.device, dtype=self.dtype)
        self.previous_goal_distance = torch.ones(
            self.num_envs, device=self.device, dtype=self.dtype
        )
        self.previous_clearance = torch.full(
            (self.num_envs,), float("inf"), device=self.device, dtype=self.dtype
        )
        self.last_clearance = torch.full(
            (self.num_envs,), float("inf"), device=self.device, dtype=self.dtype
        )
        self.min_clearance = torch.full(
            (self.num_envs,), float("inf"), device=self.device, dtype=self.dtype
        )
        self.path_length = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.previous_action = torch.zeros((self.num_envs, 2), device=self.device, dtype=self.dtype)
        self.terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.truncated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.history = torch.zeros(
            (self.num_envs, cfg.observation.history_length, cfg.raw_observation_dim),
            device=self.device,
            dtype=self.dtype,
        )

        self.starts = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._case_start_yaws = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.goals = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self.goal_radii = torch.ones(self.num_envs, device=self.device, dtype=self.dtype)
        self.start_speeds = torch.ones(self.num_envs, device=self.device, dtype=self.dtype)
        self.bounds = torch.zeros((self.num_envs, 3, 2), device=self.device, dtype=self.dtype)
        self.cyl_centers = torch.zeros(
            (self.num_envs, self._max_cylinders, 3), device=self.device, dtype=self.dtype
        )
        self.cyl_radii = torch.zeros(
            (self.num_envs, self._max_cylinders), device=self.device, dtype=self.dtype
        )
        self.cyl_heights = torch.zeros(
            (self.num_envs, self._max_cylinders), device=self.device, dtype=self.dtype
        )
        self.cyl_mask = torch.zeros(
            (self.num_envs, self._max_cylinders), device=self.device, dtype=torch.bool
        )
        self.box_centers = torch.zeros(
            (self.num_envs, self._max_boxes, 3), device=self.device, dtype=self.dtype
        )
        self.box_half_extents = torch.zeros(
            (self.num_envs, self._max_boxes, 3), device=self.device, dtype=self.dtype
        )
        self.box_mask = torch.zeros(
            (self.num_envs, self._max_boxes), device=self.device, dtype=torch.bool
        )
        self.sphere_centers = torch.zeros(
            (self.num_envs, self._max_spheres, 3), device=self.device, dtype=self.dtype
        )
        self.sphere_radii = torch.zeros(
            (self.num_envs, self._max_spheres), device=self.device, dtype=self.dtype
        )
        self.sphere_mask = torch.zeros(
            (self.num_envs, self._max_spheres), device=self.device, dtype=torch.bool
        )

        selected = _repeat_keys(
            scenario_keys
            if scenario_keys is not None
            else _weighted_keys(self.num_envs, cfg.curriculum.scenario_mix, seed),
            self.num_envs,
        )
        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.scenario_keys = [""] * self.num_envs
        self.scenario_variant_keys = [""] * self.num_envs
        self.assign_scenarios(
            selected,
            env_ids=torch.arange(self.num_envs, device=self.device),
            variant_keys=scenario_variant_keys,
        )
        self._last_scan: tuple[Tensor, Tensor, Tensor, Tensor] | None = None

    def _extract_case(self, case: PlanarScenarioCase) -> dict[str, Any]:
        scenario = case.scenario
        obstacle_map = scenario.world.obstacle_map
        cylinders = [item for item in obstacle_map.obstacles if item.kind == "cylinder"]
        boxes = [item for item in obstacle_map.obstacles if item.kind == "box"]
        spheres = [item for item in obstacle_map.obstacles if item.kind == "sphere"]
        return {
            "start": torch.as_tensor(
                scenario.start.position_ned_m, device=self.device, dtype=self.dtype
            ),
            "start_speed": torch.as_tensor(
                scenario.start.linear_velocity_body_mps[0], device=self.device, dtype=self.dtype
            ),
            "start_yaw": torch.as_tensor(
                2.0
                * math.atan2(
                    float(scenario.start.quat_xyzw[2]),
                    float(scenario.start.quat_xyzw[3]),
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            "goal": torch.as_tensor(scenario.goal_ned_m, device=self.device, dtype=self.dtype),
            "goal_radius": torch.as_tensor(
                scenario.goal_radius_m, device=self.device, dtype=self.dtype
            ),
            "bounds": torch.as_tensor(
                obstacle_map.bounds_ned_m, device=self.device, dtype=self.dtype
            ),
            "cylinders": [
                (
                    torch.as_tensor(item.center_ned_m, device=self.device, dtype=self.dtype),
                    float(item.radius_m),  # type: ignore[arg-type]
                    float(item.height_m),  # type: ignore[arg-type]
                )
                for item in cylinders
            ],
            "boxes": [
                (
                    torch.as_tensor(item.center_ned_m, device=self.device, dtype=self.dtype),
                    torch.as_tensor(
                        item.half_extents_m,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                )
                for item in boxes
            ],
            "spheres": [
                (
                    torch.as_tensor(item.center_ned_m, device=self.device, dtype=self.dtype),
                    float(item.radius_m),  # type: ignore[arg-type]
                )
                for item in spheres
            ],
        }

    def assign_scenarios(
        self,
        keys: str | Sequence[str],
        *,
        env_ids: Tensor | None = None,
        variant_keys: str | Sequence[str] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = int(env_ids.numel())
        resolved = _repeat_keys(keys, count)
        for key in resolved:
            if key not in self._scenario_groups:
                raise ValueError(f"unknown scenario key: {key}")

        if variant_keys is not None:
            resolved_variants = _repeat_keys(variant_keys, count)
            for position, (key, variant) in enumerate(
                zip(resolved, resolved_variants, strict=True)
            ):
                group = self._scenario_groups[key]
                if variant is None:
                    if len(group) != 1:
                        raise ValueError(f"scenario {key!r} requires an explicit variant")
                    resolved_variants[position] = group[0].key
                elif variant not in {case.key for case in group}:
                    raise ValueError(f"scenario variant {variant!r} does not belong to {key!r}")
        else:
            resolved_variants = [""] * count
            for key in sorted(set(resolved)):
                positions = [index for index, item in enumerate(resolved) if item == key]
                group = self._scenario_groups[key]
                if len(group) == 1:
                    chosen = [group[0].key] * len(positions)
                else:
                    indices = torch.randint(
                        len(group),
                        (len(positions),),
                        generator=self._generator,
                        device=self.device,
                    ).tolist()
                    chosen = [group[int(index)].key for index in indices]
                for position, variant in zip(positions, chosen, strict=True):
                    resolved_variants[position] = variant

        for index, key, variant in zip(env_ids.tolist(), resolved, resolved_variants, strict=True):
            self.scenario_keys[index] = key
            self.scenario_variant_keys[index] = variant
        self._apply_case_arrays(env_ids, resolved_variants)

    def _apply_case_arrays(self, env_ids: Tensor, variant_keys: list[str]) -> None:
        unique = set(variant_keys)
        for key in unique:
            mask = torch.tensor(
                [item == key for item in variant_keys], device=self.device, dtype=torch.bool
            )
            ids = env_ids[mask]
            data = self._case_data[key]
            self.starts[ids] = data["start"]
            self._case_start_yaws[ids] = data["start_yaw"]
            self.start_speeds[ids] = data["start_speed"]
            self.goals[ids] = data["goal"]
            self.goal_radii[ids] = data["goal_radius"]
            self.bounds[ids] = data["bounds"]
            self.cyl_mask[ids] = False
            self.box_mask[ids] = False
            self.sphere_mask[ids] = False
            for slot, (center, radius, height) in enumerate(data["cylinders"]):
                self.cyl_centers[ids, slot] = center
                self.cyl_radii[ids, slot] = radius
                self.cyl_heights[ids, slot] = height
                self.cyl_mask[ids, slot] = True
            for slot, (center, half) in enumerate(data["boxes"]):
                self.box_centers[ids, slot] = center
                self.box_half_extents[ids, slot] = half
                self.box_mask[ids, slot] = True
            for slot, (center, radius) in enumerate(data["spheres"]):
                self.sphere_centers[ids, slot] = center
                self.sphere_radii[ids, slot] = radius
                self.sphere_mask[ids, slot] = True

    def reset(
        self,
        env_ids: Tensor | None = None,
        *,
        scenario_keys: str | Sequence[str] | None = None,
        variant_keys: str | Sequence[str] | None = None,
        jitter: bool | None = None,
        seed: int | None = None,
        resample: bool = False,
    ) -> BatchReset:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if seed is not None:
            self._generator.manual_seed(int(seed))
        if scenario_keys is not None:
            self.assign_scenarios(scenario_keys, env_ids=env_ids, variant_keys=variant_keys)
        elif variant_keys is not None:
            raise ValueError("variant_keys requires scenario_keys")
        elif resample:
            self.assign_scenarios(
                [self.scenario_keys[index] for index in env_ids.tolist()],
                env_ids=env_ids,
            )
        if jitter is None:
            use_jitter = (
                self.cfg.curriculum.start_jitter_xy_m > 0.0
                or self.cfg.curriculum.start_jitter_yaw_rad > 0.0
            )
        else:
            use_jitter = bool(jitter)
        starts = self.starts[env_ids].clone()
        if use_jitter:
            xy = self.cfg.curriculum.start_jitter_xy_m
            yaw_jitter = self.cfg.curriculum.start_jitter_yaw_rad
            noise = (
                torch.rand(
                    (env_ids.numel(), 2),
                    generator=self._generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                * 2.0
                - 1.0
            )
            starts[:, :2] += noise * xy
            yaw_noise = (
                torch.rand(
                    (env_ids.numel(),),
                    generator=self._generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                * 2.0
                - 1.0
            ) * yaw_jitter
        else:
            yaw_noise = torch.zeros(env_ids.numel(), device=self.device, dtype=self.dtype)
        self.positions[env_ids] = starts
        self.yaws[env_ids] = self._case_start_yaws[env_ids] + yaw_noise
        self.surges[env_ids] = self.start_speeds[env_ids]
        self.yaw_rates[env_ids] = 0.0
        self.time_s[env_ids] = 0.0
        self.episode_steps[env_ids] = 0
        self.episode_ids[env_ids] += 1
        goal_distance = torch.linalg.norm(self.goals[env_ids] - self.positions[env_ids], dim=1)
        goal_distance = goal_distance.clamp_min(1e-6)
        self.initial_goal_distance[env_ids] = goal_distance
        self.previous_goal_distance[env_ids] = goal_distance
        self.path_length[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.terminated[env_ids] = False
        self.truncated[env_ids] = False
        clearance, _, _ = self._clearance()
        self.previous_clearance[env_ids] = clearance[env_ids]
        self.last_clearance[env_ids] = clearance[env_ids]
        self.min_clearance[env_ids] = clearance[env_ids]
        raw, sector, front, _ = self._scan()
        self.history[env_ids] = (
            raw[env_ids].unsqueeze(1).repeat(1, self.cfg.observation.history_length, 1)
        )
        info = {
            "scenario_key": tuple(self.scenario_keys[index] for index in env_ids.tolist()),
            "episode_id": self.episode_ids[env_ids].clone(),
            "raw_observation": raw[env_ids].clone(),
            "sector_ranges": sector[env_ids].clone(),
            "front_range": front[env_ids].clone(),
        }
        return BatchReset(self.observation, raw, info)

    @property
    def observation(self) -> Tensor:
        return self.history.reshape(self.num_envs, -1)

    def step(self, actions: Tensor) -> BatchStep:
        if self.terminated.any() or self.truncated.any():
            raise RuntimeError("reset done environments before calling step again")
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"actions must have shape ({self.num_envs}, 2)")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        low = torch.tensor(
            [-self.cfg.env.action_max_accel_mps2, -self.cfg.env.action_max_yaw_rate_rps],
            device=self.device,
            dtype=self.dtype,
        )
        high = -low
        clipped = torch.clamp(actions.to(device=self.device, dtype=self.dtype), low, high)
        self.previous_action = clipped.clone()

        substeps = int(round(self.cfg.env.dt_s / self.cfg.env.integration_step_s))
        h = self.cfg.env.integration_step_s
        collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        goal_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        current_clearance = self.last_clearance.clone()
        for _ in range(substeps):
            active = ~(collision | out_of_bounds | goal_reached)
            if not active.any():
                break
            yaw = self.yaws
            next_surge = torch.clamp(
                self.surges + clipped[:, 0] * h,
                min=1.0,
                max=2.5,
            )
            next_yaw = _wrap_angle(yaw + clipped[:, 1] * h)
            step = torch.stack(
                (
                    next_surge * torch.cos(yaw) * h,
                    next_surge * torch.sin(yaw) * h,
                    torch.zeros_like(next_surge),
                ),
                dim=1,
            )
            next_positions = self.positions + step
            self.path_length += torch.where(
                active,
                torch.linalg.norm(next_positions - self.positions, dim=1),
                torch.zeros_like(self.path_length),
            )
            self.positions = torch.where(active.unsqueeze(1), next_positions, self.positions)
            self.yaws = torch.where(active, next_yaw, self.yaws)
            self.surges = torch.where(active, next_surge, self.surges)
            self.yaw_rates = torch.where(active, clipped[:, 1], self.yaw_rates)
            self.time_s = self.time_s + active.to(self.dtype) * h
            clearance, obstacle_clearance, boundary_clearance = self._clearance()
            current_clearance = torch.where(active, clearance, current_clearance)
            self.last_clearance = torch.where(active, clearance, self.last_clearance)
            self.min_clearance = torch.where(
                active,
                torch.minimum(self.min_clearance, clearance),
                self.min_clearance,
            )
            collision = collision | (active & (obstacle_clearance < 0.0))
            out_of_bounds = out_of_bounds | (active & (boundary_clearance < 0.0))
            goal_distance = torch.linalg.norm(self.goals - self.positions, dim=1)
            goal_reached = goal_reached | (
                active & ~collision & ~out_of_bounds & (goal_distance <= self.goal_radii + 1e-9)
            )

        self.episode_steps += 1
        self.terminated = collision | out_of_bounds | goal_reached
        self.truncated = (~self.terminated) & (self.episode_steps >= self.cfg.env.max_episode_steps)
        raw, sector, front, _ = self._scan()
        self.history = torch.cat((self.history[:, 1:], raw.unsqueeze(1)), dim=1)
        goal_distance = torch.linalg.norm(self.goals - self.positions, dim=1)
        reward_terms = self._reward_terms(
            goal_distance=goal_distance,
            front_range=front,
            collision=collision,
            out_of_bounds=out_of_bounds,
            goal_reached=goal_reached,
            truncated=self.truncated,
        )
        total = reward_terms["reward_total"]
        success = goal_reached & ~collision & ~out_of_bounds & ~self.truncated
        info = {
            "scenario_key": tuple(self.scenario_keys),
            "scenario_variant": tuple(self.scenario_variant_keys),
            "episode_id": self.episode_ids.clone(),
            "rl_step": self.episode_steps.clone(),
            "simulation_time_s": self.time_s.clone(),
            "success": success,
            "goal_reached": goal_reached,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "timeout": self.truncated,
            "goal_distance_m": goal_distance,
            "initial_goal_distance_m": self.initial_goal_distance.clone(),
            "min_clearance_m": self.min_clearance.clone(),
            "path_length_m": self.path_length.clone(),
            "front_obstacle_range_m": front,
            "action_u_dot_mps2": self.previous_action[:, 0].clone(),
            "action_yaw_rate_rps": self.previous_action[:, 1].clone(),
            "sector_ranges_m": sector,
            **{key: value.clone() for key, value in reward_terms.items()},
        }
        self.previous_goal_distance = goal_distance
        self.previous_clearance = self.last_clearance
        return BatchStep(
            observation=self.observation,
            raw_observation=raw,
            reward=total,
            ppo_reward=total * self.cfg.reward.ppo_reward_scale,
            terminated=self.terminated.clone(),
            truncated=self.truncated.clone(),
            info=info,
        )

    def _scan(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        origins = self.positions + torch.stack(
            (
                self.cfg.observation.origin_body_m[0] * torch.cos(self.yaws),
                self.cfg.observation.origin_body_m[0] * torch.sin(self.yaws),
                torch.zeros_like(self.yaws),
            ),
            dim=1,
        )
        world_angles = self._sensor_angles.unsqueeze(0) + self.yaws.unsqueeze(1)
        directions = torch.stack(
            (
                torch.cos(world_angles),
                torch.sin(world_angles),
                torch.zeros_like(world_angles),
            ),
            dim=-1,
        )
        ray_ranges = self._cast_ranges(origins, directions)
        sector = ray_ranges.reshape(
            self.num_envs,
            self.cfg.observation.sector_count,
            self.cfg.observation.ray_count // self.cfg.observation.sector_count,
        ).amin(dim=2)
        front = torch.where(
            self._front_mask.unsqueeze(0),
            ray_ranges,
            torch.full_like(ray_ranges, self.cfg.env.sensor_max_range_m),
        ).amin(dim=1)
        goal_delta = self.goals - self.positions
        goal_distance = torch.linalg.norm(goal_delta, dim=1)
        bearing = _wrap_angle(torch.atan2(goal_delta[:, 1], goal_delta[:, 0]) - self.yaws)
        raw = torch.stack(
            (
                *(sector / self.cfg.env.sensor_max_range_m).unbind(dim=1),
                (goal_distance / self.initial_goal_distance).clamp(0.0, 1.0),
                (bearing / math.pi).clamp(-1.0, 1.0),
                ((self.surges - 1.0) / 1.5).clamp(0.0, 1.0),
                (self.yaw_rates / 0.5).clamp(-1.0, 1.0),
            ),
            dim=1,
        )
        self._last_scan = (raw, sector, front, sector)
        return raw, sector, front, sector

    def _cast_ranges(self, origins: Tensor, directions: Tensor) -> Tensor:
        nearest = torch.full(
            (self.num_envs, self.cfg.observation.ray_count),
            self.cfg.env.sensor_max_range_m,
            device=self.device,
            dtype=self.dtype,
        )
        if self.cyl_mask.any():
            ox = origins[:, 0, None, None] - self.cyl_centers[:, None, :, 0]
            oy = origins[:, 1, None, None] - self.cyl_centers[:, None, :, 1]
            oz = origins[:, 2, None, None]
            dx = directions[:, :, None, 0]
            dy = directions[:, :, None, 1]
            dz = directions[:, :, None, 2]
            a = dx * dx + dy * dy
            b = 2.0 * (ox * dx + oy * dy)
            c = ox * ox + oy * oy - self.cyl_radii[:, None, :] ** 2
            discriminant = b * b - 4.0 * a * c
            root = torch.sqrt(torch.clamp(discriminant, min=0.0))
            safe_a = torch.where(a > 1e-15, a, torch.ones_like(a))
            first = (-b - root) / (2.0 * safe_a)
            second = (-b + root) / (2.0 * safe_a)
            z_lower = self.cyl_centers[:, None, :, 2] - 0.5 * self.cyl_heights[:, None, :]
            z_upper = self.cyl_centers[:, None, :, 2] + 0.5 * self.cyl_heights[:, None, :]
            candidate = torch.full_like(b, float("inf"))
            for value in (first, second):
                z_hit = oz + value * dz
                valid = (
                    (a > 1e-15)
                    & (discriminant >= 0.0)
                    & (value >= 0.0)
                    & (z_hit >= z_lower - 1e-12)
                    & (z_hit <= z_upper + 1e-12)
                )
                candidate = torch.where(valid, torch.minimum(candidate, value), candidate)
            for z_plane in (z_lower, z_upper):
                cap_t = (z_plane - oz) / torch.where(torch.abs(dz) > 1e-15, dz, torch.ones_like(dz))
                cap_x = origins[:, 0, None, None] + cap_t * dx - self.cyl_centers[:, None, :, 0]
                cap_y = origins[:, 1, None, None] + cap_t * dy - self.cyl_centers[:, None, :, 1]
                valid = (
                    (torch.abs(dz) > 1e-15)
                    & (cap_t >= 0.0)
                    & (cap_x * cap_x + cap_y * cap_y <= self.cyl_radii[:, None, :] ** 2 + 1e-12)
                )
                candidate = torch.where(valid, torch.minimum(candidate, cap_t), candidate)
            candidate = torch.where(self.cyl_mask[:, None, :], candidate, float("inf"))
            nearest = torch.minimum(nearest, candidate.amin(dim=2))

        if self.box_mask.any():
            local = origins[:, None, None, :] - self.box_centers[:, None, :, :]
            near = torch.full(
                (self.num_envs, self.cfg.observation.ray_count, self._max_boxes),
                -float("inf"),
                device=self.device,
                dtype=self.dtype,
            )
            far = torch.full_like(near, float("inf"))
            for axis in range(3):
                component = directions[:, :, None, axis]
                origin = local[..., axis]
                half = self.box_half_extents[:, None, :, axis]
                parallel = torch.abs(component) <= 1e-15
                inside = (origin >= -half) & (origin <= half)
                first = (-half - origin) / torch.where(
                    parallel, torch.ones_like(component), component
                )
                second = (half - origin) / torch.where(
                    parallel, torch.ones_like(component), component
                )
                axis_near = torch.minimum(first, second)
                axis_far = torch.maximum(first, second)
                near = torch.where(parallel & inside, near, torch.maximum(near, axis_near))
                far = torch.where(parallel & inside, far, torch.minimum(far, axis_far))
                far = torch.where(parallel & ~inside, torch.full_like(far, -float("inf")), far)
            valid = (near <= far) & (far >= 0.0) & self.box_mask[:, None, :]
            candidate = torch.where(valid, torch.clamp(near, min=0.0), float("inf"))
            nearest = torch.minimum(nearest, candidate.amin(dim=2))

        if self.sphere_mask.any():
            offset = origins[:, None, None, :] - self.sphere_centers[:, None, :, :]
            b = (offset * directions[:, :, None, :]).sum(dim=-1)
            c = (offset * offset).sum(dim=-1) - self.sphere_radii[:, None, :] ** 2
            discriminant = b * b - c
            root = torch.sqrt(torch.clamp(discriminant, min=0.0))
            first = -b - root
            second = -b + root
            candidate = torch.where(
                first >= 0.0,
                first,
                torch.where(second >= 0.0, second, torch.full_like(first, float("inf"))),
            )
            candidate = torch.where(
                (discriminant >= 0.0) & self.sphere_mask[:, None, :],
                candidate,
                float("inf"),
            )
            nearest = torch.minimum(nearest, candidate.amin(dim=2))

        hit = (nearest >= self.cfg.observation.min_range_m - 1e-12) & (
            nearest <= self.cfg.env.sensor_max_range_m + 1e-12
        )
        return torch.where(hit, nearest, self.cfg.env.sensor_max_range_m)

    def _clearance(self) -> tuple[Tensor, Tensor, Tensor]:
        cos_yaw = torch.cos(self.yaws).unsqueeze(1)
        sin_yaw = torch.sin(self.yaws).unsqueeze(1)
        centers = torch.stack(
            (
                self.positions[:, None, 0] + self._hull_offsets[None, :] * cos_yaw,
                self.positions[:, None, 1] + self._hull_offsets[None, :] * sin_yaw,
                self.positions[:, None, 2].expand(-1, self._hull_offsets.numel()),
            ),
            dim=-1,
        )
        obstacle_sd = torch.full(
            (self.num_envs, centers.shape[1]), float("inf"), device=self.device, dtype=self.dtype
        )
        if self.cyl_mask.any():
            local = centers[:, :, None, :] - self.cyl_centers[:, None, :, :]
            radial = torch.hypot(local[..., 0], local[..., 1]) - self.cyl_radii[:, None, :]
            axial = torch.abs(local[..., 2]) - 0.5 * self.cyl_heights[:, None, :]
            outside = torch.hypot(torch.clamp(radial, min=0.0), torch.clamp(axial, min=0.0))
            inside = torch.clamp(torch.maximum(radial, axial), max=0.0)
            sd = torch.where(self.cyl_mask[:, None, :], outside + inside, float("inf"))
            obstacle_sd = torch.minimum(obstacle_sd, sd.amin(dim=2))
        if self.box_mask.any():
            local = centers[:, :, None, :] - self.box_centers[:, None, :, :]
            delta = torch.abs(local) - self.box_half_extents[:, None, :, :]
            outside = torch.linalg.norm(torch.clamp(delta, min=0.0), dim=-1)
            inside = torch.clamp(delta.amax(dim=-1), max=0.0)
            sd = torch.where(self.box_mask[:, None, :], outside + inside, float("inf"))
            obstacle_sd = torch.minimum(obstacle_sd, sd.amin(dim=2))
        if self.sphere_mask.any():
            local = centers[:, :, None, :] - self.sphere_centers[:, None, :, :]
            sd = torch.linalg.norm(local, dim=-1) - self.sphere_radii[:, None, :]
            sd = torch.where(self.sphere_mask[:, None, :], sd, float("inf"))
            obstacle_sd = torch.minimum(obstacle_sd, sd.amin(dim=2))
        obstacle_clearance = obstacle_sd.amin(dim=1) - self._hull_radius

        lower = self.bounds[:, :, 0].unsqueeze(1)
        upper = self.bounds[:, :, 1].unsqueeze(1)
        inside_distance = torch.minimum(centers - lower, upper - centers)
        all_inside = (inside_distance >= 0.0).all(dim=2)
        outside_offset = torch.clamp(lower - centers, min=0.0) + torch.clamp(
            centers - upper, min=0.0
        )
        outside_distance = -torch.linalg.norm(outside_offset, dim=2)
        signed = torch.where(all_inside, inside_distance.amin(dim=2), outside_distance)
        boundary_clearance = signed.amin(dim=1) - self._hull_radius
        clearance = torch.minimum(obstacle_clearance, boundary_clearance)
        return clearance, obstacle_clearance, boundary_clearance

    def _reward_terms(
        self,
        *,
        goal_distance: Tensor,
        front_range: Tensor,
        collision: Tensor,
        out_of_bounds: Tensor,
        goal_reached: Tensor,
        truncated: Tensor,
    ) -> dict[str, Tensor]:
        scale = self.cfg.reward.reward_step_scale
        distance = (
            self.cfg.reward.distance_penalty_factor
            * torch.tanh(goal_distance / self.initial_goal_distance)
            * scale
        )
        threshold = (
            self.cfg.reward.obstacle_influence_fraction * self.cfg.reward.reward_obstacle_range_m
        )
        obstacle = (
            torch.where(
                front_range < threshold,
                self.cfg.reward.near_obstacle_penalty,
                torch.where(
                    front_range < self.cfg.reward.reward_obstacle_range_m,
                    self.cfg.reward.far_obstacle_penalty,
                    torch.zeros_like(front_range),
                ),
            )
            * scale
        )
        terminal = torch.where(
            collision | out_of_bounds,
            torch.full_like(distance, self.cfg.reward.collision_penalty),
            torch.where(
                goal_reached,
                torch.full_like(distance, self.cfg.reward.goal_bonus),
                torch.where(
                    truncated,
                    torch.full_like(distance, self.cfg.reward.timeout_penalty),
                    torch.zeros_like(distance),
                ),
            ),
        )
        action_norm = torch.stack(
            (
                self.previous_action[:, 0] / self.cfg.env.action_max_accel_mps2,
                self.previous_action[:, 1] / self.cfg.env.action_max_yaw_rate_rps,
            ),
            dim=1,
        )
        action_penalty = self.cfg.reward.action_penalty_factor * action_norm.square().sum(dim=1)
        total = distance + obstacle + terminal + action_penalty
        return {
            "reward_total": total,
            "reward_distance": distance,
            "reward_obstacle": obstacle,
            "reward_terminal": terminal,
            "reward_action_penalty": action_penalty,
        }


__all__ = ["BatchReset", "BatchStep", "P1BatchCore"]
