"""Pure-controller closed loop with synthetic sensors and ALOS guidance."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from auv_core.scenarios import remus100_vehicle
from torch import Tensor

from auv_lowlevel.actions import ActuatorLimits
from auv_lowlevel.config import LowLevelConfig
from auv_lowlevel.curriculum import (
    CURRICULUM_STAGE_ORDER,
    CurriculumEpisodeSpec,
    CurriculumStageSampler,
    EpisodeManifestWriter,
)
from auv_lowlevel.path import (
    generate_curriculum_path,
    generate_pc_path_bank,
    resample_reference_path,
)

from ..actions import BatchedPureActionApplication, BatchedPureActionPipeline
from ..dynamics import BatchedRemus100Dynamics, Remus100BatchState
from ..geometry import (
    body_to_world,
    ground_velocity_body,
    legacy_ground_velocity_body,
    quaternion_from_direction,
    quaternion_from_euler,
    water_velocity_body,
    world_to_body,
    wrap_angle,
)
from ..guidance import BatchedAlosSolution, BatchedThreeDAlosGuidance
from ..path import (
    BatchedPathProjection,
    BatchedReferencePath,
    BatchedReferencePathBank,
    BatchedReferencePathSet,
)
from ..pc import (
    BatchedCommandHoldTracker,
    BatchedPCRewardComputer,
    BatchedPCTerminationManager,
    PCRewardBreakdown,
    PCTerminalKind,
    PCTerminationDecision,
    pc_route_health_terms,
    pc_terminal_kind_name,
)
from ..pc_path import (
    PC_PATH_DEFAULT_BLOCKS,
    PC_PATH_DIMS,
    PC_PATH_ENDPOINT_BLOCKS,
    build_pc_path_features,
    path_block_column_mask,
    validate_path_blocks,
)
from ..plant_randomization import sample_plant_scale_factors
from ..randomization import (
    BatchedEpisodeRandomization,
    TorchEpisodeRandomizer,
    TrainingSeedCursor,
    sample_episode_randomizations,
)
from ..sensors import SyntheticSensorOutput, SyntheticSensorSuite, pc_critic_state


@dataclass(frozen=True)
class PCCommand:
    heading_rad: Tensor
    pitch_rad: Tensor
    speed_mps: Tensor

    def normalized(self) -> Tensor:
        return torch.stack(
            [
                wrap_angle(self.heading_rad) / math.pi,
                self.pitch_rad / (math.pi / 2.0),
                self.speed_mps / 2.5,
            ],
            dim=1,
        )


@dataclass(frozen=True)
class BatchedPCObservation:
    actor_state: Tensor
    actor_path: Tensor | None
    critic_state: Tensor
    command: PCCommand
    actor_current_body: Tensor
    critic_current_ned: Tensor
    ground_velocity_body_mps: Tensor
    water_velocity_body_mps: Tensor
    sensor: SyntheticSensorOutput
    projection: BatchedPathProjection | None
    solution: BatchedAlosSolution | None


@dataclass(frozen=True)
class BatchedPCLowLevelStep:
    observation: BatchedPCObservation
    reward: PCRewardBreakdown
    terminated: Tensor
    truncated: Tensor
    info: dict[str, Any]

    @property
    def done(self) -> Tensor:
        return self.terminated | self.truncated


_MISSING_ROW_STATE = object()


def _clone_row_state(value: Any, count: int) -> Any:
    """Clone row-aligned tensors from a nested dataclass/dict state tree."""

    if isinstance(value, Tensor):
        if value.ndim >= 1 and value.shape[0] == count:
            return value.clone()
        return _MISSING_ROW_STATE
    if is_dataclass(value) and not isinstance(value, type):
        cloned: dict[str, Any] = {}
        for field in fields(value):
            state = _clone_row_state(getattr(value, field.name), count)
            if state is not _MISSING_ROW_STATE:
                cloned[field.name] = state
        if cloned:
            return ("dataclass", type(value), cloned)
        return _MISSING_ROW_STATE
    if isinstance(value, dict):
        cloned = {}
        for key, item in value.items():
            state = _clone_row_state(item, count)
            if state is not _MISSING_ROW_STATE:
                cloned[key] = state
        return cloned if cloned else _MISSING_ROW_STATE
    return _MISSING_ROW_STATE


def _merge_row_state(old: Any, new: Any, active_mask: Tensor) -> Any:
    """Keep ``old`` rows where ``active_mask`` is false and ``new`` elsewhere."""

    if isinstance(old, Tensor):
        if not isinstance(new, Tensor) or old.shape != new.shape:
            return new
        view = (active_mask.shape[0],) + (1,) * (old.ndim - 1)
        return torch.where(active_mask.view(view), new, old)
    if isinstance(old, tuple) and len(old) == 3 and old[0] == "dataclass":
        _, value_type, old_fields = old
        if not isinstance(new, value_type):
            return new
        merged = {
            name: _merge_row_state(value, getattr(new, name), active_mask)
            for name, value in old_fields.items()
        }
        return replace(new, **merged)
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return new
        merged_dict = dict(new)
        for name, value in old.items():
            merged_dict[name] = _merge_row_state(value, new.get(name), active_mask)
        return merged_dict
    return new


class BatchedPCLowLevelEnv:
    """ALOS-external pure-controller environment for C0-C4."""

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        num_envs: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        validate_finite: bool = True,
        projection_search_radius: int = 8,
        randomizer_seed: int | None = None,
        sensor_seed: int | None = None,
        path_dim: int = 0,
        path_blocks: Iterable[str] | None = None,
        eval_goal_radius_m: float | None = None,
        eval_goal_plane_miss_guard_radius_m: float | None = None,
        eval_goal_plane_miss_dwell_steps: int = 0,
        curriculum_sampler: CurriculumStageSampler | None = None,
        curriculum_manifest_path: Path | str | None = None,
        curriculum_direct_commands_enabled: bool = True,
    ) -> None:
        if config.policy_mode != "pure_controller" or config.pc_stage is None:
            raise ValueError("BatchedPCLowLevelEnv requires an explicit PC config")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if projection_search_radius < 1:
            raise ValueError("projection_search_radius must be positive")
        if config.sensors is None:
            raise ValueError("BatchedPCLowLevelEnv requires config.sensors")
        if path_dim not in (0, *PC_PATH_DIMS):
            raise ValueError(f"path_dim must be 0 or one of {PC_PATH_DIMS}")
        if path_dim == 0:
            if path_blocks is not None:
                raise ValueError("path_blocks must be omitted when path_dim=0")
            self.path_blocks = ()
        else:
            blocks = (
                (PC_PATH_ENDPOINT_BLOCKS if path_dim == 12 else PC_PATH_DEFAULT_BLOCKS)
                if path_blocks is None
                else validate_path_blocks(path_blocks)
            )
            mask = path_block_column_mask(blocks, path_dim=path_dim)
            if int(mask.numel()) != int(path_dim):
                raise ValueError("path_blocks do not match path_dim")
            self.path_blocks = blocks
        if path_dim == 12 and not config.observation.path_endpoint_context_enabled:
            raise ValueError("path_dim=12 requires observation.path_endpoint_context_enabled=true")
        if path_dim != 12 and config.observation.path_endpoint_context_enabled:
            raise ValueError("observation.path_endpoint_context_enabled requires path_dim=12")
        if eval_goal_radius_m is not None:
            if not math.isfinite(eval_goal_radius_m) or eval_goal_radius_m <= 0.0:
                raise ValueError("eval_goal_radius_m must be finite and positive")
            if eval_goal_radius_m > config.plant.goal_radius_m + 1e-12:
                raise ValueError("eval_goal_radius_m must not exceed plant.goal_radius_m")
        if eval_goal_plane_miss_guard_radius_m is not None:
            if (
                not math.isfinite(eval_goal_plane_miss_guard_radius_m)
                or eval_goal_plane_miss_guard_radius_m <= 0.0
            ):
                raise ValueError(
                    "eval_goal_plane_miss_guard_radius_m must be finite and positive"
                )
        if (
            isinstance(eval_goal_plane_miss_dwell_steps, bool)
            or not isinstance(eval_goal_plane_miss_dwell_steps, int)
            or eval_goal_plane_miss_dwell_steps < 0
        ):
            raise ValueError(
                "eval_goal_plane_miss_dwell_steps must be a non-negative integer"
            )

        self.config = config
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.validate_finite = bool(validate_finite)
        self.stage = str(config.pc_stage)
        self.pc_control_mode = str(config.pc_control_mode)
        self.actor_path_dim = int(path_dim)
        self.endpoint_enter_m = float(config.observation.path_endpoint_enter_m)
        self.endpoint_full_m = float(config.observation.path_endpoint_full_m)
        self.curriculum_direct_commands_enabled = bool(
            curriculum_direct_commands_enabled
        )
        # The direct-command curriculum is a training scaffold for C0/C1, not a
        # deployment path-following mode. Standing evaluation disables it even
        # when the environment config itself has curriculum disabled.
        self.direct_commands = (
            self.stage in {"C0", "C1"}
            and self.pc_control_mode == "alos"
            and self.curriculum_direct_commands_enabled
        )
        self.dt_s = float(config.plant.dt_s)
        self.curriculum_sampler = curriculum_sampler
        self.curriculum_specs: tuple[CurriculumEpisodeSpec, ...] | None = None
        self._curriculum_manifest = (
            None
            if curriculum_manifest_path is None
            else EpisodeManifestWriter(curriculum_manifest_path)
        )
        self._curriculum_episode_speed_mps: Tensor | None = None
        self._curriculum_current_period_s: Tensor | None = None
        self._curriculum_current_shear_scale: Tensor | None = None
        self._curriculum_stage_ids: Tensor | None = None
        self._curriculum_direct_mask: Tensor | None = None
        self._curriculum_path_points_ned_m: Tensor | None = None
        self._curriculum_path_cumulative_s_m: Tensor | None = None
        if curriculum_sampler is not None:
            if not curriculum_sampler.stages:
                raise ValueError("curriculum sampler must define at least one stage")
            max_length_m = max(
                stage.length_range_m[1] for stage in curriculum_sampler.stages.values()
            )
            self._curriculum_path_samples = max(
                2,
                int(math.ceil(max_length_m / config.reference.sample_spacing_m)) + 1,
            )
            if (
                any(stage.shear_scale > 0.0 for stage in curriculum_sampler.stages.values())
                and config.current.shear is None
            ):
                raise ValueError("curriculum stages with shear require current.shear")
        else:
            self._curriculum_path_samples = 0

        path_bank = generate_pc_path_bank(config.reference)
        if curriculum_sampler is not None:
            self._path_bank = None
            self._path_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.path = BatchedReferencePath.from_reference_path(
                path_bank[0],
                device=self.device,
                dtype=dtype,
            )
        elif len(path_bank) > 1:
            self._path_bank: BatchedReferencePathBank | None = BatchedReferencePathBank(
                path_bank,
                device=self.device,
                dtype=dtype,
            )
            self._path_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.path: BatchedReferencePath | BatchedReferencePathSet = self._path_bank.select(
                self._path_index
            )
        else:
            self._path_bank = None
            self._path_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.path = BatchedReferencePath.from_reference_path(
                path_bank[0],
                device=self.device,
                dtype=dtype,
            )
        max_start_offset_m = math.sqrt(
            2.0 * config.reference.start_jitter_xy_m**2 + config.reference.start_jitter_z_m**2
        )
        search_radius_source = self._path_bank if self._path_bank is not None else self.path
        self.reset_projection_search_radius = max(
            int(projection_search_radius),
            search_radius_source.required_search_radius(max_start_offset_m),
        )
        self.vehicle = remus100_vehicle()
        self.command_speed_range_mps = config.reference.command_speed_range_mps
        if self.command_speed_range_mps is not None:
            low, high = self.command_speed_range_mps
            if low < float(self.vehicle.min_speed_mps) - 1e-12:
                raise ValueError(
                    "reference.command_speed_range_mps lower bound is below the vehicle minimum"
                )
            if high > float(self.vehicle.max_speed_mps) + 1e-12:
                raise ValueError(
                    "reference.command_speed_range_mps upper bound exceeds the vehicle maximum"
                )
        self.domain = config.domain
        if self.domain is None:
            self._domain_command_low: Tensor | None = None
            self._domain_command_high: Tensor | None = None
            self._domain_current_low: Tensor | None = None
            self._domain_current_high: Tensor | None = None
        else:
            self._domain_command_low = torch.tensor(
                [cell.command_speed_range_mps[0] for cell in self.domain.cells],
                device=self.device,
                dtype=dtype,
            )
            self._domain_command_high = torch.tensor(
                [cell.command_speed_range_mps[1] for cell in self.domain.cells],
                device=self.device,
                dtype=dtype,
            )
            self._domain_current_low = torch.tensor(
                [cell.current_speed_range_mps[0] for cell in self.domain.cells],
                device=self.device,
                dtype=dtype,
            )
            self._domain_current_high = torch.tensor(
                [cell.current_speed_range_mps[1] for cell in self.domain.cells],
                device=self.device,
                dtype=dtype,
            )

        self.limits = ActuatorLimits(
            max_rudder_angle_rad=config.actuator.policy_max_rudder_angle_rad,
            max_stern_plane_angle_rad=config.actuator.policy_max_stern_plane_angle_rad,
            max_rudder_rate_rad_s=config.actuator.policy_max_rudder_rate_rad_s,
            max_stern_plane_rate_rad_s=config.actuator.policy_max_stern_plane_rate_rad_s,
            max_propeller_rpm=config.actuator.policy_max_propeller_rpm,
            material_tolerance_fraction=config.actuator.material_tolerance_fraction,
        )
        self.plant_limits = ActuatorLimits(
            max_rudder_angle_rad=config.actuator.plant_max_rudder_angle_rad,
            max_stern_plane_angle_rad=config.actuator.plant_max_stern_plane_angle_rad,
            max_rudder_rate_rad_s=config.actuator.plant_max_rudder_rate_rad_s,
            max_stern_plane_rate_rad_s=config.actuator.plant_max_stern_plane_rate_rad_s,
            max_propeller_rpm=config.actuator.plant_max_propeller_rpm,
            material_tolerance_fraction=config.actuator.material_tolerance_fraction,
        )
        self.dynamics = BatchedRemus100Dynamics(
            sample_time_s=self.dt_s,
            max_rudder_angle_rad=self.plant_limits.max_rudder_angle_rad,
            max_stern_plane_angle_rad=self.plant_limits.max_stern_plane_angle_rad,
            max_propeller_rpm=self.plant_limits.max_propeller_rpm,
            device=self.device,
            dtype=dtype,
            validate_finite=self.validate_finite,
        )
        self.action_pipeline = BatchedPureActionPipeline(
            self.limits,
            device=self.device,
            dtype=dtype,
            validate_finite=self.validate_finite,
        )
        self.guidance = BatchedThreeDAlosGuidance(
            config.guidance,
            config.reference,
            device=self.device,
            dtype=dtype,
            validate_finite=self.validate_finite,
        )
        self.reward_computer = BatchedPCRewardComputer(
            config,
            device=self.device,
            dtype=dtype,
        )
        self.hold_tracker = BatchedCommandHoldTracker(
            config,
            device=self.device,
            dtype=dtype,
        )
        self.termination = BatchedPCTerminationManager(
            config,
            device=self.device,
            dtype=dtype,
            goal_radius_m=eval_goal_radius_m,
            goal_plane_miss_guard_radius_m=eval_goal_plane_miss_guard_radius_m,
            goal_plane_miss_dwell_steps=eval_goal_plane_miss_dwell_steps,
        )
        self.sensors = SyntheticSensorSuite(
            config,
            num_envs=self.num_envs,
            device=self.device,
            dtype=dtype,
            seed=sensor_seed,
        )

        self._bounds = torch.as_tensor(config.plant.bounds_ned_m, device=self.device, dtype=dtype)
        self._seed_cursor = TrainingSeedCursor(
            config.evaluation.training_seed_start,
            set(
                range(
                    config.evaluation.seed_start,
                    config.evaluation.seed_start + config.evaluation.seed_count,
                )
            ),
        )
        self._auto_randomizer = TorchEpisodeRandomizer(
            config.current,
            config.reference,
            seed_start=config.evaluation.training_seed_start + self.num_envs,
            device=self.device,
            dtype=dtype,
            generator_seed=config.seed if randomizer_seed is None else randomizer_seed,
        )
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self._command_generator = torch.Generator(device=generator_device)
        self._command_generator.manual_seed(int(config.seed) + 0x51A7)

        self.eta: Tensor | None = None
        self.nu: Tensor | None = None
        self.actuator: Tensor | None = None
        self.current_ned_mps: Tensor | None = None
        self.current_base_ned_mps: Tensor | None = None
        self.current_phase_rad: Tensor | None = None
        self.current_shear_gradient: Tensor | None = (
            None
            if config.current.shear is None
            else torch.as_tensor(
                config.current.shear.velocity_gradient_per_s,
                device=self.device,
                dtype=dtype,
            )
        )
        self.current_shear_origin: Tensor | None = (
            None
            if config.current.shear is None
            else torch.as_tensor(
                config.current.shear.origin_ned_m,
                device=self.device,
                dtype=dtype,
            )
        )
        self.t_s: Tensor | None = None
        self.step_count: Tensor | None = None
        self.episode_seed: Tensor | None = None
        self.plant_scales: Tensor | None = None
        self.previous_actuator: Tensor | None = None
        self.previous_action_normalized: Tensor | None = None
        self.previous_hold_fraction: Tensor | None = None
        self.hold_fraction: Tensor | None = None
        self.sensor_output: SyntheticSensorOutput | None = None
        self.observation_value: BatchedPCObservation | None = None
        self._projection_hint: Tensor | None = None
        self._robust_current_estimate_ned_mps = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=dtype
        )
        self._guidance_previous_command = PCCommand(
            heading_rad=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
            pitch_rad=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
            speed_mps=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
        )
        self._guidance_command_initialized = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._guidance_command_rate_clipped = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.bool
        )
        self._material_saturation_ema = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._high_saturation_time_s = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._route_speed_floor_filtered_mps = torch.full(
            (self.num_envs,),
            float(config.reference.speed_mps),
            device=self.device,
            dtype=dtype,
        )
        self._route_speed_floor_boost_mps = torch.zeros(
            self.num_envs, device=self.device, dtype=dtype
        )
        self._route_speed_floor_measured_mps = torch.zeros(
            self.num_envs, device=self.device, dtype=dtype
        )
        self._route_speed_floor_initialized = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._route_speed_floor_valid = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._route_speed_floor_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._route_speed_floor_launch_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._guidance_diagnostics: dict[str, Tensor] = {}
        self._direct_target = PCCommand(
            heading_rad=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
            pitch_rad=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
            speed_mps=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
        )
        self._direct_command = PCCommand(
            heading_rad=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
            pitch_rad=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
            speed_mps=torch.zeros(self.num_envs, device=self.device, dtype=dtype),
        )
        self._direct_switch_time_s = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._direct_tau_s = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._episode_command_speed_mps = torch.full(
            (self.num_envs,),
            float(config.reference.speed_mps),
            device=self.device,
            dtype=dtype,
        )
        self._episode_initial_speed_ratio = torch.ones(
            self.num_envs, device=self.device, dtype=dtype
        )
        self._episode_initial_propeller_rpm_ratio = torch.zeros(
            self.num_envs, device=self.device, dtype=dtype
        )
        self._command_speed_filtered_below_range = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._previous_path_progress_m = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._previous_goal_distance_m = torch.zeros(
            self.num_envs, device=self.device, dtype=dtype
        )
        self._previous_ground_surge_mps = torch.zeros(
            self.num_envs, device=self.device, dtype=dtype
        )

    @property
    def observation(self) -> BatchedPCObservation:
        if self.observation_value is None:
            raise RuntimeError("reset() must be called before accessing observations")
        return self.observation_value

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def actor_state_dim(self) -> int:
        return 12

    @property
    def critic_state_dim(self) -> int:
        return 15

    def _sample_command(
        self,
        count: int,
        speeds_mps: Tensor | None = None,
    ) -> PCCommand:
        rng = self._command_generator
        tangent = (
            self.path.start_tangent_ned[0]
            if isinstance(self.path, BatchedReferencePathSet)
            else self.path.start_tangent_ned
        )
        base_heading = torch.atan2(tangent[1], tangent[0])
        horizontal = torch.linalg.norm(tangent[:2])
        base_pitch = torch.atan2(-tangent[2], horizontal)
        if self.stage == "C0":
            heading_span = math.radians(30.0)
            pitch_span = math.radians(5.0)
        else:
            heading_span = math.radians(60.0)
            pitch_span = math.radians(5.0)
        heading = (
            base_heading
            + (torch.rand(count, device=self.device, dtype=self.dtype, generator=rng) - 0.5)
            * 2.0
            * heading_span
        )
        pitch = (
            base_pitch
            + (torch.rand(count, device=self.device, dtype=self.dtype, generator=rng) - 0.5)
            * 2.0
            * pitch_span
        )
        if speeds_mps is None:
            speed_low, speed_high = 1.0, 1.5
            speed = speed_low + (speed_high - speed_low) * torch.rand(
                count, device=self.device, dtype=self.dtype, generator=rng
            )
        else:
            speed = torch.as_tensor(speeds_mps, device=self.device, dtype=self.dtype)
            if speed.shape != (count,):
                raise ValueError("speeds_mps must have shape (count,)")
        return PCCommand(heading_rad=heading, pitch_rad=pitch, speed_mps=speed)

    def _sample_switch_time(self, count: int) -> Tensor:
        return 12.0 + 12.0 * torch.rand(
            count,
            device=self.device,
            dtype=self.dtype,
            generator=self._command_generator,
        )

    def _sample_tau(self, count: int) -> Tensor:
        return 0.5 + 1.0 * torch.rand(
            count,
            device=self.device,
            dtype=self.dtype,
            generator=self._command_generator,
        )

    @staticmethod
    def _merge_rows(old: Tensor, new: Tensor, mask: Tensor) -> Tensor:
        if old.shape[0] == new.shape[0]:
            view = (old.shape[0],) + (1,) * (old.ndim - 1)
            return torch.where(mask.view(view), new, old)
        if new.shape[0] != int(mask.sum().item()):
            raise ValueError("compact replacement rows must match the selected mask")
        merged = old.clone()
        merged[mask] = new
        return merged

    def _seed_command_generator(self, seeds: list[int] | None) -> None:
        if seeds is None or not seeds:
            return
        offset = sum(int(seed) * (index + 1) for index, seed in enumerate(seeds))
        self._command_generator.manual_seed((int(self.config.seed) + 0x51A7 + offset) % (2**63 - 1))

    def _domain_uniform(self, seeds: Tensor, salt: int) -> Tensor:
        """Deterministic per-seed uniform value without extra RNG state."""

        values = seeds.to(device=self.device, dtype=torch.long)
        values = values + (0x9E3779B97F4A7C15 + int(salt) - (1 << 64))
        values = (values ^ (values >> 30)) * (0xBF58476D1CE4E5B9 - (1 << 64))
        values = (values ^ (values >> 27)) * (0x94D049BB133111EB - (1 << 64))
        values = values ^ (values >> 31)
        return (values & ((1 << 53) - 1)).to(self.dtype) / float(1 << 53)

    def _domain_values(self, seeds: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return cell index, command speed and current speed for each seed."""

        if self.domain is None or self._domain_command_low is None:
            raise RuntimeError("domain sampling is not configured")
        cell_index = torch.clamp(
            (self._domain_uniform(seeds, 0x1F3D5B79) * len(self.domain.cells)).long(),
            max=len(self.domain.cells) - 1,
        )
        command_unit = self._domain_uniform(seeds, 0x52A7E31B)
        current_unit = self._domain_uniform(seeds, 0x6C8E9CF5)
        command_speed = self._domain_command_low[cell_index] + command_unit * (
            self._domain_command_high[cell_index] - self._domain_command_low[cell_index]
        )
        current_speed = self._domain_current_low[cell_index] + current_unit * (
            self._domain_current_high[cell_index] - self._domain_current_low[cell_index]
        )
        return cell_index, command_speed, current_speed

    def _apply_domain_current(
        self, randomization: BatchedEpisodeRandomization
    ) -> BatchedEpisodeRandomization:
        """Replace the scalar current magnitude with the joint-domain sample."""

        if self.domain is None:
            return randomization
        _, _, current_speed = self._domain_values(randomization.seed)
        current = randomization.current_ned_mps
        norm = torch.linalg.norm(current[:, :2], dim=1)
        direction = torch.where(
            norm > 1e-9,
            torch.atan2(current[:, 1], current[:, 0]),
            torch.zeros_like(norm),
        )
        replacement = torch.stack(
            (
                current_speed * torch.cos(direction),
                current_speed * torch.sin(direction),
                torch.zeros_like(current_speed),
            ),
            dim=1,
        )
        return replace(randomization, current_ned_mps=replacement)

    def _reachable_speed_upper_bound(
        self,
        ground_direction_ned: Tensor,
        current_ned_mps: Tensor,
    ) -> Tensor:
        """Maximum ground speed whose required water-relative velocity is feasible."""
        if ground_direction_ned.shape != current_ned_mps.shape:
            raise ValueError("ground direction and current must have the same shape")
        direction_norm = torch.linalg.norm(ground_direction_ned, dim=1)
        direction = ground_direction_ned / torch.clamp(direction_norm[:, None], min=1e-9)
        along = (direction * current_ned_mps).sum(dim=1)
        current_sq = current_ned_mps.square().sum(dim=1)
        perpendicular_sq = torch.clamp(current_sq - along.square(), min=0.0)
        max_speed = float(self.vehicle.max_speed_mps)
        discriminant = max_speed * max_speed - perpendicular_sq
        upper = along + torch.sqrt(torch.clamp(discriminant, min=0.0))
        return torch.where(discriminant >= 0.0, upper, torch.zeros_like(upper))

    def _reset_episode_command_speeds(
        self,
        mask: Tensor,
        seeds: list[int] | None,
    ) -> None:
        if self.nu is None or self.current_ned_mps is None:
            raise RuntimeError("state must be initialized before command speed sampling")
        count = int(mask.sum().item())
        if count == 0:
            return
        if self.curriculum_sampler is not None:
            if self._curriculum_episode_speed_mps is None:
                raise RuntimeError("curriculum episode speeds have not been initialized")
            speeds = self._curriculum_episode_speed_mps[mask]
            filtered_below_range = torch.zeros(count, device=self.device, dtype=torch.bool)
        elif self.domain is not None:
            if seeds is None or len(seeds) != count:
                raise ValueError("domain sampling requires one seed per reset row")
            _, speeds, _ = self._domain_values(
                torch.as_tensor(seeds, device=self.device, dtype=torch.long)
            )
            filtered_below_range = torch.zeros(count, device=self.device, dtype=torch.bool)
        elif self.command_speed_range_mps is None:
            speeds = torch.full(
                (count,),
                float(self.config.reference.speed_mps),
                device=self.device,
                dtype=self.dtype,
            )
            filtered_below_range = torch.zeros(count, device=self.device, dtype=torch.bool)
        else:
            self._seed_command_generator(seeds)
            low, high = self.command_speed_range_mps
            sampled = low + (high - low) * torch.rand(
                count,
                device=self.device,
                dtype=self.dtype,
                generator=self._command_generator,
            )
            # Command speed must stay independent of the true current. Clamping
            # it per episode would let the actor infer the hidden current from
            # c_t. Reachability is instead enforced as a config-level worst-case
            # gate; true-current diagnostics remain info-only.
            speeds = sampled
            filtered_below_range = torch.zeros(count, device=self.device, dtype=torch.bool)
        self._episode_command_speed_mps = self._merge_rows(
            self._episode_command_speed_mps, speeds, mask
        )
        self._command_speed_filtered_below_range = self._merge_rows(
            self._command_speed_filtered_below_range,
            filtered_below_range,
            mask,
        )
        speed_ratio_range = self.config.guidance.initial_speed_ratio_range
        if speed_ratio_range is None:
            initial_speed_ratio = torch.full(
                (count,),
                float(self.config.guidance.initial_speed_ratio),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            initial_speed_ratio = speed_ratio_range[0] + (
                speed_ratio_range[1] - speed_ratio_range[0]
            ) * torch.rand(
                count,
                device=self.device,
                dtype=self.dtype,
                generator=self._command_generator,
            )
        rpm_ratio_range = self.config.guidance.initial_propeller_rpm_ratio_range
        if rpm_ratio_range is None:
            initial_rpm_ratio = torch.full(
                (count,),
                float(self.config.guidance.initial_propeller_rpm_ratio),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            initial_rpm_ratio = rpm_ratio_range[0] + (
                rpm_ratio_range[1] - rpm_ratio_range[0]
            ) * torch.rand(
                count,
                device=self.device,
                dtype=self.dtype,
                generator=self._command_generator,
            )
        self._episode_initial_speed_ratio = self._merge_rows(
            self._episode_initial_speed_ratio, initial_speed_ratio, mask
        )
        self._episode_initial_propeller_rpm_ratio = self._merge_rows(
            self._episode_initial_propeller_rpm_ratio, initial_rpm_ratio, mask
        )
        self.nu = self.nu.clone()
        initial_speed = torch.clamp(
            speeds * initial_speed_ratio,
            max=float(self.vehicle.max_speed_mps),
        )
        self.nu[mask, 0] = initial_speed
        if bool((initial_rpm_ratio > 0.0).any()):
            rpm = torch.clamp(
                initial_rpm_ratio
                * self.config.controller.speed_feedforward_rpm_per_mps
                * self.nu[mask, 0],
                max=float(self.config.actuator.plant_max_propeller_rpm),
            )
            self.actuator[mask, 2] = rpm
            self.previous_actuator[mask, 2] = rpm

    def _reset_direct_commands(self, mask: Tensor, seeds: list[int] | None) -> None:
        if self.t_s is None:
            return
        if self.curriculum_sampler is not None:
            row_mask = self._curriculum_direct_rows(mask)
            if row_mask is None or not bool(row_mask.any()):
                return
            count = int(row_mask.sum().item())
            self._seed_command_generator(seeds)
            speeds = self._episode_command_speed_mps[row_mask]
            target = self._sample_curriculum_direct_command(row_mask, speeds)
            self._direct_target = PCCommand(
                heading_rad=self._merge_rows(
                    self._direct_target.heading_rad, target.heading_rad, row_mask
                ),
                pitch_rad=self._merge_rows(
                    self._direct_target.pitch_rad, target.pitch_rad, row_mask
                ),
                speed_mps=self._merge_rows(
                    self._direct_target.speed_mps, target.speed_mps, row_mask
                ),
            )
            self._direct_command = PCCommand(
                heading_rad=self._merge_rows(
                    self._direct_command.heading_rad, target.heading_rad, row_mask
                ),
                pitch_rad=self._merge_rows(
                    self._direct_command.pitch_rad, target.pitch_rad, row_mask
                ),
                speed_mps=self._merge_rows(
                    self._direct_command.speed_mps, target.speed_mps, row_mask
                ),
            )
            self._direct_switch_time_s = self._merge_rows(
                self._direct_switch_time_s,
                self.t_s[row_mask] + self._sample_switch_time(count),
                row_mask,
            )
            stage_ids = (
                self._curriculum_stage_ids[row_mask]
                if self._curriculum_stage_ids is not None
                else torch.zeros(count, device=self.device, dtype=torch.long)
            )
            tau = torch.where(
                stage_ids == CURRICULUM_STAGE_ORDER.index("C1"),
                self._sample_tau(count),
                torch.zeros(count, device=self.device, dtype=self.dtype),
            )
            self._direct_tau_s = self._merge_rows(self._direct_tau_s, tau, row_mask)
            return
        if not self.direct_commands:
            return
        count = int(mask.sum().item())
        if count == 0:
            return
        self._seed_command_generator(seeds)
        speeds = (
            None
            if self.command_speed_range_mps is None
            and self.curriculum_sampler is None
            else self._episode_command_speed_mps[mask]
        )
        target = self._sample_command(count, speeds)
        self._direct_target = PCCommand(
            heading_rad=self._merge_rows(self._direct_target.heading_rad, target.heading_rad, mask),
            pitch_rad=self._merge_rows(self._direct_target.pitch_rad, target.pitch_rad, mask),
            speed_mps=self._merge_rows(self._direct_target.speed_mps, target.speed_mps, mask),
        )
        self._direct_command = PCCommand(
            heading_rad=self._merge_rows(
                self._direct_command.heading_rad, target.heading_rad, mask
            ),
            pitch_rad=self._merge_rows(self._direct_command.pitch_rad, target.pitch_rad, mask),
            speed_mps=self._merge_rows(self._direct_command.speed_mps, target.speed_mps, mask),
        )
        self._direct_switch_time_s = self._merge_rows(
            self._direct_switch_time_s,
            self.t_s[mask] + self._sample_switch_time(count),
            mask,
        )
        tau = (
            self._sample_tau(count)
            if self.stage == "C1"
            else torch.zeros(count, device=self.device, dtype=self.dtype)
        )
        self._direct_tau_s = self._merge_rows(self._direct_tau_s, tau, mask)

    def _advance_direct_commands(self) -> None:
        if self.t_s is None:
            return
        if self.curriculum_sampler is not None:
            row_mask = self._curriculum_direct_rows()
            if row_mask is None or not bool(row_mask.any()):
                return
            due = (self.t_s >= self._direct_switch_time_s) & row_mask
            if self._curriculum_stage_ids is not None:
                due = due & (self._curriculum_stage_ids > CURRICULUM_STAGE_ORDER.index("C0"))
            due_count = int(due.sum().item())
            if due_count:
                speeds = self._episode_command_speed_mps[due]
                target = self._sample_curriculum_direct_command(due, speeds)
                self._direct_target = PCCommand(
                    heading_rad=self._merge_rows(
                        self._direct_target.heading_rad, target.heading_rad, due
                    ),
                    pitch_rad=self._merge_rows(
                        self._direct_target.pitch_rad, target.pitch_rad, due
                    ),
                    speed_mps=self._merge_rows(
                        self._direct_target.speed_mps, target.speed_mps, due
                    ),
                )
                self._direct_switch_time_s = self._merge_rows(
                    self._direct_switch_time_s,
                    self.t_s[due] + self._sample_switch_time(due_count),
                    due,
                )
            alpha = 1.0 - torch.exp(-self.dt_s / torch.clamp(self._direct_tau_s, min=1e-9))
            heading_delta = wrap_angle(
                self._direct_target.heading_rad - self._direct_command.heading_rad
            )
            self._direct_command = PCCommand(
                heading_rad=wrap_angle(
                    self._direct_command.heading_rad + alpha * heading_delta
                ),
                pitch_rad=self._direct_command.pitch_rad
                + alpha * (self._direct_target.pitch_rad - self._direct_command.pitch_rad),
                speed_mps=self._direct_command.speed_mps
                + alpha * (self._direct_target.speed_mps - self._direct_command.speed_mps),
            )
            return
        if not self.direct_commands:
            return
        if self.stage == "C0":
            return
        due = self.t_s >= self._direct_switch_time_s
        due_count = int(due.sum().item())
        if due_count:
            speeds = (
                None
                if self.command_speed_range_mps is None
                and self.curriculum_sampler is None
                else self._episode_command_speed_mps[due]
            )
            target = self._sample_command(due_count, speeds)
            self._direct_target = PCCommand(
                heading_rad=self._merge_rows(
                    self._direct_target.heading_rad, target.heading_rad, due
                ),
                pitch_rad=self._merge_rows(self._direct_target.pitch_rad, target.pitch_rad, due),
                speed_mps=self._merge_rows(self._direct_target.speed_mps, target.speed_mps, due),
            )
            self._direct_switch_time_s = self._merge_rows(
                self._direct_switch_time_s,
                self.t_s[due] + self._sample_switch_time(due_count),
                due,
            )
        alpha = 1.0 - torch.exp(-self.dt_s / torch.clamp(self._direct_tau_s, min=1e-9))
        heading_delta = wrap_angle(
            self._direct_target.heading_rad - self._direct_command.heading_rad
        )
        self._direct_command = PCCommand(
            heading_rad=wrap_angle(self._direct_command.heading_rad + alpha * heading_delta),
            pitch_rad=self._direct_command.pitch_rad
            + alpha * (self._direct_target.pitch_rad - self._direct_command.pitch_rad),
            speed_mps=self._direct_command.speed_mps
            + alpha * (self._direct_target.speed_mps - self._direct_command.speed_mps),
        )

    def _current_estimate_for_guidance(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.sensor_output is None:
            raise RuntimeError("sensor output must be initialized")
        estimate_body = self.sensor_output.current_estimate_body
        attitude = self.sensor_output.attitude
        current_age_s = self.sensor_output.current_estimate_age_s
        attitude_quat = quaternion_from_euler(attitude[:, 0], attitude[:, 1], attitude[:, 2])
        estimate_ned = body_to_world(estimate_body, attitude_quat)
        valid = self.sensor_output.current_estimate_valid > 0.5
        valid = valid & (current_age_s <= self.config.guidance.current_estimate_max_age_s)
        valid = valid & torch.isfinite(estimate_ned).all(dim=1)
        valid = valid & (
            torch.linalg.norm(estimate_ned, dim=1)
            <= self.config.guidance.current_estimate_max_speed_mps
        )
        alpha = 1.0 - math.exp(-self.dt_s / self.config.guidance.current_estimate_tau_s)
        updated = (1.0 - alpha) * self._robust_current_estimate_ned_mps + alpha * torch.nan_to_num(
            estimate_ned
        )
        self._robust_current_estimate_ned_mps = torch.where(
            valid[:, None], updated, torch.zeros_like(updated)
        )
        return self._robust_current_estimate_ned_mps, valid, current_age_s

    def _governed_speed(self, cruise_speed_mps: Tensor) -> tuple[Tensor, Tensor]:
        guidance = self.config.guidance
        scale = torch.ones_like(cruise_speed_mps)
        if not guidance.saturation_governor_enabled:
            return cruise_speed_mps, scale
        high_speed = cruise_speed_mps > guidance.high_speed_threshold_mps
        reduction = guidance.saturation_governor_max_reduction * torch.clamp(
            (self._material_saturation_ema - guidance.saturation_governor_start)
            / guidance.saturation_governor_span,
            min=0.0,
            max=1.0,
        )
        scale = 1.0 - reduction
        emergency = self._high_saturation_time_s >= guidance.saturation_emergency_duration_s
        scale = torch.where(
            emergency,
            scale * (1.0 - guidance.saturation_emergency_extra_reduction),
            scale,
        )
        minimum_scale = max(
            guidance.saturation_governor_min_scale - guidance.saturation_emergency_extra_reduction,
            0.0,
        )
        scale = torch.clamp(scale, min=minimum_scale, max=1.0)
        scale = torch.where(high_speed, scale, torch.ones_like(scale))
        return cruise_speed_mps * scale, scale

    def _route_speed_floor(
        self,
        projection: BatchedPathProjection,
        cruise_speed_mps: Tensor,
        governed_speed_mps: Tensor,
    ) -> Tensor:
        """Raise the ground-speed target when DVL route speed is below the floor."""

        guidance = self.config.guidance
        if not guidance.route_speed_floor_enabled or self.sensor_output is None:
            self._route_speed_floor_active = torch.zeros_like(self._route_speed_floor_active)
            self._route_speed_floor_valid = torch.zeros_like(self._route_speed_floor_valid)
            self._route_speed_floor_launch_active = torch.zeros_like(
                self._route_speed_floor_launch_active
            )
            return governed_speed_mps

        sensor = self.sensor_output
        attitude_quat = quaternion_from_euler(
            sensor.attitude[:, 0], sensor.attitude[:, 1], sensor.attitude[:, 2]
        )
        dvl_ned = body_to_world(sensor.dvl_ground_body, attitude_quat)
        valid = sensor.bottom_lock_valid > 0.5
        valid = valid & (sensor.dvl_bottom_age_s <= guidance.route_speed_floor_dvl_max_age_s)
        valid = valid & torch.isfinite(dvl_ned).all(dim=1)
        valid = valid & (
            torch.linalg.norm(dvl_ned, dim=1) <= float(self.vehicle.max_speed_mps) + 1.0
        )

        route_speed = (dvl_ned * projection.tangent_ned).sum(dim=1)
        alpha = 1.0 - math.exp(-self.dt_s / guidance.route_speed_floor_filter_tau_s)
        filtered = torch.where(
            self._route_speed_floor_initialized,
            (1.0 - alpha) * self._route_speed_floor_filtered_mps + alpha * route_speed,
            route_speed,
        )
        filtered = torch.where(valid, filtered, self._route_speed_floor_filtered_mps)

        floor_speed = guidance.minimum_speed_ratio * cruise_speed_mps
        deficit = torch.clamp(floor_speed - filtered, min=0.0)
        integrated = self._route_speed_floor_boost_mps + (
            guidance.route_speed_floor_integral_gain * self.dt_s * deficit
        )
        integrated = torch.where(valid, integrated, self._route_speed_floor_boost_mps)
        integrated = torch.clamp(integrated, min=0.0, max=guidance.route_speed_floor_max_boost_mps)
        boost = torch.clamp(
            guidance.route_speed_floor_gain * deficit + integrated,
            min=0.0,
            max=guidance.route_speed_floor_max_boost_mps,
        )
        target = torch.maximum(governed_speed_mps, cruise_speed_mps + boost)
        launch_active = torch.zeros_like(valid)
        if self.t_s is not None and guidance.route_speed_floor_launch_speed_ratio > 1.0:
            launch_active = valid & (self.t_s <= guidance.route_speed_floor_launch_duration_s)
            launch_target = guidance.route_speed_floor_launch_speed_ratio * cruise_speed_mps
            target = torch.where(launch_active, torch.maximum(target, launch_target), target)
        target = torch.clamp(
            target,
            min=float(self.vehicle.min_speed_mps),
            max=float(self.vehicle.max_speed_mps),
        )
        target = torch.where(valid, target, governed_speed_mps)

        self._route_speed_floor_filtered_mps = filtered
        self._route_speed_floor_boost_mps = integrated
        self._route_speed_floor_measured_mps = route_speed
        self._route_speed_floor_initialized = self._route_speed_floor_initialized | valid
        self._route_speed_floor_valid = valid
        self._route_speed_floor_active = valid & (deficit > 0.0)
        self._route_speed_floor_launch_active = launch_active
        return target

    def _limit_guidance_command(self, target: PCCommand) -> PCCommand:
        guidance = self.config.guidance
        enabled = guidance.command_rate_limit_enabled
        if enabled is None:
            enabled = guidance.saturation_governor_enabled
        clipped = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.bool
        )
        if not enabled:
            self._guidance_previous_command = target
            self._guidance_command_initialized = torch.ones_like(self._guidance_command_initialized)
            self._guidance_command_rate_clipped = clipped
            return target
        previous = self._guidance_previous_command
        initialized = self._guidance_command_initialized
        raw_heading_delta = wrap_angle(target.heading_rad - previous.heading_rad)
        raw_pitch_delta = target.pitch_rad - previous.pitch_rad
        raw_speed_delta = target.speed_mps - previous.speed_mps
        heading_delta = torch.clamp(
            raw_heading_delta,
            -guidance.heading_rate_limit_rad_s * self.dt_s,
            guidance.heading_rate_limit_rad_s * self.dt_s,
        )
        pitch_delta = torch.clamp(
            raw_pitch_delta,
            -guidance.pitch_rate_limit_rad_s * self.dt_s,
            guidance.pitch_rate_limit_rad_s * self.dt_s,
        )
        speed_delta = torch.clamp(
            raw_speed_delta,
            -guidance.speed_rate_limit_mps2 * self.dt_s,
            guidance.speed_rate_limit_mps2 * self.dt_s,
        )
        eps = 1e-12
        clipped[:, 0] = initialized & (
            torch.abs(raw_heading_delta)
            > guidance.heading_rate_limit_rad_s * self.dt_s + eps
        )
        clipped[:, 1] = initialized & (
            torch.abs(raw_pitch_delta)
            > guidance.pitch_rate_limit_rad_s * self.dt_s + eps
        )
        clipped[:, 2] = initialized & (
            torch.abs(raw_speed_delta)
            > guidance.speed_rate_limit_mps2 * self.dt_s + eps
        )
        limited = PCCommand(
            heading_rad=wrap_angle(previous.heading_rad + heading_delta),
            pitch_rad=previous.pitch_rad + pitch_delta,
            speed_mps=previous.speed_mps + speed_delta,
        )
        command = PCCommand(
            heading_rad=torch.where(initialized, limited.heading_rad, target.heading_rad),
            pitch_rad=torch.where(initialized, limited.pitch_rad, target.pitch_rad),
            speed_mps=torch.where(initialized, limited.speed_mps, target.speed_mps),
        )
        self._guidance_previous_command = command
        self._guidance_command_initialized = torch.ones_like(initialized)
        self._guidance_command_rate_clipped = clipped
        return command

    def _reset_guidance_state(self, mask: Tensor) -> None:
        count = int(mask.sum().item())
        if count == 0:
            return
        self._robust_current_estimate_ned_mps = self._merge_rows(
            self._robust_current_estimate_ned_mps,
            torch.zeros((count, 3), device=self.device, dtype=self.dtype),
            mask,
        )
        self._guidance_previous_command = PCCommand(
            heading_rad=self._merge_rows(
                self._guidance_previous_command.heading_rad,
                torch.zeros(count, device=self.device, dtype=self.dtype),
                mask,
            ),
            pitch_rad=self._merge_rows(
                self._guidance_previous_command.pitch_rad,
                torch.zeros(count, device=self.device, dtype=self.dtype),
                mask,
            ),
            speed_mps=self._merge_rows(
                self._guidance_previous_command.speed_mps,
                torch.zeros(count, device=self.device, dtype=self.dtype),
                mask,
            ),
        )
        self._guidance_command_initialized = self._merge_rows(
            self._guidance_command_initialized,
            torch.zeros(count, device=self.device, dtype=torch.bool),
            mask,
        )
        self._guidance_command_rate_clipped = self._merge_rows(
            self._guidance_command_rate_clipped,
            torch.zeros((count, 3), device=self.device, dtype=torch.bool),
            mask,
        )
        self._material_saturation_ema = self._merge_rows(
            self._material_saturation_ema,
            torch.zeros(count, device=self.device, dtype=self.dtype),
            mask,
        )
        self._high_saturation_time_s = self._merge_rows(
            self._high_saturation_time_s,
            torch.zeros(count, device=self.device, dtype=self.dtype),
            mask,
        )
        self._route_speed_floor_filtered_mps = self._merge_rows(
            self._route_speed_floor_filtered_mps,
            self._episode_command_speed_mps[mask],
            mask,
        )
        self._route_speed_floor_boost_mps = self._merge_rows(
            self._route_speed_floor_boost_mps,
            torch.zeros(count, device=self.device, dtype=self.dtype),
            mask,
        )
        self._route_speed_floor_measured_mps = self._merge_rows(
            self._route_speed_floor_measured_mps,
            torch.zeros(count, device=self.device, dtype=self.dtype),
            mask,
        )
        self._route_speed_floor_initialized = self._merge_rows(
            self._route_speed_floor_initialized,
            torch.zeros(count, device=self.device, dtype=torch.bool),
            mask,
        )
        self._route_speed_floor_valid = self._merge_rows(
            self._route_speed_floor_valid,
            torch.zeros(count, device=self.device, dtype=torch.bool),
            mask,
        )
        self._route_speed_floor_active = self._merge_rows(
            self._route_speed_floor_active,
            torch.zeros(count, device=self.device, dtype=torch.bool),
            mask,
        )
        self._route_speed_floor_launch_active = self._merge_rows(
            self._route_speed_floor_launch_active,
            torch.zeros(count, device=self.device, dtype=torch.bool),
            mask,
        )

    def _alos_command(
        self,
    ) -> tuple[PCCommand, BatchedPathProjection, BatchedAlosSolution]:
        if self.sensor_output is None or self._projection_hint is None:
            raise RuntimeError("sensor output and projection hint must be initialized")
        position = self.sensor_output.p_nav
        projection = self.path.project(
            position,
            segment_hint=self._projection_hint,
            search_radius=self.reset_projection_search_radius,
        )
        cruise_speed = self._episode_command_speed_mps
        desired_speed, speed_scale = self._governed_speed(cruise_speed)
        desired_speed = self._route_speed_floor(projection, cruise_speed, desired_speed)
        estimated_current, current_valid, current_age = self._current_estimate_for_guidance()
        zero_current = torch.zeros_like(position)
        guidance_current = zero_current
        if self.guidance.compensate_current:
            if self.config.guidance.current_compensation_source != "estimated":
                raise RuntimeError(
                    "PC ALOS compensation must use the estimated current, not the "
                    "true-current feedforward"
                )
            guidance_current = torch.where(
                current_valid[:, None], estimated_current, zero_current
            )
        solution = self.guidance.solve(
            path=self.path,
            projection=projection,
            position_ned_m=position,
            current_ned_mps=guidance_current,
            desired_ground_speed_mps=desired_speed,
            estimated_current_ned_mps=estimated_current,
            estimated_current_valid=current_valid,
        )
        goal_distance = torch.linalg.norm(
            position - self.path.goal_ned_m_for(position.shape[0]), dim=1
        )
        minimum_speed = torch.maximum(
            torch.full_like(cruise_speed, float(self.vehicle.min_speed_mps)),
            self.config.guidance.minimum_speed_ratio * cruise_speed,
        )
        if (
            not self.config.guidance.robust_terminal_enabled
            and self.config.guidance.terminal_speed_mps > 0.0
        ):
            terminal = goal_distance <= self.config.guidance.terminal_capture_distance_m
            minimum_speed = torch.where(
                terminal,
                torch.maximum(
                    minimum_speed,
                    torch.full_like(minimum_speed, self.config.guidance.terminal_speed_mps),
                ),
                minimum_speed,
            )
        speed = torch.clamp(
            solution.command_speed_mps,
            min=minimum_speed,
            max=torch.full_like(minimum_speed, float(self.vehicle.max_speed_mps)),
        )
        target = PCCommand(
            heading_rad=solution.desired_heading_rad,
            pitch_rad=solution.desired_pitch_rad,
            speed_mps=speed,
        )
        command = self._limit_guidance_command(target)
        command = replace(
            command,
            speed_mps=torch.maximum(command.speed_mps, minimum_speed),
        )
        command_direction = torch.stack(
            (
                torch.cos(command.pitch_rad) * torch.cos(command.heading_rad),
                torch.cos(command.pitch_rad) * torch.sin(command.heading_rad),
                -torch.sin(command.pitch_rad),
            ),
            dim=1,
        )
        solution = replace(
            solution,
            desired_heading_rad=command.heading_rad,
            desired_pitch_rad=command.pitch_rad,
            desired_body_direction_ned=command_direction,
            command_speed_mps=command.speed_mps,
            reference_speed_mps=command.speed_mps,
            reference_quat_xyzw=quaternion_from_direction(command_direction),
        )
        self._projection_hint = projection.segment_index
        terminal_phase = solution.terminal_phase
        crab_angle = solution.crab_angle_rad
        self._guidance_diagnostics = {
            "current_valid": current_valid.to(self.dtype),
            "current_age_s": current_age,
            "crab_angle_rad": crab_angle
            if crab_angle is not None
            else torch.zeros_like(command.speed_mps),
            "speed_scale": speed_scale,
            "terminal_phase": terminal_phase
            if terminal_phase is not None
            else torch.zeros_like(command.speed_mps, dtype=torch.bool),
            "terminal_speed_ratio": command.speed_mps / torch.clamp(cruise_speed, min=1e-9),
            "saturation_ema": self._material_saturation_ema,
            "saturation_emergency": (
                self._high_saturation_time_s >= self.config.guidance.saturation_emergency_duration_s
            ),
            "route_speed_floor_active": self._route_speed_floor_active,
            "route_speed_floor_launch_active": self._route_speed_floor_launch_active,
            "route_speed_floor_valid": self._route_speed_floor_valid,
            "route_speed_floor_measured_mps": self._route_speed_floor_measured_mps,
            "route_speed_floor_filtered_mps": self._route_speed_floor_filtered_mps,
            "route_speed_floor_boost_mps": self._route_speed_floor_boost_mps,
            "route_speed_floor_target_mps": desired_speed,
            "route_speed_floor_ratio": self._route_speed_floor_filtered_mps
            / torch.clamp(cruise_speed, min=1e-9),
            "target_heading_rad": target.heading_rad,
            "target_pitch_rad": target.pitch_rad,
            "target_speed_mps": target.speed_mps,
            "limited_heading_rad": command.heading_rad,
            "limited_pitch_rad": command.pitch_rad,
            "limited_speed_mps": command.speed_mps,
            "heading_rate_clipped": self._guidance_command_rate_clipped[:, 0],
            "pitch_rate_clipped": self._guidance_command_rate_clipped[:, 1],
            "speed_rate_clipped": self._guidance_command_rate_clipped[:, 2],
        }
        return command, projection, solution

    def _path_command(self) -> tuple[PCCommand, BatchedPathProjection]:
        """Path-relative command without ALOS heading/pitch guidance.

        The actor never receives the heading/pitch fields in V0/V1 (its command
        dimension is 1 or 0).  The tangent is retained only for the critic's
        path-relative error diagnostics and for the path reward.
        """

        if self.sensor_output is None:
            raise RuntimeError("sensor output must be initialized")
        projection = self.path.project(
            self.sensor_output.p_nav,
            segment_hint=self._projection_hint,
            search_radius=self.reset_projection_search_radius,
        )
        tangent = projection.tangent_ned
        heading = torch.atan2(tangent[:, 1], tangent[:, 0])
        horizontal = torch.linalg.norm(tangent[:, :2], dim=1)
        pitch = torch.atan2(-tangent[:, 2], torch.clamp(horizontal, min=1e-9))
        command = PCCommand(
            heading_rad=heading,
            pitch_rad=pitch,
            speed_mps=self._episode_command_speed_mps,
        )
        self._projection_hint = projection.segment_index
        self._guidance_diagnostics = {}
        return command, projection

    def _path_reward(
        self,
        reward: PCRewardBreakdown,
        observation: BatchedPCObservation,
        *,
        previous_progress_m: Tensor,
        previous_ground_surge_mps: Tensor,
    ) -> PCRewardBreakdown:
        projection = observation.projection
        if projection is None:
            raise RuntimeError("path reward requires a path projection")
        eta = self.eta
        actuator = self.actuator
        if eta is None or actuator is None:
            raise RuntimeError("path reward requires initialized state")
        heading = wrap_angle(
            torch.atan2(projection.tangent_ned[:, 1], projection.tangent_ned[:, 0]) - eta[:, 5]
        )
        cross = projection.horizontal_error_m
        vertical = projection.vertical_error_m
        progress = projection.progress_m - previous_progress_m
        ground_surge = observation.ground_velocity_body_mps[:, 0]
        speed_smooth_penalty = (ground_surge - previous_ground_surge_mps).square() / 0.25
        energy_penalty = (actuator[:, 2] / self.config.actuator.policy_max_propeller_rpm).square()
        cfg = self.config.reward
        w_3d = cfg.w_path_3d if self.pc_control_mode == "alos" else cfg.w_3d
        path_reward = (
            -w_3d * torch.sqrt(cross.square() + vertical.square()) / 5.0
            - cfg.w_yaw * torch.abs(heading) / math.pi
            + cfg.w_prog * progress
            - cfg.w_path_speed_smooth * speed_smooth_penalty
            - cfg.w_path_energy * energy_penalty
        )
        route_terms = pc_route_health_terms(
            cross_track_m=cross,
            vertical_error_m=vertical,
            progress_m=progress,
            normalized_progress=projection.progress_m / torch.clamp(
                self.path.total_length_m_tensor, min=1e-9
            ),
            reference_speed_mps=self._episode_command_speed_mps,
            dt_s=self.config.plant.dt_s,
            reward=cfg,
        )
        path_reward = path_reward + route_terms.reward
        if self.pc_control_mode == "path_fixed_speed":
            path_reward = (
                path_reward
                - cfg.w_u * torch.abs(observation.command.speed_mps - ground_surge) / 2.5
            )
        total = reward.total + path_reward
        return replace(
            reward,
            total=total,
            ppo_scaled=total * cfg.resolved_pc_ppo_scale,
            path_cross=cross,
            path_vertical=vertical,
            path_progress=progress,
            path_heading=heading,
            speed_smooth=speed_smooth_penalty,
            energy=energy_penalty,
            route_cross_huber=route_terms.cross_huber,
            route_vertical_huber=route_terms.vertical_huber,
            route_middle_gate=route_terms.middle_gate,
            route_progress_ratio=route_terms.progress_ratio,
            route_reward=route_terms.reward,
        )

    def _ground_route_speed(self, observation: BatchedPCObservation) -> Tensor:
        """Return true ground speed projected on the reference-path tangent."""

        if self.eta is None or self._projection_hint is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        projection = self.path.project(
            self.eta[:, :3],
            segment_hint=self._projection_hint,
            search_radius=self.reset_projection_search_radius,
        )
        attitude_quat = quaternion_from_euler(self.eta[:, 3], self.eta[:, 4], self.eta[:, 5])
        ground_velocity_ned = body_to_world(observation.ground_velocity_body_mps, attitude_quat)
        return torch.clamp(
            (ground_velocity_ned * projection.tangent_ned).sum(dim=1),
            min=0.0,
        )

    def _refresh_current(self) -> None:
        variation = self.config.current.time_variation
        if self._curriculum_current_period_s is not None:
            if (
                self.current_base_ned_mps is None
                or self.current_phase_rad is None
                or self.t_s is None
            ):
                raise RuntimeError("curriculum current state is missing")
            base = self.current_base_ned_mps
            active = self._curriculum_current_period_s > 0.0
            period = torch.where(
                active,
                self._curriculum_current_period_s,
                torch.ones_like(self._curriculum_current_period_s),
            )
            omega = 2.0 * math.pi / period
            angle = omega * self.t_s + self.current_phase_rad
            base_speed = torch.hypot(base[:, 0], base[:, 1])
            base_direction = torch.atan2(base[:, 1], base[:, 0])
            speed_amplitude = torch.zeros_like(base_speed)
            direction_amplitude = torch.zeros_like(base_speed)
            if variation is not None:
                speed_amplitude = torch.where(
                    active,
                    torch.full_like(base_speed, variation.speed_amplitude_mps),
                    speed_amplitude,
                )
                direction_amplitude = torch.where(
                    active,
                    torch.full_like(base_speed, variation.direction_amplitude_rad),
                    direction_amplitude,
                )
            speed = torch.clamp(
                base_speed + speed_amplitude * torch.sin(angle),
                min=0.0,
            )
            direction = base_direction + direction_amplitude * torch.sin(
                angle + 0.5 * math.pi
            )
            velocity = torch.stack(
                (
                    speed * torch.cos(direction),
                    speed * torch.sin(direction),
                    torch.zeros_like(speed),
                ),
                dim=1,
            )
        elif variation is None:
            if self.current_base_ned_mps is None:
                raise RuntimeError("current base must be initialized")
            velocity = self.current_base_ned_mps
        else:
            if (
                self.current_base_ned_mps is None
                or self.current_phase_rad is None
                or self.t_s is None
            ):
                raise RuntimeError("time-varying current state is missing")
            base = self.current_base_ned_mps
            omega = 2.0 * math.pi / variation.period_s
            angle = omega * self.t_s + self.current_phase_rad
            base_speed = torch.hypot(base[:, 0], base[:, 1])
            base_direction = torch.atan2(base[:, 1], base[:, 0])
            speed = torch.clamp(
                base_speed + variation.speed_amplitude_mps * torch.sin(angle),
                min=0.0,
            )
            direction = base_direction + variation.direction_amplitude_rad * torch.sin(
                angle + 0.5 * math.pi
            )
            velocity = torch.stack(
                (
                    speed * torch.cos(direction),
                    speed * torch.sin(direction),
                    torch.zeros_like(speed),
                ),
                dim=1,
            )
        if self.config.current.shear is not None:
            if (
                self.eta is None
                or self.current_shear_gradient is None
                or self.current_shear_origin is None
            ):
                raise RuntimeError("sheared current state is missing")
            offset = self.eta[:, :3] - self.current_shear_origin[None, :]
            if self._curriculum_current_shear_scale is None:
                velocity = velocity + offset @ self.current_shear_gradient.T
            else:
                gradient = (
                    self.current_shear_gradient[None, :, :]
                    * self._curriculum_current_shear_scale[:, None, None]
                )
                velocity = velocity + torch.einsum(
                    "ni,nij->nj", offset, gradient.transpose(1, 2)
                )
            velocity = velocity.clone()
            velocity[:, 2] = 0.0
        self.current_ned_mps = velocity

    def _start_state(
        self,
        randomization: BatchedEpisodeRandomization,
        count: int,
        *,
        path: BatchedReferencePath | BatchedReferencePathSet | None = None,
    ) -> tuple[Tensor, Tensor]:
        path = self.path if path is None else path
        start_progress_m = self._start_progress_m_for(
            path,
            count,
            seeds=randomization.seed,
        )
        near_terminal = start_progress_m > 0.0
        if self.config.reference.start_progress_m > 0.0:
            near_point = path.point_at_s(start_progress_m)
            near_tangent = path.project(near_point).tangent_ned
            if bool(torch.any(~near_terminal).item()):
                start_point = torch.where(
                    near_terminal[:, None],
                    near_point,
                    path.start_ned_m_for(count),
                )
                start_tangent = torch.where(
                    near_terminal[:, None],
                    near_tangent,
                    path.start_tangent_ned_for(count),
                )
            else:
                start_point = near_point
                start_tangent = near_tangent
        else:
            start_point = path.start_ned_m_for(count)
            start_tangent = path.start_tangent_ned_for(count)
        start_yaw = torch.atan2(start_tangent[:, 1], start_tangent[:, 0])
        start_pitch = torch.atan2(
            -start_tangent[:, 2], torch.linalg.norm(start_tangent[:, :2], dim=1)
        )
        eta = torch.empty((count, 6), device=self.device, dtype=self.dtype)
        eta[:, :3] = start_point + randomization.start_jitter_ned_m
        eta[:, 3] = 0.0
        eta[:, 4] = start_pitch
        eta[:, 5] = wrap_angle(start_yaw + randomization.start_yaw_jitter_rad)
        nu = torch.zeros((count, 6), device=self.device, dtype=self.dtype)
        nu[:, 0] = self.config.guidance.initial_speed_ratio * float(self.config.reference.speed_mps)
        return eta, nu

    @staticmethod
    def _seed_unit_interval(seeds: Tensor) -> Tensor:
        """Deterministic per-episode uniform hash in ``[0, 1)`` for seed tensors."""

        values = seeds.to(device=seeds.device, dtype=torch.int64).bitwise_and(0xFFFFFFFF)
        values = values.bitwise_xor(values.bitwise_right_shift(16))
        values = values * 0x45D9F3B
        values = values.bitwise_xor(values.bitwise_right_shift(16))
        values = values * 0x45D9F3B
        values = values.bitwise_xor(values.bitwise_right_shift(16))
        return values.bitwise_and(0xFFFFFFFF).to(torch.float64) / float(2**32)

    def _full_path_mask_for(self, seeds: Tensor, count: int) -> Tensor:
        fraction = float(self.config.reference.start_progress_full_fraction)
        if fraction <= 0.0:
            return torch.zeros(count, device=self.device, dtype=torch.bool)
        if fraction >= 1.0:
            return torch.ones(count, device=self.device, dtype=torch.bool)
        unit = self._seed_unit_interval(seeds).to(self.device)
        return unit < fraction

    def _start_progress_m_for(
        self,
        path: BatchedReferencePath | BatchedReferencePathSet,
        count: int,
        *,
        seeds: Tensor | None = None,
    ) -> Tensor:
        start_progress = float(self.config.reference.start_progress_m)
        if start_progress <= 0.0:
            return torch.zeros(count, device=self.device, dtype=self.dtype)
        total = path.total_length_m_for(count)
        near_terminal = torch.minimum(
            torch.full_like(total, start_progress),
            torch.clamp(total - 1.0e-6, min=0.0),
        )
        fraction = float(self.config.reference.start_progress_full_fraction)
        if fraction <= 0.0:
            return near_terminal
        if seeds is None:
            raise ValueError("start_progress_full_fraction requires episode seeds")
        full_path = self._full_path_mask_for(seeds, count)
        return torch.where(full_path, torch.zeros_like(near_terminal), near_terminal)

    def _start_projection_hint(
        self,
        eta: Tensor,
        path: BatchedReferencePath | BatchedReferencePathSet,
    ) -> Tensor:
        if self.config.reference.start_progress_m <= 0.0:
            return torch.zeros(eta.shape[0], device=self.device, dtype=torch.long)
        return path.project(eta[:, :3]).segment_index

    def _build_observation(self) -> BatchedPCObservation:
        if (
            self.eta is None
            or self.nu is None
            or self.sensor_output is None
            or self.current_ned_mps is None
        ):
            raise RuntimeError("environment state must be initialized")
        if self.curriculum_sampler is not None:
            if self.pc_control_mode.startswith("path_"):
                command, projection = self._path_command()
                solution = None
            else:
                command, projection, solution = self._alos_command()
            direct_rows = self._curriculum_direct_rows()
            if direct_rows is not None and bool(direct_rows.any()):
                command = PCCommand(
                    heading_rad=torch.where(
                        direct_rows,
                        self._direct_command.heading_rad,
                        command.heading_rad,
                    ),
                    pitch_rad=torch.where(
                        direct_rows,
                        self._direct_command.pitch_rad,
                        command.pitch_rad,
                    ),
                    speed_mps=torch.where(
                        direct_rows,
                        self._direct_command.speed_mps,
                        command.speed_mps,
                    ),
                )
                if bool(direct_rows.all()):
                    solution = None
        elif self.direct_commands:
            command = self._direct_command
            projection = None
            solution = None
            if self.actor_path_dim:
                projection = self.path.project(
                    self.sensor_output.p_nav,
                    segment_hint=self._projection_hint,
                    search_radius=self.reset_projection_search_radius,
                )
                self._projection_hint = projection.segment_index
        elif self.pc_control_mode.startswith("path_"):
            command, projection = self._path_command()
            solution = None
        else:
            command, projection, solution = self._alos_command()
        ideal_actor_observations = bool(
            self.config.observation.ideal_actor_observations
        )
        actor_position_ned_m = (
            self.eta[:, :3] if ideal_actor_observations else self.sensor_output.p_nav
        )
        actor_attitude = (
            self.eta[:, 3:6]
            if ideal_actor_observations
            else self.sensor_output.attitude
        )
        actor_projection = projection
        if ideal_actor_observations and self.actor_path_dim:
            actor_projection = self.path.project(
                actor_position_ned_m,
                segment_hint=self._projection_hint,
                search_radius=self.reset_projection_search_radius,
            )
        actor_path = (
            build_pc_path_features(
                actor_projection,
                attitude=actor_attitude,
                path=self.path,
                position_ned_m=actor_position_ned_m,
                path_blocks=self.path_blocks,
                endpoint_enter_m=self.endpoint_enter_m,
                endpoint_full_m=self.endpoint_full_m,
            )
            if self.actor_path_dim
            else None
        )
        heading_error = wrap_angle(command.heading_rad - self.eta[:, 5])
        pitch_error = command.pitch_rad - self.eta[:, 4]
        if self.config.plant.velocity_contract == "nu_ground":
            ground_velocity = ground_velocity_body(self.eta, self.nu, self.current_ned_mps)
            water_velocity = water_velocity_body(self.eta, self.nu, self.current_ned_mps)
        elif self.config.plant.velocity_contract == "legacy_water_nu":
            ground_velocity = legacy_ground_velocity_body(
                self.eta, self.nu, self.current_ned_mps
            )
            water_velocity = self.nu[:, :3]
        else:
            raise ValueError(
                "unsupported plant.velocity_contract="
                f"{self.config.plant.velocity_contract!r}"
            )
        speed_feedback = self.config.observation.speed_feedback
        if speed_feedback == "ground":
            feedback_velocity = ground_velocity
        elif speed_feedback == "water_relative":
            feedback_velocity = water_velocity
        else:
            raise ValueError(f"unsupported observation.speed_feedback={speed_feedback!r}")
        privileged_surge = (
            water_velocity[:, 0]
            if self.config.plant.velocity_contract == "nu_ground"
            else ground_velocity[:, 0]
        )
        speed_error = command.speed_mps - privileged_surge
        if ideal_actor_observations:
            actor_state = torch.cat(
                [
                    actor_position_ned_m,
                    actor_attitude,
                    self.nu[:, 3:6],
                    feedback_velocity,
                ],
                dim=1,
            )
            actor_current_body = world_to_body(
                self.current_ned_mps,
                quaternion_from_euler(
                    self.eta[:, 3],
                    self.eta[:, 4],
                    self.eta[:, 5],
                ),
            )
        else:
            actor_state = self.sensor_output.actor_state(speed_feedback)
            actor_current_body = self.sensor_output.current_estimate_body
        critic_state = pc_critic_state(
            eta=self.eta,
            nu=self.nu,
            heading_error_rad=heading_error,
            pitch_error_rad=pitch_error,
            speed_error_mps=speed_error,
        )
        self.observation_value = BatchedPCObservation(
            actor_state=actor_state,
            actor_path=actor_path,
            critic_state=critic_state,
            command=command,
            actor_current_body=actor_current_body,
            critic_current_ned=self.current_ned_mps,
            ground_velocity_body_mps=ground_velocity,
            water_velocity_body_mps=water_velocity,
            sensor=self.sensor_output,
            projection=projection,
            solution=solution,
        )
        if projection is not None:
            self._previous_path_progress_m = projection.progress_m.to(self.dtype)
        return self.observation_value

    def _set_curriculum_stage_rows(
        self,
        mask: Tensor,
        specs: tuple[CurriculumEpisodeSpec, ...],
    ) -> None:
        stage_ids = torch.tensor(
            [CURRICULUM_STAGE_ORDER.index(spec.stage) for spec in specs],
            device=self.device,
            dtype=torch.long,
        )
        if self._curriculum_stage_ids is None:
            self._curriculum_stage_ids = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
        self._curriculum_stage_ids = self._merge_rows(
            self._curriculum_stage_ids, stage_ids, mask
        )
        direct = self._curriculum_stage_ids <= CURRICULUM_STAGE_ORDER.index("C1")
        if not self.curriculum_direct_commands_enabled:
            direct = torch.zeros_like(direct)
        self._curriculum_direct_mask = direct & (self.pc_control_mode == "alos")

    def _curriculum_direct_rows(self, mask: Tensor | None = None) -> Tensor | None:
        if self.curriculum_sampler is None or self._curriculum_direct_mask is None:
            return None
        if mask is None:
            return self._curriculum_direct_mask
        return self._curriculum_direct_mask & mask

    def _sample_curriculum_direct_command(
        self,
        row_mask: Tensor,
        speeds_mps: Tensor | None = None,
    ) -> PCCommand:
        count = int(row_mask.sum().item())
        if count <= 0:
            raise ValueError("direct-command sampling requires at least one row")
        indices = row_mask.nonzero(as_tuple=False).flatten()
        if isinstance(self.path, BatchedReferencePathSet):
            tangent = self.path.start_tangent_ned[indices]
        else:
            tangent = self.path.start_tangent_ned.unsqueeze(0).expand(count, -1)
        stage_ids = (
            self._curriculum_stage_ids[indices]
            if self._curriculum_stage_ids is not None
            else torch.zeros(count, device=self.device, dtype=torch.long)
        )
        base_heading = torch.atan2(tangent[:, 1], tangent[:, 0])
        horizontal = torch.linalg.norm(tangent[:, :2], dim=1)
        base_pitch = torch.atan2(-tangent[:, 2], horizontal)
        is_c0 = stage_ids == CURRICULUM_STAGE_ORDER.index("C0")
        heading_span = torch.where(
            is_c0,
            torch.full((count,), math.radians(30.0), device=self.device, dtype=self.dtype),
            torch.full((count,), math.radians(60.0), device=self.device, dtype=self.dtype),
        )
        pitch_span = torch.full(
            (count,), math.radians(5.0), device=self.device, dtype=self.dtype
        )
        rng = self._command_generator
        heading = base_heading + (
            torch.rand(count, device=self.device, dtype=self.dtype, generator=rng) - 0.5
        ) * 2.0 * heading_span
        pitch = base_pitch + (
            torch.rand(count, device=self.device, dtype=self.dtype, generator=rng) - 0.5
        ) * 2.0 * pitch_span
        if speeds_mps is None:
            speed_low, speed_high = 1.0, 1.5
            speed = speed_low + (speed_high - speed_low) * torch.rand(
                count, device=self.device, dtype=self.dtype, generator=rng
            )
        else:
            speed = torch.as_tensor(speeds_mps, device=self.device, dtype=self.dtype)
            if speed.shape != (count,):
                raise ValueError("speeds_mps must have shape (count,)")
        return PCCommand(heading_rad=heading, pitch_rad=pitch, speed_mps=speed)

    def _curriculum_specs_for_seeds(
        self, seed_values: list[int] | tuple[int, ...]
    ) -> tuple[CurriculumEpisodeSpec, ...]:
        if self.curriculum_sampler is None:
            raise RuntimeError("curriculum sampler is not configured")
        return self.curriculum_sampler.sample_batch(seed_values)

    @staticmethod
    def _curriculum_path_seed(spec: CurriculumEpisodeSpec) -> int:
        family_salt = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(f"{spec.stage}:{spec.path_family}")
        )
        return int(spec.seed) + 104729 * family_salt + 0xC0FFEE

    def _build_curriculum_path_set(
        self, specs: tuple[CurriculumEpisodeSpec, ...]
    ) -> BatchedReferencePathSet:
        if not specs:
            raise ValueError("curriculum path set requires at least one episode")
        paths = []
        for spec in specs:
            reference = replace(
                self.config.reference,
                length_m=spec.length_m,
                speed_mps=spec.speed_mps,
                depth_start_m=spec.depth_m,
                path_bank_size=1,
                path_family=spec.path_family,
                path_family_mix=None,
                start_progress_m=0.0,
                start_progress_full_fraction=0.0,
            )
            path = generate_curriculum_path(
                reference,
                spec.path_family,
                seed=self._curriculum_path_seed(spec),
            )
            paths.append(resample_reference_path(path, self._curriculum_path_samples))
        points = torch.as_tensor(
            np.stack([path.points_ned_m for path in paths]),
            device=self.device,
            dtype=self.dtype,
        )
        cumulative = torch.as_tensor(
            np.stack([path.cumulative_s_m for path in paths]),
            device=self.device,
            dtype=self.dtype,
        )
        return BatchedReferencePathSet(points, cumulative)

    def _install_curriculum_paths(
        self,
        mask: Tensor,
        specs: tuple[CurriculumEpisodeSpec, ...],
    ) -> BatchedReferencePathSet:
        selected = self._build_curriculum_path_set(specs)
        if self._curriculum_path_points_ned_m is None:
            self._curriculum_path_points_ned_m = selected.points_ned_m.clone()
            self._curriculum_path_cumulative_s_m = selected.cumulative_s_m.clone()
        else:
            self._curriculum_path_points_ned_m = self._merge_rows(
                self._curriculum_path_points_ned_m,
                selected.points_ned_m,
                mask,
            )
            self._curriculum_path_cumulative_s_m = self._merge_rows(
                self._curriculum_path_cumulative_s_m,
                selected.cumulative_s_m,
                mask,
            )
        self.path = BatchedReferencePathSet(
            self._curriculum_path_points_ned_m,
            self._curriculum_path_cumulative_s_m,
        )
        return selected

    def _apply_curriculum_current(
        self,
        randomization: BatchedEpisodeRandomization,
        specs: tuple[CurriculumEpisodeSpec, ...],
    ) -> BatchedEpisodeRandomization:
        current_speed = torch.tensor(
            [spec.current_speed_mps for spec in specs],
            device=self.device,
            dtype=self.dtype,
        )
        current = randomization.current_ned_mps
        norm = torch.linalg.norm(current[:, :2], dim=1)
        direction = torch.where(
            norm > 1e-9,
            torch.atan2(current[:, 1], current[:, 0]),
            self.config.current.direction_min_rad
            + self._domain_uniform(randomization.seed, 0x4F1BBCDC)
            * (self.config.current.direction_max_rad - self.config.current.direction_min_rad),
        )
        replacement = torch.stack(
            (
                current_speed * torch.cos(direction),
                current_speed * torch.sin(direction),
                torch.zeros_like(current_speed),
            ),
            dim=1,
        )
        return replace(randomization, current_ned_mps=replacement)

    def _set_curriculum_current_parameters(
        self,
        mask: Tensor,
        specs: tuple[CurriculumEpisodeSpec, ...],
    ) -> None:
        periods = torch.tensor(
            [0.0 if spec.current_period_s is None else spec.current_period_s for spec in specs],
            device=self.device,
            dtype=self.dtype,
        )
        shear_scales = torch.tensor(
            [spec.shear_scale for spec in specs],
            device=self.device,
            dtype=self.dtype,
        )
        if self._curriculum_current_period_s is None:
            self._curriculum_current_period_s = periods.clone()
            self._curriculum_current_shear_scale = shear_scales.clone()
        else:
            self._curriculum_current_period_s = self._merge_rows(
                self._curriculum_current_period_s, periods, mask
            )
            self._curriculum_current_shear_scale = self._merge_rows(
                self._curriculum_current_shear_scale, shear_scales, mask
            )

    def _set_curriculum_episode_speeds(
        self,
        mask: Tensor,
        specs: tuple[CurriculumEpisodeSpec, ...],
    ) -> None:
        speeds = torch.tensor(
            [spec.speed_mps for spec in specs],
            device=self.device,
            dtype=self.dtype,
        )
        if self._curriculum_episode_speed_mps is None:
            self._curriculum_episode_speed_mps = speeds.clone()
        else:
            self._curriculum_episode_speed_mps = self._merge_rows(
                self._curriculum_episode_speed_mps, speeds, mask
            )

    def _set_curriculum_noise_scales(
        self,
        mask: Tensor,
        specs: tuple[CurriculumEpisodeSpec, ...],
    ) -> None:
        scales = torch.tensor(
            [spec.noise_scale for spec in specs],
            device=self.device,
            dtype=self.dtype,
        )
        self.sensors.set_noise_scale_rows(mask, scales)

    def _record_curriculum_specs(
        self,
        mask: Tensor,
        specs: tuple[CurriculumEpisodeSpec, ...],
        *,
        reset_kind: str,
    ) -> None:
        indices = mask.nonzero(as_tuple=False).flatten().tolist()
        if len(indices) != len(specs):
            raise ValueError("curriculum specs must match the reset mask")
        if self.curriculum_specs is None or len(self.curriculum_specs) != self.num_envs:
            previous: list[CurriculumEpisodeSpec | None] = [None] * self.num_envs
        else:
            previous = list(self.curriculum_specs)
        for row, spec in zip(indices, specs, strict=True):
            previous[int(row)] = spec
            if self._curriculum_manifest is not None:
                self._curriculum_manifest.append(
                    spec,
                    row=int(row),
                    train_seed=int(spec.seed),
                    reset_kind=reset_kind,
                )
        if any(spec is None for spec in previous):
            self.curriculum_specs = None
        else:
            self.curriculum_specs = tuple(
                spec for spec in previous if spec is not None
            )

    def reset(self, seeds: list[int] | tuple[int, ...] | None = None) -> BatchedPCObservation:
        if seeds is None:
            seed_values = list(self._seed_cursor.take(self.num_envs))
        else:
            seed_values = [int(seed) for seed in seeds]
            if len(seed_values) != self.num_envs:
                raise ValueError(f"seeds must contain exactly {self.num_envs} values")
        self._auto_randomizer.reserve_until(max(seed_values) + 1)
        randomization = sample_episode_randomizations(
            seed_values,
            self.config.current,
            self.config.reference,
            device=self.device,
            dtype=self.dtype,
        )
        randomization = self._apply_domain_current(randomization)
        mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        curriculum_specs: tuple[CurriculumEpisodeSpec, ...] | None = None
        selected_path: BatchedReferencePath | BatchedReferencePathSet | None = None
        if self.curriculum_sampler is not None:
            curriculum_specs = self._curriculum_specs_for_seeds(seed_values)
            randomization = self._apply_curriculum_current(randomization, curriculum_specs)
            selected_path = self._install_curriculum_paths(mask, curriculum_specs)
            self._set_curriculum_current_parameters(mask, curriculum_specs)
            self._set_curriculum_episode_speeds(mask, curriculum_specs)
        elif self._path_bank is not None:
            self._path_index = torch.remainder(randomization.seed, self._path_bank.size).to(
                dtype=torch.long
            )
            self.path = self._path_bank.select(self._path_index)
        eta, nu = self._start_state(randomization, self.num_envs, path=selected_path)
        self.eta = eta
        self.nu = nu
        self.episode_start_progress_m = self._start_progress_m_for(
            self.path,
            self.num_envs,
            seeds=randomization.seed,
        )
        self.actuator = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self.current_base_ned_mps = randomization.current_ned_mps
        self.current_phase_rad = randomization.current_phase_rad
        self.t_s = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_seed = randomization.seed
        self.plant_scales = sample_plant_scale_factors(
            seed_values,
            self.config.plant_randomization,
            device=self.device,
            dtype=self.dtype,
        )
        self.previous_actuator = self.actuator.clone()
        self.previous_action_normalized = torch.zeros_like(self.actuator)
        self.hold_fraction = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.previous_hold_fraction = torch.zeros_like(self.hold_fraction)
        self._projection_hint = self._start_projection_hint(eta, self.path)
        if self.config.reference.start_progress_m > 0.0:
            self._previous_path_progress_m = self.episode_start_progress_m.to(self.dtype)
        self._refresh_current()
        self._reset_episode_command_speeds(mask, seed_values)
        self.sensors.reset(seed_values)
        if curriculum_specs is not None:
            self._set_curriculum_noise_scales(mask, curriculum_specs)
            self._record_curriculum_specs(mask, curriculum_specs, reset_kind="reset")
            self._set_curriculum_stage_rows(mask, curriculum_specs)
        self.sensor_output = self.sensors.prime(
            eta_true=self.eta,
            nu_true=self.nu,
            current_true_ned_mps=self.current_ned_mps,
        )
        self._reset_direct_commands(mask, seed_values)
        self.hold_tracker.reset(self.num_envs)
        self.termination.reset(self.num_envs)
        self._reset_guidance_state(mask)
        observation = self._build_observation()
        self._previous_goal_distance_m = self._goal_distance_for_reward()
        self._previous_ground_surge_mps = observation.ground_velocity_body_mps[:, 0].to(
            dtype=self.dtype
        )
        return observation

    def reset_indices(self, mask: Tensor, seeds: list[int] | None = None) -> BatchedPCObservation:
        if mask.shape != (self.num_envs,) or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor with shape (num_envs,)")
        count = int(mask.sum().item())
        seed_values = list(self._seed_cursor.take(count)) if seeds is None else seeds
        if len(seed_values) != count:
            raise ValueError("reset mask and seed count differ")
        if seed_values:
            self._auto_randomizer.reserve_until(max(int(seed) for seed in seed_values) + 1)
        randomization = sample_episode_randomizations(
            seed_values,
            self.config.current,
            self.config.reference,
            device=self.device,
            dtype=self.dtype,
        )
        self._reset_rows(mask, randomization)
        return self.observation

    def _reset_rows(self, mask: Tensor, randomization: BatchedEpisodeRandomization) -> None:
        if (
            self.eta is None
            or self.nu is None
            or self.actuator is None
            or self.current_base_ned_mps is None
            or self.current_phase_rad is None
            or self.t_s is None
            or self.step_count is None
            or self.episode_seed is None
            or self.plant_scales is None
            or self.previous_actuator is None
            or self.previous_action_normalized is None
            or self.hold_fraction is None
            or self.previous_hold_fraction is None
            or self.sensor_output is None
        ):
            raise RuntimeError("reset() must be called before reset_indices()")
        selected_count = int(mask.sum().item())
        if selected_count == 0:
            return
        if randomization.seed.shape[0] == self.num_envs:
            selected = BatchedEpisodeRandomization(
                seed=randomization.seed[mask],
                current_ned_mps=randomization.current_ned_mps[mask],
                current_phase_rad=randomization.current_phase_rad[mask],
                start_jitter_ned_m=randomization.start_jitter_ned_m[mask],
                start_yaw_jitter_rad=randomization.start_yaw_jitter_rad[mask],
            )
        elif randomization.seed.shape[0] == selected_count:
            selected = randomization
        else:
            raise ValueError(
                "episode randomization must contain either all environments or exactly the reset rows"
            )
        selected = self._apply_domain_current(selected)
        selected_specs: tuple[CurriculumEpisodeSpec, ...] | None = None
        if self.curriculum_sampler is not None:
            selected_seeds = [int(value) for value in selected.seed.tolist()]
            selected_specs = self._curriculum_specs_for_seeds(selected_seeds)
            selected = self._apply_curriculum_current(selected, selected_specs)
            selected_path = self._install_curriculum_paths(mask, selected_specs)
            self._set_curriculum_current_parameters(mask, selected_specs)
            self._set_curriculum_episode_speeds(mask, selected_specs)
        else:
            selected_path = self.path
        if self.curriculum_sampler is None and self._path_bank is not None:
            selected_indices = torch.remainder(selected.seed, self._path_bank.size).to(
                dtype=torch.long
            )
            self._path_index = self._merge_rows(self._path_index, selected_indices, mask)
            self.path = self._path_bank.select(self._path_index)
            selected_path = self._path_bank.select(selected_indices)
        eta, nu = self._start_state(selected, selected_count, path=selected_path)
        self.eta = self._merge_rows(self.eta, eta, mask)
        self.nu = self._merge_rows(self.nu, nu, mask)
        selected_start_progress = self._start_progress_m_for(
            selected_path,
            selected_count,
            seeds=selected.seed,
        )
        self.episode_start_progress_m = self._merge_rows(
            self.episode_start_progress_m,
            selected_start_progress,
            mask,
        )
        if self.config.reference.start_progress_m > 0.0:
            self._previous_path_progress_m = self._merge_rows(
                self._previous_path_progress_m,
                selected_start_progress.to(self.dtype),
                mask,
            )
        self.actuator = self._merge_rows(
            self.actuator,
            torch.zeros((selected_count, 3), device=self.device, dtype=self.dtype),
            mask,
        )
        self.current_base_ned_mps = self._merge_rows(
            self.current_base_ned_mps, selected.current_ned_mps, mask
        )
        self.current_phase_rad = self._merge_rows(
            self.current_phase_rad, selected.current_phase_rad, mask
        )
        self.t_s = self._merge_rows(self.t_s, torch.zeros_like(self.t_s), mask)
        self.step_count = self._merge_rows(self.step_count, torch.zeros_like(self.step_count), mask)
        self.episode_seed = self._merge_rows(self.episode_seed, selected.seed, mask)
        selected_plant_scales = sample_plant_scale_factors(
            [int(value) for value in selected.seed.tolist()],
            self.config.plant_randomization,
            device=self.device,
            dtype=self.dtype,
        )
        self.plant_scales = self._merge_rows(
            self.plant_scales, selected_plant_scales, mask
        )
        self.previous_actuator = self._merge_rows(
            self.previous_actuator,
            torch.zeros((selected_count, 3), device=self.device, dtype=self.dtype),
            mask,
        )
        self.previous_action_normalized = self._merge_rows(
            self.previous_action_normalized,
            torch.zeros_like(self.previous_action_normalized),
            mask,
        )
        self.hold_fraction = self._merge_rows(
            self.hold_fraction, torch.zeros_like(self.hold_fraction), mask
        )
        self.previous_hold_fraction = self._merge_rows(
            self.previous_hold_fraction, torch.zeros_like(self.previous_hold_fraction), mask
        )
        self._projection_hint = self._merge_rows(
            self._projection_hint,
            self._start_projection_hint(eta, selected_path),
            mask,
        )
        self._refresh_current()
        seeds = [int(value) for value in selected.seed.tolist()]
        self._reset_episode_command_speeds(mask, seeds)
        self.sensors.reset_rows(mask, seeds)
        if selected_specs is not None:
            self._set_curriculum_noise_scales(mask, selected_specs)
            self._record_curriculum_specs(
                mask,
                selected_specs,
                reset_kind="reset_indices",
            )
            self._set_curriculum_stage_rows(mask, selected_specs)
        self.sensor_output = self.sensors.prime_rows(
            mask,
            eta_true=self.eta,
            nu_true=self.nu,
            current_true_ned_mps=self.current_ned_mps,
            previous=self.sensor_output,
        )
        self._reset_direct_commands(mask, seeds)
        self.hold_tracker.reset_mask(mask)
        self.termination.reset_mask(mask)
        self._reset_guidance_state(mask)
        observation = self._build_observation()
        self._previous_ground_surge_mps = self._merge_rows(
            self._previous_ground_surge_mps,
            observation.ground_velocity_body_mps[mask, 0].to(dtype=self.dtype),
            mask,
        )
        self._previous_goal_distance_m = self._merge_rows(
            self._previous_goal_distance_m,
            self._goal_distance_for_reward()[mask],
            mask,
        )

    def _termination_position_ned(self) -> Tensor:
        """Position used by endpoint termination, reward and plane-miss guard.

        ``legacy`` keeps the historical coupling to ``p_nav`` whenever endpoint
        stop is enabled.  ``eta`` decouples scoring from navigation bias, while
        ``p_nav`` makes the deployment frame explicit.  The actor observation
        path is unaffected and still consumes the sensor output.
        """

        if self.sensor_output is None:
            raise RuntimeError("termination position requires initialized sensors")
        source = self.config.termination.position_source
        if source == "eta":
            return self.eta[:, :3]
        if source == "p_nav":
            return self.sensor_output.p_nav
        endpoint_stop_enabled = bool(
            self.config.termination.pc_goal_plane_miss_termination_enabled
        )
        return self.sensor_output.p_nav if endpoint_stop_enabled else self.eta[:, :3]

    def _goal_distance_for_reward(self) -> Tensor:
        if self.sensor_output is None or self.path is None:
            raise RuntimeError("goal distance requires initialized sensors and path")
        position = self._termination_position_ned()
        return torch.linalg.norm(position - self.path.goal_ned_m_for(self.num_envs), dim=1)

    def step(
        self,
        action_normalized: Tensor,
        *,
        auto_reset: bool = True,
        advance_mask: Tensor | None = None,
    ) -> BatchedPCLowLevelStep:
        if (
            self.eta is None
            or self.nu is None
            or self.actuator is None
            or self.current_ned_mps is None
            or self.t_s is None
            or self.step_count is None
            or self.previous_action_normalized is None
            or self.previous_hold_fraction is None
            or self.hold_fraction is None
            or self.sensor_output is None
        ):
            raise RuntimeError("reset() must be called before step()")
        action = torch.as_tensor(action_normalized, device=self.device, dtype=self.dtype)
        if action.shape != (self.num_envs, 3):
            raise ValueError(f"action_normalized must have shape ({self.num_envs}, 3)")
        row_snapshot: dict[str, Any] | None = None
        advance_mask_t: Tensor | None = None
        if advance_mask is not None:
            advance_mask_t = torch.as_tensor(advance_mask, device=self.device, dtype=torch.bool)
            if advance_mask_t.shape != (self.num_envs,):
                raise ValueError(f"advance_mask must have shape ({self.num_envs},)")
            if auto_reset and not bool(advance_mask_t.all()):
                raise ValueError("advance_mask with auto_reset requires all rows to be active")
            if not bool(advance_mask_t.all()):
                cloned = _clone_row_state(self.__dict__, self.num_envs)
                row_snapshot = cloned if isinstance(cloned, dict) else None
        previous_actuator = self.actuator
        previous_requested = self.previous_action_normalized
        previous_hold = self.previous_hold_fraction
        previous_ground_surge = (
            self.observation.ground_velocity_body_mps[:, 0].to(dtype=self.dtype)
            if self.observation_value is not None
            else torch.zeros_like(self.t_s)
        )
        previous_path_progress = self._previous_path_progress_m
        previous_goal_distance = self._previous_goal_distance_m
        previous_goal_ned = self.path.goal_ned_m_for(self.num_envs)
        previous_goal_vertical_error = torch.abs(self.eta[:, 2] - previous_goal_ned[:, 2])
        previous_heave_rate = self.nu[:, 2].clone()

        applied = self.action_pipeline.apply(action, previous_actuator, self.dt_s)
        saturation_alpha = 1.0 - math.exp(-self.dt_s / self.config.guidance.saturation_ema_tau_s)
        self._material_saturation_ema = (
            1.0 - saturation_alpha
        ) * self._material_saturation_ema + saturation_alpha * applied.material_saturated.to(
            self.dtype
        )
        high_saturation = (
            self._material_saturation_ema >= self.config.guidance.saturation_emergency_threshold
        )
        self._high_saturation_time_s = torch.where(
            high_saturation,
            self._high_saturation_time_s + self.dt_s,
            torch.zeros_like(self._high_saturation_time_s),
        )
        dynamics = self.dynamics.step(
            Remus100BatchState(eta=self.eta, nu=self.nu, actuator=self.actuator),
            applied.total,
            self.current_ned_mps[:, :2],
            plant_scales=self.plant_scales,
        )
        self.eta = dynamics.state.eta
        self.nu = dynamics.state.nu
        self.actuator = dynamics.state.actuator
        self.t_s = self.t_s + self.dt_s
        self.step_count = self.step_count + 1
        self._refresh_current()
        self.sensor_output = self.sensors.step(
            eta_true=self.eta,
            nu_true=self.nu,
            current_true_ned_mps=self.current_ned_mps,
        )
        self._advance_direct_commands()
        observation = self._build_observation()
        heading_error = wrap_angle(observation.command.heading_rad - self.eta[:, 5])
        pitch_error = observation.command.pitch_rad - self.eta[:, 4]
        privileged_surge = (
            observation.water_velocity_body_mps[:, 0]
            if self.config.plant.velocity_contract == "nu_ground"
            else observation.ground_velocity_body_mps[:, 0]
        )
        speed_error = observation.command.speed_mps - privileged_surge
        if self.config.reward.hold_bonus:
            self.hold_fraction = self.hold_tracker.update(
                heading_error_rad=heading_error,
                pitch_error_rad=pitch_error,
                speed_error_mps=speed_error,
            )
        else:
            self.hold_fraction = torch.zeros_like(self.t_s)

        endpoint_stop_enabled = bool(self.config.termination.pc_goal_plane_miss_termination_enabled)
        termination_position_ned_m = self._termination_position_ned()
        decision = self.termination.update(
            eta=self.eta,
            nu=self.nu,
            goal_ned_m=self.path.goal_ned_m_for(self.num_envs),
            goal_tangent_ned=self.path.goal_tangent_ned_for(self.num_envs),
            bounds_ned_m=self._bounds,
            step_count=self.step_count,
            position_ned_m=(termination_position_ned_m if endpoint_stop_enabled else None),
        )
        goal_ned_m = self.path.goal_ned_m_for(self.num_envs)
        goal_distance_m = torch.linalg.norm(termination_position_ned_m - goal_ned_m, dim=1)
        goal_vertical_error_m = torch.abs(termination_position_ned_m[:, 2] - goal_ned_m[:, 2])
        goal_plane_excess_m = torch.clamp(
            goal_distance_m - self.config.plant.goal_radius_m, min=0.0
        )
        goal_plane_miss = decision.kind == int(PCTerminalKind.GOAL_PLANE_MISS)
        route_ground_speed = self._ground_route_speed(observation)
        reward = self.reward_computer.compute(
            heading_error_rad=heading_error,
            pitch_error_rad=pitch_error,
            ground_surge_mps=privileged_surge,
            reference_speed_mps=observation.command.speed_mps,
            roll_rad=self.eta[:, 3],
            roll_rate_rad_s=self.nu[:, 3],
            actual_actuator=self.actuator,
            previous_actuator=previous_actuator,
            requested_normalized=applied.normalized,
            previous_requested_normalized=previous_requested,
            hold_fraction=self.hold_fraction,
            previous_hold_fraction=previous_hold,
            safety_terminal=decision.safety_event,
            goal_plane_miss=goal_plane_miss,
            goal_plane_excess_m=goal_plane_excess_m,
            route_ground_speed_mps=route_ground_speed,
            material_saturated=applied.material_saturated,
            previous_goal_distance_m=previous_goal_distance,
            goal_distance_m=goal_distance_m,
            goal_vertical_error_m=goal_vertical_error_m,
            previous_goal_vertical_error_m=previous_goal_vertical_error,
            heave_rate_mps=self.nu[:, 2],
            previous_heave_rate_mps=previous_heave_rate,
            terminal_kind=decision.kind,
        )
        route_health_active = any(
            weight > 1e-12
            for weight in (
                self.config.reward.w_path_cross,
                self.config.reward.w_path_vertical,
                self.config.reward.w_path_progress,
            )
        )
        if (
            self.pc_control_mode.startswith("path_")
            or self.config.reward.w_path_3d > 1e-12
            or route_health_active
        ):
            reward = self._path_reward(
                reward,
                observation,
                previous_progress_m=previous_path_progress,
                previous_ground_surge_mps=previous_ground_surge,
            )
        info = self._step_info(
            action=applied,
            decision=decision,
            observation=observation,
            goal_distance_m=goal_distance_m,
            route_ground_speed_mps=route_ground_speed,
        )
        self.previous_actuator = self.actuator
        self.previous_action_normalized = applied.normalized
        self.previous_hold_fraction = self.hold_fraction
        self._previous_ground_surge_mps = observation.ground_velocity_body_mps[:, 0].to(
            dtype=self.dtype
        )
        self._previous_goal_distance_m = goal_distance_m.to(dtype=self.dtype)

        terminated = decision.terminated
        truncated = decision.truncated
        done = terminated | truncated
        if auto_reset:
            info["final_observation"] = observation
            randomization = self._auto_randomizer.sample(self.num_envs)
            self._reset_rows(done, randomization)
        if row_snapshot is not None and advance_mask_t is not None:
            for name, old_value in row_snapshot.items():
                setattr(
                    self,
                    name,
                    _merge_row_state(old_value, getattr(self, name), advance_mask_t),
                )
        return BatchedPCLowLevelStep(
            observation=self.observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_info(
        self,
        *,
        action: BatchedPureActionApplication,
        decision: PCTerminationDecision,
        observation: BatchedPCObservation,
        goal_distance_m: Tensor,
        route_ground_speed_mps: Tensor | None = None,
    ) -> dict[str, Any]:
        assert self.eta is not None and self.nu is not None and self.actuator is not None
        assert (
            self.t_s is not None and self.step_count is not None and self.episode_seed is not None
        )
        assert self.sensor_output is not None and self.current_ned_mps is not None
        command_direction = torch.stack(
            (
                torch.cos(observation.command.pitch_rad)
                * torch.cos(observation.command.heading_rad),
                torch.cos(observation.command.pitch_rad)
                * torch.sin(observation.command.heading_rad),
                -torch.sin(observation.command.pitch_rad),
            ),
            dim=1,
        )
        command_water_relative = (
            observation.command.speed_mps[:, None] * command_direction - self.current_ned_mps
        )
        command_water_speed = torch.linalg.norm(command_water_relative, dim=1)
        command_reachability_feasible = command_water_speed <= (
            float(self.vehicle.max_speed_mps) + 1e-9
        )
        guidance_current_valid = self._guidance_diagnostics.get(
            "current_valid", torch.zeros_like(self.t_s)
        )
        guidance_current_age = self._guidance_diagnostics.get(
            "current_age_s", self.sensor_output.current_estimate_age_s
        )
        guidance_crab_angle = self._guidance_diagnostics.get(
            "crab_angle_rad", torch.zeros_like(self.t_s)
        )
        guidance_speed_scale = self._guidance_diagnostics.get(
            "speed_scale", torch.ones_like(self.t_s)
        )
        guidance_terminal_phase = self._guidance_diagnostics.get(
            "terminal_phase", torch.zeros_like(self.t_s, dtype=torch.bool)
        )
        guidance_terminal_speed_ratio = self._guidance_diagnostics.get(
            "terminal_speed_ratio", torch.ones_like(self.t_s)
        )
        guidance_saturation_ema = self._guidance_diagnostics.get(
            "saturation_ema", self._material_saturation_ema
        )
        guidance_saturation_emergency = self._guidance_diagnostics.get(
            "saturation_emergency",
            torch.zeros_like(self.t_s, dtype=torch.bool),
        )
        guidance_route_floor_active = self._guidance_diagnostics.get(
            "route_speed_floor_active",
            torch.zeros_like(self.t_s, dtype=torch.bool),
        )
        guidance_route_floor_launch_active = self._guidance_diagnostics.get(
            "route_speed_floor_launch_active",
            torch.zeros_like(self.t_s, dtype=torch.bool),
        )
        guidance_route_floor_valid = self._guidance_diagnostics.get(
            "route_speed_floor_valid",
            torch.zeros_like(self.t_s, dtype=torch.bool),
        )
        guidance_route_floor_measured = self._guidance_diagnostics.get(
            "route_speed_floor_measured_mps", torch.zeros_like(self.t_s)
        )
        guidance_route_floor_filtered = self._guidance_diagnostics.get(
            "route_speed_floor_filtered_mps", self._route_speed_floor_filtered_mps
        )
        guidance_route_floor_boost = self._guidance_diagnostics.get(
            "route_speed_floor_boost_mps", self._route_speed_floor_boost_mps
        )
        guidance_route_floor_target = self._guidance_diagnostics.get(
            "route_speed_floor_target_mps", self._episode_command_speed_mps
        )
        guidance_route_floor_ratio = self._guidance_diagnostics.get(
            "route_speed_floor_ratio", torch.ones_like(self.t_s)
        )
        guidance_target_heading = self._guidance_diagnostics.get(
            "target_heading_rad", observation.command.heading_rad
        )
        guidance_target_pitch = self._guidance_diagnostics.get(
            "target_pitch_rad", observation.command.pitch_rad
        )
        guidance_target_speed = self._guidance_diagnostics.get(
            "target_speed_mps", observation.command.speed_mps
        )
        guidance_limited_heading = self._guidance_diagnostics.get(
            "limited_heading_rad", observation.command.heading_rad
        )
        guidance_limited_pitch = self._guidance_diagnostics.get(
            "limited_pitch_rad", observation.command.pitch_rad
        )
        guidance_limited_speed = self._guidance_diagnostics.get(
            "limited_speed_mps", observation.command.speed_mps
        )
        guidance_heading_rate_clipped = self._guidance_diagnostics.get(
            "heading_rate_clipped", torch.zeros_like(self.t_s, dtype=torch.bool)
        )
        guidance_pitch_rate_clipped = self._guidance_diagnostics.get(
            "pitch_rate_clipped", torch.zeros_like(self.t_s, dtype=torch.bool)
        )
        guidance_speed_rate_clipped = self._guidance_diagnostics.get(
            "speed_rate_clipped", torch.zeros_like(self.t_s, dtype=torch.bool)
        )
        goal = self.path.goal_ned_m_for(self.num_envs)
        goal_tangent = self.path.goal_tangent_ned_for(self.num_envs)
        termination_position_ned_m = self._termination_position_ned()
        signed_goal_tangent_offset_m = torch.sum(
            (termination_position_ned_m - goal) * goal_tangent, dim=1
        )
        return {
            "seed": self.episode_seed,
            "plant_scale_factors": self.plant_scales,
            "plant_drag_scale": self.plant_scales[:, 0],
            "plant_thrust_scale": self.plant_scales[:, 1],
            "plant_fin_scale": self.plant_scales[:, 2],
            "plant_added_mass_scale": self.plant_scales[:, 3],
            "plant_actuator_time_scale": self.plant_scales[:, 4],
            "step": self.step_count,
            "time_s": self.t_s,
            "actor_state": observation.actor_state,
            "critic_state": observation.critic_state,
            "command_normalized": observation.command.normalized(),
            "command_physical": torch.stack(
                [
                    observation.command.heading_rad,
                    observation.command.pitch_rad,
                    observation.command.speed_mps,
                ],
                dim=1,
            ),
            "episode_command_speed_mps": self._episode_command_speed_mps,
            "episode_initial_speed_ratio": self._episode_initial_speed_ratio,
            "episode_initial_propeller_rpm_ratio": self._episode_initial_propeller_rpm_ratio,
            "command_speed_filtered_below_range": self._command_speed_filtered_below_range,
            "command_water_relative_speed_mps": command_water_speed,
            "command_reachability_feasible": command_reachability_feasible,
            "guidance_current_valid": guidance_current_valid,
            "guidance_current_age_s": guidance_current_age,
            "guidance_crab_angle_rad": guidance_crab_angle,
            "guidance_speed_scale": guidance_speed_scale,
            "guidance_terminal_phase": guidance_terminal_phase,
            "guidance_terminal_speed_ratio": guidance_terminal_speed_ratio,
            "guidance_saturation_ema": guidance_saturation_ema,
            "guidance_saturation_emergency": guidance_saturation_emergency,
            "guidance_route_speed_floor_active": guidance_route_floor_active,
            "guidance_route_speed_floor_launch_active": guidance_route_floor_launch_active,
            "guidance_route_speed_floor_valid": guidance_route_floor_valid,
            "guidance_route_speed_floor_measured_mps": guidance_route_floor_measured,
            "guidance_route_speed_floor_filtered_mps": guidance_route_floor_filtered,
            "guidance_route_speed_floor_boost_mps": guidance_route_floor_boost,
            "guidance_route_speed_floor_target_mps": guidance_route_floor_target,
            "guidance_route_speed_floor_ratio": guidance_route_floor_ratio,
            "guidance_target_heading_rad": guidance_target_heading,
            "guidance_target_pitch_rad": guidance_target_pitch,
            "guidance_target_speed_mps": guidance_target_speed,
            "guidance_limited_heading_rad": guidance_limited_heading,
            "guidance_limited_pitch_rad": guidance_limited_pitch,
            "guidance_limited_speed_mps": guidance_limited_speed,
            "guidance_heading_rate_clipped": guidance_heading_rate_clipped,
            "guidance_pitch_rate_clipped": guidance_pitch_rate_clipped,
            "guidance_speed_rate_clipped": guidance_speed_rate_clipped,
            "actor_current_body": observation.actor_current_body,
            "critic_current_ned": observation.critic_current_ned,
            "ground_velocity_body_mps": observation.ground_velocity_body_mps,
            "water_velocity_body_mps": observation.water_velocity_body_mps,
            "velocity_contract_code": torch.full_like(
                self.t_s,
                1.0 if self.config.plant.velocity_contract == "nu_ground" else 0.0,
            ),
            "ground_surge_mps": observation.ground_velocity_body_mps[:, 0],
            "ground_route_speed_mps": (
                torch.zeros_like(self.t_s)
                if route_ground_speed_mps is None
                else route_ground_speed_mps
            ),
            "ground_route_speed_ratio": (
                torch.zeros_like(self.t_s)
                if route_ground_speed_mps is None
                else route_ground_speed_mps / torch.clamp(self._episode_command_speed_mps, min=1e-9)
            ),
            "requested_action": action.requested_total,
            "limited_action": action.total,
            "actual_actuator": self.actuator,
            "magnitude_clipped": action.magnitude_clipped,
            "rate_clipped": action.rate_clipped,
            "constraint_violation_count": action.constraint_violation_count,
            "saturated": action.saturated,
            "material_saturated": action.material_saturated,
            "material_magnitude_clipped": action.material_magnitude_clipped,
            "material_rate_clipped": action.material_rate_clipped,
            "terminal_kind": decision.kind,
            "terminal_reason": pc_terminal_kind_name(int(decision.kind[0].item()))
            if self.num_envs == 1
            else None,
            "safety_event": decision.safety_event,
            "goal_distance_m": goal_distance_m,
            "signed_goal_tangent_offset_m": signed_goal_tangent_offset_m,
            "goal_tangent_ned": goal_tangent,
            "termination_position_ned_m": termination_position_ned_m,
            "endpoint_stop_enabled": bool(
                self.config.termination.pc_goal_plane_miss_termination_enabled
            ),
            "goal_plane_miss": decision.kind == int(PCTerminalKind.GOAL_PLANE_MISS),
            "roll_exceed_time_s": decision.roll_exceed_time_s,
            "roll_rate_exceed_time_s": decision.roll_rate_exceed_time_s,
            "p_nav": self.sensor_output.p_nav,
            "position_ned_m": self.eta[:, :3],
            "current_ned_mps": self.current_ned_mps,
            "dvl_bottom_valid": self.sensor_output.bottom_lock_valid,
            "dvl_water_valid": self.sensor_output.water_track_valid,
            "current_estimate_valid": self.sensor_output.current_estimate_valid,
            "usbl_valid": self.sensor_output.usbl_valid,
            "usbl_outlier": self.sensor_output.usbl_outlier,
            "nav_age_s": self.sensor_output.nav_age_s,
            "dvl_bottom_age_s": self.sensor_output.dvl_bottom_age_s,
            "dvl_water_age_s": self.sensor_output.dvl_water_age_s,
            "usbl_age_s": self.sensor_output.usbl_age_s,
            "current_estimate_age_s": self.sensor_output.current_estimate_age_s,
            "projection": observation.projection,
            "solution": observation.solution,
            "hold_fraction": self.hold_fraction,
        }


__all__ = [
    "BatchedPCLowLevelEnv",
    "BatchedPCLowLevelStep",
    "BatchedPCObservation",
    "PCCommand",
]
