from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from auv_core.contracts import FloatArray

SCHEMA_VERSION = "auv_odarrl_control.v1"

DEFAULT_MAX_RUDDER_ANGLE_DEG = 15.0
DEFAULT_MAX_STERN_PLANE_ANGLE_DEG = 15.0
DEFAULT_MAX_FIN_RATE_DEG_S = 150.0
DEFAULT_MAX_PROPELLER_RPM = 1525.0
DEFAULT_MATERIAL_TOLERANCE_FRACTION = 0.01
REMUS100_MAX_SPEED_MPS = 2.5
SUPPORTED_ACTUATOR_ENVELOPES = frozenset({"legacy15", "pc20"})
SUPPORTED_POLICY_MODES = frozenset({"residual", "pure_controller"})
SUPPORTED_SENSOR_SOURCES = frozenset({"simulation_truth", "synthetic", "native_isaac_oceansim"})
SUPPORTED_PLANT_BACKENDS = frozenset({"pvs_remus100", "batch_torch_remus100", "mss_remus100"})
SUPPORTED_REWARD_MODES = frozenset(
    {"legacy_path_track", "pc_command_track", "pc_command_track_v2", "pc_path_track"}
)
SUPPORTED_SENSOR_PROFILES = frozenset(
    {
        "clean",
        "mild",
        "realistic",
        "realistic_calibrated",
        "realistic_calibrated_legacy",
        "realistic_stress",
        "strong",
        "unbiased_noise_v1",
    }
)
SUPPORTED_CURRENT_EST_LPF_TIME_BASES = frozenset(
    {"control_dt_legacy", "elapsed_sample"}
)
SUPPORTED_CURRICULUM_PATH_FAMILIES = frozenset({"P0", "P1", "P2", "P3", "P4"})
SUPPORTED_PC_STAGES = frozenset({"C0", "C1", "C2", "C3", "C4", "C5", "C6"})
SUPPORTED_PC_CONTROL_MODES = frozenset({"alos", "path_fixed_speed", "path_learned_speed"})
SUPPORTED_VELOCITY_CONTRACTS = frozenset({"legacy_water_nu", "nu_ground"})


def _mapping(
    value: Any,
    path: str,
    keys: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    allowed = keys | (set() if optional is None else optional)
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    if unknown:
        raise ValueError(f"{path} has unknown keys: {unknown}")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _positive(value: Any, path: str) -> float:
    result = _number(value, path)
    if result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _non_negative(value: Any, path: str) -> float:
    result = _number(value, path)
    if result < 0.0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _optional_range(value: Any, path: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must be a [min, max] pair or null")
    lower = _number(value[0], f"{path}[0]")
    upper = _number(value[1], f"{path}[1]")
    if lower > upper:
        raise ValueError(f"{path} must satisfy min <= max")
    return lower, upper


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _optional_boolean(value: Any, path: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, path)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _vector(value: Any, size: int, path: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{path} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} must contain finite values")
    return array


def _bounds(value: Any, path: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 2):
        raise ValueError(f"{path} must have shape (3, 2)")
    if not np.all(np.isfinite(array)) or np.any(array[:, 1] <= array[:, 0]):
        raise ValueError(f"{path} must contain finite [lower, upper] bounds")
    return array


def _matrix(value: Any, rows: int, columns: int, path: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (rows, columns):
        raise ValueError(f"{path} must have shape ({rows}, {columns})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} must contain finite values")
    return array


@dataclass(frozen=True)
class PlantConfig:
    backend: str
    vehicle: str
    dt_s: float
    control_hz: float
    max_episode_steps: int
    goal_radius_m: float
    bounds_ned_m: FloatArray
    pvs_source_dir: str | None
    velocity_contract: str = "legacy_water_nu"

    def __post_init__(self) -> None:
        if self.velocity_contract not in SUPPORTED_VELOCITY_CONTRACTS:
            raise ValueError(
                "plant.velocity_contract must be one of "
                f"{sorted(SUPPORTED_VELOCITY_CONTRACTS)}"
            )


@dataclass(frozen=True)
class PlantRandomizationConfig:
    enabled: bool = False
    seed: int = 0
    drag_scale: tuple[float, float] = (1.0, 1.0)
    thrust_scale: tuple[float, float] = (1.0, 1.0)
    fin_scale: tuple[float, float] = (1.0, 1.0)
    added_mass_scale: tuple[float, float] = (1.0, 1.0)
    actuator_time_scale: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("plant_randomization.seed must be non-negative")
        for name, value in (
            ("drag_scale", self.drag_scale),
            ("thrust_scale", self.thrust_scale),
            ("fin_scale", self.fin_scale),
            ("added_mass_scale", self.added_mass_scale),
            ("actuator_time_scale", self.actuator_time_scale),
        ):
            if len(value) != 2:
                raise ValueError(f"plant_randomization.{name} must be a [min, max] pair")
            lower, upper = value
            if not all(math.isfinite(item) and item > 0.0 for item in value):
                raise ValueError(
                    f"plant_randomization.{name} values must be finite and positive"
                )
            if lower > upper:
                raise ValueError(
                    f"plant_randomization.{name} must satisfy min <= max"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "seed": self.seed,
            "drag_scale": list(self.drag_scale),
            "thrust_scale": list(self.thrust_scale),
            "fin_scale": list(self.fin_scale),
            "added_mass_scale": list(self.added_mass_scale),
            "actuator_time_scale": list(self.actuator_time_scale),
        }


@dataclass(frozen=True)
class ActuatorConfig:
    envelope: str
    policy_max_rudder_angle_rad: float
    policy_max_stern_plane_angle_rad: float
    policy_max_rudder_rate_rad_s: float
    policy_max_stern_plane_rate_rad_s: float
    policy_max_propeller_rpm: float
    plant_max_rudder_angle_rad: float
    plant_max_stern_plane_angle_rad: float
    plant_max_rudder_rate_rad_s: float
    plant_max_stern_plane_rate_rad_s: float
    plant_max_propeller_rpm: float
    material_tolerance_fraction: float

    def __post_init__(self) -> None:
        if self.envelope not in SUPPORTED_ACTUATOR_ENVELOPES:
            raise ValueError(
                f"actuator.envelope must be one of {sorted(SUPPORTED_ACTUATOR_ENVELOPES)}"
            )
        values = (
            self.policy_max_rudder_angle_rad,
            self.policy_max_stern_plane_angle_rad,
            self.policy_max_rudder_rate_rad_s,
            self.policy_max_stern_plane_rate_rad_s,
            self.policy_max_propeller_rpm,
            self.plant_max_rudder_angle_rad,
            self.plant_max_stern_plane_angle_rad,
            self.plant_max_rudder_rate_rad_s,
            self.plant_max_stern_plane_rate_rad_s,
            self.plant_max_propeller_rpm,
            self.material_tolerance_fraction,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("actuator limits and tolerances must be finite and positive")
        if self.policy_max_rudder_angle_rad > self.plant_max_rudder_angle_rad + 1e-12:
            raise ValueError("policy rudder limit exceeds the plant limit")
        if self.policy_max_stern_plane_angle_rad > self.plant_max_stern_plane_angle_rad + 1e-12:
            raise ValueError("policy stern-plane limit exceeds the plant limit")
        if self.policy_max_propeller_rpm > self.plant_max_propeller_rpm + 1e-12:
            raise ValueError("policy RPM limit exceeds the plant limit")
        if self.material_tolerance_fraction >= 1.0:
            raise ValueError("actuator.material_tolerance_fraction must be below 1")


@dataclass(frozen=True)
class ReferenceConfig:
    length_m: float
    speed_mps: float
    x_span_m: float
    lateral_amplitude_m: float
    lateral_cycles: float
    depth_start_m: float
    depth_slope_m: float
    depth_amplitude_m: float
    depth_cycles: float
    sample_spacing_m: float
    start_jitter_xy_m: float
    start_jitter_z_m: float
    start_jitter_yaw_rad: float
    path_bank_size: int = 1
    path_bank_seed: int = 0
    path_family: str | None = None
    path_family_mix: tuple[tuple[str, float], ...] | None = None
    command_speed_range_mps: tuple[float, float] | None = None
    start_progress_m: float = 0.0
    start_progress_full_fraction: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.path_bank_size, bool) or self.path_bank_size <= 0:
            raise ValueError("reference.path_bank_size must be a positive integer")
        if isinstance(self.path_bank_seed, bool) or self.path_bank_seed < 0:
            raise ValueError("reference.path_bank_seed must be a non-negative integer")
        if self.path_family is not None:
            if self.path_family not in SUPPORTED_CURRICULUM_PATH_FAMILIES:
                raise ValueError(
                    "reference.path_family must be one of "
                    f"{sorted(SUPPORTED_CURRICULUM_PATH_FAMILIES)}"
                )
        if self.path_family_mix is not None:
            if not self.path_family_mix:
                raise ValueError("reference.path_family_mix must not be empty")
            seen: set[str] = set()
            for family, weight in self.path_family_mix:
                if family not in SUPPORTED_CURRICULUM_PATH_FAMILIES:
                    raise ValueError(
                        "reference.path_family_mix contains unsupported family "
                        f"{family!r}"
                    )
                if family in seen:
                    raise ValueError(
                        f"reference.path_family_mix contains duplicate family {family!r}"
                    )
                seen.add(family)
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(
                        "reference.path_family_mix weights must be finite and positive"
                    )
        if self.path_family is not None and self.path_family_mix is not None:
            raise ValueError(
                "reference.path_family and reference.path_family_mix cannot both be set"
            )
        if not math.isfinite(self.start_progress_m) or self.start_progress_m < 0.0:
            raise ValueError("reference.start_progress_m must be finite and non-negative")
        if self.start_progress_m >= self.length_m:
            raise ValueError("reference.start_progress_m must be less than reference.length_m")
        if (
            not math.isfinite(self.start_progress_full_fraction)
            or self.start_progress_full_fraction < 0.0
            or self.start_progress_full_fraction > 1.0
        ):
            raise ValueError(
                "reference.start_progress_full_fraction must lie in [0, 1]"
            )
        if self.start_progress_full_fraction > 0.0 and self.start_progress_m <= 0.0:
            raise ValueError(
                "reference.start_progress_full_fraction requires "
                "reference.start_progress_m > 0"
            )
        if self.command_speed_range_mps is not None:
            low, high = self.command_speed_range_mps
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError("reference.command_speed_range_mps must be finite")
            if low <= 0.0 or high < low:
                raise ValueError("reference.command_speed_range_mps must be a positive [min, max]")
            if high > REMUS100_MAX_SPEED_MPS + 1e-12:
                raise ValueError(
                    "reference.command_speed_range_mps exceeds the REMUS100 speed envelope"
                )


@dataclass(frozen=True)
class CurrentVariationConfig:
    speed_amplitude_mps: float
    direction_amplitude_rad: float
    period_s: float


@dataclass(frozen=True)
class CurrentShearConfig:
    velocity_gradient_per_s: FloatArray
    origin_ned_m: FloatArray


@dataclass(frozen=True)
class CurrentConfig:
    enabled: bool
    min_speed_mps: float
    max_speed_mps: float
    direction_min_rad: float
    direction_max_rad: float
    time_variation: CurrentVariationConfig | None = None
    shear: CurrentShearConfig | None = None


@dataclass(frozen=True)
class DomainCellConfig:
    """One jointly sampled command-speed/current cell."""

    name: str
    command_speed_range_mps: tuple[float, float]
    current_speed_range_mps: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("domain.cells[].name must be non-empty")
        for field_name, values in (
            ("command_speed_range_mps", self.command_speed_range_mps),
            ("current_speed_range_mps", self.current_speed_range_mps),
        ):
            low, high = values
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(f"domain.cells[].{field_name} must be finite")
            if low < 0.0 or high < low:
                raise ValueError(f"domain.cells[].{field_name} must be a non-negative [min, max]")
        if self.command_speed_range_mps[0] <= 0.0:
            raise ValueError("domain.cells[].command_speed_range_mps must be positive")


@dataclass(frozen=True)
class DomainConfig:
    """Optional joint speed-current sampling domain for PC training."""

    cells: tuple[DomainCellConfig, ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("domain.cells must not be empty")
        names = [cell.name for cell in self.cells]
        if len(set(names)) != len(names):
            raise ValueError("domain.cells[].name must be unique")


@dataclass(frozen=True)
class GuidanceConfig:
    horizontal_lookahead_m: float
    vertical_lookahead_m: float
    min_lookahead_m: float
    max_lookahead_m: float
    horizontal_gain: float
    vertical_gain: float
    lookahead_error_gain: float
    validity_period_s: float
    compensate_current: bool
    terminal_capture_distance_m: float
    terminal_speed_mps: float
    robust_terminal_enabled: bool = False
    terminal_speed_ratio: float = 1.0
    terminal_braking_enabled: bool = False
    minimum_speed_ratio: float = 0.75
    terminal_alignment_horizon_s: float = 6.0
    terminal_alignment_min_m: float = 6.0
    terminal_alignment_max_m: float = 16.0
    terminal_path_preserving_enabled: bool = False
    terminal_reference_extension_m: float = 0.0
    current_crab_enabled: bool = False
    current_estimate_max_age_s: float = 0.6
    current_estimate_tau_s: float = 1.5
    current_estimate_max_speed_mps: float = 1.5
    current_crab_relative_speed_limit: float = 1.12
    current_crab_max_angle_rad: float = math.radians(18.0)
    speed_adaptive_lookahead_enabled: bool = False
    saturation_governor_enabled: bool = False
    command_rate_limit_enabled: bool | None = None
    saturation_ema_tau_s: float = 1.0
    saturation_governor_start: float = 0.25
    saturation_governor_span: float = 0.30
    saturation_governor_max_reduction: float = 0.12
    saturation_governor_min_scale: float = 0.88
    saturation_emergency_threshold: float = 0.70
    saturation_emergency_duration_s: float = 2.0
    saturation_emergency_extra_reduction: float = 0.05
    route_speed_floor_enabled: bool = False
    route_speed_floor_gain: float = 1.0
    route_speed_floor_integral_gain: float = 0.5
    route_speed_floor_max_boost_mps: float = 0.8
    route_speed_floor_filter_tau_s: float = 0.5
    route_speed_floor_dvl_max_age_s: float = 0.6
    route_speed_floor_launch_speed_ratio: float = 1.0
    route_speed_floor_launch_duration_s: float = 2.0
    initial_propeller_rpm_ratio: float = 0.0
    initial_speed_ratio: float = 1.0
    initial_propeller_rpm_ratio_range: tuple[float, float] | None = None
    initial_speed_ratio_range: tuple[float, float] | None = None
    high_speed_threshold_mps: float = 1.8
    heading_rate_limit_rad_s: float = math.radians(30.0)
    pitch_rate_limit_rad_s: float = math.radians(15.0)
    speed_rate_limit_mps2: float = 0.25

    def __post_init__(self) -> None:
        positive = {
            "terminal_speed_ratio": self.terminal_speed_ratio,
            "minimum_speed_ratio": self.minimum_speed_ratio,
            "terminal_alignment_horizon_s": self.terminal_alignment_horizon_s,
            "terminal_alignment_min_m": self.terminal_alignment_min_m,
            "terminal_alignment_max_m": self.terminal_alignment_max_m,
            "current_estimate_max_age_s": self.current_estimate_max_age_s,
            "current_estimate_tau_s": self.current_estimate_tau_s,
            "current_estimate_max_speed_mps": self.current_estimate_max_speed_mps,
            "current_crab_relative_speed_limit": self.current_crab_relative_speed_limit,
            "current_crab_max_angle_rad": self.current_crab_max_angle_rad,
            "saturation_ema_tau_s": self.saturation_ema_tau_s,
            "saturation_governor_span": self.saturation_governor_span,
            "saturation_emergency_duration_s": self.saturation_emergency_duration_s,
            "route_speed_floor_filter_tau_s": self.route_speed_floor_filter_tau_s,
            "route_speed_floor_dvl_max_age_s": self.route_speed_floor_dvl_max_age_s,
            "route_speed_floor_launch_speed_ratio": self.route_speed_floor_launch_speed_ratio,
            "route_speed_floor_launch_duration_s": self.route_speed_floor_launch_duration_s,
            "initial_speed_ratio": self.initial_speed_ratio,
            "high_speed_threshold_mps": self.high_speed_threshold_mps,
            "heading_rate_limit_rad_s": self.heading_rate_limit_rad_s,
            "pitch_rate_limit_rad_s": self.pitch_rate_limit_rad_s,
            "speed_rate_limit_mps2": self.speed_rate_limit_mps2,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"guidance.{name} must be finite and positive")
        if self.terminal_alignment_min_m > self.terminal_alignment_max_m:
            raise ValueError(
                "guidance.terminal_alignment_min_m must not exceed terminal_alignment_max_m"
            )
        if self.terminal_path_preserving_enabled and not self.robust_terminal_enabled:
            raise ValueError(
                "guidance.terminal_path_preserving_enabled requires "
                "robust_terminal_enabled"
            )
        if not math.isfinite(self.terminal_reference_extension_m) or (
            self.terminal_reference_extension_m < 0.0
        ):
            raise ValueError(
                "guidance.terminal_reference_extension_m must be finite and non-negative"
            )
        if (
            self.terminal_reference_extension_m > 0.0
            and not self.robust_terminal_enabled
        ):
            raise ValueError(
                "guidance.terminal_reference_extension_m requires "
                "robust_terminal_enabled"
            )
        if (
            self.terminal_reference_extension_m > 0.0
            and self.terminal_path_preserving_enabled
        ):
            raise ValueError(
                "guidance.terminal_reference_extension_m is incompatible with "
                "terminal_path_preserving_enabled"
            )
        if not 0.0 < self.terminal_speed_ratio <= 1.0:
            raise ValueError("guidance.terminal_speed_ratio must lie in (0, 1]")
        if not 0.0 < self.minimum_speed_ratio <= 1.0:
            raise ValueError("guidance.minimum_speed_ratio must lie in (0, 1]")
        if self.terminal_speed_ratio < self.minimum_speed_ratio:
            raise ValueError("guidance.terminal_speed_ratio must not be below minimum_speed_ratio")
        if not 0.0 <= self.saturation_governor_start < 1.0:
            raise ValueError("guidance.saturation_governor_start must lie in [0, 1)")
        if not 0.0 < self.saturation_governor_min_scale <= 1.0:
            raise ValueError("guidance.saturation_governor_min_scale must lie in (0, 1]")
        if not 0.0 <= self.saturation_governor_max_reduction < 1.0:
            raise ValueError("guidance.saturation_governor_max_reduction must lie in [0, 1)")
        if not 0.0 <= self.saturation_emergency_extra_reduction < 1.0:
            raise ValueError("guidance.saturation_emergency_extra_reduction must lie in [0, 1)")
        if not 0.0 <= self.saturation_emergency_threshold <= 1.0:
            raise ValueError("guidance.saturation_emergency_threshold must lie in [0, 1]")
        if not math.isfinite(self.route_speed_floor_gain) or self.route_speed_floor_gain < 0.0:
            raise ValueError("guidance.route_speed_floor_gain must be finite and non-negative")
        if (
            not math.isfinite(self.route_speed_floor_integral_gain)
            or self.route_speed_floor_integral_gain < 0.0
        ):
            raise ValueError(
                "guidance.route_speed_floor_integral_gain must be finite and non-negative"
            )
        if (
            not math.isfinite(self.route_speed_floor_max_boost_mps)
            or self.route_speed_floor_max_boost_mps < 0.0
        ):
            raise ValueError(
                "guidance.route_speed_floor_max_boost_mps must be finite and non-negative"
            )
        if (
            not math.isfinite(self.route_speed_floor_launch_speed_ratio)
            or self.route_speed_floor_launch_speed_ratio < 1.0
        ):
            raise ValueError(
                "guidance.route_speed_floor_launch_speed_ratio must be finite and at least 1"
            )
        if (
            not math.isfinite(self.initial_propeller_rpm_ratio)
            or self.initial_propeller_rpm_ratio < 0.0
        ):
            raise ValueError("guidance.initial_propeller_rpm_ratio must be finite and non-negative")
        for name, value in (
            ("initial_speed_ratio_range", self.initial_speed_ratio_range),
            (
                "initial_propeller_rpm_ratio_range",
                self.initial_propeller_rpm_ratio_range,
            ),
        ):
            if value is None:
                continue
            if len(value) != 2 or not all(math.isfinite(item) for item in value):
                raise ValueError(f"guidance.{name} must be a finite [min, max] pair")
            if value[0] < 0.0 or value[0] > value[1]:
                raise ValueError(f"guidance.{name} must satisfy 0 <= min <= max")
        if self.initial_speed_ratio_range is not None and (
            self.initial_speed_ratio_range[0] <= 0.0
        ):
            raise ValueError("guidance.initial_speed_ratio_range must be positive")


@dataclass(frozen=True)
class ControllerConfig:
    use_current_field: bool
    nominal_step_s: float
    heading_gain: float
    heading_rate_gain: float
    pitch_gain: float
    pitch_rate_gain: float
    speed_feedforward_rpm_per_mps: float
    speed_gain_rpm_per_mps: float
    speed_integral_gain_rpm_per_mps_s: float
    max_speed_integral_rpm: float


@dataclass(frozen=True)
class ResidualConfig:
    scale_rudder: float
    scale_stern: float
    scale_rpm: float
    max_action: float


@dataclass(frozen=True)
class CommandResidualConfig:
    """Optional ALOS command-space residual wrapper (Goal27 C1).

    Disabled by default. When enabled, the environment action is interpreted as
    a normalized residual on ``[psi_d, theta_d, u_d]`` and a deployable PD/PI
    controller turns the corrected command into ``[rudder, stern, rpm]``.
    """

    enabled: bool = False
    scale_heading_rad: float = 0.35
    scale_pitch_rad: float = 0.17
    scale_speed_mps: float = 0.60
    rate_heading_rad_s: float = 0.25
    rate_pitch_rad_s: float = 0.15
    rate_speed_mps2: float = 0.80
    controller_state: str = "actor"
    terminal_gate_enabled: bool = False
    terminal_gate_enter_m: float = 20.0
    terminal_gate_full_m: float = 5.0

    def __post_init__(self) -> None:
        for name, value in (
            ("terminal_gate_enter_m", self.terminal_gate_enter_m),
            ("terminal_gate_full_m", self.terminal_gate_full_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"command_residual.{name} must be finite and positive")
        if self.terminal_gate_full_m >= self.terminal_gate_enter_m:
            raise ValueError(
                "command_residual.terminal_gate_full_m must be less than "
                "terminal_gate_enter_m"
            )


@dataclass(frozen=True)
class ControllerGainResidualConfig:
    """Optional bounded controller-gain residual wrapper (Goal162).

    Disabled by default. When enabled, the environment action is interpreted as
    a normalized residual on the H3 heading/pitch PD gains. The mapping is
    ``base_gain + delta * tanh(action)`` and is therefore hard-bounded by
    construction.
    """

    enabled: bool = False
    heading_gain_delta: float = 1.0
    pitch_gain_delta: float = 1.5
    controller_state: str = "actor"
    terminal_gate_enabled: bool = True
    terminal_gate_enter_m: float = 10.0
    terminal_gate_full_m: float = 2.0

    def __post_init__(self) -> None:
        for name, value in (
            ("heading_gain_delta", self.heading_gain_delta),
            ("pitch_gain_delta", self.pitch_gain_delta),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"controller_gain_residual.{name} must be finite and positive"
                )
        if self.controller_state not in {"actor", "privileged"}:
            raise ValueError(
                "controller_gain_residual.controller_state must be 'actor' or 'privileged'"
            )
        for name, value in (
            ("terminal_gate_enter_m", self.terminal_gate_enter_m),
            ("terminal_gate_full_m", self.terminal_gate_full_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"controller_gain_residual.{name} must be finite and positive"
                )
        if self.terminal_gate_full_m >= self.terminal_gate_enter_m:
            raise ValueError(
                "controller_gain_residual.terminal_gate_full_m must be less than "
                "terminal_gate_enter_m"
            )


@dataclass(frozen=True)
class ObservationConfig:
    dim: int
    expose_current: bool
    speed_feedback: str
    current_contract: str
    path_endpoint_context_enabled: bool = False
    path_endpoint_enter_m: float = 20.0
    path_endpoint_full_m: float = 5.0
    ideal_actor_observations: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.path_endpoint_enter_m) or self.path_endpoint_enter_m <= 0.0:
            raise ValueError("observation.path_endpoint_enter_m must be finite and positive")
        if not math.isfinite(self.path_endpoint_full_m) or self.path_endpoint_full_m <= 0.0:
            raise ValueError("observation.path_endpoint_full_m must be finite and positive")
        if self.path_endpoint_full_m >= self.path_endpoint_enter_m:
            raise ValueError(
                "observation.path_endpoint_full_m must be less than path_endpoint_enter_m"
            )


@dataclass(frozen=True)
class RewardConfig:
    w_prog: float
    w_3d: float
    w_yaw: float
    w_u: float
    w_action: float
    w_sat: float
    terminal_progress_weight: float
    goal_bonus: float
    timeout_penalty: float
    out_of_bounds_penalty: float
    mode: str = "legacy_path_track"
    hold_bonus: bool = False
    w_cmd_yaw: float = 1.5
    w_cmd_pitch: float = 0.5
    w_cmd_speed: float = 1.0
    w_hold: float = 5.0
    w_action_pc: float = 0.5
    w_sat_pc: float = 0.1
    w_route_speed_pc: float = 0.0
    route_speed_floor_ratio: float = 0.75
    w_material_sat_pc: float = 0.0
    w_path_speed_smooth: float = 0.0
    w_path_energy: float = 0.0
    w_path_3d: float = 0.0
    w_path_cross: float = 0.0
    w_path_vertical: float = 0.0
    w_path_progress: float = 0.0
    route_middle_gate: float = 0.0
    path_cross_huber_delta_m: float = 0.5
    path_vertical_huber_delta_m: float = 0.5
    w_approach_vertical: float = 0.0
    w_approach_pitch: float = 0.0
    w_vertical_potential: float = 0.0
    w_heave_potential: float = 0.0
    vertical_potential_huber_delta_m: float = 0.25
    heave_potential_huber_delta_mps: float = 0.05
    potential_gamma: float = 0.995
    approach_enter_m: float = 5.0
    approach_full_m: float = 2.0
    w_phi: float = 0.05
    w_p: float = 0.02
    phi_dead_rad: float = math.radians(5.0)
    phi_soft_rad: float = math.radians(10.0)
    phi_strong_rad: float = math.radians(30.0)
    p_dead_rad_s: float = math.radians(10.0)
    p_soft_rad_s: float = math.radians(30.0)
    hold_yaw_tolerance_rad: float = math.radians(5.0)
    hold_pitch_tolerance_rad: float = math.radians(5.0)
    hold_speed_tolerance_mps: float = 0.1
    hold_time_s: float = 2.0
    pc_ppo_scale: float | None = None

    def __post_init__(self) -> None:
        if self.pc_ppo_scale is not None and (
            not math.isfinite(self.pc_ppo_scale) or self.pc_ppo_scale <= 0.0
        ):
            raise ValueError("reward.pc_ppo_scale must be finite and positive")
        for name, value in (
            ("w_route_speed_pc", self.w_route_speed_pc),
            ("w_material_sat_pc", self.w_material_sat_pc),
            ("w_path_speed_smooth", self.w_path_speed_smooth),
            ("w_path_energy", self.w_path_energy),
            ("w_path_3d", self.w_path_3d),
            ("w_path_cross", self.w_path_cross),
            ("w_path_vertical", self.w_path_vertical),
            ("w_path_progress", self.w_path_progress),
            ("route_middle_gate", self.route_middle_gate),
            ("w_approach_vertical", self.w_approach_vertical),
            ("w_approach_pitch", self.w_approach_pitch),
            ("w_vertical_potential", self.w_vertical_potential),
            ("w_heave_potential", self.w_heave_potential),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"reward.{name} must be finite and non-negative")
        for name, value in (
            ("path_cross_huber_delta_m", self.path_cross_huber_delta_m),
            ("path_vertical_huber_delta_m", self.path_vertical_huber_delta_m),
            ("vertical_potential_huber_delta_m", self.vertical_potential_huber_delta_m),
            ("heave_potential_huber_delta_mps", self.heave_potential_huber_delta_mps),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"reward.{name} must be finite and positive")
        if not math.isfinite(self.potential_gamma) or not (
            0.0 < self.potential_gamma <= 1.0
        ):
            raise ValueError("reward.potential_gamma must lie in (0, 1]")
        if not math.isfinite(self.approach_enter_m) or self.approach_enter_m <= 0.0:
            raise ValueError("reward.approach_enter_m must be finite and positive")
        if not math.isfinite(self.approach_full_m) or self.approach_full_m <= 0.0:
            raise ValueError("reward.approach_full_m must be finite and positive")
        if self.approach_full_m >= self.approach_enter_m:
            raise ValueError("reward.approach_full_m must be less than reward.approach_enter_m")
        if not math.isfinite(self.route_speed_floor_ratio) or not (
            0.0 < self.route_speed_floor_ratio <= 1.0
        ):
            raise ValueError("reward.route_speed_floor_ratio must lie in (0, 1]")

    @property
    def resolved_pc_ppo_scale(self) -> float:
        """PC PPO scaling; unset preserves the historical unscaled PC reward."""

        return 1.0 if self.pc_ppo_scale is None else self.pc_ppo_scale


@dataclass(frozen=True)
class TerminationConfig:
    phi_term_rad: float = math.radians(45.0)
    t_phi_term_s: float = 1.0
    p_term_rad_s: float = math.radians(90.0)
    t_p_term_s: float = 0.5
    safety_terminal_penalty: float = -50.0
    min_depth_m: float | None = None
    max_depth_m: float | None = None
    pc_goal_plane_miss_termination_enabled: bool = False
    goal_plane_miss_penalty: float = 0.0
    goal_plane_miss_distance_weight: float = 0.0
    position_source: str = "legacy"

    def __post_init__(self) -> None:
        if self.position_source not in {"legacy", "eta", "p_nav"}:
            raise ValueError(
                "termination.position_source must be one of ['eta', 'legacy', 'p_nav']"
            )
        positive = (
            self.phi_term_rad,
            self.t_phi_term_s,
            self.p_term_rad_s,
            self.t_p_term_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("termination thresholds and durations must be positive")
        if not math.isfinite(self.safety_terminal_penalty) or self.safety_terminal_penalty > 0.0:
            raise ValueError("safety_terminal_penalty must be finite and non-positive")
        if not math.isfinite(self.goal_plane_miss_penalty) or self.goal_plane_miss_penalty > 0.0:
            raise ValueError("goal_plane_miss_penalty must be finite and non-positive")
        if (
            not math.isfinite(self.goal_plane_miss_distance_weight)
            or self.goal_plane_miss_distance_weight < 0.0
        ):
            raise ValueError("goal_plane_miss_distance_weight must be finite and non-negative")
        if (self.min_depth_m is None) != (self.max_depth_m is None):
            raise ValueError("min_depth_m and max_depth_m must be set together")
        if self.min_depth_m is not None and self.max_depth_m is not None:
            if not math.isfinite(self.min_depth_m) or not math.isfinite(self.max_depth_m):
                raise ValueError("depth limits must be finite")
            if self.max_depth_m <= self.min_depth_m:
                raise ValueError("max_depth_m must exceed min_depth_m")


@dataclass(frozen=True)
class SensorConfig:
    profile: str
    seed: int | None = None
    current_est_lpf_time_basis: str = "control_dt_legacy"
    noise_scale: float = 1.0
    noise_scale_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.profile not in SUPPORTED_SENSOR_PROFILES:
            raise ValueError(f"sensors.profile must be one of {sorted(SUPPORTED_SENSOR_PROFILES)}")
        if self.seed is not None and self.seed < 0:
            raise ValueError("sensors.seed must be non-negative")
        if self.current_est_lpf_time_basis not in SUPPORTED_CURRENT_EST_LPF_TIME_BASES:
            raise ValueError(
                "sensors.current_est_lpf_time_basis must be one of "
                f"{sorted(SUPPORTED_CURRENT_EST_LPF_TIME_BASES)}"
            )
        if not math.isfinite(self.noise_scale) or self.noise_scale < 0.0:
            raise ValueError("sensors.noise_scale must be finite and non-negative")
        if self.noise_scale_range is not None:
            lower, upper = self.noise_scale_range
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower < 0.0
                or upper < lower
            ):
                raise ValueError(
                    "sensors.noise_scale_range must be a finite non-negative [min, max] pair"
                )
            if self.noise_scale != 1.0:
                raise ValueError(
                    "sensors.noise_scale_range and a non-default sensors.noise_scale "
                    "cannot be set together"
                )


@dataclass(frozen=True)
class NormalizationConfig:
    position_scale_m: float = 100.0
    attitude_scale_rad: float = math.pi
    angular_rate_scale_rad_s: float = 1.0
    surge_speed_scale_mps: float = 2.5
    sway_heave_speed_scale_mps: float = 1.0
    current_scale_mps: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.position_scale_m,
            self.attitude_scale_rad,
            self.angular_rate_scale_rad_s,
            self.surge_speed_scale_mps,
            self.sway_heave_speed_scale_mps,
            self.current_scale_mps,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("normalization scales must be finite and positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "position_scale_m": self.position_scale_m,
            "attitude_scale_rad": self.attitude_scale_rad,
            "angular_rate_scale_rad_s": self.angular_rate_scale_rad_s,
            "surge_speed_scale_mps": self.surge_speed_scale_mps,
            "sway_heave_speed_scale_mps": self.sway_heave_speed_scale_mps,
            "current_scale_mps": self.current_scale_mps,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    seed_start: int
    seed_count: int
    training_seed_start: int


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = False
    promotion_enabled: bool = True
    initial_stage: str = "C0"
    stage_mix: tuple[tuple[str, float], ...] = ()
    manifest_filename: str = "curriculum_episodes.jsonl"
    current_speed_range_override_mps: tuple[float, float] | None = None
    shear_scale_override: float | None = None

    def __post_init__(self) -> None:
        if self.initial_stage not in SUPPORTED_PC_STAGES:
            raise ValueError("curriculum.initial_stage must be one of C0..C6")
        if not self.enabled:
            if self.stage_mix:
                raise ValueError("curriculum.stage_mix requires curriculum.enabled=true")
            if (
                self.current_speed_range_override_mps is not None
                or self.shear_scale_override is not None
            ):
                raise ValueError(
                    "curriculum current/shear overrides require curriculum.enabled=true"
                )
            return
        if not self.stage_mix:
            raise ValueError("curriculum.stage_mix must not be empty when enabled")
        seen: set[str] = set()
        for stage, weight in self.stage_mix:
            if stage not in SUPPORTED_PC_STAGES:
                raise ValueError(f"curriculum.stage_mix contains unknown stage {stage!r}")
            if stage in seen:
                raise ValueError(f"curriculum.stage_mix contains duplicate stage {stage!r}")
            seen.add(stage)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("curriculum.stage_mix weights must be finite and positive")
        if not self.manifest_filename or Path(self.manifest_filename).is_absolute():
            raise ValueError("curriculum.manifest_filename must be a non-empty relative path")
        if self.current_speed_range_override_mps is not None:
            speed_min, speed_max = (
                float(value) for value in self.current_speed_range_override_mps
            )
            if not (math.isfinite(speed_min) and math.isfinite(speed_max)):
                raise ValueError("curriculum current speed override must be finite")
            if speed_min < 0.0 or speed_max < speed_min:
                raise ValueError(
                    "curriculum current speed override requires 0 <= MIN <= MAX"
                )
        if self.shear_scale_override is not None:
            if (
                not math.isfinite(self.shear_scale_override)
                or self.shear_scale_override < 0.0
            ):
                raise ValueError(
                    "curriculum shear scale override must be finite and non-negative"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "promotion_enabled": bool(self.promotion_enabled),
            "initial_stage": self.initial_stage,
            "stage_mix": [[stage, float(weight)] for stage, weight in self.stage_mix],
            "manifest_filename": self.manifest_filename,
            "current_speed_range_override_mps": (
                None
                if self.current_speed_range_override_mps is None
                else [float(value) for value in self.current_speed_range_override_mps]
            ),
            "shear_scale_override": (
                None
                if self.shear_scale_override is None
                else float(self.shear_scale_override)
            ),
        }


@dataclass(frozen=True)
class LowLevelConfig:
    schema_version: str
    name: str
    arm: str
    policy_mode: str
    sensor_source: str
    pc_stage: str | None
    pc_control_mode: str
    seed: int
    output: str
    plant: PlantConfig
    actuator: ActuatorConfig
    reference: ReferenceConfig
    current: CurrentConfig
    domain: DomainConfig | None
    guidance: GuidanceConfig
    controller: ControllerConfig
    residual: ResidualConfig
    command_residual: CommandResidualConfig
    controller_gain_residual: ControllerGainResidualConfig
    observation: ObservationConfig
    reward: RewardConfig
    termination: TerminationConfig
    sensors: SensorConfig | None
    normalization: NormalizationConfig
    evaluation: EvaluationConfig
    curriculum: CurriculumConfig
    plant_randomization: PlantRandomizationConfig
    config_sha256: str


def _parse_plant(value: Any) -> PlantConfig:
    data = _mapping(
        value,
        "plant",
        {
            "backend",
            "vehicle",
            "dt_s",
            "control_hz",
            "max_episode_steps",
            "goal_radius_m",
            "bounds_m",
            "pvs_source_dir",
        },
        optional={"velocity_contract"},
    )
    backend = _string(data["backend"], "plant.backend")
    if backend not in SUPPORTED_PLANT_BACKENDS or data["vehicle"] != "REMUS100":
        raise ValueError(
            "plant.backend must be one of "
            f"{sorted(SUPPORTED_PLANT_BACKENDS)} with vehicle='REMUS100'"
        )
    dt_s = _positive(data["dt_s"], "plant.dt_s")
    control_hz = _positive(data["control_hz"], "plant.control_hz")
    if abs(control_hz - 1.0 / dt_s) > 1e-9:
        raise ValueError("plant.control_hz must equal 1 / plant.dt_s")
    source = data["pvs_source_dir"]
    if source is not None:
        source = _string(source, "plant.pvs_source_dir")
    velocity_contract = _string(
        data.get("velocity_contract", "legacy_water_nu"),
        "plant.velocity_contract",
    )
    return PlantConfig(
        backend=backend,
        vehicle="REMUS100",
        dt_s=dt_s,
        control_hz=control_hz,
        max_episode_steps=_integer(data["max_episode_steps"], "plant.max_episode_steps"),
        goal_radius_m=_positive(data["goal_radius_m"], "plant.goal_radius_m"),
        bounds_ned_m=_bounds(data["bounds_m"], "plant.bounds_m"),
        pvs_source_dir=source,
        velocity_contract=velocity_contract,
    )


def _parse_plant_randomization(value: Any) -> PlantRandomizationConfig:
    if value is None:
        return PlantRandomizationConfig()
    data = _mapping(
        value,
        "plant_randomization",
        {"enabled", "seed", "drag_scale", "thrust_scale", "fin_scale"},
        optional={"added_mass_scale", "actuator_time_scale"},
    )

    def scale_range(key: str) -> tuple[float, float]:
        parsed = _optional_range(data.get(key), f"plant_randomization.{key}")
        if parsed is None:
            return (1.0, 1.0)
        if parsed[0] <= 0.0:
            raise ValueError(
                f"plant_randomization.{key} values must be strictly positive"
            )
        return parsed

    return PlantRandomizationConfig(
        enabled=_boolean(data["enabled"], "plant_randomization.enabled"),
        seed=_integer(data["seed"], "plant_randomization.seed"),
        drag_scale=scale_range("drag_scale"),
        thrust_scale=scale_range("thrust_scale"),
        fin_scale=scale_range("fin_scale"),
        added_mass_scale=scale_range("added_mass_scale"),
        actuator_time_scale=scale_range("actuator_time_scale"),
    )


def _default_actuator_config() -> ActuatorConfig:
    """Return the frozen legacy 15-degree envelope used by older configs."""

    max_rudder = math.radians(DEFAULT_MAX_RUDDER_ANGLE_DEG)
    max_stern = math.radians(DEFAULT_MAX_STERN_PLANE_ANGLE_DEG)
    max_rate = math.radians(DEFAULT_MAX_FIN_RATE_DEG_S)
    return ActuatorConfig(
        envelope="legacy15",
        policy_max_rudder_angle_rad=max_rudder,
        policy_max_stern_plane_angle_rad=max_stern,
        policy_max_rudder_rate_rad_s=max_rate,
        policy_max_stern_plane_rate_rad_s=max_rate,
        policy_max_propeller_rpm=DEFAULT_MAX_PROPELLER_RPM,
        plant_max_rudder_angle_rad=max_rudder,
        plant_max_stern_plane_angle_rad=max_stern,
        plant_max_rudder_rate_rad_s=max_rate,
        plant_max_stern_plane_rate_rad_s=max_rate,
        plant_max_propeller_rpm=DEFAULT_MAX_PROPELLER_RPM,
        material_tolerance_fraction=DEFAULT_MATERIAL_TOLERANCE_FRACTION,
    )


def _parse_actuator(value: Any) -> ActuatorConfig:
    if value is None:
        return _default_actuator_config()
    data = _mapping(
        value,
        "actuator",
        {
            "envelope",
            "max_rudder_angle_deg",
            "max_stern_plane_angle_deg",
            "max_fin_rate_deg_s",
            "max_propeller_rpm",
            "plant_max_rudder_angle_deg",
            "plant_max_stern_plane_angle_deg",
            "plant_max_fin_rate_deg_s",
            "plant_max_propeller_rpm",
            "material_tolerance_fraction",
        },
    )
    return ActuatorConfig(
        envelope=_string(data["envelope"], "actuator.envelope"),
        policy_max_rudder_angle_rad=math.radians(
            _positive(data["max_rudder_angle_deg"], "actuator.max_rudder_angle_deg")
        ),
        policy_max_stern_plane_angle_rad=math.radians(
            _positive(
                data["max_stern_plane_angle_deg"],
                "actuator.max_stern_plane_angle_deg",
            )
        ),
        policy_max_rudder_rate_rad_s=math.radians(
            _positive(data["max_fin_rate_deg_s"], "actuator.max_fin_rate_deg_s")
        ),
        policy_max_stern_plane_rate_rad_s=math.radians(
            _positive(data["max_fin_rate_deg_s"], "actuator.max_fin_rate_deg_s")
        ),
        policy_max_propeller_rpm=_positive(data["max_propeller_rpm"], "actuator.max_propeller_rpm"),
        plant_max_rudder_angle_rad=math.radians(
            _positive(
                data["plant_max_rudder_angle_deg"],
                "actuator.plant_max_rudder_angle_deg",
            )
        ),
        plant_max_stern_plane_angle_rad=math.radians(
            _positive(
                data["plant_max_stern_plane_angle_deg"],
                "actuator.plant_max_stern_plane_angle_deg",
            )
        ),
        plant_max_rudder_rate_rad_s=math.radians(
            _positive(
                data["plant_max_fin_rate_deg_s"],
                "actuator.plant_max_fin_rate_deg_s",
            )
        ),
        plant_max_stern_plane_rate_rad_s=math.radians(
            _positive(
                data["plant_max_fin_rate_deg_s"],
                "actuator.plant_max_fin_rate_deg_s",
            )
        ),
        plant_max_propeller_rpm=_positive(
            data["plant_max_propeller_rpm"], "actuator.plant_max_propeller_rpm"
        ),
        material_tolerance_fraction=_positive(
            data["material_tolerance_fraction"],
            "actuator.material_tolerance_fraction",
        ),
    )


def _parse_path_family_mix(value: Any) -> tuple[tuple[str, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError("reference.path_family_mix must be a non-empty object")
    mix: list[tuple[str, float]] = []
    for family, weight_value in value.items():
        if family not in SUPPORTED_CURRICULUM_PATH_FAMILIES:
            raise ValueError(
                "reference.path_family_mix contains unsupported family "
                f"{family!r}"
            )
        weight = _positive(
            weight_value,
            f"reference.path_family_mix.{family}",
        )
        mix.append((str(family), float(weight)))
    return tuple(mix)


def _parse_reference(value: Any) -> ReferenceConfig:
    data = _mapping(
        value,
        "reference",
        {
            "generator",
            "length_m",
            "speed_mps",
            "x_span_m",
            "lateral_amplitude_m",
            "lateral_cycles",
            "depth_start_m",
            "depth_slope_m",
            "depth_amplitude_m",
            "depth_cycles",
            "sample_spacing_m",
            "start_jitter_xy_m",
            "start_jitter_z_m",
            "start_jitter_yaw_rad",
        },
        optional={
            "path_bank_size",
            "path_bank_seed",
            "path_family",
            "path_family_mix",
            "command_speed_range_mps",
            "start_progress_m",
            "start_progress_full_fraction",
        },
    )
    if data["generator"] != "sinusoidal_3d_arclength":
        raise ValueError("unsupported reference.generator")
    path_bank_size = _integer(data.get("path_bank_size", 1), "reference.path_bank_size")
    path_bank_seed = _integer(data.get("path_bank_seed", 0), "reference.path_bank_seed")
    if path_bank_size <= 0:
        raise ValueError("reference.path_bank_size must be positive")
    if path_bank_seed < 0:
        raise ValueError("reference.path_bank_seed must be non-negative")
    path_family_value = data.get("path_family")
    path_family = (
        None
        if path_family_value is None
        else _string(path_family_value, "reference.path_family")
    )
    path_family_mix = _parse_path_family_mix(data.get("path_family_mix"))
    command_speed_range_value = data.get("command_speed_range_mps")
    command_speed_range: tuple[float, float] | None = None
    if command_speed_range_value is not None:
        command_speed = _vector(
            command_speed_range_value,
            2,
            "reference.command_speed_range_mps",
        )
        if command_speed[0] <= 0.0 or command_speed[1] < command_speed[0]:
            raise ValueError("reference.command_speed_range_mps must be a positive [min, max]")
        command_speed_range = (float(command_speed[0]), float(command_speed[1]))
    start_progress_m = _non_negative(
        data.get("start_progress_m", 0.0), "reference.start_progress_m"
    )
    start_progress_full_fraction = _number(
        data.get("start_progress_full_fraction", 0.0),
        "reference.start_progress_full_fraction",
    )
    return ReferenceConfig(
        length_m=_positive(data["length_m"], "reference.length_m"),
        speed_mps=_positive(data["speed_mps"], "reference.speed_mps"),
        x_span_m=_positive(data["x_span_m"], "reference.x_span_m"),
        lateral_amplitude_m=_non_negative(
            data["lateral_amplitude_m"], "reference.lateral_amplitude_m"
        ),
        lateral_cycles=_positive(data["lateral_cycles"], "reference.lateral_cycles"),
        depth_start_m=_non_negative(data["depth_start_m"], "reference.depth_start_m"),
        depth_slope_m=_number(data["depth_slope_m"], "reference.depth_slope_m"),
        depth_amplitude_m=_non_negative(data["depth_amplitude_m"], "reference.depth_amplitude_m"),
        depth_cycles=_positive(data["depth_cycles"], "reference.depth_cycles"),
        sample_spacing_m=_positive(data["sample_spacing_m"], "reference.sample_spacing_m"),
        start_jitter_xy_m=_non_negative(data["start_jitter_xy_m"], "reference.start_jitter_xy_m"),
        start_jitter_z_m=_non_negative(data["start_jitter_z_m"], "reference.start_jitter_z_m"),
        start_jitter_yaw_rad=_non_negative(
            data["start_jitter_yaw_rad"], "reference.start_jitter_yaw_rad"
        ),
        path_bank_size=path_bank_size,
        path_bank_seed=path_bank_seed,
        path_family=path_family,
        path_family_mix=path_family_mix,
        command_speed_range_mps=command_speed_range,
        start_progress_m=start_progress_m,
        start_progress_full_fraction=start_progress_full_fraction,
    )


def _parse_current(value: Any) -> CurrentConfig:
    data = _mapping(
        value,
        "current",
        {"enabled", "type", "speed_mps", "direction_rad", "constant_within_episode"},
        optional={"time_variation", "shear"},
    )
    current_type = data["type"]
    if current_type not in {"uniform_horizontal", "sheared_horizontal"}:
        raise ValueError("current.type must be 'uniform_horizontal' or 'sheared_horizontal'")
    constant_within_episode = _boolean(
        data["constant_within_episode"], "current.constant_within_episode"
    )
    variation_data = data.get("time_variation")
    time_variation = None if variation_data is None else _parse_current_variation(variation_data)
    shear_data = data.get("shear")
    shear = None if shear_data is None else _parse_current_shear(shear_data)
    if current_type == "sheared_horizontal" and shear is None:
        raise ValueError("sheared_horizontal current requires current.shear")
    if current_type == "uniform_horizontal" and shear is not None:
        raise ValueError("uniform_horizontal current must not define current.shear")
    if time_variation is None and not constant_within_episode:
        raise ValueError("current.constant_within_episode=false requires current.time_variation")
    if time_variation is not None and constant_within_episode:
        raise ValueError("current.time_variation requires current.constant_within_episode=false")
    speed = _vector(data["speed_mps"], 2, "current.speed_mps")
    direction = _vector(data["direction_rad"], 2, "current.direction_rad")
    if speed[0] < 0.0 or speed[1] < speed[0]:
        raise ValueError("current.speed_mps must be a non-negative [min, max] range")
    if direction[1] < direction[0]:
        raise ValueError("current.direction_rad must be [min, max]")
    if time_variation is not None and time_variation.speed_amplitude_mps > speed[0]:
        raise ValueError(
            "current.time_variation.speed_amplitude_mps must not exceed the "
            "minimum base speed, otherwise the modulated speed can become negative"
        )
    enabled = _boolean(data["enabled"], "current.enabled")
    if not enabled and time_variation is not None:
        raise ValueError("current.time_variation requires current.enabled=true")
    if not enabled and shear is not None:
        raise ValueError("current.shear requires current.enabled=true")
    return CurrentConfig(
        enabled=enabled,
        min_speed_mps=float(speed[0]),
        max_speed_mps=float(speed[1]),
        direction_min_rad=float(direction[0]),
        direction_max_rad=float(direction[1]),
        time_variation=time_variation,
        shear=shear,
    )


def _parse_domain(value: Any) -> DomainConfig | None:
    if value is None:
        return None
    data = _mapping(value, "domain", {"cells"})
    cells_data = data["cells"]
    if not isinstance(cells_data, list) or not cells_data:
        raise ValueError("domain.cells must be a non-empty list")
    cells: list[DomainCellConfig] = []
    for index, cell_value in enumerate(cells_data):
        cell = _mapping(
            cell_value,
            f"domain.cells[{index}]",
            {"name", "command_speed_range_mps", "current_speed_range_mps"},
        )
        command = _vector(
            cell["command_speed_range_mps"],
            2,
            f"domain.cells[{index}].command_speed_range_mps",
        )
        current = _vector(
            cell["current_speed_range_mps"],
            2,
            f"domain.cells[{index}].current_speed_range_mps",
        )
        cells.append(
            DomainCellConfig(
                name=_string(cell["name"], f"domain.cells[{index}].name"),
                command_speed_range_mps=(float(command[0]), float(command[1])),
                current_speed_range_mps=(float(current[0]), float(current[1])),
            )
        )
    return DomainConfig(cells=tuple(cells))


def _parse_current_variation(value: Any) -> CurrentVariationConfig:
    data = _mapping(
        value,
        "current.time_variation",
        {"speed_amplitude_mps", "direction_amplitude_rad", "period_s"},
    )
    return CurrentVariationConfig(
        speed_amplitude_mps=_non_negative(
            data["speed_amplitude_mps"],
            "current.time_variation.speed_amplitude_mps",
        ),
        direction_amplitude_rad=_non_negative(
            data["direction_amplitude_rad"],
            "current.time_variation.direction_amplitude_rad",
        ),
        period_s=_positive(data["period_s"], "current.time_variation.period_s"),
    )


def _parse_current_shear(value: Any) -> CurrentShearConfig:
    data = _mapping(
        value,
        "current.shear",
        {"velocity_gradient_per_s", "origin_ned_m"},
    )
    gradient = _matrix(
        data["velocity_gradient_per_s"],
        3,
        3,
        "current.shear.velocity_gradient_per_s",
    )
    if np.any(np.abs(gradient[2]) > 1e-12):
        raise ValueError(
            "current.shear.velocity_gradient_per_s bottom row must be zero; "
            "vertical current is not supported"
        )
    return CurrentShearConfig(
        velocity_gradient_per_s=gradient,
        origin_ned_m=_vector(data["origin_ned_m"], 3, "current.shear.origin_ned_m"),
    )


def _parse_guidance(value: Any) -> GuidanceConfig:
    data = _mapping(
        value,
        "guidance",
        {
            "horizontal_lookahead_m",
            "vertical_lookahead_m",
            "min_lookahead_m",
            "max_lookahead_m",
            "horizontal_gain",
            "vertical_gain",
            "lookahead_error_gain",
            "validity_period_s",
            "compensate_current",
            "terminal_capture_distance_m",
        },
        optional={
            "terminal_speed_mps",
            "robust_terminal_enabled",
            "terminal_speed_ratio",
            "terminal_braking_enabled",
            "minimum_speed_ratio",
            "terminal_alignment_horizon_s",
            "terminal_alignment_min_m",
            "terminal_alignment_max_m",
            "terminal_path_preserving_enabled",
            "terminal_reference_extension_m",
            "current_crab_enabled",
            "current_estimate_max_age_s",
            "current_estimate_tau_s",
            "current_estimate_max_speed_mps",
            "current_crab_relative_speed_limit",
            "current_crab_max_angle_rad",
            "speed_adaptive_lookahead_enabled",
            "saturation_governor_enabled",
            "command_rate_limit_enabled",
            "saturation_ema_tau_s",
            "saturation_governor_start",
            "saturation_governor_span",
            "saturation_governor_max_reduction",
            "saturation_governor_min_scale",
            "saturation_emergency_threshold",
            "saturation_emergency_duration_s",
            "saturation_emergency_extra_reduction",
            "route_speed_floor_enabled",
            "route_speed_floor_gain",
            "route_speed_floor_integral_gain",
            "route_speed_floor_max_boost_mps",
            "route_speed_floor_filter_tau_s",
            "route_speed_floor_dvl_max_age_s",
            "route_speed_floor_launch_speed_ratio",
            "route_speed_floor_launch_duration_s",
            "initial_speed_ratio",
            "initial_propeller_rpm_ratio",
            "initial_speed_ratio_range",
            "initial_propeller_rpm_ratio_range",
            "high_speed_threshold_mps",
            "heading_rate_limit_rad_s",
            "pitch_rate_limit_rad_s",
            "speed_rate_limit_mps2",
        },
    )
    return GuidanceConfig(
        horizontal_lookahead_m=_positive(
            data["horizontal_lookahead_m"], "guidance.horizontal_lookahead_m"
        ),
        vertical_lookahead_m=_positive(
            data["vertical_lookahead_m"], "guidance.vertical_lookahead_m"
        ),
        min_lookahead_m=_positive(data["min_lookahead_m"], "guidance.min_lookahead_m"),
        max_lookahead_m=_positive(data["max_lookahead_m"], "guidance.max_lookahead_m"),
        horizontal_gain=_non_negative(data["horizontal_gain"], "guidance.horizontal_gain"),
        vertical_gain=_non_negative(data["vertical_gain"], "guidance.vertical_gain"),
        lookahead_error_gain=_non_negative(
            data["lookahead_error_gain"], "guidance.lookahead_error_gain"
        ),
        validity_period_s=_positive(data["validity_period_s"], "guidance.validity_period_s"),
        compensate_current=_boolean(data["compensate_current"], "guidance.compensate_current"),
        terminal_capture_distance_m=_positive(
            data["terminal_capture_distance_m"], "guidance.terminal_capture_distance_m"
        ),
        terminal_speed_mps=_non_negative(
            data.get("terminal_speed_mps", 0.0), "guidance.terminal_speed_mps"
        ),
        robust_terminal_enabled=_boolean(
            data.get("robust_terminal_enabled", False),
            "guidance.robust_terminal_enabled",
        ),
        terminal_speed_ratio=_positive(
            data.get("terminal_speed_ratio", 1.0), "guidance.terminal_speed_ratio"
        ),
        terminal_braking_enabled=_boolean(
            data.get("terminal_braking_enabled", False),
            "guidance.terminal_braking_enabled",
        ),
        minimum_speed_ratio=_positive(
            data.get("minimum_speed_ratio", 0.75),
            "guidance.minimum_speed_ratio",
        ),
        terminal_alignment_horizon_s=_positive(
            data.get("terminal_alignment_horizon_s", 6.0),
            "guidance.terminal_alignment_horizon_s",
        ),
        terminal_alignment_min_m=_positive(
            data.get("terminal_alignment_min_m", 6.0),
            "guidance.terminal_alignment_min_m",
        ),
        terminal_alignment_max_m=_positive(
            data.get("terminal_alignment_max_m", 16.0),
            "guidance.terminal_alignment_max_m",
        ),
        terminal_path_preserving_enabled=_boolean(
            data.get("terminal_path_preserving_enabled", False),
            "guidance.terminal_path_preserving_enabled",
        ),
        terminal_reference_extension_m=_non_negative(
            data.get("terminal_reference_extension_m", 0.0),
            "guidance.terminal_reference_extension_m",
        ),
        current_crab_enabled=_boolean(
            data.get("current_crab_enabled", False), "guidance.current_crab_enabled"
        ),
        current_estimate_max_age_s=_positive(
            data.get("current_estimate_max_age_s", 0.6),
            "guidance.current_estimate_max_age_s",
        ),
        current_estimate_tau_s=_positive(
            data.get("current_estimate_tau_s", 1.5),
            "guidance.current_estimate_tau_s",
        ),
        current_estimate_max_speed_mps=_positive(
            data.get("current_estimate_max_speed_mps", 1.5),
            "guidance.current_estimate_max_speed_mps",
        ),
        current_crab_relative_speed_limit=_positive(
            data.get("current_crab_relative_speed_limit", 1.12),
            "guidance.current_crab_relative_speed_limit",
        ),
        current_crab_max_angle_rad=_positive(
            data.get("current_crab_max_angle_rad", math.radians(18.0)),
            "guidance.current_crab_max_angle_rad",
        ),
        speed_adaptive_lookahead_enabled=_boolean(
            data.get("speed_adaptive_lookahead_enabled", False),
            "guidance.speed_adaptive_lookahead_enabled",
        ),
        saturation_governor_enabled=_boolean(
            data.get("saturation_governor_enabled", False),
            "guidance.saturation_governor_enabled",
        ),
        command_rate_limit_enabled=_optional_boolean(
            data.get("command_rate_limit_enabled"),
            "guidance.command_rate_limit_enabled",
        ),
        saturation_ema_tau_s=_positive(
            data.get("saturation_ema_tau_s", 1.0), "guidance.saturation_ema_tau_s"
        ),
        saturation_governor_start=_non_negative(
            data.get("saturation_governor_start", 0.25),
            "guidance.saturation_governor_start",
        ),
        saturation_governor_span=_positive(
            data.get("saturation_governor_span", 0.30),
            "guidance.saturation_governor_span",
        ),
        saturation_governor_max_reduction=_non_negative(
            data.get("saturation_governor_max_reduction", 0.12),
            "guidance.saturation_governor_max_reduction",
        ),
        saturation_governor_min_scale=_positive(
            data.get("saturation_governor_min_scale", 0.88),
            "guidance.saturation_governor_min_scale",
        ),
        saturation_emergency_threshold=_non_negative(
            data.get("saturation_emergency_threshold", 0.70),
            "guidance.saturation_emergency_threshold",
        ),
        saturation_emergency_duration_s=_positive(
            data.get("saturation_emergency_duration_s", 2.0),
            "guidance.saturation_emergency_duration_s",
        ),
        saturation_emergency_extra_reduction=_non_negative(
            data.get("saturation_emergency_extra_reduction", 0.05),
            "guidance.saturation_emergency_extra_reduction",
        ),
        route_speed_floor_enabled=_boolean(
            data.get("route_speed_floor_enabled", False),
            "guidance.route_speed_floor_enabled",
        ),
        route_speed_floor_gain=_non_negative(
            data.get("route_speed_floor_gain", 1.0),
            "guidance.route_speed_floor_gain",
        ),
        route_speed_floor_integral_gain=_non_negative(
            data.get("route_speed_floor_integral_gain", 0.5),
            "guidance.route_speed_floor_integral_gain",
        ),
        route_speed_floor_max_boost_mps=_non_negative(
            data.get("route_speed_floor_max_boost_mps", 0.8),
            "guidance.route_speed_floor_max_boost_mps",
        ),
        route_speed_floor_filter_tau_s=_positive(
            data.get("route_speed_floor_filter_tau_s", 0.5),
            "guidance.route_speed_floor_filter_tau_s",
        ),
        route_speed_floor_dvl_max_age_s=_positive(
            data.get("route_speed_floor_dvl_max_age_s", 0.6),
            "guidance.route_speed_floor_dvl_max_age_s",
        ),
        route_speed_floor_launch_speed_ratio=_positive(
            data.get("route_speed_floor_launch_speed_ratio", 1.0),
            "guidance.route_speed_floor_launch_speed_ratio",
        ),
        route_speed_floor_launch_duration_s=_positive(
            data.get("route_speed_floor_launch_duration_s", 2.0),
            "guidance.route_speed_floor_launch_duration_s",
        ),
        initial_propeller_rpm_ratio=_non_negative(
            data.get("initial_propeller_rpm_ratio", 0.0),
            "guidance.initial_propeller_rpm_ratio",
        ),
        initial_speed_ratio=_positive(
            data.get("initial_speed_ratio", 1.0),
            "guidance.initial_speed_ratio",
        ),
        initial_propeller_rpm_ratio_range=_optional_range(
            data.get("initial_propeller_rpm_ratio_range"),
            "guidance.initial_propeller_rpm_ratio_range",
        ),
        initial_speed_ratio_range=_optional_range(
            data.get("initial_speed_ratio_range"),
            "guidance.initial_speed_ratio_range",
        ),
        high_speed_threshold_mps=_positive(
            data.get("high_speed_threshold_mps", 1.8),
            "guidance.high_speed_threshold_mps",
        ),
        heading_rate_limit_rad_s=_positive(
            data.get("heading_rate_limit_rad_s", math.radians(30.0)),
            "guidance.heading_rate_limit_rad_s",
        ),
        pitch_rate_limit_rad_s=_positive(
            data.get("pitch_rate_limit_rad_s", math.radians(15.0)),
            "guidance.pitch_rate_limit_rad_s",
        ),
        speed_rate_limit_mps2=_positive(
            data.get("speed_rate_limit_mps2", 0.25),
            "guidance.speed_rate_limit_mps2",
        ),
    )


def _parse_controller(value: Any) -> ControllerConfig:
    data = _mapping(
        value,
        "base_controller",
        {
            "type",
            "use_current_field",
            "nominal_step_s",
            "heading_gain",
            "heading_rate_gain",
            "pitch_gain",
            "pitch_rate_gain",
            "speed_feedforward_rpm_per_mps",
            "speed_gain_rpm_per_mps",
            "speed_integral_gain_rpm_per_mps_s",
            "max_speed_integral_rpm",
        },
    )
    if data["type"] != "remus100_tracking_pd_pi":
        raise ValueError("unsupported base_controller.type")
    return ControllerConfig(
        use_current_field=_boolean(data["use_current_field"], "base_controller.use_current_field"),
        nominal_step_s=_positive(data["nominal_step_s"], "base_controller.nominal_step_s"),
        heading_gain=_non_negative(data["heading_gain"], "base_controller.heading_gain"),
        heading_rate_gain=_non_negative(
            data["heading_rate_gain"], "base_controller.heading_rate_gain"
        ),
        pitch_gain=_non_negative(data["pitch_gain"], "base_controller.pitch_gain"),
        pitch_rate_gain=_non_negative(data["pitch_rate_gain"], "base_controller.pitch_rate_gain"),
        speed_feedforward_rpm_per_mps=_non_negative(
            data["speed_feedforward_rpm_per_mps"], "base_controller.speed_feedforward_rpm_per_mps"
        ),
        speed_gain_rpm_per_mps=_non_negative(
            data["speed_gain_rpm_per_mps"], "base_controller.speed_gain_rpm_per_mps"
        ),
        speed_integral_gain_rpm_per_mps_s=_non_negative(
            data["speed_integral_gain_rpm_per_mps_s"],
            "base_controller.speed_integral_gain_rpm_per_mps_s",
        ),
        max_speed_integral_rpm=_positive(
            data["max_speed_integral_rpm"], "base_controller.max_speed_integral_rpm"
        ),
    )


def _parse_residual(value: Any) -> ResidualConfig:
    data = _mapping(value, "residual", {"scale_rudder", "scale_stern", "scale_rpm", "max_action"})
    return ResidualConfig(
        scale_rudder=_non_negative(data["scale_rudder"], "residual.scale_rudder"),
        scale_stern=_non_negative(data["scale_stern"], "residual.scale_stern"),
        scale_rpm=_non_negative(data["scale_rpm"], "residual.scale_rpm"),
        max_action=_positive(data["max_action"], "residual.max_action"),
    )


def _parse_command_residual(value: Any) -> CommandResidualConfig:
    if value is None:
        return CommandResidualConfig()
    data = _mapping(
        value,
        "command_residual",
        set(),
        optional={
            "enabled",
            "scale_heading_rad",
            "scale_pitch_rad",
            "scale_speed_mps",
            "rate_heading_rad_s",
            "rate_pitch_rad_s",
            "rate_speed_mps2",
            "controller_state",
            "terminal_gate_enabled",
            "terminal_gate_enter_m",
            "terminal_gate_full_m",
        },
    )
    controller_state = _string(
        data.get("controller_state", "actor"), "command_residual.controller_state"
    )
    if controller_state not in {"actor", "privileged"}:
        raise ValueError(
            "command_residual.controller_state must be 'actor' or 'privileged'"
        )
    return CommandResidualConfig(
        enabled=_boolean(data.get("enabled", False), "command_residual.enabled"),
        scale_heading_rad=_positive(
            data.get("scale_heading_rad", 0.35), "command_residual.scale_heading_rad"
        ),
        scale_pitch_rad=_positive(
            data.get("scale_pitch_rad", 0.17), "command_residual.scale_pitch_rad"
        ),
        scale_speed_mps=_positive(
            data.get("scale_speed_mps", 0.60), "command_residual.scale_speed_mps"
        ),
        rate_heading_rad_s=_positive(
            data.get("rate_heading_rad_s", 0.25), "command_residual.rate_heading_rad_s"
        ),
        rate_pitch_rad_s=_positive(
            data.get("rate_pitch_rad_s", 0.15), "command_residual.rate_pitch_rad_s"
        ),
        rate_speed_mps2=_positive(
            data.get("rate_speed_mps2", 0.80), "command_residual.rate_speed_mps2"
        ),
        controller_state=controller_state,
        terminal_gate_enabled=_boolean(
            data.get("terminal_gate_enabled", False),
            "command_residual.terminal_gate_enabled",
        ),
        terminal_gate_enter_m=_positive(
            data.get("terminal_gate_enter_m", 20.0),
            "command_residual.terminal_gate_enter_m",
        ),
        terminal_gate_full_m=_positive(
            data.get("terminal_gate_full_m", 5.0),
            "command_residual.terminal_gate_full_m",
        ),
    )


def _parse_controller_gain_residual(value: Any) -> ControllerGainResidualConfig:
    if value is None:
        return ControllerGainResidualConfig()
    data = _mapping(
        value,
        "controller_gain_residual",
        set(),
        optional={
            "enabled",
            "heading_gain_delta",
            "pitch_gain_delta",
            "controller_state",
            "terminal_gate_enabled",
            "terminal_gate_enter_m",
            "terminal_gate_full_m",
        },
    )
    return ControllerGainResidualConfig(
        enabled=_boolean(
            data.get("enabled", False), "controller_gain_residual.enabled"
        ),
        heading_gain_delta=_positive(
            data.get("heading_gain_delta", 1.0),
            "controller_gain_residual.heading_gain_delta",
        ),
        pitch_gain_delta=_positive(
            data.get("pitch_gain_delta", 1.5),
            "controller_gain_residual.pitch_gain_delta",
        ),
        controller_state=_string(
            data.get("controller_state", "actor"),
            "controller_gain_residual.controller_state",
        ),
        terminal_gate_enabled=_boolean(
            data.get("terminal_gate_enabled", True),
            "controller_gain_residual.terminal_gate_enabled",
        ),
        terminal_gate_enter_m=_positive(
            data.get("terminal_gate_enter_m", 10.0),
            "controller_gain_residual.terminal_gate_enter_m",
        ),
        terminal_gate_full_m=_positive(
            data.get("terminal_gate_full_m", 2.0),
            "controller_gain_residual.terminal_gate_full_m",
        ),
    )


def _parse_observation(value: Any) -> ObservationConfig:
    data = _mapping(
        value,
        "observation",
        {"dim", "expose_current"},
        optional={
            "speed_feedback",
            "current_contract",
            "path_endpoint_context_enabled",
            "path_endpoint_enter_m",
            "path_endpoint_full_m",
            "ideal_actor_observations",
        },
    )
    dim = _integer(data["dim"], "observation.dim")
    if dim not in {26, 27}:
        raise ValueError("observation.dim must be 26 (v2) or 27 (v1 compatibility)")
    speed_feedback = _string(
        data.get("speed_feedback", "water_relative"),
        "observation.speed_feedback",
    )
    if speed_feedback not in {"water_relative", "ground"}:
        raise ValueError("observation.speed_feedback must be 'water_relative' or 'ground'")
    expose_current = _boolean(data["expose_current"], "observation.expose_current")
    current_contract = _string(
        data.get("current_contract", "oracle" if expose_current else "legacy"),
        "observation.current_contract",
    )
    if current_contract not in {"legacy", "hidden", "oracle"}:
        raise ValueError("observation.current_contract must be 'legacy', 'hidden', or 'oracle'")
    return ObservationConfig(
        dim=dim,
        expose_current=expose_current,
        speed_feedback=speed_feedback,
        current_contract=current_contract,
        path_endpoint_context_enabled=_boolean(
            data.get("path_endpoint_context_enabled", False),
            "observation.path_endpoint_context_enabled",
        ),
        path_endpoint_enter_m=_positive(
            data.get("path_endpoint_enter_m", 20.0),
            "observation.path_endpoint_enter_m",
        ),
        path_endpoint_full_m=_positive(
            data.get("path_endpoint_full_m", 5.0),
            "observation.path_endpoint_full_m",
        ),
        ideal_actor_observations=_boolean(
            data.get("ideal_actor_observations", False),
            "observation.ideal_actor_observations",
        ),
    )


def _parse_reward(value: Any) -> RewardConfig:
    keys = {
        "w_prog",
        "w_3d",
        "w_yaw",
        "w_u",
        "w_action",
        "w_sat",
        "goal_bonus",
        "timeout_penalty",
        "out_of_bounds_penalty",
    }
    data = _mapping(
        value,
        "reward",
        keys,
        optional={
            "terminal_progress_weight",
            "mode",
            "hold_bonus",
            "w_cmd_yaw",
            "w_cmd_pitch",
            "w_cmd_speed",
            "w_hold",
            "w_action_pc",
            "w_sat_pc",
            "w_route_speed_pc",
            "route_speed_floor_ratio",
            "w_material_sat_pc",
            "w_path_speed_smooth",
            "w_path_energy",
            "w_path_3d",
            "w_path_cross",
            "w_path_vertical",
            "w_path_progress",
            "route_middle_gate",
            "path_cross_huber_delta_m",
            "path_vertical_huber_delta_m",
            "w_approach_vertical",
            "w_approach_pitch",
            "w_vertical_potential",
            "w_heave_potential",
            "vertical_potential_huber_delta_m",
            "heave_potential_huber_delta_mps",
            "potential_gamma",
            "approach_enter_m",
            "approach_full_m",
            "w_phi",
            "w_p",
            "phi_dead_deg",
            "phi_soft_deg",
            "phi_strong_deg",
            "p_dead_deg_s",
            "p_soft_deg_s",
            "hold_yaw_tolerance_deg",
            "hold_pitch_tolerance_deg",
            "hold_speed_tolerance_mps",
            "hold_time_s",
            "pc_ppo_scale",
        },
    )
    mode = _string(data.get("mode", "legacy_path_track"), "reward.mode")
    if mode not in SUPPORTED_REWARD_MODES:
        raise ValueError(f"reward.mode must be one of {sorted(SUPPORTED_REWARD_MODES)}")
    phi_dead = math.radians(_non_negative(data.get("phi_dead_deg", 5.0), "reward.phi_dead_deg"))
    phi_soft = math.radians(_positive(data.get("phi_soft_deg", 10.0), "reward.phi_soft_deg"))
    phi_strong = math.radians(_positive(data.get("phi_strong_deg", 30.0), "reward.phi_strong_deg"))
    p_dead = math.radians(_non_negative(data.get("p_dead_deg_s", 10.0), "reward.p_dead_deg_s"))
    p_soft = math.radians(_positive(data.get("p_soft_deg_s", 30.0), "reward.p_soft_deg_s"))
    if not phi_dead < phi_soft < phi_strong:
        raise ValueError("reward roll thresholds must satisfy dead < soft < strong")
    if not p_dead < p_soft:
        raise ValueError("reward roll-rate thresholds must satisfy dead < soft")
    return RewardConfig(
        w_prog=_non_negative(data["w_prog"], "reward.w_prog"),
        w_3d=_non_negative(data["w_3d"], "reward.w_3d"),
        w_yaw=_non_negative(data["w_yaw"], "reward.w_yaw"),
        w_u=_non_negative(data["w_u"], "reward.w_u"),
        w_action=_non_negative(data["w_action"], "reward.w_action"),
        w_sat=_non_negative(data["w_sat"], "reward.w_sat"),
        terminal_progress_weight=_non_negative(
            data.get("terminal_progress_weight", 0.0),
            "reward.terminal_progress_weight",
        ),
        goal_bonus=_number(data["goal_bonus"], "reward.goal_bonus"),
        timeout_penalty=_number(data["timeout_penalty"], "reward.timeout_penalty"),
        out_of_bounds_penalty=_number(
            data["out_of_bounds_penalty"], "reward.out_of_bounds_penalty"
        ),
        mode=mode,
        hold_bonus=_boolean(data.get("hold_bonus", False), "reward.hold_bonus"),
        w_cmd_yaw=_non_negative(data.get("w_cmd_yaw", 1.5), "reward.w_cmd_yaw"),
        w_cmd_pitch=_non_negative(data.get("w_cmd_pitch", 0.5), "reward.w_cmd_pitch"),
        w_cmd_speed=_non_negative(data.get("w_cmd_speed", 1.0), "reward.w_cmd_speed"),
        w_hold=_non_negative(data.get("w_hold", 5.0), "reward.w_hold"),
        w_action_pc=_non_negative(data.get("w_action_pc", 0.5), "reward.w_action_pc"),
        w_sat_pc=_non_negative(data.get("w_sat_pc", 0.1), "reward.w_sat_pc"),
        w_route_speed_pc=_non_negative(
            data.get("w_route_speed_pc", 0.0), "reward.w_route_speed_pc"
        ),
        route_speed_floor_ratio=_positive(
            data.get("route_speed_floor_ratio", 0.75),
            "reward.route_speed_floor_ratio",
        ),
        w_material_sat_pc=_non_negative(
            data.get("w_material_sat_pc", 0.0), "reward.w_material_sat_pc"
        ),
        w_path_speed_smooth=_non_negative(
            data.get("w_path_speed_smooth", 0.0),
            "reward.w_path_speed_smooth",
        ),
        w_path_energy=_non_negative(
            data.get("w_path_energy", 0.0),
            "reward.w_path_energy",
        ),
        w_path_3d=_non_negative(
            data.get("w_path_3d", 0.0),
            "reward.w_path_3d",
        ),
        w_path_cross=_non_negative(
            data.get("w_path_cross", 0.0),
            "reward.w_path_cross",
        ),
        w_path_vertical=_non_negative(
            data.get("w_path_vertical", 0.0),
            "reward.w_path_vertical",
        ),
        w_path_progress=_non_negative(
            data.get("w_path_progress", 0.0),
            "reward.w_path_progress",
        ),
        route_middle_gate=_non_negative(
            data.get("route_middle_gate", 0.0),
            "reward.route_middle_gate",
        ),
        path_cross_huber_delta_m=_positive(
            data.get("path_cross_huber_delta_m", 0.5),
            "reward.path_cross_huber_delta_m",
        ),
        path_vertical_huber_delta_m=_positive(
            data.get("path_vertical_huber_delta_m", 0.5),
            "reward.path_vertical_huber_delta_m",
        ),
        w_approach_vertical=_non_negative(
            data.get("w_approach_vertical", 0.0),
            "reward.w_approach_vertical",
        ),
        w_approach_pitch=_non_negative(
            data.get("w_approach_pitch", 0.0),
            "reward.w_approach_pitch",
        ),
        w_vertical_potential=_non_negative(
            data.get("w_vertical_potential", 0.0),
            "reward.w_vertical_potential",
        ),
        w_heave_potential=_non_negative(
            data.get("w_heave_potential", 0.0),
            "reward.w_heave_potential",
        ),
        vertical_potential_huber_delta_m=_positive(
            data.get("vertical_potential_huber_delta_m", 0.25),
            "reward.vertical_potential_huber_delta_m",
        ),
        heave_potential_huber_delta_mps=_positive(
            data.get("heave_potential_huber_delta_mps", 0.05),
            "reward.heave_potential_huber_delta_mps",
        ),
        potential_gamma=_positive(
            data.get("potential_gamma", 0.995),
            "reward.potential_gamma",
        ),
        approach_enter_m=_positive(
            data.get("approach_enter_m", 5.0),
            "reward.approach_enter_m",
        ),
        approach_full_m=_positive(
            data.get("approach_full_m", 2.0),
            "reward.approach_full_m",
        ),
        w_phi=_non_negative(data.get("w_phi", 0.05), "reward.w_phi"),
        w_p=_non_negative(data.get("w_p", 0.02), "reward.w_p"),
        phi_dead_rad=phi_dead,
        phi_soft_rad=phi_soft,
        phi_strong_rad=phi_strong,
        p_dead_rad_s=p_dead,
        p_soft_rad_s=p_soft,
        hold_yaw_tolerance_rad=math.radians(
            _positive(
                data.get("hold_yaw_tolerance_deg", 5.0),
                "reward.hold_yaw_tolerance_deg",
            )
        ),
        hold_pitch_tolerance_rad=math.radians(
            _positive(
                data.get("hold_pitch_tolerance_deg", 5.0),
                "reward.hold_pitch_tolerance_deg",
            )
        ),
        hold_speed_tolerance_mps=_positive(
            data.get("hold_speed_tolerance_mps", 0.1),
            "reward.hold_speed_tolerance_mps",
        ),
        hold_time_s=_positive(data.get("hold_time_s", 2.0), "reward.hold_time_s"),
        pc_ppo_scale=(
            None
            if "pc_ppo_scale" not in data
            else _positive(data["pc_ppo_scale"], "reward.pc_ppo_scale")
        ),
    )


def _parse_termination(value: Any) -> TerminationConfig:
    if value is None:
        return TerminationConfig()
    data = _mapping(
        value,
        "termination",
        set(),
        optional={
            "phi_term_deg",
            "t_phi_term_s",
            "p_term_deg_s",
            "t_p_term_s",
            "safety_terminal_penalty",
            "min_depth_m",
            "max_depth_m",
            "pc_goal_plane_miss_termination_enabled",
            "goal_plane_miss_penalty",
            "goal_plane_miss_distance_weight",
            "position_source",
        },
    )
    min_depth = data.get("min_depth_m")
    max_depth = data.get("max_depth_m")
    return TerminationConfig(
        phi_term_rad=math.radians(
            _positive(data.get("phi_term_deg", 45.0), "termination.phi_term_deg")
        ),
        t_phi_term_s=_positive(data.get("t_phi_term_s", 1.0), "termination.t_phi_term_s"),
        p_term_rad_s=math.radians(
            _positive(data.get("p_term_deg_s", 90.0), "termination.p_term_deg_s")
        ),
        t_p_term_s=_positive(data.get("t_p_term_s", 0.5), "termination.t_p_term_s"),
        safety_terminal_penalty=_number(
            data.get("safety_terminal_penalty", -50.0),
            "termination.safety_terminal_penalty",
        ),
        min_depth_m=(None if min_depth is None else _number(min_depth, "termination.min_depth_m")),
        max_depth_m=(None if max_depth is None else _number(max_depth, "termination.max_depth_m")),
        pc_goal_plane_miss_termination_enabled=_boolean(
            data.get("pc_goal_plane_miss_termination_enabled", False),
            "termination.pc_goal_plane_miss_termination_enabled",
        ),
        goal_plane_miss_penalty=_number(
            data.get("goal_plane_miss_penalty", 0.0),
            "termination.goal_plane_miss_penalty",
        ),
        goal_plane_miss_distance_weight=_number(
            data.get("goal_plane_miss_distance_weight", 0.0),
            "termination.goal_plane_miss_distance_weight",
        ),
        position_source=_string(
            data.get("position_source", "legacy"),
            "termination.position_source",
        ),
    )


def _parse_sensors(value: Any) -> SensorConfig | None:
    if value is None:
        return None
    data = _mapping(
        value,
        "sensors",
        {"profile"},
        optional={
            "seed",
            "current_est_lpf_time_basis",
            "noise_scale",
            "noise_scale_range",
        },
    )
    seed = data.get("seed")
    return SensorConfig(
        profile=_string(data["profile"], "sensors.profile"),
        seed=None if seed is None else _integer(seed, "sensors.seed"),
        current_est_lpf_time_basis=_string(
            data.get("current_est_lpf_time_basis", "control_dt_legacy"),
            "sensors.current_est_lpf_time_basis",
        ),
        noise_scale=_non_negative(
            data.get("noise_scale", 1.0),
            "sensors.noise_scale",
        ),
        noise_scale_range=_optional_range(
            data.get("noise_scale_range"),
            "sensors.noise_scale_range",
        ),
    )


def _parse_normalization(value: Any) -> NormalizationConfig:
    if value is None:
        return NormalizationConfig()
    data = _mapping(
        value,
        "normalization",
        {
            "position_scale_m",
            "attitude_scale_rad",
            "angular_rate_scale_rad_s",
            "surge_speed_scale_mps",
            "sway_heave_speed_scale_mps",
            "current_scale_mps",
        },
    )
    return NormalizationConfig(
        position_scale_m=_positive(data["position_scale_m"], "normalization.position_scale_m"),
        attitude_scale_rad=_positive(
            data["attitude_scale_rad"], "normalization.attitude_scale_rad"
        ),
        angular_rate_scale_rad_s=_positive(
            data["angular_rate_scale_rad_s"],
            "normalization.angular_rate_scale_rad_s",
        ),
        surge_speed_scale_mps=_positive(
            data["surge_speed_scale_mps"],
            "normalization.surge_speed_scale_mps",
        ),
        sway_heave_speed_scale_mps=_positive(
            data["sway_heave_speed_scale_mps"],
            "normalization.sway_heave_speed_scale_mps",
        ),
        current_scale_mps=_positive(data["current_scale_mps"], "normalization.current_scale_mps"),
    )


def _parse_evaluation(value: Any) -> EvaluationConfig:
    data = _mapping(value, "evaluation", {"seed_start", "seed_count", "training_seed_start"})
    seed_start = _integer(data["seed_start"], "evaluation.seed_start")
    seed_count = _integer(data["seed_count"], "evaluation.seed_count")
    training_seed_start = _integer(data["training_seed_start"], "evaluation.training_seed_start")
    if seed_start < 0 or seed_count <= 0 or training_seed_start < 0:
        raise ValueError("evaluation seeds must be non-negative and non-empty")
    if training_seed_start < seed_start + seed_count:
        raise ValueError("training seed start must be outside the evaluation seed pool")
    return EvaluationConfig(seed_start, seed_count, training_seed_start)


def _parse_curriculum(value: Any) -> CurriculumConfig:
    if value is None:
        return CurriculumConfig()
    data = _mapping(
        value,
        "curriculum",
        {"enabled"},
        optional={
            "promotion_enabled",
            "initial_stage",
            "stage_mix",
            "manifest_filename",
            "current_speed_range_override_mps",
            "shear_scale_override",
        },
    )
    enabled = _boolean(data["enabled"], "curriculum.enabled")
    promotion_enabled = _boolean(
        data.get("promotion_enabled", True),
        "curriculum.promotion_enabled",
    )
    initial_stage = _string(
        data.get("initial_stage", "C0"), "curriculum.initial_stage"
    ).upper()
    raw_mix = data.get("stage_mix", [])
    if not isinstance(raw_mix, list):
        raise ValueError("curriculum.stage_mix must be a list")
    stage_mix: list[tuple[str, float]] = []
    for index, item in enumerate(raw_mix):
        if not isinstance(item, dict) or set(item) != {"stage", "weight"}:
            raise ValueError(
                f"curriculum.stage_mix[{index}] must contain stage and weight"
            )
        stage_mix.append(
            (
                _string(item["stage"], f"curriculum.stage_mix[{index}].stage").upper(),
                _number(item["weight"], f"curriculum.stage_mix[{index}].weight"),
            )
        )
    manifest_filename = _string(
        data.get("manifest_filename", "curriculum_episodes.jsonl"),
        "curriculum.manifest_filename",
    )
    raw_speed_override = data.get("current_speed_range_override_mps")
    if raw_speed_override is None:
        speed_override = None
    else:
        if not isinstance(raw_speed_override, list) or len(raw_speed_override) != 2:
            raise ValueError(
                "curriculum.current_speed_range_override_mps must be a 2-item list"
            )
        speed_override = (
            _number(
                raw_speed_override[0],
                "curriculum.current_speed_range_override_mps[0]",
            ),
            _number(
                raw_speed_override[1],
                "curriculum.current_speed_range_override_mps[1]",
            ),
        )
    raw_shear_override = data.get("shear_scale_override")
    shear_override = (
        None
        if raw_shear_override is None
        else _number(raw_shear_override, "curriculum.shear_scale_override")
    )
    return CurriculumConfig(
        enabled=enabled,
        promotion_enabled=promotion_enabled,
        initial_stage=initial_stage,
        stage_mix=tuple(stage_mix),
        manifest_filename=manifest_filename,
        current_speed_range_override_mps=speed_override,
        shear_scale_override=shear_override,
    )


def parse_config(data: dict[str, Any], *, config_sha256: str = "") -> LowLevelConfig:
    root = _mapping(
        data,
        "config",
        {
            "schema_version",
            "experiment",
            "plant",
            "reference",
            "current",
            "guidance",
            "base_controller",
            "residual",
            "observation",
            "reward",
            "evaluation",
        },
        optional={
            "actuator",
            "termination",
            "sensors",
            "normalization",
            "domain",
            "command_residual",
            "controller_gain_residual",
            "plant_randomization",
            "curriculum",
        },
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    experiment = _mapping(
        root["experiment"],
        "experiment",
        {"name", "arm", "seed", "output"},
        optional={"policy_mode", "sensor_source", "pc_stage", "pc_control_mode"},
    )
    policy_mode = _string(experiment.get("policy_mode", "residual"), "experiment.policy_mode")
    if policy_mode not in SUPPORTED_POLICY_MODES:
        raise ValueError(f"experiment.policy_mode must be one of {sorted(SUPPORTED_POLICY_MODES)}")
    sensor_source = _string(
        experiment.get("sensor_source", "simulation_truth"),
        "experiment.sensor_source",
    )
    if sensor_source not in SUPPORTED_SENSOR_SOURCES:
        raise ValueError(
            f"experiment.sensor_source must be one of {sorted(SUPPORTED_SENSOR_SOURCES)}"
        )
    pc_stage_value = experiment.get("pc_stage")
    pc_stage = (
        None if pc_stage_value is None else _string(pc_stage_value, "experiment.pc_stage").upper()
    )
    if pc_stage is not None and pc_stage not in SUPPORTED_PC_STAGES:
        raise ValueError(f"experiment.pc_stage must be one of {sorted(SUPPORTED_PC_STAGES)}")
    pc_control_mode = _string(
        experiment.get("pc_control_mode", "alos"),
        "experiment.pc_control_mode",
    )
    if pc_control_mode not in SUPPORTED_PC_CONTROL_MODES:
        raise ValueError(
            f"experiment.pc_control_mode must be one of {sorted(SUPPORTED_PC_CONTROL_MODES)}"
        )
    config = LowLevelConfig(
        schema_version=SCHEMA_VERSION,
        name=_string(experiment["name"], "experiment.name"),
        arm=_string(experiment["arm"], "experiment.arm"),
        policy_mode=policy_mode,
        sensor_source=sensor_source,
        pc_stage=pc_stage,
        pc_control_mode=pc_control_mode,
        seed=_integer(experiment["seed"], "experiment.seed"),
        output=_string(experiment["output"], "experiment.output"),
        plant=_parse_plant(root["plant"]),
        actuator=_parse_actuator(root.get("actuator")),
        reference=_parse_reference(root["reference"]),
        current=_parse_current(root["current"]),
        domain=_parse_domain(root.get("domain")),
        guidance=_parse_guidance(root["guidance"]),
        controller=_parse_controller(root["base_controller"]),
        residual=_parse_residual(root["residual"]),
        command_residual=_parse_command_residual(root.get("command_residual")),
        controller_gain_residual=_parse_controller_gain_residual(
            root.get("controller_gain_residual")
        ),
        observation=_parse_observation(root["observation"]),
        reward=_parse_reward(root["reward"]),
        termination=_parse_termination(root.get("termination")),
        sensors=_parse_sensors(root.get("sensors")),
        normalization=_parse_normalization(root.get("normalization")),
        evaluation=_parse_evaluation(root["evaluation"]),
        curriculum=_parse_curriculum(root.get("curriculum")),
        plant_randomization=_parse_plant_randomization(root.get("plant_randomization")),
        config_sha256=config_sha256,
    )
    _validate_pc_contract(config)
    _validate_reward_termination_contract(config)
    _validate_current_visibility(config)
    return config


def _validate_pc_contract(config: LowLevelConfig) -> None:
    """Reject partially configured PC runs before any environment is built.

    The legacy residual arms keep their historical 15-degree contract. A PC
    run must be explicit and internally consistent: pure-controller actions,
    a 20-degree policy/plant envelope, ground-speed feedback, no current
    compensation in ALOS, and the synthetic sensor source selected for v1.
    """

    is_pc = config.arm == "PC" or config.policy_mode == "pure_controller"
    if not is_pc:
        return
    if config.arm != "PC":
        raise ValueError("policy_mode='pure_controller' requires experiment.arm='PC'")
    if config.policy_mode != "pure_controller":
        raise ValueError("experiment.arm='PC' requires policy_mode='pure_controller'")
    if config.pc_stage is None:
        raise ValueError("PC requires experiment.pc_stage in C0..C6")
    if config.actuator.envelope != "pc20":
        raise ValueError("PC requires actuator.envelope='pc20'")
    expected_fin = math.radians(20.0)
    if not math.isclose(
        config.actuator.policy_max_rudder_angle_rad,
        expected_fin,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        config.actuator.policy_max_stern_plane_angle_rad,
        expected_fin,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("PC policy fin limits must both equal 20 degrees")
    if config.actuator.plant_max_rudder_angle_rad + 1e-12 < expected_fin:
        raise ValueError("PC plant rudder limit must be at least 20 degrees")
    if config.actuator.plant_max_stern_plane_angle_rad + 1e-12 < expected_fin:
        raise ValueError("PC plant stern-plane limit must be at least 20 degrees")
    if config.plant.backend == "pvs_remus100":
        raise ValueError("PVS REMUS100 is limited to 15 degrees and cannot back a PC20 run")
    if config.guidance.compensate_current:
        raise ValueError("PC requires guidance.compensate_current=false")
    if config.controller.use_current_field:
        raise ValueError("PC requires base_controller.use_current_field=false")
    if config.plant.velocity_contract == "legacy_water_nu":
        if config.observation.speed_feedback != "ground":
            raise ValueError(
                "legacy PC requires observation.speed_feedback='ground'"
            )
    elif config.observation.speed_feedback not in {"ground", "water_relative"}:
        raise ValueError(
            "nu_ground PC requires observation.speed_feedback in "
            "{'ground', 'water_relative'}"
        )
    if config.sensor_source != "synthetic":
        raise ValueError("PC v1 requires experiment.sensor_source='synthetic'")
    if config.sensors is None:
        raise ValueError("PC v1 requires an explicit sensors profile")
    command_speed_range = config.reference.command_speed_range_mps
    if command_speed_range is not None:
        current_bound = config.current.max_speed_mps if config.current.enabled else 0.0
        if config.current.time_variation is not None:
            current_bound += config.current.time_variation.speed_amplitude_mps
        if config.current.shear is not None:
            raise ValueError(
                "PC command-speed curriculum does not yet support sheared current; "
                "use uniform current so the worst-case reachability bound is well defined"
            )
        if command_speed_range[1] + current_bound > REMUS100_MAX_SPEED_MPS + 1e-12:
            raise ValueError(
                "PC command-speed range is not reachable for the current envelope: "
                "max command speed plus max current must stay within the REMUS100 limit"
            )
    if config.domain is not None:
        if not config.current.enabled:
            raise ValueError("domain sampling requires current.enabled=true")
        if config.current.shear is not None:
            raise ValueError("domain sampling does not support sheared current")
        if command_speed_range is None:
            raise ValueError("domain sampling requires reference.command_speed_range_mps")
        variation_bound = (
            0.0
            if config.current.time_variation is None
            else config.current.time_variation.speed_amplitude_mps
        )
        for cell in config.domain.cells:
            command_low, command_high = cell.command_speed_range_mps
            current_low, current_high = cell.current_speed_range_mps
            if (
                command_low < command_speed_range[0] - 1e-12
                or command_high > command_speed_range[1] + 1e-12
            ):
                raise ValueError(
                    f"domain cell {cell.name!r} command speed must lie inside "
                    "reference.command_speed_range_mps"
                )
            if (
                current_low < config.current.min_speed_mps - 1e-12
                or current_high > config.current.max_speed_mps + 1e-12
            ):
                raise ValueError(
                    f"domain cell {cell.name!r} current speed must lie inside current.speed_mps"
                )
            if command_high + current_high + variation_bound > REMUS100_MAX_SPEED_MPS + 1e-12:
                raise ValueError(
                    f"domain cell {cell.name!r} is not reachable for the REMUS100 envelope"
                )
    direct_command = config.pc_stage in {"C0", "C1"}
    if config.pc_control_mode == "alos":
        if config.reward.mode not in {"pc_command_track", "pc_command_track_v2"}:
            raise ValueError(
                "ALOS PC requires reward.mode='pc_command_track' or "
                "'pc_command_track_v2'"
            )
        if config.reward.hold_bonus != direct_command:
            raise ValueError("PC hold_bonus must be true exactly for C0/C1 and false for C2-C6")
    else:
        if config.pc_stage not in {"C2", "C3", "C4", "C5", "C6"}:
            raise ValueError("ALOS-free PC modes require pc_stage in {'C2', 'C3', 'C4', 'C5', 'C6'}")
        if config.reward.mode != "pc_path_track":
            raise ValueError("ALOS-free PC modes require reward.mode='pc_path_track'")
        if config.reward.hold_bonus:
            raise ValueError("ALOS-free PC modes require reward.hold_bonus=false")
        if config.reward.w_path_3d > 1e-12:
            raise ValueError("reward.w_path_3d is only supported with pc_control_mode='alos'")
    pc_path_terms = {
        "w_prog": config.reward.w_prog,
        "w_3d": config.reward.w_3d,
        "w_yaw": config.reward.w_yaw,
        "w_u": config.reward.w_u,
        "w_action": config.reward.w_action,
        "w_sat": config.reward.w_sat,
        "terminal_progress_weight": config.reward.terminal_progress_weight,
        "goal_bonus": config.reward.goal_bonus,
        "timeout_penalty": config.reward.timeout_penalty,
        "out_of_bounds_penalty": config.reward.out_of_bounds_penalty,
    }
    nonzero = {name: value for name, value in pc_path_terms.items() if abs(value) > 1e-12}
    if nonzero and config.reward.mode == "pc_command_track":
        raise ValueError(
            f"PC pc_command_track mode forbids legacy path/task reward terms: {sorted(nonzero)}"
        )
    if not config.reward.hold_bonus and config.reward.w_hold != 5.0:
        # w_hold is inert when disabled; accepting a non-default value would
        # make the config look as if C2-C6 still had a hold potential.
        raise ValueError("reward.w_hold must stay at its default when hold_bonus=false")


def _validate_reward_termination_contract(config: LowLevelConfig) -> None:
    reward = config.reward
    termination = config.termination
    if reward.mode in {"pc_command_track", "pc_command_track_v2", "pc_path_track"}:
        if not reward.phi_dead_rad <= reward.phi_soft_rad < termination.phi_term_rad:
            raise ValueError("PC roll thresholds must satisfy dead <= soft < phi_term")
        if not reward.p_dead_rad_s <= reward.p_soft_rad_s < termination.p_term_rad_s:
            raise ValueError("PC roll-rate thresholds must satisfy dead <= soft < p_term")


def _validate_current_visibility(config: LowLevelConfig) -> None:
    """Reject configs that hide the true current while still leaking it elsewhere.

    Non-oracle arms zero observation indices 17-19 and must infer the
    disturbance from history. If ALOS or the PD/PI teacher receives the true
    current, the resulting reference or water-relative speed command encodes it,
    so the policy can read the current indirectly. Oracle configs opt in with
    ``observation.expose_current=true``.
    """

    if not config.current.enabled:
        return
    contract = config.observation.current_contract
    if contract == "hidden":
        if config.observation.expose_current:
            raise ValueError("hidden current contract requires observation.expose_current=false")
        if config.observation.speed_feedback != "ground":
            raise ValueError(
                "hidden current contract requires observation.speed_feedback='ground'; "
                "water-relative surge and speed error can recover the true current"
            )
    elif contract == "oracle":
        if not config.observation.expose_current:
            raise ValueError("oracle current contract requires observation.expose_current=true")
    elif config.observation.expose_current:
        raise ValueError("observation.expose_current=true requires current_contract='oracle'")
    if config.observation.expose_current:
        return
    leaks: list[str] = []
    if config.guidance.compensate_current:
        leaks.append("guidance.compensate_current")
    if config.controller.use_current_field:
        leaks.append("base_controller.use_current_field")
    if leaks:
        raise ValueError(
            "hidden current leak: observation.expose_current=false but the true current is "
            f"still used by {', '.join(leaks)}; set them to false or expose current explicitly"
        )


def load_config(path: Path | str) -> LowLevelConfig:
    source = Path(path)
    raw_text = source.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return parse_config(data, config_sha256=hashlib.sha256(canonical).hexdigest())


__all__ = [
    "ActuatorConfig",
    "ControllerConfig",
    "CurrentConfig",
    "CurrentShearConfig",
    "CurrentVariationConfig",
    "EvaluationConfig",
    "GuidanceConfig",
    "LowLevelConfig",
    "NormalizationConfig",
    "ObservationConfig",
    "PlantConfig",
    "PlantRandomizationConfig",
    "ReferenceConfig",
    "ResidualConfig",
    "RewardConfig",
    "SensorConfig",
    "SUPPORTED_ACTUATOR_ENVELOPES",
    "SUPPORTED_POLICY_MODES",
    "SUPPORTED_REWARD_MODES",
    "SUPPORTED_SENSOR_SOURCES",
    "SUPPORTED_SENSOR_PROFILES",
    "TerminationConfig",
    "load_config",
    "parse_config",
]
