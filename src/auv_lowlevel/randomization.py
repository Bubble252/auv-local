from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np
from auv_core.contracts import FloatArray, UniformCurrent

from .config import CurrentConfig, CurrentShearConfig, ReferenceConfig


@dataclass(frozen=True)
class EpisodeRandomization:
    seed: int
    current_ned_mps: FloatArray
    current_phase_rad: float
    start_jitter_ned_m: FloatArray
    start_yaw_jitter_rad: float


@dataclass(frozen=True)
class SpatialShear:
    """Linear horizontal-current shear ``G @ (position - origin)``."""

    velocity_gradient_per_s: FloatArray
    origin_ned_m: FloatArray

    def __post_init__(self) -> None:
        gradient = np.asarray(self.velocity_gradient_per_s, dtype=np.float64)
        origin = np.asarray(self.origin_ned_m, dtype=np.float64)
        if gradient.shape != (3, 3) or not np.all(np.isfinite(gradient)):
            raise ValueError(
                "velocity_gradient_per_s must be a finite shape-(3, 3) matrix"
            )
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin_ned_m must be a finite shape-(3,) vector")
        if np.any(np.abs(gradient[2]) > 1e-12):
            raise ValueError("vertical current shear is not supported")
        object.__setattr__(self, "velocity_gradient_per_s", gradient)
        object.__setattr__(self, "origin_ned_m", origin)

    def apply(
        self,
        base_velocity_ned_mps: FloatArray,
        position_ned_m: FloatArray,
    ) -> FloatArray:
        base = np.asarray(base_velocity_ned_mps, dtype=np.float64)
        position = np.asarray(position_ned_m, dtype=np.float64)
        if base.shape != (3,) or not np.all(np.isfinite(base)):
            raise ValueError("base_velocity_ned_mps must be a finite shape-(3,) vector")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_ned_m must be a finite shape-(3,) vector")
        result = base + self.velocity_gradient_per_s @ (position - self.origin_ned_m)
        result[2] = 0.0
        return result


def spatial_shear_from_config(current: CurrentConfig) -> SpatialShear | None:
    shear: CurrentShearConfig | None = current.shear
    if shear is None:
        return None
    return SpatialShear(
        velocity_gradient_per_s=shear.velocity_gradient_per_s,
        origin_ned_m=shear.origin_ned_m,
    )


@dataclass(frozen=True)
class UniformShearCurrent:
    """Spatially sheared horizontal current with time-invariant base velocity."""

    base_velocity_ned_mps: FloatArray
    shear: SpatialShear

    def __post_init__(self) -> None:
        base = np.asarray(self.base_velocity_ned_mps, dtype=np.float64)
        if base.shape != (3,) or not np.all(np.isfinite(base)):
            raise ValueError("base_velocity_ned_mps must be a finite shape-(3,) vector")
        object.__setattr__(self, "base_velocity_ned_mps", base)

    def sample(self, position_ned_m: FloatArray, t_s: float) -> FloatArray:
        if not np.isfinite(t_s) or t_s < 0.0:
            raise ValueError("t_s must be finite and non-negative")
        return self.shear.apply(self.base_velocity_ned_mps, position_ned_m)


@dataclass(frozen=True)
class SmoothTemporalCurrent:
    """Horizontal current with deterministic smooth speed/direction modulation."""

    base_velocity_ned_mps: FloatArray
    speed_amplitude_mps: float
    direction_amplitude_rad: float
    period_s: float
    phase_rad: float
    shear: SpatialShear | None = None

    def __post_init__(self) -> None:
        base = np.asarray(self.base_velocity_ned_mps, dtype=np.float64)
        if base.shape != (3,) or not np.all(np.isfinite(base)):
            raise ValueError("base_velocity_ned_mps must be a finite shape-(3,) vector")
        if not np.isfinite(self.speed_amplitude_mps) or self.speed_amplitude_mps < 0.0:
            raise ValueError("speed_amplitude_mps must be finite and non-negative")
        if not np.isfinite(self.direction_amplitude_rad) or self.direction_amplitude_rad < 0.0:
            raise ValueError("direction_amplitude_rad must be finite and non-negative")
        if not np.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ValueError("period_s must be finite and positive")
        if not np.isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite")

    def sample(self, position_ned_m: FloatArray, t_s: float) -> FloatArray:
        position = np.asarray(position_ned_m, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_ned_m must be a finite shape-(3,) vector")
        if not np.isfinite(t_s) or t_s < 0.0:
            raise ValueError("t_s must be finite and non-negative")
        base = np.asarray(self.base_velocity_ned_mps, dtype=np.float64)
        base_speed = float(hypot(float(base[0]), float(base[1])))
        base_direction = float(atan2(float(base[1]), float(base[0])))
        omega = 2.0 * pi / self.period_s
        speed = base_speed + self.speed_amplitude_mps * sin(
            omega * t_s + self.phase_rad
        )
        speed = max(speed, 0.0)
        direction = base_direction + self.direction_amplitude_rad * sin(
            omega * t_s + self.phase_rad + 0.5 * pi
        )
        velocity = np.array(
            [speed * cos(direction), speed * sin(direction), 0.0],
            dtype=np.float64,
        )
        if self.shear is not None:
            velocity = self.shear.apply(velocity, position)
        return velocity


def episode_current_field(
    randomization: EpisodeRandomization,
    current: CurrentConfig,
) -> UniformCurrent | UniformShearCurrent | SmoothTemporalCurrent:
    shear = spatial_shear_from_config(current)
    if current.time_variation is None:
        if shear is not None:
            return UniformShearCurrent(
                base_velocity_ned_mps=randomization.current_ned_mps,
                shear=shear,
            )
        return UniformCurrent(velocity_ned_mps=randomization.current_ned_mps)
    variation = current.time_variation
    return SmoothTemporalCurrent(
        base_velocity_ned_mps=randomization.current_ned_mps,
        speed_amplitude_mps=variation.speed_amplitude_mps,
        direction_amplitude_rad=variation.direction_amplitude_rad,
        period_s=variation.period_s,
        phase_rad=randomization.current_phase_rad,
        shear=shear,
    )


def sample_episode_randomization(
    seed: int,
    current: CurrentConfig,
    reference: ReferenceConfig,
) -> EpisodeRandomization:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    sequences = np.random.SeedSequence(seed).spawn(3)
    current_rng = np.random.default_rng(sequences[0])
    jitter_xy_rng = np.random.default_rng(sequences[1])
    jitter_z_yaw_rng = np.random.default_rng(sequences[2])

    if current.enabled:
        speed = float(current_rng.uniform(current.min_speed_mps, current.max_speed_mps))
        direction = float(
            current_rng.uniform(current.direction_min_rad, current.direction_max_rad)
        )
    else:
        speed = 0.0
        direction = 0.0
    current_ned = np.array(
        [speed * np.cos(direction), speed * np.sin(direction), 0.0],
        dtype=np.float64,
    )
    if current.time_variation is None:
        current_phase_rad = 0.0
    else:
        phase_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x9E3779B9])
        )
        current_phase_rad = float(phase_rng.uniform(0.0, 2.0 * np.pi))

    jitter_xy = jitter_xy_rng.uniform(
        -reference.start_jitter_xy_m,
        reference.start_jitter_xy_m,
        size=2,
    )
    jitter_z = float(
        jitter_z_yaw_rng.uniform(
            -reference.start_jitter_z_m,
            reference.start_jitter_z_m,
        )
    )
    yaw_jitter = float(
        jitter_z_yaw_rng.uniform(
            -reference.start_jitter_yaw_rad,
            reference.start_jitter_yaw_rad,
        )
    )
    return EpisodeRandomization(
        seed=seed,
        current_ned_mps=current_ned,
        current_phase_rad=current_phase_rad,
        start_jitter_ned_m=np.array([jitter_xy[0], jitter_xy[1], jitter_z]),
        start_yaw_jitter_rad=yaw_jitter,
    )


def evaluation_seeds(start: int, count: int) -> tuple[int, ...]:
    if start < 0 or count <= 0:
        raise ValueError("evaluation seed range must be non-empty and non-negative")
    return tuple(range(start, start + count))


def training_seeds(count: int, start: int, excluded: set[int] | None = None) -> tuple[int, ...]:
    if count <= 0 or start < 0:
        raise ValueError("training seed range must be non-empty and non-negative")
    blocked = set() if excluded is None else set(excluded)
    result: list[int] = []
    candidate = start
    while len(result) < count:
        if candidate not in blocked:
            result.append(candidate)
        candidate += 1
    return tuple(result)


__all__ = [
    "EpisodeRandomization",
    "SmoothTemporalCurrent",
    "SpatialShear",
    "UniformShearCurrent",
    "episode_current_field",
    "evaluation_seeds",
    "sample_episode_randomization",
    "spatial_shear_from_config",
    "training_seeds",
]
