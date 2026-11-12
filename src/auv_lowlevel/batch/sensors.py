"""Deterministic synthetic sensor chain for the PC arm.

Truth is consumed only inside this module. The returned actor-facing state is
built from delayed, noisy, biased, scale-corrupted and dropout-held sensor
values; no true current or true position is returned in the actor API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from auv_lowlevel.config import LowLevelConfig

from .geometry import quaternion_from_euler, world_to_body, wrap_angle


@dataclass(frozen=True)
class SensorProfile:
    name: str
    dvl_bottom_hz: tuple[float, float]
    dvl_bottom_delay_s: tuple[float, float]
    dvl_bottom_noise_mps: tuple[float, float]
    dvl_bottom_bias_mps: tuple[float, float]
    dvl_bottom_scale: tuple[float, float]
    dvl_bottom_outage_start_prob: tuple[float, float]
    dvl_bottom_outage_duration_s: tuple[float, float]
    dvl_water_hz: tuple[float, float]
    dvl_water_delay_s: tuple[float, float]
    dvl_water_noise_mps: tuple[float, float]
    dvl_water_bias_mps: tuple[float, float]
    dvl_water_scale: tuple[float, float]
    dvl_water_outage_start_prob: tuple[float, float]
    dvl_water_outage_duration_s: tuple[float, float]
    imu_att_hz: tuple[float, float]
    imu_att_delay_s: tuple[float, float]
    imu_rollpitch_noise_rad: tuple[float, float]
    imu_heading_noise_rad: tuple[float, float]
    imu_rollpitch_bias_rad: tuple[float, float]
    imu_heading_bias_rad: tuple[float, float]
    imu_drift_rad_s: tuple[float, float]
    imu_mag_declination_rad: tuple[float, float]
    imu_gyro_hz: tuple[float, float]
    imu_gyro_delay_s: tuple[float, float]
    imu_gyro_noise_rad_s: tuple[float, float]
    imu_gyro_bias_rad_s: tuple[float, float]
    imu_gyro_scale: tuple[float, float]
    usbl_hz: tuple[float, float]
    usbl_delay_s: tuple[float, float]
    usbl_noise_m: tuple[float, float]
    usbl_outlier_prob: tuple[float, float]
    usbl_outlier_jump_m: tuple[float, float]
    usbl_reject_outliers: bool
    current_est_hz: tuple[float, float]
    current_est_delay_s: tuple[float, float]
    current_est_noise_mps: tuple[float, float]
    current_est_bias_mps: tuple[float, float]
    current_est_walk_mps: tuple[float, float]
    current_est_tau_s: tuple[float, float]
    current_est_outage_start_prob: tuple[float, float]
    current_est_outage_duration_s: tuple[float, float]
    nav_bias_m: tuple[float, float]
    nav_drift_sigma_m: tuple[float, float]
    nav_drift_tau_s: tuple[float, float]
    nav_usbl_gain: float
    nav_dvl_outage_drift_mps: tuple[float, float]
    nav_delay_s: tuple[float, float]
    nav_noise_m: tuple[float, float] = (0.0, 0.0)
    nav_bias_calibration_fraction: float = 0.0


def _zero_range(value: float = 0.0) -> tuple[float, float]:
    return (value, value)


PROFILES = {
    "clean": SensorProfile(
        name="clean",
        dvl_bottom_hz=_zero_range(50.0),
        dvl_bottom_delay_s=_zero_range(),
        dvl_bottom_noise_mps=_zero_range(),
        dvl_bottom_bias_mps=_zero_range(),
        dvl_bottom_scale=_zero_range(1.0),
        dvl_bottom_outage_start_prob=_zero_range(),
        dvl_bottom_outage_duration_s=_zero_range(0.0),
        dvl_water_hz=_zero_range(50.0),
        dvl_water_delay_s=_zero_range(),
        dvl_water_noise_mps=_zero_range(),
        dvl_water_bias_mps=_zero_range(),
        dvl_water_scale=_zero_range(1.0),
        dvl_water_outage_start_prob=_zero_range(),
        dvl_water_outage_duration_s=_zero_range(0.0),
        imu_att_hz=_zero_range(50.0),
        imu_att_delay_s=_zero_range(),
        imu_rollpitch_noise_rad=_zero_range(),
        imu_heading_noise_rad=_zero_range(),
        imu_rollpitch_bias_rad=_zero_range(),
        imu_heading_bias_rad=_zero_range(),
        imu_drift_rad_s=_zero_range(),
        imu_mag_declination_rad=_zero_range(),
        imu_gyro_hz=_zero_range(50.0),
        imu_gyro_delay_s=_zero_range(),
        imu_gyro_noise_rad_s=_zero_range(),
        imu_gyro_bias_rad_s=_zero_range(),
        imu_gyro_scale=_zero_range(1.0),
        usbl_hz=_zero_range(50.0),
        usbl_delay_s=_zero_range(),
        usbl_noise_m=_zero_range(),
        usbl_outlier_prob=_zero_range(),
        usbl_outlier_jump_m=_zero_range(),
        usbl_reject_outliers=True,
        current_est_hz=_zero_range(50.0),
        current_est_delay_s=_zero_range(),
        current_est_noise_mps=_zero_range(),
        current_est_bias_mps=_zero_range(),
        current_est_walk_mps=_zero_range(),
        current_est_tau_s=_zero_range(0.0),
        current_est_outage_start_prob=_zero_range(),
        current_est_outage_duration_s=_zero_range(0.0),
        nav_bias_m=_zero_range(),
        nav_drift_sigma_m=_zero_range(),
        nav_drift_tau_s=_zero_range(10.0),
        nav_usbl_gain=0.0,
        nav_dvl_outage_drift_mps=_zero_range(),
        nav_delay_s=_zero_range(),
    ),
    "mild": SensorProfile(
        name="mild",
        dvl_bottom_hz=(5.0, 10.0),
        dvl_bottom_delay_s=(0.0, 0.05),
        dvl_bottom_noise_mps=(0.002, 0.008),
        dvl_bottom_bias_mps=(0.001, 0.005),
        dvl_bottom_scale=(0.99, 1.01),
        dvl_bottom_outage_start_prob=(0.0, 0.002),
        dvl_bottom_outage_duration_s=(1.0, 2.0),
        dvl_water_hz=(5.0, 10.0),
        dvl_water_delay_s=(0.0, 0.05),
        dvl_water_noise_mps=(0.005, 0.015),
        dvl_water_bias_mps=(0.005, 0.015),
        dvl_water_scale=(0.99, 1.01),
        dvl_water_outage_start_prob=(0.0, 0.005),
        dvl_water_outage_duration_s=(1.0, 3.0),
        imu_att_hz=_zero_range(50.0),
        imu_att_delay_s=(0.0, 0.01),
        imu_rollpitch_noise_rad=(math.radians(0.1), math.radians(0.3)),
        imu_heading_noise_rad=(math.radians(0.5), math.radians(1.0)),
        imu_rollpitch_bias_rad=(math.radians(0.05), math.radians(0.2)),
        imu_heading_bias_rad=(math.radians(0.1), math.radians(0.5)),
        imu_drift_rad_s=(math.radians(0.05) / 60.0, math.radians(0.2) / 60.0),
        imu_mag_declination_rad=(math.radians(-1.0), math.radians(1.0)),
        imu_gyro_hz=_zero_range(50.0),
        imu_gyro_delay_s=(0.0, 0.01),
        imu_gyro_noise_rad_s=(0.0005, 0.002),
        imu_gyro_bias_rad_s=(0.0005, 0.001),
        imu_gyro_scale=(0.99, 1.01),
        usbl_hz=(0.5, 1.0),
        usbl_delay_s=(0.0, 0.2),
        usbl_noise_m=(0.5, 1.5),
        usbl_outlier_prob=(0.0, 0.01),
        usbl_outlier_jump_m=(5.0, 10.0),
        usbl_reject_outliers=True,
        current_est_hz=(0.5, 2.0),
        current_est_delay_s=(0.0, 0.3),
        current_est_noise_mps=(0.005, 0.02),
        current_est_bias_mps=(0.005, 0.015),
        current_est_walk_mps=(0.0005, 0.002),
        current_est_tau_s=(1.0, 5.0),
        current_est_outage_start_prob=(0.0, 0.01),
        current_est_outage_duration_s=(1.0, 3.0),
        nav_bias_m=(0.1, 0.5),
        nav_drift_sigma_m=(0.05, 0.2),
        nav_drift_tau_s=(5.0, 20.0),
        nav_usbl_gain=0.25,
        nav_dvl_outage_drift_mps=(0.001, 0.01),
        nav_delay_s=(0.0, 0.1),
    ),
    "realistic": SensorProfile(
        name="realistic",
        dvl_bottom_hz=(1.0, 10.0),
        dvl_bottom_delay_s=(0.0, 0.2),
        dvl_bottom_noise_mps=(0.005, 0.03),
        dvl_bottom_bias_mps=(0.005, 0.02),
        dvl_bottom_scale=(0.97, 1.03),
        dvl_bottom_outage_start_prob=(0.002, 0.02),
        dvl_bottom_outage_duration_s=(1.0, 10.0),
        dvl_water_hz=(1.0, 5.0),
        dvl_water_delay_s=(0.0, 0.3),
        dvl_water_noise_mps=(0.01, 0.05),
        dvl_water_bias_mps=(0.01, 0.05),
        dvl_water_scale=(0.97, 1.03),
        dvl_water_outage_start_prob=(0.002, 0.03),
        dvl_water_outage_duration_s=(1.0, 10.0),
        imu_att_hz=_zero_range(50.0),
        imu_att_delay_s=(0.0, 0.05),
        imu_rollpitch_noise_rad=(math.radians(0.1), math.radians(1.0)),
        imu_heading_noise_rad=(math.radians(0.5), math.radians(3.0)),
        imu_rollpitch_bias_rad=(math.radians(0.05), math.radians(0.4)),
        imu_heading_bias_rad=(math.radians(0.1), math.radians(1.0)),
        imu_drift_rad_s=(math.radians(0.1) / 60.0, math.radians(1.0) / 60.0),
        imu_mag_declination_rad=(math.radians(-5.0), math.radians(5.0)),
        imu_gyro_hz=_zero_range(50.0),
        imu_gyro_delay_s=(0.0, 0.05),
        imu_gyro_noise_rad_s=(0.001, 0.01),
        imu_gyro_bias_rad_s=(0.001, 0.005),
        imu_gyro_scale=(0.95, 1.05),
        usbl_hz=(0.2, 1.0),
        usbl_delay_s=(0.1, 0.5),
        usbl_noise_m=(0.5, 5.0),
        usbl_outlier_prob=(0.01, 0.03),
        usbl_outlier_jump_m=(10.0, 25.0),
        usbl_reject_outliers=True,
        current_est_hz=(0.1, 5.0),
        current_est_delay_s=(0.0, 2.0),
        current_est_noise_mps=(0.01, 0.05),
        current_est_bias_mps=(0.01, 0.03),
        current_est_walk_mps=(0.001, 0.005),
        current_est_tau_s=(1.0, 10.0),
        current_est_outage_start_prob=(0.002, 0.03),
        current_est_outage_duration_s=(1.0, 10.0),
        nav_bias_m=(0.2, 2.0),
        nav_drift_sigma_m=(0.1, 0.5),
        nav_drift_tau_s=(10.0, 60.0),
        nav_usbl_gain=0.35,
        nav_dvl_outage_drift_mps=(0.005, 0.03),
        nav_delay_s=(0.05, 0.5),
    ),
    "strong": SensorProfile(
        name="strong",
        dvl_bottom_hz=(1.0, 5.0),
        dvl_bottom_delay_s=(0.1, 0.3),
        dvl_bottom_noise_mps=(0.01, 0.05),
        dvl_bottom_bias_mps=(0.01, 0.04),
        dvl_bottom_scale=(0.95, 1.05),
        dvl_bottom_outage_start_prob=(0.01, 0.05),
        dvl_bottom_outage_duration_s=(5.0, 20.0),
        dvl_water_hz=(1.0, 3.0),
        dvl_water_delay_s=(0.1, 0.4),
        dvl_water_noise_mps=(0.02, 0.08),
        dvl_water_bias_mps=(0.02, 0.08),
        dvl_water_scale=(0.95, 1.05),
        dvl_water_outage_start_prob=(0.01, 0.05),
        dvl_water_outage_duration_s=(5.0, 20.0),
        imu_att_hz=_zero_range(50.0),
        imu_att_delay_s=(0.01, 0.08),
        imu_rollpitch_noise_rad=(math.radians(0.3), math.radians(1.5)),
        imu_heading_noise_rad=(math.radians(1.0), math.radians(5.0)),
        imu_rollpitch_bias_rad=(math.radians(0.1), math.radians(0.8)),
        imu_heading_bias_rad=(math.radians(0.5), math.radians(2.0)),
        imu_drift_rad_s=(math.radians(0.2) / 60.0, math.radians(2.0) / 60.0),
        imu_mag_declination_rad=(math.radians(-8.0), math.radians(8.0)),
        imu_gyro_hz=_zero_range(50.0),
        imu_gyro_delay_s=(0.01, 0.08),
        imu_gyro_noise_rad_s=(0.002, 0.02),
        imu_gyro_bias_rad_s=(0.002, 0.01),
        imu_gyro_scale=(0.9, 1.1),
        usbl_hz=(0.1, 0.5),
        usbl_delay_s=(0.5, 1.0),
        usbl_noise_m=(1.0, 8.0),
        usbl_outlier_prob=(0.03, 0.05),
        usbl_outlier_jump_m=(15.0, 40.0),
        usbl_reject_outliers=False,
        current_est_hz=(0.1, 2.0),
        current_est_delay_s=(0.5, 3.0),
        current_est_noise_mps=(0.02, 0.08),
        current_est_bias_mps=(0.02, 0.06),
        current_est_walk_mps=(0.002, 0.01),
        current_est_tau_s=(5.0, 20.0),
        current_est_outage_start_prob=(0.01, 0.05),
        current_est_outage_duration_s=(5.0, 20.0),
        nav_bias_m=(2.0, 5.0),
        nav_drift_sigma_m=(0.5, 1.5),
        nav_drift_tau_s=(20.0, 120.0),
        nav_usbl_gain=0.2,
        nav_dvl_outage_drift_mps=(0.01, 0.05),
        nav_delay_s=(0.2, 1.0),
    ),
}

# Goal85 contract: truth-eta scoring with zero-mean navigation noise and no
# sensor latency.  System-level navigation terms (bias, drift, calibration,
# USBL innovation gain and outage drift) are deliberately disabled; only
# per-control-step zero-mean measurements remain.  Actuator lag in the plant is
# physical dynamics and stays untouched.
PROFILES["unbiased_noise_v1"] = replace(
    PROFILES["clean"],
    name="unbiased_noise_v1",
    dvl_bottom_noise_mps=(0.005, 0.02),
    dvl_water_noise_mps=(0.005, 0.02),
    imu_rollpitch_noise_rad=(0.002, 0.01),
    imu_heading_noise_rad=(0.003, 0.015),
    imu_gyro_noise_rad_s=(0.001, 0.005),
    usbl_noise_m=(0.1, 0.5),
    current_est_noise_mps=(0.005, 0.02),
    current_est_tau_s=(0.5, 1.5),
    nav_noise_m=(0.05, 0.20),
)


# Keep the pre-2026-11-07 calibrated distribution available for historical
# reproduction, and expose the old realistic ranges as an explicit stress arm.
# The unqualified realistic_calibrated profile is now the nominal field model.
PROFILES["realistic_calibrated_legacy"] = replace(
    PROFILES["realistic"],
    name="realistic_calibrated_legacy",
    nav_usbl_gain=0.03,
    nav_bias_calibration_fraction=0.8,
)
PROFILES["realistic_stress"] = replace(
    PROFILES["realistic"],
    name="realistic_stress",
)
PROFILES["realistic_calibrated"] = replace(
    PROFILES["realistic_calibrated_legacy"],
    name="realistic_calibrated",
    current_est_tau_s=(1.5, 3.0),
    current_est_delay_s=(0.0, 0.3),
    current_est_hz=(1.0, 3.0),
    current_est_noise_mps=(0.005, 0.02),
    current_est_bias_mps=(0.005, 0.02),
    current_est_outage_duration_s=(1.0, 3.0),
    nav_delay_s=(0.02, 0.15),
    nav_bias_m=(0.1, 0.5),
    usbl_noise_m=(0.3, 1.5),
    usbl_delay_s=(0.1, 0.3),
    dvl_bottom_outage_duration_s=(1.0, 3.0),
    usbl_outlier_jump_m=(5.0, 10.0),
    usbl_outlier_prob=(0.005, 0.01),
)


@dataclass(frozen=True)
class SyntheticSensorOutput:
    p_nav: Tensor
    attitude: Tensor
    angular_rate: Tensor
    dvl_ground_body: Tensor
    dvl_water_body: Tensor
    current_estimate_body: Tensor
    usbl_position: Tensor
    bottom_lock_valid: Tensor
    water_track_valid: Tensor
    current_estimate_valid: Tensor
    usbl_valid: Tensor
    usbl_outlier: Tensor
    nav_age_s: Tensor
    dvl_bottom_age_s: Tensor
    dvl_water_age_s: Tensor
    usbl_age_s: Tensor
    current_estimate_age_s: Tensor
    nav_ou: Tensor
    nav_event: Tensor
    nav_event_usbl: Tensor
    nav_event_dvl: Tensor
    nav_bias: Tensor
    usbl_due: Tensor
    usbl_accepted: Tensor
    usbl_rejected: Tensor
    usbl_innovation: Tensor
    usbl_innovation_applied: Tensor
    usbl_raw: Tensor
    usbl_predicted: Tensor
    dvl_bottom_outage: Tensor
    nav_dvl_outage_delta: Tensor

    def actor_state(self, speed_feedback: str = "ground") -> Tensor:
        if speed_feedback == "ground":
            velocity = self.dvl_ground_body
        elif speed_feedback == "water_relative":
            velocity = self.dvl_water_body
        else:
            raise ValueError(f"unsupported speed_feedback={speed_feedback!r}")
        return torch.cat(
            [self.p_nav, self.attitude, self.angular_rate, velocity],
            dim=1,
        )


class _VectorDelayQueue:
    def __init__(self, capacity: int, dim: int, *, device: torch.device, dtype: torch.dtype):
        if capacity < 1:
            raise ValueError("delay queue capacity must be positive")
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = device
        self.dtype = dtype
        self.buffer: Tensor | None = None
        self.cursor = 0

    def reset(self, count: int, initial: Tensor | None = None) -> None:
        if initial is None:
            self.buffer = torch.zeros(
                (count, self.capacity, self.dim), device=self.device, dtype=self.dtype
            )
        else:
            value = torch.as_tensor(initial, device=self.device, dtype=self.dtype)
            if value.shape != (count, self.dim):
                raise ValueError("initial delay value has the wrong shape")
            self.buffer = value[:, None, :].expand(-1, self.capacity, -1).clone()
        self.cursor = 0

    def push(self, value: Tensor) -> None:
        if self.buffer is None:
            raise RuntimeError("delay queue must be reset before push")
        value_t = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if value_t.shape != (self.buffer.shape[0], self.dim):
            raise ValueError("delay queue value has the wrong shape")
        self.buffer[:, self.cursor, :] = value_t
        self.cursor = (self.cursor + 1) % self.capacity

    def read(self, delay_steps: Tensor) -> Tensor:
        if self.buffer is None:
            raise RuntimeError("delay queue must be reset before read")
        delay = torch.as_tensor(delay_steps, device=self.device, dtype=torch.long)
        if delay.shape != (self.buffer.shape[0],):
            raise ValueError("delay steps have the wrong shape")
        if bool((delay < 0).any()) or bool((delay >= self.capacity).any()):
            raise ValueError("delay steps must lie inside the queue capacity")
        indices = (self.cursor - 1 - delay) % self.capacity
        return self.buffer[
            torch.arange(self.buffer.shape[0], device=self.device), indices
        ]


@dataclass(frozen=True)
class _EpisodeParameters:
    dvl_bottom_rate_hz: Tensor
    dvl_bottom_delay_steps: Tensor
    dvl_bottom_noise: Tensor
    dvl_bottom_bias: Tensor
    dvl_bottom_scale: Tensor
    dvl_bottom_outage_start_prob: Tensor
    dvl_bottom_outage_duration_s: Tensor
    dvl_water_rate_hz: Tensor
    dvl_water_delay_steps: Tensor
    dvl_water_noise: Tensor
    dvl_water_bias: Tensor
    dvl_water_scale: Tensor
    dvl_water_outage_start_prob: Tensor
    dvl_water_outage_duration_s: Tensor
    imu_att_rate_hz: Tensor
    imu_att_delay_steps: Tensor
    imu_rollpitch_noise: Tensor
    imu_heading_noise: Tensor
    imu_rollpitch_bias: Tensor
    imu_heading_bias: Tensor
    imu_mag_declination: Tensor
    imu_drift_sigma: Tensor
    imu_gyro_rate_hz: Tensor
    imu_gyro_delay_steps: Tensor
    imu_gyro_noise: Tensor
    imu_gyro_bias: Tensor
    imu_gyro_scale: Tensor
    usbl_rate_hz: Tensor
    usbl_delay_steps: Tensor
    usbl_noise: Tensor
    usbl_outlier_prob: Tensor
    usbl_outlier_jump: Tensor
    current_est_rate_hz: Tensor
    current_est_delay_steps: Tensor
    current_est_noise: Tensor
    current_est_bias: Tensor
    current_est_walk: Tensor
    current_est_tau_s: Tensor
    current_est_outage_start_prob: Tensor
    current_est_outage_duration_s: Tensor
    nav_bias: Tensor
    nav_drift_sigma: Tensor
    nav_drift_tau_s: Tensor
    nav_dvl_outage_drift: Tensor
    nav_delay_steps: Tensor
    nav_noise: Tensor
    noise_scale: Tensor


class SyntheticSensorSuite:
    """Vectorized synthetic navigation, IMU, DVL and current-estimate chain."""

    def __init__(
        self,
        config: LowLevelConfig,
        *,
        num_envs: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
    ) -> None:
        if config.sensors is None:
            raise ValueError("SyntheticSensorSuite requires config.sensors")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.config = config
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.profile = PROFILES[config.sensors.profile]
        self.base_seed = int(config.sensors.seed if seed is None else seed)
        self._generator = torch.Generator(device="cpu")
        self._noise_generator = torch.Generator(device=self.device)
        self._runtime_seed_offset = 0
        self._rng_count = self.num_envs
        self._params: _EpisodeParameters | None = None
        self._t_s: Tensor | None = None
        self._next = torch.zeros((self.num_envs, 9), device=self.device, dtype=self.dtype)
        self._last_update = torch.zeros((self.num_envs, 9), device=self.device, dtype=self.dtype)
        self._dvl_bottom_outage_s = torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
        self._dvl_water_outage_s = torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
        self._current_outage_s = torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
        self._current_lpf: Tensor | None = None
        self._current_walk: Tensor | None = None
        self._attitude_drift = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._nav_ou = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._nav_event = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._nav_event_usbl = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=self.dtype
        )
        self._nav_event_dvl = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=self.dtype
        )
        self._nav_last = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._histories_primed = False
        self._dvl_bottom_last = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._dvl_water_last = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._attitude_last = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._gyro_last = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._usbl_last = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._queues = {
            "nav": _VectorDelayQueue(self._max_delay_steps(self.profile.nav_delay_s), 3, device=self.device, dtype=dtype),
            "attitude": _VectorDelayQueue(self._max_delay_steps(self.profile.imu_att_delay_s), 3, device=self.device, dtype=dtype),
            "gyro": _VectorDelayQueue(self._max_delay_steps(self.profile.imu_gyro_delay_s), 3, device=self.device, dtype=dtype),
            "bottom": _VectorDelayQueue(self._max_delay_steps(self.profile.dvl_bottom_delay_s), 3, device=self.device, dtype=dtype),
            "water": _VectorDelayQueue(self._max_delay_steps(self.profile.dvl_water_delay_s), 3, device=self.device, dtype=dtype),
            "usbl": _VectorDelayQueue(self._max_delay_steps(self.profile.usbl_delay_s), 3, device=self.device, dtype=dtype),
            "current": _VectorDelayQueue(self._max_delay_steps(self.profile.current_est_delay_s), 3, device=self.device, dtype=dtype),
        }

    def _max_delay_steps(self, delay_range: tuple[float, float]) -> int:
        return max(1, int(math.ceil(max(delay_range) / self.config.plant.dt_s)) + 1)

    def _uniform(self, values: tuple[float, float], count: int | None = None) -> Tensor:
        count = self._rng_count if count is None else count
        low, high = values
        if high < low:
            raise ValueError("sensor range upper bound must not be below lower bound")
        if high == low:
            sample = torch.full((count,), low)
        else:
            sample = low + (high - low) * torch.rand((count,), generator=self._generator)
        return sample.to(device=self.device, dtype=self.dtype)

    def _rand(self, shape: tuple[int, ...]) -> Tensor:
        return torch.rand(
            shape,
            generator=self._noise_generator,
            device=self.device,
            dtype=self.dtype,
        )

    def _randn(self, shape: tuple[int, ...]) -> Tensor:
        return torch.randn(
            shape,
            generator=self._noise_generator,
            device=self.device,
            dtype=self.dtype,
        )

    def _vector(self, values: tuple[float, float], count: int | None = None) -> Tensor:
        return self._uniform(values, count).unsqueeze(1).expand(-1, 3).clone()

    def _delay_steps(self, delay_range: tuple[float, float]) -> Tensor:
        delay_s = self._uniform(delay_range)
        return torch.round(delay_s / self.config.plant.dt_s).to(torch.long)

    def _sample_noise_scale(self, count: int) -> Tensor:
        """Sample one multiplicative noise scale per episode.

        The fixed default is deliberately allocation/RNG free so historical
        configs preserve their exact sensor-noise sequence.
        """

        sensor_config = self.config.sensors
        if sensor_config is None:
            raise RuntimeError("sensor config disappeared after suite construction")
        scale_range = sensor_config.noise_scale_range
        if scale_range is None:
            return torch.full(
                (count,),
                sensor_config.noise_scale,
                device=self.device,
                dtype=self.dtype,
            )
        lower, upper = scale_range
        if lower == upper:
            return torch.full((count,), lower, device=self.device, dtype=self.dtype)
        return self._uniform(scale_range)

    def _sample_parameters(
        self,
        seeds: list[int] | tuple[int, ...] | None = None,
        *,
        count: int | None = None,
    ) -> _EpisodeParameters:
        sample_count = self.num_envs if count is None else int(count)
        if sample_count <= 0:
            raise ValueError("parameter sample count must be positive")
        self._rng_count = sample_count
        seed_offset = 0 if seeds is None else sum(int(seed) * (index + 1) for index, seed in enumerate(seeds))
        self._runtime_seed_offset = seed_offset
        self._generator.manual_seed((self.base_seed + seed_offset) % (2**63 - 1))
        p = self.profile
        parameters = _EpisodeParameters(
            dvl_bottom_rate_hz=self._uniform(p.dvl_bottom_hz),
            dvl_bottom_delay_steps=self._delay_steps(p.dvl_bottom_delay_s),
            dvl_bottom_noise=self._uniform(p.dvl_bottom_noise_mps),
            dvl_bottom_bias=self._vector(p.dvl_bottom_bias_mps),
            dvl_bottom_scale=self._vector(p.dvl_bottom_scale),
            dvl_bottom_outage_start_prob=self._uniform(p.dvl_bottom_outage_start_prob),
            dvl_bottom_outage_duration_s=self._uniform(p.dvl_bottom_outage_duration_s),
            dvl_water_rate_hz=self._uniform(p.dvl_water_hz),
            dvl_water_delay_steps=self._delay_steps(p.dvl_water_delay_s),
            dvl_water_noise=self._uniform(p.dvl_water_noise_mps),
            dvl_water_bias=self._vector(p.dvl_water_bias_mps),
            dvl_water_scale=self._vector(p.dvl_water_scale),
            dvl_water_outage_start_prob=self._uniform(p.dvl_water_outage_start_prob),
            dvl_water_outage_duration_s=self._uniform(p.dvl_water_outage_duration_s),
            imu_att_rate_hz=self._uniform(p.imu_att_hz),
            imu_att_delay_steps=self._delay_steps(p.imu_att_delay_s),
            imu_rollpitch_noise=self._uniform(p.imu_rollpitch_noise_rad),
            imu_heading_noise=self._uniform(p.imu_heading_noise_rad),
            imu_rollpitch_bias=self._uniform(p.imu_rollpitch_bias_rad),
            imu_heading_bias=self._uniform(p.imu_heading_bias_rad),
            imu_mag_declination=self._uniform(p.imu_mag_declination_rad),
            imu_drift_sigma=self._uniform(p.imu_drift_rad_s),
            imu_gyro_rate_hz=self._uniform(p.imu_gyro_hz),
            imu_gyro_delay_steps=self._delay_steps(p.imu_gyro_delay_s),
            imu_gyro_noise=self._uniform(p.imu_gyro_noise_rad_s),
            imu_gyro_bias=self._vector(p.imu_gyro_bias_rad_s),
            imu_gyro_scale=self._vector(p.imu_gyro_scale),
            usbl_rate_hz=self._uniform(p.usbl_hz),
            usbl_delay_steps=self._delay_steps(p.usbl_delay_s),
            usbl_noise=self._uniform(p.usbl_noise_m),
            usbl_outlier_prob=self._uniform(p.usbl_outlier_prob),
            usbl_outlier_jump=self._uniform(p.usbl_outlier_jump_m),
            current_est_rate_hz=self._uniform(p.current_est_hz),
            current_est_delay_steps=self._delay_steps(p.current_est_delay_s),
            current_est_noise=self._uniform(p.current_est_noise_mps),
            current_est_bias=self._vector(p.current_est_bias_mps),
            current_est_walk=self._uniform(p.current_est_walk_mps),
            current_est_tau_s=self._uniform(p.current_est_tau_s),
            current_est_outage_start_prob=self._uniform(p.current_est_outage_start_prob),
            current_est_outage_duration_s=self._uniform(p.current_est_outage_duration_s),
            nav_bias=self._vector(p.nav_bias_m),
            nav_drift_sigma=self._uniform(p.nav_drift_sigma_m),
            nav_drift_tau_s=self._uniform(p.nav_drift_tau_s),
            nav_dvl_outage_drift=self._vector(p.nav_dvl_outage_drift_mps),
            nav_delay_steps=self._delay_steps(p.nav_delay_s),
            nav_noise=self._vector(p.nav_noise_m),
            noise_scale=self._sample_noise_scale(sample_count),
        )
        return parameters

    def _merge_reset_parameters(self, mask: Tensor, seeds: list[int]) -> None:
        if self._params is None:
            raise RuntimeError("sensors must be reset before reset_rows")
        count = int(mask.sum().item())
        if count == 0:
            return
        sampled = self._sample_parameters(seeds, count=count)
        merged: dict[str, Tensor] = {}
        for field in fields(_EpisodeParameters):
            name = field.name
            current = getattr(self._params, name)
            replacement = getattr(sampled, name)
            updated = current.clone()
            updated[mask] = replacement
            merged[name] = updated
        self._params = _EpisodeParameters(**merged)
        self._rng_count = self.num_envs

    def reset(self, seeds: list[int] | tuple[int, ...] | None = None) -> None:
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(f"seeds must contain exactly {self.num_envs} values")
        self._params = self._sample_parameters(seeds)
        self._noise_generator.manual_seed(
            (self.base_seed + 0x9E3779B9 + self._runtime_seed_offset) % (2**63 - 1)
        )
        self._t_s = torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
        self._next.zero_()
        self._last_update.zero_()
        self._dvl_bottom_outage_s.zero_()
        self._dvl_water_outage_s.zero_()
        self._current_outage_s.zero_()
        self._attitude_drift.zero_()
        self._nav_ou.zero_()
        self._nav_event.zero_()
        self._nav_event_usbl.zero_()
        self._nav_event_dvl.zero_()
        if self._params is not None:
            self._nav_event = (
                -self.profile.nav_bias_calibration_fraction * self._params.nav_bias
            )
        self._nav_last.zero_()
        self._histories_primed = False
        self._dvl_bottom_last.zero_()
        self._dvl_water_last.zero_()
        self._attitude_last.zero_()
        self._gyro_last.zero_()
        self._usbl_last.zero_()
        self._current_lpf = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        self._current_walk = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        for queue in self._queues.values():
            queue.reset(self.num_envs)

    def reset_rows(self, mask: Tensor, seeds: list[int]) -> None:
        """Reset selected sensor rows without touching the other environments."""

        if self._params is None or self._t_s is None:
            raise RuntimeError("sensors must be reset before reset_rows")
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if mask_t.shape != (self.num_envs,):
            raise ValueError("reset mask has the wrong shape")
        if int(mask_t.sum().item()) != len(seeds):
            raise ValueError("reset mask and seed count differ")
        if not bool(mask_t.any()):
            return
        self._merge_reset_parameters(mask_t, [int(seed) for seed in seeds])
        state_tensors = (
            self._t_s,
            self._next,
            self._last_update,
            self._dvl_bottom_outage_s,
            self._dvl_water_outage_s,
            self._current_outage_s,
            self._attitude_drift,
            self._nav_ou,
            self._nav_event,
            self._nav_event_usbl,
            self._nav_event_dvl,
            self._nav_last,
            self._dvl_bottom_last,
            self._dvl_water_last,
            self._attitude_last,
            self._gyro_last,
            self._usbl_last,
        )
        for tensor in state_tensors:
            view = (self.num_envs,) + (1,) * (tensor.ndim - 1)
            tensor.copy_(torch.where(mask_t.view(view), torch.zeros_like(tensor), tensor))
        assert self._params is not None
        self._nav_event[mask_t] = (
            -self.profile.nav_bias_calibration_fraction * self._params.nav_bias[mask_t]
        )
        assert self._current_lpf is not None and self._current_walk is not None
        self._current_lpf[mask_t] = 0.0
        self._current_walk[mask_t] = 0.0
        for queue in self._queues.values():
            if queue.buffer is None:
                raise RuntimeError("delay queue must be initialized")
            queue.buffer[mask_t] = 0.0

    def set_noise_scale_rows(self, mask: Tensor, values: Tensor) -> None:
        """Override the per-episode zero-mean noise multiplier on selected rows."""

        if self._params is None:
            raise RuntimeError("sensors must be reset before overriding noise scale")
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if mask_t.shape != (self.num_envs,):
            raise ValueError("noise-scale mask has the wrong shape")
        count = int(mask_t.sum().item())
        if count == 0:
            return
        scales = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        if scales.shape == (self.num_envs,):
            scales = scales[mask_t]
        if scales.shape != (count,):
            raise ValueError("noise scales must match the selected rows")
        if not bool(torch.isfinite(scales).all()) or bool((scales < 0.0).any()):
            raise ValueError("noise scales must be finite and non-negative")
        updated = self._params.noise_scale.clone()
        updated[mask_t] = scales
        self._params = replace(self._params, noise_scale=updated)

    def prime(self, *, eta_true: Tensor, nu_true: Tensor, current_true_ned_mps: Tensor) -> SyntheticSensorOutput:
        """Measure the initial state at simulation time zero without advancing it."""

        output = self.step(
            eta_true=eta_true,
            nu_true=nu_true,
            current_true_ned_mps=current_true_ned_mps,
        )
        assert self._t_s is not None
        self._t_s = self._t_s - self.config.plant.dt_s
        return output

    def prime_rows(
        self,
        mask: Tensor,
        *,
        eta_true: Tensor,
        nu_true: Tensor,
        current_true_ned_mps: Tensor,
        previous: SyntheticSensorOutput,
    ) -> SyntheticSensorOutput:
        """Prime only reset rows while preserving every other sensor row."""

        if not self._queues or self._t_s is None:
            raise RuntimeError("sensors must be reset before prime_rows")
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if mask_t.shape != (self.num_envs,):
            raise ValueError("prime mask has the wrong shape")
        if not bool(mask_t.any()):
            return previous

        state_tensors = (
            self._t_s,
            self._next,
            self._last_update,
            self._dvl_bottom_outage_s,
            self._dvl_water_outage_s,
            self._current_outage_s,
            self._attitude_drift,
            self._nav_ou,
            self._nav_event,
            self._nav_event_usbl,
            self._nav_event_dvl,
            self._nav_last,
            self._dvl_bottom_last,
            self._dvl_water_last,
            self._attitude_last,
            self._gyro_last,
            self._usbl_last,
        )
        previous_state = [tensor.clone() for tensor in state_tensors]
        assert self._current_lpf is not None and self._current_walk is not None
        previous_current_lpf = self._current_lpf.clone()
        previous_current_walk = self._current_walk.clone()
        previous_queues = {
            name: (queue.buffer.clone(), queue.cursor)
            for name, queue in self._queues.items()
            if queue.buffer is not None
        }

        fresh = self.step(
            eta_true=eta_true,
            nu_true=nu_true,
            current_true_ned_mps=current_true_ned_mps,
            _prime_rows_mask=mask_t,
        )
        dt = self.config.plant.dt_s
        for tensor, old in zip(state_tensors, previous_state, strict=True):
            view = (self.num_envs,) + (1,) * (tensor.ndim - 1)
            tensor.copy_(torch.where(mask_t.view(view), tensor, old))
        self._t_s = torch.where(mask_t, self._t_s - dt, previous_state[0])
        self._current_lpf = torch.where(mask_t[:, None], self._current_lpf, previous_current_lpf)
        self._current_walk = torch.where(mask_t[:, None], self._current_walk, previous_current_walk)

        for name, queue in self._queues.items():
            if queue.buffer is None:
                raise RuntimeError("delay queue must be initialized")
            old_buffer, old_cursor = previous_queues[name]
            fresh_buffer = queue.buffer.clone()
            buffer = old_buffer.clone()
            buffer[mask_t] = fresh_buffer[mask_t]
            queue.buffer = buffer
            queue.cursor = old_cursor

        def merged(name: str) -> Tensor:
            old = getattr(previous, name)
            new = getattr(fresh, name)
            view = (self.num_envs,) + (1,) * (old.ndim - 1)
            return torch.where(mask_t.view(view), new, old)

        return SyntheticSensorOutput(
            **{field.name: merged(field.name) for field in fields(SyntheticSensorOutput)}
        )

    @property
    def parameters(self) -> _EpisodeParameters:
        if self._params is None:
            raise RuntimeError("sensors must be reset before accessing parameters")
        return self._params

    def _prime_histories(self, mask: Tensor | None = None) -> None:
        """Fill delayed-sensor history with the first valid measurement.

        A real estimator has a valid initial sample at ``t=0`` even when a
        sensor update is delayed. Filling the remaining delay slots with zeros
        creates a large artificial transient in ``p_nav`` and every delayed
        DVL/IMU channel.
        """

        values = {
            "nav": self._nav_last,
            "attitude": self._attitude_last,
            "gyro": self._gyro_last,
            "bottom": self._dvl_bottom_last,
            "water": self._dvl_water_last,
            "usbl": self._usbl_last,
            "current": self._current_lpf,
        }
        for name, value in values.items():
            if value is None:
                raise RuntimeError(f"{name} history is not initialized")
            queue = self._queues[name]
            if queue.buffer is None:
                raise RuntimeError("delay queue must be initialized")
            if mask is None:
                queue.buffer[:] = value[:, None, :]
            else:
                queue.buffer[mask] = value[mask, None, :].expand(
                    -1, queue.capacity, -1
                )

    def _due(self, index: int, rate_hz: Tensor) -> Tensor:
        if self._t_s is None:
            raise RuntimeError("sensors must be reset before use")
        due = self._t_s + 1e-12 >= self._next[:, index]
        self._next[:, index] = torch.where(
            due,
            self._next[:, index] + 1.0 / rate_hz,
            self._next[:, index],
        )
        self._last_update[:, index] = torch.where(
            due, self._t_s, self._last_update[:, index]
        )
        return due

    def _start_outage(
        self,
        timer: Tensor,
        start_rate_per_s: Tensor,
        duration_s: Tensor,
        *,
        dt: float,
    ) -> Tensor:
        """Start outages from a per-second hazard, not once per control step.

        The profile fields retain their historical names for checkpoint/config
        compatibility, but are interpreted as expected outage starts per second.
        Converting to a per-step probability keeps the realized outage duty cycle
        independent of the 50 Hz control rate.
        """

        step_probability = -torch.expm1(-torch.clamp(start_rate_per_s, min=0.0) * dt)
        start = (timer <= 0.0) & (self._rand(step_probability.shape) < step_probability)
        duration = duration_s * self._rand(duration_s.shape)
        return torch.where(start, duration, timer)

    def step(
        self,
        *,
        eta_true: Tensor,
        nu_true: Tensor,
        current_true_ned_mps: Tensor,
        _prime_rows_mask: Tensor | None = None,
    ) -> SyntheticSensorOutput:
        if self._params is None or self._t_s is None or self._current_lpf is None or self._current_walk is None:
            raise RuntimeError("sensors must be reset before step")
        eta = torch.as_tensor(eta_true, device=self.device, dtype=self.dtype)
        nu = torch.as_tensor(nu_true, device=self.device, dtype=self.dtype)
        current_ned = torch.as_tensor(current_true_ned_mps, device=self.device, dtype=self.dtype)
        if eta.ndim != 2 or eta.shape[1] != 6 or nu.shape != eta.shape:
            raise ValueError("eta_true and nu_true must have shape (N, 6)")
        if current_ned.shape != (eta.shape[0], 3):
            raise ValueError("current_true_ned_mps must have shape (N, 3)")
        if eta.shape[0] != self.num_envs:
            raise ValueError("sensor batch size does not match reset")
        p = self._params
        dt = self.config.plant.dt_s
        count = self.num_envs
        quat = quaternion_from_euler(eta[:, 3], eta[:, 4], eta[:, 5])
        current_body = world_to_body(current_ned, quat)
        if self.config.plant.velocity_contract == "nu_ground":
            ground_body = nu[:, :3]
            water_body = ground_body - current_body
        elif self.config.plant.velocity_contract == "legacy_water_nu":
            water_body = nu[:, :3]
            ground_body = water_body + current_body
        else:
            raise ValueError(
                "unsupported plant.velocity_contract="
                f"{self.config.plant.velocity_contract!r}"
            )

        # Navigation error state: episode bias + OU drift + event offset.
        rho = torch.exp(-dt / p.nav_drift_tau_s)
        noise = self._randn((count, 3))
        self._nav_ou = rho[:, None] * self._nav_ou + (
            p.nav_drift_sigma * torch.sqrt(torch.clamp(1.0 - rho.square(), min=0.0))
        )[:, None] * noise

        # DVL bottom/water lock timers and measurements.
        self._dvl_bottom_outage_s = torch.clamp(self._dvl_bottom_outage_s - dt, min=0.0)
        self._dvl_water_outage_s = torch.clamp(self._dvl_water_outage_s - dt, min=0.0)
        self._current_outage_s = torch.clamp(self._current_outage_s - dt, min=0.0)
        self._dvl_bottom_outage_s = self._start_outage(
            self._dvl_bottom_outage_s,
            p.dvl_bottom_outage_start_prob,
            p.dvl_bottom_outage_duration_s,
            dt=dt,
        )
        self._dvl_water_outage_s = self._start_outage(
            self._dvl_water_outage_s,
            p.dvl_water_outage_start_prob,
            p.dvl_water_outage_duration_s,
            dt=dt,
        )
        self._current_outage_s = self._start_outage(
            self._current_outage_s,
            p.current_est_outage_start_prob,
            p.current_est_outage_duration_s,
            dt=dt,
        )

        bottom_due = self._due(0, p.dvl_bottom_rate_hz)
        bottom_valid = self._dvl_bottom_outage_s <= 0.0
        bottom_measurement = (
            ground_body * p.dvl_bottom_scale
            + p.dvl_bottom_bias
            + p.dvl_bottom_noise[:, None] * p.noise_scale[:, None] * self._randn(ground_body.shape)
        )
        bottom_update = bottom_due & bottom_valid
        self._dvl_bottom_last = torch.where(
            bottom_update[:, None], bottom_measurement, self._dvl_bottom_last
        )
        # Push one zero-order-hold sample every control step. Pushing only on a
        # sensor's own update ticks corrupts rows when environments have
        # different update rates.
        self._queues["bottom"].push(self._dvl_bottom_last)

        water_due = self._due(1, p.dvl_water_rate_hz)
        water_valid = self._dvl_water_outage_s <= 0.0
        water_measurement = (
            water_body * p.dvl_water_scale
            + p.dvl_water_bias
            + p.dvl_water_noise[:, None] * p.noise_scale[:, None] * self._randn(water_body.shape)
        )
        water_update = water_due & water_valid
        self._dvl_water_last = torch.where(
            water_update[:, None], water_measurement, self._dvl_water_last
        )
        self._queues["water"].push(self._dvl_water_last)

        # IMU attitude/gyro.
        att_due = self._due(2, p.imu_att_rate_hz)
        self._attitude_drift = self._attitude_drift + self._randn(self._attitude_drift.shape) * (
            p.imu_drift_sigma[:, None] * math.sqrt(dt)
        )
        attitude_candidate = eta[:, 3:6] + self._attitude_drift
        attitude_candidate[:, 0:2] += (
            p.imu_rollpitch_bias[:, None]
            + p.imu_rollpitch_noise[:, None] * p.noise_scale[:, None] * self._randn((count, 2))
        )
        attitude_candidate[:, 2] += (
            p.imu_heading_bias
            + p.imu_mag_declination
            + p.imu_heading_noise * p.noise_scale * self._randn((count,))
        )
        if self.profile.name != "clean":
            attitude_candidate[:, 2] = wrap_angle(attitude_candidate[:, 2])
        self._attitude_last = torch.where(
            att_due[:, None], attitude_candidate, self._attitude_last
        )
        self._queues["attitude"].push(self._attitude_last)

        gyro_due = self._due(3, p.imu_gyro_rate_hz)
        gyro_candidate = nu[:, 3:6] * p.imu_gyro_scale + p.imu_gyro_bias
        gyro_candidate = gyro_candidate + p.imu_gyro_noise[:, None] * p.noise_scale[:, None] * self._randn(gyro_candidate.shape)
        self._gyro_last = torch.where(gyro_due[:, None], gyro_candidate, self._gyro_last)
        self._queues["gyro"].push(self._gyro_last)

        # Raw USBL plus optional rejection; p_nav is the fused navigation output.
        usbl_due = self._due(4, p.usbl_rate_hz)
        predicted_nav = eta[:, :3] + p.nav_bias + self._nav_ou + self._nav_event
        usbl_noise = p.usbl_noise[:, None] * p.noise_scale[:, None] * self._randn((count, 3))
        usbl_outlier = self._rand((count,)) < p.usbl_outlier_prob
        jump_direction = self._randn((count, 3))
        jump_direction = jump_direction / torch.clamp(
            torch.linalg.norm(jump_direction, dim=1, keepdim=True), min=1e-9
        )
        raw_usbl = eta[:, :3] + usbl_noise + usbl_outlier[:, None] * (
            p.usbl_outlier_jump[:, None] * jump_direction
        )
        residual = raw_usbl - predicted_nav
        # Innovation gate. Scaling the gate by five times the raw USBL noise
        # makes 10 m outliers indistinguishable from noise when the noise draw
        # is itself 5 m, so they leak into the fused navigation state. Keep a
        # small noise margin plus a fixed absolute margin instead.
        threshold = torch.clamp(p.usbl_noise + 3.0, min=5.0)
        rejected = usbl_outlier & self.profile.usbl_reject_outliers & (
            torch.linalg.norm(residual, dim=1) > threshold
        )
        accepted_usbl = usbl_due & ~rejected
        gain = self.profile.nav_usbl_gain
        usbl_innovation = torch.where(
            usbl_due[:, None], raw_usbl - predicted_nav, torch.zeros_like(raw_usbl)
        )
        usbl_innovation_applied = torch.where(
            accepted_usbl[:, None],
            gain * usbl_innovation,
            torch.zeros_like(usbl_innovation),
        )
        self._nav_event_usbl = self._nav_event_usbl + usbl_innovation_applied
        self._nav_event = torch.where(
            accepted_usbl[:, None],
            self._nav_event + usbl_innovation_applied,
            self._nav_event,
        )
        self._usbl_last = torch.where(usbl_due[:, None], raw_usbl, self._usbl_last)
        self._queues["usbl"].push(self._usbl_last)

        hold = self._dvl_bottom_outage_s > 0.0
        nav_dvl_outage_delta = torch.where(
            hold[:, None],
            p.nav_dvl_outage_drift * dt,
            torch.zeros_like(self._nav_event),
        )
        self._nav_event_dvl = self._nav_event_dvl + nav_dvl_outage_delta
        self._nav_event = self._nav_event + torch.where(
            hold[:, None],
            p.nav_dvl_outage_drift * dt,
            torch.zeros_like(self._nav_event),
        )
        if bool(torch.any(p.nav_noise > 0.0)):
            nav_noise = p.nav_noise * p.noise_scale[:, None] * self._randn((count, 3))
        else:
            nav_noise = torch.zeros((count, 3), device=self.device, dtype=self.dtype)
        p_nav_current = eta[:, :3] + p.nav_bias + self._nav_ou + self._nav_event + nav_noise
        self._queues["nav"].push(p_nav_current)

        # Current estimate: correlated low-pass state + fixed bias + random walk.
        previous_current_update_s = self._last_update[:, 5].clone()
        current_due = self._due(5, p.current_est_rate_hz)
        current_valid = self._current_outage_s <= 0.0
        self._current_walk = self._current_walk + self._randn(self._current_walk.shape) * (
            p.current_est_walk[:, None] * math.sqrt(dt)
        )
        target = (
            current_body
            + p.current_est_bias
            + self._current_walk
            + p.current_est_noise[:, None] * p.noise_scale[:, None] * self._randn(current_body.shape)
        )
        # Updating only when a low-rate measurement arrives requires the elapsed
        # sample interval, not the 50 Hz control step.  The legacy basis remains
        # selectable so historical replay stays bit-compatible.
        if self.config.sensors.current_est_lpf_time_basis == "elapsed_sample":
            elapsed_s = self._t_s - previous_current_update_s
            elapsed_s = torch.where(
                previous_current_update_s > 0.0,
                elapsed_s,
                1.0 / torch.clamp(p.current_est_rate_hz, min=1e-9),
            )
            elapsed_s = torch.clamp(elapsed_s, min=dt)
        else:
            elapsed_s = torch.full_like(p.current_est_rate_hz, dt)
        alpha = torch.where(
            p.current_est_tau_s > 0.0,
            1.0 - torch.exp(-elapsed_s / torch.clamp(p.current_est_tau_s, min=1e-9)),
            torch.ones_like(p.current_est_tau_s),
        )
        updated_current = self._current_lpf + alpha[:, None] * (target - self._current_lpf)
        current_update = current_due & current_valid
        self._current_lpf = torch.where(current_update[:, None], updated_current, self._current_lpf)
        self._queues["current"].push(self._current_lpf)

        self._nav_last = p_nav_current
        if _prime_rows_mask is not None:
            self._prime_histories(_prime_rows_mask)
        elif not self._histories_primed:
            self._prime_histories()
            self._histories_primed = True

        p_nav = self._queues["nav"].read(p.nav_delay_steps)
        attitude_out = self._queues["attitude"].read(p.imu_att_delay_steps)
        gyro_out = self._queues["gyro"].read(p.imu_gyro_delay_steps)
        bottom_out = self._queues["bottom"].read(p.dvl_bottom_delay_steps)
        water_out = self._queues["water"].read(p.dvl_water_delay_steps)
        usbl_out = self._queues["usbl"].read(p.usbl_delay_steps)
        current_out = self._queues["current"].read(p.current_est_delay_steps)

        nav_age_s = p.nav_delay_steps.to(self.dtype) * dt + torch.clamp(
            self._t_s - self._last_update[:, 4], min=0.0
        )
        dvl_bottom_age_s = p.dvl_bottom_delay_steps.to(self.dtype) * dt + torch.clamp(
            self._t_s - self._last_update[:, 0], min=0.0
        )
        dvl_water_age_s = p.dvl_water_delay_steps.to(self.dtype) * dt + torch.clamp(
            self._t_s - self._last_update[:, 1], min=0.0
        )
        usbl_age_s = p.usbl_delay_steps.to(self.dtype) * dt + torch.clamp(
            self._t_s - self._last_update[:, 4], min=0.0
        )
        current_estimate_age_s = p.current_est_delay_steps.to(self.dtype) * dt + torch.clamp(
            self._t_s - self._last_update[:, 5], min=0.0
        )
        self._t_s = self._t_s + dt
        return SyntheticSensorOutput(
            p_nav=p_nav,
            attitude=attitude_out,
            angular_rate=gyro_out,
            dvl_ground_body=bottom_out,
            dvl_water_body=water_out,
            current_estimate_body=current_out,
            usbl_position=usbl_out,
            bottom_lock_valid=bottom_valid.to(self.dtype),
            water_track_valid=water_valid.to(self.dtype),
            current_estimate_valid=current_valid.to(self.dtype),
            usbl_valid=(usbl_due & ~rejected).to(self.dtype),
            usbl_outlier=usbl_outlier.to(self.dtype),
            nav_age_s=nav_age_s,
            dvl_bottom_age_s=dvl_bottom_age_s,
            dvl_water_age_s=dvl_water_age_s,
            usbl_age_s=usbl_age_s,
            current_estimate_age_s=current_estimate_age_s,
            nav_ou=self._nav_ou,
            nav_event=self._nav_event,
            nav_event_usbl=self._nav_event_usbl,
            nav_event_dvl=self._nav_event_dvl,
            nav_bias=p.nav_bias,
            usbl_due=usbl_due.to(self.dtype),
            usbl_accepted=accepted_usbl.to(self.dtype),
            usbl_rejected=(usbl_due & rejected).to(self.dtype),
            usbl_innovation=usbl_innovation,
            usbl_innovation_applied=usbl_innovation_applied,
            usbl_raw=self._usbl_last,
            usbl_predicted=predicted_nav,
            dvl_bottom_outage=hold.to(self.dtype),
            nav_dvl_outage_delta=nav_dvl_outage_delta,
        )


def pc_critic_state(
    *,
    eta: Tensor,
    nu: Tensor,
    heading_error_rad: Tensor,
    pitch_error_rad: Tensor,
    speed_error_mps: Tensor,
) -> Tensor:
    """Build the ideal 15-D critic state in the frozen PC order."""

    eta_t = torch.as_tensor(eta)
    nu_t = torch.as_tensor(nu)
    heading = torch.as_tensor(heading_error_rad, device=eta_t.device, dtype=eta_t.dtype)
    pitch = torch.as_tensor(pitch_error_rad, device=eta_t.device, dtype=eta_t.dtype)
    speed = torch.as_tensor(speed_error_mps, device=eta_t.device, dtype=eta_t.dtype)
    if eta_t.ndim != 2 or eta_t.shape[1] != 6 or nu_t.shape != eta_t.shape:
        raise ValueError("eta and nu must have shape (N, 6)")
    count = eta_t.shape[0]
    if heading.shape != (count,) or pitch.shape != (count,) or speed.shape != (count,):
        raise ValueError("reference errors must have shape (N,)")
    return torch.cat(
        [
            eta_t[:, 0:6],
            nu_t[:, 0:3],
            nu_t[:, 3:6],
            torch.stack([heading, pitch, speed], dim=1),
        ],
        dim=1,
    )


__all__ = [
    "PROFILES",
    "SensorProfile",
    "SyntheticSensorOutput",
    "SyntheticSensorSuite",
    "pc_critic_state",
]
