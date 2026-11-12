"""Batched REMUS100 dynamics mirrored from Python Vehicle Simulator.

The reference implementation lives in
``python_vehicle_simulator.vehicles.remus100``. This module keeps the same
state ordering, actuator saturation order and forward-Euler integration order,
but replaces the per-vehicle Python loop with tensor operations over a batch
dimension.

State and command conventions:

* ``eta = [x, y, z, phi, theta, psi]`` in NED, radians for angles.
* ``nu = [u, v, w, p, q, r]`` in body axes.
* ``actuator = [rudder_rad, stern_plane_rad, propeller_rpm]``.
* ``current_velocity_ned_mps`` contains only horizontal ``(north, east)``
  current; the PVS REMUS100 model does not include vertical current.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


PLANT_SCALE_NAMES = (
    "drag_scale",
    "thrust_scale",
    "fin_scale",
    "added_mass_scale",
    "actuator_time_scale",
)


@dataclass(frozen=True)
class Remus100BatchState:
    """State of a batch of REMUS100 vehicles."""

    eta: Tensor
    nu: Tensor
    actuator: Tensor


@dataclass(frozen=True)
class Remus100BatchStep:
    """Result of one batched forward-Euler step."""

    state: Remus100BatchState
    nu_dot: Tensor
    actuator_dot: Tensor
    saturated_actuator: Tensor


def _skew(vectors: Tensor) -> Tensor:
    """Return batched skew-symmetric matrices such that ``S(a) @ b = a x b``."""

    result = torch.zeros(
        (vectors.shape[0], 3, 3),
        dtype=vectors.dtype,
        device=vectors.device,
    )
    result[:, 0, 1] = -vectors[:, 2]
    result[:, 0, 2] = vectors[:, 1]
    result[:, 1, 0] = vectors[:, 2]
    result[:, 1, 2] = -vectors[:, 0]
    result[:, 2, 0] = -vectors[:, 1]
    result[:, 2, 1] = vectors[:, 0]
    return result


def _transform_matrix(offset: np.ndarray) -> np.ndarray:
    """Equivalent of PVS ``Hmtrx`` for a constant CG offset."""

    result = np.eye(6, dtype=np.float64)
    result[0:3, 3:6] = np.array(
        [
            [0.0, offset[2], -offset[1]],
            [-offset[2], 0.0, offset[0]],
            [offset[1], -offset[0], 0.0],
        ],
        dtype=np.float64,
    )
    return result


def _hoerner_cd(beam: float, draft: float) -> float:
    """PVS Hoerner cross-flow coefficient at a constant beam/draft ratio."""

    ratios = np.array(
        [
            0.0109,
            0.1766,
            0.3530,
            0.4519,
            0.4728,
            0.4929,
            0.4933,
            0.5585,
            0.6464,
            0.8336,
            0.9880,
            1.3081,
            1.6392,
            1.8600,
            2.3129,
            2.6000,
            3.0088,
            3.4508,
            3.7379,
            4.0031,
        ],
        dtype=np.float64,
    )
    coefficients = np.array(
        [
            1.9661,
            1.9657,
            1.8976,
            1.7872,
            1.5837,
            1.2786,
            1.2108,
            1.0836,
            0.9986,
            0.8796,
            0.8284,
            0.7599,
            0.6914,
            0.6571,
            0.6307,
            0.5962,
            0.5868,
            0.5859,
            0.5599,
            0.5593,
        ],
        dtype=np.float64,
    )
    return float(np.interp(beam / (2.0 * draft), ratios, coefficients))


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    """Compare devices without treating ``cuda`` and ``cuda:0`` as different."""

    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


class BatchedRemus100Dynamics:
    """Vectorized REMUS100 six-DOF Fossen model with actuator dynamics."""

    def __init__(
        self,
        *,
        sample_time_s: float = 0.02,
        max_rudder_angle_rad: float = math.radians(15.0),
        max_stern_plane_angle_rad: float = math.radians(15.0),
        max_propeller_rpm: float = 1525.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        validate_finite: bool = False,
    ) -> None:
        if not math.isfinite(sample_time_s) or sample_time_s <= 0.0:
            raise ValueError("sample_time_s must be finite and positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        for name, value in (
            ("max_rudder_angle_rad", max_rudder_angle_rad),
            ("max_stern_plane_angle_rad", max_stern_plane_angle_rad),
            ("max_propeller_rpm", max_propeller_rpm),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.sample_time_s = float(sample_time_s)
        self.max_rudder_angle_rad = float(max_rudder_angle_rad)
        self.max_stern_plane_angle_rad = float(max_stern_plane_angle_rad)
        self.max_propeller_rpm = float(max_propeller_rpm)
        self.device = torch.device(device)
        self.dtype = dtype
        self.validate_finite = bool(validate_finite)
        self._build_parameters()

    def _tensor(self, value: np.ndarray | float) -> Tensor:
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _build_parameters(self) -> None:
        rho = 1026.0
        gravity = 9.81
        length = 1.6
        diameter = 0.19
        fin_area = 0.00665
        semi_major = length / 2.0
        semi_minor = diameter / 2.0
        reference_area = 0.7 * length * diameter
        cg_offset = np.array([0.0, 0.0, 0.02], dtype=np.float64)
        cb_offset = np.zeros(3, dtype=np.float64)

        drag_coefficient = 0.42
        parasitic_drag = drag_coefficient * math.pi * semi_minor**2 / reference_area
        mass = 4.0 / 3.0 * math.pi * rho * semi_major * semi_minor**2
        inertia_x = 0.4 * mass * semi_minor**2
        inertia_y = 0.2 * mass * (semi_major**2 + semi_minor**2)
        rigid_body_cg = np.diag([mass, mass, mass, inertia_x, inertia_y, inertia_y])
        transform = _transform_matrix(cg_offset)
        rigid_body = transform.T @ rigid_body_cg @ transform

        added_roll = 0.3 * inertia_x
        eccentricity = math.sqrt(1.0 - (semi_minor / semi_major) ** 2)
        alpha_0 = (2.0 * (1.0 - eccentricity**2) / eccentricity**3) * (
            0.5 * math.log((1.0 + eccentricity) / (1.0 - eccentricity)) - eccentricity
        )
        beta_0 = 1.0 / eccentricity**2 - (1.0 - eccentricity**2) / (
            2.0 * eccentricity**3
        ) * math.log((1.0 + eccentricity) / (1.0 - eccentricity))
        k1 = alpha_0 / (2.0 - alpha_0)
        k2 = beta_0 / (2.0 - beta_0)
        k_prime = eccentricity**4 * (beta_0 - alpha_0) / (
            (2.0 - eccentricity**2)
            * (
                2.0 * eccentricity**2
                - (2.0 - eccentricity**2) * (beta_0 - alpha_0)
            )
        )
        added_mass = np.diag(
            [mass * k1, mass * k2, mass * k2, added_roll, k_prime * inertia_y, k_prime * inertia_y]
        )

        mass_matrix = rigid_body + added_mass
        mass_inverse = np.linalg.inv(mass_matrix)
        weight = mass * gravity
        buoyancy = weight
        natural_roll = math.sqrt(weight * (cg_offset[2] - cb_offset[2]) / mass_matrix[3, 3])
        natural_pitch = math.sqrt(weight * (cg_offset[2] - cb_offset[2]) / mass_matrix[4, 4])

        damping_diagonal = np.array(
            [
                mass_matrix[0, 0] / 20.0,
                mass_matrix[1, 1] / 20.0,
                mass_matrix[2, 2] / 20.0,
                mass_matrix[3, 3] * 2.0 * 0.3 * natural_roll,
                mass_matrix[4, 4] * 2.0 * 0.8 * natural_pitch,
                mass_matrix[5, 5] / 1.0,
            ],
            dtype=np.float64,
        )

        rudder_area = 2.0 * fin_area
        stern_area = 2.0 * fin_area
        aspect_ratio = diameter**2 / reference_area
        lift_slope = math.pi * aspect_ratio / (1.0 + math.sqrt(1.0 + (aspect_ratio / 2.0) ** 2))

        strips = 20
        strip_length = length / strips
        strip_positions = -length / 2.0 + strip_length / 2.0 + np.arange(strips) * strip_length
        strip_drag = _hoerner_cd(diameter, diameter)

        self._rho = rho
        self._length = length
        self._diameter = diameter
        self._semi_major = semi_major
        self._reference_area = reference_area
        self._parasitic_drag = parasitic_drag
        self._mass_matrix = self._tensor(mass_matrix)
        self._mass_inverse = self._tensor(mass_inverse)
        self._rigid_body = self._tensor(rigid_body)
        self._added_mass = self._tensor(added_mass)
        self._weight = weight
        self._buoyancy = buoyancy
        self._cg_offset = self._tensor(cg_offset)
        self._cb_offset = self._tensor(cb_offset)
        self._damping_diagonal = self._tensor(damping_diagonal)

        self._rudder_area = rudder_area
        self._stern_area = stern_area
        self._rudder_lift_coefficient = 0.5
        self._stern_lift_coefficient = 0.7
        self._rudder_x = -semi_major
        self._stern_x = -semi_major
        self._lift_slope = lift_slope

        self._strip_positions = self._tensor(strip_positions)
        self._strip_drag = strip_drag
        self._strip_scale = 0.5 * rho * diameter * strip_drag * strip_length

        self._max_rudder = self.max_rudder_angle_rad
        self._max_stern_plane = self.max_stern_plane_angle_rad
        self._max_rpm = self.max_propeller_rpm
        self._max_rudder_tensor = torch.tensor(
            self._max_rudder, device=self.device, dtype=self.dtype
        )
        self._max_stern_plane_tensor = torch.tensor(
            self._max_stern_plane, device=self.device, dtype=self.dtype
        )
        self._max_rpm_tensor = torch.tensor(
            self._max_rpm, device=self.device, dtype=self.dtype
        )
        self._rudder_time_constant = 0.1
        self._propeller_time_constant = 1.0

        self._propeller_diameter = 0.14
        self._thrust_deduction = 0.1
        self._advance_speed_scale = 0.944
        self._advance_ratio_max = 0.6632
        self._thrust_coefficient_zero = 0.4566
        self._torque_coefficient_zero = 0.0700
        self._thrust_coefficient_max = 0.1798
        self._torque_coefficient_max = 0.0312
        self._thrust_slope = (
            self._thrust_coefficient_max - self._thrust_coefficient_zero
        ) / self._advance_ratio_max
        self._torque_slope = (
            self._torque_coefficient_max - self._torque_coefficient_zero
        ) / self._advance_ratio_max

    def _validate_input(self, value: Tensor, name: str, shape: tuple[int, ...]) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if not _device_matches(value.device, self.device) or value.dtype != self.dtype:
            raise ValueError(
                f"{name} must be on {self.device} with dtype {self.dtype}, "
                f"got {value.device} and {value.dtype}"
            )
        if self.validate_finite and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains NaN or Inf")

    def _saturate(self, value: Tensor, limit_tensor: Tensor) -> Tensor:
        return torch.where(
            value.abs() >= limit_tensor,
            torch.sign(value) * limit_tensor,
            value,
        )

    def _coriolis(self, mass: Tensor, velocity: Tensor) -> Tensor:
        if mass.ndim == 2:
            mass = mass.unsqueeze(0).expand(velocity.shape[0], 6, 6)
        symmetric = 0.5 * (mass + mass.transpose(1, 2))
        m11 = symmetric[:, 0:3, 0:3]
        m12 = symmetric[:, 0:3, 3:6]
        m21 = m12.transpose(1, 2)
        m22 = symmetric[:, 3:6, 3:6]
        velocity_linear = velocity[:, 0:3]
        velocity_angular = velocity[:, 3:6]
        momentum_linear = torch.bmm(m11, velocity_linear.unsqueeze(-1)).squeeze(-1)
        momentum_linear += torch.bmm(m12, velocity_angular.unsqueeze(-1)).squeeze(-1)
        momentum_angular = torch.bmm(m21, velocity_linear.unsqueeze(-1)).squeeze(-1)
        momentum_angular += torch.bmm(m22, velocity_angular.unsqueeze(-1)).squeeze(-1)
        result = torch.zeros(
            (velocity.shape[0], 6, 6),
            dtype=self.dtype,
            device=self.device,
        )
        skew_linear = _skew(momentum_linear)
        skew_angular = _skew(momentum_angular)
        result[:, 0:3, 3:6] = -skew_linear
        result[:, 3:6, 0:3] = -skew_linear
        result[:, 3:6, 3:6] = -skew_angular
        return result

    def _rotation_zyx(self, roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
        cos_roll = torch.cos(roll)
        sin_roll = torch.sin(roll)
        cos_pitch = torch.cos(pitch)
        sin_pitch = torch.sin(pitch)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        row_0 = torch.stack(
            [
                cos_yaw * cos_pitch,
                -sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll,
                sin_yaw * sin_roll + cos_yaw * cos_roll * sin_pitch,
            ],
            dim=-1,
        )
        row_1 = torch.stack(
            [
                sin_yaw * cos_pitch,
                cos_yaw * cos_roll + sin_roll * sin_pitch * sin_yaw,
                -cos_yaw * sin_roll + sin_pitch * sin_yaw * cos_roll,
            ],
            dim=-1,
        )
        row_2 = torch.stack(
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll],
            dim=-1,
        )
        return torch.stack([row_0, row_1, row_2], dim=-2)

    def _angle_rate_matrix(self, roll: Tensor, pitch: Tensor) -> Tensor:
        cos_roll = torch.cos(roll)
        sin_roll = torch.sin(roll)
        cos_pitch = torch.cos(pitch)
        sin_pitch = torch.sin(pitch)
        row_0 = torch.stack(
            [
                torch.ones_like(roll),
                sin_roll * sin_pitch / cos_pitch,
                cos_roll * sin_pitch / cos_pitch,
            ],
            dim=-1,
        )
        row_1 = torch.stack([torch.zeros_like(roll), cos_roll, -sin_roll], dim=-1)
        row_2 = torch.stack(
            [
                torch.zeros_like(roll),
                sin_roll / cos_pitch,
                cos_roll / cos_pitch,
            ],
            dim=-1,
        )
        return torch.stack([row_0, row_1, row_2], dim=-2)

    def step(
        self,
        state: Remus100BatchState,
        command: Tensor,
        current_velocity_ned_mps: Tensor,
        *,
        dt_s: float | None = None,
        plant_scales: Tensor | None = None,
    ) -> Remus100BatchStep:
        """Advance a batch by one PVS-compatible forward-Euler step."""

        eta = state.eta
        nu = state.nu
        actuator = state.actuator
        if eta.ndim != 2 or eta.shape[1] != 6:
            raise ValueError("state.eta must have shape (N, 6)")
        batch_size = eta.shape[0]
        self._validate_input(eta, "state.eta", (batch_size, 6))
        self._validate_input(nu, "state.nu", (batch_size, 6))
        self._validate_input(actuator, "state.actuator", (batch_size, 3))
        self._validate_input(command, "command", (batch_size, 3))
        self._validate_input(
            current_velocity_ned_mps,
            "current_velocity_ned_mps",
            (batch_size, 2),
        )
        if plant_scales is None:
            plant_scales = torch.ones(
                (batch_size, len(PLANT_SCALE_NAMES)),
                dtype=self.dtype,
                device=self.device,
            )
        self._validate_input(
            plant_scales,
            "plant_scales",
            (batch_size, len(PLANT_SCALE_NAMES)),
        )
        if not bool((plant_scales > 0.0).all().item()):
            raise ValueError("plant_scales must be strictly positive")
        (
            drag_scale,
            thrust_scale,
            fin_scale,
            added_mass_scale,
            actuator_time_scale,
        ) = plant_scales.unbind(dim=1)
        dt = self.sample_time_s if dt_s is None else float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        delta_r = self._saturate(actuator[:, 0], self._max_rudder_tensor)
        delta_s = self._saturate(actuator[:, 1], self._max_stern_plane_tensor)
        rpm = self._saturate(actuator[:, 2], self._max_rpm_tensor)

        yaw = eta[:, 5]
        current_speed = torch.linalg.vector_norm(current_velocity_ned_mps, dim=1)
        current_direction = torch.atan2(
            current_velocity_ned_mps[:, 1], current_velocity_ned_mps[:, 0]
        )
        current_surge = current_speed * torch.cos(current_direction - yaw)
        current_sway = current_speed * torch.sin(current_direction - yaw)
        current_body = torch.zeros_like(nu)
        current_body[:, 0] = current_surge
        current_body[:, 1] = current_sway
        yaw_rate = nu[:, 5]
        current_acceleration = torch.zeros_like(nu)
        current_acceleration[:, 0] = yaw_rate * current_sway
        current_acceleration[:, 1] = -yaw_rate * current_surge

        relative_velocity = nu - current_body
        relative_linear = relative_velocity[:, 0:3]
        alpha = torch.atan2(relative_velocity[:, 2], relative_velocity[:, 0])
        relative_speed = torch.linalg.vector_norm(relative_linear, dim=1)

        rpm_rps = rpm / 60.0
        advance_speed = self._advance_speed_scale * relative_speed
        thrust_forward = self._rho * self._propeller_diameter**4 * (
            self._thrust_coefficient_zero * rpm_rps.abs() * rpm_rps
            + self._thrust_slope
            * (advance_speed / self._propeller_diameter)
            * rpm_rps.abs()
        )
        torque_forward = self._rho * self._propeller_diameter**5 * (
            self._torque_coefficient_zero * rpm_rps.abs() * rpm_rps
            + self._torque_slope
            * (advance_speed / self._propeller_diameter)
            * rpm_rps.abs()
        )
        thrust_reverse = (
            self._rho * self._propeller_diameter**4 * self._thrust_coefficient_zero
            * rpm_rps.abs() * rpm_rps
        )
        torque_reverse = (
            self._rho * self._propeller_diameter**5 * self._torque_coefficient_zero
            * rpm_rps.abs() * rpm_rps
        )
        thrust = torch.where(rpm_rps > 0.0, thrust_forward, thrust_reverse)
        propeller_torque = torch.where(rpm_rps > 0.0, torque_forward, torque_reverse)
        thrust = thrust * thrust_scale

        coriolis = self._coriolis(self._rigid_body, relative_velocity)
        added_mass = self._added_mass.unsqueeze(0) * added_mass_scale.view(-1, 1, 1)
        mass_matrix = self._rigid_body.unsqueeze(0) + added_mass
        mass_inverse = torch.linalg.inv(mass_matrix)
        added_coriolis = self._coriolis(added_mass, relative_velocity)
        for row, column in ((4, 0), (0, 4), (4, 2), (2, 4), (5, 0), (0, 5), (5, 1), (1, 5)):
            added_coriolis[:, row, column] = 0.0
        coriolis = coriolis + added_coriolis

        damping_diagonal = self._damping_diagonal.unsqueeze(0).expand(batch_size, 6).clone()
        damping_diagonal[:, 0] *= torch.exp(-3.0 * relative_speed)
        damping_diagonal[:, 1] *= torch.exp(-3.0 * relative_speed)
        damping_diagonal *= drag_scale.unsqueeze(1)
        damping = torch.diag_embed(damping_diagonal)
        hydrodynamic_damping = torch.bmm(
            coriolis + damping,
            relative_velocity.unsqueeze(-1),
        ).squeeze(-1)

        lift = 0.5 * self._rho * relative_speed**2 * self._reference_area * (
            self._lift_slope * alpha
        )
        drag = 0.5 * self._rho * relative_speed**2 * self._reference_area * (
            self._parasitic_drag
            + (self._lift_slope * alpha) ** 2
            / (math.pi * 0.7 * (self._diameter**2 / self._reference_area))
        )
        scaled_drag = drag * drag_scale
        lift_drag = torch.zeros_like(nu)
        lift_drag[:, 0] = torch.cos(alpha) * (-scaled_drag) + torch.sin(alpha) * lift
        lift_drag[:, 2] = -torch.sin(alpha) * scaled_drag - torch.cos(alpha) * lift

        flow_velocity = relative_velocity[:, 1].unsqueeze(1) + (
            relative_velocity[:, 5].unsqueeze(1) * self._strip_positions.unsqueeze(0)
        )
        cross_flow_strength = flow_velocity.abs() * flow_velocity
        sway_cross_flow = -(self._strip_scale * cross_flow_strength).sum(dim=1)
        yaw_cross_flow = -(
            self._strip_scale
            * self._strip_positions.unsqueeze(0)
            * cross_flow_strength
        ).sum(dim=1)
        cross_flow = torch.zeros_like(nu)
        cross_flow[:, 1] = sway_cross_flow * drag_scale
        cross_flow[:, 5] = yaw_cross_flow * drag_scale

        horizontal_speed_sq = relative_velocity[:, 0] ** 2 + relative_velocity[:, 1] ** 2
        vertical_speed_sq = relative_velocity[:, 0] ** 2 + relative_velocity[:, 2] ** 2
        rudder_drag = -0.5 * self._rho * horizontal_speed_sq * self._rudder_area * (
            self._rudder_lift_coefficient * delta_r**2
        )
        stern_drag = -0.5 * self._rho * vertical_speed_sq * self._stern_area * (
            self._stern_lift_coefficient * delta_s**2
        )
        rudder_sway = -0.5 * self._rho * horizontal_speed_sq * self._rudder_area * (
            self._rudder_lift_coefficient * delta_r
        )
        stern_heave = -0.5 * self._rho * vertical_speed_sq * self._stern_area * (
            self._stern_lift_coefficient * delta_s
        )
        rudder_drag = rudder_drag * fin_scale
        stern_drag = stern_drag * fin_scale
        rudder_sway = rudder_sway * fin_scale
        stern_heave = stern_heave * fin_scale
        actuator_force = torch.zeros_like(nu)
        actuator_force[:, 0] = (1.0 - self._thrust_deduction) * thrust + rudder_drag + stern_drag
        actuator_force[:, 1] = rudder_sway
        actuator_force[:, 2] = stern_heave
        actuator_force[:, 3] = propeller_torque / 10.0
        actuator_force[:, 4] = -self._stern_x * stern_heave
        actuator_force[:, 5] = self._rudder_x * rudder_sway

        roll = eta[:, 3]
        pitch = eta[:, 4]
        sin_pitch = torch.sin(pitch)
        cos_pitch = torch.cos(pitch)
        sin_roll = torch.sin(roll)
        cos_roll = torch.cos(roll)
        weight_minus_buoyancy = self._weight - self._buoyancy
        cg_weight = self._cg_offset * self._weight
        cb_buoyancy = self._cb_offset * self._buoyancy
        restoring = torch.zeros_like(nu)
        restoring[:, 0] = weight_minus_buoyancy * sin_pitch
        restoring[:, 1] = -weight_minus_buoyancy * cos_pitch * sin_roll
        restoring[:, 2] = -weight_minus_buoyancy * cos_pitch * cos_roll
        restoring[:, 3] = (
            -(cg_weight[1] - cb_buoyancy[1]) * cos_pitch * cos_roll
            + (cg_weight[2] - cb_buoyancy[2]) * cos_pitch * sin_roll
        )
        restoring[:, 4] = (
            (cg_weight[2] - cb_buoyancy[2]) * sin_pitch
            + (cg_weight[0] - cb_buoyancy[0]) * cos_pitch * cos_roll
        )
        restoring[:, 5] = (
            -(cg_weight[0] - cb_buoyancy[0]) * cos_pitch * sin_roll
            - (cg_weight[1] - cb_buoyancy[1]) * sin_pitch
        )

        total_force = actuator_force + lift_drag + cross_flow - hydrodynamic_damping - restoring
        nu_dot = current_acceleration + torch.bmm(
            mass_inverse, total_force.unsqueeze(-1)
        ).squeeze(-1)

        actuator_dot = torch.empty_like(actuator)
        rudder_time_constant = self._rudder_time_constant * actuator_time_scale
        propeller_time_constant = self._propeller_time_constant * actuator_time_scale
        actuator_dot[:, 0] = (command[:, 0] - delta_r) / rudder_time_constant
        actuator_dot[:, 1] = (command[:, 1] - delta_s) / rudder_time_constant
        actuator_dot[:, 2] = (command[:, 2] - rpm) / propeller_time_constant

        nu_next = nu + dt * nu_dot
        saturated_actuator = torch.stack([delta_r, delta_s, rpm], dim=1)
        actuator_next = saturated_actuator + dt * actuator_dot
        rotation = self._rotation_zyx(roll, pitch, yaw)
        angle_rate = self._angle_rate_matrix(roll, pitch)
        position_dot = torch.bmm(rotation, nu_next[:, 0:3].unsqueeze(-1)).squeeze(-1)
        euler_dot = torch.bmm(angle_rate, nu_next[:, 3:6].unsqueeze(-1)).squeeze(-1)
        eta_next = eta + dt * torch.cat((position_dot, euler_dot), dim=1)
        return Remus100BatchStep(
            state=Remus100BatchState(eta=eta_next, nu=nu_next, actuator=actuator_next),
            nu_dot=nu_dot,
            actuator_dot=actuator_dot,
            saturated_actuator=saturated_actuator,
        )


__all__ = [
    "BatchedRemus100Dynamics",
    "PLANT_SCALE_NAMES",
    "Remus100BatchState",
    "Remus100BatchStep",
]
