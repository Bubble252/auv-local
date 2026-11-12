"""Batched attitude helpers matching the scalar planner conventions."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def wrap_angle(angle_rad: Tensor) -> Tensor:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_from_euler(roll_rad: Tensor, pitch_rad: Tensor, yaw_rad: Tensor) -> Tensor:
    """Return xyzw quaternions from intrinsic ZYX Euler angles."""

    roll, pitch, yaw = torch.broadcast_tensors(roll_rad, pitch_rad, yaw_rad)
    cr = torch.cos(roll / 2.0)
    sr = torch.sin(roll / 2.0)
    cp = torch.cos(pitch / 2.0)
    sp = torch.sin(pitch / 2.0)
    cy = torch.cos(yaw / 2.0)
    sy = torch.sin(yaw / 2.0)
    quat = torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True)


def euler_from_quaternion(quat_xyzw: Tensor) -> Tensor:
    """Return intrinsic ZYX roll, pitch and yaw for xyzw quaternions."""

    quat = torch.as_tensor(quat_xyzw)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("quat_xyzw must have shape (N, 4)")
    norm = torch.linalg.norm(quat, dim=1, keepdim=True)
    x, y, z, w = (quat / torch.clamp(norm, min=1e-12)).unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(pitch_argument)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=1)


def quaternion_from_direction(direction_ned: Tensor) -> Tensor:
    """Return the shortest-arc quaternion aligning body +x with each direction."""

    direction = torch.as_tensor(direction_ned)
    if direction.ndim != 2 or direction.shape[1] != 3:
        raise ValueError("direction_ned must have shape (N, 3)")
    norm = torch.linalg.norm(direction, dim=1, keepdim=True)
    unit = torch.where(norm > 1e-12, direction / torch.clamp(norm, min=1e-12), direction)
    default_direction = torch.zeros_like(unit)
    default_direction[:, 0] = 1.0
    unit = torch.where(norm > 1e-12, unit, default_direction)
    dot = torch.clamp(unit[:, 0], -1.0, 1.0)
    body_x = torch.zeros_like(unit)
    body_x[:, 0] = 1.0
    axis = torch.cross(body_x, unit, dim=1)
    axis_norm = torch.linalg.norm(axis, dim=1, keepdim=True)
    axis_unit = axis / torch.clamp(axis_norm, min=1e-12)
    half_angle = torch.acos(dot) / 2.0
    regular = torch.cat((axis_unit * torch.sin(half_angle)[:, None], torch.cos(half_angle)[:, None]), dim=1)
    identity = direction.new_zeros((direction.shape[0], 4))
    identity[:, 3] = 1.0
    opposite = direction.new_zeros((direction.shape[0], 4))
    opposite[:, 2] = 1.0
    result = torch.where((dot > 1.0 - 1e-12)[:, None], identity, regular)
    return torch.where((dot < -1.0 + 1e-12)[:, None], opposite, result)


def body_x_direction(quat_xyzw: Tensor) -> Tensor:
    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    z = quat_xyzw[:, 2]
    w = quat_xyzw[:, 3]
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ),
        dim=1,
    )


def yaw_pitch(quat_xyzw: Tensor) -> tuple[Tensor, Tensor]:
    direction = body_x_direction(quat_xyzw)
    yaw = torch.atan2(direction[:, 1], direction[:, 0])
    pitch = torch.atan2(-direction[:, 2], torch.linalg.norm(direction[:, :2], dim=1))
    return yaw, pitch


def world_to_body(vector_ned: Tensor, quat_xyzw: Tensor) -> Tensor:
    """Rotate NED vectors into each quaternion's body frame."""

    vectors = torch.as_tensor(vector_ned)
    quat = torch.as_tensor(quat_xyzw)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vector_ned must have shape (N, 3)")
    if quat.shape != (vectors.shape[0], 4):
        raise ValueError("quat_xyzw must have shape (N, 4)")
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]
    rotation = torch.empty((vectors.shape[0], 3, 3), dtype=quat.dtype, device=quat.device)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return torch.einsum("nji,nj->ni", rotation, vectors)


def body_to_world(vector_body: Tensor, quat_xyzw: Tensor) -> Tensor:
    """Rotate body-frame vectors into NED using xyzw quaternions."""

    vectors = torch.as_tensor(vector_body)
    quat = torch.as_tensor(quat_xyzw)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vector_body must have shape (N, 3)")
    if quat.shape != (vectors.shape[0], 4):
        raise ValueError("quat_xyzw must have shape (N, 4)")
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]
    rotation = torch.empty((vectors.shape[0], 3, 3), dtype=quat.dtype, device=quat.device)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return torch.einsum("nij,nj->ni", rotation, vectors)


def ground_velocity_body(
    eta: Tensor,
    nu: Tensor,
    current_ned_mps: Tensor,
) -> Tensor:
    """Return body-frame ground velocity.

    In this project the dynamics integrate ``eta`` with ``nu`` and compute
    hydrodynamic forces from ``nu - current_body``.  Therefore ``nu[:3]`` is
    already the ground velocity; adding the current again would double-count it.
    """

    eta_t = torch.as_tensor(eta)
    nu_t = torch.as_tensor(nu)
    if eta_t.ndim != 2 or eta_t.shape[1] != 6 or nu_t.shape != eta_t.shape:
        raise ValueError("eta and nu must have shape (N, 6)")
    return nu_t[:, :3]


def legacy_ground_velocity_body(
    eta: Tensor,
    nu: Tensor,
    current_ned_mps: Tensor,
) -> Tensor:
    """Return the historical ``nu + current`` ground velocity.

    This helper exists only for bit-compatible replay of pre-Goal164 configs.
    New code must use :func:`ground_velocity_body` with ``nu_ground``.
    """

    eta_t = torch.as_tensor(eta)
    nu_t = torch.as_tensor(nu)
    current = torch.as_tensor(current_ned_mps)
    if eta_t.ndim != 2 or eta_t.shape[1] != 6 or nu_t.shape != eta_t.shape:
        raise ValueError("eta and nu must have shape (N, 6)")
    if current.shape != (eta_t.shape[0], 3):
        raise ValueError("current_ned_mps must have shape (N, 3)")
    quat = quaternion_from_euler(eta_t[:, 3], eta_t[:, 4], eta_t[:, 5])
    return nu_t[:, :3] + world_to_body(current, quat)


def water_velocity_body(
    eta: Tensor,
    nu: Tensor,
    current_ned_mps: Tensor,
) -> Tensor:
    """Return body-frame water-relative velocity ``nu - current_body``."""

    eta_t = torch.as_tensor(eta)
    nu_t = torch.as_tensor(nu)
    current = torch.as_tensor(current_ned_mps)
    if eta_t.ndim != 2 or eta_t.shape[1] != 6 or nu_t.shape != eta_t.shape:
        raise ValueError("eta and nu must have shape (N, 6)")
    if current.shape != (eta_t.shape[0], 3):
        raise ValueError("current_ned_mps must have shape (N, 3)")
    quat = quaternion_from_euler(eta_t[:, 3], eta_t[:, 4], eta_t[:, 5])
    return nu_t[:, :3] - world_to_body(current, quat)


def ground_velocity_body_for_contract(
    eta: Tensor,
    nu: Tensor,
    current_ned_mps: Tensor,
    velocity_contract: str,
) -> Tensor:
    """Return ground velocity according to the configured plant contract."""

    if velocity_contract == "legacy_water_nu":
        return legacy_ground_velocity_body(eta, nu, current_ned_mps)
    if velocity_contract == "nu_ground":
        return ground_velocity_body(eta, nu, current_ned_mps)
    raise ValueError(f"unsupported velocity contract {velocity_contract!r}")


__all__ = [
    "body_to_world",
    "body_x_direction",
    "euler_from_quaternion",
    "ground_velocity_body",
    "ground_velocity_body_for_contract",
    "legacy_ground_velocity_body",
    "quaternion_from_direction",
    "quaternion_from_euler",
    "water_velocity_body",
    "world_to_body",
    "wrap_angle",
    "yaw_pitch",
]
