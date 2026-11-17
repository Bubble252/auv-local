"""Privileged critic observations for the Goal182 dual-path policy.

The actor path block is deployable and deliberately narrow.  The critic path
block is a separate, fixed-width vector built from simulator truth and is
never mixed into the actor observation, actor history, DADE input, or command.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .geometry import quaternion_from_euler, world_to_body
from .pc_normalization import PCNormalization
from .pc_path import build_pc_path_features, normalize_path_features
from .pc_privileged_schema import (
    PC_CRITIC_PRIV_BLOCK_NAMES,
    PC_CRITIC_PRIV_DIM,
    PC_CRITIC_PRIV_FEATURE_NAMES,
    PC_CRITIC_PRIV_GROUP_SLICES,
)


def _safe_ratio(value: Tensor, limit: float) -> Tensor:
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("safety limit must be finite and positive")
    return torch.clamp(1.0 - torch.abs(value) / limit, min=0.0, max=1.0)


def build_pc_critic_privileged(
    *,
    env: Any,
    observation: Any,
    projection: Any,
    normalization: PCNormalization,
    raw_features: bool = False,
) -> Tensor:
    """Build the 30-D privileged critic vector.

    ``projection`` must be the true-``eta`` route projection, not the actor's
    ``p_nav`` projection.  The function is intentionally read-only with respect
    to environment state. When ``raw_features`` is true, dataset-dependent
    fixed scales are not applied, so an empirical normalizer owns the
    normalization contract. Unit construction (actuator limits, sensor-age
    caps, and safety ratios) remains unchanged.
    """

    if projection is None:
        raise ValueError("privileged critic route truth requires a true-eta projection")
    eta = env.eta
    if eta is None or eta.ndim != 2 or eta.shape[1] < 6:
        raise RuntimeError("privileged critic requires initialized true eta")
    count = eta.shape[0]
    attitude = eta[:, 3:6]

    route_v1 = build_pc_path_features(
        projection,
        attitude=attitude,
        path=env.path,
        path_dim=8,
    )
    route_indices = (0, 1, 2, 3, 6, 7)
    route = (
        route_v1[:, route_indices]
        if raw_features
        else normalize_path_features(route_v1, path_dim=8)[:, route_indices]
    )

    actual = env.actuator
    requested = env.previous_action_normalized
    if actual is None or requested is None:
        raise RuntimeError("privileged critic requires actuator history")
    max_rudder = float(env.limits.max_rudder_angle_rad)
    max_stern = float(env.limits.max_stern_plane_angle_rad)
    max_rpm = float(env.limits.max_propeller_rpm)
    if max_rudder <= 0.0 or max_stern <= 0.0 or max_rpm <= 0.0:
        raise ValueError("actuator limits must be positive")
    actual_norm = torch.stack(
        (
            actual[:, 0] / max_rudder,
            actual[:, 1] / max_stern,
            actual[:, 2] / max_rpm,
        ),
        dim=1,
    )
    tracking_residual = requested - actual_norm
    saturation = env._material_saturation_ema.to(device=eta.device, dtype=eta.dtype)
    actuator = torch.cat(
        (actual_norm, requested, tracking_residual, saturation[:, None]),
        dim=1,
    )

    true_current = env.current_ned_mps
    if true_current is None:
        raise RuntimeError("privileged critic requires true current")
    true_current_value = (
        true_current if raw_features else normalization.normalize_current(true_current)
    )
    current_body = world_to_body(
        true_current,
        quaternion_from_euler(eta[:, 3], eta[:, 4], eta[:, 5]),
    )
    current_innovation = current_body - observation.actor_current_body
    if not raw_features:
        current_innovation = normalization.normalize_current(current_innovation)
    disturbance = torch.cat((true_current_value, current_innovation), dim=1)

    sensor = observation.sensor
    nav_innovation = eta[:, :3] - sensor.p_nav
    if not raw_features:
        nav_innovation = normalization.normalize_actor_state(
            torch.cat(
                (
                    nav_innovation,
                    torch.zeros((count, 9), device=eta.device, dtype=eta.dtype),
                ),
                dim=1,
            )
        )[:, :3]
    depth_margin = torch.ones((count,), device=eta.device, dtype=eta.dtype)
    termination = env.config.termination
    if termination.min_depth_m is not None and termination.max_depth_m is not None:
        lower = eta[:, 2] - float(termination.min_depth_m)
        upper = float(termination.max_depth_m) - eta[:, 2]
        span = max(float(termination.max_depth_m) - float(termination.min_depth_m), 1e-6)
        depth_margin = torch.clamp(torch.minimum(lower, upper) / span, min=0.0, max=1.0)
    roll_margin = _safe_ratio(eta[:, 3], float(termination.phi_term_rad))
    pitch_margin = _safe_ratio(eta[:, 4], math.pi / 2.0)
    roll_rate_margin = _safe_ratio(env.nu[:, 3], float(termination.p_term_rad_s))
    safety_margin = torch.minimum(
        torch.minimum(depth_margin, roll_margin),
        torch.minimum(pitch_margin, roll_rate_margin),
    )
    sensor_safety = torch.cat(
        (
            torch.clamp(sensor.usbl_age_s / 10.0, min=0.0, max=1.0)[:, None],
            torch.clamp(sensor.dvl_bottom_age_s / 10.0, min=0.0, max=1.0)[:, None],
            sensor.usbl_valid[:, None],
            sensor.bottom_lock_valid[:, None],
            nav_innovation,
            safety_margin[:, None],
        ),
        dim=1,
    )
    features = torch.cat((route, actuator, disturbance, sensor_safety), dim=1)
    if features.shape != (count, PC_CRITIC_PRIV_DIM):
        raise RuntimeError(
            f"privileged critic feature width mismatch: {tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("privileged critic features contain NaN or Inf")
    return features


__all__ = [
    "PC_CRITIC_PRIV_BLOCK_NAMES",
    "PC_CRITIC_PRIV_DIM",
    "PC_CRITIC_PRIV_FEATURE_NAMES",
    "PC_CRITIC_PRIV_GROUP_SLICES",
    "build_pc_critic_privileged",
]
