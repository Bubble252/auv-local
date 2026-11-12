"""Path-relative PC observation blocks for the S15-P1 pilot.

The path block is deliberately separate from the 12-D actor state.  It is
encoded by its own encoder and fused inside ``z_t``; it never enters the raw
history token or DADE.  All features are computed from the navigation position
projected onto the reference path, never from simulator truth or the true
current field.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from .geometry import wrap_angle
from .path import BatchedPathProjection

PC_PATH_DIM_V1 = 8
PC_PATH_DIM_ENDPOINT = 12
PC_PATH_DIMS = (PC_PATH_DIM_V1, PC_PATH_DIM_ENDPOINT)
PC_PATH_FEATURE_NAMES_V1 = (
    "signed_cross_track_m",
    "signed_vertical_error_m",
    "tangent_heading_error_rad",
    "tangent_pitch_error_rad",
    "progress_ratio",
    "remaining_ratio",
    "curvature_1_per_m",
    "lookahead_heading_delta_rad",
)
PC_PATH_ENDPOINT_FEATURE_NAMES = (
    "goal_distance_proximity",
    "goal_tangent_offset_gated",
    "goal_heading_error_gated",
    "goal_pitch_error_gated",
)
PC_PATH_FEATURE_NAMES = PC_PATH_FEATURE_NAMES_V1
PC_PATH_FEATURE_NAMES_BY_DIM = {
    PC_PATH_DIM_V1: PC_PATH_FEATURE_NAMES_V1,
    PC_PATH_DIM_ENDPOINT: PC_PATH_FEATURE_NAMES_V1 + PC_PATH_ENDPOINT_FEATURE_NAMES,
}
PC_PATH_FEATURE_SCALES_V1 = (
    5.0,
    5.0,
    math.pi,
    math.pi / 2.0,
    1.0,
    1.0,
    1.0,
    math.pi,
)
PC_PATH_ENDPOINT_FEATURE_SCALES = (
    1.0,
    5.0,
    math.pi,
    math.pi / 2.0,
)
PC_PATH_FEATURE_SCALES = PC_PATH_FEATURE_SCALES_V1
PC_PATH_FEATURE_SCALES_BY_DIM = {
    PC_PATH_DIM_V1: PC_PATH_FEATURE_SCALES_V1,
    PC_PATH_DIM_ENDPOINT: PC_PATH_FEATURE_SCALES_V1 + PC_PATH_ENDPOINT_FEATURE_SCALES,
}
PC_PATH_DIM = PC_PATH_DIM_V1
PC_PATH_BLOCK_NAMES = (
    "path_geom",
    "path_context",
    "path_velocity",
    "endpoint_context",
)
PC_PATH_BLOCK_FEATURE_INDICES: dict[str, tuple[int, ...]] = {
    "path_geom": (0, 1, 2, 3),
    "path_context": (4, 5, 6, 7),
    # P1 keeps the optional path-velocity block closed.  Reserving the block
    # name now avoids changing the schema when it is added in a later stage.
    "path_velocity": (),
    "endpoint_context": (8, 9, 10, 11),
}
PC_PATH_DEFAULT_BLOCKS = ("path_geom", "path_context")
PC_PATH_ENDPOINT_BLOCKS = ("path_geom", "path_context", "endpoint_context")


def pc_path_dim_for_blocks(blocks: Iterable[str]) -> int:
    active = set(validate_path_blocks(blocks))
    return PC_PATH_DIM_ENDPOINT if "endpoint_context" in active else PC_PATH_DIM_V1


def pc_path_feature_names_for_dim(path_dim: int) -> tuple[str, ...]:
    try:
        return PC_PATH_FEATURE_NAMES_BY_DIM[int(path_dim)]
    except KeyError as exc:
        raise ValueError(f"path_dim must be one of {PC_PATH_DIMS}") from exc


def pc_path_feature_scales_for_dim(path_dim: int) -> tuple[float, ...]:
    try:
        return PC_PATH_FEATURE_SCALES_BY_DIM[int(path_dim)]
    except KeyError as exc:
        raise ValueError(f"path_dim must be one of {PC_PATH_DIMS}") from exc


def validate_path_blocks(blocks: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(item) for item in blocks)
    unknown = sorted(set(values) - set(PC_PATH_BLOCK_NAMES))
    if unknown:
        raise ValueError(f"unknown path observation blocks: {unknown}")
    if len(set(values)) != len(values):
        raise ValueError("path observation blocks must not contain duplicates")
    if "path_velocity" in values and not PC_PATH_BLOCK_FEATURE_INDICES["path_velocity"]:
        raise ValueError("path_velocity has no P1 feature columns and must stay closed")
    return values


def path_block_column_mask(blocks: Iterable[str], *, path_dim: int | None = None) -> Tensor:
    active = set(validate_path_blocks(blocks))
    inferred_dim = pc_path_dim_for_blocks(active)
    dim = inferred_dim if path_dim is None else int(path_dim)
    if dim not in PC_PATH_DIMS:
        raise ValueError(f"path_dim must be one of {PC_PATH_DIMS}")
    if dim == PC_PATH_DIM_V1 and "endpoint_context" in active:
        raise ValueError("endpoint_context requires path_dim=12")
    columns = [False] * dim
    for name, indices in PC_PATH_BLOCK_FEATURE_INDICES.items():
        if name in active:
            for index in indices:
                columns[index] = True
    return torch.tensor(columns, dtype=torch.bool)


def pc_path_schema_hash(
    blocks: Iterable[str] = PC_PATH_DEFAULT_BLOCKS,
    *,
    path_dim: int | None = None,
) -> str:
    validated = validate_path_blocks(blocks)
    dim = pc_path_dim_for_blocks(validated) if path_dim is None else int(path_dim)
    if dim not in PC_PATH_DIMS:
        raise ValueError(f"path_dim must be one of {PC_PATH_DIMS}")
    if dim == PC_PATH_DIM_V1 and "endpoint_context" in validated:
        raise ValueError("endpoint_context requires path_dim=12")
    payload = {
        "version": "pc_path_v1" if dim == PC_PATH_DIM_V1 else "pc_path_v2_endpoint",
        "dim": dim,
        "features": pc_path_feature_names_for_dim(dim),
        "scales": pc_path_feature_scales_for_dim(dim),
        "blocks": sorted(validated),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalize_path_features(features: Tensor, *, path_dim: int | None = None) -> Tensor:
    value = torch.as_tensor(features)
    if value.ndim != 2:
        raise ValueError("path features must have shape (N, path_dim)")
    dim = int(value.shape[1]) if path_dim is None else int(path_dim)
    if value.shape[1] != dim or dim not in PC_PATH_DIMS:
        raise ValueError(f"path features must have shape (N, path_dim), path_dim in {PC_PATH_DIMS}")
    scales = value.new_tensor(pc_path_feature_scales_for_dim(dim))
    return value / scales


def build_pc_path_features(
    projection: BatchedPathProjection,
    *,
    attitude: Tensor,
    path: Any,
    position_ned_m: Tensor | None = None,
    path_blocks: Iterable[str] = PC_PATH_DEFAULT_BLOCKS,
    endpoint_enter_m: float = 20.0,
    endpoint_full_m: float = 5.0,
    lookahead_m: float = 3.0,
) -> Tensor:
    """Build the versioned path block from ``p_nav`` projection and sensor attitude."""

    blocks = validate_path_blocks(path_blocks)
    tangent = torch.as_tensor(projection.tangent_ned)
    position = torch.as_tensor(attitude)
    if tangent.ndim != 2 or tangent.shape[1] != 3:
        raise ValueError("projection tangent must have shape (N, 3)")
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("attitude must have shape (N, 3)")
    if projection.progress_m.shape != (tangent.shape[0],):
        raise ValueError("projection progress must have shape (N,)")
    if lookahead_m <= 0.0:
        raise ValueError("path lookahead must be positive")

    horizontal = torch.linalg.norm(tangent[:, :2], dim=1)
    tangent_heading = torch.atan2(tangent[:, 1], tangent[:, 0])
    tangent_pitch = torch.atan2(-tangent[:, 2], horizontal)
    heading_error = wrap_angle(tangent_heading - position[:, 2])
    pitch_error = wrap_angle(tangent_pitch - position[:, 1])

    total = path.total_length_m_for(tangent.shape[0])
    progress_ratio = torch.clamp(projection.progress_m / torch.clamp(total, min=1e-9), 0.0, 1.0)
    remaining_ratio = 1.0 - progress_ratio

    lookahead_s = torch.clamp(projection.progress_m + lookahead_m, max=total)
    lookahead_point = path.point_at_s(lookahead_s)
    delta = lookahead_point - projection.point_ned_m
    delta_horizontal = torch.linalg.norm(delta[:, :2], dim=1)
    lookahead_heading = torch.where(
        delta_horizontal > 1e-9,
        torch.atan2(delta[:, 1], delta[:, 0]),
        tangent_heading,
    )
    lookahead_delta = wrap_angle(lookahead_heading - tangent_heading)

    features = [
        projection.horizontal_error_m,
        projection.vertical_error_m,
        heading_error,
        pitch_error,
        progress_ratio,
        remaining_ratio,
        projection.curvature_1_per_m,
        lookahead_delta,
    ]
    if "endpoint_context" in blocks:
        if position_ned_m is None:
            raise ValueError("endpoint_context requires position_ned_m")
        if not math.isfinite(endpoint_enter_m) or endpoint_enter_m <= 0.0:
            raise ValueError("endpoint_enter_m must be finite and positive")
        if not math.isfinite(endpoint_full_m) or endpoint_full_m <= 0.0:
            raise ValueError("endpoint_full_m must be finite and positive")
        if endpoint_full_m >= endpoint_enter_m:
            raise ValueError("endpoint_full_m must be less than endpoint_enter_m")
        nav_position = torch.as_tensor(position_ned_m)
        if nav_position.shape != (tangent.shape[0], 3):
            raise ValueError("position_ned_m must have shape (N, 3)")
        count = tangent.shape[0]
        goal = path.goal_ned_m_for(count)
        goal_tangent = path.goal_tangent_ned_for(count)
        delta = goal - nav_position
        goal_distance = torch.linalg.norm(delta, dim=1)
        signed_tangent_offset = torch.sum((nav_position - goal) * goal_tangent, dim=1)
        horizontal = torch.linalg.norm(delta[:, :2], dim=1)
        goal_heading = torch.where(
            horizontal > 1e-9,
            torch.atan2(delta[:, 1], delta[:, 0]),
            position[:, 2],
        )
        goal_pitch = torch.where(
            goal_distance > 1e-9,
            torch.atan2(-delta[:, 2], horizontal),
            position[:, 1],
        )
        s = torch.clamp(
            (endpoint_enter_m - goal_distance) / (endpoint_enter_m - endpoint_full_m),
            0.0,
            1.0,
        )
        endpoint_gate = s * s * (3.0 - 2.0 * s)
        distance_ratio = torch.clamp(goal_distance / endpoint_enter_m, 0.0, 1.0)
        features.extend(
            [
                endpoint_gate * (1.0 - distance_ratio),
                endpoint_gate * signed_tangent_offset,
                endpoint_gate * wrap_angle(goal_heading - position[:, 2]),
                endpoint_gate * wrap_angle(goal_pitch - position[:, 1]),
            ]
        )
    return torch.stack(features, dim=1)


__all__ = [
    "PC_PATH_BLOCK_FEATURE_INDICES",
    "PC_PATH_BLOCK_NAMES",
    "PC_PATH_DEFAULT_BLOCKS",
    "PC_PATH_DIM",
    "PC_PATH_DIM_ENDPOINT",
    "PC_PATH_DIM_V1",
    "PC_PATH_DIMS",
    "PC_PATH_ENDPOINT_BLOCKS",
    "PC_PATH_ENDPOINT_FEATURE_NAMES",
    "PC_PATH_ENDPOINT_FEATURE_SCALES",
    "PC_PATH_FEATURE_NAMES",
    "PC_PATH_FEATURE_NAMES_BY_DIM",
    "PC_PATH_FEATURE_NAMES_V1",
    "PC_PATH_FEATURE_SCALES",
    "PC_PATH_FEATURE_SCALES_BY_DIM",
    "PC_PATH_FEATURE_SCALES_V1",
    "build_pc_path_features",
    "normalize_path_features",
    "path_block_column_mask",
    "pc_path_dim_for_blocks",
    "pc_path_feature_names_for_dim",
    "pc_path_feature_scales_for_dim",
    "pc_path_schema_hash",
    "validate_path_blocks",
]
