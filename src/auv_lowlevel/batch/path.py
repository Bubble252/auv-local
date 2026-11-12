"""Tensorized path projection for the batched low-level environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from auv_lowlevel.path import ReferencePath


@dataclass(frozen=True)
class BatchedPathProjection:
    """Vectorized equivalent of :class:`auv_lowlevel.path.PathProjection`."""

    segment_index: Tensor
    progress_m: Tensor
    point_ned_m: Tensor
    tangent_ned: Tensor
    lateral_axis_ned: Tensor
    vertical_axis_ned: Tensor
    horizontal_error_m: Tensor
    vertical_error_m: Tensor
    error_3d_m: Tensor
    curvature_1_per_m: Tensor


class BatchedReferencePath:
    """Fixed polyline retained as tensors and projected over an environment batch."""

    def __init__(
        self,
        points_ned_m: Tensor,
        cumulative_s_m: Tensor,
        *,
        name: str = "batched_reference",
        speed_mps: float = 1.2,
    ) -> None:
        points = torch.as_tensor(points_ned_m)
        cumulative = torch.as_tensor(cumulative_s_m)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("points_ned_m must have shape (N, 3) with N >= 2")
        if cumulative.shape != (points.shape[0],):
            raise ValueError("cumulative_s_m must match the number of points")
        if points.dtype != cumulative.dtype:
            raise ValueError("points and cumulative arclength must share a dtype")
        if not points.is_floating_point():
            raise ValueError("points_ned_m must be floating point")
        if not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(cumulative).all()):
            raise ValueError("path tensors must contain finite values")
        if float(cumulative[0]) != 0.0 or not bool((cumulative[1:] > cumulative[:-1]).all()):
            raise ValueError("cumulative_s_m must start at zero and be strictly increasing")
        if float(cumulative[-1]) <= 0.0:
            raise ValueError("path length must be positive")
        self.points_ned_m = points
        self.cumulative_s_m = cumulative
        self.name = name
        self.speed_mps = float(speed_mps)
        self._starts = points[:-1]
        differences = points[1:] - points[:-1]
        geometric_lengths = torch.linalg.norm(differences, dim=1)
        self._tangents = differences / geometric_lengths[:, None]
        # The scalar reference clips ``along`` with cumulative arclength differences.
        # Keeping that exact value preserves the progress tie-break at shared vertices.
        self._segment_lengths = cumulative[1:] - cumulative[:-1]
        self._total_length_tensor = cumulative[-1]
        self._total_length_m = float(self._total_length_tensor)
        self._minimum_segment_length_m = float(self._segment_lengths.min())

    @property
    def device(self) -> torch.device:
        return self.points_ned_m.device

    @property
    def dtype(self) -> torch.dtype:
        return self.points_ned_m.dtype

    @property
    def total_length_m(self) -> float:
        """Host-side path length for diagnostics and legacy callers."""

        return self._total_length_m

    @property
    def total_length_m_tensor(self) -> Tensor:
        """Device-resident path length for use inside tensorized hot paths."""

        return self._total_length_tensor

    def required_search_radius(self, max_offset_m: float) -> int:
        """Return a conservative segment window covering an offset near the start."""

        if max_offset_m < 0.0:
            raise ValueError("max_offset_m must be non-negative")
        return max(1, math.ceil(max_offset_m / self._minimum_segment_length_m) + 2)

    @property
    def start_ned_m(self) -> Tensor:
        return self.points_ned_m[0]

    @property
    def goal_ned_m(self) -> Tensor:
        return self.points_ned_m[-1]

    @property
    def start_tangent_ned(self) -> Tensor:
        return self._tangents[0]

    @property
    def goal_tangent_ned(self) -> Tensor:
        return self._tangents[-1]

    def start_ned_m_for(self, count: int) -> Tensor:
        return self.start_ned_m[None, :].expand(count, 3)

    def start_tangent_ned_for(self, count: int) -> Tensor:
        return self.start_tangent_ned[None, :].expand(count, 3)

    def goal_tangent_ned_for(self, count: int) -> Tensor:
        return self.goal_tangent_ned[None, :].expand(count, 3)

    def goal_ned_m_for(self, count: int) -> Tensor:
        return self.goal_ned_m[None, :].expand(count, 3)

    def total_length_m_for(self, count: int) -> Tensor:
        return self.total_length_m_tensor.expand(count)

    @classmethod
    def from_reference_path(
        cls,
        path: ReferencePath,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> BatchedReferencePath:
        return cls(
            torch.as_tensor(path.points_ned_m, device=device, dtype=dtype),
            torch.as_tensor(path.cumulative_s_m, device=device, dtype=dtype),
            name=path.name,
            speed_mps=path.speed_mps,
        )

    def point_at_s(self, s_m: Tensor) -> Tensor:
        """Return path points for a tensor of arclengths."""

        s = torch.as_tensor(s_m, device=self.device, dtype=self.dtype)
        if s.ndim != 1:
            raise ValueError("s_m must have shape (N,)")
        target = torch.clamp(
            s,
            min=torch.zeros_like(s),
            max=self.total_length_m_tensor.expand_as(s),
        )
        index = torch.searchsorted(self.cumulative_s_m, target, right=True) - 1
        index = torch.clamp(index, 0, self.points_ned_m.shape[0] - 2)
        start_s = self.cumulative_s_m[index]
        length = self.cumulative_s_m[index + 1] - start_s
        fraction = (target - start_s) / length
        start = self.points_ned_m[index]
        return start + fraction[:, None] * (self.points_ned_m[index + 1] - start)

    def curvature_at_s(self, s_m: Tensor) -> Tensor:
        """Return the same discrete curvature as ``ReferencePath.curvature_at_s``."""

        s = torch.as_tensor(s_m, device=self.device, dtype=self.dtype)
        if s.ndim != 1:
            raise ValueError("s_m must have shape (N,)")
        target = torch.clamp(
            s,
            min=torch.zeros_like(s),
            max=self.total_length_m_tensor.expand_as(s),
        )
        index = torch.searchsorted(self.cumulative_s_m, target, right=True) - 1
        index = torch.clamp(index, 1, self._tangents.shape[0] - 2)
        before = self._tangents[index - 1]
        after = self._tangents[index]
        turn = torch.acos(torch.clamp(torch.sum(before * after, dim=1), -1.0, 1.0))
        arc_length = self.cumulative_s_m[index + 1] - self.cumulative_s_m[index - 1]
        return turn / torch.clamp(arc_length, min=1e-12)

    def project(
        self,
        position_ned_m: Tensor,
        *,
        segment_hint: Tensor | None = None,
        search_radius: int = 8,
    ) -> BatchedPathProjection:
        """Project ``(N, 3)`` positions onto the polyline.

        The scalar reference chooses the lowest distance and, within ``1e-12``,
        the larger progress. That tie-break matters at shared segment vertices
        and is reproduced explicitly.

        When ``segment_hint`` is provided, only the segments within
        ``search_radius`` of each hint are evaluated. The caller must provide a
        hint close to the true segment; the closed-loop environment uses the
        previous projection segment and performs a full search on reset.
        """

        positions = torch.as_tensor(position_ned_m, device=self.device, dtype=self.dtype)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("position_ned_m must have shape (N, 3)")
        if segment_hint is None:
            return self._project_global(positions)
        if search_radius < 1:
            raise ValueError("search_radius must be positive")
        hints = torch.as_tensor(segment_hint, device=self.device, dtype=torch.long)
        if hints.shape != (positions.shape[0],):
            raise ValueError("segment_hint must have shape (N,)")
        return self._project_window(positions, hints, search_radius=search_radius)

    def _project_global(self, positions: Tensor) -> BatchedPathProjection:
        """Project against every segment in the fixed path."""

        delta_from_start = positions[:, None, :] - self._starts[None, :, :]
        raw_along = torch.sum(
            delta_from_start * self._tangents[None, :, :], dim=2
        )
        along = torch.clamp(raw_along, min=0.0)
        along = torch.minimum(along, self._segment_lengths[None, :])
        position_minus_start_squared = torch.sum(delta_from_start * delta_from_start, dim=2)
        distance_sq = position_minus_start_squared - 2.0 * along * raw_along + along * along
        distance_sq = torch.clamp(distance_sq, min=0.0)
        progress = self.cumulative_s_m[:-1][None, :] + along

        minimum = torch.min(distance_sq, dim=1).values
        tied = distance_sq <= (minimum + 1e-12)[:, None]
        best_tied_progress = torch.where(tied, progress, torch.full_like(progress, -torch.inf))
        index = torch.argmax(best_tied_progress, dim=1)
        row = torch.arange(positions.shape[0], device=self.device)
        best_progress = progress[row, index]
        best_along = along[row, index]
        return self._assemble_projection(
            positions=positions,
            segment_index=index,
            along_m=best_along,
            progress_m=best_progress,
        )

    def _project_window(
        self,
        positions: Tensor,
        hints: Tensor,
        *,
        search_radius: int,
    ) -> BatchedPathProjection:
        """Project against a small segment window around each hint."""

        offsets = torch.arange(
            -search_radius,
            search_radius + 1,
            device=self.device,
            dtype=torch.long,
        )
        candidate_indices = hints[:, None] + offsets[None, :]
        valid = (candidate_indices >= 0) & (candidate_indices < self._tangents.shape[0])
        candidate_indices = torch.clamp(
            candidate_indices, 0, self._tangents.shape[0] - 1
        )

        starts = self._starts[candidate_indices]
        tangents = self._tangents[candidate_indices]
        lengths = self._segment_lengths[candidate_indices]
        cumulative = self.cumulative_s_m[candidate_indices]
        delta_from_start = positions[:, None, :] - starts
        raw_along = torch.sum(delta_from_start * tangents, dim=2)
        along = torch.clamp(raw_along, min=0.0)
        along = torch.minimum(along, lengths)
        position_minus_start_squared = torch.sum(delta_from_start * delta_from_start, dim=2)
        distance_sq = position_minus_start_squared - 2.0 * along * raw_along + along * along
        distance_sq = torch.clamp(distance_sq, min=0.0)
        distance_sq = torch.where(valid, distance_sq, torch.full_like(distance_sq, torch.inf))
        progress = cumulative + along

        minimum = torch.min(distance_sq, dim=1).values
        tied = valid & (distance_sq <= (minimum + 1e-12)[:, None])
        best_tied_progress = torch.where(
            tied, progress, torch.full_like(progress, -torch.inf)
        )
        local_index = torch.argmax(best_tied_progress, dim=1)
        row = torch.arange(positions.shape[0], device=self.device)
        return self._assemble_projection(
            positions=positions,
            segment_index=candidate_indices[row, local_index],
            along_m=along[row, local_index],
            progress_m=progress[row, local_index],
        )

    def _assemble_projection(
        self,
        *,
        positions: Tensor,
        segment_index: Tensor,
        along_m: Tensor,
        progress_m: Tensor,
    ) -> BatchedPathProjection:
        tangent = self._tangents[segment_index]
        projection = self._starts[segment_index] + along_m[:, None] * tangent

        horizontal_norm = torch.linalg.norm(tangent[:, :2], dim=1)
        regular_lateral = torch.stack(
            (
                -tangent[:, 1] / horizontal_norm,
                tangent[:, 0] / horizontal_norm,
                torch.zeros_like(horizontal_norm),
            ),
            dim=1,
        )
        fallback_lateral = torch.stack(
            (
                torch.zeros_like(horizontal_norm),
                torch.ones_like(horizontal_norm),
                torch.zeros_like(horizontal_norm),
            ),
            dim=1,
        )
        lateral = torch.where((horizontal_norm > 1e-12)[:, None], regular_lateral, fallback_lateral)
        vertical = torch.cross(tangent, lateral, dim=1)
        vertical = vertical / torch.linalg.norm(vertical, dim=1, keepdim=True)

        offset = positions - projection
        return BatchedPathProjection(
            segment_index=segment_index,
            progress_m=progress_m,
            point_ned_m=projection,
            tangent_ned=tangent,
            lateral_axis_ned=lateral,
            vertical_axis_ned=vertical,
            horizontal_error_m=torch.sum(offset * lateral, dim=1),
            vertical_error_m=torch.sum(offset * vertical, dim=1),
            error_3d_m=torch.linalg.norm(offset, dim=1),
            curvature_1_per_m=self.curvature_at_s(progress_m),
        )


class BatchedReferencePathBank:
    """A fixed bank of equal-length paths with per-environment selection."""

    CUMULATIVE_ATOL_M = 1.0e-3
    CUMULATIVE_RTOL = 1.0e-5

    def __init__(
        self,
        paths: tuple[ReferencePath, ...] | list[ReferencePath],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if not paths:
            raise ValueError("path bank must contain at least one path")
        points = [torch.as_tensor(path.points_ned_m, device=device, dtype=dtype) for path in paths]
        cumulative = [
            torch.as_tensor(path.cumulative_s_m, device=device, dtype=dtype)
            for path in paths
        ]
        shape = points[0].shape
        if shape[0] < 2 or shape[1] != 3:
            raise ValueError("bank paths must have shape (N, 3) with N >= 2")
        if any(item.shape != shape for item in points):
            raise ValueError("bank paths must contain the same number of samples")
        if any(item.shape != cumulative[0].shape for item in cumulative):
            raise ValueError("bank paths must share arclength sample counts")
        if any(
            not torch.allclose(
                item,
                cumulative[0],
                atol=self.CUMULATIVE_ATOL_M,
                rtol=self.CUMULATIVE_RTOL,
            )
            for item in cumulative[1:]
        ):
            raise ValueError("bank paths must share the same cumulative arclength")
        self.points_ned_m = torch.stack(points, dim=0)
        self.cumulative_s_m = cumulative[0]
        self.speed_mps = float(paths[0].speed_mps)
        self.name = "batched_reference_path_bank"
        self._segment_lengths = self.cumulative_s_m[1:] - self.cumulative_s_m[:-1]
        self._minimum_segment_length_m = float(self._segment_lengths.min())

    @property
    def size(self) -> int:
        return int(self.points_ned_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.points_ned_m.device

    @property
    def dtype(self) -> torch.dtype:
        return self.points_ned_m.dtype

    def required_search_radius(self, max_offset_m: float) -> int:
        if max_offset_m < 0.0:
            raise ValueError("max_offset_m must be non-negative")
        return max(1, math.ceil(max_offset_m / self._minimum_segment_length_m) + 2)

    def select(self, indices: Tensor) -> BatchedReferencePathSet:
        selected = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        if selected.ndim != 1:
            raise ValueError("path bank indices must have shape (N,)")
        if bool((selected < 0).any()) or bool((selected >= self.size).any()):
            raise ValueError("path bank index is out of range")
        return BatchedReferencePathSet(
            self.points_ned_m[selected],
            self.cumulative_s_m,
            name=self.name,
            speed_mps=self.speed_mps,
        )


class BatchedReferencePathSet:
    """Per-environment paths with optional per-row cumulative arclength."""

    def __init__(
        self,
        points_ned_m: Tensor,
        cumulative_s_m: Tensor,
        *,
        name: str = "batched_reference_path_set",
        speed_mps: float = 1.2,
    ) -> None:
        points = torch.as_tensor(points_ned_m)
        cumulative = torch.as_tensor(cumulative_s_m)
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 2:
            raise ValueError("points_ned_m must have shape (N, P, 3) with P >= 2")
        if cumulative.shape == (points.shape[1],):
            cumulative = cumulative.unsqueeze(0).expand(points.shape[0], -1).contiguous()
        if cumulative.shape != (points.shape[0], points.shape[1]):
            raise ValueError("cumulative_s_m must have shape (P,) or (N, P)")
        if not points.is_floating_point() or not cumulative.is_floating_point():
            raise ValueError("path tensors must be floating point")
        if points.dtype != cumulative.dtype:
            raise ValueError("points and cumulative arclength must share a dtype")
        if not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(cumulative).all()):
            raise ValueError("path tensors must contain finite values")
        if not bool((cumulative[:, 0] == 0.0).all()) or not bool(
            (cumulative[:, 1:] > cumulative[:, :-1]).all()
        ):
            raise ValueError(
                "cumulative_s_m must start at zero and be strictly increasing per row"
            )
        self.points_ned_m = points
        self.cumulative_s_m = cumulative
        self.name = name
        self.speed_mps = float(speed_mps)
        differences = points[:, 1:] - points[:, :-1]
        lengths = torch.linalg.norm(differences, dim=2)
        if bool((lengths <= 1e-12).any()):
            raise ValueError("path set contains duplicate adjacent points")
        self._starts = points[:, :-1]
        self._tangents = differences / lengths[:, :, None]
        self._segment_lengths = cumulative[:, 1:] - cumulative[:, :-1]
        self._total_length_tensor = cumulative[:, -1]
        self._total_length_m = float(self._total_length_tensor.max())

    @property
    def num_paths(self) -> int:
        return int(self.points_ned_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.points_ned_m.device

    @property
    def dtype(self) -> torch.dtype:
        return self.points_ned_m.dtype

    @property
    def total_length_m(self) -> float:
        return self._total_length_m

    @property
    def total_length_m_tensor(self) -> Tensor:
        return self._total_length_tensor

    @property
    def start_ned_m(self) -> Tensor:
        return self.points_ned_m[:, 0]

    @property
    def start_tangent_ned(self) -> Tensor:
        return self._tangents[:, 0]

    @property
    def goal_tangent_ned(self) -> Tensor:
        return self._tangents[:, -1]

    @property
    def goal_ned_m(self) -> Tensor:
        return self.points_ned_m[:, -1]

    def start_ned_m_for(self, count: int) -> Tensor:
        if count != self.num_paths:
            raise ValueError("path-set size must match the environment batch")
        return self.start_ned_m

    def start_tangent_ned_for(self, count: int) -> Tensor:
        if count != self.num_paths:
            raise ValueError("path-set size must match the environment batch")
        return self.start_tangent_ned

    def goal_tangent_ned_for(self, count: int) -> Tensor:
        if count != self.num_paths:
            raise ValueError("path-set size must match the environment batch")
        return self.goal_tangent_ned

    def goal_ned_m_for(self, count: int) -> Tensor:
        if count != self.num_paths:
            raise ValueError("path-set size must match the environment batch")
        return self.goal_ned_m

    def total_length_m_for(self, count: int) -> Tensor:
        if count != self.num_paths:
            raise ValueError("path-set size must match the environment batch")
        return self.total_length_m_tensor

    def point_at_s(self, s_m: Tensor) -> Tensor:
        s = torch.as_tensor(s_m, device=self.device, dtype=self.dtype)
        if s.shape != (self.num_paths,):
            raise ValueError("s_m must have shape (N,) matching the path set")
        target = torch.minimum(torch.clamp(s, min=0.0), self._total_length_tensor)
        rows = torch.arange(self.num_paths, device=self.device)
        index = torch.searchsorted(self.cumulative_s_m, target[:, None], right=True) - 1
        index = torch.clamp(index, 0, self.points_ned_m.shape[1] - 2)
        index = index[:, 0]
        start_s = self.cumulative_s_m[rows, index]
        length = self.cumulative_s_m[rows, index + 1] - start_s
        fraction = (target - start_s) / length
        start = self.points_ned_m[rows, index]
        end = self.points_ned_m[rows, index + 1]
        return start + fraction[:, None] * (end - start)

    def curvature_at_s(self, s_m: Tensor) -> Tensor:
        s = torch.as_tensor(s_m, device=self.device, dtype=self.dtype)
        if s.shape != (self.num_paths,):
            raise ValueError("s_m must have shape (N,) matching the path set")
        target = torch.minimum(torch.clamp(s, min=0.0), self._total_length_tensor)
        index = torch.searchsorted(self.cumulative_s_m, target[:, None], right=True) - 1
        index = torch.clamp(index, 1, self._tangents.shape[1] - 2)
        index = index[:, 0]
        rows = torch.arange(self.num_paths, device=self.device)
        before = self._tangents[rows, index - 1]
        after = self._tangents[rows, index]
        turn = torch.acos(torch.clamp(torch.sum(before * after, dim=1), -1.0, 1.0))
        arc_length = (
            self.cumulative_s_m[rows, index + 1]
            - self.cumulative_s_m[rows, index - 1]
        )
        return turn / torch.clamp(arc_length, min=1e-12)

    def project(
        self,
        position_ned_m: Tensor,
        *,
        segment_hint: Tensor | None = None,
        search_radius: int = 8,
    ) -> BatchedPathProjection:
        positions = torch.as_tensor(position_ned_m, device=self.device, dtype=self.dtype)
        if positions.shape != (self.num_paths, 3):
            raise ValueError("position_ned_m must have shape (N, 3) matching the path set")
        if segment_hint is None:
            return self._project_global(positions)
        if search_radius < 1:
            raise ValueError("search_radius must be positive")
        hints = torch.as_tensor(segment_hint, device=self.device, dtype=torch.long)
        if hints.shape != (self.num_paths,):
            raise ValueError("segment_hint must have shape (N,)")
        return self._project_window(positions, hints, search_radius=search_radius)

    def _project_global(self, positions: Tensor) -> BatchedPathProjection:
        rows = torch.arange(self.num_paths, device=self.device)
        delta = positions[:, None, :] - self._starts
        raw_along = torch.sum(delta * self._tangents, dim=2)
        along = torch.clamp(raw_along, min=0.0)
        along = torch.minimum(along, self._segment_lengths)
        distance_sq = torch.sum(delta * delta, dim=2) - 2.0 * along * raw_along + along * along
        distance_sq = torch.clamp(distance_sq, min=0.0)
        progress = self.cumulative_s_m[:, :-1] + along
        minimum = torch.min(distance_sq, dim=1).values
        tied = distance_sq <= (minimum + 1e-12)[:, None]
        best_tied_progress = torch.where(tied, progress, torch.full_like(progress, -torch.inf))
        index = torch.argmax(best_tied_progress, dim=1)
        return self._assemble_projection(
            positions=positions,
            segment_index=index,
            along_m=along[rows, index],
            progress_m=progress[rows, index],
        )

    def _project_window(
        self, positions: Tensor, hints: Tensor, *, search_radius: int
    ) -> BatchedPathProjection:
        offsets = torch.arange(
            -search_radius, search_radius + 1, device=self.device, dtype=torch.long
        )
        candidate = hints[:, None] + offsets[None, :]
        valid = (candidate >= 0) & (candidate < self._tangents.shape[1])
        candidate = torch.clamp(candidate, 0, self._tangents.shape[1] - 1)
        rows = torch.arange(self.num_paths, device=self.device)[:, None]
        starts = self._starts[rows, candidate]
        tangents = self._tangents[rows, candidate]
        lengths = self._segment_lengths[rows, candidate]
        cumulative = self.cumulative_s_m[rows, candidate]
        delta = positions[:, None, :] - starts
        raw_along = torch.sum(delta * tangents, dim=2)
        along = torch.clamp(raw_along, min=0.0)
        along = torch.minimum(along, lengths)
        distance_sq = torch.sum(delta * delta, dim=2) - 2.0 * along * raw_along + along * along
        distance_sq = torch.clamp(distance_sq, min=0.0)
        distance_sq = torch.where(valid, distance_sq, torch.full_like(distance_sq, torch.inf))
        progress = cumulative + along
        minimum = torch.min(distance_sq, dim=1).values
        tied = valid & (distance_sq <= (minimum + 1e-12)[:, None])
        best_tied_progress = torch.where(tied, progress, torch.full_like(progress, -torch.inf))
        local = torch.argmax(best_tied_progress, dim=1)
        row = torch.arange(self.num_paths, device=self.device)
        segment_index = candidate[row, local]
        return self._assemble_projection(
            positions=positions,
            segment_index=segment_index,
            along_m=along[row, local],
            progress_m=progress[row, local],
        )

    def _assemble_projection(
        self,
        *,
        positions: Tensor,
        segment_index: Tensor,
        along_m: Tensor,
        progress_m: Tensor,
    ) -> BatchedPathProjection:
        rows = torch.arange(self.num_paths, device=self.device)
        tangent = self._tangents[rows, segment_index]
        projection = self._starts[rows, segment_index] + along_m[:, None] * tangent
        horizontal_norm = torch.linalg.norm(tangent[:, :2], dim=1)
        regular_lateral = torch.stack(
            (
                -tangent[:, 1] / horizontal_norm,
                tangent[:, 0] / horizontal_norm,
                torch.zeros_like(horizontal_norm),
            ),
            dim=1,
        )
        fallback_lateral = torch.stack(
            (
                torch.zeros_like(horizontal_norm),
                torch.ones_like(horizontal_norm),
                torch.zeros_like(horizontal_norm),
            ),
            dim=1,
        )
        lateral = torch.where(
            (horizontal_norm > 1e-12)[:, None], regular_lateral, fallback_lateral
        )
        vertical = torch.cross(tangent, lateral, dim=1)
        vertical = vertical / torch.linalg.norm(vertical, dim=1, keepdim=True)
        offset = positions - projection
        return BatchedPathProjection(
            segment_index=segment_index,
            progress_m=progress_m,
            point_ned_m=projection,
            tangent_ned=tangent,
            lateral_axis_ned=lateral,
            vertical_axis_ned=vertical,
            horizontal_error_m=torch.sum(offset * lateral, dim=1),
            vertical_error_m=torch.sum(offset * vertical, dim=1),
            error_3d_m=torch.linalg.norm(offset, dim=1),
            curvature_1_per_m=self.curvature_at_s(progress_m),
        )


__all__ = [
    "BatchedPathProjection",
    "BatchedReferencePath",
    "BatchedReferencePathBank",
    "BatchedReferencePathSet",
]
