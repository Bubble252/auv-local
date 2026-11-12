from __future__ import annotations

import math
from dataclasses import dataclass, replace
from math import acos, atan2, hypot

import numpy as np
from auv_core.attitude import forward_quaternion
from auv_core.contracts import FloatArray, Trajectory

from .config import ReferenceConfig


def _interp_rows(x: FloatArray, xp: FloatArray, fp: FloatArray) -> FloatArray:
    return np.column_stack([np.interp(x, xp, fp[:, axis]) for axis in range(fp.shape[1])])


@dataclass(frozen=True)
class PathProjection:
    segment_index: int
    progress_m: float
    point_ned_m: FloatArray
    tangent_ned: FloatArray
    lateral_axis_ned: FloatArray
    vertical_axis_ned: FloatArray
    horizontal_error_m: float
    vertical_error_m: float
    error_3d_m: float
    curvature_1_per_m: float


@dataclass(frozen=True)
class ReferencePath:
    name: str
    points_ned_m: FloatArray
    cumulative_s_m: FloatArray
    speed_mps: float

    def __post_init__(self) -> None:
        points = np.asarray(self.points_ned_m, dtype=np.float64)
        cumulative = np.asarray(self.cumulative_s_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("points_ned_m must have shape (N, 3) with N >= 2")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_ned_m must contain finite values")
        if cumulative.shape != (points.shape[0],):
            raise ValueError("cumulative_s_m must match the number of points")
        if not np.all(np.isfinite(cumulative)) or cumulative[0] != 0.0:
            raise ValueError("cumulative_s_m must be finite and start at zero")
        if np.any(np.diff(cumulative) <= 0.0):
            raise ValueError("cumulative_s_m must be strictly increasing")
        if not np.allclose(
            np.linalg.norm(np.diff(points, axis=0), axis=1),
            np.diff(cumulative),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("cumulative_s_m must equal the polyline arclength")
        if not np.isfinite(self.speed_mps) or self.speed_mps <= 0.0:
            raise ValueError("speed_mps must be finite and positive")
        object.__setattr__(self, "points_ned_m", points)
        object.__setattr__(self, "cumulative_s_m", cumulative)

    @property
    def total_length_m(self) -> float:
        return float(self.cumulative_s_m[-1])

    @property
    def tangents_ned(self) -> FloatArray:
        differences = np.diff(self.points_ned_m, axis=0)
        return differences / np.linalg.norm(differences, axis=1)[:, None]

    @property
    def start_ned_m(self) -> FloatArray:
        return self.points_ned_m[0].copy()

    @property
    def goal_ned_m(self) -> FloatArray:
        return self.points_ned_m[-1].copy()

    def point_at_s(self, s_m: float) -> FloatArray:
        s_value = float(np.clip(s_m, 0.0, self.total_length_m))
        index = int(np.searchsorted(self.cumulative_s_m, s_value, side="right") - 1)
        index = min(max(index, 0), self.points_ned_m.shape[0] - 2)
        segment_length = float(self.cumulative_s_m[index + 1] - self.cumulative_s_m[index])
        fraction = (s_value - self.cumulative_s_m[index]) / segment_length
        return self.points_ned_m[index] + fraction * (
            self.points_ned_m[index + 1] - self.points_ned_m[index]
        )

    def curvature_at_s(self, s_m: float) -> float:
        s_value = float(np.clip(s_m, 0.0, self.total_length_m))
        index = int(np.searchsorted(self.cumulative_s_m, s_value, side="right") - 1)
        index = min(max(index, 1), self.tangents_ned.shape[0] - 2)
        before = self.tangents_ned[index - 1]
        after = self.tangents_ned[index]
        turn = acos(float(np.clip(np.dot(before, after), -1.0, 1.0)))
        arc_length = float(self.cumulative_s_m[index + 1] - self.cumulative_s_m[index - 1])
        return turn / max(arc_length, 1e-12)

    def project(self, position_ned_m: FloatArray) -> PathProjection:
        position = np.asarray(position_ned_m, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_ned_m must be a finite shape-(3,) vector")

        points = self.points_ned_m
        tangents = self.tangents_ned
        best: tuple[float, float, int, FloatArray] | None = None
        for index, (start, tangent) in enumerate(zip(points[:-1], tangents, strict=True)):
            length = float(self.cumulative_s_m[index + 1] - self.cumulative_s_m[index])
            along = float(np.clip(np.dot(position - start, tangent), 0.0, length))
            candidate = start + along * tangent
            distance_sq = float(np.dot(position - candidate, position - candidate))
            progress = float(self.cumulative_s_m[index] + along)
            if (
                best is None
                or distance_sq < best[0] - 1e-12
                or (abs(distance_sq - best[0]) <= 1e-12 and progress > best[1])
            ):
                best = (distance_sq, progress, index, candidate)
        assert best is not None
        _, progress, index, projection = best
        tangent = tangents[index]
        horizontal_norm = hypot(float(tangent[0]), float(tangent[1]))
        if horizontal_norm > 1e-12:
            lateral = np.array(
                [-tangent[1] / horizontal_norm, tangent[0] / horizontal_norm, 0.0],
                dtype=np.float64,
            )
        else:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        vertical = np.cross(tangent, lateral)
        vertical /= np.linalg.norm(vertical)

        offset = position - projection
        horizontal_error = float(np.dot(offset, lateral))
        vertical_error = float(np.dot(offset, vertical))
        return PathProjection(
            segment_index=index,
            progress_m=progress,
            point_ned_m=projection,
            tangent_ned=tangent,
            lateral_axis_ned=lateral,
            vertical_axis_ned=vertical,
            horizontal_error_m=horizontal_error,
            vertical_error_m=vertical_error,
            error_3d_m=float(np.linalg.norm(offset)),
            curvature_1_per_m=self.curvature_at_s(progress),
        )

    def trajectory(self, trajectory_id: str = "odarrl_control_reference") -> Trajectory:
        durations = self.cumulative_s_m / self.speed_mps
        tangents = np.vstack((self.tangents_ned, self.tangents_ned[-1]))
        quaternions = np.vstack([forward_quaternion(tangent) for tangent in tangents])
        linear_velocity = np.zeros((self.points_ned_m.shape[0], 3), dtype=np.float64)
        linear_velocity[:, 0] = self.speed_mps
        return Trajectory(
            trajectory_id=trajectory_id,
            t_s=durations,
            position_ned_m=self.points_ned_m.copy(),
            quat_xyzw=quaternions,
            linear_velocity_body_mps=linear_velocity,
            angular_velocity_body_rps=np.zeros_like(linear_velocity),
            feasibility="tracked",
        )

    def endpoint_heading_rad(self) -> float:
        tangent = self.tangents_ned[-1]
        return float(atan2(tangent[1], tangent[0]))

    def endpoint_pitch_rad(self) -> float:
        tangent = self.tangents_ned[-1]
        horizontal = hypot(float(tangent[0]), float(tangent[1]))
        return float(atan2(-tangent[2], horizontal))


def generate_sinusoidal_3d_path(config: ReferenceConfig) -> ReferencePath:
    dense_count = max(4001, int(np.ceil(config.length_m / 0.002)))
    parameter = np.linspace(0.0, 1.0, dense_count)
    raw: FloatArray = np.column_stack(
        (
            config.x_span_m * parameter,
            config.lateral_amplitude_m
            * np.sin(2.0 * np.pi * config.lateral_cycles * parameter),
            config.depth_start_m
            + config.depth_slope_m * parameter
            + config.depth_amplitude_m
            * np.sin(2.0 * np.pi * config.depth_cycles * parameter),
        )
    )
    raw_segments = np.diff(raw, axis=0)
    raw_cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(raw_segments, axis=1))))
    raw_length = float(raw_cumulative[-1])
    if raw_length <= 1e-9:
        raise ValueError("reference generator produced a zero-length path")

    sample_count = max(2, int(np.ceil(config.length_m / config.sample_spacing_m)) + 1)
    target_s: FloatArray = np.linspace(0.0, raw_length, sample_count, dtype=np.float64)
    resampled = _interp_rows(target_s, raw_cumulative, raw)
    scale = config.length_m / raw_length
    points = resampled[0] + (resampled - resampled[0]) * scale

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("reference generator produced duplicate adjacent points")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return ReferencePath(
        name="sinusoidal_3d_arclength",
        points_ned_m=points,
        cumulative_s_m=cumulative,
        speed_mps=config.speed_mps,
    )


def generate_sinusoidal_3d_path_bank(
    config: ReferenceConfig,
) -> tuple[ReferencePath, ...]:
    """Return an equal-length bank of deterministic sinusoidal path variants.

    The first bank entry is always the unmodified configuration. Remaining
    entries vary lateral/depth geometry while preserving ``length_m`` and
    sample spacing, so existing path-progress metrics remain comparable.
    """

    if config.path_bank_size == 1:
        return (generate_sinusoidal_3d_path(config),)

    rng = np.random.default_rng(config.path_bank_seed)
    paths: list[ReferencePath] = []
    for index in range(config.path_bank_size):
        if index == 0:
            variant = config
        else:
            span_low = max(1.0, 0.8 * config.x_span_m)
            span_high = max(span_low + 1e-6, 1.2 * config.x_span_m)
            lateral_low = 0.25 * config.lateral_amplitude_m
            lateral_high = max(lateral_low + 1e-6, 1.75 * config.lateral_amplitude_m)
            depth_low = max(2.0, config.depth_start_m - 4.0)
            depth_high = max(depth_low + 1e-6, config.depth_start_m + 8.0)
            variant = replace(
                config,
                x_span_m=float(rng.uniform(span_low, span_high)),
                lateral_amplitude_m=float(rng.uniform(lateral_low, lateral_high)),
                lateral_cycles=float(
                    rng.uniform(
                        max(0.25, 0.5 * config.lateral_cycles),
                        max(0.5, 1.5 * config.lateral_cycles),
                    )
                ),
                depth_start_m=float(rng.uniform(depth_low, depth_high)),
                depth_slope_m=float(rng.uniform(-2.0, 4.0)),
                depth_amplitude_m=float(
                    rng.uniform(0.0, max(0.2, 1.2 * config.depth_amplitude_m))
                ),
                depth_cycles=float(
                    rng.uniform(
                        max(0.25, 0.5 * config.depth_cycles),
                        max(0.5, 1.5 * config.depth_cycles),
                    )
                ),
            )
        path = generate_sinusoidal_3d_path(variant)
        paths.append(
            ReferencePath(
                name=f"sinusoidal_3d_arclength_bank_{index:03d}",
                points_ned_m=path.points_ned_m,
                cumulative_s_m=path.cumulative_s_m,
                speed_mps=path.speed_mps,
            )
        )
    return tuple(paths)


SUPPORTED_CURRICULUM_PATH_FAMILIES = ("P0", "P1", "P2", "P3", "P4")


def _make_reference_path(
    *,
    name: str,
    points_ned_m: FloatArray,
    speed_mps: float,
) -> ReferencePath:
    points = np.asarray(points_ned_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("path points must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("path points must be finite")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("path contains duplicate adjacent points")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return ReferencePath(
        name=name,
        points_ned_m=points,
        cumulative_s_m=cumulative,
        speed_mps=speed_mps,
    )


def _resample_to_reference_contract(
    *,
    name: str,
    raw_points_ned_m: FloatArray,
    length_m: float,
    sample_spacing_m: float,
    speed_mps: float,
) -> ReferencePath:
    raw = np.asarray(raw_points_ned_m, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] < 2:
        raise ValueError("raw path must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw path must be finite")
    if not math.isfinite(length_m) or length_m <= 0.0:
        raise ValueError("path length must be finite and positive")
    if not math.isfinite(sample_spacing_m) or sample_spacing_m <= 0.0:
        raise ValueError("path sample spacing must be finite and positive")

    raw_segments = np.diff(raw, axis=0)
    raw_segment_lengths = np.linalg.norm(raw_segments, axis=1)
    if np.any(raw_segment_lengths <= 1e-12):
        raise ValueError("raw path contains duplicate adjacent points")
    raw_cumulative = np.concatenate(([0.0], np.cumsum(raw_segment_lengths)))
    raw_length = float(raw_cumulative[-1])
    if raw_length <= 1e-9:
        raise ValueError("path generator produced a zero-length path")

    sample_count = max(2, int(np.ceil(length_m / sample_spacing_m)) + 1)
    target_s = np.linspace(0.0, raw_length, sample_count, dtype=np.float64)
    resampled = _interp_rows(target_s, raw_cumulative, raw)
    # Re-sampling through a polyline corner creates a tiny chord shortcut.
    # Scale the realized sampled polyline, not only the analytic raw length, so
    # the final ReferencePath total length is exact to floating-point tolerance.
    realized_length = float(
        np.linalg.norm(np.diff(resampled, axis=0), axis=1).sum()
    )
    if realized_length <= 1e-12:
        raise ValueError("resampled path has zero length")
    scale = float(length_m) / realized_length
    points = resampled[0] + (resampled - resampled[0]) * scale
    return _make_reference_path(name=name, points_ned_m=points, speed_mps=speed_mps)


def _bounded_segment_lengths(
    *,
    total_length_m: float,
    segment_count: int,
    minimum_segment_m: float,
    rng: np.random.Generator,
) -> FloatArray:
    if segment_count < 2:
        raise ValueError("curriculum path requires at least two segments")
    if minimum_segment_m <= 0.0:
        raise ValueError("minimum segment length must be positive")
    minimum_total = segment_count * minimum_segment_m
    if total_length_m < minimum_total + 1e-9:
        raise ValueError(
            "path length is too short for the requested family: "
            f"{total_length_m:.3f} m < {minimum_total:.3f} m"
        )
    residual = total_length_m - minimum_total
    weights = rng.uniform(0.5, 1.5, size=segment_count)
    return minimum_segment_m + residual * weights / float(weights.sum())


def _sample_constant_curvature_segment(
    *,
    start_xy: FloatArray,
    start_heading_rad: float,
    curvature_per_m: float,
    segment_length_m: float,
    depth_start_m: float,
    depth_slope: float,
    sample_step_m: float,
) -> tuple[FloatArray, float]:
    sample_count = max(2, int(np.ceil(segment_length_m / sample_step_m)) + 1)
    local_s = np.linspace(0.0, segment_length_m, sample_count, dtype=np.float64)[1:]
    heading_end = start_heading_rad + curvature_per_m * segment_length_m
    if abs(curvature_per_m) < 1e-12:
        delta_xy = np.column_stack(
            (
                local_s * math.cos(start_heading_rad),
                local_s * math.sin(start_heading_rad),
            )
        )
    else:
        headings = start_heading_rad + curvature_per_m * local_s
        delta_xy = np.column_stack(
            (
                (np.sin(headings) - math.sin(start_heading_rad)) / curvature_per_m,
                (math.cos(start_heading_rad) - np.cos(headings)) / curvature_per_m,
            )
        )
    xy = start_xy + delta_xy
    return xy, heading_end


def _generate_bounded_curvature_path(
    config: ReferenceConfig,
    *,
    family: str,
    rng: np.random.Generator,
    name: str,
) -> ReferencePath:
    if family == "P3":
        segment_count = int(rng.integers(4, 7))
        minimum_segment_m = 5.0
        max_turn_rad = math.radians(30.0)
        max_curvature = 0.08
    elif family == "P4":
        segment_count = int(rng.integers(5, 8))
        minimum_segment_m = 4.0
        max_turn_rad = math.radians(40.0)
        max_curvature = 0.10
    else:
        raise ValueError(f"unsupported bounded-curvature family {family!r}")

    lengths = _bounded_segment_lengths(
        total_length_m=config.length_m,
        segment_count=segment_count,
        minimum_segment_m=minimum_segment_m,
        rng=rng,
    )
    depth_slope = float(rng.uniform(-0.01, 0.01))
    heading = 0.0
    xy = np.zeros((1, 2), dtype=np.float64)
    z = [float(config.depth_start_m)]
    sample_step_m = min(0.05, config.sample_spacing_m / 4.0)
    first_sign = -1.0 if rng.random() < 0.5 else 1.0
    for index, segment_length in enumerate(lengths):
        sign = first_sign if index % 2 == 0 else -first_sign
        curvature_limit_for_segment = min(max_curvature, max_turn_rad / segment_length)
        curvature = sign * float(rng.uniform(0.35, 1.0)) * curvature_limit_for_segment
        xy_segment, heading = _sample_constant_curvature_segment(
            start_xy=xy[-1],
            start_heading_rad=heading,
            curvature_per_m=curvature,
            segment_length_m=float(segment_length),
            depth_start_m=float(config.depth_start_m),
            depth_slope=depth_slope,
            sample_step_m=sample_step_m,
        )
        xy = np.vstack((xy, xy_segment))
        previous_z = z[-1]
        local_count = xy_segment.shape[0]
        z.extend(
            (previous_z + depth_slope * segment_length * (step + 1) / local_count)
            for step in range(local_count)
        )
    raw = np.column_stack((xy, np.asarray(z, dtype=np.float64)))
    return _resample_to_reference_contract(
        name=name,
        raw_points_ned_m=raw,
        length_m=config.length_m,
        sample_spacing_m=config.sample_spacing_m,
        speed_mps=config.speed_mps,
    )


def _generate_polyline_path(
    config: ReferenceConfig,
    *,
    family: str,
    rng: np.random.Generator,
    name: str,
) -> ReferencePath:
    if family == "P2":
        segment_count = int(rng.integers(3, 5))
        minimum_segment_m = 8.0
        min_turn_rad = math.radians(10.0)
        max_turn_rad = math.radians(20.0)
    else:
        raise ValueError(f"unsupported polyline family {family!r}")

    lengths = _bounded_segment_lengths(
        total_length_m=config.length_m,
        segment_count=segment_count,
        minimum_segment_m=minimum_segment_m,
        rng=rng,
    )
    depth_slope = float(rng.uniform(-0.01, 0.01))
    heading = 0.0
    point = np.array([0.0, 0.0, float(config.depth_start_m)], dtype=np.float64)
    points = [point.copy()]
    turn_sign = -1.0 if rng.random() < 0.5 else 1.0
    for index, segment_length in enumerate(lengths):
        if index > 0:
            turn = turn_sign * float(rng.uniform(min_turn_rad, max_turn_rad))
            heading += turn
            if abs(heading) > math.radians(35.0):
                turn_sign = -turn_sign
        point = point + np.array(
            [
                segment_length * math.cos(heading),
                segment_length * math.sin(heading),
                depth_slope * segment_length,
            ],
            dtype=np.float64,
        )
        points.append(point.copy())
    return _resample_to_reference_contract(
        name=name,
        raw_points_ned_m=np.stack(points),
        length_m=config.length_m,
        sample_spacing_m=config.sample_spacing_m,
        speed_mps=config.speed_mps,
    )


def generate_curriculum_path(
    config: ReferenceConfig,
    family: str,
    *,
    seed: int,
) -> ReferencePath:
    """Generate one deterministic P0-P4 curriculum-family path."""

    if family not in SUPPORTED_CURRICULUM_PATH_FAMILIES:
        raise ValueError(
            f"unsupported curriculum path family {family!r}; "
            f"expected one of {SUPPORTED_CURRICULUM_PATH_FAMILIES}"
        )
    rng = np.random.default_rng(int(seed))
    name = f"{family.lower()}_curriculum_path_s{int(seed)}"
    if family == "P0":
        heading = float(rng.uniform(-math.radians(8.0), math.radians(8.0)))
        depth_slope = float(rng.uniform(-0.005, 0.005))
        direction = np.array(
            [math.cos(heading), math.sin(heading), depth_slope],
            dtype=np.float64,
        )
        direction = direction / float(np.linalg.norm(direction))
        sample_count = max(2, int(np.ceil(config.length_m / config.sample_spacing_m)) + 1)
        s = np.linspace(0.0, config.length_m, sample_count, dtype=np.float64)
        points = np.array([0.0, 0.0, float(config.depth_start_m)]) + s[:, None] * direction
        return _make_reference_path(name=name, points_ned_m=points, speed_mps=config.speed_mps)

    if family == "P1":
        variant = replace(
            config,
            x_span_m=max(1.0, 0.9 * config.length_m),
            lateral_amplitude_m=float(rng.uniform(0.5, 1.0)),
            lateral_cycles=float(rng.uniform(0.5, 1.0)),
            depth_slope_m=float(rng.uniform(-0.01, 0.01)),
            depth_amplitude_m=float(rng.uniform(0.1, 0.2)),
            depth_cycles=float(rng.uniform(0.5, 1.0)),
        )
        path = generate_sinusoidal_3d_path(variant)
        return _resample_to_reference_contract(
            name=name,
            raw_points_ned_m=path.points_ned_m,
            length_m=config.length_m,
            sample_spacing_m=config.sample_spacing_m,
            speed_mps=config.speed_mps,
        )

    if family == "P2":
        if bool(rng.integers(0, 2)):
            variant = replace(
                config,
                x_span_m=max(1.0, 0.9 * config.length_m),
                lateral_amplitude_m=float(rng.uniform(1.0, 2.0)),
                lateral_cycles=float(rng.uniform(1.0, 2.0)),
                depth_slope_m=float(rng.uniform(-0.01, 0.01)),
                depth_amplitude_m=float(rng.uniform(0.15, 0.35)),
                depth_cycles=float(rng.uniform(0.5, 1.5)),
            )
            path = generate_sinusoidal_3d_path(variant)
            return _resample_to_reference_contract(
                name=f"{name}_sine",
                raw_points_ned_m=path.points_ned_m,
                length_m=config.length_m,
                sample_spacing_m=config.sample_spacing_m,
                speed_mps=config.speed_mps,
            )
        return _generate_polyline_path(
            config,
            family=family,
            rng=rng,
            name=f"{name}_polyline",
        )

    return _generate_bounded_curvature_path(config, family=family, rng=rng, name=name)


def _apportion_path_family_counts(
    *,
    size: int,
    weights: FloatArray,
) -> list[int]:
    if size <= 0:
        raise ValueError("path bank size must be positive")
    positive = weights > 0.0
    if not bool(positive.any()):
        raise ValueError("path family mix must contain a positive weight")
    raw = size * weights / float(weights.sum())
    counts = [int(math.floor(value)) for value in raw]
    family_count = int(positive.sum())
    if size >= family_count:
        for index in range(len(counts)):
            if counts[index] > 0 or not positive[index]:
                continue
            donors = [
                donor
                for donor in range(len(counts))
                if positive[donor] and counts[donor] > 1
            ]
            if not donors:
                raise ValueError("could not apportion at least one path per family")
            donor = max(donors, key=lambda item: (raw[item] - counts[item], counts[item]))
            counts[donor] -= 1
            counts[index] += 1
    remaining = size - sum(counts)
    fractions = [
        (raw[index] - counts[index]) if positive[index] else -1.0
        for index in range(len(counts))
    ]
    for _ in range(remaining):
        index = max(range(len(counts)), key=lambda item: fractions[item])
        counts[index] += 1
        fractions[index] -= 1.0
    return counts


def generate_mixed_path_bank(
    config: ReferenceConfig,
    family_mix: tuple[tuple[str, float], ...],
    *,
    size: int | None = None,
    seed: int | None = None,
) -> tuple[ReferencePath, ...]:
    """Generate a deterministic bank containing multiple P0-P4 families."""

    if not family_mix:
        raise ValueError("family_mix must not be empty")
    families: list[str] = []
    weights: list[float] = []
    for family, weight in family_mix:
        if family not in SUPPORTED_CURRICULUM_PATH_FAMILIES:
            raise ValueError(f"unsupported curriculum path family {family!r}")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("path family weights must be finite and positive")
        if family in families:
            raise ValueError(f"duplicate path family in mix: {family!r}")
        families.append(family)
        weights.append(float(weight))
    bank_size = config.path_bank_size if size is None else int(size)
    bank_seed = config.path_bank_seed if seed is None else int(seed)
    if bank_seed < 0:
        raise ValueError("path bank seed must be non-negative")
    weight_array = np.asarray(weights, dtype=np.float64)
    counts = _apportion_path_family_counts(size=bank_size, weights=weight_array)
    sequence = np.repeat(np.arange(len(families), dtype=np.int64), counts)
    rng = np.random.default_rng(bank_seed + 0xC0FFEE)
    sequence = sequence[rng.permutation(sequence.shape[0])]
    paths: list[ReferencePath] = []
    for index, family_index in enumerate(sequence.tolist()):
        family = families[int(family_index)]
        path_seed = bank_seed + 104729 * index + 1009 * int(family_index)
        paths.append(generate_curriculum_path(config, family, seed=path_seed))
    return tuple(paths)


def generate_pc_path_bank(config: ReferenceConfig) -> tuple[ReferencePath, ...]:
    """Dispatch to legacy or curriculum path generation without changing defaults."""

    if config.path_family is None and config.path_family_mix is None:
        return generate_sinusoidal_3d_path_bank(config)
    if config.path_family_mix is not None:
        family_mix = config.path_family_mix
    elif config.path_family is not None:
        family_mix = ((config.path_family, 1.0),)
    else:
        raise RuntimeError("unreachable path-family dispatch state")
    return generate_mixed_path_bank(config, family_mix)


def resample_reference_path(path: ReferencePath, sample_count: int) -> ReferencePath:
    """Resample a path to a fixed number of points without changing its geometry."""

    if isinstance(sample_count, bool) or int(sample_count) < 2:
        raise ValueError("sample_count must be an integer >= 2")
    count = int(sample_count)
    if path.points_ned_m.shape[0] == count:
        return path
    target_s = np.linspace(0.0, path.total_length_m, count, dtype=np.float64)
    points = _interp_rows(target_s, path.cumulative_s_m, path.points_ned_m)
    realized = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if realized <= 1e-12:
        raise ValueError("resampled path has zero length")
    points = points[0] + (points - points[0]) * (path.total_length_m / realized)
    return _make_reference_path(
        name=path.name,
        points_ned_m=points,
        speed_mps=path.speed_mps,
    )


__all__ = [
    "SUPPORTED_CURRICULUM_PATH_FAMILIES",
    "PathProjection",
    "ReferencePath",
    "generate_curriculum_path",
    "generate_mixed_path_bank",
    "generate_pc_path_bank",
    "resample_reference_path",
    "generate_sinusoidal_3d_path",
    "generate_sinusoidal_3d_path_bank",
]
