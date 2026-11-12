from __future__ import annotations

import numpy as np
import pytest

from auv_lowlevel.config import load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path
from auv_lowlevel.randomization import (
    evaluation_seeds,
    sample_episode_randomization,
    training_seeds,
)


def _config():
    return load_config("configs/lowlevel/odarrl_control_base.json")


def test_generated_path_has_frozen_length_and_monotonic_arclength() -> None:
    config = _config()
    path = generate_sinusoidal_3d_path(config.reference)

    assert path.points_ned_m.shape[1] == 3
    assert path.total_length_m == pytest.approx(35.0, abs=1e-3)
    assert path.cumulative_s_m[0] == 0.0
    assert np.all(np.diff(path.cumulative_s_m) > 0.0)
    assert np.allclose(
        np.linalg.norm(np.diff(path.points_ned_m, axis=0), axis=1),
        np.diff(path.cumulative_s_m),
    )
    assert path.start_ned_m.tolist() == pytest.approx([0.0, 0.0, 10.0])


def test_path_generation_is_deterministic_and_trajectory_matches_points() -> None:
    config = _config()
    first = generate_sinusoidal_3d_path(config.reference)
    second = generate_sinusoidal_3d_path(config.reference)
    trajectory = first.trajectory()

    assert np.array_equal(first.points_ned_m, second.points_ned_m)
    assert np.array_equal(first.cumulative_s_m, second.cumulative_s_m)
    assert trajectory.position_ned_m.shape == first.points_ned_m.shape
    assert trajectory.quat_xyzw.shape == (first.points_ned_m.shape[0], 4)
    assert np.all(np.diff(trajectory.t_s) > 0.0)


def test_path_projection_recovers_points_and_reports_orthogonal_errors() -> None:
    config = _config()
    path = generate_sinusoidal_3d_path(config.reference)
    target_s = 12.5
    point = path.point_at_s(target_s)
    tangent = path.project(point).tangent_ned
    offset = 0.3 * path.project(point).lateral_axis_ned + 0.2 * path.project(
        point
    ).vertical_axis_ned
    projection = path.project(point + offset)

    assert projection.progress_m == pytest.approx(target_s, abs=0.11)
    assert projection.horizontal_error_m == pytest.approx(0.3, abs=1e-6)
    assert projection.vertical_error_m == pytest.approx(0.2, abs=1e-6)
    assert projection.error_3d_m == pytest.approx(np.hypot(0.3, 0.2), abs=1e-6)
    assert np.dot(tangent, offset) == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(projection.curvature_1_per_m)


def test_episode_randomization_is_reproducible_and_bounded() -> None:
    config = _config()
    first = sample_episode_randomization(17, config.current, config.reference)
    second = sample_episode_randomization(17, config.current, config.reference)
    another = sample_episode_randomization(18, config.current, config.reference)

    assert np.array_equal(first.current_ned_mps, second.current_ned_mps)
    assert np.array_equal(first.start_jitter_ned_m, second.start_jitter_ned_m)
    assert not np.array_equal(first.current_ned_mps, another.current_ned_mps)
    speed = float(np.linalg.norm(first.current_ned_mps))
    assert config.current.min_speed_mps <= speed <= config.current.max_speed_mps
    assert first.current_ned_mps[2] == 0.0
    assert np.all(np.abs(first.start_jitter_ned_m[:2]) <= config.reference.start_jitter_xy_m)
    assert abs(first.start_jitter_ned_m[2]) <= config.reference.start_jitter_z_m
    assert abs(first.start_yaw_jitter_rad) <= config.reference.start_jitter_yaw_rad


def test_evaluation_and_training_seed_pools_are_disjoint() -> None:
    config = _config()
    evaluation = set(evaluation_seeds(config.evaluation.seed_start, config.evaluation.seed_count))
    training = set(
        training_seeds(
            100,
            config.evaluation.training_seed_start,
            excluded=evaluation,
        )
    )

    assert not evaluation.intersection(training)
    assert min(training) >= config.evaluation.training_seed_start
