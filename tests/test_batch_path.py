from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.path import (
    BatchedReferencePath,
    BatchedReferencePathBank,
)
from auv_lowlevel.config import load_config
from auv_lowlevel.path import (
    generate_sinusoidal_3d_path,
    generate_sinusoidal_3d_path_bank,
)


def _path() -> BatchedReferencePath:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    return BatchedReferencePath.from_reference_path(generate_sinusoidal_3d_path(config.reference))


def test_projection_matches_scalar_for_random_positions() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    reference = generate_sinusoidal_3d_path(config.reference)
    batched = BatchedReferencePath.from_reference_path(reference)
    rng = np.random.default_rng(12)
    positions = rng.uniform(-10.0, 40.0, size=(200, 3))

    expected = [reference.project(position) for position in positions]
    actual = batched.project(torch.from_numpy(positions))

    np.testing.assert_allclose(
        actual.progress_m.numpy(), [item.progress_m for item in expected], atol=1e-12
    )
    np.testing.assert_allclose(
        actual.point_ned_m.numpy(),
        np.stack([item.point_ned_m for item in expected]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.tangent_ned.numpy(),
        np.stack([item.tangent_ned for item in expected]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.horizontal_error_m.numpy(),
        [item.horizontal_error_m for item in expected],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.vertical_error_m.numpy(),
        [item.vertical_error_m for item in expected],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.error_3d_m.numpy(), [item.error_3d_m for item in expected], atol=1e-12
    )
    np.testing.assert_allclose(
        actual.curvature_1_per_m.numpy(),
        [item.curvature_1_per_m for item in expected],
        atol=1e-12,
    )


def test_shared_vertices_use_larger_progress_tie_break() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    reference = generate_sinusoidal_3d_path(config.reference)
    batched = BatchedReferencePath.from_reference_path(reference)
    vertex_indices = np.arange(1, reference.points_ned_m.shape[0] - 1)
    positions = reference.points_ned_m[vertex_indices]

    actual = batched.project(torch.from_numpy(positions))
    expected = [reference.project(position) for position in positions]

    assert actual.segment_index.numpy().tolist() == [item.segment_index for item in expected]
    np.testing.assert_allclose(
        actual.progress_m.numpy(), [item.progress_m for item in expected], atol=1e-12
    )


def test_windowed_projection_matches_global_near_path() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    reference = generate_sinusoidal_3d_path(config.reference)
    batched = BatchedReferencePath.from_reference_path(reference)
    s_values = np.linspace(0.0, reference.total_length_m, 500)
    positions = np.stack([reference.point_at_s(s_value) for s_value in s_values])
    positions[:, 1] += 0.01 * np.sin(s_values)
    positions[:, 2] += 0.02 * np.cos(s_values)
    position_tensor = torch.from_numpy(positions)

    expected = batched.project(position_tensor)
    hints = torch.empty_like(expected.segment_index)
    hints[0] = 0
    hints[1:] = expected.segment_index[:-1]
    actual = batched.project(position_tensor, segment_hint=hints, search_radius=4)

    assert torch.equal(actual.segment_index, expected.segment_index)
    torch.testing.assert_close(actual.progress_m, expected.progress_m, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(actual.point_ned_m, expected.point_ned_m, atol=1e-12, rtol=0.0)


def test_point_and_curvature_queries_match_scalar() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    reference = generate_sinusoidal_3d_path(config.reference)
    batched = BatchedReferencePath.from_reference_path(reference)
    s_values = np.linspace(-1.0, reference.total_length_m + 1.0, 300)
    s = torch.from_numpy(s_values)

    np.testing.assert_allclose(
        batched.point_at_s(s).numpy(),
        np.stack([reference.point_at_s(value) for value in s_values]),
    )
    np.testing.assert_allclose(
        batched.curvature_at_s(s).numpy(),
        [reference.curvature_at_s(value) for value in s_values],
    )


def test_path_bank_default_preserves_existing_fixed_path() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    reference = generate_sinusoidal_3d_path(config.reference)
    bank = generate_sinusoidal_3d_path_bank(config.reference)

    assert config.reference.path_bank_size == 1
    assert len(bank) == 1
    np.testing.assert_array_equal(bank[0].points_ned_m, reference.points_ned_m)
    np.testing.assert_array_equal(bank[0].cumulative_s_m, reference.cumulative_s_m)


def test_path_bank_is_reproducible_and_preserves_comparable_geometry() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    reference = replace(
        config.reference,
        path_bank_size=8,
        path_bank_seed=37,
    )

    first = generate_sinusoidal_3d_path_bank(reference)
    second = generate_sinusoidal_3d_path_bank(reference)

    assert len(first) == len(second) == 8
    assert first[0].points_ned_m.tolist() == generate_sinusoidal_3d_path(
        config.reference
    ).points_ned_m.tolist()
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.points_ned_m, right.points_ned_m)
        np.testing.assert_array_equal(left.cumulative_s_m, right.cumulative_s_m)
    for path in first:
        assert path.points_ned_m.shape == first[0].points_ned_m.shape
        assert path.cumulative_s_m[-1] == pytest.approx(
            first[0].cumulative_s_m[-1], abs=1e-4
        )
        assert path.cumulative_s_m[-1] == pytest.approx(reference.length_m, abs=1e-4)
        np.testing.assert_allclose(
            path.cumulative_s_m,
            first[0].cumulative_s_m,
            atol=1e-4,
            rtol=0.0,
        )
    assert any(
        not np.allclose(path.points_ned_m, first[0].points_ned_m)
        for path in first[1:]
    )


def test_path_set_queries_match_independent_single_paths() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    reference = replace(
        config.reference,
        path_bank_size=8,
        path_bank_seed=19,
    )
    paths = generate_sinusoidal_3d_path_bank(reference)
    bank = BatchedReferencePathBank(paths, dtype=torch.float64)
    indices = torch.tensor([0, 3, 1, 7], dtype=torch.long)
    path_set = bank.select(indices)
    single_paths = [BatchedReferencePath.from_reference_path(paths[index]) for index in indices.tolist()]

    s_values = torch.tensor([0.0, 1.7, 9.3, 34.99], dtype=torch.float64)
    for row, single in enumerate(single_paths):
        s_query = s_values[row : row + 1]
        torch.testing.assert_close(
            path_set.point_at_s(s_values)[row],
            single.point_at_s(s_query)[0],
            atol=5e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            path_set.curvature_at_s(s_values)[row],
            single.curvature_at_s(s_query)[0],
            atol=5e-5,
            rtol=0.0,
        )

    np.testing.assert_allclose(
        path_set.start_ned_m.numpy(),
        np.stack([path.start_ned_m for path in single_paths]),
    )
    np.testing.assert_allclose(
        path_set.start_tangent_ned.numpy(),
        np.stack([path.start_tangent_ned for path in single_paths]),
    )
    np.testing.assert_allclose(
        path_set.goal_ned_m.numpy(),
        np.stack([path.goal_ned_m for path in single_paths]),
    )
    np.testing.assert_allclose(
        path_set.total_length_m_tensor.numpy(),
        [path.total_length_m for path in single_paths],
        atol=1e-4,
        rtol=0.0,
    )

    rng = np.random.default_rng(23)
    positions = torch.empty((len(single_paths), 3), dtype=torch.float64)
    for row, index in enumerate(indices.tolist()):
        path = paths[index]
        center = path.point_at_s(float(rng.uniform(0.0, path.cumulative_s_m[-1])))
        positions[row] = torch.as_tensor(center + rng.normal(0.0, 0.4, size=3))

    actual = path_set.project(positions)
    expected = [
        path.project(positions[row : row + 1].numpy())
        for row, path in enumerate(single_paths)
    ]
    for field in (
        "progress_m",
        "point_ned_m",
        "tangent_ned",
        "lateral_axis_ned",
        "vertical_axis_ned",
        "horizontal_error_m",
        "vertical_error_m",
        "error_3d_m",
        "curvature_1_per_m",
    ):
        np.testing.assert_allclose(
            getattr(actual, field).numpy(),
            np.concatenate(
                [np.asarray(getattr(item, field)) for item in expected], axis=0
            ),
            atol=5e-5,
            rtol=0.0,
        )
    assert actual.segment_index.tolist() == [item.segment_index for item in expected]

    windowed = path_set.project(
        positions,
        segment_hint=actual.segment_index,
        search_radius=4,
    )
    torch.testing.assert_close(windowed.progress_m, actual.progress_m, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(windowed.point_ned_m, actual.point_ned_m, atol=1e-5, rtol=0.0)
