from __future__ import annotations

import numpy as np
import pytest

from auv_lowlevel.bc import ACTION_SCALES
from auv_lowlevel.observation_schema import (
    OBSERVATION_SCHEMA_V1,
    OBSERVATION_SCHEMA_V2,
)
from auv_lowlevel.residual import (
    build_residual_labels,
    label_statistics,
    mask_residual_actor_inputs,
)


def test_residual_labels_use_configured_action_authority() -> None:
    observation = np.zeros((2, 27), dtype=np.float32)
    base = np.zeros((2, 3), dtype=np.float32)
    teacher = np.stack(
        [
            0.5 * ACTION_SCALES * np.array([0.3, 0.2, 0.15]),
            2.0 * ACTION_SCALES * np.array([0.3, 0.2, 0.15]),
        ]
    ).astype(np.float32)
    labels = build_residual_labels(
        observation,
        teacher,
        base,
        np.array([0.3, 0.2, 0.15], dtype=np.float32),
        np.array([1, 1]),
        np.array([0, 0], dtype=np.int8),
    )

    assert np.allclose(labels.target_normalized[0], 0.5)
    assert np.allclose(labels.target_normalized[1], 1.0)
    stats = label_statistics(labels)
    assert stats["sample_count"] == 2
    assert stats["episode_count"] == 1
    assert stats["target_clip_fraction"] == [0.5, 0.5, 0.5]


def test_residual_labels_reject_zero_authority() -> None:
    with np.testing.assert_raises(ValueError):
        build_residual_labels(
            np.zeros((1, 27), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.array([0.3, 0.0, 0.15], dtype=np.float32),
            np.array([0]),
            np.array([0], dtype=np.int8),
        )


def test_residual_labels_and_masks_support_26d_v2() -> None:
    observation = np.arange(52, dtype=np.float32).reshape(2, 26)
    labels = build_residual_labels(
        observation,
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.array([0.3, 0.2, 0.15], dtype=np.float32),
        np.array([10, 10]),
        np.array([0, 0], dtype=np.int8),
        observation_schema="v2",
    )

    assert labels.observation.shape == (2, 26)
    assert labels.observation_schema == "v2"
    masked = mask_residual_actor_inputs(
        observation,
        expected_schema=OBSERVATION_SCHEMA_V2,
    )
    assert masked.shape == observation.shape
    if not np.all(masked[:, [17, 18, 19, 23, 24, 25]] == 0.0):
        raise AssertionError("26D residual actor inputs were not masked")
    if not np.array_equal(
        masked[:, [index for index in range(26) if index not in {17, 18, 19, 23, 24, 25}]],
        observation[:, [index for index in range(26) if index not in {17, 18, 19, 23, 24, 25}]],
    ):
        raise AssertionError("26D mask changed visible inputs")


def test_residual_actor_mask_checks_schema() -> None:
    with pytest.raises(ValueError, match="schema mismatch"):
        mask_residual_actor_inputs(
            np.zeros((1, 27), dtype=np.float32),
            expected_schema=OBSERVATION_SCHEMA_V2,
        )
    masked = mask_residual_actor_inputs(
        np.ones((1, 27), dtype=np.float32),
        expected_schema=OBSERVATION_SCHEMA_V1,
    )
    assert np.count_nonzero(masked) == 21
    with pytest.raises(ValueError, match="schema"):
        build_residual_labels(
            np.zeros((1, 26), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.array([0.3, 0.2, 0.15], dtype=np.float32),
            np.array([0]),
            np.array([0], dtype=np.int8),
            observation_schema="v1",
        )


def test_terminal_residual_governor_is_smooth_and_monotone() -> None:
    from auv_lowlevel.residual import TerminalResidualGovernor

    governor = TerminalResidualGovernor(zero_radius_m=1.25, full_scale_radius_m=4.0)
    assert governor.scale(1.25) == pytest.approx(0.0)
    assert governor.scale(4.0) == pytest.approx(1.0)
    distances = np.linspace(1.25, 4.0, 101)
    scales = np.asarray([governor.scale(float(distance)) for distance in distances])
    assert np.all(np.diff(scales) >= -1e-15)
    assert 0.0 < governor.scale(2.625) < 1.0
    assert governor.scale(7.0) == pytest.approx(1.0)


def test_terminal_residual_governor_validates_configuration() -> None:
    from auv_lowlevel.residual import TerminalResidualGovernor

    with pytest.raises(ValueError, match="exceed"):
        TerminalResidualGovernor(zero_radius_m=2.0, full_scale_radius_m=2.0)
    with pytest.raises(ValueError, match="floor"):
        TerminalResidualGovernor(
            zero_radius_m=1.25,
            full_scale_radius_m=4.0,
            floor=1.01,
        )
    governor = TerminalResidualGovernor(zero_radius_m=1.25, full_scale_radius_m=4.0)
    with pytest.raises(ValueError, match="distance"):
        governor.scale(float("nan"))
