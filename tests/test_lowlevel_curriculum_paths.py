from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from auv_lowlevel.config import load_config, parse_config
from auv_lowlevel.path import (
    SUPPORTED_CURRICULUM_PATH_FAMILIES,
    generate_curriculum_path,
    generate_mixed_path_bank,
    generate_pc_path_bank,
    generate_sinusoidal_3d_path_bank,
)

ROOT = Path(__file__).resolve().parents[1]
GOAL85_CONFIG = (
    ROOT
    / "configs"
    / "lowlevel"
    / "pc_goal85_unbiased_default_candidate_20261110.json"
)


def _config():
    return load_config(GOAL85_CONFIG)


def _config_payload():
    return json.loads(GOAL85_CONFIG.read_text(encoding="utf-8"))


def _sample_curvature(path) -> float:
    s_values = np.linspace(0.0, path.total_length_m, 800)
    return max(path.curvature_at_s(float(value)) for value in s_values)


def _segments_intersect_xy(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float(np.cross(q - p, r - p))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return bool(
            min(p[0], q[0]) - 1e-12 <= r[0] <= max(p[0], q[0]) + 1e-12
            and min(p[1], q[1]) - 1e-12 <= r[1] <= max(p[1], q[1]) + 1e-12
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    eps = 1e-10
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    if abs(o1) <= eps and on_segment(a, b, c):
        return True
    if abs(o2) <= eps and on_segment(a, b, d):
        return True
    if abs(o3) <= eps and on_segment(c, d, a):
        return True
    if abs(o4) <= eps and on_segment(c, d, b):
        return True
    return False


@pytest.mark.parametrize("family", ["P2", "P3", "P4"])
def test_curriculum_path_xy_projection_does_not_self_intersect(family: str) -> None:
    reference = _config().reference
    for seed in range(8100, 8120):
        points = generate_curriculum_path(reference, family, seed=seed).points_ned_m[:, :2]
        for first in range(len(points) - 1):
            for second in range(first + 2, len(points) - 1):
                if points[second, 0] >= points[first + 1, 0] - 1e-12:
                    break
                assert not _segments_intersect_xy(
                    points[first],
                    points[first + 1],
                    points[second],
                    points[second + 1],
                ), f"{family} seed={seed} segments {first} and {second} intersect"


def test_default_pc_path_bank_keeps_legacy_sinusoidal_bank_bit_exact() -> None:
    reference = _config().reference
    legacy = generate_sinusoidal_3d_path_bank(reference)
    dispatched = generate_pc_path_bank(reference)
    assert len(legacy) == len(dispatched)
    for left, right in zip(legacy, dispatched, strict=True):
        assert left.name == right.name
        np.testing.assert_array_equal(left.points_ned_m, right.points_ned_m)
        np.testing.assert_array_equal(left.cumulative_s_m, right.cumulative_s_m)


@pytest.mark.parametrize("family", SUPPORTED_CURRICULUM_PATH_FAMILIES)
def test_curriculum_paths_preserve_seed_length_and_spacing(family: str) -> None:
    reference = _config().reference
    first = generate_curriculum_path(reference, family, seed=8101)
    second = generate_curriculum_path(reference, family, seed=8101)
    np.testing.assert_array_equal(first.points_ned_m, second.points_ned_m)
    np.testing.assert_array_equal(first.cumulative_s_m, second.cumulative_s_m)
    assert first.name == second.name
    assert first.speed_mps == pytest.approx(reference.speed_mps)
    assert first.points_ned_m.shape == (
        int(math.ceil(reference.length_m / reference.sample_spacing_m)) + 1,
        3,
    )
    assert first.total_length_m == pytest.approx(reference.length_m, abs=1e-5)
    assert np.all(np.diff(first.cumulative_s_m) > 0.0)
    assert np.all(np.diff(first.points_ned_m[:, 0]) > 0.0)


def test_p0_is_straight_and_has_negligible_curvature() -> None:
    path = generate_curriculum_path(_config().reference, "P0", seed=8102)
    assert _sample_curvature(path) < 1e-4
    lateral_drift = path.points_ned_m[:, 1] - path.points_ned_m[0, 1]
    assert float(np.max(np.abs(lateral_drift))) < 5.0


@pytest.mark.parametrize(
    "family, curvature_limit",
    [("P3", 0.08), ("P4", 0.10)],
)
def test_bounded_curvature_families_respect_geometry_limits(
    family: str, curvature_limit: float
) -> None:
    path = generate_curriculum_path(_config().reference, family, seed=8103)
    assert _sample_curvature(path) <= curvature_limit + 1e-9
    assert np.all(np.diff(path.points_ned_m[:, 0]) > 0.0)
    assert np.isfinite(path.points_ned_m).all()


def test_p2_bank_covers_sine_and_polyline_subtypes() -> None:
    reference = _config().reference
    names = {
        generate_curriculum_path(reference, "P2", seed=seed).name
        for seed in range(20, 28)
    }
    assert any(name.endswith("_sine") for name in names)
    assert any(name.endswith("_polyline") for name in names)


def test_mixed_path_bank_is_deterministic_and_matches_requested_mix() -> None:
    reference = _config().reference
    mix = (("P0", 0.7), ("P1", 0.3))
    first = generate_mixed_path_bank(reference, mix, size=10, seed=42)
    second = generate_mixed_path_bank(reference, mix, size=10, seed=42)
    assert len(first) == len(second) == 10
    for left, right in zip(first, second, strict=True):
        assert left.name == right.name
        np.testing.assert_array_equal(left.points_ned_m, right.points_ned_m)
    counts = {
        family: sum(path.name.startswith(family.lower() + "_") for path in first)
        for family in ("P0", "P1")
    }
    assert counts == {"P0": 7, "P1": 3}


def test_path_family_config_parser_round_trips() -> None:
    payload = _config_payload()
    payload["reference"]["path_family"] = "P3"
    payload["reference"]["path_bank_size"] = 4
    config = parse_config(payload)
    assert config.reference.path_family == "P3"
    assert config.reference.path_family_mix is None
    paths = generate_pc_path_bank(config.reference)
    assert len(paths) == 4
    assert all(path.name.startswith("p3_") for path in paths)

    payload = _config_payload()
    payload["reference"]["path_family_mix"] = {"P0": 0.7, "P2": 0.3}
    payload["reference"]["path_bank_size"] = 10
    config = parse_config(payload)
    assert config.reference.path_family_mix == (("P0", 0.7), ("P2", 0.3))
    paths = generate_pc_path_bank(config.reference)
    assert sum(path.name.startswith("p0_") for path in paths) == 7
    assert sum(path.name.startswith("p2_") for path in paths) == 3


@pytest.mark.parametrize(
    "reference_update, message",
    [
        ({"path_family": "P9"}, "path_family"),
        ({"path_family_mix": {"P9": 1.0}}, "unsupported family"),
        ({"path_family_mix": {"P0": 0.0}}, "positive"),
        ({"path_family_mix": {"P0": 1.0, "P1": -0.1}}, "positive"),
        (
            {"path_family": "P0", "path_family_mix": {"P1": 1.0}},
            "cannot both be set",
        ),
    ],
)
def test_path_family_config_parser_rejects_invalid_values(
    reference_update: dict, message: str
) -> None:
    payload = _config_payload()
    payload["reference"].update(reference_update)
    with pytest.raises(ValueError, match=message):
        parse_config(payload)


def test_path_generator_rejects_too_short_length_for_family() -> None:
    reference = _config().reference
    short = reference.__class__(
        **{
            **reference.__dict__,
            "length_m": 20.0,
        }
    )
    with pytest.raises(ValueError, match="too short"):
        generate_curriculum_path(short, "P2", seed=1)
