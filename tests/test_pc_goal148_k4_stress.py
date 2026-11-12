from __future__ import annotations

import json
from pathlib import Path

import pytest

from auv_lowlevel.config import load_config
from scripts.evaluate_pc_ppo import _build_curriculum_sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs/lowlevel/goal148/profile_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_goal148_manifest_has_only_strict_stress_levels() -> None:
    manifest = _manifest()
    assert manifest["goal_radius_m"] == pytest.approx(1.25)
    assert set(manifest["levels"]) == {
        "nominal",
        "current15",
        "current20",
        "plant20",
        "combined20",
    }
    assert "default" not in manifest["levels"]


def test_goal148_stress_profiles_preserve_success_contract() -> None:
    base = load_config(REPO_ROOT / "configs/lowlevel/goal136/curriculum/C0.json")
    manifest = _manifest()
    for level, entry in manifest["levels"].items():
        config = load_config(REPO_ROOT / entry["path"])
        assert config.plant.goal_radius_m == pytest.approx(1.25), level
        assert config.reward == base.reward, level
        assert config.termination == base.termination, level
        assert config.guidance == base.guidance, level
        assert config.curriculum == base.curriculum, level
        assert config.termination.max_depth_m == pytest.approx(200.0), level
        assert config.termination.min_depth_m == pytest.approx(0.0), level


def test_goal148_current_doses_are_frozen() -> None:
    expected = {
        "nominal": (0.25, 0.45),
        "current15": (0.375, 0.675),
        "current20": (0.5, 0.9),
        "plant20": (0.25, 0.45),
        "combined20": (0.5, 0.9),
    }
    manifest = _manifest()
    for level, (minimum, maximum) in expected.items():
        config = load_config(REPO_ROOT / manifest["levels"][level]["path"])
        assert config.current.min_speed_mps == pytest.approx(minimum), level
        assert config.current.max_speed_mps == pytest.approx(maximum), level


def test_goal148_plant_randomization_only_on_plant_levels() -> None:
    manifest = _manifest()
    for level, entry in manifest["levels"].items():
        config = load_config(REPO_ROOT / entry["path"])
        if level in {"plant20", "combined20"}:
            assert config.plant_randomization.enabled, level
            assert config.plant_randomization.seed == 13420, level
            assert config.plant_randomization.thrust_scale == pytest.approx((0.8, 1.2)), level
        else:
            assert not config.plant_randomization.enabled, level


def test_goal148_k4_controller_override_is_only_pitch_gain() -> None:
    base = load_config(REPO_ROOT / "configs/lowlevel/goal136/curriculum/C0.json").controller
    k4 = load_config(REPO_ROOT / "configs/lowlevel/goal147/controller_k4.json").controller
    assert k4.pitch_gain == pytest.approx(4.0)
    assert k4.pitch_rate_gain == pytest.approx(base.pitch_rate_gain)
    assert k4.heading_gain == pytest.approx(base.heading_gain)


def test_goal148_curriculum_current_override_keeps_other_c1_fields() -> None:
    base = _build_curriculum_sampler("C1")
    stressed = _build_curriculum_sampler(
        "C1", current_speed_range_mps=(0.5, 0.9), shear_scale=2.0
    )
    assert base is not None and stressed is not None
    for seed in (20, 33, 59):
        baseline = base.sample(seed)
        sample = stressed.sample(seed)
        assert 0.5 <= sample.current_speed_mps <= 0.9
        assert sample.shear_scale == pytest.approx(2.0)
        assert sample.depth_m == pytest.approx(baseline.depth_m)
        assert sample.path_family == baseline.path_family
        assert sample.noise_scale == pytest.approx(baseline.noise_scale)
        assert sample.length_m == pytest.approx(baseline.length_m)
        assert sample.speed_mps == pytest.approx(baseline.speed_mps)


def test_goal148_curriculum_override_requires_a_stage_or_valid_range() -> None:
    with pytest.raises(ValueError, match="ordered"):
        _build_curriculum_sampler("C1", current_speed_range_mps=(0.5, 0.4))
    with pytest.raises(ValueError, match="require a curriculum stage"):
        _build_curriculum_sampler(None, shear_scale=1.0)
