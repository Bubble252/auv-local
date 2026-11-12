from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from auv_lowlevel.config import CurriculumConfig, load_config
from auv_lowlevel.curriculum import (
    CurriculumStageSampler,
    curriculum_stages_with_overrides,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs/lowlevel/goal150/profile_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_goal150_env_keeps_success_contract_and_k4() -> None:
    base = load_config(REPO_ROOT / "configs/lowlevel/goal148/env_nominal.json")
    env = load_config(REPO_ROOT / "configs/lowlevel/goal150/env_current07_k4.json")
    assert env.plant.goal_radius_m == pytest.approx(1.25)
    assert env.reward == base.reward
    assert env.termination == base.termination
    assert env.guidance == base.guidance
    assert env.controller.pitch_gain == pytest.approx(4.0)
    assert env.controller.pitch_rate_gain == pytest.approx(base.controller.pitch_rate_gain)
    assert env.curriculum.enabled
    assert env.curriculum.stage_mix == (("C1", 1.0),)
    assert env.curriculum.current_speed_range_override_mps == pytest.approx((0.275, 0.50))
    assert env.curriculum.shear_scale_override == pytest.approx(0.75)


def test_goal150_sampler_uses_current07_override() -> None:
    env = load_config(REPO_ROOT / "configs/lowlevel/goal150/env_current07_k4.json")
    stages = curriculum_stages_with_overrides(
        [stage for stage, _weight in env.curriculum.stage_mix],
        current_speed_range_mps=env.curriculum.current_speed_range_override_mps,
        shear_scale=env.curriculum.shear_scale_override,
    )
    sampler = CurriculumStageSampler(stages=stages, stage_mix=env.curriculum.stage_mix)
    for seed in (20, 33, 59):
        spec = sampler.sample(seed)
        assert 0.275 <= spec.current_speed_mps <= 0.50
        assert spec.shear_scale == pytest.approx(0.75)


def test_goal150_ppo_profiles_are_one_million_and_vary_entropy() -> None:
    manifest = _manifest()
    entropies = []
    for name in ("ppo_ent1e4", "ppo_ent1e3", "ppo_ent1e2"):
        path = REPO_ROOT / manifest["profiles"][name]["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        entropies.append(payload["algorithm"]["entropy_coef"])
        assert payload["environment"]["config"] == "configs/lowlevel/goal150/env_current07_k4.json"
        assert payload["environment"]["num_envs"] * payload["runner"]["num_steps_per_env"] * payload["training"]["iterations"] == 1_000_000
        assert payload["training"]["initial_residual_zero"] is False
    assert entropies == pytest.approx([1e-4, 1e-3, 1e-2])


def test_goal150_sac_profile_is_one_million_command_residual() -> None:
    manifest = _manifest()
    payload = json.loads(
        (REPO_ROOT / manifest["profiles"]["sac"]["path"]).read_text(encoding="utf-8")
    )
    assert payload["environment_config"] == "configs/lowlevel/goal150/env_current07_k4.json"
    assert payload["action_interface"] == "command"
    assert payload["training"]["total_env_transitions"] == 1_000_192


def test_goal150_override_requires_enabled_curriculum() -> None:
    with pytest.raises(ValueError, match="require curriculum.enabled"):
        replace(CurriculumConfig(), current_speed_range_override_mps=(0.275, 0.50))
    with pytest.raises(ValueError, match="MIN <= MAX"):
        CurriculumConfig(
            enabled=True,
            stage_mix=(("C1", 1.0),),
            current_speed_range_override_mps=(0.5, 0.4),
        )
