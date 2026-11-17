from __future__ import annotations

from pathlib import Path

from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "configs" / "lowlevel" / "goal182"
SEEDS = (86201, 86202, 86203)


def test_goal182_formal_configs_have_full_curriculum_and_three_seeds() -> None:
    paths = sorted(GOAL.glob("train_formal_*_5d_m_seed*.json"))

    assert len(paths) == 12
    for path in paths:
        config = load_training_config(path)
        assert config.iterations == 2600
        assert config.training_profile == "PPO-FLAT-4096x48-E5-MB4"
        assert any(str(seed) in path.name for seed in SEEDS)
        environment = load_config(ROOT / Path(config.environment_config))
        assert environment.curriculum.enabled is True
        assert environment.curriculum.promotion_enabled is True
        assert environment.curriculum.initial_stage == "C0"
        assert environment.curriculum.stage_mix == (("C0", 0.9), ("C1", 0.1))


def test_goal182_formal_alos_configs_use_a3_vertical_gain() -> None:
    for mode in ("trres_cmd", "trres_act"):
        for seed in SEEDS:
            path = GOAL / f"train_formal_{mode}_5d_m_seed{seed}.json"
            config = load_training_config(path)
            environment_path = Path(config.environment_config)
            assert environment_path.name == "env_alos_formal_a3_3d.json"
            environment = load_config(ROOT / environment_path)
            assert environment.guidance.vertical_gain == 2.5


def test_goal182_formal_envs_use_frozen_60s_episode_protocol() -> None:
    for name in ("env_path_fixed_speed_formal.json", "env_alos_formal_a3_3d.json"):
        environment = load_config(GOAL / name)
        assert environment.plant.max_episode_steps == 3000
        assert environment.plant.dt_s == 0.02


def test_goal182_formal_curriculum_sampler_covers_all_promotion_stages() -> None:
    from auv_lowlevel.curriculum import (
        CURRICULUM_STAGE_ORDER,
        CurriculumStageSampler,
        curriculum_stage_mix_target,
        curriculum_stages_with_overrides,
    )

    for name in ("env_path_fixed_speed_formal.json", "env_alos_formal_a3_3d.json"):
        environment = load_config(GOAL / name)
        stages = curriculum_stages_with_overrides(
            [stage for stage, _weight in environment.curriculum.stage_mix]
        )
        assert set(stages) == set(CURRICULUM_STAGE_ORDER)
        sampler = CurriculumStageSampler(
            stages=stages, stage_mix=environment.curriculum.stage_mix
        )
        for stage in CURRICULUM_STAGE_ORDER:
            sampler.set_stage_mix(
                curriculum_stage_mix_target(stage, preview_enabled=True)
            )
            assert sampler.sample(12345, stage=stage).stage == stage
