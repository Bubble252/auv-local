from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "configs" / "lowlevel" / "goal182"
SEEDS = (86201, 86202, 86203)


def _module_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_goal182_formal_mod4_configs_are_complete_and_three_seed() -> None:
    paths = sorted(GOAL.glob("train_formal_mod4_*_5d_m_seed*.json"))

    assert len(paths) == 18
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


def test_goal182_formal_mod4_configs_encode_full_module_set() -> None:
    for path in sorted(GOAL.glob("train_formal_mod4_*_5d_m_seed*.json")):
        payload = _module_payload(path)
        pc = payload["pc"]
        assert pc["actor_current_encoder_enabled"] is True
        assert pc["critic_path_dim"] == 30
        assert pc["dade_enabled"] is True
        assert pc["path_dim"] == 5
        assert pc["bc_path_dim"] == 5
        expected_command_dim = 0 if "_pe2e_" in path.name else 3
        assert pc["actor_command_dim"] == expected_command_dim


def test_goal182_formal_mod4_bc_warmstarts_use_current_encoder_checkpoints() -> None:
    expected = {
        "pe2e": "d85718a432a287f2ae255528f645bc20f4445aba875c9fb9c44f008c578094b2",
        "alosb": "47cda13ed97e77d372d1495096dd72259987c5466c13a5d2023f45000b402749",
    }
    for path in sorted(GOAL.glob("train_formal_mod4_*_5d_m_seed*.json")):
        payload = _module_payload(path)
        name = path.name
        if "_finetune_" not in name and "_residual_" not in name:
            continue
        control = "pe2e" if "_pe2e_" in name else "alosb"
        if "_finetune_" in name:
            assert payload["experiment"]["warmstart_actor_sha256"] == expected[control]
        else:
            assert payload["base_policy"]["checkpoint_sha256"] == expected[control]


def test_goal182_formal_mod2_configs_remain_labeled_as_mod2() -> None:
    paths = sorted(GOAL.glob("train_formal_mod2_*_5d_m_seed*.json"))

    assert len(paths) == 12
    for path in paths:
        config = load_training_config(path)
        assert config.iterations == 2600
        payload = _module_payload(path)
        assert payload["pc"]["actor_current_encoder_enabled"] is False
        assert payload["pc"]["critic_path_dim"] == 30


def test_goal182_formal_alos_mod4_configs_use_a3_vertical_gain() -> None:
    for path in sorted(GOAL.glob("train_formal_mod4_alosb_*_5d_m_seed*.json")):
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


def test_goal182_matrix_checker_rejects_mod2_config_as_mod4() -> None:
    spec = importlib.util.spec_from_file_location(
        "check_goal182_matrices", ROOT / "scripts" / "check_goal182_matrices.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arm = {
        "arm_id": "G182_INT_STACK_MLP_PE2E_BCFT_MOD4",
        "learning_composition": "BCFT",
    }
    errors = module._validate_formal_config(
        arm,
        "PE2E",
        "MOD4",
        "configs/lowlevel/goal182/train_formal_mod2_pe2e_finetune_5d_m_seed86201.json",
    )
    assert any("MODULE_SET_CONFIG_MISMATCH" in error for error in errors)
    assert any("BC_CHECKPOINT_INCOMPATIBLE" in error for error in errors)
