from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
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
        assert environment.curriculum.speed_protocol == "SPEED_V3_CENTER_OUT_5S"


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


def test_goal182_bc_warmstart_finetune_uses_guarded_recipe() -> None:
    paths = sorted(GOAL.glob("train_formal_mod4_*_finetune_5d_m_seed*.json"))

    assert len(paths) == 6
    for path in paths:
        payload = _module_payload(path)
        algorithm = payload["algorithm"]
        assert algorithm["learning_rate_actor"] == 1e-5
        assert algorithm["learning_rate_critic"] == 3e-4
        assert algorithm["schedule"] == "fixed"
        assert payload["runner"]["empirical_normalization"] is False
        assert payload["policy"]["init_noise_std"] == 0.05


def test_goal182_reference_recipe_is_left_for_residual_and_trres() -> None:
    paths = sorted(
        list(GOAL.glob("train_formal_mod4_*_residual_5d_m_seed*.json"))
        + list(GOAL.glob("train_formal_mod4_*_trres_*_5d_m_seed*.json"))
    )

    assert len(paths) == 12
    for path in paths:
        payload = _module_payload(path)
        assert payload["algorithm"]["learning_rate_actor"] == 3e-4
        assert payload["algorithm"]["learning_rate_critic"] == 3e-4
        assert payload["algorithm"]["schedule"] == "adaptive"
        assert payload["runner"]["empirical_normalization"] is True
        assert payload["policy"]["init_noise_std"] == 0.3


def test_goal182_c0_recipe_diagnostic_is_c0_only_and_guarded() -> None:
    path = GOAL / "train_c0_diagnostic_mod4_pe2e_finetune_5d_m_seed86201.json"
    config = load_training_config(path)
    payload = _module_payload(path)

    assert config.iterations == 72
    environment = load_config(ROOT / Path(config.environment_config))
    assert environment.curriculum.initial_stage == "C0"
    assert environment.curriculum.promotion_enabled is False
    assert environment.curriculum.stage_mix == (("C0", 1.0),)
    assert payload["algorithm"]["learning_rate_actor"] == 1e-5
    assert payload["algorithm"]["learning_rate_critic"] == 3e-4
    assert payload["algorithm"]["schedule"] == "fixed"
    assert payload["runner"]["empirical_normalization"] is False
    assert payload["policy"]["init_noise_std"] == 0.05
    assert "/diagnostic/" in payload["experiment"]["output"]


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
        assert environment.curriculum.speed_protocol == "SPEED_V3_CENTER_OUT_5S"


def test_goal182_envs_only_use_default_two_metre_radius() -> None:
    for path in sorted(GOAL.glob("*.json")):
        payload = _module_payload(path)
        plant = payload.get("plant")
        if not isinstance(plant, dict) or "goal_radius_m" not in plant:
            continue
        assert float(plant["goal_radius_m"]) == 2.0


def test_goal182_ppo_trainer_has_no_strict125_eval_path() -> None:
    source = (ROOT / "scripts" / "train_batch_lowlevel_ppo.py").read_text(encoding="utf-8")
    assert "curriculum-eval-strict-config" not in source
    assert "strict125" not in source
    assert "combine_eval_summaries" not in source


def test_goal182_ppo_trainer_saves_zero_update_checkpoint_for_pc_arms() -> None:
    source = (ROOT / "scripts" / "train_batch_lowlevel_ppo.py").read_text(encoding="utf-8")
    assert 'if is_pc:\n            initial_checkpoint_path = output / "model_initial.pt"' in source
    assert (
        'if frozen_bc_residual:\n            initial_checkpoint_path = output / "model_initial.pt"'
        not in source
    )


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
            [stage for stage, _weight in environment.curriculum.stage_mix],
            speed_protocol=environment.curriculum.speed_protocol,
        )
        assert set(stages) == set(CURRICULUM_STAGE_ORDER)
        sampler = CurriculumStageSampler(
            stages=stages,
            stage_mix=environment.curriculum.stage_mix,
            speed_protocol=environment.curriculum.speed_protocol,
        )
        for stage in CURRICULUM_STAGE_ORDER:
            sampler.set_stage_mix(curriculum_stage_mix_target(stage, preview_enabled=True))
            assert sampler.sample(12345, stage=stage).stage == stage


def test_goal182_formal_env_resamples_speed_every_five_seconds() -> None:
    from auv_lowlevel.curriculum import (
        CurriculumStageSampler,
        curriculum_stages_with_overrides,
    )

    environment = load_config(GOAL / "env_path_fixed_speed_formal.json")
    stages = curriculum_stages_with_overrides(
        [stage for stage, _weight in environment.curriculum.stage_mix],
        speed_protocol=environment.curriculum.speed_protocol,
    )
    sampler = CurriculumStageSampler(
        stages=stages,
        stage_mix=(("C0", 1.0),),
        current_variation_speed_amplitude_mps=0.10,
        speed_protocol=environment.curriculum.speed_protocol,
    )
    env = BatchedPCLowLevelEnv(
        environment,
        num_envs=2,
        curriculum_sampler=sampler,
        path_dim=5,
        device="cpu",
        dtype=torch.float64,
    )
    env.reset([90000, 90001])
    specs = env.curriculum_specs
    assert specs is not None and len(specs) == 2
    for _ in range(260):
        result = env.step(torch.zeros((2, 3), dtype=torch.float64), auto_reset=False)
    assert torch.equal(
        result.info["speed_segment_index"],
        torch.ones(2, dtype=torch.long),
    )
    assert torch.allclose(
        result.info["speed_target_mps"],
        torch.tensor([spec.speed_segment_plan_mps[1] for spec in specs], dtype=torch.float64),
    )
    assert torch.allclose(
        result.info["speed_resample_period_s"],
        torch.full((2,), 5.0, dtype=torch.float64),
    )


def test_goal182_eval_sampler_stays_fixed_speed() -> None:
    from scripts.evaluate_pc_ppo import _build_curriculum_sampler

    from auv_lowlevel.curriculum import CURRICULUM_STAGE_ORDER

    for stage in CURRICULUM_STAGE_ORDER:
        sampler = _build_curriculum_sampler(stage)
        assert sampler is not None
        assert sampler.speed_protocol == "legacy_v1"
        spec = sampler.sample(12345, stage=stage)
        assert spec.speed_resample_period_s is None
        assert spec.speed_segment_plan_mps == (spec.speed_mps,)


def test_goal182_c2_axis_paths_are_geometry_isolated() -> None:
    from dataclasses import replace

    from auv_lowlevel.curriculum import (
        CurriculumStageSampler,
        curriculum_stages_with_overrides,
    )
    from auv_lowlevel.path import (
        apply_curriculum_vertical_profile,
        generate_curriculum_axis_path,
        generate_curriculum_path,
    )

    environment = load_config(GOAL / "env_path_fixed_speed_formal.json")
    stages = curriculum_stages_with_overrides(
        ["C2", "C3", "C4", "C5", "C6"],
        speed_protocol=environment.curriculum.speed_protocol,
    )
    sampler = CurriculumStageSampler(
        stages=stages,
        stage_mix=(("C2", 1.0),),
        speed_protocol=environment.curriculum.speed_protocol,
    )
    specs = sampler.sample_batch(range(20), stage="C2")
    vertical = next(spec for spec in specs if spec.route_axis == "vert_only")
    horizontal = next(spec for spec in specs if spec.route_axis == "horiz_only")
    reference = replace(
        environment.reference,
        length_m=vertical.length_m,
        depth_start_m=vertical.depth_m,
        depth_amplitude_m=vertical.vertical_amplitude_m,
        depth_cycles=vertical.vertical_cycles,
        lateral_amplitude_m=0.0,
    )
    vertical_path = generate_curriculum_axis_path(reference, route_axis="vert_only", seed=1)
    assert max(abs(vertical_path.points_ned_m[:, 1])) < 1e-9
    vertical_z = vertical_path.points_ned_m[:, 2]
    assert float(vertical_z.max() - vertical_z.min()) > 1.0

    reference = replace(
        reference,
        depth_start_m=horizontal.depth_m,
        depth_amplitude_m=0.0,
        lateral_amplitude_m=horizontal.horizontal_amplitude_m,
        lateral_cycles=1.0,
    )
    horizontal_path = generate_curriculum_axis_path(reference, route_axis="horiz_only", seed=1)
    assert max(abs(horizontal_path.points_ned_m[:, 2] - horizontal.depth_m)) < 1e-9
    horizontal_y = horizontal_path.points_ned_m[:, 1]
    assert float(horizontal_y.max() - horizontal_y.min()) > 1.0

    coupled_spec = sampler.sample(12345, stage="C6")
    base_reference = replace(
        environment.reference,
        length_m=coupled_spec.length_m,
        depth_start_m=coupled_spec.depth_m,
    )
    base = generate_curriculum_path(base_reference, coupled_spec.path_family, seed=7)
    coupled = apply_curriculum_vertical_profile(
        base_reference,
        base,
        amplitude_m=coupled_spec.vertical_amplitude_m,
        cycles=coupled_spec.vertical_cycles,
    )
    coupled_range = float(coupled.points_ned_m[:, 2].max() - coupled.points_ned_m[:, 2].min())
    base_range = float(base.points_ned_m[:, 2].max() - base.points_ned_m[:, 2].min())
    assert coupled_range > base_range + 1.0


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
