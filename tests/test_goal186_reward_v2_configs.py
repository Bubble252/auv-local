from __future__ import annotations

from pathlib import Path

import pytest

from auv_lowlevel.config import load_config
from auv_lowlevel.ppo_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
GOAL186 = ROOT / "configs/lowlevel/goal186"
ENV_CONFIG = GOAL186 / "env_path_fixed_speed_formal_empirical_reward_v2.json"
MODULES = ("mod4", "mod3", "mod2", "mod1", "mod0")


def test_goal186_env_declares_only_the_approved_reward_delta() -> None:
    config = load_config(ENV_CONFIG)

    assert config.reward.w_action_feasible_gap_pc == pytest.approx(5.0)
    assert config.reward.w_path_vertical == pytest.approx(2.5)
    assert config.reward.w_sat_pc == pytest.approx(0.0)
    assert config.reward.w_action_clip_attempt_pc == pytest.approx(0.0)


@pytest.mark.parametrize("module", MODULES)
def test_goal186_training_configs_use_reward_v2_and_freeze_profile(module: str) -> None:
    config = load_training_config(
        GOAL186 / f"train_formal_{module}_pe2e_bcres_reward_v2_seed86201.json"
    )
    environment = load_config(ROOT / config.environment_config)

    assert environment.reward.w_action_feasible_gap_pc == pytest.approx(5.0)
    assert environment.reward.w_path_vertical == pytest.approx(2.5)
    assert environment.reward.w_sat_pc == pytest.approx(0.0)
    assert config.training_profile == "PPO-FLAT-4096x48-E5-MB4"
    assert environment.curriculum.enabled is True
    assert environment.curriculum.stage_mix


def test_goal186_mod4_is_mod3_plus_privileged_critic_path() -> None:
    mod3 = load_training_config(GOAL186 / "train_formal_mod3_pe2e_bcres_reward_v2_seed86201.json")
    mod4 = load_training_config(GOAL186 / "train_formal_mod4_pe2e_bcres_reward_v2_seed86201.json")

    assert mod4.pc.critic_path_dim == 30
    assert mod4.pc.actor_current_encoder_enabled is True
    assert mod4.pc.dade_enabled is True
    assert mod4.base_checkpoint == mod3.base_checkpoint
