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
