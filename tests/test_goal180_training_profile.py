from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_goal180_configs as goal180  # noqa: E402

from auv_lowlevel.ppo_config import parse_training_config  # noqa: E402


def test_goal180_generated_training_uses_canonical_profile() -> None:
    payload = goal180._training("P0_PATH_BASE", goal180.BASE_ENV_PATH)
    config = parse_training_config(payload)

    assert config.training_profile == "PPO-FLAT-4096x48-E5-MB4"
    assert config.num_envs == 4096
    assert config.runner["num_steps_per_env"] == 48
    assert config.algorithm["num_learning_epochs"] == 5
    assert config.algorithm["num_mini_batches"] == 4
    assert config.iterations == 5
    assert (
        config.num_envs
        * config.runner["num_steps_per_env"]
        * config.iterations
        == goal180.SCREEN_TRANSITIONS
        == 983_040
    )
    assert goal180.FORMAL_MIN_SEEDS == 3
    assert goal180.FORMAL_ITERATIONS_PER_SEED == 16
