from __future__ import annotations

import sys
from pathlib import Path

import pytest

from auv_lowlevel.ppo_config import load_training_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pc_p1_optimization_20261105 as module  # noqa: E402


def test_x1_changes_only_path_encoder_alpha() -> None:
    original = load_training_config(
        Path(
            "configs/lowlevel/"
            "odarrl_control_ppo_pc_abc_p1_full_path_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
        )
    )
    optimized = load_training_config(Path(module.ALPHA1_CONFIG))

    assert optimized.pc is not None
    assert original.pc is not None
    assert optimized.pc.path_encoder_alpha == 1.0
    assert original.pc.path_encoder_alpha == 0.01
    assert optimized.pc.path_encoder_hidden_dims == original.pc.path_encoder_hidden_dims
    assert optimized.pc.path_dim == original.pc.path_dim
    assert optimized.environment_config == original.environment_config
    assert optimized.policy == original.policy
    assert optimized.runner == original.runner
    assert optimized.algorithm == original.algorithm


def test_x1_and_x2_keep_fixed_seed_and_ablation_contract() -> None:
    x1 = module._plan(module.ARMS["x1"])
    x2 = module._plan(module.ARMS["x2"])

    assert x1.iterations == 32
    assert x2.iterations == 128
    assert x2.num_envs == x1.num_envs == 128
    assert x2.num_steps == x1.num_steps == 256
    assert x2.train_seed == x1.train_seed == 91430
    assert x2.eval_seed_start == x1.eval_seed_start == 1000
    assert x2.eval_seed_count == x1.eval_seed_count == 20
    assert x2.path_ablations == x1.path_ablations == module.p1.PATH_ABLATIONS


def test_baseline_resolution_requires_an_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        module._resolve_baseline_eval_root(tmp_path / "missing")


def test_optimizer_arms_change_only_their_declared_variable() -> None:
    reference = load_training_config(Path(module.ALPHA1_CONFIG))
    reward_scale = load_training_config(Path(module.REWARD_SCALE1_CONFIG))
    actor_lr = load_training_config(Path(module.ACTOR_LR_1E4_CONFIG))
    minibatch_norm = load_training_config(Path(module.MINIBATCH_ADV_NORM_CONFIG))

    assert reward_scale.environment_config != reference.environment_config
    assert reward_scale.algorithm == reference.algorithm
    assert reward_scale.pc == reference.pc

    assert actor_lr.environment_config == reference.environment_config
    assert actor_lr.pc == reference.pc
    assert actor_lr.algorithm["learning_rate_actor"] == 1e-4
    assert actor_lr.algorithm["learning_rate_critic"] == reference.algorithm[
        "learning_rate_critic"
    ]

    assert minibatch_norm.environment_config == reference.environment_config
    assert minibatch_norm.pc == reference.pc
    assert minibatch_norm.algorithm["normalize_advantage_per_mini_batch"] is True

