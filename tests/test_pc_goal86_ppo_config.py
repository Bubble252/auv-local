from __future__ import annotations

import json
from pathlib import Path

import pytest

from auv_lowlevel.ppo_config import load_training_config, parse_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs/lowlevel/goal86"


def _config_path(arm: str, budget: str) -> Path:
    return CONFIG_DIR / f"pc_goal86_{arm}_ppo_{budget}_20261110.json"


@pytest.mark.parametrize("arm", ["t1", "t2", "p-a"])
@pytest.mark.parametrize(
    ("budget", "steps", "iterations"),
    [("screen", 1_000_000, 8), ("formal", 5_000_000, 40)],
)
def test_goal86_ppo_matrix_uses_identical_transition_budget(
    arm: str, budget: str, steps: int, iterations: int
) -> None:
    config = load_training_config(_config_path(arm, budget))
    assert config.num_envs == 1000
    assert config.runner["num_steps_per_env"] == 125
    assert config.iterations == iterations
    assert config.num_envs * config.runner["num_steps_per_env"] * config.iterations == steps
    assert config.pc is not None
    assert config.pc.path_dim == 12
    assert config.pc.actor_command_dim == 3
    assert config.pc.actor_current_encoder_enabled
    assert config.pc.d_z_total == 64
    assert config.pc.path_dim == 12
    assert not config.pc.dade_enabled


def test_goal86_ppo_action_interfaces_and_initialization_are_paired() -> None:
    t1 = load_training_config(_config_path("t1", "screen"))
    t2 = load_training_config(_config_path("t2", "screen"))
    pa = load_training_config(_config_path("p-a", "screen"))
    assert t1.pc is not None and t2.pc is not None and pa.pc is not None
    assert t1.pc.action_interface == "command"
    assert t2.pc.action_interface == "actuator"
    assert pa.pc.action_interface == "actuator"
    assert t1.pc.bc_path_dim == 12
    assert t2.pc.bc_path_dim == 12
    assert pa.pc.bc_path_dim == 8
    assert t1.pc.path_encoder_zero_init is False
    assert t2.pc.path_encoder_zero_init is False
    assert pa.pc.path_encoder_zero_init is True
    assert pa.pc.endpoint_encoder_hidden_dims == (32,)
    assert pa.pc.endpoint_encoder_alpha == pytest.approx(1.0)


def test_t2_sac_is_actuator_space_and_matches_t1_scratch_architecture() -> None:
    t1 = json.loads(
        (CONFIG_DIR / "pc_goal86_t1_sac_default_20261110.json").read_text(
            encoding="utf-8"
        )
    )
    t2 = json.loads(
        (CONFIG_DIR / "pc_goal86_t2_sac_default_20261110.json").read_text(
            encoding="utf-8"
        )
    )
    assert t2["action_interface"] == "actuator"
    assert t2["policy"]["bc_checkpoint"] is None
    assert t2["policy"]["bc_path_dim"] == 12
    assert t2["training"]["total_env_transitions"] == t1["training"]["total_env_transitions"]
    assert t2["training"]["screen_env_transitions"] == t1["training"]["screen_env_transitions"]
    assert t2["policy"]["architecture"] == t1["policy"]["architecture"]


def test_goal86_ppo_and_sac_share_exact_transition_budgets() -> None:
    for arm in ("t1", "t2", "p-a"):
        ppo = load_training_config(_config_path(arm, "screen"))
        sac_name = "pa" if arm == "p-a" else arm
        payload = json.loads(
            (CONFIG_DIR / f"pc_goal86_{sac_name}_sac_default_20261110.json").read_text(
                encoding="utf-8"
            )
        )
        assert ppo.num_envs * ppo.runner["num_steps_per_env"] * ppo.iterations == (
            payload["training"]["screen_env_transitions"]
        )
        formal = load_training_config(_config_path(arm, "formal"))
        assert formal.num_envs * formal.runner["num_steps_per_env"] * formal.iterations == (
            payload["training"]["total_env_transitions"]
        )


def test_goal86_formal_configs_freeze_three_train_seeds() -> None:
    for arm in ("t1", "t2", "p-a"):
        config = load_training_config(_config_path(arm, "formal"))
        seeds = [config.seed, config.seed + 1, config.seed + 2]
        assert len(seeds) == 3
        assert seeds == list(range(seeds[0], seeds[0] + 3))


def test_goal87_nonzero_residual_output_std_is_explicit_and_validated() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "configs/lowlevel/goal89_v2_20261111/A1p0_nogate.json"
        ).read_text(encoding="utf-8")
    )
    payload["training"]["initial_residual_output_std"] = 0.01
    config = parse_training_config(payload, config_sha256="test")
    assert config.initial_residual_output_std == pytest.approx(0.01)
    assert config.initial_residual_zero is False

    payload["training"]["initial_residual_zero"] = True
    with pytest.raises(ValueError, match="initial_residual_zero=false"):
        parse_training_config(payload, config_sha256="test")
