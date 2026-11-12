from __future__ import annotations

from pathlib import Path

from auv_lowlevel.batch.pc_path import PC_PATH_DEFAULT_BLOCKS
from auv_lowlevel.batch.pc_policy import pc_schema_hash
from auv_lowlevel.ppo_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "configs/lowlevel/odarrl_control_ppo_pc_v0_fixed_speed_alpha1_lr1e-4_dz64.json"
V1 = ROOT / "configs/lowlevel/odarrl_control_ppo_pc_v1_learned_speed_alpha1_lr1e-4_dz64.json"


def test_v0_v1_training_contracts_differ_only_in_actor_speed_input() -> None:
    v0 = load_training_config(V0)
    v1 = load_training_config(V1)

    assert v0.pc is not None and v1.pc is not None
    assert v0.pc.actor_command_dim == 1
    assert v1.pc.actor_command_dim == 0
    assert v0.pc.critic_command_dim == v1.pc.critic_command_dim == 1
    assert v0.pc.path_dim == v1.pc.path_dim == 8
    assert v0.pc.history_path_dim == v1.pc.history_path_dim == 8
    assert v0.pc.dade_history_mode == v1.pc.dade_history_mode == "encoded"
    assert v0.pc.path_curriculum == v1.pc.path_curriculum == ()
    assert v0.pc.path_encoder_alpha == v1.pc.path_encoder_alpha == 1.0
    assert (
        float(v0.algorithm["learning_rate_actor"])
        == float(v1.algorithm["learning_rate_actor"])
        == 1e-4
    )
    assert v0.history is not None and v1.history is not None
    assert (
        v0.history.short_window,
        v0.history.long_window,
        v0.history.short_stride,
        v0.history.long_stride,
    ) == (
        v1.history.short_window,
        v1.history.long_window,
        v1.history.short_stride,
        v1.history.long_stride,
    )


def test_v0_v1_schema_keeps_path_and_history_identical() -> None:
    common = {
        "path_dim": 8,
        "path_blocks": PC_PATH_DEFAULT_BLOCKS,
        "dade_history_mode": "encoded",
        "critic_command_dim": 1,
        "history_path_dim": 8,
    }
    v0_hash = pc_schema_hash(actor_command_dim=1, **common)
    v1_hash = pc_schema_hash(actor_command_dim=0, **common)

    assert v0_hash != v1_hash
    assert pc_schema_hash(actor_command_dim=1, **common) == v0_hash
    assert pc_schema_hash(actor_command_dim=0, **common) == v1_hash
