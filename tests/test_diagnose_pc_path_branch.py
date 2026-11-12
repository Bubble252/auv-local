from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from auv_lowlevel.ppo_config import load_training_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_pc_path_branch import diagnose  # noqa: E402


def test_diagnose_script_writes_path_effective_scale(tmp_path: Path) -> None:
    config = Path(
        "configs/lowlevel/odarrl_control_ppo_pc_abc_p1_full_path_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
    )
    training = load_training_config(config)
    policy_config = training.rsl_train_cfg()["policy"]
    from auv_lowlevel.batch.pc_policy import (
        PC_ACTOR_STATE_DIM,
        PC_CRITIC_STATE_DIM,
        PCActorCritic,
        PCPolicyLayout,
    )

    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, path_dim=8)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        **{key: value for key, value in policy_config.items() if key != "class_name"},
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)

    result = diagnose(config, [checkpoint], bc_checkpoint=None)

    assert result["schema_version"] == "auv_pc_path_branch_diagnosis.v1"
    diagnostics = result["checkpoints"][0]["diagnostics"]
    assert diagnostics["path_encoder_present"] is True
    assert json.dumps(result)


def test_diagnose_script_ignores_boolean_buffers(tmp_path: Path) -> None:
    config = Path(
        "configs/lowlevel/odarrl_control_ppo_pc_abc_p1_full_path_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
    )
    training = load_training_config(config)
    policy_config = training.rsl_train_cfg()["policy"]
    from auv_lowlevel.batch.pc_policy import (
        PC_ACTOR_STATE_DIM,
        PC_CRITIC_STATE_DIM,
        PCActorCritic,
        PCPolicyLayout,
    )

    actor_layout = PCPolicyLayout(state_dim=PC_ACTOR_STATE_DIM, path_dim=8)
    critic_layout = PCPolicyLayout(state_dim=PC_CRITIC_STATE_DIM)
    policy = PCActorCritic(
        actor_layout.flat_dim,
        critic_layout.flat_dim,
        3,
        **{key: value for key, value in policy_config.items() if key != "class_name"},
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
    bc_checkpoint = tmp_path / "bc.pt"
    torch.save({"state_dict": policy.state_dict()}, bc_checkpoint)

    result = diagnose(config, [checkpoint], bc_checkpoint=bc_checkpoint)

    assert result["bc_reference"] is not None
    assert all(
        value is None or isinstance(value, float)
        for value in result["bc_reference"]["comparison_to_final"]["norms"].values()
    )

