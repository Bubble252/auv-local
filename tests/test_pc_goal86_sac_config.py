from __future__ import annotations

import json
from pathlib import Path

import pytest

from auv_lowlevel.batch.pc_bc import read_pc_bc_checkpoint_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
T1_CONFIG = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_t1_sac_default_20261110.json"
PA_CONFIG = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_pa_sac_default_20261110.json"
D3_CHECKPOINT = REPO_ROOT / "checkpoints/goal26/d3/pc_bc_actor.pt"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_t1_sac_uses_full_path_z_t_without_bc_prior() -> None:
    payload = _load(T1_CONFIG)
    policy = payload["policy"]
    observation = payload["observation"]
    assert payload["action_interface"] == "command"
    assert payload["actuator_residual_scale"] == pytest.approx(0.25)
    assert policy["architecture"] == "pc_z_t"
    assert policy["bc_checkpoint"] is None
    assert observation["path_dim"] == 12
    assert policy["bc_path_dim"] == 12
    assert observation["path_dim"] - policy["bc_path_dim"] == 0
    assert payload["training"]["net_arch"] == [256, 128]
    assert payload["training"]["num_envs"] == 256
    assert payload["training"]["screen_env_transitions"] == 1_000_000
    assert payload["training"]["total_env_transitions"] == 5_000_000
    assert payload["experiment"]["train_seeds"] == [86001, 86002, 86003]
    assert (REPO_ROOT / payload["environment_config"]).is_file()


@pytest.mark.skipif(not D3_CHECKPOINT.is_file(), reason="D3 BC checkpoint is unavailable")
def test_pa_sac_explicitly_splits_path8_and_endpoint4() -> None:
    payload = _load(PA_CONFIG)
    policy = payload["policy"]
    observation = payload["observation"]
    assert payload["action_interface"] == "actuator"
    assert payload["actuator_residual_scale"] == pytest.approx(0.25)
    assert policy["architecture"] == "pc_z_t"
    assert observation["path_dim"] == 12
    assert policy["bc_path_dim"] == 8
    assert observation["path_dim"] - policy["bc_path_dim"] == 4
    assert policy["bc_checkpoint"] == "checkpoints/goal26/d3/pc_bc_actor.pt"
    metadata = read_pc_bc_checkpoint_metadata(REPO_ROOT / policy["bc_checkpoint"])
    assert metadata["path_dim"] == policy["bc_path_dim"]
    assert metadata["actor_hidden_dims"] == payload["training"]["net_arch"]
    assert metadata["actor_command_dim"] == 0
    assert metadata["actor_output_dim"] == 3


@pytest.mark.parametrize(
    ("path", "seeds"),
    [
        (T1_CONFIG, [86001, 86002, 86003]),
        (PA_CONFIG, [86101, 86102, 86103]),
    ],
)
def test_sac_configs_freeze_identical_budget_and_three_seeds(path: Path, seeds: list[int]) -> None:
    payload = _load(path)
    assert payload["training"]["num_envs"] == 256
    assert payload["training"]["screen_env_transitions"] == 1_000_000
    assert payload["training"]["total_env_transitions"] == 5_000_000
    assert payload["experiment"]["train_seeds"] == seeds
