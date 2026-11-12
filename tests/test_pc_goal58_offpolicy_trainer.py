from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from auv_lowlevel.config import load_config
from auv_rl.ddpg.agent import DDPGAgent, DDPGConfig
from auv_rl.ddpg.pc_command_residual_env import (
    PCCommandResidualOffPolicyEnv,
    zero_initialize_pc_residual_actor,
)
from auv_rl.ddpg.pc_vector_trainer import train_pc_command_residual_ddpg

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = REPO_ROOT / "configs" / "lowlevel" / "pc_goal27_r1_reward_v2_default_20261109.json"


def _env() -> PCCommandResidualOffPolicyEnv:
    config = load_config(ENV_CONFIG)
    config = replace(
        config,
        plant=replace(config.plant, max_episode_steps=8),
        reference=replace(config.reference, path_bank_size=8, path_bank_seed=41),
        evaluation=replace(config.evaluation, training_seed_start=240),
    )
    return PCCommandResidualOffPolicyEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        path_dim=8,
    )


def test_trainer_writes_checkpoint_and_residual_metrics(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    env = _env()
    agent = DDPGAgent(
        DDPGConfig(
            observation_dim=env.observation_dim,
            action_low=(-1.0, -1.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
            hidden_sizes=(16,),
            gru_hidden_size=8,
            history_length=3,
            batch_size=4,
            buffer_size=64,
            learning_starts=4,
            privileged_dim=env.privileged_dim,
            actor_output_sigma_init=0.1,
            actor_hidden_sigma_init=0.2,
            per_beta_steps=16,
            policy_delay=1,
            twin_critics=True,
            device="cpu",
        ),
        seed=240,
    )
    zero_initialize_pc_residual_actor(agent)
    metrics = train_pc_command_residual_ddpg(
        env,
        agent,
        total_env_transitions=8,
        seed=240,
        update_to_data_ratio=1.0,
        output=tmp_path,
    )
    assert metrics.env_transitions >= 8
    assert metrics.gradient_steps > 0
    assert np.isfinite(metrics.residual_abs_mean)
    assert (tmp_path / "final.pt").is_file()
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "training_metrics.json").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "training_curve.csv").is_file()

    loaded = DDPGAgent.load(
        tmp_path / "final.pt",
        device="cpu",
        expected_observation_version=env.observation_version,
    )
    observations, _ = env.reset([20, 21])
    for env_id in range(env.num_envs):
        loaded.begin_episode_for_env(env_id, int(env.episode_ids[env_id]))
    before = loaded.act_batch([0, 1], observations, evaluate=True)
    for env_id in range(env.num_envs):
        loaded.begin_episode_for_env(env_id, int(env.episode_ids[env_id]))
    after = loaded.act_batch([0, 1], observations, evaluate=True)
    assert np.allclose(before, after, atol=1e-9)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observation_dim"] == env.observation_dim
    assert manifest["privileged_dim"] == env.privileged_dim
