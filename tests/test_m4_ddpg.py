from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from auv_rl.ddpg.agent import DDPGAgent, DDPGConfig  # noqa: E402
from auv_rl.ddpg.networks import NoisyLinear  # noqa: E402
from auv_rl.ddpg.replay import PrioritizedSequenceReplayBuffer, SumTree  # noqa: E402
from auv_rl.ddpg.trainer import train_ddpg  # noqa: E402
from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig  # noqa: E402


def test_noisy_linear_eval_uses_mean_weights_and_training_has_learnable_rho() -> None:
    layer = NoisyLinear(4, 3, use_noisy=True)
    layer.eval()
    input_tensor = torch.ones(1, 4)
    first = layer(input_tensor, sample_noise=False)
    second = layer(input_tensor, sample_noise=False)
    assert torch.allclose(first, second)
    assert layer.weight_rho.requires_grad
    assert layer.bias_rho.requires_grad


def test_noisy_linear_uses_factorized_gaussian_weight_noise() -> None:
    layer = NoisyLinear(5, 4, use_noisy=True)
    torch.manual_seed(7)
    noise = layer._sample_noise((4, 5), torch.device("cpu"))
    assert noise.shape == (4, 5)
    row = noise[0]
    assert torch.count_nonzero(row) == row.numel()
    reconstructed = row.unsqueeze(0) * (noise[:, 0:1] / row[0])
    assert torch.allclose(noise, reconstructed, atol=1e-6)

    bias_noise = layer._sample_noise((4,), torch.device("cpu"))
    assert bias_noise.shape == (4,)


def test_ddpg_feature_switches_control_gru_noisy_and_pper() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        gru_hidden_size=4,
        history_length=3,
        use_noisy=False,
        use_gru=False,
        use_pper=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=1)
    assert agent.actor.body.gru is None
    assert agent.critic.body.gru is None
    assert not agent.replay.use_pper
    noisy_layers = [module for module in agent.actor.modules() if isinstance(module, NoisyLinear)]
    assert noisy_layers
    assert all(not layer.use_noisy for layer in noisy_layers)


def test_gru_history_is_reset_between_episodes() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        gru_hidden_size=4,
        history_length=3,
        use_noisy=False,
        use_gru=True,
        use_pper=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=3)
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    agent.begin_episode(0)
    first_action = agent.act(observation, evaluate=True)
    agent.begin_episode(1)
    second_action = agent.act(observation, evaluate=True)
    assert first_action == pytest.approx(second_action)


def test_sumtree_update_and_sequence_replay_stays_within_episode() -> None:
    tree = SumTree(4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    assert tree.total == pytest.approx(3.0)
    index, priority = tree.get(1.5)
    assert index == 1
    assert priority == pytest.approx(2.0)

    replay = PrioritizedSequenceReplayBuffer(
        64,
        observation_dim=3,
        action_dim=2,
        history_length=3,
        use_pper=True,
        seed=2,
    )
    for step in range(6):
        replay.add(
            np.full(3, step, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            float(step),
            np.full(3, step + 1, dtype=np.float32),
            step == 5,
            episode_id=0,
            episode_step=step,
        )
    for step in range(4):
        replay.add(
            np.full(3, 100 + step, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            float(step),
            np.full(3, 100 + step + 1, dtype=np.float32),
            step == 3,
            episode_id=1,
            episode_step=step,
        )
    batch = replay.sample(16, beta=0.5)
    for row in range(batch.episode_ids.shape[0]):
        valid = batch.episode_ids[row][batch.valid_masks[row] > 0.5]
        assert np.all(valid == valid[0])


def test_checkpoint_restore_matches_deterministic_action(tmp_path) -> None:
    config = DDPGConfig(
        observation_dim=5,
        hidden_sizes=(8, 8),
        gru_hidden_size=4,
        history_length=4,
        use_noisy=True,
        use_gru=True,
        use_pper=True,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=11)
    observation = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    agent.begin_episode(0)
    expected = agent.act(observation, evaluate=True)
    checkpoint = tmp_path / "model.pt"
    agent.save(checkpoint)
    restored = DDPGAgent.load(checkpoint)
    restored.begin_episode(0)
    actual = restored.act(observation, evaluate=True)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_agent_update_runs_with_noisy_gru_pper() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        gru_hidden_size=4,
        history_length=3,
        batch_size=4,
        buffer_size=64,
        learning_starts=1,
        use_noisy=True,
        use_gru=True,
        use_pper=True,
        per_beta_steps=10,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=5)
    rng = np.random.default_rng(0)
    agent.begin_episode(0)
    for _idx in range(16):
        observation = rng.normal(size=4).astype(np.float32)
        action = rng.uniform(low=[-0.5, -math.radians(15)], high=[0.5, math.radians(15)])
        next_observation = rng.normal(size=4).astype(np.float32)
        agent.observe(
            observation,
            action.astype(np.float32),
            0.1,
            next_observation,
            False,
        )
    metrics = agent.update(0)
    assert metrics is not None
    for value in (
        metrics.critic_loss,
        metrics.actor_loss,
        metrics.mean_q,
        metrics.mean_td_error,
        metrics.beta,
    ):
        assert np.isfinite(value)


def test_train_ddpg_writes_reproducible_artifacts(tmp_path) -> None:
    env = PlanarAvoidanceEnv(
        PlanarEnvConfig(
            active_scenario_keys=("s1",),
            max_episode_steps=20,
        )
    )
    payload = {
        "training": {
            "seed": 0,
            "batch_size": 4,
            "buffer_size": 64,
            "learning_starts": 4,
            "history_length": 2,
            "net_arch": [8, 8],
            "gru_hidden_size": 4,
            "use_noisy": True,
            "use_gru": True,
            "use_pper": True,
            "eval_seeds": [0],
        },
        "output": str(tmp_path),
    }
    result = train_ddpg(
        env,
        payload,
        output=tmp_path,
        seed=0,
        total_timesteps=12,
        device="cpu",
    )
    assert result == 0
    for name in (
        "config.json",
        "manifest.json",
        "train_episodes.csv",
        "eval_episodes.csv",
        "model.pt",
    ):
        assert (tmp_path / name).is_file()
    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observation_source"] == "oracle"
    assert manifest["perception_mode"] == "oracle"
    assert manifest["observation_spec"]["shape"] == [17]
    assert manifest["action_spec"]["low"] == pytest.approx([-0.5, -math.radians(15.0)])
