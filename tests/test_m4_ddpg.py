from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from auv_rl.ddpg.agent import DDPGAgent, DDPGConfig  # noqa: E402
from auv_rl.ddpg.diagnostics import (  # noqa: E402
    action_grid,
    critic_health_passes,
    diagnose_critic,
    saturation_gate_passes,
)
from auv_rl.ddpg.networks import NoisyLinear, PolicyActor  # noqa: E402
from auv_rl.ddpg.replay import PrioritizedSequenceReplayBuffer, SumTree  # noqa: E402
from auv_rl.ddpg.trainer import _scenario_keys_for_timestep, train_ddpg  # noqa: E402
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


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "silu"])
def test_actor_and_critic_support_configurable_activations(activation: str) -> None:
    config = DDPGConfig(
        observation_dim=17,
        hidden_sizes=(16, 16),
        gru_hidden_size=4,
        history_length=3,
        use_noisy=False,
        use_gru=True,
        activation=activation,
        layer_norm=activation == "silu",
        device="cpu",
    )
    agent = DDPGAgent(config, seed=2)
    observations = torch.randn(2, 3, 17)
    actions = torch.zeros(2, 3, 2, requires_grad=True)
    q_values, _ = agent.critic(observations, actions, sample_noise=False)
    q_values.sum().backward()
    assert q_values.shape == (2, 3, 1)
    assert actions.grad is not None


def test_critic_health_diagnostics_and_action_grid() -> None:
    config = DDPGConfig(
        observation_dim=17,
        hidden_sizes=(16, 16),
        gru_hidden_size=4,
        history_length=3,
        use_noisy=False,
        use_gru=False,
        use_pper=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=4)
    rng = np.random.default_rng(4)
    observations = rng.uniform(0.0, 1.0, size=(12, 17)).astype(np.float32)
    health = diagnose_critic(agent, observations)
    assert action_grid(agent).shape == (9, 2)
    assert health["sample_count"] == 12
    assert 0.0 <= health["activation_active_fraction"] <= 1.0
    assert np.isfinite(health["q_range_mean"])
    assert np.isfinite(health["dq_da_mean"])
    assert isinstance(critic_health_passes(health), bool)
    assert isinstance(saturation_gate_passes(health), bool)
    assert saturation_gate_passes(health) is False


def test_new_checkpoint_fields_round_trip_and_legacy_config_defaults(tmp_path) -> None:
    config = DDPGConfig(
        observation_dim=3,
        hidden_sizes=(4, 4),
        use_noisy=False,
        use_gru=False,
        use_pper=False,
        activation="silu",
        layer_norm=True,
        actor_output_noise=False,
        actor_output_weight_init_bound=0.003,
        policy_delay=2,
        twin_critics=True,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=8)
    checkpoint = tmp_path / "twin.pt"
    agent.save(checkpoint)
    restored = DDPGAgent.load(checkpoint)
    assert restored.config.activation == "silu"
    assert restored.config.layer_norm is True
    assert restored.config.actor_output_noise is False
    assert restored.config.actor_output_weight_init_bound == pytest.approx(0.003)
    assert restored.critic2 is not None

    legacy_agent = DDPGAgent(
        DDPGConfig(
            observation_dim=3,
            hidden_sizes=(4, 4),
            use_noisy=False,
            use_gru=False,
            use_pper=False,
            device="cpu",
        ),
        seed=8,
    )
    legacy_source = tmp_path / "legacy_source.pt"
    legacy_agent.save(legacy_source)
    legacy_payload = torch.load(legacy_source, map_location="cpu", weights_only=False)
    legacy_config = dict(legacy_payload["config"])
    for key in (
        "activation",
        "actor_activation",
        "critic_activation",
        "layer_norm",
        "actor_output_noise",
        "actor_output_sigma_init",
        "actor_noise_sigma_trainable",
        "actor_output_weight_init_bound",
        "policy_delay",
        "twin_critics",
        "target_policy_noise_sigma",
        "target_policy_noise_clip",
        "saturation_penalty",
        "saturation_threshold_fraction",
    ):
        legacy_config.pop(key, None)
    legacy_payload["config"] = legacy_config
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy_payload, legacy_path)
    loaded = DDPGAgent.load(legacy_path)
    assert loaded.config.activation == "relu"
    assert loaded.config.actor_output_noise is True
    assert loaded.config.policy_delay == 1


def test_frozen_actor_noise_flag_round_trips(tmp_path) -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        use_noisy=True,
        use_gru=False,
        use_pper=False,
        actor_hidden_sigma_init=0.01,
        actor_noise_sigma_trainable=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=19)
    checkpoint = tmp_path / "frozen_noise.pt"
    agent.save(checkpoint)

    restored = DDPGAgent.load(checkpoint)
    assert restored.config.actor_noise_sigma_trainable is False
    actor_noisy = [module for module in restored.actor.modules() if isinstance(module, NoisyLinear)]
    assert actor_noisy
    assert all(not module.weight_rho.requires_grad for module in actor_noisy)


def test_actor_output_layer_can_disable_noise_and_use_small_initialization() -> None:
    config = DDPGConfig(
        observation_dim=17,
        hidden_sizes=(16, 16),
        use_noisy=True,
        use_gru=False,
        use_pper=False,
        actor_output_noise=False,
        actor_output_weight_init_bound=0.003,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=9)
    noisy_layers = [module for module in agent.actor.modules() if isinstance(module, NoisyLinear)]
    assert noisy_layers[-1].use_noisy is False
    assert all(layer.use_noisy for layer in noisy_layers[:-1])
    assert float(noisy_layers[-1].weight_mu.abs().max()) <= 0.003

    observations = torch.randn(2, 1, 17)
    raw, _ = agent.actor.forward_raw(observations, sample_noise=True)
    raw.sum().backward()
    assert torch.isfinite(raw).all()


def test_twin_critic_update_backpropagates_each_critic_once() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        history_length=3,
        use_noisy=False,
        use_gru=False,
        use_pper=False,
        twin_critics=True,
        batch_size=2,
        learning_starts=2,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=11)
    agent.begin_episode(0)
    for step in range(6):
        observation = np.full(4, step / 10.0, dtype=np.float32)
        next_observation = np.full(4, (step + 1) / 10.0, dtype=np.float32)
        agent.observe(
            observation,
            np.zeros(2, dtype=np.float64),
            0.1,
            next_observation,
            step == 5,
        )
    metrics = agent.update(0)
    assert metrics is not None
    assert np.isfinite(metrics.critic_loss)


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


def test_sumtree_batch_get_and_update_match_scalar_operations() -> None:
    tree = SumTree(8)
    tree.update_many(
        np.asarray([0, 3, 5], dtype=np.int64),
        np.asarray([1.0, 2.0, 4.0], dtype=np.float64),
    )
    assert tree.total == pytest.approx(7.0)
    prefixes = np.asarray([0.0, 0.5, 1.5, 3.5, 6.5], dtype=np.float64)
    indices, priorities = tree.get_many(prefixes)
    scalar = [tree.get(prefix) for prefix in prefixes.tolist()]
    assert indices.tolist() == [index for index, _ in scalar]
    assert priorities.tolist() == pytest.approx([priority for _, priority in scalar])

    tree.update_many(
        np.asarray([0, 3], dtype=np.int64),
        np.asarray([3.0, 1.0], dtype=np.float64),
    )
    assert tree.total == pytest.approx(8.0)
    assert tree.get(4.5)[0] == 5


def test_vectorized_sequence_gather_matches_slot_by_slot_reference() -> None:
    replay = PrioritizedSequenceReplayBuffer(
        32,
        observation_dim=3,
        action_dim=2,
        history_length=4,
        use_pper=True,
        seed=7,
    )
    for env_id in (0, 1):
        for episode_id in range(3):
            for step in range(5):
                value = float(env_id * 100 + episode_id * 10 + step)
                replay.add(
                    np.full(3, value, dtype=np.float32),
                    np.full(2, value, dtype=np.float32),
                    value,
                    np.full(3, value + 0.5, dtype=np.float32),
                    step == 4,
                    episode_id=episode_id,
                    episode_step=step,
                    env_id=env_id,
                )

    batch = replay.sample(20, beta=0.5)
    for row, anchor in enumerate(batch.indices.tolist()):
        sequence = replay._sequence_indices(anchor)
        offset = replay.history_length - len(sequence)
        expected_observations = np.zeros((4, 3), dtype=np.float32)
        expected_actions = np.zeros((4, 2), dtype=np.float32)
        expected_rewards = np.zeros(4, dtype=np.float32)
        expected_next = np.zeros((4, 3), dtype=np.float32)
        expected_dones = np.ones(4, dtype=np.float32)
        expected_masks = np.zeros(4, dtype=np.float32)
        expected_episode_ids = np.full(4, -1, dtype=np.int64)
        expected_env_ids = np.full(4, -1, dtype=np.int64)
        for position, sequence_index in enumerate(sequence):
            slot = offset + position
            expected_observations[slot] = replay._observations[sequence_index]
            expected_actions[slot] = replay._actions[sequence_index]
            expected_rewards[slot] = replay._rewards[sequence_index]
            expected_next[slot] = replay._next_observations[sequence_index]
            expected_dones[slot] = replay._dones[sequence_index]
            expected_masks[slot] = 1.0
            expected_episode_ids[slot] = replay._episode_ids[sequence_index]
            expected_env_ids[slot] = replay._env_ids[sequence_index]

        np.testing.assert_array_equal(batch.observations[row], expected_observations)
        np.testing.assert_array_equal(batch.actions[row], expected_actions)
        np.testing.assert_array_equal(batch.rewards[row], expected_rewards)
        np.testing.assert_array_equal(batch.next_observations[row], expected_next)
        np.testing.assert_array_equal(batch.dones[row], expected_dones)
        np.testing.assert_array_equal(batch.valid_masks[row], expected_masks)
        np.testing.assert_array_equal(batch.episode_ids[row], expected_episode_ids)
        np.testing.assert_array_equal(batch.env_ids[row], expected_env_ids)


def test_actor_eval_is_deterministic_after_rng_is_consumed() -> None:
    config = DDPGConfig(
        observation_dim=5,
        hidden_sizes=(8, 8),
        gru_hidden_size=4,
        history_length=4,
        use_noisy=True,
        use_gru=True,
        use_pper=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=17)
    observation = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    agent.begin_episode(0)
    first = agent.act(observation, evaluate=True)
    torch.randn(4096)
    agent.begin_episode(0)
    second = agent.act(observation, evaluate=True)
    assert np.allclose(first, second, atol=1e-7, rtol=0.0)


def test_eval_forward_ignores_noisy_rho_and_training_uses_it() -> None:
    actor = PolicyActor(
        5,
        (-0.5, -0.2),
        (0.5, 0.2),
        hidden_sizes=(8, 8),
        use_gru=False,
        use_noisy=True,
    )
    inputs = torch.linspace(-1.0, 1.0, 5).reshape(1, 5)
    mean_action, _ = actor(inputs, sample_noise=False)
    with torch.no_grad():
        for module in actor.modules():
            if isinstance(module, NoisyLinear):
                module.weight_rho.fill_(50.0)
                module.bias_rho.fill_(50.0)
    mean_action_after, _ = actor(inputs, sample_noise=False)
    noisy_action, _ = actor(inputs, sample_noise=True)
    assert torch.allclose(mean_action, mean_action_after, atol=1e-7, rtol=0.0)
    assert not torch.allclose(mean_action, noisy_action, atol=1e-6, rtol=0.0)


def test_noisy_forward_samples_different_weights_each_call() -> None:
    actor = PolicyActor(
        4,
        (-0.5, -0.2),
        (0.5, 0.2),
        hidden_sizes=(16, 8),
        use_gru=False,
        use_noisy=True,
    )
    inputs = torch.linspace(-1.0, 1.0, 4).reshape(1, 4)
    first, _ = actor(inputs, sample_noise=True)
    second, _ = actor(inputs, sample_noise=True)
    assert not torch.allclose(first, second, atol=1e-8, rtol=0.0)


def test_checkpoint_reload_is_deterministic_after_rng_drift(tmp_path) -> None:
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
    agent = DDPGAgent(config, seed=23)
    observation = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    agent.begin_episode(0)
    torch.randn(1024)
    expected = agent.act(observation, evaluate=True)
    checkpoint = tmp_path / "model.pt"
    agent.save(checkpoint)
    restored = DDPGAgent.load(checkpoint)
    torch.randn(2048)
    restored.begin_episode(0)
    actual = restored.act(observation, evaluate=True)
    assert np.allclose(actual, expected, atol=1e-6, rtol=0.0)


def test_target_networks_ignore_noise_and_are_repeatable() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        gru_hidden_size=4,
        history_length=3,
        use_noisy=True,
        use_gru=True,
        use_pper=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=29)
    observations = torch.randn(2, 3, 4)
    with torch.no_grad():
        first_actions, _ = agent.target_actor(observations, sample_noise=False)
        torch.randn(512)
        second_actions, _ = agent.target_actor(observations, sample_noise=False)
        first_q, _ = agent.target_critic(observations, first_actions, sample_noise=False)
        torch.randn(512)
        second_q, _ = agent.target_critic(observations, second_actions, sample_noise=False)
    assert torch.allclose(first_actions, second_actions, atol=1e-7, rtol=0.0)
    assert torch.allclose(first_q, second_q, atol=1e-7, rtol=0.0)


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


def test_curriculum_switches_scenarios_at_frozen_step() -> None:
    training = {
        "curriculum": {
            "stage1_scenarios": ["s1"],
            "stage1_steps": 10000,
            "stage2_scenarios": ["s1", "s2"],
        }
    }
    assert _scenario_keys_for_timestep(training, timestep=9999, default_keys=("s1", "s2")) == (
        "s1",
    )
    assert _scenario_keys_for_timestep(training, timestep=10000, default_keys=("s1", "s2")) == (
        "s1",
        "s2",
    )


def test_privileged_features_reach_only_the_critic() -> None:
    config = DDPGConfig(
        observation_dim=4,
        privileged_dim=5,
        hidden_sizes=(8, 8),
        gru_hidden_size=4,
        history_length=2,
        batch_size=4,
        buffer_size=64,
        learning_starts=1,
        use_noisy=False,
        use_gru=False,
        use_pper=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=1)
    actor_first = agent.actor.body.head.layers[0]
    critic_first = agent.critic.body.head.layers[0]
    assert isinstance(actor_first, NoisyLinear)
    assert isinstance(critic_first, NoisyLinear)
    assert actor_first.in_features == 4
    assert critic_first.in_features == 4 + 5 + 2

    agent.begin_episode(0)
    rng = np.random.default_rng(0)
    for _ in range(8):
        agent.observe(
            rng.normal(size=4).astype(np.float32),
            np.zeros(2, dtype=np.float32),
            0.0,
            rng.normal(size=4).astype(np.float32),
            False,
            privileged=np.ones(5, dtype=np.float32),
            next_privileged=np.ones(5, dtype=np.float32),
        )
    metrics = agent.update(0)
    assert metrics is not None
    assert np.isfinite(metrics.critic_loss)


def test_bc_auxiliary_loss_decays_and_keeps_actor_dimension() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        gru_hidden_size=4,
        history_length=2,
        batch_size=4,
        buffer_size=64,
        learning_starts=1,
        use_noisy=False,
        use_gru=True,
        use_pper=False,
        bc_auxiliary_weight=0.1,
        bc_auxiliary_decay_updates=10,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=1)
    agent.configure_behavior_cloning(
        np.zeros((4, 2, 4), dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
    )
    agent.begin_episode(0)
    rng = np.random.default_rng(1)
    for _ in range(8):
        agent.observe(
            rng.normal(size=4).astype(np.float32),
            np.zeros(2, dtype=np.float32),
            0.0,
            rng.normal(size=4).astype(np.float32),
            False,
        )
    assert agent.bc_auxiliary_beta == pytest.approx(0.1)
    metrics = agent.update(0)
    assert metrics is not None
    assert 0.0 < agent.bc_auxiliary_beta < 0.1
    assert agent.actor.body.head.layers[0].in_features == 4


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


def test_actor_warmup_updates_freezes_actor_until_threshold() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        history_length=3,
        use_noisy=False,
        use_gru=False,
        use_pper=False,
        batch_size=2,
        learning_starts=2,
        actor_warmup_updates=2,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=13)
    agent.begin_episode(0)
    for step in range(6):
        observation = np.full(4, step / 10.0, dtype=np.float32)
        next_observation = np.full(4, (step + 1) / 10.0, dtype=np.float32)
        agent.observe(observation, np.zeros(2, dtype=np.float64), 0.1, next_observation, False)

    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in agent.critic.parameters()]

    assert agent.update(0) is not None
    assert agent.update(1) is not None
    for before, after in zip(actor_before, agent.actor.parameters(), strict=True):
        assert torch.equal(before, after.detach())
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(critic_before, agent.critic.parameters(), strict=True)
    )

    assert agent.update(2) is not None
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(actor_before, agent.actor.parameters(), strict=True)
    )


def test_actor_warmup_is_rejected_with_cuda_graph() -> None:
    with pytest.raises(ValueError, match="cuda_graph_update"):
        DDPGConfig(
            observation_dim=4,
            history_length=3,
            use_noisy=False,
            use_gru=False,
            actor_warmup_updates=10,
            cuda_graph_update=True,
            device="cpu",
        )


def test_actor_hidden_sigma_init_controls_noisy_hidden_layers() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        history_length=3,
        use_noisy=True,
        use_gru=False,
        actor_hidden_sigma_init=0.05,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=21)
    hidden = next(module for module in agent.actor.modules() if isinstance(module, NoisyLinear))
    sigma = torch.nn.functional.softplus(hidden.weight_rho).detach()
    assert float(sigma.max()) == pytest.approx(0.05, rel=1e-3)
    assert float(sigma.min()) == pytest.approx(0.05, rel=1e-3)


def test_actor_hidden_sigma_init_must_be_positive() -> None:
    with pytest.raises(ValueError, match="actor_hidden_sigma_init"):
        DDPGConfig(
            observation_dim=4,
            history_length=3,
            use_gru=False,
            actor_hidden_sigma_init=0.0,
            device="cpu",
        )


def test_actor_noise_sigma_can_be_frozen_while_means_train() -> None:
    config = DDPGConfig(
        observation_dim=4,
        hidden_sizes=(8, 8),
        history_length=3,
        use_noisy=True,
        use_gru=False,
        use_pper=False,
        batch_size=2,
        learning_starts=2,
        actor_hidden_sigma_init=0.01,
        actor_noise_sigma_trainable=False,
        device="cpu",
    )
    agent = DDPGAgent(config, seed=17)
    actor_noisy = [module for module in agent.actor.modules() if isinstance(module, NoisyLinear)]
    assert actor_noisy
    assert all(not module.weight_rho.requires_grad for module in actor_noisy)
    assert all(not module.bias_rho.requires_grad for module in actor_noisy)
    weight_rho_before = [module.weight_rho.detach().clone() for module in actor_noisy]
    bias_rho_before = [module.bias_rho.detach().clone() for module in actor_noisy]
    weight_before = [module.weight_mu.detach().clone() for module in actor_noisy]

    agent.begin_episode(0)
    for step in range(6):
        observation = np.full(4, step / 10.0, dtype=np.float32)
        next_observation = np.full(4, (step + 1) / 10.0, dtype=np.float32)
        agent.observe(observation, np.zeros(2, dtype=np.float64), 0.1, next_observation, False)
    for index in range(4):
        assert agent.update(index) is not None

    for weight_before_rho, bias_before_rho, module in zip(
        weight_rho_before, bias_rho_before, actor_noisy, strict=True
    ):
        assert torch.equal(weight_before_rho, module.weight_rho.detach())
        assert torch.equal(bias_before_rho, module.bias_rho.detach())
    assert any(
        not torch.equal(before, module.weight_mu.detach())
        for before, module in zip(weight_before, actor_noisy, strict=True)
    )


def test_actor_noise_sigma_trainable_must_be_bool() -> None:
    with pytest.raises(TypeError, match="actor_noise_sigma_trainable"):
        DDPGConfig(
            observation_dim=4,
            history_length=3,
            use_gru=False,
            actor_noise_sigma_trainable=0,  # type: ignore[arg-type]
            device="cpu",
        )
