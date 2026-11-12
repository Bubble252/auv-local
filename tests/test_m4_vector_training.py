from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from auv_rl.ddpg.agent import DDPGAgent, DDPGConfig  # noqa: E402
from auv_rl.ddpg.replay import PrioritizedSequenceReplayBuffer, SumTree  # noqa: E402
from auv_rl.ddpg.vector_trainer import (  # noqa: E402
    evaluate_parallel_ddpg,
    load_vector_training_state,
    train_parallel_ddpg,
)
from auv_rl.envs.parallel_planar import P1TensorVectorEnv  # noqa: E402
from auv_rl.envs.planar_avoidance import PlanarEnvConfig  # noqa: E402


def _agent_config(**overrides: object) -> DDPGConfig:
    values: dict[str, object] = {
        "observation_dim": 5,
        "hidden_sizes": (8, 8),
        "gru_hidden_size": 4,
        "history_length": 3,
        "batch_size": 8,
        "buffer_size": 256,
        "learning_starts": 1,
        "use_noisy": True,
        "use_gru": True,
        "use_pper": False,
        "device": "cpu",
    }
    values.update(overrides)
    return DDPGConfig(**values)  # type: ignore[arg-type]


def _observations(seed: int, count: int, dimension: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, dimension)).astype(np.float32)


@pytest.mark.parametrize("use_gru", [True, False])
def test_act_batch_matches_legacy_act_for_n1(use_gru: bool) -> None:
    config = _agent_config(use_gru=use_gru)
    batch_agent = DDPGAgent(config, seed=3)
    legacy_agent = copy.deepcopy(batch_agent)
    batch_agent.begin_episode_for_env(0, 0)
    legacy_agent.begin_episode(0)

    observations = _observations(seed=7, count=12)
    for observation in observations:
        expected = legacy_agent.act(observation, evaluate=True)
        actual = batch_agent.act_batch([0], observation[None, :], evaluate=True)[0]
        assert np.allclose(actual, expected, atol=1e-7, rtol=0.0)

    assert len(batch_agent.env_history(0)) == config.history_length


def test_act_batch_isolates_history_and_reset_between_envs() -> None:
    agent = DDPGAgent(_agent_config(), seed=11)
    standalone_a = copy.deepcopy(agent)
    standalone_b = copy.deepcopy(agent)
    agent.begin_episode_for_env(0, 10)
    agent.begin_episode_for_env(1, 20)
    standalone_a.begin_episode(10)
    standalone_b.begin_episode(20)

    observations_a = _observations(seed=21, count=6)
    observations_b = _observations(seed=22, count=5)

    for step in range(4):
        actual = agent.act_batch([0], observations_a[step][None, :], evaluate=True)[0]
        expected = standalone_a.act(observations_a[step], evaluate=True)
        assert np.allclose(actual, expected, atol=1e-7, rtol=0.0)
    for step in range(2):
        actual = agent.act_batch([1], observations_b[step][None, :], evaluate=True)[0]
        expected = standalone_b.act(observations_b[step], evaluate=True)
        assert np.allclose(actual, expected, atol=1e-7, rtol=0.0)

    joint = agent.act_batch(
        [0, 1],
        np.stack((observations_a[4], observations_b[2]), axis=0),
        evaluate=True,
    )
    expected_a = standalone_a.act(observations_a[4], evaluate=True)
    expected_b = standalone_b.act(observations_b[2], evaluate=True)
    assert np.allclose(joint[0], expected_a, atol=1e-7, rtol=0.0)
    assert np.allclose(joint[1], expected_b, atol=1e-7, rtol=0.0)

    agent.begin_episode_for_env(0, 11)
    standalone_a.begin_episode(11)
    after_reset = agent.act_batch(
        [0, 1],
        np.stack((observations_a[5], observations_b[3]), axis=0),
        evaluate=True,
    )
    expected_a = standalone_a.act(observations_a[5], evaluate=True)
    expected_b = standalone_b.act(observations_b[3], evaluate=True)
    assert np.allclose(after_reset[0], expected_a, atol=1e-7, rtol=0.0)
    assert np.allclose(after_reset[1], expected_b, atol=1e-7, rtol=0.0)
    assert len(agent.env_history(0)) == 1
    assert len(agent.env_history(1)) == agent.config.history_length


def test_act_batch_noise_is_independent_per_env_and_eval_is_deterministic() -> None:
    config = _agent_config(use_gru=False, history_length=1)
    agent = DDPGAgent(config, seed=5)
    env_ids = tuple(range(16))
    for env_id in env_ids:
        agent.begin_episode_for_env(env_id, env_id)
    observation = np.linspace(-0.2, 0.2, config.observation_dim, dtype=np.float32)
    observations = np.tile(observation, (len(env_ids), 1))

    torch.manual_seed(123)
    noisy_actions = agent.act_batch(env_ids, observations, evaluate=False)
    assert noisy_actions.shape == (len(env_ids), config.action_dim)
    assert not np.allclose(noisy_actions, noisy_actions[0], atol=1e-9, rtol=0.0)

    deterministic_actions = agent.act_batch(env_ids, observations, evaluate=True)
    assert np.allclose(deterministic_actions, deterministic_actions[0], atol=1e-7, rtol=0.0)


def test_sumtree_batch_priority_update_matches_sequential_updates() -> None:
    batch_tree = SumTree(8)
    sequential_tree = SumTree(8)
    indices = np.array([1, 3, 5, 7], dtype=np.int64)
    priorities = np.array([0.2, 0.5, 1.3, 0.7], dtype=np.float64)

    batch_tree.update_many(indices, priorities)
    for index, priority in zip(indices.tolist(), priorities.tolist(), strict=True):
        sequential_tree.update(index, priority)

    assert np.allclose(batch_tree.tree, sequential_tree.tree, atol=1e-12, rtol=0.0)
    assert batch_tree.total == pytest.approx(sequential_tree.total)


def test_replay_sequence_batch_never_crosses_env_or_episode() -> None:
    replay = PrioritizedSequenceReplayBuffer(
        64,
        observation_dim=3,
        action_dim=2,
        history_length=3,
        use_pper=True,
        seed=2,
    )
    for env_id in (0, 1):
        for step in range(5):
            replay.add(
                np.full(3, env_id * 100 + step, dtype=np.float32),
                np.full(2, env_id, dtype=np.float32),
                float(step),
                np.full(3, env_id * 100 + step + 1, dtype=np.float32),
                step == 4,
                episode_id=env_id * 100,
                episode_step=step,
                env_id=env_id,
            )

    batch = replay.sample(64, beta=0.5)
    for row in range(batch.batch_size):
        valid = batch.valid_masks[row] > 0.5
        assert np.all(batch.episode_ids[row][valid] == batch.episode_ids[row][valid][0])
        assert np.all(batch.env_ids[row][valid] == batch.env_ids[row][valid][0])


def test_replay_sequence_uses_env_id_even_when_episode_ids_collide() -> None:
    replay = PrioritizedSequenceReplayBuffer(
        64,
        observation_dim=1,
        action_dim=1,
        history_length=4,
        use_pper=True,
        seed=4,
    )
    for env_id in (0, 1):
        for step in range(6):
            replay.add(
                np.array([env_id * 100 + step], dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                0.0,
                np.array([env_id * 100 + step + 1], dtype=np.float32),
                False,
                episode_id=0,
                episode_step=step,
                env_id=env_id,
            )

    batch = replay.sample(64, beta=0.5)
    for row in range(batch.batch_size):
        valid = batch.valid_masks[row] > 0.5
        env_id = int(batch.env_ids[row][valid][0])
        observations = batch.observations[row][valid, 0]
        assert np.all(np.floor(observations / 100.0) == env_id)


def test_observe_batch_keeps_truncated_and_terminated_done_semantics() -> None:
    agent = DDPGAgent(_agent_config(history_length=2), seed=17)
    observation = np.zeros(5, dtype=np.float32)
    next_observation = np.ones(5, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)

    agent.begin_episode_for_env(0, 0)
    agent.observe_batch(
        [0],
        observation[None, :],
        action[None, :],
        np.array([0.0], dtype=np.float32),
        next_observation[None, :],
        np.array([False], dtype=np.bool_),
    )
    agent.begin_episode_for_env(1, 1)
    agent.observe_batch(
        [1],
        observation[None, :],
        action[None, :],
        np.array([0.0], dtype=np.float32),
        next_observation[None, :],
        np.array([True], dtype=np.bool_),
    )

    batch = agent.replay.sample(16, beta=1.0)
    for row in range(batch.batch_size):
        valid = batch.valid_masks[row] > 0.5
        episode_id = int(batch.episode_ids[row][valid][0])
        expected_done = 1.0 if episode_id == 1 else 0.0
        assert np.all(batch.dones[row][valid] == expected_done)


def test_checkpoint_round_trip_preserves_batch_actions(tmp_path: Path) -> None:
    agent = DDPGAgent(_agent_config(), seed=23)
    for env_id in (0, 1):
        agent.begin_episode_for_env(env_id, env_id)
    observations_a = _observations(seed=31, count=4)
    observations_b = _observations(seed=32, count=4)
    expected_actions = []
    for step in range(4):
        expected_actions.append(
            agent.act_batch(
                [0, 1],
                np.stack((observations_a[step], observations_b[step]), axis=0),
                evaluate=True,
            )
        )

    checkpoint = tmp_path / "model.pt"
    agent.save(checkpoint)
    restored = DDPGAgent.load(checkpoint)
    for env_id in (0, 1):
        restored.begin_episode_for_env(env_id, env_id)
    for step in range(4):
        actual = restored.act_batch(
            [0, 1],
            np.stack((observations_a[step], observations_b[step]), axis=0),
            evaluate=True,
        )
        assert np.allclose(actual, expected_actions[step], atol=1e-7, rtol=0.0)


def test_replay_state_round_trip_preserves_sampling_stream() -> None:
    replay = PrioritizedSequenceReplayBuffer(
        32,
        5,
        2,
        history_length=3,
        use_pper=True,
        seed=17,
    )
    for episode_id in range(4):
        for step in range(7):
            replay.add(
                np.full(5, episode_id * 10 + step, dtype=np.float32),
                np.array([episode_id, step], dtype=np.float32),
                float(step),
                np.full(5, step + 0.5, dtype=np.float32),
                step == 6,
                episode_id=episode_id,
                episode_step=step,
                env_id=episode_id % 2,
            )
    first = replay.sample(8, beta=0.5)
    replay.update_priorities(first.indices, np.linspace(0.1, 1.0, 8).astype(np.float32))

    state = replay.state_dict()
    restored = PrioritizedSequenceReplayBuffer(32, 5, 2, history_length=3, use_pper=True, seed=99)
    restored.load_state_dict(state)

    np.testing.assert_array_equal(restored._tree.tree, replay._tree.tree)
    np.testing.assert_array_equal(restored._observations, replay._observations)
    np.testing.assert_array_equal(restored._valid, replay._valid)
    assert restored._episode_index == replay._episode_index
    expected = replay.sample(8, beta=0.75)
    actual = restored.sample(8, beta=0.75)
    for field in (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "dones",
        "valid_masks",
        "weights",
        "indices",
        "episode_ids",
        "env_ids",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_environment_training_state_round_trip_is_exact() -> None:
    config = PlanarEnvConfig(active_scenario_keys=("s1", "s2"), max_episode_steps=5)
    environment = P1TensorVectorEnv(2, config, seed=5)
    observations, _ = environment.reset(seed=5)
    action = np.array([[0.1, 0.02], [-0.05, -0.01]], dtype=np.float64)
    for _ in range(3):
        observations, _, _, _, _ = environment.step(action)
    state = environment.training_state_dict()

    restored = P1TensorVectorEnv(2, config, seed=99)
    restored.load_training_state_dict(state)
    expected_observations, expected_rewards, expected_terminated, expected_truncated, _ = (
        environment.step(action)
    )
    actual_observations, actual_rewards, actual_terminated, actual_truncated, _ = restored.step(
        action
    )
    np.testing.assert_array_equal(actual_observations, expected_observations)
    np.testing.assert_array_equal(actual_rewards, expected_rewards)
    np.testing.assert_array_equal(actual_terminated, expected_terminated)
    np.testing.assert_array_equal(actual_truncated, expected_truncated)
    np.testing.assert_array_equal(restored.episode_ids, environment.episode_ids)
    np.testing.assert_array_equal(restored.episode_steps, environment.episode_steps)


def test_train_parallel_ddpg_writes_outputs_and_exact_update_ratio(tmp_path: Path) -> None:
    num_envs = 16
    env = P1TensorVectorEnv(
        num_envs,
        PlanarEnvConfig(active_scenario_keys=("s1",), max_episode_steps=50),
        seed=0,
    )
    agent = DDPGAgent(
        DDPGConfig(
            observation_dim=17,
            hidden_sizes=(8, 8),
            gru_hidden_size=4,
            history_length=2,
            batch_size=8,
            buffer_size=256,
            learning_starts=4,
            use_noisy=True,
            use_gru=True,
            use_pper=True,
            per_beta_steps=10,
            device="cpu",
        ),
        seed=0,
    )

    metrics = train_parallel_ddpg(
        env,
        agent,
        total_env_transitions=num_envs,
        seed=0,
        update_to_data_ratio=1.0,
        output=tmp_path,
    )

    assert metrics.env_transitions == num_envs
    assert metrics.gradient_steps == 12
    assert metrics.actual_update_to_data_ratio == pytest.approx(1.0)
    assert metrics.total_update_to_data_ratio == pytest.approx(0.75)
    for name in (
        "manifest.json",
        "training_metrics.json",
        "gradient_updates.csv",
        "model.pt",
    ):
        assert (tmp_path / name).is_file()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_envs"] == num_envs
    assert manifest["observation_source"] == "analytic_13sector"
    assert manifest["evaluation_source"] == "analytic_13sector"
    assert manifest["periodic_evaluation_source"] == "analytic_13sector"
    assert manifest["final_evaluation_source"] == "oceansim_simplified_pcl"
    assert manifest["metrics"]["gradient_steps"] == 12
    gradient_rows = (tmp_path / "gradient_updates.csv").read_text(encoding="utf-8").splitlines()
    assert len(gradient_rows) == 13


def test_train_parallel_ddpg_does_not_overwrite_transition_with_reset_observation(
    tmp_path: Path,
) -> None:
    env = P1TensorVectorEnv(
        1,
        PlanarEnvConfig(active_scenario_keys=("s1",), max_episode_steps=1),
        seed=0,
    )
    agent = DDPGAgent(
        DDPGConfig(
            observation_dim=17,
            hidden_sizes=(8, 8),
            gru_hidden_size=4,
            history_length=2,
            batch_size=4,
            buffer_size=64,
            learning_starts=100,
            use_noisy=False,
            use_gru=True,
            use_pper=False,
            device="cpu",
        ),
        seed=0,
    )

    metrics = train_parallel_ddpg(
        env,
        agent,
        total_env_transitions=1,
        seed=0,
        update_to_data_ratio=0.0,
    )
    assert metrics.gradient_steps == 0
    final_observation = agent.replay._next_observations[0].copy()
    reset_observation = env._batch_observation()[0][0].copy()
    assert not np.allclose(final_observation, reset_observation, atol=1e-7, rtol=0.0)
    assert agent.replay._dones[0] == pytest.approx(0.0)
    assert env.episode_steps[0] == 0

    assert math.isfinite(metrics.control_steps_per_s)


def test_evaluate_parallel_ddpg_is_deterministic_and_isolated(tmp_path: Path) -> None:
    env = P1TensorVectorEnv(
        1,
        PlanarEnvConfig(active_scenario_keys=("s1",), max_episode_steps=4),
        seed=0,
    )
    agent = DDPGAgent(
        DDPGConfig(
            observation_dim=17,
            hidden_sizes=(8, 8),
            gru_hidden_size=4,
            history_length=2,
            batch_size=4,
            buffer_size=64,
            learning_starts=100,
            use_noisy=False,
            use_gru=True,
            use_pper=False,
            device="cpu",
        ),
        seed=0,
    )
    agent.begin_episode_for_env(0, 0)
    assert len(agent.env_history(0)) == 0

    rows, summary = evaluate_parallel_ddpg(
        agent,
        env,
        env_transitions=0,
        gradient_steps=0,
        scenario_keys=("s1",),
        seeds=(0,),
    )

    assert len(rows) == 1
    assert rows[0]["scenario_key"] == "s1"
    assert bool(rows[0]["timeout"]) is True
    assert summary.episodes == 1
    assert summary.scenario_episodes == {"s1": 1}
    assert summary.finite
    assert math.isfinite(summary.health["q_range_mean"])
    assert len(agent.env_history(0)) == 0


def test_train_parallel_ddpg_writes_periodic_evaluation_and_best_checkpoint(
    tmp_path: Path,
) -> None:
    env = P1TensorVectorEnv(
        1,
        PlanarEnvConfig(active_scenario_keys=("s1",), max_episode_steps=2),
        seed=0,
    )
    agent = DDPGAgent(
        DDPGConfig(
            observation_dim=17,
            hidden_sizes=(8, 8),
            gru_hidden_size=4,
            history_length=2,
            batch_size=4,
            buffer_size=64,
            learning_starts=1,
            use_noisy=False,
            use_gru=True,
            use_pper=False,
            device="cpu",
        ),
        seed=0,
    )

    metrics = train_parallel_ddpg(
        env,
        agent,
        total_env_transitions=3,
        seed=0,
        update_to_data_ratio=0.0,
        output=tmp_path,
        evaluation_every_env_transitions=2,
        evaluation_seeds=(0,),
        evaluation_scenario_keys=("s1",),
    )

    assert metrics.env_transitions == 3
    for name in (
        "evaluation_history.json",
        "best_evaluation.json",
        "checkpoint_best.pt",
        "checkpoint_latest.pt",
        "episodes.csv",
    ):
        assert (tmp_path / name).is_file()
    history = json.loads((tmp_path / "evaluation_history.json").read_text(encoding="utf-8"))
    assert [item["summary"]["env_transitions"] for item in history] == [0, 2, 3]


def test_vector_training_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    def make_run() -> tuple[P1TensorVectorEnv, DDPGAgent]:
        env = P1TensorVectorEnv(
            2,
            PlanarEnvConfig(active_scenario_keys=("s1",), max_episode_steps=3),
            seed=7,
        )
        agent = DDPGAgent(
            DDPGConfig(
                observation_dim=17,
                hidden_sizes=(8, 8),
                gru_hidden_size=4,
                history_length=2,
                batch_size=4,
                buffer_size=128,
                learning_starts=2,
                use_noisy=True,
                use_gru=True,
                use_pper=True,
                per_beta_steps=20,
                device="cpu",
            ),
            seed=7,
        )
        return env, agent

    def run(output: Path, total: int, resume: dict | None = None):
        env, agent = make_run()
        return train_parallel_ddpg(
            env,
            agent,
            total_env_transitions=total,
            seed=7,
            update_to_data_ratio=1.0,
            output=output,
            evaluation_every_env_transitions=2,
            evaluation_seeds=(0,),
            evaluation_scenario_keys=("s1",),
            resume_state=resume,
        )

    interrupted = tmp_path / "interrupted"
    partial_metrics = run(interrupted, 4)
    state_path = interrupted / "training_state.pt"
    assert state_path.is_file()
    resume_state = load_vector_training_state(state_path)
    assert resume_state["env_transitions"] == 4
    assert resume_state["agent"]["replay"]["size"] == 4

    resumed = tmp_path / "resumed"
    resumed_metrics = run(resumed, 6, resume_state)
    uninterrupted = tmp_path / "uninterrupted"
    expected_metrics = run(uninterrupted, 6)

    assert partial_metrics.env_transitions == 4
    assert resumed_metrics.env_transitions == expected_metrics.env_transitions == 6
    assert resumed_metrics.vector_control_steps == expected_metrics.vector_control_steps
    assert resumed_metrics.environment_substeps == expected_metrics.environment_substeps
    assert resumed_metrics.gradient_steps == expected_metrics.gradient_steps
    assert resumed_metrics.actual_update_to_data_ratio == pytest.approx(
        expected_metrics.actual_update_to_data_ratio
    )

    resumed_state = load_vector_training_state(resumed / "training_state.pt")
    expected_state = load_vector_training_state(uninterrupted / "training_state.pt")
    for name, expected in expected_state["agent"]["modules"].items():
        actual = resumed_state["agent"]["modules"][name]
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key].numpy(), value.numpy())
    for name in (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "dones",
        "env_ids",
        "episode_ids",
        "episode_steps",
        "valid",
        "tree",
    ):
        np.testing.assert_array_equal(
            resumed_state["agent"]["replay"][name],
            expected_state["agent"]["replay"][name],
        )
    assert resumed_state["gradient_steps"] == expected_state["gradient_steps"]
    assert resumed_state["environment_substeps"] == expected_state["environment_substeps"]
    assert resumed_state["best_score"] == expected_state["best_score"]
    assert resumed_state["episode_rows"] == expected_state["episode_rows"]
