from __future__ import annotations

import pytest
import torch

from auv_lowlevel.batch.history import BatchTorchHistoryVecEnv, HistoryLayout


class _DummyEnv:
    def __init__(self, num_envs: int = 2, observation_dim: int = 3) -> None:
        self.num_envs = num_envs
        self.num_obs = observation_dim
        self.num_actions = 2
        self.device = torch.device("cpu")
        self.step_dt = 0.02
        self.max_episode_length = 10
        self.unwrapped = self
        self.cfg = {"backend": "dummy", "observation_dim": observation_dim}
        self.total_env_steps = 0
        self.env_step_rate_hz = 0.0
        self._step = 0
        self.current_ned_mps = torch.tensor(
            [[0.2, -0.1, 0.0], [-0.3, 0.05, 0.0]], dtype=torch.float32
        )[:num_envs]
        self.observations = self._observations()

    def _observations(self) -> torch.Tensor:
        base = torch.arange(self.num_envs * self.num_obs, dtype=torch.float32)
        return base.reshape(self.num_envs, self.num_obs) + float(self._step * 100)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self.observations, {"observations": {}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        self._step = 0
        self.observations = self._observations()
        return self.observations, {"observations": {}}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        del actions
        self._step += 1
        self.observations = self._observations()
        self.total_env_steps += self.num_envs
        rewards = torch.zeros(self.num_envs)
        dones = torch.tensor([0.0] * (self.num_envs - 1) + [1.0])
        return self.observations, rewards, dones, {"observations": {}}


def test_history_layout_round_trip() -> None:
    layout = HistoryLayout(observation_dim=4, short_window=2, long_window=3)
    current = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    short = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    long = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    flat = layout.flatten(current, short, long)
    restored = layout.split(flat)

    assert layout.policy_input_dim == 24
    assert flat.shape == (2, 24)
    assert torch.equal(restored[0], current)
    assert torch.equal(restored[1], short)
    assert torch.equal(restored[2], long)


def test_history_layout_rejects_bad_windows_and_shapes() -> None:
    with pytest.raises(ValueError, match="short_window"):
        HistoryLayout(observation_dim=4, short_window=5, long_window=3)
    with pytest.raises(ValueError, match="positive"):
        HistoryLayout(observation_dim=4, short_window=0, long_window=3)
    with pytest.raises(ValueError, match="strides"):
        HistoryLayout(observation_dim=4, short_window=2, long_window=3, long_stride=0)

    layout = HistoryLayout(observation_dim=4, short_window=2, long_window=3)
    with pytest.raises(ValueError, match="current"):
        layout.flatten(torch.zeros(1, 3), torch.zeros(1, 2, 4), torch.zeros(1, 3, 4))


def test_history_wrapper_shifts_and_resets_episode_buffers() -> None:
    inner = _DummyEnv()
    env = BatchTorchHistoryVecEnv(inner, short_window=2, long_window=3)

    initial, _ = env.get_observations()
    current, short, long = env.layout.split(initial)
    assert torch.equal(current, inner.observations)
    assert torch.all(short == inner.observations[:, None, :])
    assert torch.all(long == inner.observations[:, None, :])

    flat, _, dones, _ = env.step(torch.zeros(2, 2))
    current, short, long = env.layout.split(flat)
    assert dones.tolist() == [0.0, 1.0]
    assert torch.equal(current, inner.observations)

    # Env 0 shifts both windows; env 1 was done and is filled with its reset obs.
    assert torch.equal(short[0, -1], inner.observations[0])
    assert torch.equal(long[0, -1], inner.observations[0])
    assert torch.all(short[1] == inner.observations[1])
    assert torch.all(long[1] == inner.observations[1])

    reset, _ = env.reset()
    current, short, long = env.layout.split(reset)
    assert torch.equal(current, inner.observations)
    assert torch.all(short == inner.observations[:, None, :])
    assert torch.all(long == inner.observations[:, None, :])


def test_history_wrapper_exposes_rsl_rl_contract() -> None:
    env = BatchTorchHistoryVecEnv(_DummyEnv(), short_window=2, long_window=3)
    assert env.num_obs == 3 * (1 + 2 + 3)
    assert env.cfg["base_observation_dim"] == 3
    assert env.cfg["history_short_window"] == 2
    assert env.cfg["history_long_window"] == 3
    assert env.cfg["history_short_stride"] == 1
    assert env.cfg["history_long_stride"] == 1
    assert env.cfg["observation_dim"] == env.num_obs
    assert env.max_episode_length == 10


def test_history_wrapper_subsamples_both_windows_without_changing_width() -> None:
    inner = _DummyEnv(num_envs=2)
    env = BatchTorchHistoryVecEnv(
        inner,
        short_window=2,
        long_window=3,
        short_stride=2,
        long_stride=2,
    )
    assert env.num_obs == 3 * (1 + 2 + 3)
    for _ in range(4):
        flat, _, _, _ = env.step(torch.zeros(2, 2))

    current, short, long = env.layout.split(flat)
    assert torch.equal(current[0], inner.observations[0])
    assert short[0].tolist() == [[200.0, 201.0, 202.0], [400.0, 401.0, 402.0]]
    assert long[0].tolist() == [
        [0.0, 1.0, 2.0],
        [200.0, 201.0, 202.0],
        [400.0, 401.0, 402.0],
    ]
    assert env.cfg["history_short_stride"] == 2
    assert env.cfg["history_long_stride"] == 2


def test_history_wrapper_places_true_current_only_in_critic_extras() -> None:
    inner = _DummyEnv(num_envs=2)
    env = BatchTorchHistoryVecEnv(
        inner,
        short_window=2,
        long_window=3,
        privileged_current=True,
    )

    flat, extras = env.get_observations()
    current, _, _ = env.layout.split(flat)
    critic = extras["observations"]["critic"]
    assert env.num_privileged_obs == inner.num_obs + 3
    assert env.critic_observation_dim == inner.num_obs + 3
    assert flat.shape == (2, inner.num_obs * 6)
    assert torch.equal(current, inner.observations)
    assert torch.equal(critic[:, : inner.num_obs], inner.observations)
    assert torch.equal(critic[:, inner.num_obs :], inner.current_ned_mps)

    flat_before = flat.clone()
    inner.current_ned_mps = torch.tensor(
        [[-0.4, 0.3, 0.0], [0.1, 0.2, 0.0]], dtype=torch.float32
    )
    flat_after, extras_after = env.get_observations()
    assert torch.equal(flat_after, flat_before)
    assert torch.equal(
        extras_after["observations"]["critic"][:, inner.num_obs :],
        inner.current_ned_mps,
    )

    _, _, _, infos = env.step(torch.zeros(2, 2))
    assert "critic" in infos["observations"]
    assert infos["observations"]["critic"].shape == (2, inner.num_obs + 3)


def test_current_frame_dropout_hides_direct_and_history_copies() -> None:
    inner = _DummyEnv(num_envs=2)
    env = BatchTorchHistoryVecEnv(
        inner,
        short_window=2,
        long_window=3,
        current_frame_dropout_prob=1.0,
    )

    flat, _ = env.get_observations()
    current, short, long = env.layout.split(flat)
    assert torch.count_nonzero(current) == 0
    assert torch.count_nonzero(short) == 0
    assert torch.count_nonzero(long) == 0

    flat, _, _, _ = env.step(torch.zeros(2, 2))
    current, short, long = env.layout.split(flat)
    assert torch.count_nonzero(current[0]) == 0
    assert torch.count_nonzero(short[0, -1]) == 0
    assert torch.count_nonzero(long[0, -1]) == 0
    assert torch.count_nonzero(short[0, :-1]) > 0
    assert torch.count_nonzero(long[0, :-1]) > 0

    # Env 1 resets in this step, so its bootstrap window must hide all rows.
    assert torch.count_nonzero(current[1]) == 0
    assert torch.count_nonzero(short[1]) == 0
    assert torch.count_nonzero(long[1]) == 0


def test_current_frame_dropout_disabled_preserves_observation() -> None:
    inner = _DummyEnv(num_envs=2)
    plain = BatchTorchHistoryVecEnv(inner, short_window=2, long_window=3)
    disabled = BatchTorchHistoryVecEnv(
        inner,
        short_window=2,
        long_window=3,
        current_frame_dropout_prob=0.0,
    )
    plain_flat, _ = plain.get_observations()
    disabled_flat, _ = disabled.get_observations()
    assert torch.equal(plain_flat, disabled_flat)
