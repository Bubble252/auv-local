from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = WORKSPACE_ROOT / "third_party" / "stable-baselines3"
if SB3_ROOT.is_dir() and str(SB3_ROOT) not in sys.path:
    sys.path.insert(0, str(SB3_ROOT))

pytest.importorskip("stable_baselines3")

from auv_rl.sac.pc_vec_env import PCHistoryVecEnv  # noqa: E402


class _FakeBatchEnv:
    num_envs = 2
    observation_dim = 3
    action_dim = 3
    render_mode = None

    def __init__(self) -> None:
        self.observations = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        self.actions: np.ndarray | None = None
        self._completed = [{"success": True, "env": 0}]

    def reset(self):
        self.observations = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        return self.observations.copy(), {"episode_id": np.array([0, 1])}

    def step(self, actions):
        self.actions = np.asarray(actions)
        previous = self.observations.copy()
        self.observations = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
        return (
            self.observations.copy(),
            np.array([1.0, -1.0], dtype=np.float32),
            np.array([True, False]),
            np.array([False, False]),
            {
                "final_observations": previous,
                "episode_id": np.array([2, 1]),
            },
        )

    def drain_completed_infos(self):
        rows = self._completed
        self._completed = []
        return rows

    def set_action_mode(self, mode: str) -> None:
        self.mode = mode

    def set_path_zero(self, enabled: bool) -> None:
        self.path_zero = enabled


def test_history_vec_env_stacks_and_resets_done_history() -> None:
    wrapper = PCHistoryVecEnv(_FakeBatchEnv(), history_length=3)  # type: ignore[arg-type]
    observations = wrapper.reset()
    assert observations.shape == (2, 9)
    assert np.allclose(observations[0].reshape(3, 3), np.array([[1, 2, 3]] * 3))
    assert np.allclose(observations[1].reshape(3, 3), np.array([[4, 5, 6]] * 3))

    actions = np.zeros((2, 3), dtype=np.float32)
    observations, rewards, dones, infos = wrapper.step(actions)
    assert np.array_equal(dones, np.array([True, False]))
    assert np.allclose(observations[0].reshape(3, 3), np.array([[7, 8, 9]] * 3))
    assert np.allclose(observations[1].reshape(3, 3), np.array([[4, 5, 6], [4, 5, 6], [10, 11, 12]]))
    assert infos[0]["TimeLimit.truncated"] is False
    assert infos[0]["terminal_observation"].shape == observations.shape[1:]
    assert np.allclose(
        infos[0]["terminal_observation"].reshape(3, 3),
        np.array([[1.0, 2.0, 3.0]] * 3),
    )
    assert wrapper.drain_completed_infos() == [{"success": True, "env": 0}]


class _TorchInfoBatchEnv(_FakeBatchEnv):
    def step(self, actions):
        observations, rewards, terminated, truncated, info = super().step(actions)
        info["episode_id"] = torch.as_tensor(info["episode_id"])
        info["final_observations"] = torch.as_tensor(info["final_observations"])
        return observations, rewards, terminated, truncated, info


def test_history_vec_env_splits_torch_info_per_row() -> None:
    wrapper = PCHistoryVecEnv(_TorchInfoBatchEnv(), history_length=2)  # type: ignore[arg-type]
    wrapper.reset()
    _, _, _, infos = wrapper.step(np.zeros((2, 3), dtype=np.float32))
    assert infos[0]["episode_id"] == 2
    assert infos[1]["episode_id"] == 1
    assert infos[0]["terminal_observation"].shape == wrapper.observation_space.shape


def test_history_vec_env_rejects_bad_action_shape() -> None:
    wrapper = PCHistoryVecEnv(_FakeBatchEnv(), history_length=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actions have the wrong shape"):
        wrapper.step_async(np.zeros((2, 2), dtype=np.float32))
