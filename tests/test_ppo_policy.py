from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from auv_rl.ppo_policy import PPOActorPolicy


def _oceansim_runner_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_p1_oceansim_ppo_policy",
        Path("scripts/train_p1_oceansim.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(path: Path, *, frame_dim: int = 2, history_length: int = 2) -> Path:
    input_dim = frame_dim * history_length
    actor = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ELU(),
        nn.Linear(input_dim, 2),
    )
    with torch.no_grad():
        actor[0].weight.copy_(torch.eye(input_dim))
        actor[0].bias.zero_()
        actor[2].weight.zero_()
        actor[2].weight[0, -2:] = torch.tensor([1.0, 0.0])
        actor[2].weight[1, -2:] = torch.tensor([0.0, 1.0])
        actor[2].bias.zero_()
    state = {f"actor.{key}": value for key, value in actor.state_dict().items()}
    state.update(
        {
            "critic.0.weight": torch.zeros((1, input_dim)),
            "critic.0.bias": torch.zeros(1),
            "std": torch.ones(2),
        }
    )
    torch.save({"iter": 42, "model_state_dict": state}, path)
    return path


def test_ppo_policy_loads_rsl_rl_state_and_applies_history(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    policy = PPOActorPolicy.load(
        checkpoint,
        frame_dim=2,
        history_length=2,
        action_scale=(0.5, 0.2),
        device="cpu",
    )

    policy.begin_episode(0)
    first = policy.act(np.array([0.4, 0.6], dtype=np.float32))
    second = policy.act(np.array([0.2, 0.4], dtype=np.float32))
    assert policy.checkpoint_iteration == 42
    np.testing.assert_allclose(first, [0.2, 0.12], atol=1e-6)
    np.testing.assert_allclose(second, [0.1, 0.08], atol=1e-6)


def test_ppo_policy_rejects_wrong_frame_shape(tmp_path: Path) -> None:
    policy = PPOActorPolicy.load(
        _checkpoint(tmp_path / "checkpoint.pt"),
        frame_dim=2,
        history_length=2,
        action_scale=(0.5, 0.2),
    )

    with pytest.raises(ValueError, match="observation must have shape"):
        policy.act(np.zeros(3, dtype=np.float32))


def test_ppo_policy_rejects_nonfinite_observation(tmp_path: Path) -> None:
    policy = PPOActorPolicy.load(
        _checkpoint(tmp_path / "checkpoint.pt"),
        frame_dim=2,
        history_length=2,
        action_scale=(0.5, 0.2),
    )

    with pytest.raises(ValueError, match="finite"):
        policy.act(np.array([np.nan, 0.0], dtype=np.float32))


def test_ppo_policy_rejects_history_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input dimension"):
        PPOActorPolicy.load(
            _checkpoint(tmp_path / "checkpoint.pt"),
            frame_dim=2,
            history_length=3,
            action_scale=(0.5, 0.2),
        )


def test_oceansim_ppo_evaluation_consumes_single_frame_observations(tmp_path: Path) -> None:
    module = _oceansim_runner_module()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", frame_dim=17, history_length=2)

    class FakeEnv:
        builder = SimpleNamespace(spec=SimpleNamespace(shape=(17,)))
        limits = SimpleNamespace(max_accel_mps2=0.5, max_yaw_rate_rps=0.2)

        def __init__(self) -> None:
            self.resets = 0
            self.steps = 0
            self.episode_steps = 0
            self._state: Any = None

        def reset(
            self, *, seed: int, options: dict[str, object]
        ) -> tuple[np.ndarray, dict[str, object]]:
            assert options["jitter"] is False
            self.resets += 1
            self.episode_steps = 0
            self._state = SimpleNamespace(
                t_s=0.0,
                position_ned_m=np.zeros(3, dtype=np.float64),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
            return np.zeros(17, dtype=np.float32), {"sensor_points_total": 10}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            assert action.shape == (2,)
            assert np.all(np.abs(action) <= np.array([0.5, 0.2]))
            self.steps += 1
            self.episode_steps += 1
            assert self._state is not None
            self._state.t_s += 0.02
            self._state.position_ned_m[0] += 0.02
            return (
                np.zeros(17, dtype=np.float32),
                0.0,
                self.episode_steps >= 2,
                False,
                {"success": self.episode_steps >= 2, "scenario_key": "s1"},
            )

    env = FakeEnv()
    rows = module._evaluate_ppo_checkpoint(
        env,
        checkpoint=checkpoint,
        device="cpu",
        scenarios=("s1", "s2"),
        seeds=(0, 1),
        history_length=2,
    )

    assert len(rows) == 4
    assert env.resets == 4
    assert env.steps == 8
    assert [row["eval_seed"] for row in rows] == [0, 1, 0, 1]
    assert [row["scenario_key"] for row in rows] == ["s1", "s1", "s2", "s2"]
    assert all(row["reset_sensor_points_total"] == 10 for row in rows)
    assert all(len(row["trajectory"]) == 3 for row in rows)


def test_ppo_latency_gate_uses_history_adapter(tmp_path: Path) -> None:
    module = _oceansim_runner_module()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", frame_dim=17, history_length=2)

    class FakeEnv:
        builder = SimpleNamespace(spec=SimpleNamespace(shape=(17,), version="2.0-120m"))
        limits = SimpleNamespace(max_accel_mps2=0.5, max_yaw_rate_rps=0.2)
        config = SimpleNamespace(active_scenario_keys=("s2",), dt_s=0.02)
        perception = SimpleNamespace(observation_source="fake", perception_mode="fake")

        def reset(
            self, *, seed: int, options: dict[str, object]
        ) -> tuple[np.ndarray, dict[str, object]]:
            assert options == {"scenario_key": "s2", "jitter": False}
            return np.zeros(17, dtype=np.float32), {}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            assert action.shape == (2,)
            return np.zeros(17, dtype=np.float32), 0.0, False, False, {}

    result = module._run_latency_gate(
        FakeEnv(),
        checkpoint,
        policy_kind="ppo",
        output=tmp_path / "latency",
        device="cpu",
        threshold_ms=20.0,
        steps=3,
        warmup_steps=1,
        scenario_key="s2",
        history_length=2,
    )

    assert result["policy_kind"] == "ppo"
    assert result["step_latency"]["steps"] == 3
    assert (tmp_path / "latency" / "latency_gate.json").is_file()
