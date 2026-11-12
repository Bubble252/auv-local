from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from auv_lowlevel.batch.env import BatchedLowLevelEnv, BatchedLowLevelStep
from auv_lowlevel.batch.observation import OBSERVATION_SCHEMA_V2
from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.batch.vec_env import BatchTorchVecEnv
from auv_lowlevel.config import load_config


def _short_config():
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    return replace(
        config,
        plant=replace(config.plant, max_episode_steps=2),
    )


def test_vec_env_contract_and_timeout_auto_reset() -> None:
    env = BatchTorchVecEnv(_short_config(), num_envs=3, validate=True)
    observations, extras = env.get_observations()
    assert observations.shape == (3, 27)
    assert observations.dtype == torch.float32
    assert extras == {"observations": {}}
    assert env.num_actions == 3
    assert env.num_obs == 27
    assert env.num_privileged_obs is None
    assert env.max_episode_length == 2
    assert env.step_dt == pytest.approx(0.02)
    assert env.unwrapped is env

    first = env.step(torch.zeros((3, 3), dtype=torch.float32))
    assert first[0].shape == (3, 27)
    assert first[1].shape == (3,)
    assert first[2].shape == (3,)
    assert first[3]["observations"] == {}
    assert not bool(first[2].any())
    assert torch.equal(env.episode_length_buf, torch.ones(3, dtype=torch.long))

    second = env.step(torch.zeros((3, 3), dtype=torch.float32))
    assert bool(second[2].all())
    assert bool(second[3]["time_outs"].all())
    assert torch.equal(env.episode_length_buf, torch.zeros(3, dtype=torch.long))
    assert second[3]["log"]["/env/reset_count"].item() == pytest.approx(3.0)
    assert torch.isfinite(second[0]).all()


def test_vec_env_supports_26d_schema_without_changing_backend() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    env = BatchTorchVecEnv(
        config,
        num_envs=2,
        schema=OBSERVATION_SCHEMA_V2,
    )
    observations, _ = env.get_observations()
    stepped, _, _, _ = env.step(torch.zeros((2, 3), dtype=torch.float32))
    assert env.num_obs == 26
    assert env.cfg["observation_schema"] == "v2"
    assert env.cfg["observation_dim"] == 26
    assert observations.shape == (2, 26)
    assert stepped.shape == (2, 26)
    assert observations.dtype == torch.float32


def test_vec_env_records_cpu_step_timing_window() -> None:
    env = BatchTorchVecEnv(
        _short_config(),
        num_envs=2,
        record_step_timings=True,
    )
    env.step(torch.zeros((2, 3), dtype=torch.float32))
    env.step(torch.zeros((2, 3), dtype=torch.float32))

    summary = env.drain_step_latency_stats()

    assert summary["mode"] == "cpu_wall_clock"
    assert summary["count"] == 2
    assert float(summary["p50_ms"]) > 0.0
    assert float(summary["p95_ms"]) >= float(summary["p50_ms"])
    assert env.drain_step_latency_stats()["count"] == 0


def test_vec_env_exposes_opt_in_compile_without_changing_default(monkeypatch) -> None:
    calls: list[tuple[object, str, bool, bool]] = []

    def fake_compile(function, *, mode, fullgraph, dynamic):
        calls.append((function, mode, fullgraph, dynamic))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    env = BatchTorchVecEnv(
        _short_config(),
        num_envs=2,
        compile_step=True,
        compile_mode="default",
    )
    assert calls and calls[0][1:] == ("default", False, False)
    assert env.cfg["compile_step"] is True
    assert env.cfg["compile_mode"] == "default"
    env.step(torch.zeros((2, 3), dtype=torch.float32))

    with pytest.raises(ValueError, match="compile_mode"):
        BatchTorchVecEnv(_short_config(), num_envs=1, compile_mode="reduce-overhead")


def test_auto_reset_only_replaces_done_rows() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    env = BatchedLowLevelEnv(config, num_envs=2)
    env.reset([41, 42])
    assert env.eta is not None
    env.eta[0, :3] = env.path.goal_ned_m
    before = env.observation.clone()

    result = env.step(torch.zeros((2, 3), dtype=torch.float64), auto_reset=True)

    assert bool(result.terminated[0])
    assert not bool(result.done[1])
    assert torch.equal(result.observation[1], result.info["final_observation"][1])
    assert not torch.equal(result.observation[0], result.info["final_observation"][0])
    assert not torch.equal(result.observation[0], before[0])


def test_auto_reset_step_does_not_explicitly_read_host_tensor_values(monkeypatch) -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    env = BatchedLowLevelEnv(config, num_envs=2, validate_finite=False, validate_shadow=False)
    env.reset([71, 72])
    assert env.eta is not None
    env.eta[0, :3] = env.path.goal_ned_m

    def forbidden_item(self: torch.Tensor) -> float:
        raise AssertionError("step path must not call Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    result = env.step(torch.zeros((2, 3), dtype=torch.float64), auto_reset=True)
    assert result.observation.shape == (2, 27)


def test_auto_reset_step_does_not_call_nonzero(monkeypatch) -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    env = BatchedLowLevelEnv(config, num_envs=2, validate_finite=False, validate_shadow=False)
    env.reset([81, 82])
    assert env.eta is not None
    env.eta[0, :3] = env.path.goal_ned_m

    def forbidden_nonzero(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("step path must not call Tensor.nonzero()")

    monkeypatch.setattr(torch.Tensor, "nonzero", forbidden_nonzero)
    result = env.step(torch.zeros((2, 3), dtype=torch.float64), auto_reset=True)
    assert result.observation.shape == (2, 27)


def test_auto_reset_step_does_not_read_host_path_length(monkeypatch) -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    env = BatchedLowLevelEnv(config, num_envs=2, validate_finite=False, validate_shadow=False)
    env.reset([91, 92])
    assert env.eta is not None
    env.eta[0, :3] = env.path.goal_ned_m

    def forbidden_length(self: BatchedReferencePath) -> float:
        raise AssertionError("step path must use the device-resident path length")

    monkeypatch.setattr(BatchedReferencePath, "total_length_m", property(forbidden_length))
    result = env.step(torch.zeros((2, 3), dtype=torch.float64), auto_reset=True)
    assert result.observation.shape == (2, 27)


def test_episode_metrics_divide_accumulated_values_by_steps_not_episodes() -> None:
    env = BatchTorchVecEnv(_short_config(), num_envs=2)

    def step_result() -> BatchedLowLevelStep:
        projection = SimpleNamespace(
            error_3d_m=torch.full((2,), 3.0),
            horizontal_error_m=torch.full((2,), 2.0),
            vertical_error_m=torch.full((2,), 1.0),
        )
        return cast(
            BatchedLowLevelStep,
            SimpleNamespace(
                info={
                    "projection_after": projection,
                    "residual_normalized": torch.ones((2, 3)),
                    "material_saturated": torch.tensor([False, True]),
                    "constraint_violation_count": torch.tensor([2.0, 0.0]),
                    "base_command": torch.zeros((2, 3)),
                    "terminal_kind": torch.zeros(2, dtype=torch.long),
                },
                observation=torch.zeros((2, 26)),
                truncated=torch.tensor([False, False]),
            ),
        )

    actions = torch.zeros((2, 3))
    rewards = torch.zeros(2)
    env._update_logs(
        step_result(), rewards, actions, torch.tensor([1, 1]), torch.tensor([False, False])
    )
    log = env._update_logs(
        step_result(), rewards, actions, torch.tensor([2, 2]), torch.tensor([True, True])
    )

    assert log["/episode/length_mean"].item() == pytest.approx(2.0)
    assert log["/episode/three_d_rmse_mean"].item() == pytest.approx(3.0)
    assert log["/episode/residual_abs_mean"].item() == pytest.approx(1.0)
    assert log["/episode/material_saturation_rate"].item() == pytest.approx(0.5)
    assert log["/episode/constraint_violations_mean"].item() == pytest.approx(1.0)


def test_vec_env_returns_ppo_scaled_reward_and_logs_raw_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = BatchTorchVecEnv(_short_config(), num_envs=2)
    projection = SimpleNamespace(
        error_3d_m=torch.zeros(2),
        horizontal_error_m=torch.zeros(2),
        vertical_error_m=torch.zeros(2),
    )
    result = SimpleNamespace(
        observation=torch.zeros((2, 27)),
        reward=SimpleNamespace(
            total=torch.tensor([100.0, -50.0]),
            ppo_scaled=torch.tensor([1.0, -0.5]),
        ),
        terminated=torch.tensor([True, True]),
        truncated=torch.tensor([False, False]),
        done=torch.tensor([True, True]),
        info={
            "projection_after": projection,
            "residual_normalized": torch.zeros((2, 3)),
            "material_saturated": torch.tensor([False, False]),
            "constraint_violation_count": torch.tensor([0.0, 0.0]),
            "base_command": torch.zeros((2, 3)),
            "terminal_kind": torch.ones(2, dtype=torch.long),
        },
    )
    monkeypatch.setattr(env.env, "step", lambda actions, auto_reset: result)

    _, rewards, _, extras = env.step(torch.zeros((2, 3)))

    assert torch.equal(rewards, torch.tensor([1.0, -0.5]))
    assert extras["log"]["/episode/return_mean"].item() == pytest.approx(25.0)
