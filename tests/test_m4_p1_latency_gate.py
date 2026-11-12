from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _runner_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_p1_oceansim_latency",
        Path("scripts/train_p1_oceansim.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latency_gate_requires_median_and_p95_under_threshold() -> None:
    module = _runner_module()

    passed = module._latency_gate({"median_ms": 19.0, "p95_ms": 20.0}, threshold_ms=20.0)
    failed = module._latency_gate({"median_ms": 19.9, "p95_ms": 20.1}, threshold_ms=20.0)

    assert passed["passed"] is True
    assert passed["median_pass"] is True
    assert passed["p95_pass"] is True
    assert failed["passed"] is False
    assert failed["median_pass"] is True
    assert failed["p95_pass"] is False


def test_latency_gate_rejects_invalid_threshold() -> None:
    module = _runner_module()

    with pytest.raises(ValueError, match="threshold"):
        module._latency_gate({"median_ms": 1.0, "p95_ms": 1.0}, threshold_ms=0.0)


def test_simulation_app_aa_setting_defaults_and_validates() -> None:
    module = _runner_module()

    assert module._simulation_app_config({})["anti_aliasing"] == 3
    assert module._simulation_app_config({"oceansim": {"anti_aliasing": 0}})["anti_aliasing"] == 0
    for value in (True, -1, 5, 1.0, "0"):
        with pytest.raises(ValueError, match="anti_aliasing"):
            module._simulation_app_config({"oceansim": {"anti_aliasing": value}})


def test_simulation_app_performance_profile_is_explicit() -> None:
    module = _runner_module()

    config = module._simulation_app_config(
        {"oceansim": {"anti_aliasing": 0, "render_profile": "isaaclab_performance_v1"}}
    )

    assert config["anti_aliasing"] == 0
    assert config["multi_gpu"] is False
    assert "--/rtx/reflections/enabled=false" in config["extra_args"]
    assert "--/rtx-transient/dlssg/enabled=false" in config["extra_args"]
    with pytest.raises(ValueError, match="render_profile"):
        module._simulation_app_config({"oceansim": {"render_profile": "unknown"}})


def test_simulation_app_cpu_thread_limit_validates() -> None:
    module = _runner_module()

    assert module._simulation_app_config({})["limit_cpu_threads"] == 32
    assert (
        module._simulation_app_config({"oceansim": {"limit_cpu_threads": 16}})["limit_cpu_threads"]
        == 16
    )
    for value in (True, 0, 129, 1.5, "16"):
        with pytest.raises(ValueError, match="limit_cpu_threads"):
            module._simulation_app_config({"oceansim": {"limit_cpu_threads": value}})


def test_oceansim_aa_off_config_is_single_variable() -> None:
    base = json.loads(Path("configs/rl/p1_ppo_oceansim_120m50hz.json").read_text(encoding="utf-8"))
    candidate = json.loads(
        Path("configs/rl/p1_ppo_oceansim_120m50hz_aa_off_v4.json").read_text(encoding="utf-8")
    )

    assert candidate["oceansim"].pop("anti_aliasing") == 0
    assert candidate == base


def test_oceansim_performance_config_is_single_variable() -> None:
    aa_off = json.loads(
        Path("configs/rl/p1_ppo_oceansim_120m50hz_aa_off_v4.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        Path("configs/rl/p1_ppo_oceansim_120m50hz_aa_off_perf_v5.json").read_text(encoding="utf-8")
    )

    assert candidate["oceansim"].pop("render_profile") == "isaaclab_performance_v1"
    assert candidate == aa_off


def test_oceansim_16_thread_config_is_single_variable() -> None:
    base = json.loads(Path("configs/rl/p1_ppo_oceansim_120m50hz.json").read_text(encoding="utf-8"))
    candidate = json.loads(
        Path("configs/rl/p1_ppo_oceansim_120m50hz_16threads_v6.json").read_text(encoding="utf-8")
    )

    assert candidate["oceansim"].pop("limit_cpu_threads") == 16
    assert candidate == base


def test_measure_policy_latency_includes_policy_and_environment_steps() -> None:
    module = _runner_module()

    class FakeEnv:
        config = SimpleNamespace(active_scenario_keys=("s2",))

        def __init__(self) -> None:
            self.resets = 0
            self.steps = 0

        def reset(
            self, *, seed: int, options: dict[str, object]
        ) -> tuple[np.ndarray, dict[str, object]]:
            assert options["scenario_key"] == "s2"
            self.resets += 1
            return np.zeros(17, dtype=np.float32), {}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            assert action.shape == (2,)
            self.steps += 1
            return np.zeros(17, dtype=np.float32), 0.0, False, False, {}

    class FakePolicy:
        def __init__(self) -> None:
            self.episodes: list[int] = []
            self.actions = 0

        def begin_episode(self, episode_id: int) -> None:
            self.episodes.append(episode_id)

        def act(self, observation: np.ndarray, *, evaluate: bool = True) -> np.ndarray:
            assert evaluate is True
            assert observation.shape == (17,)
            self.actions += 1
            return np.zeros(2, dtype=np.float64)

    env = FakeEnv()
    policy = FakePolicy()
    result = module._measure_policy_latency(
        env,
        policy,
        scenario_key="s2",
        steps=4,
        warmup_steps=2,
    )

    assert env.resets == 1
    assert env.steps == 6
    assert policy.actions == 6
    assert policy.episodes == [0]
    assert result["steps"] == 4
    assert result["warmup_steps"] == 2
    assert result["scenario"] == "s2"
    assert result["median_ms"] >= 0.0
    assert result["p95_ms"] >= result["median_ms"]
