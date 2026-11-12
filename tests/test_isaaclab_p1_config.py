from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from auv_isaaclab.config import ConfigError, P1AvoidanceEnvCfg, load_config
from auv_isaaclab.tasks.p1_avoidance.agents.rsl_rl_ppo_cfg import build_rsl_rl_runner_cfg

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs" / "ppo" / "isaaclab"
BASE = CONFIG_DIR / "p1_ppo_base.json"
SMOKE = CONFIG_DIR / "p1_ppo_smoke.json"
SMOKE_MIXED = CONFIG_DIR / "p1_ppo_smoke_mixed.json"
MAIN = CONFIG_DIR / "p1_ppo_main.json"
SCHEMA = CONFIG_DIR / "schema" / "ppo_experiment.schema.json"


def _payload(path: Path = BASE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_base_config_resolves_contract() -> None:
    bundle = load_config(BASE)
    cfg = bundle.config

    assert cfg.env.num_envs == 4096
    assert cfg.raw_observation_dim == 17
    assert cfg.policy_input_dim == 272
    assert cfg.rollout_transitions == 131072
    assert cfg.minibatch_size == 16384
    assert cfg.config_sha256() == bundle.config_sha256
    assert len(bundle.config_sha256) == 64


def test_smoke_extends_base() -> None:
    bundle = load_config(SMOKE)

    assert [path.name for path in bundle.source_files] == [
        "p1_ppo_base.json",
        "p1_ppo_smoke.json",
    ]
    assert bundle.config.env.num_envs == 64
    assert bundle.config.env.max_episode_steps == 600
    assert bundle.config.curriculum.scenario_mix["s1"] == 1.0
    assert bundle.config.curriculum.scenario_mix["s2"] == 0.0
    assert bundle.config.ppo.num_mini_batches == 4


def test_smoke_mixed_extends_smoke() -> None:
    bundle = load_config(SMOKE_MIXED)

    assert [path.name for path in bundle.source_files] == [
        "p1_ppo_base.json",
        "p1_ppo_smoke.json",
        "p1_ppo_smoke_mixed.json",
    ]
    assert bundle.config.curriculum.scenario_mix["s1"] == 0.5
    assert bundle.config.curriculum.scenario_mix["s2"] == 0.5
    assert bundle.config.evaluation.scenario_keys == ("s1", "s2")


def test_main_config_keeps_full_scale() -> None:
    bundle = load_config(MAIN)

    assert bundle.config.env.num_envs == 4096
    assert bundle.config.ppo.max_iterations == 5000
    assert bundle.config.ppo.save_interval == 100


def test_runtime_overrides_are_applied_and_recorded() -> None:
    bundle = load_config(
        MAIN,
        {
            "num_envs": 128,
            "seed": 7,
            "output": "outputs/ppo_isaaclab/override",
            "max_iterations": 20,
            "device": "cuda:1",
            "resume": "outputs/ppo_isaaclab/checkpoint.pt",
        },
    )

    assert bundle.config.env.num_envs == 128
    assert bundle.config.experiment.seed == 7
    assert bundle.config.experiment.output == "outputs/ppo_isaaclab/override"
    assert bundle.config.ppo.max_iterations == 20
    assert bundle.cli_overrides["device"] == "cuda:1"
    assert bundle.cli_overrides["resume"].endswith("checkpoint.pt")


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unsupported overrides"):
        load_config(BASE, {"batch_size": 256})


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["unexpected"] = 1

    with pytest.raises(ConfigError, match="unknown keys: unexpected"):
        load_config(_write(tmp_path / "bad.json", payload))


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["env"]["typo"] = 1

    with pytest.raises(ConfigError, match="env: unknown keys: typo"):
        load_config(_write(tmp_path / "bad.json", payload))


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    del payload["ppo"]["gamma"]

    with pytest.raises(ConfigError, match="ppo: missing keys: gamma"):
        load_config(_write(tmp_path / "bad.json", payload))


def test_non_integer_integration_ratio_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["env"]["integration_step_s"] = 0.015

    with pytest.raises(ConfigError, match="integer multiple"):
        load_config(_write(tmp_path / "bad.json", payload))


def test_reward_range_must_not_exceed_sensor_range(tmp_path: Path) -> None:
    payload = _payload()
    payload["reward"]["reward_obstacle_range_m"] = 200.0

    with pytest.raises(ConfigError, match="reward_obstacle_range_m"):
        load_config(_write(tmp_path / "bad.json", payload))


def test_reward_step_scale_is_tied_to_dt(tmp_path: Path) -> None:
    payload = _payload()
    payload["reward"]["reward_step_scale"] = 0.2

    with pytest.raises(ConfigError, match="reward_step_scale"):
        load_config(_write(tmp_path / "bad.json", payload))


def test_scenario_mix_must_sum_to_one(tmp_path: Path) -> None:
    payload = _payload()
    payload["curriculum"]["scenario_mix"] = {
        "s1": 0.5,
        "s2": 0.4,
        "c1": 0.0,
        "c2": 0.0,
        "c3": 0.0,
    }

    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(_write(tmp_path / "bad.json", payload))


def test_rollout_must_divide_into_minibatches(tmp_path: Path) -> None:
    payload = _payload()
    payload["ppo"]["num_mini_batches"] = 7

    with pytest.raises(ConfigError, match="divisible"):
        load_config(_write(tmp_path / "bad.json", payload))


def test_extends_cycle_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", {"extends": "b.json"})
    _write(tmp_path / "b.json", {"extends": "a.json"})

    with pytest.raises(ConfigError, match="extends cycle"):
        load_config(tmp_path / "a.json")


def test_config_hash_changes_with_tunable_value(tmp_path: Path) -> None:
    payload = _payload()
    first = load_config(_write(tmp_path / "first.json", payload)).config_sha256
    payload["ppo"]["learning_rate"] = 0.0002
    second = load_config(_write(tmp_path / "second.json", payload)).config_sha256

    assert first != second


def test_runner_cfg_maps_ppo_fields() -> None:
    cfg = load_config(BASE).config
    runner = build_rsl_rl_runner_cfg(cfg.ppo)

    assert runner["num_steps_per_env"] == 32
    assert runner["policy"]["actor_hidden_dims"] == [256, 256]
    assert runner["policy"]["critic_hidden_dims"] == [256, 256]
    assert runner["algorithm"]["class_name"] == "PPO"
    assert runner["algorithm"]["gamma"] == pytest.approx(0.99798)
    assert runner["algorithm"]["num_mini_batches"] == 8


def test_from_json_returns_env_cfg() -> None:
    cfg = P1AvoidanceEnvCfg.from_json(BASE)

    assert isinstance(cfg, P1AvoidanceEnvCfg)
    assert cfg.policy_input_dim == 272


def test_schema_is_strict_everywhere() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version",
        "experiment",
        "env",
        "observation",
        "reward",
        "curriculum",
        "ppo",
        "evaluation",
    }
    for name, definition in schema["$defs"].items():
        assert definition["additionalProperties"] is False, name


def test_eval_configs_are_json_objects() -> None:
    for name in ("eval_analytic.json", "eval_oceansim.json"):
        payload = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert payload["schema_version"] == "auv_ppo_isaaclab.eval.v1"
        assert payload["seeds"] == [0, 1, 2]
