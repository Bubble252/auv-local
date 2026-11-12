from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from auv_lowlevel.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "lowlevel" / "goal162"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "name",
    ["env_h3_gain_near10_2_train.json", "env_h3_gain_near10_2_eval.json"],
)
def test_goal162_environment_contract(name: str) -> None:
    path = CONFIG_ROOT / name
    payload = _load_json(path)
    config = load_config(path)
    assert payload["base_controller"]["heading_gain"] == pytest.approx(3.0)
    assert payload["base_controller"]["pitch_gain"] == pytest.approx(4.0)
    assert payload["base_controller"]["heading_rate_gain"] == pytest.approx(0.6)
    assert payload["base_controller"]["pitch_rate_gain"] == pytest.approx(1.0)
    assert payload["command_residual"]["enabled"] is False
    gain = payload["controller_gain_residual"]
    assert gain == {
        "enabled": True,
        "heading_gain_delta": 1.0,
        "pitch_gain_delta": 1.5,
        "controller_state": "actor",
        "terminal_gate_enabled": True,
        "terminal_gate_enter_m": 10.0,
        "terminal_gate_full_m": 2.0,
    }
    assert config.controller_gain_residual.heading_gain_delta == pytest.approx(1.0)
    assert config.controller_gain_residual.pitch_gain_delta == pytest.approx(1.5)
    assert config.reference.speed_mps == pytest.approx(1.2)
    worst_case = (
        config.reference.speed_mps
        + config.current.max_speed_mps
        + (0.0 if config.current.time_variation is None else config.current.time_variation.speed_amplitude_mps)
    )
    assert worst_case <= 2.5 + 1e-12


@pytest.mark.parametrize("seed", [86201, 86202, 86203])
def test_goal162_training_configs_are_paired_and_one_million(seed: int) -> None:
    path = CONFIG_ROOT / f"train_g1_gain_near10_2_seed{seed}.json"
    payload = _load_json(path)
    assert payload["action_interface"] == "controller_gain"
    assert payload["environment_config"] == "configs/lowlevel/goal162/env_h3_gain_near10_2_train.json"
    assert payload["training"]["seed"] == seed
    assert payload["training"]["num_envs"] == 1024
    assert payload["training"]["batch_size"] == 512
    assert payload["training"]["update_to_data_ratio"] == pytest.approx(0.125)
    assert payload["training"]["total_env_transitions"] == 1_000_000
    assert payload["training"]["screen_env_transitions"] == 1_000_000
    assert payload["residual_terminal_gate_enabled"] is False


def test_goal162_manifest_hashes_and_speed_contract() -> None:
    manifest = _load_json(CONFIG_ROOT / "goal162_manifest.json")
    assert manifest["action_interface"] == "controller_gain"
    assert manifest["action_dim"] == 2
    assert manifest["train_seeds"] == [86201, 86202, 86203]
    assert manifest["compute"] == {
        "num_envs": 1024,
        "batch_size": 512,
        "update_to_data_ratio": 0.125,
        "total_env_transitions": 1_000_000,
    }
    assert manifest["speed_contract"] == {
        "reference_speed_mps": 1.2,
        "remus100_max_speed_mps": 2.5,
        "curriculum_v2_active": False,
    }
    for key, path_key in (
        ("train_env_sha256", "train_env"),
        ("eval_env_sha256", "eval_env"),
        ("eval_policy_sha256", "eval_policy"),
    ):
        assert manifest[key] == _sha256(ROOT / manifest[path_key])
