from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from auv_lowlevel.config import REMUS100_MAX_SPEED_MPS, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/lowlevel/goal160"
ARMS = (
    "A_actuator_near10_2",
    "B_command_near10_2",
    "C_command_terminal_train",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("arm", ARMS)
def test_goal160_env_contract(arm: str) -> None:
    path = CONFIG_ROOT / f"env_{arm.lower()}.json"
    raw = _load_json(path)
    assert raw["experiment"]["pc_stage"] == "C3"
    assert raw["experiment"]["pc_control_mode"] == "alos"
    env = load_config(path)
    assert env.reference.speed_mps == pytest.approx(1.2)
    assert env.reference.speed_mps <= REMUS100_MAX_SPEED_MPS
    if env.current.enabled:
        current_upper = env.current.max_speed_mps
        variation = (
            0.0
            if env.current.time_variation is None
            else env.current.time_variation.speed_amplitude_mps
        )
        assert env.reference.speed_mps + current_upper + variation <= (
            REMUS100_MAX_SPEED_MPS + 1e-12
        )
    if arm == "A_actuator_near10_2":
        assert env.command_residual.enabled is False
    else:
        assert env.command_residual.enabled is True
    if arm == "C_command_terminal_train":
        assert env.command_residual.terminal_gate_enabled is True
        assert env.command_residual.terminal_gate_enter_m == pytest.approx(10.0)
        assert env.command_residual.terminal_gate_full_m == pytest.approx(2.0)


@pytest.mark.parametrize("arm", ARMS)
def test_goal160_train_configs_are_one_million_and_match_interface(arm: str) -> None:
    expected_interface = "actuator" if arm.startswith("A_") else "command"
    for seed in (86201, 86202, 86203):
        path = CONFIG_ROOT / f"train_{arm.lower()}_seed{seed}.json"
        payload = _load_json(path)
        assert payload["action_interface"] == expected_interface
        assert payload["training"]["seed"] == seed
        assert payload["training"]["total_env_transitions"] == 1_000_000
        assert payload["training"]["screen_env_transitions"] == 1_000_000
        env_path = ROOT / payload["environment_config"]
        assert env_path.is_file()


def test_goal160_manifest_hashes_and_speed_contract() -> None:
    manifest = _load_json(CONFIG_ROOT / "goal160_manifest.json")
    assert manifest["train_speed_mps"] == pytest.approx(1.2)
    assert manifest["max_speed_mps"] == pytest.approx(2.5)
    assert set(manifest["arms"]) == set(ARMS)
    for arm, entry in manifest["arms"].items():
        env_path = ROOT / entry["environment_config"]
        assert _sha256(env_path) == entry["environment_sha256"]
        assert entry["train_speed_mps"] <= REMUS100_MAX_SPEED_MPS
    for entry in manifest["train_configs"]:
        path = ROOT / entry["path"]
        assert _sha256(path) == entry["sha256"]
    eval_env = ROOT / manifest["command_eval_environment"]["path"]
    assert _sha256(eval_env) == manifest["command_eval_environment"]["sha256"]
    actuator_eval_env = ROOT / manifest["actuator_eval_environment"]["path"]
    assert _sha256(actuator_eval_env) == manifest["actuator_eval_environment"]["sha256"]
    for path in (eval_env, actuator_eval_env):
        raw = _load_json(path)
        assert raw["plant"]["goal_radius_m"] == pytest.approx(2.0)
        assert raw["reference"]["speed_mps"] == pytest.approx(1.2)
    eval_policy = ROOT / manifest["command_eval_policy"]["path"]
    assert _sha256(eval_policy) == manifest["command_eval_policy"]["sha256"]
    eval_payload = _load_json(eval_policy)
    assert eval_payload["action_interface"] == "command"
    assert eval_payload["residual_terminal_gate_enabled"] is False
    assert eval_payload["environment_config"] == manifest["command_eval_environment"]["path"]
