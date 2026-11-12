from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs/lowlevel/goal153/manifest.json"
RUNNER = REPO_ROOT / "scripts/run_goal153_reward_pilot.sh"


def test_goal153_reward_pilot_configs_are_frozen() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["frozen_before_measurement"] is True
    assert manifest["seed"] == 86201
    assert manifest["transitions"] == 1_000_000
    assert manifest["bc_initialized"] is True
    assert set(manifest["configs"]) == {"R0", "R1", "R2"}
    expected = {
        "R0": (0.0, 0.0),
        "R1": (5.0, 0.0),
        "R2": (5.0, 2.0),
    }
    for name, (vertical, heave) in expected.items():
        env = json.loads(
            (REPO_ROOT / manifest["configs"][name]["env"]).read_text(encoding="utf-8")
        )
        train = json.loads(
            (REPO_ROOT / manifest["configs"][name]["train"]).read_text(encoding="utf-8")
        )
        assert env["reward"]["w_approach_vertical"] == 1.0
        assert env["reward"]["w_vertical_potential"] == vertical
        assert env["reward"]["w_heave_potential"] == heave
        assert train["environment_config"] == manifest["configs"][name]["env"]
        assert train["policy"]["bc_checkpoint"] == "checkpoints/goal26/d3/pc_bc_actor.pt"
        assert train["residual_terminal_gate_enabled"] is False
        assert train["training"]["seed"] == 86201
        assert train["training"]["screen_env_transitions"] == 1_000_000
        assert train["training"]["total_env_transitions"] == 1_000_000
        assert train["action_interface"] == "actuator"


def test_goal153_runner_keeps_path_curriculum_semantics() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    train_command = script.split("scripts/train_pc_command_residual_sac.py", 1)[1]
    train_command = train_command.split("echo $? >", 1)[0]
    assert "--curriculum-disable-direct-command" in train_command
