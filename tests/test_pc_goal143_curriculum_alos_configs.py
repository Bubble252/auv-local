from __future__ import annotations

import json
from pathlib import Path

from auv_lowlevel.ppo_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/lowlevel/goal143"


def test_goal143_alos_ppo_configs_keep_encoder_contract_and_frozen_budget() -> None:
    for arm in ("C-full-alos", "C-no-current-alos"):
        path = CONFIG_DIR / f"{arm}_ppo_pilot.json"
        config = load_training_config(path)
        assert config.pc is not None
        assert config.pc.actor_current_encoder_enabled is True
        assert config.pc.actor_current_encoder_alpha == 0.0
        assert config.iterations == 40
        assert config.environment_config.endswith("curriculum/C0.json")
        payload = json.loads(path.read_text())
        assert payload["experiment"]["name"].startswith("pc_goal143_")


def test_goal143_alos_sac_configs_keep_current_dim_and_frozen_budget() -> None:
    for arm in ("C-full-alos", "C-no-current-alos"):
        payload = json.loads((CONFIG_DIR / f"{arm}_sac_pilot.json").read_text())
        assert payload["observation"]["current_dim"] == 3
        assert payload["policy"]["current_encoder_alpha"] == 0.0
        assert payload["training"]["total_env_transitions"] == 1_000_192
        assert payload["training"]["num_envs"] == 256
        assert payload["evaluation"]["seed_start"] == 20
        assert payload["evaluation"]["seed_count"] == 40


def test_goal143_manifest_freezes_control_change_and_heldout() -> None:
    manifest = json.loads((CONFIG_DIR / "goal143_manifest.json").read_text())
    assert manifest["control_change"].startswith("C0/C1 use ALOS")
    assert manifest["environment_config"].endswith("curriculum/C0.json")
    assert len(manifest["pilots"]) == 2
    for row in manifest["pilots"]:
        assert row["heldout_closed"] is True
        assert row["dev_seed_count"] == 40
        assert row["curriculum_direct_commands_enabled"] is False
        assert row["force_current_encoder_alpha_zero"] is (
            row["arm"] == "C-no-current-alos"
        )

