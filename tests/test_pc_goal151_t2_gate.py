from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json"
MANIFEST = REPO_ROOT / "configs/lowlevel/goal151/profile_manifest.json"


def test_goal151_configs_only_change_gate_and_experiment() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    for name, full_m in (("near10_2", 2.0), ("shell10_3", 3.0)):
        payload = json.loads(
            (REPO_ROOT / f"configs/lowlevel/goal151/{name}.json").read_text(encoding="utf-8")
        )
        assert payload["residual_terminal_gate_enabled"] is True
        assert payload["residual_terminal_gate_enter_m"] == 10.0
        assert payload["residual_terminal_gate_full_m"] == full_m
        assert payload["action_interface"] == source["action_interface"]
        assert payload["actuator_residual_scale"] == source["actuator_residual_scale"]
        assert payload["policy"] == source["policy"]
        assert payload["training"] == source["training"]


def test_goal151_manifest_freezes_source_and_variants() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["frozen_before_measurement"] is True
    assert manifest["source_config"] == "configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json"
    assert set(manifest["variants"]) == {"near10_2", "shell10_3"}
    assert manifest["variants"]["near10_2"]["learner_terminal_gate_mode"] == "none"
    assert manifest["variants"]["shell10_3"]["learner_terminal_gate_mode"] == "shell"
