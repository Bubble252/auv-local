from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs/lowlevel/goal153/goal153_mid_manifest.json"


def test_goal153_mid_configs_enable_train_time_shell() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["variant"] == "mid_only"
    assert set(manifest["configs"]) == {"R0", "R1", "R2"}
    for entry in manifest["configs"].values():
        payload = json.loads((REPO_ROOT / entry["path"]).read_text(encoding="utf-8"))
        assert payload["residual_terminal_gate_enabled"] is True
        assert payload["residual_terminal_gate_enter_m"] == 10.0
        assert payload["residual_terminal_gate_full_m"] == 2.0
        assert payload["learner_terminal_gate_mode"] == "shell"
        assert payload["policy"]["bc_checkpoint"] == "checkpoints/goal26/d3/pc_bc_actor.pt"
        assert payload["training"]["total_env_transitions"] == 1_000_000
