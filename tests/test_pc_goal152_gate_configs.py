from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "configs/lowlevel/goal151/near10_2.json"
OUT = REPO_ROOT / "configs/lowlevel/goal152"
SEEDS = (86201, 86202, 86203)


def test_goal152_configs_freeze_gate_and_protocol() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    for seed in SEEDS:
        payload = json.loads((OUT / f"near10_2_seed{seed}.json").read_text(encoding="utf-8"))
        assert payload["residual_terminal_gate_enabled"] is True
        assert payload["residual_terminal_gate_enter_m"] == 10.0
        assert payload["residual_terminal_gate_full_m"] == 2.0
        assert payload["training"]["seed"] == seed
        assert payload["training"]["screen_env_transitions"] == 1_000_000
        assert payload["training"]["total_env_transitions"] == 1_000_000
        assert payload["training"]["num_envs"] == 256
        assert payload["training"]["batch_size"] == 256
        assert payload["experiment"]["heldout_seed_start"] == 100
        assert payload["experiment"]["heldout_seed_count"] == 50
        assert payload["action_interface"] == base["action_interface"]
        assert payload["actuator_residual_scale"] == base["actuator_residual_scale"]
        assert payload["environment_config"] == base["environment_config"]
        assert payload["policy"] == base["policy"]


def test_goal152_manifest_freezes_three_seeds() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_before_measurement"] is True
    assert manifest["gate"]["learner_terminal_gate_mode"] == "none"
    assert set(manifest["configs"]) == {str(seed) for seed in SEEDS}
    for seed in SEEDS:
        path = REPO_ROOT / manifest["configs"][str(seed)]["path"]
        assert path.is_file()
