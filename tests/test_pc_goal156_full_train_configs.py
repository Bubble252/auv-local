from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts/build_goal156_full_train_configs.py"
SPEC = importlib.util.spec_from_file_location("goal156_builder", BUILDER)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _base() -> dict:
    return json.loads(builder.BASE_CONFIG.read_text(encoding="utf-8"))


def test_goal156_preserves_full_residual_and_5m_budget() -> None:
    base = _base()
    for seed in builder.SEEDS:
        payload = builder._config(base, seed)
        assert payload["action_interface"] == "actuator"
        assert payload["policy"]["architecture"] == "pc_z_t"
        assert payload["policy"]["bc_checkpoint"] is None
        assert payload["training"]["seed"] == seed
        assert payload["training"]["num_envs"] == 256
        assert payload["training"]["batch_size"] == 256
        assert payload["training"]["total_env_transitions"] == 5_000_000
        assert payload["training"]["screen_env_transitions"] == 5_000_000
        assert payload["residual_terminal_gate_enabled"] is False
        assert "learner_terminal_gate_mode" not in payload
        assert payload["evaluation"]["residual_scale"] == 1.0


def test_goal156_config_paths_are_formal_and_seed_specific() -> None:
    base = _base()
    for seed in builder.SEEDS:
        payload = builder._config(base, seed)
        assert f"full_seed{seed}" in payload["experiment"]["output"]
        assert "goal156_full_train_eval_gate" in payload["experiment"]["output"]
