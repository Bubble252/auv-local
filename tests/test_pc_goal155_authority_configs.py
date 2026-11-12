from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts/build_goal155_authority_training_configs.py"
SPEC = importlib.util.spec_from_file_location("goal155_builder", BUILDER)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _base() -> dict:
    return json.loads(builder.BASE_CONFIG.read_text(encoding="utf-8"))


def test_goal155_has_three_arms_and_three_frozen_seeds() -> None:
    assert tuple(builder.ARMS) == ("near10_2", "mid_only", "far_off_only")
    assert builder.SEEDS == (86201, 86202, 86203)
    assert builder.SCREEN_TRANSITIONS == 1_000_000


def test_goal155_gate_semantics_are_distinct() -> None:
    base = _base()
    near = builder._arm_config(base, "near10_2", 86201)
    mid = builder._arm_config(base, "mid_only", 86201)
    far = builder._arm_config(base, "far_off_only", 86201)

    assert near["residual_terminal_gate_enabled"] is True
    assert near["learner_terminal_gate_mode"] == "none"
    assert (near["residual_terminal_gate_enter_m"], near["residual_terminal_gate_full_m"]) == (
        10.0,
        2.0,
    )

    assert mid["residual_terminal_gate_enabled"] is True
    assert mid["learner_terminal_gate_mode"] == "shell"
    assert (mid["residual_terminal_gate_enter_m"], mid["residual_terminal_gate_full_m"]) == (
        10.0,
        2.0,
    )

    assert far["residual_terminal_gate_enabled"] is True
    assert far["learner_terminal_gate_mode"] == "none"
    assert (far["residual_terminal_gate_enter_m"], far["residual_terminal_gate_full_m"]) == (
        10.0001,
        10.0,
    )


def test_goal155_configs_preserve_budget_and_scratch_contract() -> None:
    base = _base()
    for arm in builder.ARMS:
        for seed in builder.SEEDS:
            payload = builder._arm_config(base, arm, seed)
            assert payload["action_interface"] == "actuator"
            assert payload["policy"]["architecture"] == "pc_z_t"
            assert payload["policy"]["bc_checkpoint"] is None
            assert payload["training"]["seed"] == seed
            assert payload["training"]["num_envs"] == 256
            assert payload["training"]["batch_size"] == 256
            assert payload["training"]["total_env_transitions"] == 1_000_000
            assert payload["evaluation"]["residual_scale"] == 1.0
