from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_pc_goal136_dline_ppo_configs import (
    ENV_CONFIGS,
    ITERATIONS,
    NUM_ENVS,
    NUM_STEPS_PER_ENV,
    SCREEN_SEEDS,
    TOTAL_ENV_TRANSITIONS,
    build_training_config,
)


def test_ppo_dline_budget_and_axes_are_frozen(tmp_path: Path) -> None:
    assert tuple(ENV_CONFIGS) == ("N0", "D1", "D2")
    assert SCREEN_SEEDS == (86201, 86202, 86203)
    assert ITERATIONS * NUM_ENVS * NUM_STEPS_PER_ENV == TOTAL_ENV_TRANSITIONS == 1_000_000
    assert ITERATIONS == 40
    assert NUM_STEPS_PER_ENV == 25


@pytest.mark.parametrize("arm", tuple(ENV_CONFIGS))
def test_ppo_dline_configs_keep_same_algorithm_contract(tmp_path: Path, arm: str) -> None:
    env_path = tmp_path / f"{arm}.json"
    env_path.write_text(json.dumps({"arm": arm}), encoding="utf-8")
    payload = build_training_config(
        arm,
        SCREEN_SEEDS[0],
        environment_config=env_path,
        artifact_root=tmp_path,
    )
    assert payload["schema_version"] == "auv_odarrl_control_ppo.v1"
    assert payload["environment"]["config"] == str(env_path)
    assert payload["environment"]["num_envs"] == NUM_ENVS
    assert payload["runner"]["num_steps_per_env"] == NUM_STEPS_PER_ENV
    assert payload["training"]["iterations"] == ITERATIONS
    assert payload["experiment"]["seed"] == SCREEN_SEEDS[0]
    assert payload["experiment"]["arm"] == "PC"
    assert arm in payload["experiment"]["name"]
    assert payload["pc"]["dade_enabled"] is False
    assert payload["pc"]["action_interface"] == "actuator"
    assert payload["pc"]["actuator_residual_scale"] == 1.5
    assert payload["pc"]["base_residual_bias"] == [0.0, -0.25, 0.0]
    assert payload["algorithm"]["num_learning_epochs"] == 4
    assert payload["algorithm"]["num_mini_batches"] == 32
    assert payload["base_policy"]["mode"] == "frozen_pc_bc_residual"
    assert "/screen/" in payload["experiment"]["output"]
