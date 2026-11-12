from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from auv_lowlevel.curriculum import LinearAlphaSchedule, linear_alpha_value
from auv_lowlevel.ppo_config import parse_training_config

ROOT = Path(__file__).resolve().parents[1]
PPO_CONFIG = ROOT / "configs" / "lowlevel" / "goal86" / "pc_goal86_t2_ppo_screen_20261110.json"
SAC_SCRIPT = ROOT / "scripts" / "train_pc_command_residual_sac.py"


def _load_sac_script():
    spec = importlib.util.spec_from_file_location("train_pc_command_residual_sac", SAC_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linear_alpha_schedule_endpoints_and_clamp() -> None:
    schedule = LinearAlphaSchedule(start=0.0, end=1.0, transitions=100)
    assert schedule.value_at(0) == pytest.approx(0.0)
    assert schedule.value_at(50) == pytest.approx(0.5)
    assert schedule.value_at(100) == pytest.approx(1.0)
    assert schedule.value_at(250) == pytest.approx(1.0)
    assert linear_alpha_value(25, start=1.0, end=0.0, transitions=100) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        schedule.value_at(-1)


def test_zero_transition_alpha_schedule_jumps_to_end() -> None:
    schedule = LinearAlphaSchedule(start=0.0, end=0.75, transitions=0)
    assert schedule.value_at(0) == pytest.approx(0.75)
    assert schedule.as_dict() == {"start": 0.0, "end": 0.75, "transitions": 0}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": -0.1, "end": 1.0, "transitions": 10},
        {"start": 0.0, "end": float("nan"), "transitions": 10},
        {"start": 0.0, "end": 1.0, "transitions": -1},
        {"start": 0.0, "end": 1.0, "transitions": True},
    ],
)
def test_linear_alpha_schedule_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        LinearAlphaSchedule(**kwargs)


def test_ppo_config_parses_actor_current_encoder_alpha_ramp() -> None:
    payload = json.loads(PPO_CONFIG.read_text(encoding="utf-8"))
    payload["pc"]["actor_current_encoder_enabled"] = True
    payload["pc"]["actor_current_encoder_alpha"] = 1.0
    payload["pc"]["actor_current_encoder_alpha_start"] = 0.0
    payload["pc"]["actor_current_encoder_alpha_end"] = 1.0
    payload["pc"]["actor_current_encoder_alpha_ramp_transitions"] = 100_000
    config = parse_training_config(payload)
    assert config.pc is not None
    assert config.pc.actor_current_encoder_alpha_schedule == (0.0, 1.0, 100_000)
    rsl_policy = config.rsl_train_cfg()["policy"]
    assert rsl_policy["actor_current_encoder_alpha"] == pytest.approx(0.0)
    assert "actor_current_encoder_alpha_start" not in rsl_policy
    assert "actor_current_encoder_alpha_end" not in rsl_policy
    assert "actor_current_encoder_alpha_ramp_transitions" not in rsl_policy


@pytest.mark.parametrize(
    "override",
    [
        {"actor_current_encoder_alpha_start": 0.0},
        {"actor_current_encoder_alpha_end": 1.0},
        {"actor_current_encoder_alpha_ramp_transitions": 100},
        {
            "actor_current_encoder_alpha_start": 0.0,
            "actor_current_encoder_alpha_end": 1.0,
            "actor_current_encoder_alpha_ramp_transitions": 100,
            "actor_current_encoder_enabled": False,
        },
    ],
)
def test_ppo_config_rejects_incomplete_or_disabled_alpha_ramp(override) -> None:
    payload = json.loads(PPO_CONFIG.read_text(encoding="utf-8"))
    payload["pc"].pop("actor_current_encoder_alpha_start", None)
    payload["pc"].pop("actor_current_encoder_alpha_end", None)
    payload["pc"].pop("actor_current_encoder_alpha_ramp_transitions", None)
    payload["pc"].update(override)
    with pytest.raises(ValueError):
        parse_training_config(payload)


def test_sac_alpha_schedule_helper_matches_training_contract() -> None:
    module = _load_sac_script()
    initial, schedule = module._current_encoder_alpha_schedule(
        {
            "current_encoder_alpha": 1.0,
            "current_encoder_alpha_start": 0.0,
            "current_encoder_alpha_end": 1.0,
            "current_encoder_alpha_ramp_transitions": 100_000,
        },
        current_dim=3,
    )
    assert initial == pytest.approx(0.0)
    assert schedule is not None
    assert schedule.value_at(50_000) == pytest.approx(0.5)
    assert schedule.value_at(100_000) == pytest.approx(1.0)

    legacy_initial, legacy_schedule = module._current_encoder_alpha_schedule(
        {"current_encoder_alpha": 0.25}, current_dim=3
    )
    assert legacy_initial == pytest.approx(0.25)
    assert legacy_schedule is None

    with pytest.raises(ValueError, match="current_dim=3"):
        module._current_encoder_alpha_schedule(
            {
                "current_encoder_alpha_start": 0.0,
                "current_encoder_alpha_end": 1.0,
                "current_encoder_alpha_ramp_transitions": 100,
            },
            current_dim=0,
        )
