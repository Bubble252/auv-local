from __future__ import annotations

import json
from pathlib import Path

from auv_lowlevel.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "lowlevel"


def _config(scenario: str) -> dict:
    path = CONFIG_DIR / (
        f"odarrl_control_pc_c3_random_realistic_v1_stress_teacher_fixed_"
        f"{scenario}_20261108.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_stress_configs_use_stress_profile_and_keep_teacher_controls() -> None:
    for scenario in ("default", "strict125"):
        data = _config(scenario)
        assert data["sensors"]["profile"] == "realistic_stress"
        assert data["sensors"]["seed"] == 82300
        assert data["guidance"]["current_crab_enabled"] is True
        assert data["guidance"]["robust_terminal_enabled"] is True
        assert data["guidance"]["route_speed_floor_enabled"] is True
        assert data["termination"]["pc_goal_plane_miss_termination_enabled"] is True


def test_stress_configs_are_loadable_with_expected_goal_radius() -> None:
    default = load_config(
        CONFIG_DIR
        / "odarrl_control_pc_c3_random_realistic_v1_stress_teacher_fixed_default_20261108.json"
    )
    strict = load_config(
        CONFIG_DIR
        / "odarrl_control_pc_c3_random_realistic_v1_stress_teacher_fixed_strict125_20261108.json"
    )
    assert default.sensors is not None and default.sensors.profile == "realistic_stress"
    assert strict.sensors is not None and strict.sensors.profile == "realistic_stress"
    assert default.plant.goal_radius_m == 2.0
    assert strict.plant.goal_radius_m == 1.25
