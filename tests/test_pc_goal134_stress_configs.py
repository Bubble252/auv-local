from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from auv_lowlevel.config import load_config
from scripts.build_pc_goal134_stress_configs import (
    PLANT_RANGE,
    PLANT_SEED,
    STRESS_LEVELS,
    build_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs/lowlevel/goal134"


@pytest.mark.parametrize("scenario", ("strict", "default"))
def test_nominal_config_matches_goal85_base(scenario: str) -> None:
    built = build_config(scenario, "nominal")
    base_path = (
        REPO_ROOT
        / f"configs/lowlevel/pc_goal85_unbiased_{scenario}125_candidate_20261110.json"
        if scenario == "strict"
        else REPO_ROOT / "configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    built.pop("experiment")
    base.pop("experiment")
    # The tracked base does not yet contain Goal125's reward overlay.
    for key in (
        "approach_enter_m",
        "approach_full_m",
        "terminal_progress_weight",
        "w_approach_pitch",
        "w_approach_vertical",
    ):
        base["reward"][key] = built["reward"][key]
    assert built == base


def test_current_scaling_preserves_direction_distribution() -> None:
    nominal = build_config("default", "nominal")
    current20 = build_config("default", "current20")
    assert current20["current"]["speed_mps"] == pytest.approx([0.5, 0.9])
    assert current20["current"]["time_variation"]["speed_amplitude_mps"] == pytest.approx(0.2)
    assert np.allclose(
        current20["current"]["shear"]["velocity_gradient_per_s"],
        [[0.008, 0.0, 0.0], [0.0, -0.006, 0.0], [0.0, 0.0, 0.0]],
    )
    assert current20["current"]["direction_rad"] == nominal["current"]["direction_rad"]
    assert (
        current20["current"]["time_variation"]["direction_amplitude_rad"]
        == nominal["current"]["time_variation"]["direction_amplitude_rad"]
    )


def test_plant20_uses_all_five_channels() -> None:
    config = build_config("strict", "plant20")
    randomization = config["plant_randomization"]
    assert randomization["enabled"] is True
    assert randomization["seed"] == PLANT_SEED
    for key in (
        "drag_scale",
        "thrust_scale",
        "fin_scale",
        "added_mass_scale",
        "actuator_time_scale",
    ):
        assert randomization[key] == list(PLANT_RANGE)


def test_combined20_is_current20_plus_plant20() -> None:
    current20 = build_config("default", "current20")
    plant20 = build_config("default", "plant20")
    combined = build_config("default", "combined20")
    assert combined["current"] == current20["current"]
    assert combined["plant_randomization"] == plant20["plant_randomization"]


def test_goal134_configs_load_and_keep_unbiased_sensors() -> None:
    for level in STRESS_LEVELS:
        for scenario in ("strict", "default"):
            path = CONFIG_DIR / f"pc_goal134_{level}_{scenario}_20261111.json"
            config = load_config(path)
            assert config.sensors.profile == "unbiased_noise_v1"
            expected_radius = 1.25 if scenario == "strict" else 2.0
            assert config.plant.goal_radius_m == pytest.approx(expected_radius)
            if level.startswith("current"):
                assert config.plant_randomization.enabled is False
            if level in {"plant20", "combined20"}:
                assert config.plant_randomization.enabled is True
                assert config.plant_randomization.drag_scale == PLANT_RANGE
