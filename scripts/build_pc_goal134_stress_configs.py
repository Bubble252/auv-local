#!/usr/bin/env python3
"""Build the frozen Goal134 stress configs from tracked Goal85 base configs.

The base files are the tracked Goal85 ``candidate`` configs.  Goal125's
``B15/default_B15`` artifacts add a small reward overlay; the overlay is applied
here explicitly so the nominal Goal134 configs reproduce those artifacts without
depending on an untracked artifact directory.

Only two physical stress dimensions are changed:

* current speed / speed fluctuation / spatial shear are scaled together;
* the five Goal85b plant-scale channels are randomized in ``[0.8, 1.2]``.

Sensor profile stays ``unbiased_noise_v1``.  No p_nav/eta bias or delay is added.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = {
    "strict": ROOT / "configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json",
    "default": ROOT / "configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json",
}
DEFAULT_OUTPUT_DIR = ROOT / "configs/lowlevel/goal134"

# Exact reward fields added by Goal125's B15/default_B15 artifacts.
REWARD_OVERLAY = {
    "approach_enter_m": 5.0,
    "approach_full_m": 2.0,
    "terminal_progress_weight": 8.0,
    "w_approach_pitch": 0.5,
    "w_approach_vertical": 1.0,
}

PLANT_SEED = 13420
PLANT_RANGE = (0.8, 1.2)
STRESS_LEVELS = {
    "nominal": (1.0, None),
    "current15": (1.5, None),
    "current20": (2.0, None),
    "plant20": (1.0, PLANT_RANGE),
    "combined20": (2.0, PLANT_RANGE),
}
SCENARIOS = ("strict", "default")


def _scale_current(current: dict[str, Any], factor: float) -> dict[str, Any]:
    if factor <= 0.0:
        raise ValueError("current scale factor must be positive")
    scaled = copy.deepcopy(current)
    low, high = scaled["speed_mps"]
    scaled["speed_mps"] = [float(low) * factor, float(high) * factor]
    variation = scaled.get("time_variation")
    if isinstance(variation, dict):
        variation["speed_amplitude_mps"] = (
            float(variation["speed_amplitude_mps"]) * factor
        )
    shear = scaled.get("shear")
    if isinstance(shear, dict):
        shear["velocity_gradient_per_s"] = [
            [float(value) * factor for value in row]
            for row in shear["velocity_gradient_per_s"]
        ]
    return scaled


def _plant_randomization(scale_range: tuple[float, float] | None) -> dict[str, Any]:
    if scale_range is None:
        lower, upper = 1.0, 1.0
        enabled = False
    else:
        lower, upper = scale_range
        enabled = True
    return {
        "enabled": enabled,
        "seed": PLANT_SEED,
        "drag_scale": [lower, upper],
        "thrust_scale": [lower, upper],
        "fin_scale": [lower, upper],
        "added_mass_scale": [lower, upper],
        "actuator_time_scale": [lower, upper],
    }


def build_config(scenario: str, level: str) -> dict[str, Any]:
    if scenario not in BASE_CONFIG:
        raise ValueError(f"unknown scenario: {scenario}")
    if level not in STRESS_LEVELS:
        raise ValueError(f"unknown stress level: {level}")
    data = json.loads(BASE_CONFIG[scenario].read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    reward = data.setdefault("reward", {})
    reward.update(REWARD_OVERLAY)
    factor, plant_range = STRESS_LEVELS[level]
    data["current"] = _scale_current(data["current"], factor)
    if plant_range is None:
        data.pop("plant_randomization", None)
    else:
        data["plant_randomization"] = _plant_randomization(plant_range)
    sensors = data.get("sensors", {})
    if sensors.get("profile") != "unbiased_noise_v1":
        raise ValueError("Goal134 requires sensors.profile=unbiased_noise_v1")
    name = f"pc_goal134_{level}_{scenario}_20261111"
    data.setdefault("experiment", {})["name"] = name
    data["experiment"]["output"] = f"artifacts/lowlevel/{name}"
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in STRESS_LEVELS:
        for scenario in SCENARIOS:
            data = build_config(scenario, level)
            target = output_dir / f"pc_goal134_{level}_{scenario}_20261111.json"
            target.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
