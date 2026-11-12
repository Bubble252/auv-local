#!/usr/bin/env python3
"""Build the frozen Goal85b B1 environment configs from Goal85 configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "lowlevel"

LEVELS = {
    "m0": None,
    "m05": (0.95, 1.05),
    "m10": (0.90, 1.10),
}
SCENARIOS = ("default", "strict125")
ROOTS = ("baseline", "candidate")


def _plant_randomization(level: str) -> dict[str, object]:
    scale = LEVELS[level]
    if scale is None:
        return {
            "enabled": False,
            "seed": 8505,
            "drag_scale": [1.0, 1.0],
            "thrust_scale": [1.0, 1.0],
            "fin_scale": [1.0, 1.0],
            "added_mass_scale": [1.0, 1.0],
            "actuator_time_scale": [1.0, 1.0],
        }
    lower, upper = scale
    return {
        "enabled": True,
        "seed": 8505,
        "drag_scale": [lower, upper],
        "thrust_scale": [lower, upper],
        "fin_scale": [lower, upper],
        "added_mass_scale": [lower, upper],
        "actuator_time_scale": [1.0, 1.0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=CONFIG_DIR / "goal85b"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        for scenario in SCENARIOS:
            for root in ROOTS:
                source = (
                    CONFIG_DIR
                    / f"pc_goal85_unbiased_{scenario}_{root}_20261110.json"
                )
                data = json.loads(source.read_text(encoding="utf-8"))
                name = f"pc_goal85b_{level}_{scenario}_{root}_20261110"
                data["plant_randomization"] = _plant_randomization(level)
                data["experiment"]["name"] = name
                data["experiment"]["output"] = f"artifacts/lowlevel/{name}"
                target = args.output_dir / f"{name}.json"
                target.write_text(
                    json.dumps(data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
