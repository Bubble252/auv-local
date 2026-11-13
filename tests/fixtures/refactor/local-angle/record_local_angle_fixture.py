#!/usr/bin/env python3
"""Record scalar angle wrapper outputs for refactor equivalence checks."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

from auv_local.line_detour import _wrap_angle as line_detour_wrap_angle
from auv_rl.bc.teacher import _wrap_angle as teacher_wrap_angle
from auv_rl.observation import _wrap_angle as observation_wrap_angle
from auv_rl.planar import _wrap_angle as planar_wrap_angle

ANGLES = (-7.0, -math.pi, -3.5, -0.1, 0.0, 0.1, 3.5, math.pi, 7.0)
WRAPPERS: dict[str, Callable[[float], float]] = {
    "line_detour": line_detour_wrap_angle,
    "planar": planar_wrap_angle,
    "observation": observation_wrap_angle,
    "teacher": teacher_wrap_angle,
}


def build_fixture() -> dict[str, object]:
    """Build a deterministic JSON-compatible fixture."""
    return {
        "randomized": False,
        "inputs": {"angles_rad": list(ANGLES)},
        "outputs": {
            name: [wrapper(angle) for angle in ANGLES] for name, wrapper in WRAPPERS.items()
        },
        "exception": None,
        "manifest": {
            "candidate": "LOCAL-A1",
            "helper": "scalar wrap_angle",
            "unit": "rad",
        },
    }


def main() -> int:
    """Write the fixture to the requested path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
