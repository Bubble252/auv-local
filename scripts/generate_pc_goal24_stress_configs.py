#!/usr/bin/env python3
"""Generate teacher-fixed PC stress configs without touching nominal configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "lowlevel"
OUTPUT_ROOT = "artifacts/lowlevel/pc_v1_stress_mix_20261108/envs"
BASES = {
    "default": "odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_default_20261108.json",
    "strict125": "odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_strict125_20261108.json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_config(base: dict[str, Any], scenario: str) -> dict[str, Any]:
    data = copy.deepcopy(base)
    name = f"pc_c3_random_realistic_v1_stress_teacher_fixed_{scenario}_20261108"
    data["experiment"]["name"] = name
    data["experiment"]["output"] = f"{OUTPUT_ROOT}/{name}"
    data["sensors"]["profile"] = "realistic_stress"
    data["sensors"]["seed"] = 82300
    return data


def main() -> int:
    args = _parse_args()
    written: list[Path] = []
    for scenario, base_name in BASES.items():
        base_path = CONFIG_DIR / base_name
        data = json.loads(base_path.read_text(encoding="utf-8"))
        variant = build_config(data, scenario)
        output = CONFIG_DIR / (
            f"odarrl_control_pc_c3_random_realistic_v1_stress_teacher_fixed_"
            f"{scenario}_20261108.json"
        )
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing config: {output}")
        output.write_text(json.dumps(variant, indent=2) + "\n", encoding="utf-8")
        written.append(output)
    for path in written:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
