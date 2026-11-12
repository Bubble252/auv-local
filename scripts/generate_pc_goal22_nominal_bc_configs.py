#!/usr/bin/env python3
"""Generate the Goal22 nominal/ideal BC environment matrix without rewriting old configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "lowlevel"
BASE_DEFAULT = CONFIG_DIR / "odarrl_control_pc_c3_random_realistic_v1_nominal_20261107.json"
BASE_STRICT = (
    CONFIG_DIR
    / "odarrl_control_pc_c3_random_realistic_v1_nominal_strict125_20261107.json"
)
OUTPUT_ROOT = "artifacts/lowlevel/pc_v0d_nominal_bc_rebase_20261107/envs"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant(
    base: dict[str, Any],
    *,
    path_dim: int,
    strict125: bool,
    ideal_actor: bool,
) -> dict[str, Any]:
    data = copy.deepcopy(base)
    family = "ideal" if ideal_actor else "nominal"
    scenario = "strict125" if strict125 else "default"
    name = f"pc_c3_random_realistic_v1_{family}_bc_n{path_dim}_{scenario}_20261107"
    data["experiment"]["name"] = name
    data["experiment"]["output"] = f"{OUTPUT_ROOT}/{name}"
    data["observation"]["path_endpoint_context_enabled"] = path_dim == 12
    data["observation"]["ideal_actor_observations"] = ideal_actor
    data["termination"]["pc_goal_plane_miss_termination_enabled"] = True
    data["termination"]["goal_plane_miss_penalty"] = 0.0
    data["termination"]["goal_plane_miss_distance_weight"] = 0.0
    if strict125:
        data["plant"]["goal_radius_m"] = 1.25
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("all", "nominal", "ideal"),
        default="all",
        help="generate only one observation family while preserving the other",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bases = {False: _load(BASE_DEFAULT), True: _load(BASE_STRICT)}
    written: list[Path] = []
    for ideal_actor in (False, True):
        family = "ideal" if ideal_actor else "nominal"
        if args.family not in ("all", family):
            continue
        for path_dim in (8, 12):
            for strict125 in (False, True):
                data = _variant(
                    bases[strict125],
                    path_dim=path_dim,
                    strict125=strict125,
                    ideal_actor=ideal_actor,
                )
                scenario = "strict125" if strict125 else "default"
                path = CONFIG_DIR / (
                    f"odarrl_control_pc_c3_random_realistic_v1_{family}_bc_"
                    f"n{path_dim}_{scenario}_20261107.json"
                )
                if path.exists() and not args.overwrite:
                    raise FileExistsError(f"refusing to overwrite existing config: {path}")
                path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                written.append(path)
    for path in written:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
