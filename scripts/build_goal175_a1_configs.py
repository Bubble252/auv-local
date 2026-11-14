#!/usr/bin/env python3
"""Materialize the Goal175 A1 COMP/NONE scenario pair without changing route geometry."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import build_pc_standing_eval_configs as standing

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_ENV = REPO_ROOT / "configs/lowlevel/goal173/env_route_health_train_nu_ground.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/lowlevel/goal175_alos_comp_mlp_20261114/configs"
MODES = ("ALOS-COMP", "ALOS-NONE")
PRESSURES = ("nominal", "current10", "noise_n2", "plant20")
RADII = ("default", "strict125")
SCHEMA_VERSION = "auv_goal175_a1_manifest.v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _canonical_scenario_id(scenario_id: str) -> str:
    return scenario_id.removesuffix("__a-ALOS-NONE")


def _validate_config(payload: dict[str, Any], *, mode: str, path: Path) -> None:
    guidance = payload["guidance"]
    expected = {
        "ALOS-COMP": (True, False, "estimated"),
        "ALOS-NONE": (False, False, "none"),
    }[mode]
    observed = (
        guidance["compensate_current"],
        guidance["current_crab_enabled"],
        guidance["current_compensation_source"],
    )
    if observed != expected:
        raise ValueError(f"{path}: ALOS mode drifted: {observed!r} != {expected!r}")
    if guidance["terminal_alignment_min_m"] != 50.0 or guidance["terminal_alignment_max_m"] != 50.0:
        raise ValueError(f"{path}: Goal173 50 m terminal alignment drifted")
    reference = payload["reference"]
    if reference["path_family"] is not None or reference["path_family_mix"] is not None:
        raise ValueError(f"{path}: route family drifted from legacy_sine")
    frozen = {
        "generator": "sinusoidal_3d_arclength",
        "length_m": 35.0,
        "speed_mps": 1.2,
        "path_bank_seed": 41,
        "path_bank_size": 32,
    }
    for key, value in frozen.items():
        if reference.get(key) != value:
            raise ValueError(f"{path}: reference.{key} drifted: {reference.get(key)!r} != {value!r}")
    if payload["observation"]["current_contract"] != "hidden" or payload["observation"]["expose_current"]:
        raise ValueError(f"{path}: actor current observation is not hidden")
    if payload["experiment"]["pc_stage"] != "C3":
        raise ValueError(f"{path}: pc_stage must remain C3")


def build(*, base_env_path: Path, suite_path: Path, output_dir: Path) -> dict[str, Any]:
    suite = _load(suite_path)
    base_env = _load(base_env_path)
    scenarios: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_dir = output_dir / mode
        manifest = standing.build_manifest(
            suite=suite,
            base_env=base_env,
            tier_id="A_screen",
            suite_path=suite_path,
            base_env_path=base_env_path,
            output_dir=mode_dir,
            radii=list(RADII),
            pressure_ids=list(PRESSURES),
            routes=["legacy_sine"],
            lengths=[35.0],
            speeds=[1.2],
            alos_current_mode=mode,
        )
        for scenario in manifest["scenarios"]:
            scenario_id = _canonical_scenario_id(str(scenario["id"]))
            payload = copy.deepcopy(scenario["payload"])
            config_path = mode_dir / f"{scenario_id}.json"
            payload["experiment"]["name"] = f"goal175_a1_{scenario_id}__{mode.lower()}"[:240]
            payload["experiment"]["output"] = _relative(config_path.with_suffix(""))
            _validate_config(payload, mode=mode, path=config_path)
            standing._write(config_path, payload)
            row = scenarios.setdefault(
                scenario_id,
                {
                    "scenario": scenario_id,
                    "pressure": scenario["axes"]["pressure"],
                    "radius": scenario["axes"]["goal_radius"],
                    "goal_radius_m": scenario["axes"]["goal_radius_m"],
                    "configs": {},
                    "sha256": {},
                },
            )
            row["configs"][mode] = _relative(config_path)
            row["sha256"][mode] = standing._sha256(config_path)

    if len(scenarios) != 8:
        raise ValueError(f"expected 8 A1 scenarios, got {len(scenarios)}")
    for scenario_id, row in scenarios.items():
        if set(row["configs"]) != set(MODES):
            raise ValueError(f"{scenario_id}: missing COMP/NONE config pair")

    result = {
        "schema_version": SCHEMA_VERSION,
        "goal_id": "Goal175",
        "phase": "A1",
        "base_environment_config": _relative(base_env_path),
        "base_environment_sha256": standing._sha256(base_env_path),
        "suite": _relative(suite_path),
        "suite_sha256": standing._sha256(suite_path),
        "axes": {
            "alos_current_modes": list(MODES),
            "pressure_profiles": list(PRESSURES),
            "goal_radii": list(RADII),
            "route_profile": "legacy_sine",
            "reference_length_m": 35.0,
            "reference_speed_mps": 1.2,
            "path_bank_seed": 41,
            "path_bank_size": 32,
            "terminal_alignment_m": 50.0,
        },
        "scenario_count": len(scenarios),
        "scenarios": [scenarios[key] for key in sorted(scenarios)],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "a1_manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"manifest": str(manifest_path), "scenario_count": len(scenarios)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-env", type=Path, default=DEFAULT_BASE_ENV)
    parser.add_argument("--suite", type=Path, default=standing.DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = args.base_env if args.base_env.is_absolute() else REPO_ROOT / args.base_env
    suite = args.suite if args.suite.is_absolute() else REPO_ROOT / args.suite
    output = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    print(json.dumps(build(base_env_path=base, suite_path=suite, output_dir=output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
