#!/usr/bin/env python3
"""Materialize Goal182 evaluation scenarios, including absolute depth axes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from auv_lowlevel._paths import resolve_from_root

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs/governance/goal182_eval_matrix.json"
DEFAULT_BASE_ENV = REPO_ROOT / "configs/lowlevel/goal182/env_path_fixed_speed_formal.json"
DEFAULT_PRESSURE_SUITE = REPO_ROOT / "configs/lowlevel/eval/pc_standing_eval_suite_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/lowlevel/goal182/eval_configs"
MANIFEST_SCHEMA = "auv_goal182_eval_env_manifest.v1"
DEFAULT_RADIUS_M = 2.0  # magic-number-ok: Goal182 active evaluation fixes plant.goal_radius_m
ROUTE_LENGTH_M = 70.0  # magic-number-ok: Goal182 EV3/EV4 route length
DEPTH_ENVELOPE_MARGIN_M = 10.0  # magic-number-ok: termination/bounds headroom above absolute depth
PLANT_SEED = 13420  # magic-number-ok: frozen Goal182 pressure profile plant seed
ROUTE_AMPLITUDES_M = (0.5, 2.0, 5.0)  # magic-number-ok: Goal182 EV3 amplitude axis
ROUTE_CYCLES = (2.0, 3.0, 5.0)  # magic-number-ok: Goal182 EV3 cycles axis
PRESSURE_PROFILE_IDS = {
    "nominal": "nominal",
    "current": "current10",
    "noise": "noise_n2",
    "plant": "plant20",
}


def _amplitude_tag(value: float) -> str:
    return f"{value:g}".replace(".", "P")


def _route_specs() -> dict[str, dict[str, float]]:
    specs: dict[str, dict[str, float]] = {
        "R_STRAIGHT": {
            "lateral_amplitude_m": 0.0,
            "lateral_cycles": 0.0,
            "depth_amplitude_m": 0.0,
            "depth_cycles": 0.0,
            "depth_slope_m": 0.0,
        }
    }
    for amplitude in ROUTE_AMPLITUDES_M:
        for cycles in ROUTE_CYCLES:
            tag = f"A{_amplitude_tag(amplitude)}_C{cycles:g}"
            specs[f"R_VERT_{tag}"] = {
                "lateral_amplitude_m": 0.0,
                "lateral_cycles": 0.0,
                "depth_amplitude_m": float(amplitude),
                "depth_cycles": float(cycles),
                "depth_slope_m": 0.0,
            }
    for amplitude in ROUTE_AMPLITUDES_M:
        for cycles in ROUTE_CYCLES:
            tag = f"A{_amplitude_tag(amplitude)}_C{cycles:g}"
            specs[f"R_LAT_{tag}"] = {
                "lateral_amplitude_m": float(amplitude),
                "lateral_cycles": float(cycles),
                "depth_amplitude_m": 0.0,
                "depth_cycles": 0.0,
                "depth_slope_m": 0.0,
            }
    for level, amplitude, cycles in (("S", 0.5, 2.0), ("M", 2.0, 3.0), ("L", 5.0, 5.0)):
        specs[f"R_3D_{level}"] = {
            "lateral_amplitude_m": amplitude,
            "lateral_cycles": cycles,
            "depth_amplitude_m": amplitude,
            "depth_cycles": cycles,
            "depth_slope_m": 0.0,
        }
    return specs


ROUTE_SPECS = _route_specs()
ROUTE_GENERALIZATION_ROUTES = tuple(ROUTE_SPECS)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _num_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def scenario_id(*, pressure: str, route: str, depth_m: float, speed_mps: float) -> str:
    return (
        f"p-{pressure}__r-{route}__d{_num_tag(depth_m)}"
        f"__l{_num_tag(ROUTE_LENGTH_M)}__v{_num_tag(speed_mps)}__g-default"
    )


def _tier(matrix: dict[str, Any], tier_id: str) -> dict[str, Any]:
    tiers = matrix.get("tiers")
    if not isinstance(tiers, list):
        raise ValueError("eval matrix tiers must be a list")
    for tier in tiers:
        if isinstance(tier, dict) and tier.get("tier_id") == tier_id:
            return tier
    raise ValueError(f"unknown eval tier: {tier_id}")


def _selected(values: list[Any], requested: list[Any] | None, *, name: str) -> list[Any]:
    if requested is None:
        return list(values)
    missing = sorted(set(requested) - set(values), key=str)
    if missing:
        raise ValueError(f"unsupported {name}: {missing}; available={values}")
    return [value for value in values if value in set(requested)]


def _pressure_profile_map(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = suite.get("pressure_profiles")
    if not isinstance(profiles, list):
        raise ValueError("pressure suite must define pressure_profiles")
    result: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, dict) or "id" not in profile:
            raise ValueError("every pressure profile must be an object with an id")
        result[str(profile["id"])] = profile
    missing = sorted(set(PRESSURE_PROFILE_IDS.values()) - set(result))
    if missing:
        raise ValueError(f"pressure suite is missing required profiles: {missing}")
    return result


def _scale_shear(current: dict[str, Any], scale: float) -> None:
    if scale < 0.0:
        raise ValueError("current shear scale must be non-negative")
    shear = current.get("shear")
    if not isinstance(shear, dict):
        if scale != 0.0:
            raise ValueError("profile requests non-zero shear but base current has no shear")
        return
    gradient = shear.get("velocity_gradient_per_s")
    if not isinstance(gradient, list):
        raise ValueError("current.shear.velocity_gradient_per_s must be a 2-D array")
    shear["velocity_gradient_per_s"] = [
        [float(value) * float(scale) for value in row] for row in gradient
    ]


def apply_pressure(payload: dict[str, Any], profile: dict[str, Any]) -> None:
    current = payload.setdefault("current", {})
    if not isinstance(current, dict):
        raise ValueError("environment current must be an object")
    speed_range = profile.get("current_speed_range_mps", "use_base_config")
    if speed_range != "use_base_config":
        if not isinstance(speed_range, list) or len(speed_range) != 2:
            raise ValueError("pressure current_speed_range_mps must be a 2-item list")
        current["speed_mps"] = [float(speed_range[0]), float(speed_range[1])]
    shear_scale = profile.get("current_shear_scale", "use_base_config")
    if shear_scale != "use_base_config":
        _scale_shear(current, float(shear_scale))
    noise_range = profile.get("noise_scale_range")
    if noise_range is not None:
        if not isinstance(noise_range, list) or len(noise_range) != 2:
            raise ValueError("pressure noise_scale_range must be a null or 2-item list")
        sensors = payload.setdefault("sensors", {})
        if not isinstance(sensors, dict):
            raise ValueError("environment sensors must be an object")
        sensors["noise_scale"] = 1.0
        sensors["noise_scale_range"] = [float(noise_range[0]), float(noise_range[1])]
    plant_range = profile.get("plant_scale_range")
    if plant_range is not None:
        if not isinstance(plant_range, list) or len(plant_range) != 2:
            raise ValueError("pressure plant_scale_range must be a null or 2-item list")
        lower, upper = float(plant_range[0]), float(plant_range[1])
        payload["plant_randomization"] = {
            "enabled": True,
            "seed": PLANT_SEED,
            "drag_scale": [lower, upper],
            "thrust_scale": [lower, upper],
            "fin_scale": [lower, upper],
            "added_mass_scale": [lower, upper],
            "actuator_time_scale": [lower, upper],
        }


def apply_depth_envelope(payload: dict[str, Any], *, depth_m: float) -> None:
    if depth_m <= 0.0:
        raise ValueError("absolute depth must be positive")
    reference = payload.setdefault("reference", {})
    plant = payload.setdefault("plant", {})
    termination = payload.setdefault("termination", {})
    if not isinstance(reference, dict) or not isinstance(plant, dict):
        raise ValueError("environment reference and plant must be objects")
    if not isinstance(termination, dict):
        raise ValueError("environment termination must be an object")
    reference["depth_start_m"] = float(depth_m)
    required_max_depth = float(depth_m) + DEPTH_ENVELOPE_MARGIN_M
    current_max_depth = float(termination.get("max_depth_m", 0.0))
    termination["max_depth_m"] = max(current_max_depth, required_max_depth)
    bounds = plant.get("bounds_m")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 3
        or not isinstance(bounds[2], list)
        or len(bounds[2]) != 2
    ):
        raise ValueError("plant.bounds_m must be a 3x2 nested list")
    bounds[2][0] = min(float(bounds[2][0]), 0.0)
    bounds[2][1] = max(float(bounds[2][1]), required_max_depth)


def apply_route(payload: dict[str, Any], *, route: str, depth_m: float, speed_mps: float) -> None:
    if route not in ROUTE_SPECS:
        raise ValueError(f"unsupported Goal182 route: {route}; available={sorted(ROUTE_SPECS)}")
    reference = payload.setdefault("reference", {})
    if not isinstance(reference, dict):
        raise ValueError("environment reference must be an object")
    reference.update(ROUTE_SPECS[route])
    reference["path_family"] = None
    reference["path_family_mix"] = None
    reference["path_bank_size"] = 1
    reference["length_m"] = ROUTE_LENGTH_M
    reference["speed_mps"] = float(speed_mps)
    reference["x_span_m"] = ROUTE_LENGTH_M
    reference["depth_start_m"] = float(depth_m)


def apply_eval_contract(payload: dict[str, Any], *, scenario: str, stage: str = "C6") -> None:
    plant = payload.setdefault("plant", {})
    if not isinstance(plant, dict):
        raise ValueError("environment plant must be an object")
    plant["goal_radius_m"] = DEFAULT_RADIUS_M
    curriculum = payload.get("curriculum")
    if isinstance(curriculum, dict):
        curriculum["enabled"] = False
        curriculum.pop("stage_mix", None)
    experiment = payload.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("environment experiment must be an object")
    source_name = str(experiment.get("name", "goal182_eval"))
    experiment["name"] = f"{source_name}_{scenario}"[:240]
    experiment["output"] = f"artifacts/lowlevel/goal182/eval_configs/{scenario}"
    experiment["pc_stage"] = stage


def _scenario(
    *,
    sid: str,
    path: Path,
    payload: dict[str, Any],
    pressure: str,
    route: str,
    depth_m: float,
    speed_mps: float,
) -> dict[str, Any]:
    return {
        "id": sid,
        "config": _relative(path),
        "payload": payload,
        "axes": {
            "pressure": pressure,
            "route_profile": route,
            "depth_m": depth_m,
            "length_m": ROUTE_LENGTH_M,
            "speed_mps": speed_mps,
            "goal_radius": "default",
            "goal_radius_m": DEFAULT_RADIUS_M,
            "stage": "C6",
        },
    }


def build_manifest(
    *,
    matrix: dict[str, Any],
    base_env: dict[str, Any],
    tier_id: str,
    matrix_path: Path,
    base_env_path: Path,
    output_dir: Path,
    depths: list[float] | None = None,
    routes: list[str] | None = None,
    pressures: list[str] | None = None,
    pressure_suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tier = _tier(matrix, tier_id)
    if tier_id not in {"DEPTH_GENERALIZATION", "ROUTE_GENERALIZATION"}:
        raise ValueError(
            "this builder materializes DEPTH_GENERALIZATION or ROUTE_GENERALIZATION only"
        )
    radii = [float(value) for value in tier.get("radii_m", [])]
    if radii != [DEFAULT_RADIUS_M]:
        raise ValueError(f"Goal182 active radii must be [2.0]; got {radii}")
    stages = [str(value) for value in tier.get("stage_levels", [])]
    if stages != ["C6"]:
        raise ValueError(f"Goal182 evaluation tier must use C6; got {stages}")
    speed_mps = float(base_env.get("reference", {}).get("speed_mps", 0.0))
    if not 0.0 < speed_mps <= 2.5:
        raise ValueError(f"base reference speed must be in (0, 2.5]; got {speed_mps}")

    scenarios: list[dict[str, Any]] = []
    if tier_id == "DEPTH_GENERALIZATION":
        requested_pressures = (
            ["nominal"] if pressures is None else _selected(["nominal"], pressures, name="pressure")
        )
        if requested_pressures != ["nominal"]:
            raise ValueError(f"DEPTH_GENERALIZATION must use nominal pressure; got {pressures}")
        requested_depths = [
            float(value)
            for value in _selected(
                [float(value) for value in tier.get("depths_m", [])], depths, name="depth"
            )
        ]
        requested_routes = [
            str(value)
            for value in _selected(
                [str(value) for value in tier.get("route_profiles", [])], routes, name="route"
            )
        ]
        if not requested_depths or not requested_routes:
            raise ValueError("no depth/route scenarios selected")
        depth_axis = requested_depths
        for depth_m in requested_depths:
            for route in requested_routes:
                sid = scenario_id(
                    pressure="nominal", route=route, depth_m=depth_m, speed_mps=speed_mps
                )
                payload = copy.deepcopy(base_env)
                apply_depth_envelope(payload, depth_m=depth_m)
                apply_route(payload, route=route, depth_m=depth_m, speed_mps=speed_mps)
                apply_depth_envelope(payload, depth_m=depth_m)
                apply_eval_contract(payload, scenario=sid, stage="C6")
                scenarios.append(
                    _scenario(
                        sid=sid,
                        path=output_dir / f"{sid}.json",
                        payload=payload,
                        pressure="nominal",
                        route=route,
                        depth_m=depth_m,
                        speed_mps=speed_mps,
                    )
                )
    elif tier_id == "ROUTE_GENERALIZATION":
        expected_route_count = int(tier.get("route_profile_count", 0))
        if expected_route_count != len(ROUTE_GENERALIZATION_ROUTES):
            raise ValueError(
                f"ROUTE_GENERALIZATION route count {expected_route_count} != "
                f"generated {len(ROUTE_GENERALIZATION_ROUTES)}"
            )
        if depths is not None:
            raise ValueError("ROUTE_GENERALIZATION has no absolute-depth axis")
        requested_pressures = [
            str(value)
            for value in _selected(
                [str(value) for value in tier.get("pressure_levels", [])],
                pressures,
                name="pressure",
            )
        ]
        requested_routes = [
            str(value)
            for value in _selected(list(ROUTE_GENERALIZATION_ROUTES), routes, name="route")
        ]
        if not requested_pressures or not requested_routes:
            raise ValueError("no pressure/route scenarios selected")
        suite = pressure_suite
        if suite is None:
            suite = _load_json(_resolve(DEFAULT_PRESSURE_SUITE))
        profiles = _pressure_profile_map(suite)
        depth_m = float(base_env.get("reference", {}).get("depth_start_m", 10.0))
        depth_axis = [depth_m]
        for pressure in requested_pressures:
            profile = profiles[PRESSURE_PROFILE_IDS[pressure]]
            for route in requested_routes:
                sid = scenario_id(
                    pressure=pressure, route=route, depth_m=depth_m, speed_mps=speed_mps
                )
                payload = copy.deepcopy(base_env)
                apply_pressure(payload, profile)
                apply_route(payload, route=route, depth_m=depth_m, speed_mps=speed_mps)
                apply_depth_envelope(payload, depth_m=depth_m)
                apply_eval_contract(payload, scenario=sid, stage="C6")
                scenarios.append(
                    _scenario(
                        sid=sid,
                        path=output_dir / f"{sid}.json",
                        payload=payload,
                        pressure=pressure,
                        route=route,
                        depth_m=depth_m,
                        speed_mps=speed_mps,
                    )
                )
    else:
        raise ValueError(
            "this builder materializes DEPTH_GENERALIZATION or ROUTE_GENERALIZATION only"
        )

    scenario_count = len(scenarios)
    train_seed_count = int(tier.get("train_seed_count", len(matrix.get("train_seeds", []))))
    matrix_eval_seeds = [int(value) for value in matrix.get("eval_seeds", [])]
    if not matrix_eval_seeds:
        raise ValueError("eval matrix must define eval_seeds")
    if matrix_eval_seeds != list(
        range(matrix_eval_seeds[0], matrix_eval_seeds[0] + len(matrix_eval_seeds))
    ):
        raise ValueError("eval matrix eval_seeds must be contiguous for the standing runner")
    full_selection = depths is None and routes is None and pressures is None
    tier_expected_cells = int(tier.get("expected_cells", 0))
    computed_cells = scenario_count * train_seed_count
    if full_selection and computed_cells != tier_expected_cells:
        raise ValueError(
            f"expected_cells {tier_expected_cells} != scenario_count {scenario_count} x "
            f"train_seed_count {train_seed_count}"
        )
    expected_cells = tier_expected_cells if full_selection else computed_cells
    expected_episodes = int(tier.get("expected_episodes_per_arm", 0))
    if not full_selection:
        eval_seed_count = int(tier.get("eval_seed_count", 0))
        expected_episodes = computed_cells * eval_seed_count
    return {
        "schema_version": MANIFEST_SCHEMA,
        "goal_id": matrix.get("goal_id"),
        "matrix_id": matrix.get("matrix_id"),
        "matrix": _relative(matrix_path),
        "matrix_sha256": _sha256(matrix_path),
        "tier": tier_id,
        "base_environment_config": _relative(base_env_path),
        "base_environment_sha256": _sha256(base_env_path),
        "axes": {
            "depth_profiles_m": depth_axis,
            "route_profiles": requested_routes,
            "pressure_profiles": requested_pressures,
            "stage_levels": stages,
            "goal_radii_m": radii,
        },
        "eval_seeds": len(matrix_eval_seeds),
        "eval_seeds_start": matrix_eval_seeds[0],
        "eval_seed_values": matrix_eval_seeds,
        "train_seed_count": train_seed_count,
        "scenario_count": scenario_count,
        "expected_cells": expected_cells,
        "expected_episodes_per_arm": expected_episodes,
        "scenarios": scenarios,
    }


def write_manifest(manifest: dict[str, Any], *, output_dir: Path, overwrite: bool) -> None:
    for scenario in manifest["scenarios"]:
        path = Path(scenario["config"])
        if path.exists() and not overwrite:
            raise FileExistsError(f"config already exists (use --overwrite): {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(scenario["payload"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        scenario["sha256"] = _sha256(path)
        scenario.pop("payload")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists (use --overwrite): {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--tier", default="DEPTH_GENERALIZATION")
    parser.add_argument("--base-env", type=Path, default=DEFAULT_BASE_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--depth", nargs="*", type=float, default=None)
    parser.add_argument("--route", nargs="*", default=None)
    parser.add_argument("--pressure", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    matrix_path = _resolve(args.matrix)
    base_env_path = _resolve(args.base_env)
    output_dir = _resolve(args.output_dir)
    manifest = build_manifest(
        matrix=_load_json(matrix_path),
        base_env=_load_json(base_env_path),
        tier_id=args.tier,
        matrix_path=matrix_path,
        base_env_path=base_env_path,
        output_dir=output_dir,
        depths=args.depth,
        routes=args.route,
        pressures=args.pressure,
    )
    if args.dry_run:
        preview = copy.deepcopy(manifest)
        for scenario in preview["scenarios"]:
            scenario.pop("payload")
        print(json.dumps(preview, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    write_manifest(manifest, output_dir=output_dir, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "tier": args.tier,
                "scenario_count": manifest["scenario_count"],
                "depths_m": manifest["axes"]["depth_profiles_m"],
                "route_profiles": manifest["axes"]["route_profiles"],
                "pressure_profiles": manifest["axes"]["pressure_profiles"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
