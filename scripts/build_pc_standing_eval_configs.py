#!/usr/bin/env python3
"""Materialize the frozen PC standing-eval axes as auditable environment configs.

This builder deliberately changes only explicit evaluation axes:

* pressure: current speed/shear, sensor-noise scale, plant randomization;
* geometry: route family, reference length and speed;
* ALOS ocean-current mode: default ALOS-COMP, optional ALOS-NONE ablation;
* success radius: default=2.0 m or strict125=1.25 m.

It disables the training curriculum in generated configs so one scenario does
not silently mix stages. The source config supplies all non-evaluation fields.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO_ROOT / "configs/lowlevel/eval/pc_standing_eval_suite_v1.json"
DEFAULT_BASE_ENV = REPO_ROOT / "configs/lowlevel/goal136/curriculum/C1.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/lowlevel/pc_standing_eval/configs"
MANIFEST_SCHEMA = "auv_pc_standing_eval_env_manifest.v1"
PLANT_SEED = 13420


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


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _num_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def scenario_id(
    *,
    pressure: str,
    route: str,
    length_m: float,
    speed_mps: float,
    radius: str,
    alos_current_mode: str = "ALOS-COMP",
    default_alos_current_mode: str = "ALOS-COMP",
) -> str:
    base = (
        f"p-{pressure}__r-{route}__l{_num_tag(length_m)}"
        f"__v{_num_tag(speed_mps)}__g-{radius}"
    )
    if alos_current_mode == default_alos_current_mode:
        return base
    return f"{base}__a-{alos_current_mode}"


def _profile_map(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = suite.get("pressure_profiles")
    if not isinstance(profiles, list):
        raise ValueError("suite.pressure_profiles must be a list")
    result: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, dict) or "id" not in profile:
            raise ValueError("every pressure profile must be an object with an id")
        result[str(profile["id"])] = profile
    return result


def _tier_map(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tiers = suite.get("tiers")
    if not isinstance(tiers, list):
        raise ValueError("suite.tiers must be a list")
    result: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        if not isinstance(tier, dict) or "id" not in tier:
            raise ValueError("every tier must be an object with an id")
        result[str(tier["id"])] = tier
    return result


def _alos_current_mode_map(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modes = suite.get("alos_current_modes")
    if not isinstance(modes, list):
        raise ValueError("suite.alos_current_modes must be a list")
    result: dict[str, dict[str, Any]] = {}
    for mode in modes:
        if not isinstance(mode, dict) or "id" not in mode:
            raise ValueError("every ALOS current mode must be an object with an id")
        result[str(mode["id"])] = mode
    if set(result) != {"ALOS-NONE", "ALOS-COMP"}:
        raise ValueError("suite must define exactly ALOS-NONE and ALOS-COMP")
    return result


def _selected(values: Iterable[Any], requested: list[Any] | None, *, name: str) -> list[Any]:
    available = list(values)
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available), key=str)
    if missing:
        raise ValueError(f"unsupported {name}: {missing}; available={available}")
    return [value for value in available if value in set(requested)]


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
        [float(value) * float(scale) for value in row]
        for row in gradient
    ]


def apply_pressure(
    payload: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Apply only the pressure fields declared by one suite profile."""

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
    return payload


def apply_geometry(
    payload: dict[str, Any],
    *,
    route: str,
    length_m: float,
    speed_mps: float,
) -> dict[str, Any]:
    reference = payload.setdefault("reference", {})
    if not isinstance(reference, dict):
        raise ValueError("environment reference must be an object")
    if route == "legacy_sine":
        reference["path_family"] = None
        reference["path_family_mix"] = None
    elif route in {"P0", "P1", "P2", "P3", "P4"}:
        reference["path_family"] = route
        reference["path_family_mix"] = None
    else:
        raise ValueError(f"unsupported route profile: {route}")
    if not 0.0 < float(speed_mps) <= 2.5:
        raise ValueError("reference speed must be in (0, 2.5] m/s")
    reference["length_m"] = float(length_m)
    reference["speed_mps"] = float(speed_mps)
    return payload


def apply_alos_current_mode(
    payload: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Apply exactly one canonical ALOS ocean-current mode."""

    guidance = payload.setdefault("guidance", {})
    if not isinstance(guidance, dict):
        raise ValueError("environment guidance must be an object")
    guidance["compensate_current"] = bool(profile["compensate_current"])
    guidance["current_crab_enabled"] = bool(profile["current_crab_enabled"])
    guidance["current_compensation_source"] = str(profile["current_compensation_source"])
    return payload


def apply_eval_contract(
    payload: dict[str, Any],
    *,
    radius: str,
    goal_radius_m: float,
    scenario: str,
) -> dict[str, Any]:
    plant = payload.setdefault("plant", {})
    if not isinstance(plant, dict):
        raise ValueError("environment plant must be an object")
    plant["goal_radius_m"] = float(goal_radius_m)

    curriculum = payload.get("curriculum")
    if curriculum is not None:
        if not isinstance(curriculum, dict):
            raise ValueError("environment curriculum must be an object")
        curriculum["enabled"] = False
        curriculum.pop("stage_mix", None)

    experiment = payload.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("environment experiment must be an object")
    source_name = str(experiment.get("name", "pc_standing_eval"))
    experiment["name"] = f"{source_name}_{scenario}_{radius}"[:240]
    experiment["output"] = f"artifacts/lowlevel/pc_standing_eval/{scenario}/{radius}"
    return payload


def build_manifest(
    *,
    suite: dict[str, Any],
    base_env: dict[str, Any],
    tier_id: str,
    suite_path: Path,
    base_env_path: Path,
    output_dir: Path,
    radii: list[str] | None = None,
    pressure_ids: list[str] | None = None,
    routes: list[str] | None = None,
    lengths: list[float] | None = None,
    speeds: list[float] | None = None,
    alos_current_mode: str | None = None,
) -> dict[str, Any]:
    tiers = _tier_map(suite)
    if tier_id not in tiers:
        raise ValueError(f"unknown tier: {tier_id}; available={sorted(tiers)}")
    tier = tiers[tier_id]
    profiles = _profile_map(suite)
    alos_modes = _alos_current_mode_map(suite)
    default_alos_mode = str(suite.get("default_alos_current_mode", "ALOS-COMP"))
    selected_alos_mode = str(alos_current_mode or default_alos_mode)
    if selected_alos_mode not in alos_modes:
        raise ValueError(
            f"unsupported ALOS current mode: {selected_alos_mode}; "
            f"available={sorted(alos_modes)}"
        )

    requested_pressures = _selected(tier["pressure_profiles"], pressure_ids, name="pressure profile")
    requested_routes = _selected(tier["route_profiles"], routes, name="route profile")
    requested_lengths = [
        float(value)
        for value in _selected(tier["length_profiles_m"], lengths, name="length profile")
    ]
    requested_speeds = [
        float(value)
        for value in _selected(tier["speed_profiles_mps"], speeds, name="speed profile")
    ]
    goal_radii = suite.get("success_contract", {}).get("goal_radii_m")
    if not isinstance(goal_radii, dict):
        raise ValueError("suite.success_contract.goal_radii_m must be an object")
    requested_radii = _selected(goal_radii.keys(), radii, name="goal radius")

    scenarios: list[dict[str, Any]] = []
    for pressure in requested_pressures:
        profile = profiles[pressure]
        for route in requested_routes:
            for length_m in requested_lengths:
                for speed_mps in requested_speeds:
                    for radius in requested_radii:
                        sid = scenario_id(
                            pressure=pressure,
                            route=route,
                            length_m=length_m,
                            speed_mps=speed_mps,
                            radius=radius,
                            alos_current_mode=selected_alos_mode,
                            default_alos_current_mode=default_alos_mode,
                        )
                        payload = copy.deepcopy(base_env)
                        apply_pressure(payload, profile)
                        apply_geometry(
                            payload,
                            route=route,
                            length_m=length_m,
                            speed_mps=speed_mps,
                        )
                        apply_alos_current_mode(payload, alos_modes[selected_alos_mode])
                        apply_eval_contract(
                            payload,
                            radius=radius,
                            goal_radius_m=float(goal_radii[radius]),
                            scenario=sid,
                        )
                        path = output_dir / f"{sid}.json"
                        scenarios.append(
                            {
                                "id": sid,
                                "config": _relative(path),
                                "payload": payload,
                                "axes": {
                                    "pressure": pressure,
                                    "route": route,
                                    "length_m": length_m,
                                    "speed_mps": speed_mps,
                                    "goal_radius": radius,
                                    "goal_radius_m": float(goal_radii[radius]),
                                    "alos_current_mode": selected_alos_mode,
                                },
                            }
                        )

    if not scenarios:
        raise ValueError("no scenarios selected")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "suite": _relative(suite_path),
        "suite_sha256": _sha256(suite_path),
        "tier": tier_id,
        "tier_purpose": tier.get("purpose"),
        "training_seeds": tier.get("training_seeds"),
        "eval_seeds": tier.get("eval_seeds"),
        "eval_seeds_start": int(suite.get("dev_seeds", {}).get("start", 20)),
        "base_environment_config": _relative(base_env_path),
        "base_environment_sha256": _sha256(base_env_path),
        "axes": {
            "pressure_profiles": requested_pressures,
            "route_profiles": requested_routes,
            "length_profiles_m": requested_lengths,
            "speed_profiles_mps": requested_speeds,
            "goal_radii": requested_radii,
            "alos_current_mode": selected_alos_mode,
            "alos_current_modes": sorted(alos_modes),
        },
        "speed_max_mps": suite.get("speed_max_mps"),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def write_manifest(manifest: dict[str, Any], *, output_dir: Path, overwrite: bool) -> None:
    for scenario in manifest["scenarios"]:
        path = Path(scenario["config"])
        if path.exists() and not overwrite:
            raise FileExistsError(f"config already exists (use --overwrite): {path}")
        _write(path, scenario["payload"])
        scenario["sha256"] = _sha256(path)
        scenario.pop("payload")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists (use --overwrite): {manifest_path}")
    _write(manifest_path, manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--base-env", type=Path, default=DEFAULT_BASE_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pressure", nargs="*", default=None)
    parser.add_argument("--route", nargs="*", default=None)
    parser.add_argument("--length", nargs="*", type=float, default=None)
    parser.add_argument("--speed", nargs="*", type=float, default=None)
    parser.add_argument("--radius", nargs="*", default=None)
    parser.add_argument("--alos-current-mode", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    suite_path = _resolve(args.suite)
    base_env_path = _resolve(args.base_env)
    output_dir = _resolve(args.output_dir)
    manifest = build_manifest(
        suite=_load_json(suite_path),
        base_env=_load_json(base_env_path),
        tier_id=args.tier,
        suite_path=suite_path,
        base_env_path=base_env_path,
        output_dir=output_dir,
        radii=args.radius,
        pressure_ids=args.pressure,
        routes=args.route,
        lengths=args.length,
        speeds=args.speed,
        alos_current_mode=args.alos_current_mode,
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
                "speed_profiles_mps": manifest["axes"]["speed_profiles_mps"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
