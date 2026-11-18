from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from auv_lowlevel.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_goal182_eval_configs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_goal182_eval_configs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs() -> tuple[dict, dict, Path, Path]:
    builder = _load()
    matrix_path = builder.DEFAULT_MATRIX
    base_path = builder.DEFAULT_BASE_ENV
    return (
        json.loads(matrix_path.read_text(encoding="utf-8")),
        json.loads(base_path.read_text(encoding="utf-8")),
        matrix_path,
        base_path,
    )


def _manifest(tmp_path: Path, **kwargs) -> dict:
    builder = _load()
    matrix, base, matrix_path, base_path = _inputs()
    defaults = {
        "matrix": matrix,
        "base_env": base,
        "tier_id": "DEPTH_GENERALIZATION",
        "matrix_path": matrix_path,
        "base_env_path": base_path,
        "output_dir": tmp_path,
    }
    defaults.update(kwargs)
    return builder.build_manifest(**defaults)


def test_depth_manifest_separates_four_depths_and_two_routes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    matrix, _, _, _ = _inputs()
    assert manifest["scenario_count"] == 8
    assert manifest["expected_cells"] == 24
    assert manifest["axes"]["depth_profiles_m"] == [10.0, 80.0, 250.0, 500.0]
    assert manifest["axes"]["route_profiles"] == ["R_STRAIGHT", "R_3D_M"]
    assert manifest["axes"]["goal_radii_m"] == [2.0]
    assert manifest["eval_seeds_start"] == matrix["eval_seeds"][0]
    assert manifest["eval_seeds"] == len(matrix["eval_seeds"])


def test_depth_and_route_geometry_are_materialized(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, depths=[500.0], routes=["R_STRAIGHT", "R_3D_M"])
    payloads = {
        scenario["axes"]["route_profile"]: scenario["payload"] for scenario in manifest["scenarios"]
    }
    straight = payloads["R_STRAIGHT"]
    assert straight["reference"]["depth_start_m"] == 500.0
    assert straight["reference"]["length_m"] == 70.0
    assert straight["reference"]["path_bank_size"] == 1
    assert straight["reference"]["lateral_amplitude_m"] == 0.0
    assert straight["reference"]["depth_amplitude_m"] == 0.0
    assert straight["termination"]["max_depth_m"] >= 510.0
    assert straight["plant"]["bounds_m"][2][1] >= 510.0
    assert straight["plant"]["goal_radius_m"] == 2.0
    assert straight["curriculum"]["enabled"] is False

    three_d = payloads["R_3D_M"]
    assert three_d["reference"]["lateral_amplitude_m"] == 2.0
    assert three_d["reference"]["lateral_cycles"] == 3.0
    assert three_d["reference"]["depth_amplitude_m"] == 2.0
    assert three_d["reference"]["depth_cycles"] == 3.0


def test_generated_depth_config_parses_and_keeps_absolute_depth(tmp_path: Path) -> None:
    builder = _load()
    manifest = _manifest(tmp_path, depths=[500.0], routes=["R_3D_M"])
    builder.write_manifest(manifest, output_dir=tmp_path, overwrite=False)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    scenario = saved["scenarios"][0]
    config_path = ROOT / scenario["config"]
    config = load_config(config_path)
    assert config.reference.depth_start_m == 500.0
    assert config.termination.max_depth_m >= 510.0
    assert config.plant.goal_radius_m == 2.0
    assert config.curriculum.enabled is False


def test_rejects_unregistered_depth_or_route(tmp_path: Path) -> None:
    builder = _load()
    matrix, base, matrix_path, base_path = _inputs()
    try:
        builder.build_manifest(
            matrix=matrix,
            base_env=base,
            tier_id="DEPTH_GENERALIZATION",
            matrix_path=matrix_path,
            base_env_path=base_path,
            output_dir=tmp_path,
            depths=[400.0],
        )
    except ValueError as error:
        assert "unsupported depth" in str(error)
    else:
        raise AssertionError("400 m must not be accepted by the frozen depth tier")


def test_rejects_non_goal182_tier(tmp_path: Path) -> None:
    builder = _load()
    matrix, base, matrix_path, base_path = _inputs()
    try:
        builder.build_manifest(
            matrix=matrix,
            base_env=base,
            tier_id="MAIN_RANKING",
            matrix_path=matrix_path,
            base_env_path=base_path,
            output_dir=tmp_path,
        )
    except ValueError as error:
        assert "DEPTH_GENERALIZATION or ROUTE_GENERALIZATION only" in str(error)
    else:
        raise AssertionError("MAIN_RANKING must not be silently materialized here")


def test_route_manifest_materializes_22_routes_and_four_pressures(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, tier_id="ROUTE_GENERALIZATION")
    assert manifest["scenario_count"] == 88
    assert manifest["expected_cells"] == 264
    assert manifest["expected_episodes_per_arm"] == 792
    assert manifest["axes"]["depth_profiles_m"] == [10.0]
    assert manifest["axes"]["pressure_profiles"] == ["nominal", "current", "noise", "plant"]
    routes = manifest["axes"]["route_profiles"]
    assert len(routes) == 22
    assert routes[0] == "R_STRAIGHT"
    assert routes[-3:] == ["R_3D_S", "R_3D_M", "R_3D_L"]
    assert sum(route.startswith("R_VERT_") for route in routes) == 9
    assert sum(route.startswith("R_LAT_") for route in routes) == 9


def test_route_geometry_and_pressure_are_materialized(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        tier_id="ROUTE_GENERALIZATION",
        routes=["R_STRAIGHT", "R_VERT_A2_C3", "R_LAT_A5_C5", "R_3D_L"],
        pressures=["nominal", "current", "noise", "plant"],
    )
    payloads = {
        (scenario["axes"]["pressure"], scenario["axes"]["route_profile"]): scenario["payload"]
        for scenario in manifest["scenarios"]
    }
    straight = payloads[("nominal", "R_STRAIGHT")]["reference"]
    assert straight["lateral_amplitude_m"] == 0.0
    assert straight["depth_amplitude_m"] == 0.0
    assert straight["depth_slope_m"] == 0.0
    assert straight["length_m"] == 70.0

    vertical = payloads[("nominal", "R_VERT_A2_C3")]["reference"]
    assert vertical["lateral_amplitude_m"] == 0.0
    assert vertical["depth_amplitude_m"] == 2.0
    assert vertical["depth_cycles"] == 3.0

    lateral = payloads[("nominal", "R_LAT_A5_C5")]["reference"]
    assert lateral["lateral_amplitude_m"] == 5.0
    assert lateral["lateral_cycles"] == 5.0
    assert lateral["depth_amplitude_m"] == 0.0

    three_d = payloads[("nominal", "R_3D_L")]["reference"]
    assert three_d["lateral_amplitude_m"] == 5.0
    assert three_d["lateral_cycles"] == 5.0
    assert three_d["depth_amplitude_m"] == 5.0
    assert three_d["depth_cycles"] == 5.0

    current = payloads[("current", "R_STRAIGHT")]
    assert current["current"]["speed_mps"] == [0.3, 0.55]
    noise = payloads[("noise", "R_STRAIGHT")]
    assert noise["sensors"]["noise_scale_range"] == [0.5, 1.0]
    plant = payloads[("plant", "R_STRAIGHT")]
    assert plant["plant_randomization"]["enabled"] is True
    assert plant["plant_randomization"]["seed"] == 13420


def test_generated_route_config_parses_and_sets_c6_eval_contract(tmp_path: Path) -> None:
    builder = _load()
    manifest = _manifest(
        tmp_path,
        tier_id="ROUTE_GENERALIZATION",
        routes=["R_3D_M"],
        pressures=["current"],
    )
    builder.write_manifest(manifest, output_dir=tmp_path, overwrite=False)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    scenario = saved["scenarios"][0]
    config_path = ROOT / scenario["config"]
    config = load_config(config_path)
    assert config.reference.length_m == 70.0
    assert config.reference.depth_amplitude_m == 2.0
    assert config.reference.lateral_amplitude_m == 2.0
    assert config.plant.goal_radius_m == 2.0
    assert config.curriculum.enabled is False


def test_rejects_unregistered_route_generalization_route(tmp_path: Path) -> None:
    builder = _load()
    matrix, base, matrix_path, base_path = _inputs()
    try:
        builder.build_manifest(
            matrix=matrix,
            base_env=base,
            tier_id="ROUTE_GENERALIZATION",
            matrix_path=matrix_path,
            base_env_path=base_path,
            output_dir=tmp_path,
            routes=["R_NOT_A_ROUTE"],
        )
    except ValueError as error:
        assert "unsupported route" in str(error)
    else:
        raise AssertionError("unregistered route must not be materialized")
