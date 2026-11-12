from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_pc_standing_eval_configs.py"
SPEC = importlib.util.spec_from_file_location("standing_eval_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _inputs() -> tuple[dict, dict, Path, Path]:
    suite_path = builder.DEFAULT_SUITE
    base_path = builder.DEFAULT_BASE_ENV
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    return suite, base, suite_path, base_path


def _manifest(tmp_path: Path, **kwargs) -> dict:
    suite, base, suite_path, base_path = _inputs()
    defaults = {
        "suite": suite,
        "base_env": base,
        "tier_id": "A_screen",
        "suite_path": suite_path,
        "base_env_path": base_path,
        "output_dir": tmp_path,
        "radii": ["default", "strict125"],
    }
    defaults.update(kwargs)
    return builder.build_manifest(**defaults)


def test_a_screen_expands_pressure_and_both_radii(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest["scenario_count"] == 8
    assert manifest["axes"]["speed_profiles_mps"] == [1.2]
    assert manifest["speed_max_mps"] == 2.5
    ids = {item["id"] for item in manifest["scenarios"]}
    assert "p-nominal__r-legacy_sine__l35__v1p2__g-default" in ids
    assert "p-plant20__r-legacy_sine__l35__v1p2__g-strict125" in ids


def test_pressure_and_geometry_are_applied_explicitly(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        tier_id="D_formal",
        pressure_ids=["current10", "plant20", "combined20"],
        routes=["P3"],
        lengths=[45.0],
        speeds=[2.5],
        radii=["strict125"],
    )
    assert manifest["scenario_count"] == 3
    payloads = {item["axes"]["pressure"]: item["payload"] for item in manifest["scenarios"]}

    current10 = payloads["current10"]
    assert current10["current"]["speed_mps"] == [0.3, 0.55]
    assert current10["current"]["shear"]["velocity_gradient_per_s"][0][0] == 0.004
    assert current10["reference"]["path_family"] == "P3"
    assert current10["reference"]["length_m"] == 45.0
    assert current10["reference"]["speed_mps"] == 2.5
    assert current10["plant"]["goal_radius_m"] == 1.25
    assert current10["curriculum"]["enabled"] is False

    plant20 = payloads["plant20"]
    assert plant20["plant_randomization"]["enabled"] is True
    assert plant20["plant_randomization"]["drag_scale"] == [0.8, 1.2]

    combined = payloads["combined20"]
    assert combined["sensors"]["noise_scale_range"] == [0.5, 1.0]
    assert combined["plant_randomization"]["actuator_time_scale"] == [0.8, 1.2]
    assert combined["current"]["speed_mps"] == [0.3, 0.55]


def test_written_manifest_is_self_contained(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, radii=["strict125"])
    builder.write_manifest(manifest, output_dir=tmp_path, overwrite=False)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["scenario_count"] == len(saved["scenarios"])
    for scenario in saved["scenarios"]:
        assert "payload" not in scenario
        config_path = builder.REPO_ROOT / scenario["config"]
        assert config_path.is_file()
        assert scenario["sha256"] == builder._sha256(config_path)


def test_rejects_speed_above_remus_limit(tmp_path: Path) -> None:
    try:
        _manifest(tmp_path, speeds=[2.6])
    except ValueError as error:
        assert "unsupported speed profile" in str(error)
    else:
        raise AssertionError("2.6 m/s must not be accepted")
