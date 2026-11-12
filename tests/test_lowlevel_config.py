from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from auv_lowlevel.config import load_config, parse_config
from auv_lowlevel.ppo_config import load_training_config

BASE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "lowlevel" / "odarrl_control_base.json"
)
V2_CONFIG = BASE_CONFIG.parent / "odarrl_control_v2_26d.json"
CONFIG_DIR = BASE_CONFIG.parent
REPO_ROOT = BASE_CONFIG.parents[2]


def _raw_config() -> dict:
    return json.loads(BASE_CONFIG.read_text(encoding="utf-8"))


def test_base_config_loads_with_frozen_contract_values() -> None:
    config = load_config(BASE_CONFIG)

    assert config.schema_version == "auv_odarrl_control.v1"
    assert config.arm == "B0"
    assert config.plant.dt_s == pytest.approx(0.02)
    assert config.plant.control_hz == pytest.approx(50.0)
    assert config.reference.speed_mps == pytest.approx(1.2)
    assert config.reference.length_m == pytest.approx(35.0)
    assert config.observation.dim == 27
    assert config.evaluation.seed_count == 50
    assert config.config_sha256


def test_config_hash_is_canonical_and_content_sensitive(tmp_path: Path) -> None:
    first = load_config(BASE_CONFIG)
    second = load_config(BASE_CONFIG)
    changed = _raw_config()
    changed["reward"]["w_action"] = 0.051
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")

    assert first.config_sha256 == second.config_sha256
    assert parse_config(changed).config_sha256 == ""
    assert load_config(changed_path).config_sha256 != first.config_sha256


def test_config_rejects_unknown_and_missing_fields() -> None:
    unknown = _raw_config()
    unknown["reward"]["unexpected"] = 1.0
    with pytest.raises(ValueError, match="unknown keys"):
        parse_config(unknown)

    missing = _raw_config()
    del missing["plant"]["dt_s"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_config(missing)


def test_config_accepts_26d_and_rejects_invalid_observation_dimension() -> None:
    v2 = copy.deepcopy(_raw_config())
    v2["observation"]["dim"] = 26
    assert parse_config(v2).observation.dim == 26

    wrong_dim = copy.deepcopy(_raw_config())
    wrong_dim["observation"]["dim"] = 25
    with pytest.raises(ValueError, match="observation.dim"):
        parse_config(wrong_dim)

    overlap = copy.deepcopy(_raw_config())
    overlap["evaluation"]["training_seed_start"] = 49
    with pytest.raises(ValueError, match="training seed start"):
        parse_config(overlap)


def test_guidance_command_rate_limit_switch_is_optional_and_round_trips() -> None:
    legacy = _raw_config()
    legacy["guidance"].pop("command_rate_limit_enabled", None)
    assert parse_config(legacy).guidance.command_rate_limit_enabled is None

    for value in (True, False, None):
        raw = _raw_config()
        raw["guidance"]["command_rate_limit_enabled"] = value
        assert parse_config(raw).guidance.command_rate_limit_enabled is value

    invalid = _raw_config()
    invalid["guidance"]["command_rate_limit_enabled"] = "true"
    with pytest.raises(ValueError, match="command_rate_limit_enabled"):
        parse_config(invalid)


def test_observation_contract_defaults_preserve_legacy_and_oracle_configs() -> None:
    oracle = parse_config(_raw_config())
    assert oracle.observation.current_contract == "oracle"
    assert oracle.observation.speed_feedback == "water_relative"

    hidden = copy.deepcopy(_raw_config())
    hidden["observation"]["expose_current"] = False
    hidden["guidance"]["compensate_current"] = False
    hidden["base_controller"]["use_current_field"] = False
    assert parse_config(hidden).observation.current_contract == "legacy"


def test_hidden_contract_requires_ground_speed_feedback() -> None:
    hidden = copy.deepcopy(_raw_config())
    hidden["observation"].update(
        {
            "expose_current": False,
            "current_contract": "hidden",
            "speed_feedback": "ground",
        }
    )
    hidden["guidance"]["compensate_current"] = False
    hidden["base_controller"]["use_current_field"] = False
    parsed = parse_config(hidden)
    assert parsed.observation.current_contract == "hidden"
    assert parsed.observation.speed_feedback == "ground"

    leaked = copy.deepcopy(hidden)
    leaked["observation"]["speed_feedback"] = "water_relative"
    with pytest.raises(ValueError, match="speed_feedback='ground'"):
        parse_config(leaked)

    exposed = copy.deepcopy(hidden)
    exposed["observation"]["expose_current"] = True
    with pytest.raises(ValueError, match="expose_current=false"):
        parse_config(exposed)


@pytest.mark.parametrize(
    "config_path",
    sorted(CONFIG_DIR.glob("*hidden*.json")),
    ids=lambda path: path.name,
)
def test_hidden_named_configs_use_the_strict_hidden_contract(config_path: Path) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    environment_path = config_path
    if "environment" in raw:
        environment_path = REPO_ROOT / load_training_config(config_path).environment_config

    config = load_config(environment_path)
    assert config.observation.current_contract == "hidden"
    assert config.observation.expose_current is False
    assert config.observation.speed_feedback == "ground"


def test_oracle_contract_requires_exposed_current() -> None:
    oracle = copy.deepcopy(_raw_config())
    oracle["observation"]["current_contract"] = "oracle"
    oracle["observation"]["expose_current"] = False
    with pytest.raises(ValueError, match="requires observation.expose_current=true"):
        parse_config(oracle)

    legacy = copy.deepcopy(_raw_config())
    legacy["observation"]["current_contract"] = "legacy"
    with pytest.raises(ValueError, match="requires current_contract='oracle'"):
        parse_config(legacy)


def test_canonical_v2_config_matches_v1_except_observation_dimension() -> None:
    v1_raw = _raw_config()
    v2_raw = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    v1_raw["observation"]["dim"] = 26

    assert v2_raw == v1_raw
    config = load_config(V2_CONFIG)
    assert config.observation.dim == 26
    assert config.config_sha256 == (
        "c19a5867ae298aaf373339e172be3d78e492c0b62c821d303e2ca2b60e0f72e7"
    )


def test_config_rejects_inconsistent_control_rate_and_nonpositive_values() -> None:
    rate = copy.deepcopy(_raw_config())
    rate["plant"]["control_hz"] = 49.0
    with pytest.raises(ValueError, match="control_hz"):
        parse_config(rate)

    speed = copy.deepcopy(_raw_config())
    speed["reference"]["speed_mps"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        parse_config(speed)


def test_reference_path_bank_defaults_and_validation() -> None:
    default = parse_config(_raw_config())
    assert default.reference.path_bank_size == 1
    assert default.reference.path_bank_seed == 0

    banked = copy.deepcopy(_raw_config())
    banked["reference"]["path_bank_size"] = 8
    banked["reference"]["path_bank_seed"] = 37
    parsed = parse_config(banked)
    assert parsed.reference.path_bank_size == 8
    assert parsed.reference.path_bank_seed == 37

    invalid_size = copy.deepcopy(banked)
    invalid_size["reference"]["path_bank_size"] = 0
    with pytest.raises(ValueError, match="path_bank_size"):
        parse_config(invalid_size)

    invalid_seed = copy.deepcopy(banked)
    invalid_seed["reference"]["path_bank_seed"] = -1
    with pytest.raises(ValueError, match="path_bank_seed"):
        parse_config(invalid_seed)


def test_config_rejects_hidden_current_leak_for_non_oracle_arms() -> None:
    leaked = _raw_config()
    leaked["observation"]["expose_current"] = False
    leaked["guidance"]["compensate_current"] = True
    leaked["base_controller"]["use_current_field"] = False
    with pytest.raises(ValueError, match="hidden current leak"):
        parse_config(leaked)

    leaked["guidance"]["compensate_current"] = False
    leaked["base_controller"]["use_current_field"] = True
    with pytest.raises(ValueError, match="hidden current leak"):
        parse_config(leaked)

    clean = copy.deepcopy(leaked)
    clean["base_controller"]["use_current_field"] = False
    assert parse_config(clean).observation.expose_current is False


def test_all_shipped_lowlevel_configs_load() -> None:
    configs = sorted(
        path
        for path in CONFIG_DIR.glob("odarrl_control_*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("schema_version")
        == "auv_odarrl_control.v1"
    )
    assert configs
    for path in configs:
        config = load_config(path)
        assert config.config_sha256
        assert config.observation.dim in {26, 27}


def test_goal22_nominal_and_ideal_bc_config_matrix() -> None:
    for family, ideal_actor in (("nominal", False), ("ideal", True)):
        for path_dim in (8, 12):
            for scenario in ("default", "strict125"):
                path = CONFIG_DIR / (
                    f"odarrl_control_pc_c3_random_realistic_v1_{family}_bc_"
                    f"n{path_dim}_{scenario}_20261107.json"
                )
                config = load_config(path)
                assert config.observation.ideal_actor_observations is ideal_actor
                assert config.observation.path_endpoint_context_enabled is (path_dim == 12)
                assert config.termination.pc_goal_plane_miss_termination_enabled is True
                assert config.plant.goal_radius_m == (1.25 if scenario == "strict125" else 2.0)
