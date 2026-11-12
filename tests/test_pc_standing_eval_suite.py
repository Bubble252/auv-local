from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_pc_standing_eval_suite.py"
)
SPEC = importlib.util.spec_from_file_location("standing_eval_validator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def test_canonical_suite_is_valid() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    assert validator.validate(payload) == []


def test_suite_rejects_pnav_axis_and_open_heldout() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    payload["success_contract"]["p_nav_used_as_performance_axis"] = True
    payload["heldout_seeds"]["status"] = "open"
    errors = validator.validate(payload)
    assert any("p_nav_used_as_performance_axis" in error for error in errors)
    assert any("heldout_seeds.status" in error for error in errors)


def test_suite_freezes_exactly_two_alos_current_modes() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    modes = {item["id"]: item for item in payload["alos_current_modes"]}
    assert set(modes) == {"ALOS-NONE", "ALOS-COMP"}
    assert modes["ALOS-NONE"]["compensate_current"] is False
    assert modes["ALOS-NONE"]["current_crab_enabled"] is False
    assert modes["ALOS-COMP"]["compensate_current"] is True
    assert modes["ALOS-COMP"]["current_crab_enabled"] is False
    assert modes["ALOS-COMP"]["current_compensation_source"] == "estimated"

    modes["ALOS-CRAB"] = {
        "id": "ALOS-CRAB",
        "compensate_current": False,
        "current_crab_enabled": True,
        "current_compensation_source": "none",
    }
    payload["alos_current_modes"] = list(modes.values())
    errors = validator.validate(payload)
    assert any("exactly" in error and "ALOS-NONE" in error for error in errors)


def test_suite_requires_t0_comp_and_none_control() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    paradigms = {item["id"]: item for item in payload["paradigms"]}
    assert paradigms["T0"]["alos_current_mode"] == "ALOS-COMP"
    assert paradigms["T0-NONE"]["alos_current_mode"] == "ALOS-NONE"

    paradigms["T0"]["alos_current_mode"] = "ALOS-NONE"
    errors = validator.validate(payload)
    assert any("T0 must use ALOS-COMP" in error for error in errors)


def test_suite_requires_remus_speed_ceiling_in_curriculum_and_eval() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    assert payload["speed_max_mps"] == 2.5
    assert 2.5 in payload["speed_profiles_mps"]
    assert payload["speed_curriculum"]["stages_mps"]["C6"][1] == 2.5
    for tier in ("B_robustness", "C_full_geometry", "D_formal"):
        tier_payload = next(item for item in payload["tiers"] if item["id"] == tier)
        assert 2.5 in tier_payload["speed_profiles_mps"]


def test_suite_rejects_nonmonotonic_or_truncated_speed_curriculum() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    payload["speed_curriculum"]["stages_mps"]["C5"][1] = 2.6
    payload["speed_curriculum"]["stages_mps"]["C6"][1] = 2.4
    errors = validator.validate(payload)
    assert any("non-decreasing" in error for error in errors)
    assert any("cover the full" in error for error in errors)


def test_suite_rejects_speed_above_remus_envelope() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    payload["speed_curriculum"]["stages_mps"]["C6"][1] = 2.6
    errors = validator.validate(payload)
    assert any("REMUS100 envelope" in error for error in errors)

    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    payload["speed_profiles_mps"].append(2.6)
    errors = validator.validate(payload)
    assert any("speed_profiles_mps" in error and "REMUS100" in error for error in errors)


def test_suite_rejects_high_speed_tier_that_omits_25() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    tier = next(item for item in payload["tiers"] if item["id"] == "D_formal")
    tier["speed_profiles_mps"] = [0.9, 1.2, 1.4]
    errors = validator.validate(payload)
    assert any("D_formal" in error and "2.5" in error for error in errors)


def test_suite_requires_shared_final_stage_for_curriculum_order() -> None:
    payload = json.loads(validator.DEFAULT_SUITE.read_text(encoding="utf-8"))
    hard = next(item for item in payload["curricula"] if item["id"] == "C-hard")
    hard["final_stage"] = "C5"
    hard["eval_stage"] = "C5"
    errors = validator.validate(payload)
    assert any("same final_stage" in error for error in errors)


def test_validator_cli_reports_ok(capsys) -> None:
    assert validator.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["paradigm_count"] >= 6
