from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from auv_lowlevel.config import load_config
from scripts.evaluate_pc_ppo import _validate_controller_pitch_override


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_goal147_profiles_change_only_declared_section_fields() -> None:
    manifest = json.loads(
        (REPO_ROOT / "configs/lowlevel/goal147/profile_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    base = json.loads(
        (REPO_ROOT / manifest["base_config"]).read_text(encoding="utf-8")
    )
    for name, entry in manifest["arms"].items():
        for section, key in (("guidance", "guidance"), ("base_controller", "controller")):
            profile = entry[key]
            if profile is None:
                continue
            payload = json.loads(
                (REPO_ROOT / profile["path"]).read_text(encoding="utf-8")
            )[section]
            changed = {
                field: value
                for field, value in payload.items()
                if base[section].get(field) != value
            }
            assert changed == profile["changes"], (name, key)


def test_goal147_slowdown_profiles_are_valid() -> None:
    b85 = load_config(REPO_ROOT / "configs/lowlevel/goal147/guidance_b85.json")
    b75 = load_config(REPO_ROOT / "configs/lowlevel/goal147/guidance_b75.json")
    assert b85.guidance.terminal_braking_enabled
    assert b85.guidance.terminal_speed_ratio == pytest.approx(0.85)
    assert b75.guidance.terminal_braking_enabled
    assert b75.guidance.terminal_speed_ratio == pytest.approx(0.75)


def test_controller_override_allows_only_pitch_gains() -> None:
    base = load_config(REPO_ROOT / "configs/lowlevel/goal136/curriculum/C0.json").controller
    k4 = load_config(REPO_ROOT / "configs/lowlevel/goal147/controller_k4.json").controller
    r2 = load_config(REPO_ROOT / "configs/lowlevel/goal147/controller_r2.json").controller

    _validate_controller_pitch_override(base, k4)
    _validate_controller_pitch_override(base, r2)

    with pytest.raises(ValueError, match="pitch_gain and pitch_rate_gain"):
        _validate_controller_pitch_override(
            base, replace(k4, heading_gain=base.heading_gain + 1.0)
        )
