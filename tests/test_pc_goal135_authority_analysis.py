from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_pc_goal135_authority import classify_arm

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    REPO_ROOT
    / "configs/lowlevel/goal135/pc_goal135_authority_sac_base_20261111.json"
)


def test_goal135_base_config_freezes_architecture_and_screen_budget() -> None:
    payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "auv_pc_goal135_authority_sac.v1"
    assert payload["policy"]["bc_path_dim"] == 8
    assert payload["observation"]["path_dim"] == 12
    assert payload["training"]["total_env_transitions"] == 1_000_192
    assert payload["training"]["num_envs"] == 256
    assert payload["training"]["batch_size"] == 256
    assert payload["training"]["authority_regularization"]["form"] == "hinge_rms"
    assert payload["base_residual_bias"] == [0.0, -0.25, 0.0]
    assert payload["residual_terminal_gate_enabled"] is True


def test_goal135_classifies_cross_seed_gain() -> None:
    status, _ = classify_arm(
        nominal_strict=26.0,
        nominal_default=33.0,
        current15_strict=16.0,
        current15_default=24.0,
        nominal_ok_seeds=4,
        current15_ok_seeds=4,
        authority_rms=0.22,
        timeout=0,
        safety=0,
    )
    assert status == "AUTHORITY_SAC_GAIN"


def test_goal135_classifies_nominal_only_and_single_seed() -> None:
    nominal_only, _ = classify_arm(
        nominal_strict=26.0,
        nominal_default=33.0,
        current15_strict=14.5,
        current15_default=24.0,
        nominal_ok_seeds=4,
        current15_ok_seeds=2,
        authority_rms=0.22,
        timeout=0,
        safety=0,
    )
    assert nominal_only == "NOMINAL_ONLY_GAIN"
    single, _ = classify_arm(
        nominal_strict=25.0,
        nominal_default=32.5,
        current15_strict=15.0,
        current15_default=24.0,
        nominal_ok_seeds=1,
        current15_ok_seeds=0,
        authority_rms=0.25,
        timeout=0,
        safety=0,
    )
    assert single == "SINGLE_SEED_ONLY"


def test_goal135_classifies_authority_collapse_and_harm() -> None:
    collapse, _ = classify_arm(
        nominal_strict=24.5,
        nominal_default=33.0,
        current15_strict=14.5,
        current15_default=24.0,
        nominal_ok_seeds=0,
        current15_ok_seeds=0,
        authority_rms=0.18,
        timeout=0,
        safety=0,
    )
    assert collapse == "DIRECTION_SIGNAL_INSUFFICIENT"
    harm, _ = classify_arm(
        nominal_strict=24.0,
        nominal_default=33.0,
        current15_strict=16.0,
        current15_default=24.0,
        nominal_ok_seeds=0,
        current15_ok_seeds=4,
        authority_rms=0.2,
        timeout=1,
        safety=0,
    )
    assert harm == "AUTHORITY_SAC_HARM"
