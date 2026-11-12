from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.path import BatchedReferencePathBank
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.path import generate_sinusoidal_3d_path_bank
from scripts.build_pc_goal136_dline_configs import (
    ARMS,
    BASE_ENV_CONFIG,
    NOMINAL_ANCHOR,
    SAMPLING_MODE,
    SCREEN_SEEDS,
    SCREEN_TRANSITIONS,
    build_environment_config,
    build_training_config,
)


def test_n0_environment_matches_goal134_nominal_contract() -> None:
    built = build_environment_config("N0")
    base = json.loads(BASE_ENV_CONFIG.read_text(encoding="utf-8"))
    built.pop("experiment")
    base.pop("experiment")
    assert built == base


def test_dline_arms_widen_only_registered_supported_axes() -> None:
    n0 = build_environment_config("N0")
    d1 = build_environment_config("D1")
    d2 = build_environment_config("D2")

    assert n0.get("domain") is None
    assert d1.get("domain") is None
    assert d2.get("domain") is None
    assert n0["reference"]["speed_mps"] == d1["reference"]["speed_mps"]
    assert n0["reference"]["speed_mps"] == d2["reference"]["speed_mps"]

    assert d1["current"]["speed_mps"] == [0.25, 0.55]
    assert d2["current"]["speed_mps"] == [0.25, 0.675]
    assert d1["current"]["time_variation"]["period_s"] == 10.0
    assert d2["current"]["time_variation"]["period_s"] == 6.0
    assert d1["plant_randomization"]["drag_scale"] == [0.9, 1.1]
    assert d2["plant_randomization"]["drag_scale"] == [0.8, 1.2]
    assert d1["reference"]["path_bank_size"] == 48
    assert d2["reference"]["path_bank_size"] == 64


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_dline_environment_configs_load(tmp_path: Path, arm: str) -> None:
    path = tmp_path / f"{arm}.json"
    path.write_text(json.dumps(build_environment_config(arm)), encoding="utf-8")
    config = load_config(path)
    assert config.sensors is not None
    assert config.sensors.profile == "unbiased_noise_v1"
    assert config.plant.goal_radius_m == pytest.approx(1.25)
    assert config.termination.position_source == "eta"
    assert config.observation.current_contract == "hidden"
    assert config.observation.ideal_actor_observations is False
    if arm == "N0":
        assert config.plant_randomization.enabled is False
        assert config.current.max_speed_mps == pytest.approx(0.45)
    else:
        assert config.plant_randomization.enabled is True
        assert config.current.shear is not None


def test_training_configs_match_sac_protocol_and_disable_authority(tmp_path: Path) -> None:
    root = tmp_path
    env_path = root / "env.json"
    env_path.write_text(json.dumps(build_environment_config("D2")), encoding="utf-8")
    for seed in SCREEN_SEEDS:
        payload = build_training_config(
            "D2",
            seed,
            environment_config=env_path,
            artifact_root=root,
        )
        assert payload["training"]["seed"] == seed
        assert payload["training"]["total_env_transitions"] == SCREEN_TRANSITIONS
        assert payload["training"]["authority_regularization"]["enabled"] is False
        assert payload["evaluation"]["residual_scale"] == 2.0
        assert payload["experiment"]["arm"] == "D2"
        assert payload["experiment"]["algorithm"] == "SAC"
        assert str(root / "screen" / "D2" / f"S_s{seed}") in payload["experiment"]["output"]


def test_dline_arm_parameters_are_frozen() -> None:
    assert tuple(ARMS) == ("N0", "D1", "D2")
    assert ARMS["N0"] == {}
    assert ARMS["D1"]["plant_scale"] == (0.9, 1.1)
    assert ARMS["D2"]["plant_scale"] == (0.8, 1.2)


def test_dline_phase_a1_is_joint_wide_without_fake_anchor() -> None:
    assert SAMPLING_MODE == "joint_wide"
    assert NOMINAL_ANCHOR["implemented_in_phase_a1"] is False
    assert NOMINAL_ANCHOR["status"] == "deferred_a2_after_a1"


def test_d2_wide_path_bank_is_selectable(tmp_path: Path) -> None:
    path = tmp_path / "D2.json"
    path.write_text(json.dumps(build_environment_config("D2")), encoding="utf-8")
    config = load_config(path)
    paths = generate_sinusoidal_3d_path_bank(config.reference)
    assert len(paths) == 64
    bank = BatchedReferencePathBank(paths, dtype=torch.float64)
    selected = bank.select(torch.tensor([0, 63], dtype=torch.long))
    assert selected.num_paths == 2

    env = BatchedPCLowLevelEnv(
        config,
        num_envs=2,
        device="cpu",
        dtype=torch.float64,
        validate_finite=True,
        path_dim=12,
    )
    observation = env.reset(seeds=[86201, 86202])
    assert observation.actor_state.shape == (2, 12)
    assert torch.isfinite(observation.actor_state).all()
    assert env._path_bank is not None
    assert env._path_bank.size == 64
