from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.sensors import PROFILES, SyntheticSensorSuite
from auv_lowlevel.config import load_config, parse_config
from auv_lowlevel.curriculum import (
    CurriculumStageSampler,
    EpisodeManifestWriter,
    default_curriculum_stages,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = (
    ROOT / "configs" / "lowlevel" / "pc_goal85_unbiased_default_candidate_20261110.json"
)
FORMAL_ENV = (
    ROOT / "configs" / "lowlevel" / "goal184" / "env_path_fixed_speed_formal_empirical.json"
)


def _sensor_config(profile: str = "unbiased_small_noise_v3"):
    data = json.loads(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    data["sensors"] = {"profile": profile, "seed": 4242}
    return parse_config(data)


def _inputs(num_envs: int, current_x: float):
    eta = torch.zeros((num_envs, 6), dtype=torch.float64)
    nu = torch.zeros((num_envs, 6), dtype=torch.float64)
    current = torch.tensor([[current_x, 0.0, 0.0]] * num_envs, dtype=torch.float64)
    return eta, nu, current


def test_v3_is_v2_with_short_current_filter_tau() -> None:
    v2 = PROFILES["unbiased_small_noise_v2"]
    v3 = PROFILES["unbiased_small_noise_v3"]

    assert v3.name == "unbiased_small_noise_v3"
    assert v2.current_est_tau_s == (0.5, 1.5)
    assert v3.current_est_tau_s == (0.05, 0.5)
    changed = {
        field.name for field in fields(v2) if getattr(v2, field.name) != getattr(v3, field.name)
    }
    assert changed == {"name", "current_est_tau_s"}
    assert v3.current_est_delay_s == (0.0, 0.0)
    assert v3.current_est_outage_start_prob == (0.0, 0.0)
    assert v3.current_est_outage_duration_s == (0.0, 0.0)


def test_v3_profile_is_accepted_by_config_parser() -> None:
    config = _sensor_config()
    assert config.sensors is not None
    assert config.sensors.profile == "unbiased_small_noise_v3"


def test_default_stages_define_outage_only_for_c5_and_c6() -> None:
    stages = default_curriculum_stages()
    for name in ("C0", "C1", "C2", "C3", "C4"):
        assert stages[name].current_outage_start_hazard_per_s == (0.0, 0.0)
        assert stages[name].current_outage_duration_s == (0.0, 0.0)

    assert stages["C5"].current_outage_start_hazard_per_s == (0.002, 0.008)
    assert stages["C5"].current_outage_duration_s == (0.05, 0.3)
    assert stages["C6"].current_outage_start_hazard_per_s == (0.008, 0.020)
    assert stages["C6"].current_outage_duration_s == (0.05, 0.3)
    assert stages["C6"].current_estimator_level == "degraded_outage"

    for name in ("C5", "C6"):
        payload = stages[name].as_dict()
        assert payload["current_outage_start_hazard_per_s"] == list(
            stages[name].current_outage_start_hazard_per_s
        )
        assert payload["current_outage_duration_s"] == list(stages[name].current_outage_duration_s)


def test_stage_rejects_outage_contract_violations() -> None:
    stage = default_curriculum_stages()["C5"]
    with pytest.raises(ValueError, match="must not exceed 0.3 s"):
        replace(stage, current_outage_duration_s=(0.05, 0.31))
    with pytest.raises(ValueError, match="positive outage hazard requires"):
        replace(stage, current_outage_duration_s=(0.0, 0.0))
    with pytest.raises(ValueError, match="positive outage duration requires"):
        replace(stage, current_outage_start_hazard_per_s=(0.0, 0.0))
    with pytest.raises(ValueError, match="non-negative"):
        replace(stage, current_outage_start_hazard_per_s=(-0.001, 0.008))


def test_sampler_draws_outage_within_stage_range_and_is_reproducible(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C5", 1.0),))
    specs = sampler.sample_batch(range(256), stage="C5")
    assert specs == sampler.sample_batch(range(256), stage="C5")
    assert all(0.002 <= spec.current_outage_start_hazard_per_s <= 0.008 for spec in specs)
    assert all(0.05 <= spec.current_outage_duration_s <= 0.3 for spec in specs)
    assert len({spec.current_outage_start_hazard_per_s for spec in specs}) > 1
    assert len({spec.current_outage_duration_s for spec in specs}) > 1

    c0 = sampler.sample(0, stage="C0")
    assert c0.current_outage_start_hazard_per_s == 0.0
    assert c0.current_outage_duration_s == 0.0

    writer = EpisodeManifestWriter(tmp_path / "curriculum_episodes.jsonl")
    writer.append(specs[0])
    row = writer.read()[0]
    assert row["current_outage_start_hazard_per_s"] == specs[0].current_outage_start_hazard_per_s
    assert row["current_outage_duration_s"] == specs[0].current_outage_duration_s


def test_set_current_outage_rows_validates_and_applies() -> None:
    config = _sensor_config()
    suite = SyntheticSensorSuite(config, num_envs=3)
    suite.reset([1, 2, 3])
    mask = torch.tensor([True, False, True])
    suite.set_current_outage_rows(
        mask,
        torch.tensor([0.002, 0.0, 0.008]),
        torch.tensor([0.05, 0.0, 0.3]),
    )
    assert suite._params is not None
    assert torch.allclose(
        suite._params.current_est_outage_start_prob,
        torch.tensor([0.002, 0.0, 0.008], dtype=torch.float64),
    )
    assert torch.allclose(
        suite._params.current_est_outage_duration_s,
        torch.tensor([0.05, 0.0, 0.3], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        suite.set_current_outage_rows(mask, torch.tensor([-1.0, 0.0, 0.0]), torch.ones(3))
    with pytest.raises(ValueError, match="must not exceed 0.3 s"):
        suite.set_current_outage_rows(mask, torch.zeros(3), torch.full((3,), 0.31))
    with pytest.raises(ValueError, match="match the selected rows"):
        suite.set_current_outage_rows(
            mask,
            torch.tensor([0.0, 0.0, 0.0, 0.0]),
            torch.ones(4),
        )


def test_outage_freezes_estimate_and_never_reads_true_current() -> None:
    config = _sensor_config()
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([11, 12])
    suite.set_current_outage_rows(
        torch.ones(2, dtype=torch.bool),
        torch.full((2,), 1.0e6, dtype=torch.float64),
        torch.full((2,), 0.3, dtype=torch.float64),
    )

    # The override above exercises the runtime setter; profile fields stay as
    # configured for provenance. Both current values change every step, but the
    # estimator must remain frozen for the whole forced outage.
    for step in range(50):
        eta, nu, current = _inputs(2, current_x=0.2 + 0.01 * step)
        output = suite.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)
        assert torch.equal(output.current_estimate_valid, torch.zeros(2, dtype=torch.float64))
        assert torch.equal(
            output.current_estimate_body,
            torch.zeros((2, 3), dtype=torch.float64),
        )


def test_outage_recovery_warm_starts_from_post_outage_current() -> None:
    config = _sensor_config()
    suite = SyntheticSensorSuite(config, num_envs=2)
    suite.reset([21, 22])
    suite.set_current_outage_rows(
        torch.ones(2, dtype=torch.bool),
        torch.full((2,), 1.0e6, dtype=torch.float64),
        torch.full((2,), 0.02, dtype=torch.float64),
    )
    eta, nu, current = _inputs(2, current_x=0.2)
    first = suite.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)
    assert torch.equal(first.current_estimate_valid, torch.zeros(2, dtype=torch.float64))
    # Disable the hazard after the forced outage starts so recovery is observable.
    suite.set_current_outage_rows(
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )

    eta, nu, current = _inputs(2, current_x=0.8)
    output = first
    for _ in range(20):
        output = suite.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)
    assert torch.equal(output.current_estimate_valid, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(
        output.current_estimate_body[:, 0],
        torch.full((2,), 0.8, dtype=torch.float64),
        atol=0.02,
        rtol=0.0,
    )


def test_same_seed_produces_identical_outage_sequence() -> None:
    config = _sensor_config()
    first = SyntheticSensorSuite(config, num_envs=3)
    second = SyntheticSensorSuite(config, num_envs=3)
    for suite in (first, second):
        suite.reset([31, 32, 33])
        suite.set_current_outage_rows(
            torch.ones(3, dtype=torch.bool),
            torch.full((3,), 0.5, dtype=torch.float64),
            torch.full((3,), 0.2, dtype=torch.float64),
        )
    for step in range(40):
        eta, nu, current = _inputs(3, current_x=0.1 + 0.01 * step)
        left = first.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)
        right = second.step(eta_true=eta, nu_true=nu, current_true_ned_mps=current)
        assert torch.equal(left.current_estimate_valid, right.current_estimate_valid)
        assert torch.equal(left.current_estimate_body, right.current_estimate_body)


def test_env_reset_wires_stage_outage_parameters() -> None:
    config = load_config(FORMAL_ENV)
    assert config.sensors is not None
    assert config.sensors.profile == "unbiased_small_noise_v3"

    for stage, expected_zero in (("C0", True), ("C5", False), ("C6", False)):
        sampler = CurriculumStageSampler(stage_mix=((stage, 1.0),))
        env = BatchedPCLowLevelEnv(
            config,
            num_envs=2,
            dtype=torch.float64,
            curriculum_sampler=sampler,
        )
        env.reset([101, 102])
        assert env.curriculum_specs is not None
        specs = env.curriculum_specs
        assert all(spec.stage == stage for spec in specs)
        assert env.sensors._params is not None
        hazards = env.sensors._params.current_est_outage_start_prob
        durations = env.sensors._params.current_est_outage_duration_s
        if expected_zero:
            assert torch.equal(hazards, torch.zeros(2, dtype=torch.float64))
            assert torch.equal(durations, torch.zeros(2, dtype=torch.float64))
        else:
            stage_spec = default_curriculum_stages()[stage]
            for index, spec in enumerate(specs):
                assert stage_spec.current_outage_start_hazard_per_s[0] <= (
                    spec.current_outage_start_hazard_per_s
                    <= stage_spec.current_outage_start_hazard_per_s[1]
                )
                assert hazards[index].item() == pytest.approx(
                    spec.current_outage_start_hazard_per_s
                )
                assert durations[index].item() == pytest.approx(spec.current_outage_duration_s)
