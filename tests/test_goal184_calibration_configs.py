from __future__ import annotations

import hashlib
import json
from pathlib import Path

from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import CurriculumStageSampler, default_curriculum_stages
from auv_lowlevel.ppo_config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = (
    ROOT
    / "configs"
    / "lowlevel"
    / "goal184"
    / "calibration"
    / "stage_matched_v4_teacher_alos_truth_100it_fixedlr"
)
STAGES = tuple(f"C{index}" for index in range(7))
WARMSTART_SHA256 = "86cab3fef9ee5784c0d3ef9441b1a409636df45204902713bb2fc950ebb1c086"


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_goal184_stage_matched_calibration_cover_all_stages() -> None:
    manifest = _payload(CALIBRATION / "manifest.json")
    assert manifest["stages"] == list(STAGES)
    assert manifest["sensor_profile"] == "unbiased_small_noise_v3"
    assert manifest["normalization_mode"] == "empirical"
    assert manifest["noise_std"] == 0.05
    assert manifest["learning_rate"] == 1e-12
    assert manifest["schedule"] == "fixed"
    assert manifest["iterations"] == 100
    assert manifest["warmstart_actor_sha256"] == WARMSTART_SHA256
    for group in ("env", "train"):
        assert set(manifest["files"][group]) == set(STAGES)
        for relative in manifest["files"][group].values():
            assert manifest["files_sha256"][relative] == _sha256(ROOT / relative)


def test_goal184_stage_matched_calibration_envs_are_single_stage_v3() -> None:
    for stage in STAGES:
        path = CALIBRATION / f"env_{stage}.json"
        payload = _payload(path)
        assert payload["normalization"]["mode"] == "empirical"
        assert payload["sensors"]["profile"] == "unbiased_small_noise_v3"
        assert payload["plant"]["goal_radius_m"] == 2.0
        assert payload["plant"]["max_episode_steps"] == 3000
        assert payload["curriculum"]["initial_stage"] == stage
        assert payload["curriculum"]["promotion_enabled"] is False
        assert payload["curriculum"]["stage_mix"] == [{"stage": stage, "weight": 1.0}]
        environment = load_config(path)
        assert environment.normalization.mode == "empirical"
        assert environment.curriculum.initial_stage == stage
        assert environment.curriculum.promotion_enabled is False


def test_goal184_stage_matched_calibration_train_configs_are_negligible_updates() -> None:
    for stage in STAGES:
        path = CALIBRATION / f"train_{stage}.json"
        payload = _payload(path)
        algorithm = payload["algorithm"]
        assert algorithm["learning_rate"] == 1e-12
        assert algorithm["schedule"] == "fixed"
        assert "learning_rate_actor" not in algorithm
        assert "learning_rate_critic" not in algorithm
        assert "learning_rate_dade_head" not in algorithm
        assert payload["policy"]["init_noise_std"] == 0.05
        assert payload["training"]["iterations"] == 100
        assert payload["runner"]["save_interval"] == 100
        assert payload["runner"]["empirical_normalization"] is False
        assert payload["experiment"]["warmstart_actor_sha256"] == WARMSTART_SHA256
        assert payload["environment"]["config"] == (
            f"configs/lowlevel/goal184/calibration/"
            f"stage_matched_v4_teacher_alos_truth_100it_fixedlr/env_{stage}.json"
        )
        training = load_training_config(path)
        assert training.iterations == 100
        assert training.training_profile == "PPO-FLAT-4096x48-E5-MB4"
        assert training.runner["empirical_normalization"] is False


def test_goal184_stage_matched_calibration_envs_sample_their_stage_ranges() -> None:
    expected = default_curriculum_stages()
    for stage in STAGES:
        environment = load_config(CALIBRATION / f"env_{stage}.json")
        sampler = CurriculumStageSampler(
            stage_mix=[(str(item[0]), float(item[1])) for item in environment.curriculum.stage_mix],
            speed_protocol=environment.curriculum.speed_protocol,
        )
        specs = [sampler.sample(91_000 + index) for index in range(100)]
        assert {spec.stage for spec in specs} == {stage}
        depth_range = expected[stage].depth_range_m
        noise_range = expected[stage].noise_scale_range
        current_range = expected[stage].current_speed_range_mps
        assert all(depth_range[0] <= spec.depth_m <= depth_range[1] for spec in specs)
        assert all(noise_range[0] <= spec.noise_scale <= noise_range[1] for spec in specs)
        assert all(
            current_range[0] <= spec.current_speed_mps <= current_range[1]
            for spec in specs
        )


def test_goal184_stage_matched_calibration_outage_is_only_on_c5_c6() -> None:
    expected = default_curriculum_stages()
    for stage in STAGES:
        environment = load_config(CALIBRATION / f"env_{stage}.json")
        sampler = CurriculumStageSampler(
            stage_mix=[(str(item[0]), float(item[1])) for item in environment.curriculum.stage_mix],
            speed_protocol=environment.curriculum.speed_protocol,
        )
        specs = [sampler.sample(92_000 + index) for index in range(100)]
        hazard_range = expected[stage].current_outage_start_hazard_per_s
        duration_range = expected[stage].current_outage_duration_s
        assert all(
            hazard_range[0] <= spec.current_outage_start_hazard_per_s <= hazard_range[1]
            for spec in specs
        )
        assert all(
            duration_range[0] <= spec.current_outage_duration_s <= duration_range[1]
            for spec in specs
        )
        if stage in {"C0", "C1", "C2", "C3", "C4"}:
            assert all(spec.current_outage_start_hazard_per_s == 0.0 for spec in specs)
            assert all(spec.current_outage_duration_s == 0.0 for spec in specs)
