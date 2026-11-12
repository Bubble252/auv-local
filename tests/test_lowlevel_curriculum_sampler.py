from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from auv_lowlevel.config import REMUS100_MAX_SPEED_MPS
from auv_lowlevel.curriculum import (
    CURRICULUM_EPISODE_MANIFEST_SCHEMA,
    CURRICULUM_STAGE_ORDER,
    SPEED_CURRICULUM_V2_RANGES_MPS,
    CurriculumDecision,
    CurriculumMixRamp,
    CurriculumPromotionController,
    CurriculumStageSampler,
    EpisodeManifestWriter,
    StageEvaluation,
    curriculum_stage_mix_target,
    curriculum_stratum,
    default_curriculum_stages,
    high_speed_curriculum_stages,
)


def _passing_evaluation(default: float = 0.92) -> StageEvaluation:
    return StageEvaluation(
        default_success=default,
        strict_success=0.75,
        timeout_rate=0.02,
        plane_miss_rate=0.02,
        safety_rate=0.0,
        anchor_default_success=0.90,
        stratified_default_success={"depth_low": 0.90, "depth_high": 0.84},
    )


def test_default_stage_protocol_covers_c0_to_c6_in_order() -> None:
    stages = default_curriculum_stages()
    assert tuple(stages) == CURRICULUM_STAGE_ORDER
    assert stages["C0"].noise_scale_range == (0.0, 0.0)
    assert stages["C4"].actor_current_encoder is True
    assert stages["C6"].depth_range_m == (400.0, 500.0)
    assert stages["C6"].current_period_range_s == (4.0, 8.0)


def test_curriculum_speed_ranges_stay_inside_remus_envelope() -> None:
    stages = default_curriculum_stages()
    assert max(stage.speed_range_mps[1] for stage in stages.values()) <= (
        REMUS100_MAX_SPEED_MPS
    )
    with pytest.raises(ValueError, match="REMUS100 speed envelope"):
        replace(stages["C6"], speed_range_mps=(1.2, REMUS100_MAX_SPEED_MPS + 0.01))


def test_high_speed_curriculum_v2_is_explicit_and_reaches_remus_ceiling() -> None:
    v1 = default_curriculum_stages()
    v2 = high_speed_curriculum_stages()

    assert v1["C6"].speed_range_mps == (1.20, 1.40)
    assert v2["C6"].speed_range_mps == (0.90, REMUS100_MAX_SPEED_MPS)
    assert SPEED_CURRICULUM_V2_RANGES_MPS["C6"][1] == REMUS100_MAX_SPEED_MPS
    assert tuple(v2) == CURRICULUM_STAGE_ORDER

    upper_bounds = [v2[stage].speed_range_mps[1] for stage in CURRICULUM_STAGE_ORDER]
    assert upper_bounds == sorted(upper_bounds)
    assert all(upper <= REMUS100_MAX_SPEED_MPS for upper in upper_bounds)
    for stage in CURRICULUM_STAGE_ORDER:
        assert v2[stage].depth_range_m == v1[stage].depth_range_m
        assert v2[stage].current_speed_range_mps == v1[stage].current_speed_range_mps


def test_high_speed_curriculum_v2_sampler_respects_joint_ground_speed_cap() -> None:
    sampler = CurriculumStageSampler(
        stages=high_speed_curriculum_stages(),
        stage_mix=(("C6", 1.0),),
        current_variation_speed_amplitude_mps=0.10,
    )
    specs = sampler.sample_batch(range(2048), stage="C6")
    assert all(spec.speed_mps <= REMUS100_MAX_SPEED_MPS for spec in specs)
    assert all(
        spec.speed_mps + spec.current_speed_mps + 0.10
        <= REMUS100_MAX_SPEED_MPS + 1e-12
        for spec in specs
    )
    assert max(spec.speed_mps for spec in specs) >= 2.30


def test_high_speed_curriculum_v2_all_stages_respect_joint_ground_speed_cap() -> None:
    """REMUS100's 2.5 m/s ground-speed cap holds for every v2 stage."""

    variation = 0.20
    sampler = CurriculumStageSampler(
        stages=high_speed_curriculum_stages(),
        stage_mix=tuple((stage, 1.0) for stage in CURRICULUM_STAGE_ORDER),
        current_variation_speed_amplitude_mps=variation,
    )
    for stage in CURRICULUM_STAGE_ORDER:
        specs = sampler.sample_batch(range(4096), stage=stage)
        upper = SPEED_CURRICULUM_V2_RANGES_MPS[stage][1]
        assert max(spec.speed_mps for spec in specs) <= upper + 1e-12
        assert max(spec.speed_mps for spec in specs) <= (
            REMUS100_MAX_SPEED_MPS - variation + 1e-12
        )
        assert all(
            spec.speed_mps + spec.current_speed_mps + variation
            <= REMUS100_MAX_SPEED_MPS + 1e-12
            for spec in specs
        )


def test_sampler_couples_speed_and_current_inside_remus_envelope() -> None:
    stages = default_curriculum_stages()
    stages["C6"] = replace(
        stages["C6"],
        speed_range_mps=(2.40, REMUS100_MAX_SPEED_MPS),
        current_speed_range_mps=(0.40, 0.675),
    )
    sampler = CurriculumStageSampler(
        stages=stages,
        stage_mix=(("C6", 1.0),),
        current_variation_speed_amplitude_mps=0.1,
    )
    specs = sampler.sample_batch(range(256), stage="C6")
    assert all(spec.speed_mps <= REMUS100_MAX_SPEED_MPS for spec in specs)
    assert all(spec.speed_mps >= 2.40 for spec in specs)
    assert all(
        spec.speed_mps + spec.current_speed_mps + 0.1
        <= REMUS100_MAX_SPEED_MPS + 1e-12
        for spec in specs
    )


def test_max_speed_endpoint_requires_zero_current_and_variation() -> None:
    stages = default_curriculum_stages()
    stages["C6"] = replace(
        stages["C6"],
        speed_range_mps=(REMUS100_MAX_SPEED_MPS, REMUS100_MAX_SPEED_MPS),
        current_speed_range_mps=(0.40, 0.675),
    )
    sampler = CurriculumStageSampler(
        stages=stages,
        stage_mix=(("C6", 1.0),),
        current_variation_speed_amplitude_mps=0.0,
    )
    specs = sampler.sample_batch(range(256), stage="C6")
    assert all(spec.speed_mps == REMUS100_MAX_SPEED_MPS for spec in specs)
    assert all(spec.current_speed_mps == 0.0 for spec in specs)

    with pytest.raises(ValueError, match="cannot fit"):
        CurriculumStageSampler(
            stages=stages,
            stage_mix=(("C6", 1.0),),
            current_variation_speed_amplitude_mps=0.1,
        ).sample(0, stage="C6")


def test_sampler_variation_bound_must_stay_inside_remus_envelope() -> None:
    with pytest.raises(ValueError, match="variation"):
        CurriculumStageSampler(
            current_variation_speed_amplitude_mps=REMUS100_MAX_SPEED_MPS + 0.01
        )


def test_stage_sampler_is_deterministic_and_within_frozen_ranges() -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C4", 1.0),))
    first = sampler.sample(12345)
    second = sampler.sample(12345)
    assert first == second
    assert first.stage == "C4"
    stage = default_curriculum_stages()["C4"]
    assert stage.depth_range_m[0] <= first.depth_m <= stage.depth_range_m[1]
    assert stage.noise_scale_range[0] <= first.noise_scale <= stage.noise_scale_range[1]
    assert first.path_family in {name for name, _weight in stage.path_family_mix}
    assert first.actor_current_encoder is True
    assert first.current_encoder_alpha_start == 0.0
    assert first.current_encoder_alpha_end == 1.0


def test_stage_sampler_stage_mix_and_seed_sensitivity() -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 0.7), ("C6", 0.3)))
    specs = [sampler.sample(seed) for seed in range(2000)]
    c0 = sum(spec.stage == "C0" for spec in specs)
    c6 = sum(spec.stage == "C6" for spec in specs)
    assert c0 + c6 == len(specs)
    assert 0.66 < c0 / len(specs) < 0.74
    assert all(spec.stage == "C0" for spec in sampler.sample_batch(range(5), stage="C0"))
    assert sampler.sample(1) != sampler.sample(2)


def test_disabled_current_encoder_stages_have_zero_alpha() -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C3", 1.0),))
    spec = sampler.sample(7)
    assert spec.actor_current_encoder is False
    assert spec.current_encoder_alpha_start == 0.0
    assert spec.current_encoder_alpha_end == 0.0


def test_curriculum_stratum_is_derived_from_episode_metadata() -> None:
    c0 = CurriculumStageSampler(stage_mix=(("C0", 1.0),)).sample(20)
    assert curriculum_stratum(c0, initial_heading_rad=-0.1) == "start_heading:neg"
    assert curriculum_stratum(c0, initial_heading_rad=0.1) == "start_heading:nonneg"

    c1 = CurriculumStageSampler(stage_mix=(("C1", 1.0),)).sample(20)
    assert curriculum_stratum(c1) == f"path:{c1.path_family}"

    c3 = CurriculumStageSampler(stage_mix=(("C3", 1.0),)).sample(20)
    expected = "depth:low" if c3.depth_m < 115.0 else "depth:high"
    assert curriculum_stratum(c3) == expected


def test_promotion_requires_two_consecutive_gate_passes() -> None:
    controller = CurriculumPromotionController()
    evaluation = _passing_evaluation()
    assert controller.observe(evaluation) is CurriculumDecision.CONTINUE
    assert controller.stage == "C0"
    assert controller.observe(evaluation) is CurriculumDecision.PROMOTE
    assert controller.stage == "C1"
    assert len(controller.history) == 2


def test_severe_failure_twice_rolls_back_one_stage() -> None:
    controller = CurriculumPromotionController(start_stage="C1")
    evaluation = StageEvaluation(
        default_success=0.75,
        strict_success=0.50,
        timeout_rate=0.30,
        plane_miss_rate=0.30,
        safety_rate=0.0,
        anchor_default_success=0.90,
        stratified_default_success={"depth_low": 0.72, "depth_high": 0.70},
    )
    assert controller.observe(evaluation) is CurriculumDecision.CONTINUE
    assert controller.observe(evaluation) is CurriculumDecision.ROLLBACK
    assert controller.stage == "C0"
    assert controller.rollbacks_used == 1


def test_safety_guard_failure_twice_rolls_back() -> None:
    controller = CurriculumPromotionController(start_stage="C2")
    evaluation = StageEvaluation(
        default_success=0.82,
        strict_success=0.58,
        timeout_rate=0.20,
        plane_miss_rate=0.10,
        safety_rate=0.02,
        anchor_default_success=0.90,
        stratified_default_success={"depth_low": 0.82, "depth_high": 0.80},
    )
    assert controller.observe(evaluation) is CurriculumDecision.CONTINUE
    assert controller.observe(evaluation) is CurriculumDecision.ROLLBACK
    assert controller.stage == "C1"


def test_stalled_metric_triggers_stalled_decision() -> None:
    controller = CurriculumPromotionController(start_stage="C1")
    evaluation = StageEvaluation(
        default_success=0.80,
        strict_success=0.60,
        timeout_rate=0.10,
        plane_miss_rate=0.10,
        safety_rate=0.0,
        anchor_default_success=0.90,
        stratified_default_success={"depth_low": 0.80, "depth_high": 0.78},
    )
    assert controller.observe(evaluation) is CurriculumDecision.CONTINUE
    assert controller.observe(evaluation) is CurriculumDecision.CONTINUE
    assert controller.observe(evaluation) is CurriculumDecision.CURRICULUM_STALLED


def test_budget_and_current_encoder_guards() -> None:
    controller = CurriculumPromotionController()
    budget = controller.observe(_passing_evaluation(), completed_transitions=400_000)
    assert budget is CurriculumDecision.STAGE_BUDGET_EXHAUSTED


def test_rollback_reentry_uses_per_stage_budget_not_global_transitions() -> None:
    controller = CurriculumPromotionController(start_stage="C0")
    passing = _passing_evaluation()
    failing = StageEvaluation(
        default_success=0.65,
        strict_success=0.55,
        timeout_rate=0.0,
        plane_miss_rate=0.35,
        safety_rate=0.0,
        anchor_default_success=0.90,
        stratified_default_success={"path:P0": 0.63, "path:P1": 0.70},
    )

    assert controller.observe(passing, completed_transitions=100_000) is CurriculumDecision.CONTINUE
    assert controller.observe(passing, completed_transitions=200_000) is CurriculumDecision.PROMOTE
    assert controller.stage == "C1"
    assert controller.observe(failing, completed_transitions=300_000) is CurriculumDecision.CONTINUE
    assert controller.observe(failing, completed_transitions=400_000) is CurriculumDecision.ROLLBACK
    assert controller.stage == "C0"

    assert controller.observe(passing, completed_transitions=500_000) is CurriculumDecision.CONTINUE
    assert controller.stage_consumed_transitions == 300_000
    assert (
        controller.observe(passing, completed_transitions=600_000)
        is CurriculumDecision.STAGE_BUDGET_EXHAUSTED
    )

    controller = CurriculumPromotionController(start_stage="C4")
    rejected = controller.observe(
        StageEvaluation(
            default_success=0.82,
            strict_success=0.60,
            timeout_rate=0.10,
            plane_miss_rate=0.10,
            safety_rate=0.0,
            anchor_default_success=0.90,
            stratified_default_success={"depth_low": 0.82, "depth_high": 0.80},
            current_encoder_constant_degradation=0.08,
            current_encoder_fast_layer_compensation=0.0,
        )
    )
    assert rejected is CurriculumDecision.CURRENT_ENCODER_REJECTED


def test_episode_manifest_round_trip(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C5", 1.0),))
    writer = EpisodeManifestWriter(tmp_path / "episodes.jsonl")
    spec = sampler.sample(901, stage="C5")
    writer.append(spec, train_seed=86201, episode_index=3)
    writer.append(spec, train_seed=86201, episode_index=4)
    rows = writer.read()
    assert len(rows) == 2
    assert rows[0]["schema_version"] == CURRICULUM_EPISODE_MANIFEST_SCHEMA
    assert rows[0]["stage"] == "C5"
    assert rows[0]["seed"] == 901
    assert rows[0]["train_seed"] == 86201
    assert json.loads(json.dumps(rows[0], sort_keys=True)) == rows[0]


def test_sampler_rejects_unknown_stage_and_bad_mix() -> None:
    with pytest.raises(ValueError, match="unknown stages"):
        CurriculumStageSampler(stage_mix=(("C9", 1.0),))
    with pytest.raises(ValueError, match="weights"):
        CurriculumStageSampler(stage_mix=(("C0", 0.0),))


def test_stage_mix_target_matches_frozen_protocol_and_preview() -> None:
    without_preview = dict(curriculum_stage_mix_target("C2", preview_enabled=False))
    assert without_preview == pytest.approx(
        {"C2": 0.70 / 0.95, "C1": 0.15 / 0.95, "C0": 0.10 / 0.95}
    )
    with_preview = dict(curriculum_stage_mix_target("C2", preview_enabled=True))
    assert with_preview == pytest.approx({"C2": 0.70, "C1": 0.15, "C0": 0.10, "C3": 0.05})
    assert dict(curriculum_stage_mix_target("C6", preview_enabled=True)) == pytest.approx(
        {"C6": 0.80, "C5": 0.15, "C0": 0.05}
    )


def test_stage_mix_ramp_is_linear_and_sets_sampler_mix() -> None:
    start = (("C2", 1.0),)
    target = curriculum_stage_mix_target("C3", preview_enabled=False)
    ramp = CurriculumMixRamp(
        start_mix=start,
        target_mix=target,
        start_transition=1000,
        transitions=100,
    )
    assert dict(ramp.mix_at(1000)) == pytest.approx({"C2": 1.0})
    midpoint = dict(ramp.mix_at(1050))
    expected = dict(start)
    for key, weight in target:
        expected[key] = 0.5 * expected.get(key, 0.0) + 0.5 * weight
    assert midpoint == pytest.approx(expected)
    assert dict(ramp.mix_at(1100)) == pytest.approx(dict(target))

    sampler = CurriculumStageSampler(stage_mix=start)
    sampler.set_stage_mix(ramp.mix_at(1100))
    assert dict(sampler.stage_mix) == pytest.approx(dict(target))
    sampled = sampler.sample_batch(range(2000))
    counts = {key: sum(spec.stage == key for spec in sampled) / len(sampled) for key, _ in target}
    for key, weight in target:
        assert counts[key] == pytest.approx(weight, abs=0.035)

