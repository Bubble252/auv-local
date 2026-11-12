from __future__ import annotations

import json
from pathlib import Path

import pytest

from auv_lowlevel.curriculum import (
    CurriculumDecision,
    CurriculumPromotionController,
    CurriculumStageSampler,
    StageEvaluation,
    curriculum_stage_mix_target,
    default_curriculum_stages,
)
from auv_lowlevel.curriculum_trainer import (
    CurriculumTrainerHook,
    combine_eval_summaries,
    parse_eval_summary,
)


def _summary(default_success: int, strict_success: int) -> dict:
    episodes = []
    for index in range(4):
        episodes.append(
            {
                "success": index < default_success,
                "success_eta_tight_at_done": index < strict_success,
                "timeout": False,
                "goal_plane_miss": index >= default_success,
                "safety": False,
            }
        )
    return {"episodes": episodes}


def test_parse_and_combine_eval_summaries() -> None:
    default = parse_eval_summary(_summary(default_success=2, strict_success=1))
    assert default.default_success == 0.5
    assert default.strict_success == 0.25

    combined = combine_eval_summaries(
        _summary(default_success=3, strict_success=3),
        _summary(default_success=1, strict_success=1),
    )
    assert combined.default_success == 0.75
    assert combined.strict_success == 0.25
    assert combined.plane_miss_rate == 0.75


def _stratified_summary(
    successes: dict[str, list[bool]],
    *,
    strict_successes: dict[str, list[bool]] | None = None,
) -> dict:
    strict_successes = successes if strict_successes is None else strict_successes
    episodes = []
    for stratum, values in successes.items():
        for index, success in enumerate(values):
            episodes.append(
                {
                    "seed": len(episodes),
                    "success": success,
                    "success_eta_tight_at_done": strict_successes[stratum][index],
                    "timeout": False,
                    "goal_plane_miss": not success,
                    "safety": False,
                    "curriculum": {"stratum": stratum},
                }
            )
    return {"episodes": episodes}


def test_parse_eval_summary_computes_strata_from_episode_metadata() -> None:
    summary = _stratified_summary(
        {
            "start_heading:neg": [True] * 19 + [False],
            "start_heading:nonneg": [True] * 18 + [False, False],
        }
    )
    evaluation = parse_eval_summary(summary, require_stratified=True)
    assert evaluation.stratified_default_success == {
        "start_heading:neg": 0.95,
        "start_heading:nonneg": 0.90,
    }


def test_stratified_gate_rejects_missing_strata() -> None:
    gate = default_curriculum_stages()["C0"].gate
    with pytest.raises(ValueError, match="curriculum.stratum"):
        combine_eval_summaries(_summary(4, 4), _summary(4, 4), gate=gate)


def test_stratified_gate_uses_every_stratum_not_only_overall_success() -> None:
    gate = default_curriculum_stages()["C0"].gate
    passing = _stratified_summary(
        {
            "start_heading:neg": [True] * 19 + [False],
            "start_heading:nonneg": [True] * 19 + [False],
        }
    )
    combined = combine_eval_summaries(passing, passing, gate=gate)
    assert combined.default_success == 0.95
    assert combined.gate_passes(gate)

    failing_stratum = _stratified_summary(
        {
            "start_heading:neg": [True] * 19 + [False],
            "start_heading:nonneg": [False] * 10 + [True] * 10,
        }
    )
    combined = combine_eval_summaries(failing_stratum, failing_stratum, gate=gate)
    assert combined.default_success == 0.725
    assert not combined.gate_passes(gate)


def test_hook_promotes_after_two_passes(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    controller = CurriculumPromotionController(start_stage="C0")
    evaluations = [
        StageEvaluation(
            0.95, 0.80, 0.0, 0.0, 0.0,
            stratified_default_success={"default": 0.95},
        ),
        StageEvaluation(
            0.95, 0.80, 0.0, 0.0, 0.0,
            stratified_default_success={"default": 0.95},
        ),
    ]
    hook = CurriculumTrainerHook(
        controller=controller,
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=100,
        evaluate=lambda _stage, _steps: evaluations.pop(0),
    )

    assert hook.maybe_update(99) is None
    assert hook.maybe_update(100) == CurriculumDecision.CONTINUE
    assert hook.maybe_update(200) == CurriculumDecision.PROMOTE
    assert hook.stage == "C1"
    assert hook.stage_mix == curriculum_stage_mix_target("C1", preview_enabled=True)
    records = [
        json.loads(line)
        for line in (tmp_path / "curriculum_decisions.jsonl").read_text().splitlines()
    ]
    assert [row["decision"] for row in records] == ["CONTINUE", "PROMOTE"]


def test_hook_rolls_back_after_two_severe_failures(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C2", 1.0),))
    controller = CurriculumPromotionController(start_stage="C2")
    evaluations = [
        StageEvaluation(0.10, 0.05, 0.0, 0.0, 0.0),
        StageEvaluation(0.10, 0.05, 0.0, 0.0, 0.0),
    ]
    hook = CurriculumTrainerHook(
        controller=controller,
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=100,
        evaluate=lambda _stage, _steps: evaluations.pop(0),
    )

    assert hook.maybe_update(100) == CurriculumDecision.CONTINUE
    assert hook.maybe_update(200) == CurriculumDecision.ROLLBACK
    assert hook.stage == "C1"
    assert hook.stage_mix == curriculum_stage_mix_target("C1", preview_enabled=True)


def test_stage_aware_current_encoder_alpha(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    controller = CurriculumPromotionController(start_stage="C0")
    hook = CurriculumTrainerHook(
        controller=controller,
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=100,
        evaluate=lambda _stage, _steps: StageEvaluation(0.0, 0.0, 0.0, 1.0, 0.0),
    )
    assert hook.current_encoder_alpha(0) == 0.0
    controller.stage = "C4"
    hook._stage_entered_at = 1_000
    assert hook.current_encoder_alpha(1_000) == 0.0
    assert hook.current_encoder_alpha(51_000) == 0.5
    assert hook.current_encoder_alpha(101_000) == 1.0
    controller.stage = "C6"
    assert hook.current_encoder_alpha(101_000) == 1.0


def test_forced_zero_current_encoder_alpha_overrides_stage_ramp(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C4", 1.0),))
    hook = CurriculumTrainerHook(
        controller=CurriculumPromotionController(start_stage="C4"),
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=100,
        evaluate=lambda _stage, _steps: StageEvaluation(0.0, 0.0, 0.0, 1.0, 0.0),
        force_current_encoder_alpha_zero=True,
    )
    for completed in (0, 1_000, 51_000, 101_000):
        assert hook.current_encoder_alpha(completed) == 0.0
    state = json.loads((tmp_path / "curriculum_state.json").read_text())
    assert state["force_current_encoder_alpha_zero"] is True
