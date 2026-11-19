from __future__ import annotations

import json
from pathlib import Path

import pytest

from auv_lowlevel.curriculum import (
    CurriculumDecision,
    CurriculumPromotionController,
    CurriculumStageSampler,
    RouteGateThresholds,
    RouteGateWindow,
    StageEvaluation,
    curriculum_stage_mix_target,
    default_curriculum_stages,
)
from auv_lowlevel.curriculum_trainer import (
    CurriculumTrainerHook,
    parse_eval_summary,
)


def _summary(
    success_count: int = 4,
    *,
    completion: float = 0.95,
    safety: float = 0.0,
    pre_goal_3d: float = 0.30,
) -> dict:
    episodes = []
    for index in range(4):
        episodes.append(
            {
                "success": index < success_count,
                "success_eta_tight_at_done": index < success_count,
                "timeout": index >= success_count,
                "goal_plane_miss": index >= success_count,
                "safety": index < round(safety * 4),
                "path_completion_ratio": completion,
                "path_horizontal_rmse_m": 0.20,
                "path_vertical_rmse_m": 0.20,
                "path_three_d_rmse_m": 0.30,
                "pre_goal_3d_rmse_m": pre_goal_3d,
            }
        )
    return {"episodes": episodes}


def test_parse_eval_summary_keeps_terminal_outcomes_diagnostic() -> None:
    evaluation = parse_eval_summary(_summary(success_count=0, completion=0.95, pre_goal_3d=999.0))
    gate = default_curriculum_stages()["C0"].gate
    assert evaluation.default_success == 0.0
    assert evaluation.timeout_rate == 1.0
    assert evaluation.plane_miss_rate == 1.0
    assert evaluation.path_completion_ratio == 0.95
    assert evaluation.pre_goal_3d_rmse_m == 999.0
    assert evaluation.gate_passes(gate) is True


def test_parse_eval_summary_requires_completion_telemetry() -> None:
    summary = _summary()
    for episode in summary["episodes"]:
        del episode["path_completion_ratio"]
    with pytest.raises(ValueError, match="path_completion_ratio"):
        parse_eval_summary(summary)


def _stage_eval(
    *,
    safety: float = 0.0,
    completion: float = 0.95,
    success: float = 0.95,
) -> StageEvaluation:
    return StageEvaluation(
        safety_rate=safety,
        path_completion_ratio=completion,
        cross_track_rmse_m=0.20,
        vertical_rmse_m=0.20,
        default_success=success,
        eta_tight_success_rate=success,
        timeout_rate=0.0 if success > 0.0 else 1.0,
        plane_miss_rate=0.0 if success > 0.0 else 1.0,
        path_three_d_rmse_m=0.30,
        pre_goal_3d_rmse_m=0.30,
    )


def test_hook_promotes_after_two_passes(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    controller = CurriculumPromotionController(start_stage="C0")
    evaluations = [
        _stage_eval(),
        _stage_eval(),
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


def test_hook_state_roundtrip_restores_promotion_and_schedule(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    controller = CurriculumPromotionController(start_stage="C0")
    evaluations = [_stage_eval(), _stage_eval()]
    hook = CurriculumTrainerHook(
        controller=controller,
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=100,
        evaluate=lambda _stage, _steps: evaluations.pop(0),
    )
    assert hook.maybe_update(100) == CurriculumDecision.CONTINUE
    assert hook.maybe_update(200) == CurriculumDecision.PROMOTE
    state = hook.state_dict()

    restored_sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    restored = CurriculumTrainerHook(
        controller=CurriculumPromotionController(start_stage="C0"),
        sampler=restored_sampler,
        output_dir=tmp_path / "restored",
        eval_interval_transitions=100,
        evaluate=lambda _stage, _steps: _stage_eval(),
    )
    restored.load_state(state)

    assert restored.stage == "C1"
    assert restored.stage_mix == curriculum_stage_mix_target("C1", preview_enabled=True)
    assert restored._next_eval_transition == 300
    assert restored.controller.stage_consumed_transitions == hook.controller.stage_consumed_transitions


def test_hook_fails_closed_at_stage_hard_cap_without_eval(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    controller = CurriculumPromotionController(start_stage="C0")
    evaluations: list[str] = []

    def evaluate(stage: str, _transitions: int) -> StageEvaluation:
        evaluations.append(stage)
        return _stage_eval()

    hook = CurriculumTrainerHook(
        controller=controller,
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=10_000_000,
        evaluate=evaluate,
    )

    assert hook.check_stage_budget(1_000, 149) is None
    status = hook.check_stage_budget(2_000, 150)
    assert status is not None and status.exhausted is True
    assert evaluations == []
    rows = [
        json.loads(line)
        for line in (tmp_path / "curriculum_stage_budget.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["decision"] == "STAGE_BUDGET_EXHAUSTED"
    assert rows[-1]["stage"] == "C0"
    assert rows[-1]["consumed_iterations"] == 150
    state = json.loads((tmp_path / "curriculum_state.json").read_text())
    assert state["last_decision"] == "STAGE_BUDGET_EXHAUSTED"


def test_stage_boundary_eval_waits_for_soft_budget_and_retries_after_window(
    tmp_path: Path,
) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    controller = CurriculumPromotionController(start_stage="C0")
    evaluations = [
        _stage_eval(),
        _stage_eval(),
    ]
    calls: list[tuple[str, int]] = []

    def evaluate(stage: str, transitions: int) -> StageEvaluation:
        calls.append((stage, transitions))
        return evaluations.pop(0)

    hook = CurriculumTrainerHook(
        controller=controller,
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=1,
        evaluate=evaluate,
        stage_boundary_evals=True,
        stage_boundary_retry_iterations=20,
    )

    assert hook.maybe_update(10_000, completed_iterations=71) is None
    assert calls == []
    assert hook.maybe_update(20_000, completed_iterations=72) == CurriculumDecision.CONTINUE
    assert calls == [("C0", 20_000)]
    assert hook.maybe_update(30_000, completed_iterations=91) is None
    assert len(calls) == 1
    assert hook.maybe_update(40_000, completed_iterations=92) == CurriculumDecision.PROMOTE
    assert calls == [("C0", 20_000), ("C0", 40_000)]
    assert hook.stage == "C1"
    assert hook._next_stage_eval_iteration == 200

    state = json.loads((tmp_path / "curriculum_state.json").read_text())
    assert state["eval_schedule"] == "stage_boundary"
    assert state["next_eval_iteration"] == 200


def test_hook_rolls_back_after_two_safety_failures(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C2", 1.0),))
    controller = CurriculumPromotionController(start_stage="C2")
    evaluations = [
        _stage_eval(safety=0.20, success=0.0),
        _stage_eval(safety=0.20, success=0.0),
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
        evaluate=lambda _stage, _steps: _stage_eval(completion=0.0, success=0.0),
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
        evaluate=lambda _stage, _steps: _stage_eval(completion=0.0, success=0.0),
        force_current_encoder_alpha_zero=True,
    )
    for completed in (0, 1_000, 51_000, 101_000):
        assert hook.current_encoder_alpha(completed) == 0.0
    state = json.loads((tmp_path / "curriculum_state.json").read_text())
    assert state["force_current_encoder_alpha_zero"] is True


def _training_route_metrics(*, safety: float = 0.0) -> dict[str, float]:
    return {
        "/episode/cross_track_rmse_mean": 0.10,
        "/episode/vertical_rmse_mean": 0.10,
        "/episode/pre_goal_3d_rmse_mean": 0.15,
        "/episode/path_completion_mean": 0.95,
        "/episode/safety_termination_rate": safety,
    }


def _route_thresholds() -> RouteGateThresholds:
    return RouteGateThresholds(
        cross_track_rmse_max_m=0.25,
        vertical_rmse_max_m=0.25,
        completion_min=0.80,
        safety_max=0.0,
    )


def test_enforce_route_gate_promotes_on_training_rollout_20_3(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    evaluations: list[tuple[str, int]] = []

    def evaluate(stage: str, transitions: int) -> StageEvaluation:
        evaluations.append((stage, transitions))
        return _stage_eval()

    hook = CurriculumTrainerHook(
        controller=CurriculumPromotionController(start_stage="C0"),
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=10_000_000,
        evaluate=evaluate,
        routine_gate=RouteGateWindow(
            window_size=20,
            consecutive_required=3,
            mode="enforce",
        ),
        route_gate_thresholds=_route_thresholds(),
    )

    decision = None
    for iteration in range(1, 75):
        decision = hook.maybe_update(
            iteration * 1_000,
            completed_iterations=iteration,
            training_metrics=_training_route_metrics(),
        )

    assert decision == CurriculumDecision.PROMOTE
    assert hook.stage == "C1"
    assert evaluations == [("C0", 74_000)]
    record = json.loads((tmp_path / "curriculum_decisions.jsonl").read_text().splitlines()[-1])
    assert record["reason"].startswith("routine training-rollout route gate passed")
    assert record["route_gate"]["should_promote"] is True
    assert record["route_gate"]["consecutive_passes"] == 3
    route_rows = [
        json.loads(line)
        for line in (tmp_path / "curriculum_route_gate.jsonl").read_text().splitlines()
    ]
    assert len(route_rows) == 74
    assert route_rows[-1]["should_promote"] is True


def test_hook_skips_stage_budget_when_promotion_disabled(tmp_path: Path) -> None:
    hook = CurriculumTrainerHook(
        controller=CurriculumPromotionController(start_stage="C0"),
        sampler=CurriculumStageSampler(stage_mix=(("C0", 1.0),)),
        output_dir=tmp_path,
        eval_interval_transitions=10_000_000,
        evaluate=lambda _stage, _steps: _stage_eval(),
        promotion_enabled=False,
    )

    assert hook.check_stage_budget(1_000_000, 1_000) is None
    assert not (tmp_path / "curriculum_stage_budget.jsonl").exists()


def test_enforce_route_gate_boundary_safety_veto_keeps_stage(tmp_path: Path) -> None:
    sampler = CurriculumStageSampler(stage_mix=(("C0", 1.0),))
    evaluations: list[tuple[str, int]] = []

    def evaluate(stage: str, transitions: int) -> StageEvaluation:
        evaluations.append((stage, transitions))
        return _stage_eval(safety=0.25, success=0.0)

    gate = RouteGateWindow(window_size=2, consecutive_required=2, mode="enforce")
    hook = CurriculumTrainerHook(
        controller=CurriculumPromotionController(start_stage="C0"),
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=10_000_000,
        evaluate=evaluate,
        routine_gate=gate,
        route_gate_thresholds=_route_thresholds(),
    )

    decision = None
    for iteration in range(1, 74):
        decision = hook.maybe_update(
            iteration * 1_000,
            completed_iterations=iteration,
            training_metrics=_training_route_metrics(),
        )

    assert decision == CurriculumDecision.CONTINUE
    assert hook.stage == "C0"
    assert evaluations == [("C0", 73_000)]
    assert gate.last_status["reason"] == "STAGE_RESET"
    record = json.loads((tmp_path / "curriculum_decisions.jsonl").read_text().splitlines()[-1])
    assert record["reason"].startswith("frozen-dev boundary safety veto")
    assert record["route_gate"]["should_promote"] is True


def test_enforce_route_gate_requires_frozen_thresholds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires frozen route thresholds"):
        CurriculumTrainerHook(
            controller=CurriculumPromotionController(start_stage="C0"),
            sampler=CurriculumStageSampler(stage_mix=(("C0", 1.0),)),
            output_dir=tmp_path,
            eval_interval_transitions=100,
            evaluate=lambda _stage, _steps: _stage_eval(completion=0.0, success=0.0),
            routine_gate=RouteGateWindow(mode="enforce"),
        )


def test_route_gate_uses_stage_local_iterations(tmp_path: Path) -> None:
    """A late-entered stage must not inherit the global iteration counter."""

    controller = CurriculumPromotionController(start_stage="C0")
    assert controller.check_stage_budget(112).exhausted is False
    controller.stage = "C1"
    controller._stage_entry_iteration = 112

    gate = RouteGateWindow(window_size=2, consecutive_required=2, mode="enforce")
    hook = CurriculumTrainerHook(
        controller=controller,
        sampler=CurriculumStageSampler(stage_mix=(("C1", 1.0),)),
        output_dir=tmp_path,
        eval_interval_transitions=10_000_000,
        evaluate=lambda _stage, _steps: _stage_eval(),
        routine_gate=gate,
        route_gate_thresholds=_route_thresholds(),
    )

    for completed in range(113, 121):
        hook.maybe_update(
            1_000_000 + completed,
            completed_iterations=completed,
            training_metrics=_training_route_metrics(),
        )

    assert hook.stage == "C1"
    assert gate.last_status["eligible"] is False
    assert gate.last_status["reason"] == "SOFT_ITERATION_CAP_NOT_REACHED"
    assert gate.last_status["min_iterations"] == 108
    assert controller.stage_consumed_iterations == 8
