from __future__ import annotations

from auv_lowlevel.curriculum_resume import curriculum_state_from_records


def _record(iteration: int, stage: str, steps: int, mix: list[list[object]] | None = None):
    return {
        "iteration": iteration,
        "curriculum_stage": stage,
        "curriculum_stage_mix": mix or [[stage, 1.0]],
        "total_env_steps": steps,
    }


def test_reconstructs_current_stage_budget() -> None:
    records = [
        _record(1, "C0", 100),
        _record(2, "C0", 200),
        _record(3, "C1", 300),
        _record(4, "C1", 400),
        _record(5, "C1", 500),
    ]

    state = curriculum_state_from_records(records, upto_iteration=5)

    assert state is not None
    assert state["stage"] == "C1"
    assert state["completed_iterations"] == 5
    assert state["completed_transitions"] == 500
    controller = state["controller_state"]
    assert controller["stage_consumed_iterations"] == {"C1": 3}
    assert controller["stage_consumed"] == {"C1": 300}
    assert controller["stage_entry_iteration"] == 5
    assert controller["stage_entry_transition"] == 500


def test_ignores_records_after_checkpoint() -> None:
    records = [
        _record(1, "C0", 100),
        _record(2, "C5", 200, [["C5", 0.7], ["C4", 0.3]]),
        _record(3, "C0", 300),
    ]

    state = curriculum_state_from_records(records, upto_iteration=2)

    assert state is not None
    assert state["stage"] == "C5"
    assert state["stage_mix"] == [["C5", 0.7], ["C4", 0.3]]
    assert state["controller_state"]["stage_consumed_iterations"] == {"C5": 1}


def test_empty_records_return_none() -> None:
    assert curriculum_state_from_records([], upto_iteration=10) is None
    assert (
        curriculum_state_from_records([_record(20, "C0", 100)], upto_iteration=10)
        is None
    )
