"""Reconstruct curriculum trainer state from iteration records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def curriculum_state_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    upto_iteration: int,
) -> dict[str, Any] | None:
    """Build a curriculum resume payload for checkpoints without embedded state.

    Legacy checkpoints do not contain ``curriculum_state``. The on-disk
    ``curriculum_state.json`` can also be overwritten by a failed resume
    attempt before it is read. Per-iteration records are append-only up to the
    checkpoint and retain the active stage and stage mix, so they are the
    authoritative fallback.
    """

    completed = [
        record
        for record in records
        if record.get("iteration") is not None
        and int(record["iteration"]) <= int(upto_iteration)
    ]
    if not completed:
        return None
    last = completed[-1]
    stage_value = last.get("curriculum_stage")
    if stage_value is None:
        return None
    stage = str(stage_value)

    start = len(completed) - 1
    while start > 0 and str(completed[start - 1].get("curriculum_stage")) == stage:
        start -= 1
    first_iteration = int(completed[start]["iteration"])
    last_iteration = int(last["iteration"])
    last_steps = int(last.get("total_env_steps") or 0)
    entry_steps = (
        int(completed[start - 1].get("total_env_steps") or 0) if start > 0 else 0
    )

    return {
        "schema_version": "auv_pc_curriculum_trainer_state.v1",
        "completed_transitions": last_steps,
        "completed_iterations": last_iteration,
        "stage": stage,
        "stage_mix": last.get("curriculum_stage_mix"),
        "promotion_enabled": True,
        "frozen_stage": None,
        "controller_state": {
            "stage": stage,
            "consecutive_passes": 0,
            "consecutive_unsafe": 0,
            "consecutive_stalls": 0,
            "rollback_counts": {},
            "last_completion_ratio": None,
            "stage_consumed": {stage: max(0, last_steps - entry_steps)},
            "stage_consumed_iterations": {
                stage: max(1, last_iteration - first_iteration + 1)
            },
            "stage_entry_transition": entry_steps,
            "stage_entry_iteration": first_iteration - 1,
        },
        "last_decision": None,
        "rollbacks_used": 0,
        "force_current_encoder_alpha_zero": False,
        "eval_schedule": "stage_boundary",
        "next_eval_transition": None,
        "next_eval_iteration": None,
        "finalized": False,
    }
