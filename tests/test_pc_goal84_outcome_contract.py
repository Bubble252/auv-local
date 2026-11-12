from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from auv_lowlevel.outcome_contract import (
    score_success_contract,
    summarize_success_contract,
    validate_success_frame,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_ppo_module() -> ModuleType:
    path = ROOT / "scripts" / "evaluate_pc_ppo.py"
    spec = importlib.util.spec_from_file_location("evaluate_pc_ppo_goal84", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_success_frames_have_distinct_selected_semantics() -> None:
    p_nav = score_success_contract(
        p_nav_tight_success=True,
        eta_tight_ever=False,
        eta_tight_at_done=False,
        success_frame="p_nav",
    )
    eta = score_success_contract(
        p_nav_tight_success=False,
        eta_tight_ever=True,
        eta_tight_at_done=False,
        success_frame="eta",
    )
    both = score_success_contract(
        p_nav_tight_success=False,
        eta_tight_ever=True,
        eta_tight_at_done=False,
        success_frame="both",
    )

    assert p_nav["success_selected"] is True
    assert p_nav["success_eta_tight_ever"] is False
    assert eta["success_selected"] is True
    assert eta["success_pnav_tight"] is False
    assert both["success_selected"] is None
    assert both["success_frame_gap_eta_minus_pnav"] == pytest.approx(1.0)


def test_success_summary_reports_pnav_eta_and_gap() -> None:
    rows = [
        score_success_contract(
            p_nav_tight_success=True,
            eta_tight_ever=True,
            eta_tight_at_done=True,
            success_frame="both",
        ),
        score_success_contract(
            p_nav_tight_success=False,
            eta_tight_ever=True,
            eta_tight_at_done=False,
            success_frame="both",
        ),
        score_success_contract(
            p_nav_tight_success=False,
            eta_tight_ever=False,
            eta_tight_at_done=False,
            success_frame="both",
        ),
    ]

    summary = summarize_success_contract(rows, success_frame="both")

    assert summary["success_rate_pnav_tight"] == pytest.approx(1 / 3)
    assert summary["success_rate_eta_tight_ever"] == pytest.approx(2 / 3)
    assert summary["success_rate_eta_tight_at_done"] == pytest.approx(1 / 3)
    assert summary["success_frame_gap_eta_minus_pnav"] == pytest.approx(1 / 3)
    assert summary["success_rate_selected"] is None


def test_invalid_success_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported success frame"):
        validate_success_frame("protocol")


def test_eta_reporting_does_not_change_episode_control_fields() -> None:
    module = _load_ppo_module()
    metrics = module._new_episode_metrics(1)
    metrics["active_steps"][:] = [4]
    metrics["path_steps"][:] = [4]
    metrics["goal_terminations"][:] = [0]
    metrics["pnav_goal_capture_seen"][:] = [False]
    metrics["eta_goal_capture_seen"][:] = [True]
    metrics["eta_goal_capture_at_done"][:] = [False]
    metrics["reward_sum"][:] = [2.0]

    p_nav_rows = module._episode_records(
        metrics,
        (20,),
        path_enabled=False,
        path_length_m=10.0,
        action_full_scale=(0.5, 0.25, 1000.0),
        success_frame="p_nav",
    )
    eta_rows = module._episode_records(
        metrics,
        (20,),
        path_enabled=False,
        path_length_m=10.0,
        action_full_scale=(0.5, 0.25, 1000.0),
        success_frame="eta",
    )

    assert p_nav_rows[0]["success_selected"] is False
    assert eta_rows[0]["success_selected"] is True
    assert p_nav_rows[0]["success"] is eta_rows[0]["success"] is False
    assert p_nav_rows[0]["steps"] == eta_rows[0]["steps"] == 4
    assert p_nav_rows[0]["mean_reward"] == eta_rows[0]["mean_reward"] == 0.5
    assert metrics["goal_terminations"][0] == 0
    assert metrics["reward_sum"][0] == pytest.approx(2.0)
    assert np.isclose(metrics["active_steps"][0], 4)
