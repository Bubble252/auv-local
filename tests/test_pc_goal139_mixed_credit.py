from __future__ import annotations

from scripts.analyze_pc_goal139_mixed_credit import _ppo_status, _sac_status
from scripts.build_pc_goal139_mixed_credit_configs import (
    PPO_ARMS,
    SAC_ARMS,
    build_ppo_environment_config,
    build_sac_environment_config,
)


def test_goal139_sac_environment_fractions_are_frozen() -> None:
    m50 = build_sac_environment_config("M50")
    m75 = build_sac_environment_config("M75")
    assert SAC_ARMS["M50"]["start_progress_full_fraction"] == 0.5
    assert SAC_ARMS["M75"]["start_progress_full_fraction"] == 0.25
    assert m50["reference"]["start_progress_m"] == 20.0
    assert m50["reference"]["start_progress_full_fraction"] == 0.5
    assert m75["reference"]["start_progress_full_fraction"] == 0.25


def test_goal139_ppo_environment_is_t15_only() -> None:
    env = build_ppo_environment_config()
    assert env["reference"]["start_progress_m"] == 20.0
    assert env["reference"]["start_progress_full_fraction"] == 0.0
    assert [PPO_ARMS[arm]["gamma"] for arm in ("G995", "G980", "G950")] == [
        0.995,
        0.98,
        0.95,
    ]


def test_goal139_sac_status_prefers_zero_advantage_only_with_guards() -> None:
    status, _ = _sac_status(
        helped_seeds=2,
        zero_seeds=2,
        control_nominal_guard=True,
        zero_nominal_guard=True,
        safety_bad=False,
        path_bad=False,
        authority_bad=False,
    )
    assert status == "MIXED_ZERO_ADVANTAGE"
    status, _ = _sac_status(
        helped_seeds=2,
        zero_seeds=1,
        control_nominal_guard=True,
        zero_nominal_guard=True,
        safety_bad=False,
        path_bad=False,
        authority_bad=False,
    )
    assert status == "MIXED_ANCHOR_SIGNAL"
    status, _ = _sac_status(
        helped_seeds=2,
        zero_seeds=2,
        control_nominal_guard=False,
        zero_nominal_guard=True,
        safety_bad=False,
        path_bad=False,
        authority_bad=False,
    )
    assert status == "MIXED_NOMINAL_TRADEOFF"


def test_goal139_ppo_status_prioritizes_residual_collapse() -> None:
    status, _ = _ppo_status(
        helped_seeds=3,
        zero_seeds=3,
        control_nominal_guard=True,
        zero_nominal_guard=True,
        safety_bad=False,
        path_bad=False,
        residual_collapse=True,
    )
    assert status == "PPO_RESIDUAL_COLLAPSE_PERSISTS"
    status, _ = _ppo_status(
        helped_seeds=2,
        zero_seeds=2,
        control_nominal_guard=True,
        zero_nominal_guard=True,
        safety_bad=False,
        path_bad=False,
        residual_collapse=False,
    )
    assert status == "PPO_CREDIT_ZERO_ADVANTAGE"
