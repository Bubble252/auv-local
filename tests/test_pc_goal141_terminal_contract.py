from __future__ import annotations

import math
from pathlib import Path

import torch

from auv_lowlevel.batch.pc import BatchedPCTerminationManager, PCTerminalKind
from auv_lowlevel.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lowlevel" / "pc_goal85_unbiased_strict125_candidate_20261110.json"
GOAL = torch.tensor([[100.0, 20.0, 10.0]], dtype=torch.float64)
TANGENT = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)


def _manager() -> BatchedPCTerminationManager:
    config = load_config(CONFIG)
    manager = BatchedPCTerminationManager(
        config,
        device="cpu",
        dtype=torch.float64,
        goal_radius_m=config.plant.goal_radius_m,
        goal_plane_miss_guard_radius_m=config.plant.goal_radius_m,
        goal_plane_miss_dwell_steps=0,
    )
    manager.reset(1)
    return manager


def _warm_roll_safety(manager: BatchedPCTerminationManager) -> None:
    config = manager.config
    eta = torch.tensor([[0.0, 0.0, 10.0, math.radians(60.0), 0.0, 0.0]], dtype=torch.float64)
    nu = torch.zeros((1, 6), dtype=torch.float64)
    for step in range(60):
        decision = manager.update(
            eta=eta,
            nu=nu,
            goal_ned_m=GOAL,
            goal_tangent_ned=TANGENT,
            bounds_ned_m=torch.as_tensor(config.plant.bounds_ned_m),
            step_count=torch.tensor([step], dtype=torch.long),
        )
    assert float(decision.roll_exceed_time_s[0]) >= config.termination.t_phi_term_s


def test_goal_wins_over_roll_safety_in_endpoint_stop_branch() -> None:
    manager = _manager()
    _warm_roll_safety(manager)
    config = manager.config
    eta = torch.tensor([[100.0, 20.0, 10.0, math.radians(60.0), 0.0, 0.0]], dtype=torch.float64)
    nu = torch.zeros((1, 6), dtype=torch.float64)
    decision = manager.update(
        eta=eta,
        nu=nu,
        goal_ned_m=GOAL,
        goal_tangent_ned=TANGENT,
        bounds_ned_m=torch.as_tensor(config.plant.bounds_ned_m),
        step_count=torch.tensor([60], dtype=torch.long),
    )
    assert int(decision.kind[0]) == int(PCTerminalKind.GOAL)
    assert not bool(decision.safety_event[0])


def test_goal_plane_miss_wins_over_roll_safety_in_endpoint_stop_branch() -> None:
    manager = _manager()
    _warm_roll_safety(manager)
    config = manager.config
    eta = torch.tensor([[103.0, 20.0, 10.0, math.radians(60.0), 0.0, 0.0]], dtype=torch.float64)
    nu = torch.zeros((1, 6), dtype=torch.float64)
    decision = manager.update(
        eta=eta,
        nu=nu,
        goal_ned_m=GOAL,
        goal_tangent_ned=TANGENT,
        bounds_ned_m=torch.as_tensor(config.plant.bounds_ned_m),
        step_count=torch.tensor([60], dtype=torch.long),
    )
    assert int(decision.kind[0]) == int(PCTerminalKind.GOAL_PLANE_MISS)
    assert not bool(decision.safety_event[0])

