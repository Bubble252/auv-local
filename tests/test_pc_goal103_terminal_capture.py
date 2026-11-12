from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load(name: str, relative: str):
    script = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load("goal103_terminal_capture", "scripts/analyze_pc_goal103_terminal_capture.py")


def _trace(seed: int = 20) -> dict[str, np.ndarray]:
    steps = 4
    eta = np.zeros((steps, 6), dtype=np.float64)
    goal = np.tile(np.array([10.0, 0.0, 0.0]), (steps, 1))
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (steps, 1))
    signed = np.array([-1.0, -0.1, 0.2, 0.4], dtype=np.float64)
    eta[:, 0] = goal[:, 0] + signed
    eta[2:, 1] = 0.1
    eta[2:, 2] = -0.05
    return {
        "episode_seed": np.full(steps, seed, dtype=np.int64),
        "step": np.arange(steps, dtype=np.int64),
        "eta": eta,
        "goal_ned_m": goal,
        "goal_tangent_ned": tangent,
        "signed_goal_tangent_offset_eta_m": signed,
        "goal_distance_eta_m": np.linalg.norm(eta[:, :3] - goal, axis=1),
        "goal_distance_pnav_m": np.linalg.norm(eta[:, :3] - goal, axis=1),
        "endpoint_entry_reached": np.array([False, False, True, True]),
        "material_saturated": np.array([False, False, True, True]),
        "actuator_residual_applied": np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.3, 0.2, 0.1]]
        ),
        "time_s": np.arange(steps, dtype=np.float64),
    }


def test_episode_geometry_uses_first_plane_crossing() -> None:
    row = audit._episode_geometry(_trace(), seed=20, goal_radius_m=1.25)
    assert row["plane_crossing_step"] == 2
    assert row["endpoint_entry_reached"] is True
    assert row["signed_tangent_offset_m"] >= 0.0
    assert row["miss_distance_at_plane_m"] > 0.0
    assert row["perpendicular_miss_m"] >= 0.0


def _failure(seed: int, *, excess: float, clip: float, pnav_capture: bool, eta_capture: bool):
    return {
        "seed": seed,
        "endpoint_entry_reached": True,
        "radial_excess_m": excess,
        "magnitude_clip_rate": clip,
        "rate_clip_rate": clip,
        "goal_capture_pnav": pnav_capture,
        "goal_capture_eta": eta_capture,
    }


def test_shell_dominant_classification() -> None:
    rows = [
        _failure(seed, excess=0.1, clip=0.1, pnav_capture=True, eta_capture=False)
        for seed in range(20, 30)
    ]
    rows += [
        _failure(seed, excess=0.8, clip=0.8, pnav_capture=False, eta_capture=False)
        for seed in range(30, 40)
    ]
    result = audit._classify(rows)
    assert result["status"] == "TERMINAL_SHELL_DOMINANT"
    assert result["shell_failures"] == 10


def test_authority_limited_classification() -> None:
    rows = [
        _failure(seed, excess=0.8, clip=0.8, pnav_capture=False, eta_capture=False)
        for seed in range(20, 35)
    ]
    rows += [
        _failure(seed, excess=0.1, clip=0.1, pnav_capture=False, eta_capture=False)
        for seed in range(35, 40)
    ]
    result = audit._classify(rows)
    assert result["status"] == "CONTROL_AUTHORITY_LIMITED"
    assert result["authority_failures"] == 15


def test_nav_eta_disagreement_classification() -> None:
    rows = [
        _failure(seed, excess=0.8, clip=0.1, pnav_capture=True, eta_capture=False)
        for seed in range(20, 30)
    ]
    rows += [
        _failure(seed, excess=0.8, clip=0.1, pnav_capture=False, eta_capture=False)
        for seed in range(30, 35)
    ]
    rows += [
        _failure(seed, excess=0.1, clip=0.1, pnav_capture=False, eta_capture=False)
        for seed in range(35, 40)
    ]
    result = audit._classify(rows)
    assert result["status"] == "NAV_ETA_DISAGREEMENT"
    assert result["nav_eta_failures"] == 10
