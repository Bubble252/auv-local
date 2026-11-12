from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_pc_goal24_dagger_seeds.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("select_pc_goal24_dagger_seeds", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    seed: int,
    *,
    success: bool,
    safety: bool = False,
    timeout: bool = False,
    plane_miss: bool = False,
    rmse: float = 1.0,
    saturation: float = 0.0,
) -> dict:
    return {
        "seed": seed,
        "success": success,
        "safety": safety,
        "timeout": timeout,
        "goal_plane_miss": plane_miss,
        "pre_goal_3d_rmse_m": rmse,
        "material_saturation_rate": saturation,
    }


def test_selection_prioritizes_safety_and_ignores_successful_low_saturation() -> None:
    module = _load_module()
    payload = {
        "student_episodes": [
            _record(1, success=True, rmse=0.1),
            _record(2, success=False, plane_miss=True, rmse=3.0),
            _record(3, success=False, safety=True, rmse=0.5),
            _record(4, success=True, saturation=0.8, rmse=1.0),
        ]
    }

    selected = module._select(payload, 2)

    assert [row["seed"] for row in selected] == [3, 2]


def test_selection_marks_partial_pool() -> None:
    module = _load_module()
    payload = {
        "student_episodes": [
            _record(1, success=True),
            _record(2, success=False, timeout=True, rmse=2.0),
        ]
    }

    selected = module._select(payload, 10)

    assert [row["seed"] for row in selected] == [2]


def test_selection_can_exclude_seeds_from_another_scenario() -> None:
    module = _load_module()
    payload = {
        "student_episodes": [
            _record(1, success=False, plane_miss=True, rmse=4.0),
            _record(2, success=False, plane_miss=True, rmse=3.0),
            _record(3, success=False, timeout=True, rmse=1.0),
        ]
    }

    selected = module._select(payload, 2, excluded_seeds={1})

    assert [row["seed"] for row in selected] == [3, 2]
