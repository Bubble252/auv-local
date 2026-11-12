from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_pc_goal24_stress_mixes.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_pc_goal24_stress_mixes", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stress_mix_plan_preserves_total_budget() -> None:
    module = _load_module()
    for arm, plan in module.PLAN.items():
        total = sum(
            values
            for group in ("teacher", "targeted", "stress")
            for values in plan[group].values()
        )
        assert total == module.EXPECTED_TOTAL_SAMPLES, arm
