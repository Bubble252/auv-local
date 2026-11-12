from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_pc_s14_capability_20261104.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("summarize_pc_s14_capability", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_metrics_use_delta_prefix() -> None:
    module = _load_module()

    paired = module._paired(
        [{"train_seed": 94030, "tag": "default", "success_rate": 0.75}],
        [{"train_seed": 94030, "tag": "default", "success_rate": 0.625}],
        {"default"},
    )["default"]

    assert paired["delta_success_rate"]["mean"] == 0.125
    assert "success_rate" not in paired
