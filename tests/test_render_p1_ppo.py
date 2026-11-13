from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _render_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "render_p1_rollout_2d",
        Path("scripts/render_p1_rollout_2d.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gallery_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_p1_vis_gallery",
        Path("scripts/build_p1_vis_gallery.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_resolves_fixed_curriculum_variants() -> None:
    module = _render_module()

    cases, key = module._resolve_cases("c3_gate_00")

    assert key == "c3_gate_00"
    assert key in {case.key for case in cases}
    assert {case.key for case in cases} >= {"s1", "s2", "c1_00", "c2_00", "c3_gate_00"}


def test_gallery_summary_contains_analytic_gate_metrics() -> None:
    summary_path = Path("artifacts/ppo_isaaclab/final_evaluation.json")
    if not summary_path.is_file():
        pytest.skip("IsaacLab final evaluation artifact is not included in the clean source checkout")
    module = _gallery_module()

    summary = module._evaluation_summary_html(summary_path)

    assert "解析最终评估" in summary
    assert "S1 无障碍" in summary
    assert "C3 连续避障" in summary
    assert "C3 offset gate" in summary
    assert "零碰撞、零越界、零超时" in summary
    assert "不代表 OceanSim" in summary
