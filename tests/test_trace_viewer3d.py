from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")

from auv_core.contracts import Obstacle

from auv_local.trace_viewer3d import (
    TraceDocument,
    build_figure,
    group_by_scenario,
    render_trace_directory,
)


def _sample(t_s: float, x_m: float, *, replanned: bool) -> dict[str, object]:
    position = [x_m, 0.0, -5.0]
    return {
        "state": {
            "t_s": t_s,
            "position_ned_m": position,
            "linear_velocity_body_mps": [1.0, 0.0, 0.0],
        },
        "reference": {"position_ned_m": [x_m + 0.1, 0.0, -5.0]},
        "command": {
            "rudder_angle_rad": 0.02,
            "stern_plane_angle_rad": -0.01,
            "propeller_rpm": 600.0,
        },
        "local_path_ned_m": [position, [x_m + 0.2, 0.0, -5.0]],
        "replanned": replanned,
    }


def _trace_payload(
    *,
    case_name: str = "s1_seed0",
    scenario_name: str = "m2_s1_empty",
    seed: int = 0,
) -> dict[str, object]:
    return {
        "case_name": case_name,
        "scenario_name": scenario_name,
        "seed": seed,
        "termination_reason": "goal",
        "initial_state": {"position_ned_m": [-8.0, 0.0, -5.0]},
        "goal_ned_m": [15.0, 0.0, -5.0],
        "samples": [_sample(0.02, -7.98, replanned=True), _sample(0.04, -7.96, replanned=False)],
    }


def _box() -> Obstacle:
    return Obstacle(
        obstacle_id="box-1",
        kind="box",
        center_ned_m=np.array([0.0, 0.0, -5.0], dtype=np.float64),
        half_extents_m=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )


def test_trace_document_builds_interactive_3d_figure() -> None:
    document = TraceDocument.from_dict(_trace_payload())

    figure = build_figure([document], [_box()])

    layout = figure.to_dict()["layout"]
    assert layout["scene"]["dragmode"] == "orbit"
    assert "3D trajectory review" in str(layout["title"]["text"])
    data = figure.to_dict()["data"]
    actual = next(trace for trace in data if str(trace["name"]).startswith("seed 0: actual"))
    assert list(actual["x"][:2]) == pytest.approx([-8.0, -7.98])
    assert list(actual["z"][:2]) == pytest.approx([5.0, 5.0])
    local_plans = next(trace for trace in data if str(trace["name"]).endswith("local plans"))
    assert None in list(local_plans["x"])


def test_trace_directory_renders_one_html_per_scenario(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    output_dir = tmp_path / "3d"
    trace_dir.mkdir()
    for seed, scenario in enumerate(("m2_s1_empty", "m2_s2_single_cylinder", "m2_s3_offset_gate")):
        payload = _trace_payload(case_name=f"s{seed + 1}_seed0", scenario_name=scenario, seed=0)
        (trace_dir / f"s{seed + 1}_seed0.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    outputs = render_trace_directory(
        trace_dir,
        output_dir,
        obstacles_by_scenario={"m2_s2_single_cylinder": (_box(),)},
    )

    assert len(outputs) == 3
    assert len(group_by_scenario([TraceDocument.from_dict(_trace_payload())])) == 1
    assert (output_dir / "plotly.min.js").is_file()
    for output in outputs:
        assert output.is_file()
        assert output.stat().st_size > 1_000
        assert "plotly.min.js" in output.read_text(encoding="utf-8")
