from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal99_distance_schedule.py"
)
SPEC = importlib.util.spec_from_file_location("goal99_analyzer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _episodes(successes: int) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "success": offset < successes,
            "success_pnav_tight": offset < successes,
            "success_eta_tight_ever": offset < successes,
            "success_eta_tight_at_done": offset < successes,
            "timeout": False,
            "safety": False,
        }
        for offset, seed in enumerate(range(20, 60))
    ]


def _write_goal98(
    root: Path, *, arm: str, scenario: str, scale: float, successes: int
) -> None:
    run = root / f"{arm}_{scenario}_s{int(round(scale * 100)):03d}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": _episodes(successes),
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA[arm],
                "actuator_residual_scale_override": scale,
                "metrics": {
                    "path_three_d_rmse_m": 2.70,
                    "goal_plane_miss_rate": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        run / "trace.npz",
        step=np.arange(3, dtype=np.int64),
        episode_seed=np.array([20, 20, 20], dtype=np.int64),
        goal_distance_eta_m=np.array([3.0, 2.5, 2.0], dtype=np.float64),
        terminal_kind=np.zeros(3, dtype=np.int64),
        actuator_residual_applied=np.zeros((3, 3), dtype=np.float64),
    )


def _write_goal98_all(root: Path) -> None:
    for arm in audit.ARMS:
        for scenario in audit.SCENARIOS:
            for scale in audit.CONSTANT_SCALES:
                _write_goal98(
                    root,
                    arm=arm,
                    scenario=scenario,
                    scale=scale,
                    successes=audit.EXPECTED_CONSTANT_SUCCESS[
                        (scale, arm, scenario)
                    ],
                )


def _write_schedule(
    root: Path,
    *,
    arm: str,
    scenario: str,
    schedule_id: str,
    successes: int,
    corrupt_boundary: bool = False,
) -> None:
    spec = audit.SCHEDULES[schedule_id]
    run = root / f"{arm}_{scenario}_{schedule_id}"
    run.mkdir(parents=True)
    episodes = _episodes(successes)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA[arm],
                "actuator_residual_scale_override": None,
                "actuator_residual_scale_near": spec["near"],
                "actuator_residual_scale_far": spec["far"],
                "actuator_residual_scale_switch_distance_m": spec[
                    "switch_distance_m"
                ],
                "actuator_residual_scale_schedule_enabled": True,
                "metrics": {
                    "path_three_d_rmse_m": 2.70,
                    "goal_plane_miss_rate": (
                        0.0 if "_strict125" not in run.name else 0.2
                    ),
                    "actuator_residual_magnitude_clip_rate": 0.5,
                    "actuator_residual_rate_clip_rate": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    distance = np.array([3.0, 2.5, 1.5], dtype=np.float64)
    scale = np.where(distance > spec["switch_distance_m"], spec["far"], spec["near"])
    if corrupt_boundary:
        scale = scale.copy()
        scale[0] = spec["near"]
    np.savez(
        run / "trace.npz",
        step=np.arange(3, dtype=np.int64),
        episode_seed=np.array([20, 20, 20], dtype=np.int64),
        goal_distance_eta_m=distance,
        terminal_kind=np.zeros(3, dtype=np.int64),
        actuator_residual_applied=np.zeros((3, 3), dtype=np.float64),
        actuator_residual_scale=scale,
        actuator_residual_schedule_distance_m=distance,
    )


def _write_schedule_all(
    root: Path,
    *,
    overrides: dict[str, int] | None = None,
    corrupt_group: str | None = None,
) -> None:
    overrides = overrides or {}
    for arm in audit.ARMS:
        for scenario in audit.SCENARIOS:
            group = f"{arm}_{scenario}"
            target = 31 if scenario == "default" else 20
            for schedule_id in audit.SCHEDULES:
                _write_schedule(
                    root,
                    arm=arm,
                    scenario=scenario,
                    schedule_id=schedule_id,
                    successes=overrides.get(group, target),
                    corrupt_boundary=group == corrupt_group,
                )


def test_schedule_supported_when_all_groups_reach_frozen(tmp_path: Path) -> None:
    goal98 = tmp_path / "goal98"
    goal99 = tmp_path / "goal99"
    _write_goal98_all(goal98)
    _write_schedule_all(goal99)
    gate = audit._gate(goal99, goal98, expected_seeds=set(range(20, 60)))
    assert gate["status"] == "SCHEDULE_SUPPORTED"
    assert gate["qualifying_schedule"] == "S20"


def test_schedule_partial_when_strict_remains_below_frozen(tmp_path: Path) -> None:
    goal98 = tmp_path / "goal98"
    goal99 = tmp_path / "goal99"
    _write_goal98_all(goal98)
    _write_schedule_all(goal99, overrides={"P_D3_strict125": 19})
    gate = audit._gate(goal99, goal98, expected_seeds=set(range(20, 60)))
    assert gate["status"] == "SCHEDULE_PARTIAL"
    assert gate["geometry_limited_candidates"] == []


def test_schedule_invalid_when_trace_boundary_is_wrong(tmp_path: Path) -> None:
    goal98 = tmp_path / "goal98"
    goal99 = tmp_path / "goal99"
    _write_goal98_all(goal98)
    _write_schedule_all(goal99, corrupt_group="P_D3_default")
    try:
        audit._gate(goal99, goal98, expected_seeds=set(range(20, 60)))
    except audit.AuditError as exc:
        assert "trace scale does not match schedule boundary" in str(exc)
    else:
        raise AssertionError("expected AuditError")
