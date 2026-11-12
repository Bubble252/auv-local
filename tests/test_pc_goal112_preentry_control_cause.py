from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "goal112_preentry_control_cause",
    SCRIPTS / "analyze_pc_goal112_preentry_control_cause.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _trace(*, command_offset: float = 0.0, actual_offset: float = 0.0) -> dict[str, np.ndarray]:
    steps = 40
    distance = 3.0 - 0.05 * np.arange(steps, dtype=np.float64)
    time_s = np.arange(steps, dtype=np.float64) * 0.1
    eta = np.zeros((steps, 6), dtype=np.float64)
    eta[:, 0] = distance
    eta[:, 5] = math.pi + actual_offset
    goal = np.zeros((steps, 3), dtype=np.float64)
    command = np.zeros((steps, 3), dtype=np.float64)
    command[:, 0] = math.pi + command_offset
    command[:, 1] = 0.0
    command[:, 2] = 1.5
    heading_error = audit._wrap_angle(command[:, 0] - eta[:, 5])
    return {
        "time_s": time_s,
        "goal_distance_eta_m": distance,
        "eta": eta,
        "goal_ned_m": goal,
        "command": command,
        "ground_speed_mps": np.full(steps, 1.5, dtype=np.float64),
        "heading_error_rad": np.asarray(heading_error, dtype=np.float64),
        "pitch_error_rad": np.zeros(steps, dtype=np.float64),
        "actuator_residual_effective": np.zeros((steps, 3), dtype=np.float64),
        "material_saturated": np.zeros(steps, dtype=bool),
        "guidance_terminal_phase": np.ones(steps, dtype=np.float64),
    }


def test_wrap_angle_and_circular_mean_handle_pi_boundary() -> None:
    assert audit._wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    value = audit._circular_mean(np.asarray([math.pi - 0.1, -math.pi + 0.1]))
    assert abs(audit._wrap_angle(value - math.pi)) < 1e-9


@pytest.mark.parametrize(
    ("command_offset", "actual_offset", "expected"),
    (
        (0.0, 0.0, "ALIGNED_CONTROL_POSITION_OFFSET"),
        (0.5, 0.5, "REFERENCE_BAD_ONLY"),
        (0.0, -0.5, "TRACKING_BAD_ONLY"),
        (0.5, 0.0, "MIXED_REFERENCE_AND_TRACKING"),
    ),
)
def test_control_cause_four_way_partition(
    command_offset: float,
    actual_offset: float,
    expected: str,
) -> None:
    arrays = audit._validate_arrays(_trace(command_offset=command_offset, actual_offset=actual_offset), seed=20)
    result = audit._control_at_shell(arrays, shell_m=2.0, seed=20)
    assert result["entered"] is True
    assert result["cause"] == expected


def test_control_threshold_is_inclusive() -> None:
    result = audit._classify_control(
        command_los_heading_rad=0.35,
        command_los_pitch_rad=0.0,
        tracking_heading_rad=0.0,
        tracking_pitch_rad=0.0,
    )
    assert result["reference_bad"] is True
    assert result["cause"] == "REFERENCE_BAD_ONLY"


def test_tracking_consistency_guard_rejects_mismatch() -> None:
    trace = _trace()
    trace["heading_error_rad"][20] = 0.2
    arrays = audit._validate_arrays(trace, seed=21)
    with pytest.raises(audit.AuditError, match="tracking errors disagree"):
        audit._control_at_shell(arrays, shell_m=2.0, seed=21)


def test_episode_control_metrics_records_earliest_bad_shell() -> None:
    trace = _trace(command_offset=0.5, actual_offset=0.5)
    metrics = audit._episode_control_metrics(trace, seed=22)
    assert metrics["shells"]["3.0"]["cause"] == "REFERENCE_BAD_ONLY"
    assert metrics["shells"]["2.5"]["cause"] == "REFERENCE_BAD_ONLY"
    assert metrics["shells"]["2.0"]["cause"] == "REFERENCE_BAD_ONLY"
    assert metrics["earliest_bad_shell_m"] == 3.0


def _record(cause: str, earliest: float | None = None) -> dict[str, object]:
    return {"primary": {"cause": cause}, "earliest_bad_shell_m": earliest}


def _summary(counts: dict[str, int]) -> dict[str, object]:
    total = sum(counts.values())
    fractions = {cause: counts.get(cause, 0) / total for cause in audit.CAUSES}
    reference = counts.get("REFERENCE_BAD_ONLY", 0) + counts.get("MIXED_REFERENCE_AND_TRACKING", 0)
    tracking = counts.get("TRACKING_BAD_ONLY", 0) + counts.get("MIXED_REFERENCE_AND_TRACKING", 0)
    return {
        "n": total,
        "counts": {cause: counts.get(cause, 0) for cause in audit.CAUSES},
        "fractions": fractions,
        "reference_involved_n": reference,
        "reference_involved_fraction": reference / total,
        "tracking_involved_n": tracking,
        "tracking_involved_fraction": tracking / total,
    }


@pytest.mark.parametrize(
    ("counts", "status"),
    (
        (
            {
                "REFERENCE_BAD_ONLY": 8,
                "TRACKING_BAD_ONLY": 0,
                "MIXED_REFERENCE_AND_TRACKING": 0,
                "ALIGNED_CONTROL_POSITION_OFFSET": 3,
            },
            "GUIDANCE_REFERENCE_LIMITED",
        ),
        (
            {
                "REFERENCE_BAD_ONLY": 0,
                "TRACKING_BAD_ONLY": 8,
                "MIXED_REFERENCE_AND_TRACKING": 0,
                "ALIGNED_CONTROL_POSITION_OFFSET": 3,
            },
            "TRACKING_LIMITED",
        ),
        (
            {
                "REFERENCE_BAD_ONLY": 2,
                "TRACKING_BAD_ONLY": 2,
                "MIXED_REFERENCE_AND_TRACKING": 6,
                "ALIGNED_CONTROL_POSITION_OFFSET": 1,
            },
            "JOINT_REFERENCE_AND_TRACKING",
        ),
        (
            {
                "REFERENCE_BAD_ONLY": 3,
                "TRACKING_BAD_ONLY": 1,
                "MIXED_REFERENCE_AND_TRACKING": 1,
                "ALIGNED_CONTROL_POSITION_OFFSET": 6,
            },
            "ALIGNED_CONTROL_POSITION_OFFSET",
        ),
        (
            {
                "REFERENCE_BAD_ONLY": 4,
                "TRACKING_BAD_ONLY": 4,
                "MIXED_REFERENCE_AND_TRACKING": 0,
                "ALIGNED_CONTROL_POSITION_OFFSET": 3,
            },
            "AMBIGUOUS_HANDOFF",
        ),
    ),
)
def test_gate_frozen_branches(counts: dict[str, int], status: str) -> None:
    gate = audit._classify_gate(_summary(counts))
    assert gate["status"] == status
    assert gate["primary_n"] == 11


def test_gate_rejects_wrong_primary_size() -> None:
    with pytest.raises(audit.AuditError, match="must contain 11"):
        audit._classify_gate(_summary({"ALIGNED_CONTROL_POSITION_OFFSET": 10}))


def test_episode_contract_rejects_wrong_success_count(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    episodes = [
        {"seed": seed, "success": seed < 52, "timeout": False, "safety": False}
        for seed in audit.DEV_SEEDS
    ]
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA,
                "actuator_residual_scale_override": audit.EXPECTED_EVAL_SCALE,
                "actuator_residual_channel_gains": [1.0, 1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(audit.AuditError, match="success 32/40"):
        audit._load_episodes(run, scenario="default", profile="C0")
