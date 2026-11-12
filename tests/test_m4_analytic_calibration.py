from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import calibrate_p1_analytic_sensor as calibration  # pyright: ignore[reportMissingImports]  # noqa: E402


def test_pairing_pose_set_covers_required_scenarios_and_ranges() -> None:
    poses = calibration._all_poses()
    assert len(poses["s1"]) >= 100
    assert len(poses["s2"]) >= 400

    s2 = poses["s2"]
    categories = {pose.category for pose in s2}
    assert "start" in categories
    assert any(category.startswith("approach_") for category in categories)
    assert any(category.startswith("bypass_") for category in categories)

    nominal = np.asarray(
        [pose.nominal_surface_m for pose in s2 if pose.nominal_surface_m is not None],
        dtype=np.float64,
    )
    assert nominal.min() <= 2.0 + 1e-9
    assert nominal.max() >= 20.0 - 1e-9
    assert any("b-65.0" in pose.pose_id for pose in s2)
    assert any("b+65.0" in pose.pose_id for pose in s2)


def _row(
    *,
    scenario: str,
    pose_id: str,
    analytic_range_m: float,
    ocean_range_m: float,
    analytic_saturated: bool,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "pose_id": pose_id,
        "analytic_range_m": analytic_range_m,
        "ocean_range_m": ocean_range_m,
        "abs_error_m": abs(ocean_range_m - analytic_range_m),
        "dangerous_miss": bool(ocean_range_m < 19.9 and analytic_range_m >= 19.9),
        "analytic_saturated": analytic_saturated,
    }


def test_equivalence_gate_passes_clean_rows_and_rejects_dangerous_miss() -> None:
    args = argparse.Namespace(local_range_m=20.0)
    observation_contract = {"all_checks_pass": True}
    clean_rows = [
        _row(
            scenario="s1",
            pose_id="s1_0",
            analytic_range_m=20.0,
            ocean_range_m=20.0,
            analytic_saturated=True,
        ),
        _row(
            scenario="s2",
            pose_id="s2_0",
            analytic_range_m=10.0,
            ocean_range_m=10.05,
            analytic_saturated=False,
        ),
    ]
    summary = calibration._summarize(
        clean_rows,
        args,
        observation_contract=observation_contract,
    )
    assert summary["gates"]["all"] is True
    assert summary["metrics"]["dangerous_miss_count"] == 0

    miss_rows = [
        *clean_rows,
        _row(
            scenario="s2",
            pose_id="s2_1",
            analytic_range_m=20.0,
            ocean_range_m=5.0,
            analytic_saturated=True,
        ),
    ]
    miss_summary = calibration._summarize(
        miss_rows,
        args,
        observation_contract=observation_contract,
    )
    assert miss_summary["gates"]["no_dangerous_miss"] is False
    assert miss_summary["gates"]["all"] is False
    assert miss_summary["metrics"]["dangerous_miss_count"] == 1


def test_equivalence_gate_requires_observation_contract() -> None:
    args = argparse.Namespace(local_range_m=20.0)
    rows = [
        _row(
            scenario="s1",
            pose_id="s1_0",
            analytic_range_m=20.0,
            ocean_range_m=20.0,
            analytic_saturated=True,
        )
    ]
    summary = calibration._summarize(
        rows,
        args,
        observation_contract={"all_checks_pass": False},
    )
    assert summary["gates"]["observation_contract"] is False
    assert summary["gates"]["all"] is False


def test_dry_run_writes_pose_and_analytic_artifacts(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output=tmp_path,
        local_range_m=20.0,
        max_range_m=120.0,
    )
    poses = calibration._all_poses()
    assert calibration._dry_run(poses, args) == 0
    assert (tmp_path / "dry_run.json").is_file()
    assert (tmp_path / "dry_run_poses.jsonl").is_file()
    assert (tmp_path / "dry_run_analytic.csv").is_file()
    summary = json.loads((tmp_path / "dry_run.json").read_text(encoding="utf-8"))
    assert summary["observation_contract"]["shape"] == [17]
    assert summary["observation_contract"]["all_checks_pass"] is True
