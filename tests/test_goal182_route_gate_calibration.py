from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_SCRIPT = ROOT / "scripts" / "calibrate_goal182_route_gate.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "calibrate_goal182_route_gate", CALIBRATION_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CALIBRATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(count: int = 40) -> list[dict]:
    return [
        {
            "episode_metrics": {
                "/episode/cross_track_rmse_mean": 1.0 + 0.01 * index,
                "/episode/vertical_rmse_mean": 2.0 + 0.02 * index,
                "/episode/path_completion_mean": 0.90 - 0.001 * index,
            }
        }
        for index in range(count)
    ]


def test_calibration_uses_twenty_iteration_windows() -> None:
    module = _load()
    calibration = module.calibrate(_records(), window_size=20)
    cross = calibration["statistics"]["cross_track_rmse_max_m"]
    vertical = calibration["statistics"]["vertical_rmse_max_m"]
    completion = calibration["statistics"]["completion_ratio"]
    assert cross["iteration_count"] == 40
    assert cross["window_count"] == 21
    assert vertical["mean"] == pytest.approx(
        statistics.fmean(
            [
                statistics.fmean([2.0 + 0.02 * index for index in range(start, start + 20)])
                for start in range(21)
            ]
        )
    )
    assert completion["mean"] == pytest.approx(
        statistics.fmean(
            [
                statistics.fmean([0.90 - 0.001 * index for index in range(start, start + 20)])
                for start in range(21)
            ]
        )
    )


def test_thresholds_apply_margin_and_frozen_completion_bounds() -> None:
    module = _load()
    calibration = module.calibrate(_records(), window_size=20)
    thresholds = module.build_thresholds(
        calibration,
        cross_margin=0.10,
        vertical_margin=0.10,
        completion_floor=0.80,
        completion_ceiling=0.90,
    )
    cross = calibration["statistics"]["cross_track_rmse_max_m"]
    vertical = calibration["statistics"]["vertical_rmse_max_m"]
    assert thresholds["cross_track_rmse_max_m"] == pytest.approx(
        max(cross["mean"] * 1.10, cross["mean"] + 2.0 * cross["std"])
    )
    assert thresholds["vertical_rmse_max_m"] == pytest.approx(
        max(vertical["mean"] * 1.10, vertical["mean"] + 2.0 * vertical["std"])
    )
    completion = calibration["statistics"]["completion_ratio"]
    assert thresholds["completion_min"] == pytest.approx(
        max(0.80, min(0.90, completion["mean"] - 0.05))
    )
    assert thresholds["safety_max"] == 0.0


def test_calibration_requires_complete_telemetry() -> None:
    module = _load()
    records = _records(20)
    del records[0]["episode_metrics"]["/episode/vertical_rmse_mean"]
    with pytest.raises(ValueError, match="need at least 20"):
        module.calibrate(records, window_size=20)


def test_calibration_records_source_and_window_scheme() -> None:
    module = _load()
    calibration = module.calibrate(_records(), window_size=20, calibration_source="REFERENCE_BC")
    assert calibration["calibration_source"] == "REFERENCE_BC"
    assert calibration["window_scheme"] == "TRAINING_ROLLOUT_20_ITERATION_WINDOWS"
    assert calibration["stage_coverage"]["records_without_stage"] == 40


def test_calibration_reports_and_enforces_stage_coverage() -> None:
    module = _load()
    records = _records()
    for record in records:
        record["curriculum_stage"] = "C0"
    calibration = module.calibrate(
        records,
        window_size=20,
        required_stages=("C0",),
    )
    assert calibration["stage_coverage"] == {
        "required": ["C0"],
        "counts": {"C0": 40},
        "missing": [],
        "insufficient": [],
        "min_records_per_stage": 0,
        "records_without_stage": 0,
    }
    with pytest.raises(ValueError, match="do not cover required stages: C1"):
        module.calibrate(
            records,
            window_size=20,
            required_stages=("C0", "C1"),
        )
    with pytest.raises(ValueError, match="too few iterations for required stages: C0"):
        module.calibrate(
            records,
            window_size=20,
            required_stages=("C0",),
            min_stage_records=41,
        )


def test_frozen_table_contains_formula_margins_stats_and_record_sha(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in _records()),
        encoding="utf-8",
    )
    output = tmp_path / "goal182_route_gate_pe2e_20_3.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(CALIBRATION_SCRIPT),
            "--records",
            str(records),
            "--control-contract",
            "PE2E",
            "--calibration-source",
            "REFERENCE_BC",
            "--reference-id",
            "frozen_bc_pe2e",
            "--reference-sha256",
            "abc123",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    table = json.loads(output.with_suffix(".thresholds.json").read_text(encoding="utf-8"))
    assert table["status"] == "FROZEN"
    assert table["calibration_source"] == "REFERENCE_BC"
    assert table["window_scheme"] == "TRAINING_ROLLOUT_20_ITERATION_WINDOWS"
    assert table["reference"] == {"id": "frozen_bc_pe2e", "sha256": "abc123"}
    assert "max(mean" in table["formula"]["cross_track_rmse_max_m"]
    assert table["margins"]["cross"] == 0.10
    assert table["statistics"]["cross_track_rmse_max_m"]["window_count"] == 21
    assert table["stage_coverage"]["required"] == []
    assert table["record_files"][0]["sha256"]
    assert table["thresholds"]["completion_min"] == pytest.approx(0.8305)


def test_cli_rejects_calibration_without_required_stage_coverage(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    staged = _records()
    for record in staged:
        record["curriculum_stage"] = "C0"
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in staged),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(CALIBRATION_SCRIPT),
            "--records",
            str(records),
            "--control-contract",
            "PE2E",
            "--require-stages",
            "C0,C1",
            "--output",
            str(tmp_path / "thresholds.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "do not cover required stages: C1" in completed.stderr


def test_stage_calibration_configs_cover_c0_through_c6() -> None:
    root = ROOT / "configs" / "lowlevel" / "goal182" / "calibration"
    stages = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
    for stage in stages:
        environment = json.loads((root / f"env_{stage}.json").read_text(encoding="utf-8"))
        curriculum = environment["curriculum"]
        assert curriculum["initial_stage"] == stage
        assert curriculum["promotion_enabled"] is False
        assert curriculum["stage_mix"] == [{"stage": stage, "weight": 1.0}]
        assert environment["plant"]["goal_radius_m"] == 2.0
        assert environment["plant"]["max_episode_steps"] == 3000
        assert environment["plant"]["dt_s"] == 0.02
        for control in ("pe2e", "alosb"):
            config = json.loads(
                (root / f"reference_{control}_{stage}.json").read_text(encoding="utf-8")
            )
            assert config["environment"]["config"] == (
                f"configs/lowlevel/goal182/calibration/env_{stage}.json"
            )
            assert config["training"]["iterations"] == 100
            assert config["runner"]["save_interval"] == 100
            assert config["base_policy"]["residual_scale"] == pytest.approx(1e-6)


def test_seven_stage_sweep_satisfies_coverage_and_window_requirements() -> None:
    module = _load()
    stages = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
    records = []
    for stage in stages:
        for index in range(40):
            record = _records(1)[0]
            record["curriculum_stage"] = stage
            record["episode_metrics"]["/episode/cross_track_rmse_mean"] += 0.01 * index
            records.append(record)
    calibration = module.calibrate(
        records,
        window_size=20,
        required_stages=stages,
        min_stage_records=20,
    )
    assert calibration["stage_coverage"]["counts"] == {stage: 40 for stage in stages}
    assert calibration["stage_coverage"]["missing"] == []
    assert calibration["stage_coverage"]["insufficient"] == []
    assert calibration["statistics"]["cross_track_rmse_max_m"]["iteration_count"] == 280


def test_calibration_windows_stay_inside_each_stage_and_emit_stage_thresholds() -> None:
    module = _load()
    records = []
    for stage, base in (("C0", 1.0), ("C1", 100.0)):
        for index in range(40):
            record = _records(1)[0]
            record["curriculum_stage"] = stage
            record["episode_metrics"]["/episode/cross_track_rmse_mean"] = base + 0.01 * index
            records.append(record)
    calibration = module.calibrate(
        records,
        window_size=20,
        required_stages=("C0", "C1"),
        min_stage_records=20,
    )
    assert calibration["statistics_by_stage"]["C0"]["cross_track_rmse_max_m"]["mean"] < 2.0
    assert calibration["statistics_by_stage"]["C1"]["cross_track_rmse_max_m"]["mean"] > 99.0
    assert calibration["statistics"]["cross_track_rmse_max_m"]["window_count"] == 42
    thresholds = module.build_thresholds_by_stage(
        calibration,
        cross_margin=0.10,
        vertical_margin=0.10,
    )
    assert set(thresholds) == {"C0", "C1"}
    assert thresholds["C0"]["cross_track_rmse_max_m"] < thresholds["C1"]["cross_track_rmse_max_m"]
