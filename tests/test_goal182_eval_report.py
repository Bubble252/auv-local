from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILDER_SCRIPT = ROOT / "scripts" / "build_goal182_eval_report.py"
AUDITOR_SCRIPT = ROOT / "scripts" / "audit_goal182_artifacts.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(run_dir: Path, *, seed_start: int) -> None:
    run_dir.mkdir(parents=True)
    episodes = [
        {
            "episode_id": seed_start,
            "seed": seed_start,
            "success": True,
            "timeout": False,
            "safety": False,
            "goal_plane_miss": False,
            "material_saturation_rate": 0.0,
            "path_steps": 10,
            "pre_goal_path_steps": 8,
            "path_horizontal_rmse_m": 0.1,
            "path_vertical_rmse_m": 0.2,
            "path_three_d_rmse_m": 0.3,
            "pre_goal_3d_rmse_m": 0.25,
            "path_completion_ratio": 1.0,
        },
        {
            "episode_id": seed_start + 1,
            "seed": seed_start + 1,
            "success": False,
            "timeout": True,
            "safety": False,
            "goal_plane_miss": False,
            "material_saturation_rate": 0.1,
            "path_steps": 20,
            "pre_goal_path_steps": 12,
            "path_horizontal_rmse_m": 0.2,
            "path_vertical_rmse_m": 0.4,
            "path_three_d_rmse_m": 0.5,
            "pre_goal_3d_rmse_m": 0.5,
            "path_completion_ratio": 0.5,
        },
    ]
    metrics = {
        "success_rate": 0.5,
        "timeout_rate": 0.5,
        "safety_termination_rate": 0.0,
        "goal_plane_miss_rate": 0.0,
        "path_horizontal_rmse_m": math.sqrt((0.1**2 + 0.2**2) / 2.0),
        "path_vertical_rmse_m": math.sqrt((0.2**2 + 0.4**2) / 2.0),
        "path_three_d_rmse_m": math.sqrt((0.3**2 + 0.5**2) / 2.0),
        "pre_goal_3d_rmse_m": math.sqrt((8 * 0.25**2 + 12 * 0.5**2) / 20.0),
        "path_completion_ratio_mean": 0.75,
        "material_saturation_rate": 0.05,
    }
    summary = {
        "schema_version": "auv_pc_ppo_evaluation.v1",
        "checkpoint_sha256": "abc123",
        "metrics": metrics,
        "episodes": episodes,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    np.savez_compressed(
        run_dir / "trace.npz",
        episode_id=np.asarray([seed_start, seed_start + 1], dtype=np.int64),
        seed=np.asarray([seed_start, seed_start + 1], dtype=np.int64),
        active=np.asarray([True, True]),
        path_error_3d=np.asarray([0.1, 0.2]),
        path_horizontal_error=np.asarray([0.05, 0.1]),
        path_vertical_error=np.asarray([0.01, 0.02]),
    )


def _manifest(root: Path) -> Path:
    jobs = []
    for index, pressure in enumerate(("nominal", "current10")):
        run_dir = root / f"run-{pressure}"
        _write_run(run_dir, seed_start=90001 + 2 * index)
        jobs.append(
            {
                "arm_tag": "arm",
                "training_seed": 86201,
                "scenario": f"p-{pressure}__r-P0__l35__v1p2__g-default",
                "output": str(run_dir),
            }
        )
    path = root / "run_manifest.json"
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    return path


def test_build_report_writes_pressure_separated_canonical_artifacts(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _manifest(tmp_path)

    manifest, rows, markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )
    builder._write_outputs(tmp_path, manifest, rows, markdown)

    assert manifest["expected_cells"] == 2
    assert manifest["valid_cells"] == 2
    assert {row["pressure_profile"] for row in rows} == {"nominal", "current10"}
    assert "## Pressure: nominal" in markdown
    assert "## Pressure: current10" in markdown
    assert (tmp_path / "report_rows.jsonl").is_file()


def test_build_report_rejects_retired_strict125_radius(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["jobs"][0]["scenario"] = "p-nominal__r-P0__l35__v1p2__g-strict125"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, rows, _markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )

    assert rows[0]["failure_codes"] == ["INVALID_GOAL182_RADIUS"]
    assert manifest["valid_cells"] == 1


def test_build_report_rejects_missing_radius(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["jobs"][0]["scenario"] = "p-nominal__r-P0__l35__v1p2"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, rows, _markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )

    assert rows[0]["failure_codes"] == ["INVALID_GOAL182_RADIUS"]
    assert manifest["valid_cells"] == 1


def test_build_report_rejects_missing_paired_zero_for_bcres(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for job in payload["jobs"]:
        job["learning_composition"] = "BCRES"
        job["is_zero_baseline"] = False
        job["zero_baseline_status"] = "INVALID_MISSING_PAIRED_ZERO"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, rows, _markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )

    assert all("INVALID_INCOMPLETE_EVAL" in row["failure_codes"] for row in rows)
    assert manifest["valid_cells"] == 0


def _paired_manifest(root: Path, *, zero_seed_start: int) -> Path:
    scenario = "p-nominal__r-P0__l35__v1p2__g-default"
    policy_dir = root / "policy"
    zero_dir = root / "zero"
    _write_run(policy_dir, seed_start=90001)
    _write_run(zero_dir, seed_start=zero_seed_start)
    jobs = [
        {
            "arm_tag": "bcres",
            "training_seed": 86201,
            "scenario": scenario,
            "output": str(policy_dir),
            "learning_composition": "BCRES",
            "is_zero_baseline": False,
            "zero_baseline_status": "PAIRED",
            "paired_zero_tag": "zero_bcres",
        },
        {
            "arm_tag": "zero_bcres",
            "scenario": scenario,
            "output": str(zero_dir),
            "learning_composition": "BCRES",
            "is_zero_baseline": True,
            "zero_baseline_status": "ZERO_BASELINE",
        },
    ]
    path = root / "run_manifest.json"
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    return path


def test_paired_zero_requires_identical_eval_seeds(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _paired_manifest(tmp_path, zero_seed_start=90001)

    manifest, rows, _markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )

    policy = next(row for row in rows if not row["is_zero_baseline"])
    assert policy["valid"] is True
    assert manifest["paired_zero_cells"] == 1


def test_paired_zero_seed_mismatch_invalidates_policy_cell(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _paired_manifest(tmp_path, zero_seed_start=90009)

    manifest, rows, _markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )

    policy = next(row for row in rows if not row["is_zero_baseline"])
    assert "INVALID_INCOMPLETE_EVAL:paired_zero_seed_mismatch" in policy["failure_codes"]
    assert manifest["valid_cells"] == 0
    assert manifest["valid_zero_cells"] == 1


def test_pe2e_composition_arm_is_valid_only_when_explicitly_unpaired(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    run_dir = tmp_path / "pe2e"
    _write_run(run_dir, seed_start=90001)
    for status, expected_valid in (("ZERO_BASELINE_UNPAIRED", True), ("PAIRED", False)):
        manifest_path = tmp_path / f"run_manifest_{status}.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "jobs": [
                        {
                            "arm_tag": "pe2e_bcft",
                            "training_seed": 86201,
                            "scenario": "p-nominal__r-P0__l35__v1p2__g-default",
                            "output": str(run_dir),
                            "learning_composition": "BCFT",
                            "control_contract": "CC_PE2E",
                            "is_zero_baseline": False,
                            "zero_baseline_status": status,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        manifest, rows, _markdown = builder.build_report(
            run_manifest_path=manifest_path,
            goal_id="Goal182",
            matrix_id="G182-EVAL-TEST",
            tier="MAIN_RANKING",
            phase="formal",
            training_profile="PPO-FLAT-4096x48-E5-MB4",
            reporting_rule="report every pressure separately; never pool pressure levels before ranking",
        )
        assert rows[0]["valid"] is expected_valid
        assert manifest["valid_cells"] == int(expected_valid)
        if not expected_valid:
            assert "INVALID_ZERO_BASELINE_PROTOCOL" in rows[0]["failure_codes"]


def test_build_report_marks_pe2e_unpaired_without_invalidating_cells(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for job in payload["jobs"]:
        job["learning_composition"] = "PE2E"
        job["control_contract"] = "CC_PE2E"
        job["is_zero_baseline"] = False
        job["zero_baseline_status"] = "ZERO_BASELINE_UNPAIRED"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, rows, _markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )

    assert manifest["valid_cells"] == 2
    assert manifest["unpaired_zero_cells"] == 2
    assert all(row["zero_baseline_status"] == "ZERO_BASELINE_UNPAIRED" for row in rows)


def test_depth_report_keeps_depth_separate_from_route(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for job in payload["jobs"]:
        job["scenario"] = "p-nominal__r-R_3D_M__d500__l70__v1p2__g-default"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, rows, _markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="DEPTH_GENERALIZATION",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )

    assert manifest["valid_cells"] == 2
    assert {row["route_profile"] for row in rows} == {"R_3D_M"}
    assert {row["depth_m"] for row in rows} == {500.0}


def test_depth_report_rejects_missing_depth(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for job in payload["jobs"]:
        job["scenario"] = "p-nominal__r-R_3D_M__l70__v1p2__g-default"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, rows, _markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="DEPTH_GENERALIZATION",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )

    assert all("INVALID_GOAL182_DEPTH" in row["failure_codes"] for row in rows)
    assert manifest["valid_cells"] == 0


def test_eval_set_auditor_recomputes_report_rows_from_run_artifacts(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    auditor = _load("audit_goal182_artifacts_eval_set", AUDITOR_SCRIPT)
    manifest_path = _manifest(tmp_path)
    manifest, rows, markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )
    builder._write_outputs(tmp_path, manifest, rows, markdown)

    assert auditor.audit_eval_set(tmp_path, expected_runs=2) == []


def test_eval_set_auditor_rejects_report_metric_drift(tmp_path: Path) -> None:
    builder = _load("build_goal182_eval_report", BUILDER_SCRIPT)
    auditor = _load("audit_goal182_artifacts_eval_set", AUDITOR_SCRIPT)
    manifest_path = _manifest(tmp_path)
    manifest, rows, markdown = builder.build_report(
        run_manifest_path=manifest_path,
        goal_id="Goal182",
        matrix_id="G182-EVAL-TEST",
        tier="MAIN_RANKING",
        phase="formal",
        training_profile="PPO-FLAT-4096x48-E5-MB4",
        reporting_rule="report every pressure separately; never pool pressure levels before ranking",
    )
    builder._write_outputs(tmp_path, manifest, rows, markdown)
    rows[0]["success"] = 0.0
    (tmp_path / "report_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    errors = auditor.audit_eval_set(tmp_path)
    assert any("success cannot be recomputed" in error for error in errors)
