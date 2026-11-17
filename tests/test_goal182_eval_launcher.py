from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_goal182_eval.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_goal182_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_dry_run_wires_protocol_and_disables_legacy_analyzer(tmp_path: Path) -> None:
    config = tmp_path / "env.json"
    config.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"scenarios": [{"id": "p-nominal__r-P0__l35__v1p2__g-default", "config": str(config)}]}
        ),
        encoding="utf-8",
    )
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps(
            {
                "schema_version": "auv_pc_standing_eval_arms.v1",
                "arms": [
                    {
                        "tag": "zero",
                        "evaluator": "sac",
                        "training_config": str(config),
                        "action_mode": "zero-residual",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--arms",
            str(arms),
            "--output-root",
            str(tmp_path / "eval"),
            "--matrix-id",
            "G182-EVAL",
            "--tier",
            "MAIN_RANKING",
            "--phase",
            "smoke",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PASS goal182 eval matrix" in completed.stdout
    assert "--no-analyze" in completed.stdout


def test_formal_phase_rejects_draft_matrix(tmp_path: Path) -> None:
    matrix = json.loads(
        (ROOT / "configs" / "governance" / "goal182_eval_matrix.json").read_text(encoding="utf-8")
    )
    matrix["matrix_status"] = "DRAFT"
    for tier in matrix["tiers"]:
        tier["status"] = "DRAFT"
    draft = tmp_path / "draft_eval_matrix.json"
    draft.write_text(json.dumps(matrix), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--arms",
            str(tmp_path / "arms.json"),
            "--output-root",
            str(tmp_path / "eval"),
            "--eval-matrix",
            str(draft),
            "--matrix-id",
            "G182-EVAL",
            "--tier",
            "MAIN_RANKING",
            "--phase",
            "formal",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "FROZEN matrix" in completed.stderr


def test_launcher_rejects_retired_strict125_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "axes": {"goal_radii": ["default", "strict125"]},
                "scenarios": [
                    {
                        "id": "p-nominal__r-P0__l35__v1p2__g-strict125",
                        "config": str(tmp_path / "env.json"),
                        "axes": {"goal_radius": "strict125", "goal_radius_m": 1.25},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps(
            {
                "schema_version": "auv_pc_standing_eval_arms.v1",
                "arms": [
                    {
                        "tag": "zero",
                        "evaluator": "sac",
                        "training_config": str(tmp_path / "env.json"),
                        "action_mode": "zero-residual",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--arms",
            str(arms),
            "--output-root",
            str(tmp_path / "eval"),
            "--matrix-id",
            "G182-EVAL",
            "--tier",
            "MAIN_RANKING",
            "--phase",
            "smoke",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "retired" in completed.stderr
    assert "strict125" in completed.stderr


def test_launcher_rejects_manifest_without_goal_radius(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": "p-nominal__r-P0__l35__v1p2",
                        "config": str(tmp_path / "env.json"),
                        "axes": {"path_family": "P0"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps(
            {
                "schema_version": "auv_pc_standing_eval_arms.v1",
                "arms": [
                    {
                        "tag": "zero",
                        "evaluator": "sac",
                        "training_config": str(tmp_path / "env.json"),
                        "action_mode": "zero-residual",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--arms",
            str(arms),
            "--output-root",
            str(tmp_path / "eval"),
            "--matrix-id",
            "G182-EVAL",
            "--tier",
            "MAIN_RANKING",
            "--phase",
            "smoke",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "missing goal radius" in completed.stderr


def test_runner_command_contains_canonical_inputs(tmp_path: Path) -> None:
    module = _module()
    args = module._parse_args.__globals__["argparse"].Namespace(
        python=sys.executable,
        manifest=tmp_path / "m.json",
        arms=tmp_path / "a.json",
        output_root=tmp_path / "out",
        device="cuda",
        dtype="float64",
        gpus="0 1",
        max_parallel=2,
        zero_tag="zero",
        seed_start=90001,
        seed_count=3,
        timeout_s=None,
        sb3_root=None,
        tags=["arm"],
        scenarios=["p-nominal"],
        overwrite=False,
        skip_existing=True,
        dry_run=True,
        legacy_analyze=False,
    )

    command = module._runner_command(args)

    assert "--manifest" in command
    assert "--arms" in command
    assert "--no-analyze" in command
    assert "--dry-run" in command
    assert command[command.index("--seed-start") + 1] == "90001"


def test_depth_tier_rejects_manifest_without_absolute_depth(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": "p-nominal__r-R_3D_M__l70__v1p2__g-default",
                        "config": str(tmp_path / "env.json"),
                        "axes": {"route_profile": "R_3D_M", "goal_radius": "default"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps(
            {
                "schema_version": "auv_pc_standing_eval_arms.v1",
                "arms": [
                    {
                        "tag": "zero",
                        "evaluator": "sac",
                        "training_config": str(tmp_path / "env.json"),
                        "action_mode": "zero-residual",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--arms",
            str(arms),
            "--output-root",
            str(tmp_path / "eval"),
            "--matrix-id",
            "G182-EVAL",
            "--tier",
            "DEPTH_GENERALIZATION",
            "--phase",
            "smoke",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "missing depth_m" in completed.stderr


def test_launcher_rejects_bcres_without_paired_zero(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": "p-nominal__r-P0__l35__v1p2__g-default",
                        "config": str(tmp_path / "env.json"),
                        "axes": {"goal_radius": "default"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps(
            {
                "schema_version": "auv_pc_standing_eval_arms.v1",
                "arms": [
                    {
                        "tag": "bcres",
                        "evaluator": "ppo",
                        "training_config": str(tmp_path / "train.json"),
                        "checkpoint": str(tmp_path / "policy.pt"),
                        "learning_composition": "BCRES",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--arms",
            str(arms),
            "--output-root",
            str(tmp_path / "eval"),
            "--matrix-id",
            "G182-EVAL",
            "--tier",
            "MAIN_RANKING",
            "--phase",
            "smoke",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "missing paired_zero_tag" in completed.stderr
