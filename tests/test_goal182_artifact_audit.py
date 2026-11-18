from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_goal182_artifacts.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_goal182_artifacts", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    episodes = [
        {
            "episode_id": 90001,
            "seed": 90001,
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
            "episode_id": 90002,
            "seed": 90002,
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
    }
    summary = {
        "schema_version": "auv_pc_ppo_evaluation.v1",
        "metrics": metrics,
        "episodes": episodes,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    np.savez_compressed(
        run_dir / "trace.npz",
        episode_id=np.asarray([90001, 90002], dtype=np.int64),
        seed=np.asarray([90001, 90002], dtype=np.int64),
        active=np.asarray([True, True]),
        path_error_3d=np.asarray([0.1, 0.2]),
        path_horizontal_error=np.asarray([0.05, 0.1]),
        path_vertical_error=np.asarray([0.01, 0.02]),
    )


def _write_goal184_train_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "final.pt"
    checkpoint.write_bytes(b"goal184-final-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    records = [
        {
            "iteration": index,
            "curriculum_stage": "C0" if index == 1 else "C6",
            "losses": {"policy": 0.1, "value": 0.2},
            "losses_finite": True,
            "learning_rate": 1e-5,
            "mean_noise_std": 0.05,
            "policy_update_metrics": {
                "approx_kl": 0.003,
                "clip_fraction": 0.02,
                "ratio_mean": 1.0,
                "ratio_max": 1.5,
            },
        }
        for index in (1, 2)
    ]
    manifest = {
        "status": "completed",
        "training_profile": "PPO-FLAT-4096x48-E5-MB4",
        "curriculum_final_stage": "C6",
        "curriculum_current_gate": "TRAINING_ROLLOUT_20_3",
        "curriculum": {
            "enabled": True,
            "manifest_filename": "curriculum_episodes.jsonl",
        },
        "iteration_records": records,
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_sha256": checkpoint_sha,
        "normalization_mode": "empirical",
        "final_checkpoint_metadata": {
            "pc_stage": "C6",
            "normalization": {"mode": "empirical"},
            "pc": {
                "normalization_mode": "empirical",
                "normalization_state": {"count": 1},
                "actor_normalization_schema_hash": "a" * 64,
                "critic_normalization_schema_hash": "b" * 64,
            },
        },
    }
    (run_dir / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_dir / "curriculum_episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "auv_pc_curriculum_episode_manifest.v1",
                    "stage": f"C{index}",
                    "depth_m": 10.0,
                    "speed_mps": 1.0,
                    "current_speed_mps": 0.1,
                }
            )
            + "\n"
            for index in range(7)
        ),
        encoding="utf-8",
    )


def test_goal182_artifact_audit_accepts_recomputable_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    assert _module().audit_run(run_dir, expected_episodes=2) == []


def test_goal182_artifact_audit_requires_episode_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    (run_dir / "episodes.jsonl").unlink()

    errors = _module().audit_run(run_dir)
    assert any("episodes.jsonl" in error and "missing" in error for error in errors)


def test_goal182_artifact_audit_rejects_trace_id_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    trace = np.load(run_dir / "trace.npz", allow_pickle=False)
    payload = {name: trace[name] for name in trace.files}
    trace.close()
    payload["episode_id"] = np.asarray([90001, 99999], dtype=np.int64)
    np.savez_compressed(run_dir / "trace.npz", **payload)

    errors = _module().audit_run(run_dir)
    assert any("episode_id set differs" in error for error in errors)


def test_goal184_training_audit_accepts_complete_curriculum_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "goal184" / "pe2e_bcft_seed86201"
    _write_goal184_train_run(run_dir)

    errors, summary = _module().audit_training_run(run_dir, expected_iterations=2)
    assert errors == []
    assert summary["stage_counts"] == {f"C{index}": 1 for index in range(7)}
    assert summary["iterations"]["records"] == 2


def test_goal184_training_audit_rejects_missing_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_goal184_train_run(run_dir)
    episodes = run_dir / "curriculum_episodes.jsonl"
    rows = episodes.read_text(encoding="utf-8").splitlines()
    episodes.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    errors, _summary = _module().audit_training_run(run_dir, expected_iterations=2)
    assert any("stage C6 has 0 episodes" in error for error in errors)


def test_goal184_training_audit_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_goal184_train_run(run_dir)
    manifest_path = run_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["iteration_records"][0]["policy_update_metrics"]["approx_kl"] = float("inf")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors, _summary = _module().audit_training_run(run_dir, expected_iterations=2)
    assert any("invalid policy metrics" in error for error in errors)


def test_goal184_training_audit_rejects_gate_and_checkpoint_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_goal184_train_run(run_dir)
    manifest_path = run_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["curriculum_current_gate"] = "LEGACY_EVAL_INTERVAL"
    manifest["final_checkpoint_metadata"]["pc"]["normalization_state"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors, _summary = _module().audit_training_run(run_dir, expected_iterations=2)
    assert any("curriculum_current_gate" in error for error in errors)
    assert any("normalization_state" in error for error in errors)
