from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "audit_p1_ppo_seed",
        Path("scripts/audit_p1_ppo_seed.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(tmp_path: Path) -> tuple[Path, Path, Path]:
    best = tmp_path / "checkpoint_best.pt"
    final = tmp_path / "checkpoint_final.pt"
    best.write_bytes(b"best")
    final.write_bytes(b"final")
    module = _module()
    summary: dict[str, Any] = {
        "status": "PASS",
        "evaluation_gate": "PASS",
        "iterations_completed": 2,
        "max_iterations": 2,
        "num_envs": 8,
        "num_steps_per_env": 2,
        "seed": 3,
        "training_s": 1.0,
        "config_sha256": "config",
        "eval_spec_sha256": "eval",
        "eval_errors": [],
        "model_parameters_finite": True,
        "losses_finite": True,
        "deterministic_action_finite": True,
        "records": [
            {"iteration": 0, "losses": {"entropy": 1.0}},
            {"iteration": 1, "losses": {"entropy": 0.5}},
        ],
        "best_iteration": 1,
        "best_score": [1.0, 2.0],
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": module._sha256(best),
        "checkpoint_final": str(final),
        "checkpoint_final_sha256": module._sha256(final),
        "checkpoint_sha256": module._sha256(final),
        "best_gates": {"pass": True},
        "evaluations": [
            {"iteration": 0, "status": "FAIL_GATE"},
            {"iteration": 1, "status": "PASS"},
        ],
    }
    path = tmp_path / "training_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path, best, final


def test_seed_audit_passes_complete_summary(tmp_path: Path) -> None:
    summary, _, _ = _write_summary(tmp_path)

    audit = _module().build_audit(summary, expect_iterations=2, expect_envs=8)

    assert audit["status"] == "PASS"
    assert audit["best_iteration"] == 1
    assert audit["last_twenty_percent"] == {
        "count": 1,
        "passes": 1,
        "failures": 0,
        "failed_iterations": [],
    }
    assert all(audit["acceptance_checks"].values())
    assert all(audit["diagnostic_checks"].values())


def test_seed_audit_fails_on_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    summary, best, _ = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["best_checkpoint_sha256"] = "wrong"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    assert best.is_file()

    audit = _module().build_audit(summary, expect_iterations=2, expect_envs=8)

    assert audit["status"] == "FAIL"
    assert audit["acceptance_checks"]["best_checkpoint_sha_matches"] is False


def test_seed_audit_fails_on_incomplete_records(tmp_path: Path) -> None:
    summary, _, _ = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["records"] = payload["records"][:1]
    summary.write_text(json.dumps(payload), encoding="utf-8")

    audit = _module().build_audit(summary, expect_iterations=2, expect_envs=8)

    assert audit["status"] == "FAIL"
    assert audit["acceptance_checks"]["records_complete"] is False


def test_seed_audit_keeps_failed_final_eval_as_diagnostic(tmp_path: Path) -> None:
    summary, _, _ = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["evaluation_gate"] = "FAIL_GATE"
    payload["evaluations"] = [
        {"iteration": 1, "status": "PASS"},
        {"iteration": 1, "status": "PASS"},
        {"iteration": 2, "status": "FAIL_GATE"},
    ]
    summary.write_text(json.dumps(payload), encoding="utf-8")

    audit = _module().build_audit(summary, expect_iterations=2, expect_envs=8)

    assert audit["status"] == "PASS"
    assert audit["selected_checkpoint_gate_pass"] is True
    assert audit["diagnostic_checks"]["final_evaluation_gate_pass"] is False
