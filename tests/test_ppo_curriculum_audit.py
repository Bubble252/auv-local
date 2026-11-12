from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "audit_p1_curriculum_stage",
        Path("scripts/audit_p1_curriculum_stage.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gates() -> dict[str, Any]:
    return json.loads(Path("configs/ppo/isaaclab/final_gates.json").read_text(encoding="utf-8"))


def _scenario(successes: int, episodes: int) -> dict[str, Any]:
    return {
        "episodes": episodes,
        "successes": successes,
        "mean_time_to_goal_s": 10.0,
        "mean_min_clearance_m": 1.0,
    }


def _eval_summary(
    stage: str,
    *,
    successes: int,
    episodes: int,
    checkpoint_sha256: str,
    gate_successes: int = 2,
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "checkpoint_sha256": checkpoint_sha256,
        "gates": {
            "pass": True,
            "action_saturation_yaw": 0.1,
            "both_action_saturation": 0.02,
        },
        "scenarios": {
            "s1": _scenario(3, 3),
            "s2": _scenario(3, 3),
            stage: _scenario(successes, episodes),
        },
        "scenario_instances": {
            "s1/s1": _scenario(3, 3),
            "s2/s2": _scenario(3, 3),
            "c3/c3_gate_00": _scenario(gate_successes, 3),
        },
    }


def _write_inputs(tmp_path: Path, stage: str = "c3") -> tuple[Path, Path]:
    best = tmp_path / "checkpoint_best.pt"
    best.write_bytes(b"best")
    module = _module()
    sha = module._sha256(best)
    training = {
        "status": "PASS",
        "iterations_completed": 5000,
        "max_iterations": 5000,
        "num_envs": 4096,
        "seed": 0,
        "config_sha256": "config",
        "eval_spec_sha256": "eval-spec",
        "eval_errors": [],
        "model_parameters_finite": True,
        "losses_finite": True,
        "deterministic_action_finite": True,
        "best_iteration": 4000,
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha,
        "evaluation_gate": "FAIL_GATE",
        "evaluation_count": 100,
    }
    evaluation = _eval_summary(
        stage,
        successes=19 if stage == "c3" else 22,
        episodes=21 if stage == "c3" else 24,
        checkpoint_sha256=sha,
    )
    training_path = tmp_path / "training_summary.json"
    eval_path = tmp_path / "eval_summary.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    eval_path.write_text(json.dumps(evaluation), encoding="utf-8")
    return training_path, eval_path


def test_curriculum_audit_passes_complete_stage(tmp_path: Path) -> None:
    training, evaluation = _write_inputs(tmp_path)

    audit = _module().build_audit(
        stage="c3",
        training_path=training,
        eval_path=evaluation,
        gates=_gates(),
    )

    assert audit["status"] == "PASS"
    assert audit["best_iteration"] == 4000
    assert audit["diagnostics"]["final_evaluation_gate"] == "FAIL_GATE"
    assert audit["failed_checks"] == []


def test_curriculum_audit_fails_on_eval_checkpoint_mismatch(tmp_path: Path) -> None:
    training, evaluation = _write_inputs(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "wrong"
    evaluation.write_text(json.dumps(payload), encoding="utf-8")

    audit = _module().build_audit(
        stage="c3",
        training_path=training,
        eval_path=evaluation,
        gates=_gates(),
    )

    assert audit["status"] == "FAIL"
    assert "best_checkpoint_sha_matches_eval" in audit["failed_checks"]


def test_curriculum_audit_fails_on_low_c3_gate_variant(tmp_path: Path) -> None:
    training, evaluation = _write_inputs(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["scenario_instances"]["c3/c3_gate_00"] = _scenario(1, 3)
    evaluation.write_text(json.dumps(payload), encoding="utf-8")

    audit = _module().build_audit(
        stage="c3",
        training_path=training,
        eval_path=evaluation,
        gates=_gates(),
    )

    assert audit["status"] == "FAIL"
    assert "c3_gate_variant_pass" in audit["failed_checks"]
