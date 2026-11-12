"""Audit one PPO curriculum arm against the frozen G6 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one P1 PPO curriculum stage")
    parser.add_argument("--stage", choices=("c1", "c2", "c3"), required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path("configs/ppo/isaaclab/final_gates.json"),
    )
    parser.add_argument("--expect-iterations", type=int, default=5000)
    parser.add_argument("--expect-envs", type=int, default=4096)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario(summary: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, Mapping) or not isinstance(scenarios.get(key), Mapping):
        raise ValueError(f"eval summary is missing scenario {key!r}")
    return scenarios[key]


def _instance(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    instances = summary.get("scenario_instances")
    if not isinstance(instances, Mapping) or not isinstance(instances.get(name), Mapping):
        raise ValueError(f"eval summary is missing scenario instance {name!r}")
    return instances[name]


def _finite(value: Any, where: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where}: expected a finite number")
    return result


def build_audit(
    *,
    stage: str,
    training_path: Path,
    eval_path: Path,
    gates: Mapping[str, Any],
    expect_iterations: int = 5000,
    expect_envs: int = 4096,
) -> dict[str, Any]:
    if stage not in {"c1", "c2", "c3"}:
        raise ValueError(f"unsupported curriculum stage: {stage!r}")
    if gates.get("schema_version") != "auv_ppo_final_gates.v1":
        raise ValueError("unsupported final gate schema")

    training = _load_json(training_path)
    evaluation = _load_json(eval_path)
    best_path_text = training.get("best_checkpoint")
    best_path = Path(best_path_text) if isinstance(best_path_text, str) else None
    actual_best_sha = _sha256(best_path) if best_path is not None and best_path.is_file() else None
    best_sha = training.get("best_checkpoint_sha256")
    eval_sha = evaluation.get("checkpoint_sha256")
    gate = evaluation.get("gates")
    if not isinstance(gate, Mapping):
        raise ValueError("eval summary is missing gates")

    s1 = _scenario(evaluation, "s1")
    s2 = _scenario(evaluation, "s2")
    stage_metrics = _scenario(evaluation, stage)
    curriculum_gates = gates["curriculum"]
    minimum = int(curriculum_gates[f"{stage}_min_successes"])
    c3_gate: Mapping[str, Any] | None = None
    c3_gate_required: int | None = None
    if stage == "c3":
        c3_gate = _instance(evaluation, "c3/c3_gate_00")
        c3_gate_required = int(curriculum_gates["c3_gate_min_successes"])
    c3_gate_metrics: Mapping[str, Any] = c3_gate or {}

    checks = {
        "training_status_pass": training.get("status") == "PASS",
        "iterations_complete": (
            training.get("iterations_completed")
            == training.get("max_iterations")
            == expect_iterations
        ),
        "num_envs_matches": training.get("num_envs") == expect_envs,
        "training_eval_errors_empty": not training.get("eval_errors"),
        "model_parameters_finite": training.get("model_parameters_finite") is True,
        "losses_finite": training.get("losses_finite") is True,
        "deterministic_action_finite": training.get("deterministic_action_finite") is True,
        "best_checkpoint_exists": actual_best_sha is not None,
        "best_checkpoint_sha_matches_training": (
            best_sha is not None and actual_best_sha == best_sha
        ),
        "best_checkpoint_sha_matches_eval": best_sha is not None and eval_sha == best_sha,
        "eval_status_pass": evaluation.get("status") == "PASS",
        "eval_gate_pass": gate.get("pass") is True,
        "s1_gate_pass": int(s1.get("successes", -1)) >= int(gates["s1_min_successes"]),
        "s2_gate_pass": int(s2.get("successes", -1)) >= int(gates["s2_min_successes"]),
        "yaw_saturation_gate_pass": (
            _finite(gate.get("action_saturation_yaw"), "gates.action_saturation_yaw")
            < float(gates["max_yaw_saturation"])
        ),
        "both_saturation_gate_pass": (
            _finite(gate.get("both_action_saturation"), "gates.both_action_saturation")
            < float(gates["max_both_saturation"])
        ),
        "curriculum_gate_pass": int(stage_metrics.get("successes", -1)) >= minimum,
        "c3_gate_variant_pass": (
            stage != "c3" or int(c3_gate_metrics.get("successes", -1)) >= int(c3_gate_required or 0)
        ),
    }
    failed = [name for name, passed in checks.items() if passed is not True]
    return {
        "schema": "auv_p1_ppo_curriculum_stage_audit.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "stage": stage,
        "source_training_summary": str(training_path),
        "source_eval_summary": str(eval_path),
        "seed": training.get("seed"),
        "config_sha256": training.get("config_sha256"),
        "eval_spec_sha256": training.get("eval_spec_sha256"),
        "best_iteration": training.get("best_iteration"),
        "best_checkpoint": str(best_path) if best_path is not None else None,
        "best_checkpoint_sha256": best_sha,
        "eval_checkpoint_sha256": eval_sha,
        "minimum_successes": minimum,
        "stage_metrics": dict(stage_metrics),
        "s1": dict(s1),
        "s2": dict(s2),
        "action_saturation_yaw": gate.get("action_saturation_yaw"),
        "both_action_saturation": gate.get("both_action_saturation"),
        "c3_gate_00": None if c3_gate is None else dict(c3_gate),
        "c3_gate_required": c3_gate_required,
        "checks": checks,
        "failed_checks": failed,
        "diagnostics": {
            "final_evaluation_gate": training.get("evaluation_gate"),
            "final_checkpoint_sha256": training.get("checkpoint_final_sha256"),
            "evaluation_count": training.get("evaluation_count"),
        },
    }


def main() -> int:
    args = _parse_args()
    audit = build_audit(
        stage=args.stage,
        training_path=args.training_summary,
        eval_path=args.eval_summary,
        gates=_load_json(args.gates),
        expect_iterations=args.expect_iterations,
        expect_envs=args.expect_envs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "stage": audit["stage"],
                "best_iteration": audit["best_iteration"],
                "best_checkpoint_sha256": audit["best_checkpoint_sha256"],
                "stage_successes": audit["stage_metrics"].get("successes"),
                "stage_episodes": audit["stage_metrics"].get("episodes"),
                "failed_checks": audit["failed_checks"],
            },
            indent=2,
        )
    )
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
