"""Audit a completed PPO seed and write compact, reproducible evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one completed P1 PPO seed")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def _resolve_checkpoint(summary_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, summary_path.parent / path]
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _entropy_values(records: Any) -> list[float] | None:
    if not isinstance(records, list) or not records:
        return None
    values: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            return None
        losses = record.get("losses")
        if not isinstance(losses, dict):
            return None
        try:
            entropy = float(losses["entropy"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(entropy):
            return None
        values.append(entropy)
    return values


def _best_evaluation(summary: dict[str, Any]) -> dict[str, Any] | None:
    best_iteration = summary.get("best_iteration")
    evaluations = summary.get("evaluations")
    if not isinstance(evaluations, list):
        return None
    for evaluation in evaluations:
        if isinstance(evaluation, dict) and evaluation.get("iteration") == best_iteration:
            return evaluation
    return None


def _last_twenty_percent(summary: dict[str, Any]) -> dict[str, Any]:
    evaluations = summary.get("evaluations")
    if not isinstance(evaluations, list):
        return {"count": 0, "failures": 0, "passes": 0}
    completed = int(summary.get("iterations_completed") or 0)
    threshold = int(completed * 0.8)
    selected = [
        value
        for value in evaluations
        if isinstance(value, dict) and int(value.get("iteration") or 0) >= threshold
    ]
    failures = [value for value in selected if value.get("status") != "PASS"]
    return {
        "count": len(selected),
        "passes": len(selected) - len(failures),
        "failures": len(failures),
        "failed_iterations": [value.get("iteration") for value in failures],
    }


def build_audit(
    summary_path: Path,
    *,
    expect_iterations: int = 5000,
    expect_envs: int = 4096,
) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    summary = _load_json(summary_path)
    records = summary.get("records")
    entropy_values = _entropy_values(records)
    best_path = _resolve_checkpoint(summary_path, summary.get("best_checkpoint"))
    final_path = _resolve_checkpoint(summary_path, summary.get("checkpoint_final"))
    best_sha = _sha256(best_path) if best_path is not None and best_path.is_file() else None
    final_sha = _sha256(final_path) if final_path is not None and final_path.is_file() else None
    iterations_completed = summary.get("iterations_completed")
    max_iterations = summary.get("max_iterations")
    best_gates = summary.get("best_gates")
    evaluations = summary.get("evaluations")

    last_twenty_percent = _last_twenty_percent(summary)
    acceptance_checks = {
        "status_pass": summary.get("status") == "PASS",
        "iterations_complete": (iterations_completed == max_iterations == expect_iterations),
        "num_envs_matches": summary.get("num_envs") == expect_envs,
        "eval_errors_empty": not summary.get("eval_errors"),
        "model_parameters_finite": summary.get("model_parameters_finite") is True,
        "losses_finite": summary.get("losses_finite") is True,
        "deterministic_action_finite": summary.get("deterministic_action_finite") is True,
        "records_complete": isinstance(records, list) and len(records) == expect_iterations,
        "entropy_finite": entropy_values is not None,
        "evaluations_present": isinstance(evaluations, list) and bool(evaluations),
        "best_gate_pass": isinstance(best_gates, dict) and best_gates.get("pass") is True,
        "best_checkpoint_exists": best_sha is not None,
        "best_checkpoint_sha_matches": (
            best_sha is not None and best_sha == summary.get("best_checkpoint_sha256")
        ),
        "final_checkpoint_exists": final_sha is not None,
        "final_checkpoint_sha_matches": (
            final_sha is not None
            and final_sha == summary.get("checkpoint_final_sha256")
            and final_sha == summary.get("checkpoint_sha256")
        ),
        "last_twenty_percent_majority_pass": (
            last_twenty_percent["count"] > 0
            and last_twenty_percent["passes"] > last_twenty_percent["failures"]
        ),
    }
    diagnostic_checks = {
        "final_evaluation_gate_pass": summary.get("evaluation_gate") == "PASS",
    }
    status = "PASS" if all(acceptance_checks.values()) else "FAIL"
    return {
        "schema": "auv_p1_ppo_seed_audit.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "source_summary": str(summary_path),
        "seed": summary.get("seed"),
        "num_envs": summary.get("num_envs"),
        "num_steps_per_env": summary.get("num_steps_per_env"),
        "iterations_completed": iterations_completed,
        "max_iterations": max_iterations,
        "training_s": summary.get("training_s"),
        "config_sha256": summary.get("config_sha256"),
        "eval_spec_sha256": summary.get("eval_spec_sha256"),
        "checkpoint": str(final_path) if final_path is not None else None,
        "checkpoint_sha256": final_sha,
        "best_checkpoint": str(best_path) if best_path is not None else None,
        "best_iteration": summary.get("best_iteration"),
        "best_score": summary.get("best_score"),
        "best_checkpoint_sha256": best_sha,
        "selected_checkpoint": "best",
        "selected_checkpoint_gate_pass": (
            isinstance(best_gates, dict) and best_gates.get("pass") is True
        ),
        "best_gates": best_gates,
        "best_evaluation": _best_evaluation(summary),
        "final_evaluation": evaluations[-1]
        if isinstance(evaluations, list) and evaluations
        else None,
        "evaluation_count": len(evaluations) if isinstance(evaluations, list) else 0,
        "final_evaluation_gate": summary.get("evaluation_gate"),
        "eval_errors": summary.get("eval_errors"),
        "last_twenty_percent": last_twenty_percent,
        "entropy_min": min(entropy_values) if entropy_values is not None else None,
        "entropy_positive": summary.get("entropy_positive"),
        "acceptance_basis": (
            "Selected best checkpoint; final checkpoint and last-iteration gate are diagnostics."
        ),
        "acceptance_checks": acceptance_checks,
        "diagnostic_checks": diagnostic_checks,
    }


def main() -> int:
    args = _parse_args()
    audit = build_audit(
        args.summary,
        expect_iterations=args.expect_iterations,
        expect_envs=args.expect_envs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "seed": audit["seed"],
                "best_iteration": audit["best_iteration"],
                "best_checkpoint_sha256": audit["best_checkpoint_sha256"],
                "final_evaluation_gate": audit["final_evaluation_gate"],
                "failed_checks": [
                    key for key, value in audit["acceptance_checks"].items() if value is not True
                ],
            },
            indent=2,
        )
    )
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
