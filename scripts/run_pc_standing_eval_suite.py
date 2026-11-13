#!/usr/bin/env python3
"""Run a declared arm set over a materialized PC standing-eval manifest.

The runner is intentionally a thin dispatcher around the existing evaluators.
It does not change their success, terminal or reward contracts. Each arm must
name its evaluator and training config explicitly, so zero-residual, BC, PPO
and SAC results cannot be silently mixed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from auv_lowlevel._paths import resolve_from_root

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS_SCHEMA = "auv_pc_standing_eval_arms.v1"
SUPPORTED_EVALUATORS = {
    "sac": REPO_ROOT / "scripts/evaluate_pc_sac.py",
    "ppo": REPO_ROOT / "scripts/evaluate_pc_ppo.py",
    "bc_residual": REPO_ROOT / "scripts/evaluate_pc_bc_residual.py",
}
ANALYZER = REPO_ROOT / "scripts/analyze_pc_goal153_eval_grid.py"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolve(path: str | Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _relative(path: str | Path) -> str:
    try:
        return str(_resolve(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(_resolve(path))


def _validate_seed_window(start: int, count: int) -> None:
    if start < 0:
        raise ValueError("seed start must be non-negative")
    if count <= 0:
        raise ValueError("seed count must be positive")
    seeds = set(range(start, start + count))
    overlap = seeds & set(range(100, 150))
    if overlap:
        raise ValueError(f"held-out seeds 100-149 are closed: overlap={sorted(overlap)[:5]}")


def load_arms(
    path: Path,
    *,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("schema_version") != ARMS_SCHEMA:
        raise ValueError(f"arms schema_version must be {ARMS_SCHEMA!r}")
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a non-empty list")
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    requested = set(tags or [])
    found_requested: set[str] = set()
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict):
            raise ValueError(f"arms[{index}] must be an object")
        tag = str(arm.get("tag", ""))
        if not tag:
            raise ValueError(f"arms[{index}].tag must be non-empty")
        if tag in seen:
            raise ValueError(f"duplicate arm tag: {tag}")
        seen.add(tag)
        evaluator = arm.get("evaluator")
        if evaluator not in SUPPORTED_EVALUATORS:
            raise ValueError(
                f"arm {tag!r} evaluator must be one of {sorted(SUPPORTED_EVALUATORS)}"
            )
        training_config = arm.get("training_config")
        if not isinstance(training_config, str) or not training_config:
            raise ValueError(f"arm {tag!r} training_config must be a non-empty string")
        checkpoint = arm.get("checkpoint")
        action_mode = str(arm.get("action_mode", "policy"))
        if action_mode == "policy" and not checkpoint:
            raise ValueError(f"arm {tag!r} policy evaluation requires checkpoint")
        if evaluator in {"ppo", "bc_residual"} and not checkpoint:
            raise ValueError(f"arm {tag!r} {evaluator} evaluation requires checkpoint")
        extra_args = arm.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(
            isinstance(value, str) for value in extra_args
        ):
            raise ValueError(f"arm {tag!r} extra_args must be a list of strings")
        if requested and tag not in requested:
            continue
        found_requested.add(tag)
        selected.append(arm)
    missing = requested - found_requested
    if missing:
        raise ValueError(f"requested arm tags not found: {sorted(missing)}")
    if not selected:
        raise ValueError("no arms selected")
    return selected


def _append_common_eval_args(
    command: list[str],
    *,
    scenario_config: Path,
    output: Path,
    seed_start: int,
    seed_count: int,
    device: str,
    dtype: str,
    overwrite: bool,
) -> None:
    command.extend(
        [
            "--environment-config",
            str(scenario_config),
            "--output",
            str(output),
            "--seed-start",
            str(seed_start),
            "--seed-count",
            str(seed_count),
            "--device",
            device,
            "--dtype",
            dtype,
            "--success-frame",
            "both",
        ]
    )
    if overwrite:
        command.append("--overwrite")


def build_command(
    *,
    arm: dict[str, Any],
    scenario: dict[str, Any],
    output: Path,
    python: str,
    seed_start: int,
    seed_count: int,
    device: str,
    dtype: str,
    overwrite: bool,
) -> list[str]:
    evaluator = str(arm["evaluator"])
    script = SUPPORTED_EVALUATORS[evaluator]
    scenario_config = _resolve(scenario["config"])
    checkpoint = arm.get("checkpoint")
    action_mode = str(arm.get("action_mode", "policy"))
    residual_scale = float(arm.get("residual_scale", 1.0))
    command = [python, str(script)]

    if evaluator == "sac":
        command.extend(["--config", str(_resolve(arm["training_config"]))])
        _append_common_eval_args(
            command,
            scenario_config=scenario_config,
            output=output,
            seed_start=seed_start,
            seed_count=seed_count,
            device=device,
            dtype=dtype,
            overwrite=overwrite,
        )
        command.extend(
            [
                "--action-mode",
                action_mode,
                "--residual-scale",
                str(residual_scale),
                "--trace-output",
                str(output),
                "--trace-reward-breakdown",
                "--rich-trace",
                "--disable-curriculum-direct-command",
            ]
        )
        if checkpoint:
            command.extend(["--checkpoint", str(_resolve(str(checkpoint)))])
        gate_mode = arm.get("learner_terminal_gate_mode")
        if gate_mode is not None:
            command.extend(["--learner-terminal-gate-mode", str(gate_mode)])
    elif evaluator == "ppo":
        command.extend(
            [
                "--training-config",
                str(_resolve(arm["training_config"])),
                "--checkpoint",
                str(_resolve(str(checkpoint))),
            ]
        )
        _append_common_eval_args(
            command,
            scenario_config=scenario_config,
            output=output,
            seed_start=seed_start,
            seed_count=seed_count,
            device=device,
            dtype=dtype,
            overwrite=overwrite,
        )
        command.extend(
            [
                "--action-mode",
                action_mode,
                "--trace-output",
                str(output / "trace.npz"),
                "--trace-reward-breakdown",
                "--disable-curriculum-direct-command",
            ]
        )
        if arm.get("curriculum_stage") is not None:
            command.extend(["--curriculum-stage", str(arm["curriculum_stage"])])
    elif evaluator == "bc_residual":
        command.extend(
            [
                "--config",
                str(_resolve(arm["training_config"])),
                "--checkpoint",
                str(_resolve(str(checkpoint))),
            ]
        )
        _append_common_eval_args(
            command,
            scenario_config=scenario_config,
            output=output,
            seed_start=seed_start,
            seed_count=seed_count,
            device=device,
            dtype=dtype,
            overwrite=overwrite,
        )
        command.extend(
            [
                "--action-mode",
                action_mode,
                "--residual-scale",
                str(residual_scale),
                "--disable-curriculum-direct-command",
            ]
        )
    else:
        raise AssertionError(f"unhandled evaluator: {evaluator}")

    command.extend(str(value) for value in arm.get("extra_args", []))
    return command


def build_jobs(
    *,
    manifest: dict[str, Any],
    arms: list[dict[str, Any]],
    output_root: Path,
    python: str,
    seed_start: int,
    seed_count: int,
    device: str,
    dtype: str,
    overwrite: bool,
    scenarios: list[str] | None = None,
) -> list[dict[str, Any]]:
    _validate_seed_window(seed_start, seed_count)
    scenario_payloads = manifest.get("scenarios")
    if not isinstance(scenario_payloads, list) or not scenario_payloads:
        raise ValueError("manifest.scenarios must be a non-empty list")
    requested = set(scenarios or [])
    found: set[str] = set()
    jobs: list[dict[str, Any]] = []
    for scenario in scenario_payloads:
        sid = str(scenario.get("id", ""))
        if not sid:
            raise ValueError("every scenario must have a non-empty id")
        if requested and sid not in requested:
            continue
        found.add(sid)
        for arm in arms:
            tag = str(arm["tag"])
            training_seed = arm.get("training_seed")
            run_tag = tag if training_seed is None else f"{tag}_s{training_seed}"
            output = output_root / run_tag / sid
            command = build_command(
                arm=arm,
                scenario=scenario,
                output=output,
                python=python,
                seed_start=seed_start,
                seed_count=seed_count,
                device=device,
                dtype=dtype,
                overwrite=overwrite,
            )
            jobs.append(
                {
                    "arm_tag": tag,
                    "run_tag": run_tag,
                    "paradigm": arm.get("paradigm"),
                    "evaluator": arm["evaluator"],
                    "scenario": sid,
                    "output": _relative(output),
                    "command": command,
                }
            )
    missing = requested - found
    if missing:
        raise ValueError(f"requested scenarios not found: {sorted(missing)}")
    if not jobs:
        raise ValueError("no evaluation jobs selected")
    return jobs


def has_paired_arms(jobs: list[dict[str, Any]], *, zero_tag: str) -> bool:
    """Return whether at least one non-zero policy arm is present.

    A zero-only selection is a valid baseline run, but the paired analyzer needs
    a policy tag to compute deltas.  Skipping the analyzer in that case avoids a
    false failure while leaving full multi-arm screens unchanged.
    """

    return any(str(job.get("arm_tag")) != zero_tag for job in jobs)


def _pythonpath(sb3_root: Path | None) -> str:
    parts = [str(REPO_ROOT / "src")]
    if sb3_root is not None:
        parts.append(str(sb3_root))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _run_job(
    job: dict[str, Any],
    *,
    log_root: Path,
    gpu: str | None,
    timeout_s: float | None,
    skip_existing: bool,
    pythonpath: str,
) -> dict[str, Any]:
    output = _resolve(job["output"])
    summary = output / "summary.json"
    if skip_existing and summary.is_file():
        return {**job, "status": "skipped_existing"}
    output.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{job['run_tag']}__{job['scenario']}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                job["command"],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        status = "ok" if completed.returncode == 0 else "failed"
        return {
            **job,
            "status": status,
            "exit_code": completed.returncode,
            "log": _relative(log_path),
        }
    except subprocess.TimeoutExpired:
        return {**job, "status": "timeout", "log": _relative(log_path)}


def run_jobs(
    jobs: list[dict[str, Any]],
    *,
    max_parallel: int,
    gpus: list[str],
    log_root: Path,
    timeout_s: float | None,
    skip_existing: bool,
    pythonpath: str,
) -> list[dict[str, Any]]:
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)] if gpus else None
            futures[
                executor.submit(
                    _run_job,
                    job,
                    log_root=log_root,
                    gpu=gpu,
                    timeout_s=timeout_s,
                    skip_existing=skip_existing,
                    pythonpath=pythonpath,
                )
            ] = job
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['status']}: {result['run_tag']}/{result['scenario']}",
                flush=True,
            )
    return sorted(results, key=lambda item: (item["run_tag"], item["scenario"]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sb3-root", type=Path, default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--gpus", default="", help="space-separated GPU ids, round-robin")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze", dest="analyze", action="store_true", default=True)
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    parser.add_argument("--zero-tag", default="zero")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = _resolve(args.manifest)
    arms_path = _resolve(args.arms)
    output_root = _resolve(args.output_root)
    manifest = _load_json(manifest_path)
    arms = load_arms(arms_path, tags=args.tags)
    seed_start = int(manifest.get("eval_seeds_start", 20) if args.seed_start is None else args.seed_start)
    seed_count = int(manifest.get("eval_seeds", 40) if args.seed_count is None else args.seed_count)
    gpus = args.gpus.split()
    pythonpath = _pythonpath(None if args.sb3_root is None else _resolve(args.sb3_root))
    jobs = build_jobs(
        manifest=manifest,
        arms=arms,
        output_root=output_root,
        python=args.python,
        seed_start=seed_start,
        seed_count=seed_count,
        device=args.device,
        dtype=args.dtype,
        overwrite=args.overwrite,
        scenarios=args.scenarios,
    )
    plan = {
        "schema_version": "auv_pc_standing_eval_run_plan.v1",
        "manifest": _relative(manifest_path),
        "arms": _relative(arms_path),
        "output_root": _relative(output_root),
        "seed_start": seed_start,
        "seed_count": seed_count,
        "device": args.device,
        "dtype": args.dtype,
        "gpus": gpus,
        "job_count": len(jobs),
        "jobs": jobs,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    results = run_jobs(
        jobs,
        max_parallel=args.max_parallel,
        gpus=gpus,
        log_root=output_root / "logs",
        timeout_s=args.timeout_s,
        skip_existing=args.skip_existing,
        pythonpath=pythonpath,
    )
    plan["results"] = results
    (output_root / "run_manifest.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    failed = [item for item in results if item["status"] not in {"ok", "skipped_existing"}]
    paired_arms = has_paired_arms(jobs, zero_tag=args.zero_tag)
    if args.analyze and not paired_arms:
        print("zero-only selection: paired analyzer skipped; baseline summaries are complete")
    if args.analyze and paired_arms and any(
        (_resolve(item["output"]) / "summary.json").is_file() for item in results
    ):
        command = [
            args.python,
            str(ANALYZER),
            "--root",
            str(output_root),
            "--output-json",
            str(output_root / "grid_summary.json"),
            "--output-md",
            str(output_root / "grid_summary.md"),
            "--zero-tag",
            args.zero_tag,
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": pythonpath},
            check=False,
        )
        if completed.returncode != 0:
            print("analyzer failed; inspect grid_summary files", file=sys.stderr)
            failed.append({"status": "analyzer_failed", "exit_code": completed.returncode})
    if failed:
        print(json.dumps({"failed": failed}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(f"standing eval complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
