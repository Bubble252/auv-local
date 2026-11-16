#!/usr/bin/env python3
"""Run the frozen Goal175 A2 72-cell formal evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
A1_ARMS_PATH = REPO_ROOT / "configs/lowlevel/goal175/a1_arms.json"
DEFAULT_ROOT = REPO_ROOT / "artifacts/lowlevel/goal175_alos_comp_mlp_a2_20261114"
EVALUATOR = REPO_ROOT / "scripts/evaluate_pc_ppo.py"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_run(manifest: dict[str, Any], arm: str) -> dict[str, Any]:
    for run in manifest["train_runs"]:
        if run["arm"] == arm:
            return run
    raise ValueError(f"{arm}: missing from train_runs")


def build_jobs(
    *,
    root: Path,
    python: str,
    seed_start: int,
    seed_count: int,
    device: str,
    dtype: str,
) -> list[dict[str, Any]]:
    manifest = _load(root / "configs/a2_manifest.json")
    a1_arms = _load(A1_ARMS_PATH)
    zero_by_arm = {str(arm["tag"]): arm for arm in a1_arms["arms"]}
    zero_checkpoint = _resolve(a1_arms["checkpoint"])
    zero_training_config = _resolve(a1_arms["training_config"])
    zero_checkpoint_sha = str(a1_arms["checkpoint_sha256"])
    if _sha256(zero_checkpoint) != zero_checkpoint_sha:
        raise ValueError(f"{zero_checkpoint}: zero-baseline checkpoint SHA drifted")

    jobs: list[dict[str, Any]] = []
    for arm_name in manifest["policy_eval_arms"]:
        run = _policy_run(manifest, str(arm_name))
        mode = str(run["alos_current_mode"])
        checkpoint = Path(str(run["output"])) / "final.pt"
        training_config = Path(str(run["training_config"]))
        if not checkpoint.is_file():
            raise ValueError(f"{checkpoint}: missing policy checkpoint")
        checkpoint_sha = _sha256(checkpoint)
        for scenario in manifest["eval_scenarios"]:
            scenario_id = str(scenario["scenario"])
            config = Path(str(scenario["configs"][mode]))
            output = root / "eval" / str(arm_name) / scenario_id
            command = [
                python,
                str(EVALUATOR),
                "--training-config",
                str(training_config),
                "--checkpoint",
                str(checkpoint),
                "--environment-config",
                str(config),
                "--output",
                str(output),
                "--seed-start",
                str(seed_start),
                "--seed-count",
                str(seed_count),
                # A2 evaluation configs carry a fixed pressure profile.  Passing
                # --curriculum-stage would resample the pressure axis and
                # invalidate the matrix cell.
                "--disable-curriculum-direct-command",
                "--device",
                device,
                "--dtype",
                dtype,
                "--action-mode",
                "policy",
                "--success-frame",
                "both",
                "--trace-output",
                str(output / "trace.npz"),
                "--allow-alos-current-mode-override",
            ]
            if scenario["pressure"] == "noise_n2":
                command.append("--allow-sensor-override")
            jobs.append(
                {
                    "arm": str(arm_name),
                    "base_arm": str(run["base_arm"]),
                    "role": "policy",
                    "action_mode": "policy",
                    "alos_current_mode": mode,
                    "scenario": scenario_id,
                    "pressure": str(scenario["pressure"]),
                    "radius": str(scenario["radius"]),
                    "training_config": str(training_config),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "environment_config": str(config),
                    "environment_sha256": str(scenario["sha256"][mode]),
                    "output": _relative(output),
                    "command": command,
                }
            )

    for baseline in manifest["zero_baselines"]:
        arm_name = str(baseline["arm"])
        mode = str(baseline["alos_current_mode"])
        action_mode = str(baseline["action_mode"])
        if arm_name not in zero_by_arm:
            raise ValueError(f"{arm_name}: missing zero-baseline arm declaration")
        declared = zero_by_arm[arm_name]
        if declared["action_mode"] != action_mode or declared["alos_current_mode"] != mode:
            raise ValueError(f"{arm_name}: zero-baseline action/mode drifted")
        for scenario in manifest["eval_scenarios"]:
            scenario_id = str(scenario["scenario"])
            config = Path(str(scenario["configs"][mode]))
            output = root / "eval" / arm_name / scenario_id
            command = [
                python,
                str(EVALUATOR),
                "--training-config",
                str(zero_training_config),
                "--checkpoint",
                str(zero_checkpoint),
                "--environment-config",
                str(config),
                "--output",
                str(output),
                "--seed-start",
                str(seed_start),
                "--seed-count",
                str(seed_count),
                "--disable-curriculum-direct-command",
                "--device",
                device,
                "--dtype",
                dtype,
                "--action-mode",
                action_mode,
                "--success-frame",
                "both",
                "--trace-output",
                str(output / "trace.npz"),
                "--allow-alos-current-mode-override",
            ]
            if scenario["pressure"] == "noise_n2":
                command.append("--allow-sensor-override")
            jobs.append(
                {
                    "arm": arm_name,
                    "base_arm": None,
                    "role": "zero",
                    "action_mode": action_mode,
                    "alos_current_mode": mode,
                    "scenario": scenario_id,
                    "pressure": str(scenario["pressure"]),
                    "radius": str(scenario["radius"]),
                    "training_config": str(zero_training_config),
                    "checkpoint": str(zero_checkpoint),
                    "checkpoint_sha256": zero_checkpoint_sha,
                    "environment_config": str(config),
                    "environment_sha256": str(scenario["sha256"][mode]),
                    "output": _relative(output),
                    "command": command,
                }
            )
    if len(jobs) != 72:
        raise ValueError(f"A2 must produce exactly 72 cells, got {len(jobs)}")
    return jobs


def _run_job(
    job: dict[str, Any],
    *,
    log_root: Path,
    gpu: str,
    sb3_root: Path | None,
    timeout_s: float | None,
    skip_existing: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output = _resolve(job["output"])
    complete = (output / "summary.json").is_file() and (output / "trace.npz").is_file()
    if complete and skip_existing and not overwrite:
        return {**job, "status": "skipped_existing"}
    if overwrite and output.exists():
        if "eval" not in output.parts:
            raise ValueError(f"refusing to overwrite a non-eval path: {output}")
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        return {**job, "status": "failed", "error": "non-empty incomplete output"}
    output.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{job['arm']}__{job['scenario']}.log"
    environment = os.environ.copy()
    pythonpath = [str(REPO_ROOT / "src")]
    if sb3_root is not None:
        pythonpath.append(str(sb3_root))
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath if item)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                list(job["command"]),
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {**job, "status": "timeout", "log": _relative(log_path)}
    return {
        **job,
        "status": "ok" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "log": _relative(log_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument(
        "--sb3-root",
        type=Path,
        default=None,
        help="stable-baselines3 checkout prepended to PYTHONPATH on the server",
    )
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = _resolve(args.root)
    if args.seed_start != 20 or args.seed_count != 40:
        raise ValueError("Goal175 A2 fixes dev seeds 20-59")
    jobs = build_jobs(
        root=root,
        python=args.python,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        device=args.device,
        dtype=args.dtype,
    )
    if args.dry_run:
        print(json.dumps({"job_count": len(jobs), "jobs": jobs}, indent=2, ensure_ascii=False))
        return 0

    gpus = args.gpus.split()
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(
                _run_job,
                job,
                log_root=root / "logs" / "eval_a2",
                gpu=gpus[index % len(gpus)],
                sb3_root=None if args.sb3_root is None else _resolve(args.sb3_root),
                timeout_s=args.timeout_s,
                skip_existing=args.skip_existing,
                overwrite=args.overwrite,
            ): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']}: {result['arm']}/{result['scenario']}", flush=True)

    results.sort(key=lambda item: (item["arm"], item["scenario"]))
    manifest = {
        "schema_version": "auv_goal175_a2_run_manifest.v1",
        "goal_id": "Goal175",
        "phase": "A2",
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "device": args.device,
        "dtype": args.dtype,
        "gpus": gpus,
        "max_parallel": args.max_parallel,
        "job_count": len(jobs),
        "jobs": results,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    failed = [item for item in results if item["status"] not in {"ok", "skipped_existing"}]
    if failed:
        print(json.dumps({"failed": failed}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(f"Goal175 A2 evaluation complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
