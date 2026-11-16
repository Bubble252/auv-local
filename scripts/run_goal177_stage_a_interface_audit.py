#!/usr/bin/env python3
"""Launch the Goal177 Stage A action-interface matrix.

Each cell is a separate evaluator process.  This keeps CUDA state and policy
capture isolated while allowing the full matrix to run in parallel across GPUs.
The launcher writes a run manifest; the worker writes ``cell.json`` and the
auditor checks completeness and Gate A.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs/governance/goal177_matrix.json"
DEFAULT_ZERO_ARMS = REPO_ROOT / "configs/lowlevel/goal175/a1_arms.json"
DEFAULT_WORKER = REPO_ROOT / "scripts/goal177_stage_a_cell.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--goal175-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--zero-arms", type=Path, default=DEFAULT_ZERO_ARMS)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--gpus", default="0 1 6 7")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--levels", nargs="+", default=None)
    parser.add_argument("--radii", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_record(a2_manifest: dict[str, Any], arm: str) -> dict[str, Any]:
    for record in a2_manifest.get("train_runs", []):
        if isinstance(record, dict) and record.get("arm") == arm:
            return record
    raise ValueError(f"{arm}: missing from Goal175 a2_manifest.train_runs")


def _scenario_map(a2_manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in a2_manifest.get("eval_scenarios", []):
        if not isinstance(record, dict):
            continue
        result[(str(record["pressure"]), str(record["radius"]))] = record
    return result


def _build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    matrix = _load(args.matrix)
    if matrix.get("goal_id") != "Goal177":
        raise ValueError(f"unexpected matrix goal_id: {matrix.get('goal_id')!r}")
    a2_manifest = _load(args.goal175_root / "configs/a2_manifest.json")
    scenarios = _scenario_map(a2_manifest)
    policy_run = _train_record(a2_manifest, "P_RUD_075_COMP")
    policy_training = Path(str(policy_run["training_config"]))
    policy_checkpoint = Path(str(policy_run["output"])) / "final.pt"
    if not policy_checkpoint.is_file():
        raise FileNotFoundError(policy_checkpoint)
    zero_arms = _load(args.zero_arms)
    zero_by_arm = {
        str(item.get("arm", item.get("tag"))): item for item in zero_arms["arms"]
    }
    zero_training = (args.worktree / str(zero_arms["training_config"])).resolve()
    zero_checkpoint = (args.worktree / str(zero_arms["checkpoint"])).resolve()
    if not zero_training.is_file():
        raise FileNotFoundError(zero_training)
    if not zero_checkpoint.is_file():
        raise FileNotFoundError(zero_checkpoint)

    arms = [record for record in matrix["arms"] if isinstance(record, dict)]
    levels = [str(value) for value in matrix["levels"]]
    radii = [str(value) for value in matrix["radii"]]
    if args.arms:
        keep = set(args.arms)
        arms = [record for record in arms if str(record["arm"]) in keep]
    if args.levels:
        keep = set(args.levels)
        levels = [value for value in levels if value in keep]
    if args.radii:
        keep = set(args.radii)
        radii = [value for value in radii if value in keep]
    if args.smoke:
        arms = [record for record in arms if record["arm"] == "A_S075_RUDDER_ONLY"]
        levels = ["current10"] if "current10" in levels else levels[:1]
        radii = ["default"] if "default" in radii else radii[:1]
    if not arms or not levels or not radii:
        raise ValueError("selection produced an empty Stage A matrix")

    stage_root = args.root / "stage_a"
    jobs: list[dict[str, Any]] = []
    for arm in arms:
        arm_name = str(arm["arm"])
        for level in levels:
            for radius in radii:
                scenario = scenarios.get((level, radius))
                if scenario is None:
                    raise ValueError(f"missing Goal175 scenario for {level}/{radius}")
                env_config = Path(str(scenario["configs"]["ALOS-COMP"]))
                if not env_config.is_file():
                    raise FileNotFoundError(env_config)
                output = stage_root / "cells" / arm_name / str(scenario["scenario"])
                command = [
                    args.python,
                    str(args.worker),
                    "--repo-root",
                    str(args.worktree),
                    "--output",
                    str(output),
                    "--training-config",
                    str(policy_training),
                    "--checkpoint",
                    str(policy_checkpoint),
                    "--environment-config",
                    str(env_config),
                    "--arm",
                    arm_name,
                    "--stage",
                    "A",
                    "--level",
                    level,
                    "--radius",
                    radius,
                    "--scenario",
                    str(scenario["scenario"]),
                    "--action-mode",
                    "policy",
                    "--seed-start",
                    str(args.seed_start),
                    "--seed-count",
                    str(args.seed_count),
                    "--device",
                    "cuda",
                    "--dtype",
                    "float64",
                    "--actuator-residual-scale",
                    str(float(arm["authority_scale"])),
                    "--actuator-residual-channel-gains",
                    ",".join(str(float(value)) for value in arm["actuator_residual_channel_gains"]),
                    "--capture-stride",
                    "10",
                    "--max-jacobian-samples",
                    "128",
                ]
                if level == "noise_n2":
                    command.append("--allow-sensor-override")
                if args.overwrite:
                    command.append("--overwrite")
                jobs.append(
                    {
                        "arm": arm_name,
                        "role": "policy",
                        "level": level,
                        "radius": radius,
                        "scenario": str(scenario["scenario"]),
                        "action_mode": "policy",
                        "training_config": str(policy_training),
                        "checkpoint": str(policy_checkpoint),
                        "checkpoint_sha256": _sha256(policy_checkpoint),
                        "environment_config": str(env_config),
                        "environment_sha256": _sha256(env_config),
                        "output": str(output),
                        "command": command,
                    }
                )

    for baseline_name in matrix["baselines"]:
        if baseline_name not in zero_by_arm:
            raise ValueError(f"{baseline_name}: missing zero baseline declaration")
        baseline = zero_by_arm[baseline_name]
        for level in levels:
            for radius in radii:
                scenario = scenarios.get((level, radius))
                if scenario is None:
                    raise ValueError(f"missing Goal175 scenario for {level}/{radius}")
                env_config = Path(str(scenario["configs"]["ALOS-COMP"]))
                output = stage_root / "cells" / baseline_name / str(scenario["scenario"])
                command = [
                    args.python,
                    str(args.worker),
                    "--repo-root",
                    str(args.worktree),
                    "--output",
                    str(output),
                    "--training-config",
                    str(zero_training),
                    "--checkpoint",
                    str(zero_checkpoint),
                    "--environment-config",
                    str(env_config),
                    "--arm",
                    baseline_name,
                    "--stage",
                    "A",
                    "--level",
                    level,
                    "--radius",
                    radius,
                    "--scenario",
                    str(scenario["scenario"]),
                    "--action-mode",
                    str(baseline["action_mode"]),
                    "--seed-start",
                    str(args.seed_start),
                    "--seed-count",
                    str(args.seed_count),
                    "--device",
                    "cuda",
                    "--dtype",
                    "float64",
                    "--skip-jacobian",
                ]
                if level == "noise_n2":
                    command.append("--allow-sensor-override")
                if args.overwrite:
                    command.append("--overwrite")
                jobs.append(
                    {
                        "arm": baseline_name,
                        "role": "zero",
                        "level": level,
                        "radius": radius,
                        "scenario": str(scenario["scenario"]),
                        "action_mode": str(baseline["action_mode"]),
                        "training_config": str(zero_training),
                        "checkpoint": str(zero_checkpoint),
                        "checkpoint_sha256": _sha256(zero_checkpoint),
                        "environment_config": str(env_config),
                        "environment_sha256": _sha256(env_config),
                        "output": str(output),
                        "command": command,
                    }
                )
    return jobs


def _run_job(
    job: dict[str, Any],
    *,
    log_root: Path,
    gpu: str,
    pythonpath: str,
    skip_existing: bool,
    timeout_s: float,
) -> dict[str, Any]:
    output = Path(job["output"])
    complete = (output / "cell.json").is_file()
    if complete and skip_existing:
        try:
            cell = _load(output / "cell.json")
            if cell.get("status") == "ok":
                return {**job, "status": "skipped_existing", "gpu": gpu}
        except Exception:
            pass
    if output.exists() and any(output.iterdir()) and not any(
        str(Path(value)).endswith("--overwrite") for value in job["command"]
    ):
        return {
            **job,
            "status": "failed",
            "gpu": gpu,
            "exit_code": None,
            "error": "non-empty incomplete output; use --overwrite",
        }
    output.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{job['arm']}__{job['scenario']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = pythonpath
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                job["command"],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_s,
            )
        status = "ok" if completed.returncode == 0 else "failed"
        return {
            **job,
            "status": status,
            "gpu": gpu,
            "exit_code": int(completed.returncode),
            "log": str(log_path),
        }
    except subprocess.TimeoutExpired:
        return {
            **job,
            "status": "timeout",
            "gpu": gpu,
            "exit_code": None,
            "log": str(log_path),
            "error": f"timeout after {timeout_s}s",
        }


def main() -> int:
    args = _parse_args()
    if not args.gpus.split():
        raise ValueError("at least one GPU is required")
    if args.max_parallel <= 0:
        raise ValueError("max-parallel must be positive")
    if args.seed_count <= 0 or args.seed_start < 0:
        raise ValueError("invalid seed range")
    if args.seed_start + args.seed_count > 100:
        raise ValueError("Stage A must not enter held-out seeds >=100")
    root = args.root.resolve()
    stage_root = root / "stage_a"
    jobs = _build_jobs(args)
    gpus = args.gpus.split()
    pythonpath = os.pathsep.join(
        [
            str(args.worktree / "src"),
            "/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src",
            "/data/yhx/underwater-robotics/src/stable-baselines3",
        ]
    )
    manifest_path = stage_root / "run_manifest.json"
    manifest = {
        "schema_version": "goal177_stage_a_run_manifest.v1",
        "goal_id": "Goal177",
        "phase": "A",
        "root": str(root),
        "goal175_root": str(args.goal175_root),
        "worktree": str(args.worktree),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "gpus": gpus,
        "max_parallel": args.max_parallel,
        "job_count": len(jobs),
        "jobs": [{**job, "status": "pending"} for job in jobs],
    }
    _write_json(manifest_path, manifest)
    lock = threading.Lock()
    records = {index: row for index, row in enumerate(manifest["jobs"])}

    def update(index: int, result: dict[str, Any]) -> None:
        with lock:
            records[index] = result
            manifest["jobs"] = [records[key] for key in sorted(records)]
            _write_json(manifest_path, manifest)

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            future = executor.submit(
                _run_job,
                job,
                log_root=stage_root / "logs",
                gpu=gpu,
                pythonpath=pythonpath,
                skip_existing=args.skip_existing,
                timeout_s=args.job_timeout,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                result = {
                    **jobs[index],
                    "status": "failed",
                    "exit_code": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            update(index, result)
            print(
                f"{result.get('status'):>16} {result.get('arm')} "
                f"{result.get('level')}/{result.get('radius')}",
                flush=True,
            )
    counts: dict[str, int] = {}
    for row in records.values():
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    manifest["status_counts"] = counts
    _write_json(manifest_path, manifest)
    print(json.dumps({"root": str(root), "job_count": len(jobs), "status_counts": counts}))
    return 0 if counts.get("failed", 0) == 0 and counts.get("timeout", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
