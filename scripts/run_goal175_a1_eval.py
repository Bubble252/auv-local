#!/usr/bin/env python3
"""Run the frozen Goal175 A1 COMP/NONE eval-only cell matrix."""

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
MATRIX_PATH = REPO_ROOT / "configs/governance/goal175_matrix.json"
ARMS_PATH = REPO_ROOT / "configs/lowlevel/goal175/a1_arms.json"
DEFAULT_ROOT = REPO_ROOT / "artifacts/lowlevel/goal175_alos_comp_mlp_20261114"
EVALUATOR = REPO_ROOT / "scripts/evaluate_pc_ppo.py"
CHECKER = REPO_ROOT / "scripts/check_goal175_protocol.py"
AUDITOR = REPO_ROOT / "scripts/audit_goal175_artifacts.py"
ANALYZER = REPO_ROOT / "scripts/analyze_goal175_a1.py"


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


def build_jobs(
    *,
    root: Path,
    python: str,
    checkpoint_override: Path | None,
    training_config_override: Path | None,
    seed_start: int,
    seed_count: int,
    device: str,
    dtype: str,
) -> list[dict[str, Any]]:
    manifest = _load(root / "configs/a1_manifest.json")
    arms_payload = _load(ARMS_PATH)
    training_config = (
        _resolve(arms_payload["training_config"])
        if training_config_override is None
        else _resolve(training_config_override)
    )
    checkpoint = (
        _resolve(arms_payload["checkpoint"])
        if checkpoint_override is None
        else _resolve(checkpoint_override)
    )
    if _sha256(checkpoint) != arms_payload["checkpoint_sha256"]:
        raise ValueError(f"{checkpoint}: checkpoint SHA drifted")
    if _sha256(training_config) != _load_checkpoint_metadata(checkpoint)["training_config_sha256"]:
        raise ValueError(f"{checkpoint}: training-config hash does not match {training_config}")

    jobs: list[dict[str, Any]] = []
    for arm in arms_payload["arms"]:
        mode = str(arm["alos_current_mode"])
        for scenario in manifest["scenarios"]:
            scenario_id = str(scenario["scenario"])
            config = _resolve(scenario["configs"][mode])
            output = root / "eval" / str(arm["tag"]) / scenario_id
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
                "--curriculum-stage",
                "C3",
                "--disable-curriculum-direct-command",
                "--device",
                device,
                "--dtype",
                dtype,
                "--action-mode",
                str(arm["action_mode"]),
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
                    "arm": str(arm["tag"]),
                    "role": str(arm["role"]),
                    "alos_current_mode": mode,
                    "scenario": scenario_id,
                    "pressure": str(scenario["pressure"]),
                    "radius": str(scenario["radius"]),
                    "environment_config": _relative(config),
                    "environment_sha256": str(scenario["sha256"][mode]),
                    "output": _relative(output),
                    "command": command,
                }
            )
    if len(jobs) != 40:
        raise ValueError(f"A1 must produce exactly 40 cells, got {len(jobs)}")
    return jobs


def _load_checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    import torch

    from auv_lowlevel.batch.checkpoint import read_pc_checkpoint_metadata

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return read_pc_checkpoint_metadata(state)


def _run_job(
    job: dict[str, Any],
    *,
    log_root: Path,
    gpu: str | None,
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
            raise ValueError(f"refusing to overwrite a non-A1 path: {output}")
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        return {**job, "status": "failed", "error": "non-empty incomplete output"}
    output.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{job['arm']}__{job['scenario']}.log"
    command = list(job["command"])
    environment = os.environ.copy()
    pythonpath = [str(REPO_ROOT / "src")]
    if sb3_root is not None:
        pythonpath.append(str(sb3_root))
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in pythonpath if item
    )
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {**job, "status": "timeout", "log": _relative(log_path)}
    status = "ok" if result.returncode == 0 else "failed"
    return {
        **job,
        "status": status,
        "exit_code": result.returncode,
        "log": _relative(log_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training-config", type=Path, default=None)
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
        raise ValueError("Goal175 A1 fixes dev seeds 20-59")
    protocol_command = [
        args.python,
        str(CHECKER),
        "--phase",
        "a1",
        "--config-root",
        str(root / "configs"),
    ]
    if args.checkpoint is not None:
        protocol_command.extend(["--checkpoint", str(_resolve(args.checkpoint))])
    if args.training_config is not None:
        protocol_command.extend(["--training-config", str(_resolve(args.training_config))])
    protocol = subprocess.run(
        protocol_command,
        cwd=REPO_ROOT,
        check=False,
    )
    if protocol.returncode != 0:
        return protocol.returncode
    jobs = build_jobs(
        root=root,
        python=args.python,
        checkpoint_override=args.checkpoint,
        training_config_override=args.training_config,
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
        raise ValueError("--gpus must contain at least one GPU id for a CUDA run")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(
                _run_job,
                job,
                log_root=root / "logs",
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
    run_manifest = {
        "schema_version": "auv_goal175_a1_run_manifest.v1",
        "goal_id": "Goal175",
        "phase": "A1",
        "matrix": _relative(MATRIX_PATH),
        "matrix_sha256": _sha256(MATRIX_PATH),
        "arms_sha256": _sha256(ARMS_PATH),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "device": args.device,
        "dtype": args.dtype,
        "gpus": gpus,
        "max_parallel": args.max_parallel,
        "job_count": len(jobs),
        "jobs": results,
    }
    run_manifest_path = root / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    failed = [item for item in results if item["status"] not in {"ok", "skipped_existing"}]
    if failed:
        print(json.dumps({"failed": failed}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    for command in (
        [args.python, str(AUDITOR), "--root", str(root)],
        [args.python, str(ANALYZER), "--root", str(root)],
    ):
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
    print(f"Goal175 A1 complete and audited: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
