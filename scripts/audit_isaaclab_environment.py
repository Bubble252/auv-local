from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_NAMES = (
    "isaacsim",
    "isaacsim-core",
    "isaacsim-kernel",
    "isaacsim-rl",
    "isaaclab",
    "isaaclab-tasks",
    "torch",
    "torchvision",
    "numpy",
    "gymnasium",
    "stable-baselines3",
    "warp-lang",
)

TRAINING_PATTERN = "train_p1_oceansim[.]py|run_p1_ablation_sequence[.]sh"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the isolated IsaacLab environment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式"),
    )
    parser.add_argument(
        "--oceansim-python",
        type=Path,
        default=Path("/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python"),
    )
    parser.add_argument(
        "--isaaclab-home",
        type=Path,
        default=Path("/data/yhx/underwater-robotics/simulator/IsaacLab"),
    )
    parser.add_argument(
        "--isaaclab-venv",
        type=Path,
        default=Path("/data/yhx/underwater-robotics/envs/venvs/isaaclab"),
    )
    return parser.parse_args()


def _run(command: list[str], timeout: int = 30, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError:
        return {"command": command, "returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"command": command, "returncode": None, "stdout": "", "stderr": "timeout"}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _pip_freeze_fingerprint(python: Path) -> dict[str, Any]:
    result = _run([str(python), "-m", "pip", "freeze"], timeout=60)
    text = result["stdout"]
    return {
        "returncode": result["returncode"],
        "package_count": len([line for line in text.splitlines() if line.strip()]),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
        "stderr": result["stderr"],
    }


def _python_environment(python: Path) -> dict[str, Any]:
    code = f"""
import importlib.metadata as metadata
import json
import platform
import sys
names = {PACKAGE_NAMES!r}
out = {{
    "executable": sys.executable,
    "version": sys.version,
    "platform": platform.platform(),
}}
for name in names:
    try:
        out[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        out[name] = None
print(json.dumps(out, ensure_ascii=False))
"""
    result = _run([str(python), "-c", code], timeout=60)
    payload: dict[str, Any] = {
        "command": result["command"],
        "returncode": result["returncode"],
        "stderr": result["stderr"],
    }
    if result["stdout"]:
        try:
            payload.update(json.loads(result["stdout"]))
        except json.JSONDecodeError:
            payload["raw_stdout"] = result["stdout"]
    return payload


def _pip_check(python: Path) -> dict[str, Any]:
    result = _run([str(python), "-m", "pip", "check"], timeout=60)
    return {
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def _gpu_inventory() -> dict[str, Any]:
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    gpus: list[dict[str, Any]] = []
    if query["returncode"] == 0:
        for row in csv.reader(query["stdout"].splitlines()):
            if len(row) != 7:
                continue
            gpus.append(
                {
                    "index": int(row[0]),
                    "name": row[1].strip(),
                    "uuid": row[2].strip(),
                    "memory_total_mib": int(row[3]),
                    "memory_used_mib": int(row[4]),
                    "utilization_gpu_percent": int(row[5]),
                    "driver_version": row[6].strip(),
                }
            )
    compute_query = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    compute_processes: list[dict[str, Any]] = []
    if compute_query["returncode"] == 0:
        for row in csv.reader(compute_query["stdout"].splitlines()):
            if len(row) != 3:
                continue
            compute_processes.append(
                {
                    "pid": int(row[0]),
                    "process_name": row[1].strip(),
                    "used_memory_mib": int(row[2]),
                }
            )
    return {
        "query": {"returncode": query["returncode"], "stderr": query["stderr"]},
        "gpus": gpus,
        "compute_processes": compute_processes,
    }


def _training_processes() -> dict[str, Any]:
    result = _run(["pgrep", "-af", TRAINING_PATTERN], timeout=10)
    lines = [line for line in result["stdout"].splitlines() if line.strip()]
    return {
        "pattern": TRAINING_PATTERN,
        "returncode": result["returncode"],
        "lines": lines,
        "running": bool(lines),
    }


def _disk(path: str) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gib": usage.total / (1024**3),
        "used_gib": usage.used / (1024**3),
        "free_gib": usage.free / (1024**3),
        "used_percent": 100.0 * usage.used / usage.total if usage.total else None,
    }


def _git_state(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"path": str(path), "exists": False}
    head = _run(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=20)
    branch = _run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"], timeout=20)
    describe = _run(
        ["git", "-C", str(path), "describe", "--tags", "--always", "--dirty"], timeout=20
    )
    remote = _run(["git", "-C", str(path), "config", "--get", "remote.origin.url"], timeout=20)
    return {
        "path": str(path),
        "exists": True,
        "head": head["stdout"] or None,
        "branch": branch["stdout"] or None,
        "describe": describe["stdout"] or None,
        "remote": remote["stdout"] or None,
    }


def _existing_install(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "git": _git_state(path),
        "version_file": (path / "VERSION").read_text(encoding="utf-8").strip()
        if (path / "VERSION").exists()
        else None,
        "pyvenv_cfg": (path / "pyvenv.cfg").read_text(encoding="utf-8")
        if (path / "pyvenv.cfg").exists()
        else None,
    }


def main() -> int:
    args = _parse_args()
    started = time.time()
    payload: dict[str, Any] = {
        "schema": "auv_isaaclab_environment_audit.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "repo_path": str(args.repo),
        "python_running_audit": sys.executable,
        "system": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "load_average": list(os.getloadavg()),
            "memory_total_gib": _memory_total_gib(),
        },
        "gpu": _gpu_inventory(),
        "training_processes": _training_processes(),
        "repo_git": _git_state(args.repo),
        "oceansim_python": _python_environment(args.oceansim_python),
        "oceansim_pip_check": _pip_check(args.oceansim_python),
        "oceansim_pip_freeze": _pip_freeze_fingerprint(args.oceansim_python),
        "existing_isaaclab_home": _existing_install(args.isaaclab_home),
        "existing_isaaclab_venv": _existing_install(args.isaaclab_venv),
        "disk": {
            "root": _disk("/"),
            "data": _disk("/data") if Path("/data").exists() else None,
            "repo": _disk(str(args.repo)) if args.repo.exists() else None,
        },
        "compatibility_matrix": {
            "isaaclab_v2.3.x": ["isaac_sim_4.5", "isaac_sim_5.0", "isaac_sim_5.1"],
            "isaaclab_v2.2.x": ["isaac_sim_4.5", "isaac_sim_5.0"],
            "selected": {
                "isaaclab": "v2.2.1",
                "isaac_sim": "5.0.0.0",
                "python": "3.11",
                "torch": "2.7.0+cu128",
            },
            "selection_reason": "v2.2.X explicitly supports Isaac Sim 5.0; v2.3.2 pins Pillow 11.3.0, which conflicts with isaacsim-kernel 5.0.0.0 requiring 11.2.1.",
            "source": "https://github.com/isaac-sim/IsaacLab README compatibility table",
        },
        "environment_variables": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMNI_KIT_ACCEPT_EULA",
                "OV_CACHE_DIR",
                "OV_DATA_DIR",
                "PIP_CACHE_DIR",
                "TMPDIR",
            )
        },
        "elapsed_s": time.time() - started,
        "status": "COMPLETE",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _memory_total_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            kib = float(line.split()[1])
            return kib / (1024**2)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
