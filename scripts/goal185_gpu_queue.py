#!/usr/bin/env python3
"""Goal185 GPU-safe training queue.

Safety rules:
- only launch on a GPU when free memory covers this job's declared need plus
  a margin that is reserved for other users;
- other users' current usage must stay below --other-max-mb;
- every job runs with a hard allocator cap (AUV_GPU_MEM_LIMIT_MB), so it cannot
  take memory that was not free at launch;
- if other users grow into the reserved margin, or free memory falls below the
  floor, this queue stops its own job first. It never signals another user's
  process.
- deadlines stop this queue's own job with SIGINT, then SIGTERM, then SIGKILL.

Queue file (JSON list):
[
  {
    "name": "mod3_seed86201",
    "config": "configs/lowlevel/goal185/train_formal_mod3_...json",
    "output": "/data/.../full_wave/mod3/pe2e_bcres_seed86201",
    "seed": 86201,
    "iterations": 2600,
    "need_mb": 22000,
    "deadline": "2026-11-19T12:00:00",
    "gpus": [0, 1],
    "extra_args": ["--curriculum-eval-stage-boundaries", "..."]
  }
]
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

LOCK_DIR = Path("/tmp/auv_gpu_queue_locks")


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def pid_user(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Uid:"):
                    return pwd.getpwuid(int(line.split()[1])).pw_name
    except (FileNotFoundError, ProcessLookupError, PermissionError, KeyError):
        pass
    return "unknown"


def gpu_snapshot() -> dict[int, dict[str, Any]]:
    gpus: dict[int, dict[str, Any]] = {}
    for line in run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines():
        index, uuid, total, used = [part.strip() for part in line.split(",")]
        gpus[int(index)] = {
            "index": int(index),
            "uuid": uuid,
            "total_mb": int(float(total)),
            "used_mb": int(float(used)),
            "processes": [],
        }
    try:
        proc_lines = run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    except subprocess.CalledProcessError:
        proc_lines = []
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    for line in proc_lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        uuid, pid, memory = parts
        try:
            memory_mb = int(float(memory))
        except ValueError:
            memory_mb = 0
        by_uuid.setdefault(uuid, []).append(
            {"pid": int(pid), "mem_mb": memory_mb, "user": pid_user(int(pid))}
        )
    me = pwd.getpwuid(os.getuid()).pw_name
    for gpu in gpus.values():
        procs = by_uuid.get(gpu["uuid"], [])
        gpu["processes"] = procs
        gpu["other_mb"] = sum(p["mem_mb"] for p in procs if p["user"] != me)
        known = sum(p["mem_mb"] for p in procs)
        gpu["other_mb"] += max(0, gpu["used_mb"] - known)
        gpu["free_mb"] = gpu["total_mb"] - gpu["used_mb"]
        gpu["me"] = me
    return gpus


def pid_mem_mb(gpu_state: dict[str, Any], pid: int) -> int:
    for proc in gpu_state.get("processes", []):
        if proc["pid"] == pid:
            return int(proc["mem_mb"])
    return 0


def active_ramp_mb(running: list[dict[str, Any]], gpu: int, gpu_state: dict[str, Any]) -> int:
    """Memory that our own queue jobs still expect to grow into."""
    total = 0
    for entry in running:
        if entry["gpu"] != gpu:
            continue
        current = pid_mem_mb(gpu_state, entry["proc"].pid)
        total += max(0, int(entry["job"]["need_mb"]) - current)
    return total


def parse_gpus(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def lock_path(gpu: int) -> Path:
    return LOCK_DIR / f"gpu{gpu}.lock"


def read_lock(gpu: int) -> dict[str, Any] | None:
    path = lock_path(gpu)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    pid = int(payload.get("pid", -1))
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        path.unlink(missing_ok=True)
        return None
    return payload


def write_lock(gpu: int, payload: dict[str, Any]) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path(gpu).write_text(json.dumps(payload), encoding="utf-8")


def release_lock(gpu: int) -> None:
    lock_path(gpu).unlink(missing_ok=True)


def stop_job(proc: subprocess.Popen[Any], grace_s: int, log) -> int:
    pgid = proc.pid
    log(f"stop: SIGINT pgid={pgid}")
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return int(proc.poll() or 0)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return int(proc.returncode or 0)
        time.sleep(2)
    log("stop: escalation SIGTERM")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return int(proc.poll() or 0)
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            return int(proc.returncode or 0)
        time.sleep(1)
    log("stop: escalation SIGKILL")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=30)
    return int(proc.returncode or -9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--python", default=os.environ.get("AUV_PYTHON", sys.executable))
    parser.add_argument("--gpus", default="0-7")
    parser.add_argument("--other-max-mb", type=int, default=4096)
    parser.add_argument("--margin-mb", type=int, default=2048)
    parser.add_argument("--preempt-floor-mb", type=int, default=512)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--grace-seconds", type=int, default=120)
    parser.add_argument("--state", type=Path, default=Path("goal185_queue_state.json"))
    parser.add_argument("--log", type=Path, default=Path("goal185_queue.log"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    allowed = parse_gpus(args.gpus)
    jobs = json.loads(args.queue.read_text(encoding="utf-8"))
    for job in jobs:
        job.setdefault("need_mb", 22000)
        job.setdefault("iterations", 2600)
        job.setdefault("extra_args", [])
        job.setdefault("gpus", allowed)
        job.setdefault("status", "pending")

    def log(message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        if not args.dry_run:
            with args.log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    pending = [j for j in jobs if j["status"] == "pending"]
    running: list[dict[str, Any]] = []
    log(f"queue start: jobs={len(pending)} gpus={allowed} dry_run={args.dry_run}")

    while pending or running:
        snapshot = gpu_snapshot()
        for gpu in allowed:
            if gpu not in snapshot:
                continue
            state = snapshot[gpu]
            log(
                f"gpu{gpu}: free={state['free_mb']}MiB other={state['other_mb']}MiB "
                f"procs={[(p['user'], p['mem_mb']) for p in state['processes']]}"
            )
        if args.dry_run:
            for job in pending:
                for gpu in job["gpus"]:
                    if gpu not in snapshot:
                        continue
                    state = snapshot[gpu]
                    available = state["free_mb"]
                    eligible = (
                        state["other_mb"] <= args.other_max_mb
                        and available >= job["need_mb"] + args.margin_mb
                    )
                    log(
                        f"dry-run {job['name']} gpu{gpu}: eligible={eligible} "
                        f"need={job['need_mb']} free={state['free_mb']} "
                        f"other={state['other_mb']}"
                    )
            return 0

        now = datetime.now()
        for entry in list(running):
            proc = entry["proc"]
            code = proc.poll()
            if code is not None:
                log(f"job {entry['job']['name']} exited rc={code}")
                if args.stop_on_failure and code != 0:
                    log("stop-on-failure: abort queue")
                    return 1
                if entry.get("status") == "running" and code == 0:
                    entry["job"]["status"] = "completed"
                elif entry.get("status") == "running":
                    entry["job"]["status"] = "failed"
                release_lock(entry["gpu"])
                running.remove(entry)
                continue

            job = entry["job"]
            gpu_state = snapshot.get(entry["gpu"])
            reason = None
            if job.get("deadline"):
                deadline = datetime.fromisoformat(job["deadline"])
                if now >= deadline:
                    reason = "deadline"
            if gpu_state is not None:
                if gpu_state["free_mb"] < args.preempt_floor_mb:
                    reason = "free-floor"
                elif gpu_state["other_mb"] > entry["other_baseline_mb"] + args.margin_mb:
                    reason = "other-growth"
            if reason:
                log(f"job {job['name']} preempt reason={reason}")
                entry["status"] = "stopping"
                code = stop_job(proc, args.grace_seconds, log)
                log(f"job {job['name']} stopped rc={code} reason={reason}")
                job["status"] = "stopped"
                job["stop_reason"] = reason
                job["stop_rc"] = code
                release_lock(entry["gpu"])
                running.remove(entry)

        for job in list(pending):
            chosen = None
            for gpu in job["gpus"]:
                if gpu not in allowed or gpu not in snapshot:
                    continue
                state = snapshot[gpu]
                if read_lock(gpu) is not None:
                    continue
                if state["other_mb"] > args.other_max_mb:
                    continue
                available = state["free_mb"] - active_ramp_mb(running, gpu, state)
                if available < job["need_mb"] + args.margin_mb:
                    continue
                chosen = gpu
                break
            if chosen is None:
                continue
            state = snapshot[chosen]
            cap_mb = (
                state["free_mb"]
                - active_ramp_mb(running, chosen, state)
                - args.margin_mb
            )
            if cap_mb < job["need_mb"]:
                continue
            wrapper = Path(__file__).with_name("_gpu_mem_cap.py")
            command = [
                args.python,
                str(wrapper),
                str(Path("scripts/train_batch_lowlevel_ppo.py")),
                "--training-config",
                job["config"],
                "--iterations",
                str(job["iterations"]),
                "--output",
                job["output"],
                "--seed",
                str(job["seed"]),
                *job["extra_args"],
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(chosen)
            env["AUV_GPU_MEM_LIMIT_MB"] = str(cap_mb)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            log(
                f"launch {job['name']} gpu{chosen} cap={cap_mb}MiB "
                f"other={state['other_mb']}MiB free={state['free_mb']}MiB"
            )
            proc = subprocess.Popen(
                command,
                env=env,
                start_new_session=True,
                stdout=open(Path(job["output"]).parent / f"{job['name']}.log", "ab"),
                stderr=subprocess.STDOUT,
            )
            write_lock(
                chosen,
                {"pid": proc.pid, "job": job["name"], "started": now.isoformat()},
            )
            job["status"] = "running"
            job["gpu"] = chosen
            running.append(
                {
                    "job": job,
                    "gpu": chosen,
                    "proc": proc,
                    "other_baseline_mb": state["other_mb"],
                    "status": "running",
                }
            )
            pending.remove(job)

        args.state.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
        if pending or running:
            time.sleep(args.poll_seconds)

    log("queue done")
    args.state.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
