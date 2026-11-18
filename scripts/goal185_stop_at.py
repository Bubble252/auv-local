#!/usr/bin/env python3
"""Stop this project's training jobs at a wall-clock deadline.

Only processes whose /proc/<pid>/cmdline still contains
``train_batch_lowlevel_ppo.py`` and the expected output path are signalled.
After exit, writes an audit file with the last iteration record and the newest
model_*.pt so a partial run stays traceable.

Example:
  python goal185_stop_at.py --at 2026-11-18T23:55:00 \
    --job pe2e_bcft:1011579:/data/.../full_wave1/pe2e_bcft_seed86201 \
    --job pe2e_bcres:1011764:/data/.../full_wave1/pe2e_bcres_seed86201
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path


def log(path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_expected(pid: int, output: Path) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, ProcessLookupError):
        return False
    return "train_batch_lowlevel_ppo.py" in cmdline and str(output) in cmdline


def last_iteration(run_dir: Path) -> int | None:
    path = run_dir / "iteration_records.jsonl"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - (1 << 16)))
            lines = handle.read().decode("utf-8", errors="replace").strip().splitlines()
        for line in reversed(lines):
            try:
                return int(json.loads(line)["iteration"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    except FileNotFoundError:
        return None
    return None


def newest_checkpoint(run_dir: Path) -> Path | None:
    candidates = [
        path for path in run_dir.glob("model_*.pt") if path.name != "model_initial.pt"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def stop(pid: int, grace: int, log_path: Path) -> int:
    log(log_path, f"SIGINT pid={pid}")
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return 0
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return 0
        time.sleep(2)
    log(log_path, f"SIGTERM pid={pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return 0
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return 0
        time.sleep(1)
    log(log_path, f"SIGKILL pid={pid}")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return -9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--at", required=True, help="ISO local time, e.g. 2026-11-18T23:55:00")
    parser.add_argument("--job", action="append", required=True, help="name:pid:run_dir")
    parser.add_argument("--grace-seconds", type=int, default=180)
    parser.add_argument("--log", type=Path, default=Path("goal185_stop_at.log"))
    args = parser.parse_args()

    deadline = datetime.fromisoformat(args.at)
    jobs = []
    for spec in args.job:
        name, pid, run_dir = spec.split(":", 2)
        jobs.append((name, int(pid), Path(run_dir)))

    log(args.log, f"armed at={deadline.isoformat()} jobs={[(n, p) for n, p, _ in jobs]}")
    while True:
        remaining = (deadline - datetime.now()).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(30.0, max(1.0, remaining)))

    for name, pid, run_dir in jobs:
        if not is_expected(pid, run_dir):
            log(args.log, f"skip {name} pid={pid}: process not matching expected run")
            continue
        started = datetime.now().isoformat(timespec="seconds")
        log(args.log, f"stopping {name} pid={pid}")
        rc = stop(pid, args.grace_seconds, args.log)
        checkpoint = newest_checkpoint(run_dir)
        audit = {
            "name": name,
            "pid": pid,
            "run_dir": str(run_dir),
            "stopped_at": started,
            "deadline": deadline.isoformat(),
            "return_code": rc,
            "last_iteration": last_iteration(run_dir),
            "newest_checkpoint": None if checkpoint is None else str(checkpoint),
            "newest_checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
            "final_pt_exists": (run_dir / "final.pt").exists(),
            "note": (
                "SIGINT/SIGTERM stop; trainer has no graceful signal handler, "
                "so the newest periodic model_*.pt and flushed jsonl records are "
                "the authoritative partial state."
            ),
        }
        audit_path = run_dir / "stopped_by_deadline_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        log(args.log, f"audit {name} -> {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
