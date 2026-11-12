from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

SCALING_NAME = re.compile(r"^num_envs_(\d+)\.json$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P1 IsaacLab scaling probes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def summarize(input_dir: Path) -> dict[str, Any]:
    scaling: list[dict[str, Any]] = []
    reset_stress: dict[str, Any] | None = None
    failures: list[str] = []
    for path in sorted(input_dir.glob("num_envs_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            failures.append(str(path))
        match = SCALING_NAME.match(path.name)
        if match is not None:
            payload["num_envs"] = int(match.group(1))
            scaling.append(payload)
        elif path.name == "num_envs_64_reset_mixed.json":
            reset_stress = payload

    scaling.sort(key=lambda item: item["num_envs"])
    required_counts = {1, 64, 256, 1024, 4096}
    observed_counts = {int(item["num_envs"]) for item in scaling}
    missing_counts = sorted(required_counts - observed_counts)
    all_pass = not failures and not missing_counts and reset_stress is not None
    if reset_stress is not None and block_status(reset_stress) != "PASS":
        all_pass = False

    return {
        "schema": "auv_p1_isaaclab_scaling_summary.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PASS" if all_pass else "FAIL",
        "required_env_counts": sorted(required_counts),
        "missing_env_counts": missing_counts,
        "failed_files": failures,
        "scaling": [scaling_row(item) for item in scaling],
        "reset_stress": reset_stress_row(reset_stress) if reset_stress is not None else None,
        "max_env_steps_per_s": max(
            (float(item["env_steps_per_s"]) for item in scaling), default=None
        ),
    }


def block_status(payload: dict[str, Any]) -> str:
    return str(payload.get("status", "MISSING"))


def scaling_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_envs": int(payload["num_envs"]),
        "status": payload["status"],
        "control_steps_per_s": float(payload["control_steps_per_s"]),
        "env_steps_per_s": float(payload["env_steps_per_s"]),
        "step_latency_p50_ms": float(payload["step_latency_p50_ms"]),
        "step_latency_p95_ms": float(payload["step_latency_p95_ms"]),
        "steps_completed": int(payload["steps_completed"]),
        "all_observation_finite": bool(payload["all_observation_finite"]),
        "max_group_abs_diff": float(payload["max_group_abs_diff"]),
        "stale_done_steps": int(payload["stale_done_steps"]),
        "gpu_memory_after_mib": payload.get("gpu_memory_after_mib"),
        "gpu_peak_allocated_mib": payload.get("gpu_peak_allocated_mib"),
        "config_sha256": payload.get("config_sha256"),
    }


def reset_stress_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload["status"],
        "num_envs": int(payload["num_envs_requested"]),
        "steps_completed": int(payload["steps_completed"]),
        "terminated_events": int(payload["terminated_events"]),
        "truncated_events": int(payload["truncated_events"]),
        "reset_events": int(payload["reset_events"]),
        "stale_done_steps": int(payload["stale_done_steps"]),
        "max_group_abs_diff": float(payload["max_group_abs_diff"]),
        "max_history_reset_error": float(payload["max_history_reset_error"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "num_envs",
        "status",
        "control_steps_per_s",
        "env_steps_per_s",
        "step_latency_p50_ms",
        "step_latency_p95_ms",
        "steps_completed",
        "all_observation_finite",
        "max_group_abs_diff",
        "stale_done_steps",
        "gpu_memory_after_mib",
        "gpu_peak_allocated_mib",
        "config_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    summary = summarize(args.input)
    write_csv(args.csv, summary["scaling"])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
