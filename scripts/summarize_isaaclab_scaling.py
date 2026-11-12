from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

PRIMARY_COUNTS = (1, 64, 256, 1024)
FILENAME_PATTERN = re.compile(r"num_envs_(\d+)(?:_repeat_(\d+))?\.json$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate IsaacLab scaling probe results")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--required-repeats", type=int, default=3)
    parser.add_argument("--required-steps", type=int, default=200)
    return parser.parse_args()


def _load_runs(directory: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("num_envs_*.json")):
        match = FILENAME_PATTERN.search(path.name)
        if not match:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payload = {
                "status": "FAIL_INVALID_JSON",
                "error": repr(exc),
                "num_envs": int(match.group(1)),
            }
        payload["_path"] = str(path)
        payload["_repeat_from_filename"] = int(match.group(2)) if match.group(2) else 0
        runs.append(payload)
    return runs


def _metric(run: dict[str, Any], name: str) -> float | None:
    value = run.get(name)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    gpu = run.get("gpu")
    if isinstance(gpu, dict) and name in gpu:
        value = gpu.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _values(runs: list[dict[str, Any]], name: str) -> list[float]:
    return [value for run in runs if (value := _metric(run, name)) is not None]


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _is_pass(run: dict[str, Any], required_steps: int) -> bool:
    return (
        run.get("status") == "PASS"
        and int(run.get("steps_completed") or 0) >= required_steps
        and bool(run.get("all_observation_finite", False))
        and int(run.get("num_envs_actual") or 0) == int(run.get("num_envs") or -1)
    )


def _failure(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": Path(run["_path"]).name,
        "repeat": run.get("_repeat_from_filename"),
        "status": run.get("status"),
        "error_stage": run.get("error_stage"),
        "error": run.get("error"),
        "exception_type": run.get("exception_type"),
    }


def _aggregate(
    runs: list[dict[str, Any]], required_repeats: int, required_steps: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_count: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        count = int(run.get("num_envs") or 0)
        by_count.setdefault(count, []).append(run)

    configs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for count in sorted(by_count):
        group = by_count[count]
        passed = [run for run in group if _is_pass(run, required_steps)]
        failed = [run for run in group if not _is_pass(run, required_steps)]
        oom_count = sum(1 for run in group if run.get("status") == "FAIL_OOM")
        nan_count = sum(1 for run in group if run.get("status") == "FAIL_NONFINITE_OBSERVATION")
        downgrade_count = sum(
            1
            for run in group
            if int(run.get("num_envs_actual") or -1) != int(run.get("num_envs") or 0)
        )
        actual_counts = sorted(
            {
                int(run.get("num_envs_actual"))
                for run in group
                if run.get("num_envs_actual") is not None
            }
        )
        config = {
            "num_envs_requested": count,
            "num_envs_actual_observed": actual_counts,
            "attempted": len(group),
            "passed": len(passed),
            "failed": len(failed),
            "required_repeats": required_repeats,
            "required_steps": required_steps,
            "repeat_gate": len(passed) >= required_repeats,
            "all_runs_passed": not failed,
            "oom_count": oom_count,
            "nan_count": nan_count,
            "silent_downgrade_count": downgrade_count,
            "metrics": {
                name: _summary(_values(passed, name))
                for name in (
                    "env_steps_per_s",
                    "control_steps_per_s",
                    "time_to_first_step_s",
                    "app_ready_s",
                    "reset_s",
                    "step_latency_p50_ms",
                    "step_latency_p95_ms",
                    "gpu_memory_peak_mib",
                    "gpu_memory_peak_delta_mib",
                    "gpu_peak_allocated_mib",
                    "cpu_time_ratio",
                    "process_peak_rss_mib",
                )
            },
            "status": "PASS"
            if len(passed) >= required_repeats and not failed
            else "PARTIAL"
            if passed
            else "FAIL",
        }
        configs.append(config)
        for run in failed:
            item = _failure(run)
            item["num_envs_requested"] = count
            failures.append(item)
    return configs, failures


def _by_count(configs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(config["num_envs_requested"]): config for config in configs}


def _write_csv(path: Path, configs: list[dict[str, Any]]) -> None:
    columns = [
        "num_envs_requested",
        "num_envs_actual_observed",
        "attempted",
        "passed",
        "failed",
        "required_repeats",
        "repeat_gate",
        "status",
        "env_steps_per_s_mean",
        "env_steps_per_s_min",
        "env_steps_per_s_max",
        "control_steps_per_s_mean",
        "step_latency_p50_ms_mean",
        "step_latency_p95_ms_mean",
        "time_to_first_step_s_mean",
        "app_ready_s_mean",
        "reset_s_mean",
        "gpu_memory_peak_mib_mean",
        "gpu_memory_peak_delta_mib_mean",
        "gpu_peak_allocated_mib_mean",
        "cpu_time_ratio_mean",
        "process_peak_rss_mib_mean",
        "oom_count",
        "nan_count",
        "silent_downgrade_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for config in configs:
            metrics = config["metrics"]
            row: dict[str, Any] = {
                "num_envs_requested": config["num_envs_requested"],
                "num_envs_actual_observed": "|".join(
                    str(value) for value in config["num_envs_actual_observed"]
                ),
                "attempted": config["attempted"],
                "passed": config["passed"],
                "failed": config["failed"],
                "required_repeats": config["required_repeats"],
                "repeat_gate": config["repeat_gate"],
                "status": config["status"],
                "oom_count": config["oom_count"],
                "nan_count": config["nan_count"],
                "silent_downgrade_count": config["silent_downgrade_count"],
            }
            for name in columns:
                if name.endswith("_mean"):
                    metric_name = name[: -len("_mean")]
                    row[name] = metrics.get(metric_name, {}).get("mean")
                elif name.endswith("_min"):
                    metric_name = name[: -len("_min")]
                    row[name] = metrics.get(metric_name, {}).get("min")
                elif name.endswith("_max"):
                    metric_name = name[: -len("_max")]
                    row[name] = metrics.get(metric_name, {}).get("max")
            writer.writerow(row)


def main() -> int:
    args = _parse_args()
    runs = _load_runs(args.input)
    if not runs:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "schema": "auv_isaaclab_scaling_report.v1",
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "status": "NO_DATA",
                    "input": str(args.input),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"no probe JSON files found in {args.input}")
        return 1

    configs, failures = _aggregate(runs, args.required_repeats, args.required_steps)
    indexed = _by_count(configs)
    base = indexed.get(1, {})
    large = indexed.get(1024, {})
    base_throughput = base.get("metrics", {}).get("env_steps_per_s", {}).get("mean")
    large_throughput = large.get("metrics", {}).get("env_steps_per_s", {}).get("mean")
    speedup = (
        large_throughput / base_throughput
        if base_throughput and large_throughput and base_throughput > 0
        else None
    )
    efficiency = speedup / 1024.0 if speedup is not None else None
    primary_present = all(count in indexed for count in PRIMARY_COUNTS)
    primary_gates = {
        count: bool(indexed.get(count, {}).get("repeat_gate"))
        and indexed.get(count, {}).get("status") == "PASS"
        for count in PRIMARY_COUNTS
    }
    gates = {
        "primary_configs_present": primary_present,
        "primary_configs_pass": all(primary_gates.values()),
        "config_1024_passed": primary_gates.get(1024, False),
        "config_1024_no_oom": int(large.get("oom_count") or 0) == 0,
        "config_1024_no_nan": int(large.get("nan_count") or 0) == 0,
        "config_1024_no_silent_downgrade": int(large.get("silent_downgrade_count") or 0) == 0,
        "config_1024_actual_envs_ok": 1024 in large.get("num_envs_actual_observed", []),
    }
    gates["engineering_gate_pass"] = all(gates.values())
    report = {
        "schema": "auv_isaaclab_scaling_report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(args.input),
        "required_repeats": args.required_repeats,
        "required_steps": args.required_steps,
        "configs": configs,
        "primary_gates": primary_gates,
        "gates": gates,
        "scaling": {
            "throughput_1_env_steps_per_s": base_throughput,
            "throughput_1024_env_steps_per_s": large_throughput,
            "speedup_1024_vs_1": speedup,
            "ideal_linear_speedup": 1024,
            "parallel_efficiency_1024_vs_1": efficiency,
            "linear_claim_allowed": efficiency is not None and efficiency >= 0.9,
            "note": "Linear claim is allowed only when parallel efficiency is at least 0.9; otherwise report measured speedup only.",
        },
        "failures": failures,
        "status": "COMPLETE" if gates["engineering_gate_pass"] else "PARTIAL",
    }
    _write_csv(args.csv, configs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if gates["engineering_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
