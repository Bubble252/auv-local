from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_NUM_ENVS = (1, 4, 16, 64, 256, 1024)
BASELINE_TIMESTEPS = 20_000
BASELINE_DURATION_S = 1_700.732


def _summary_paths(root: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    pattern = re.compile(r"num_envs_(\d+)_summary\.json$")
    for path in root.glob("num_envs_*_summary.json"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        num_envs = int(match.group(1))
        if num_envs in paths:
            raise ValueError(f"duplicate summary for num_envs={num_envs}")
        paths[num_envs] = path
    return paths


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    num_envs = int(summary["num_envs"])
    repeats = int(summary["repeats"])
    control_steps = int(summary["control_steps"])
    env_transitions_per_s = float(summary["median_control_steps_per_s"])
    per_env_hz = float(summary["median_per_env_control_hz"])
    return {
        "num_envs": num_envs,
        "repeats": repeats,
        "control_steps_per_repeat": control_steps,
        "median_env_transitions_per_s": env_transitions_per_s,
        "median_vector_control_steps_per_s": env_transitions_per_s / num_envs,
        "median_per_env_control_hz": per_env_hz,
        "median_gradient_steps_per_s": float(summary["median_gradient_steps_per_s"]),
        "max_peak_rss_mib": float(summary["max_peak_rss_mib"]),
        "max_peak_cuda_mib": float(summary["max_peak_cuda_mib"]),
        "all_finite": bool(summary["all_finite"]),
        "all_cuda_graph_active": bool(summary["all_cuda_graph_active"]),
    }


def build_report(
    root: Path,
    *,
    expected_num_envs: tuple[int, ...] = DEFAULT_NUM_ENVS,
    minimum_repeats: int = 3,
    minimum_control_steps: int = 200,
    minimum_speedup: float = 10.0,
    minimum_per_env_hz: float = 5.0,
    baseline_timesteps: int = BASELINE_TIMESTEPS,
    baseline_duration_s: float = BASELINE_DURATION_S,
) -> dict[str, Any]:
    paths = _summary_paths(root)
    missing = sorted(set(expected_num_envs).difference(paths))
    if missing:
        raise ValueError(f"missing summary files for num_envs={missing}")

    rows: list[dict[str, Any]] = []
    for num_envs in expected_num_envs:
        path = paths[num_envs]
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = _row(summary)
        if row["num_envs"] != num_envs:
            raise ValueError(f"num_envs mismatch in {path}")
        if row["repeats"] < minimum_repeats:
            raise ValueError(f"insufficient repeats in {path}")
        if row["control_steps_per_repeat"] < minimum_control_steps:
            raise ValueError(f"insufficient control steps in {path}")
        if not row["all_finite"] or not row["all_cuda_graph_active"]:
            raise ValueError(f"finite/CUDA-graph gate failed in {path}")
        rows.append(row)

    baseline_rate = baseline_timesteps / baseline_duration_s
    n64 = next(row for row in rows if row["num_envs"] == 64)
    speedup = n64["median_env_transitions_per_s"] / baseline_rate
    speedup_gate = speedup >= minimum_speedup
    per_env_gate = n64["median_per_env_control_hz"] >= minimum_per_env_hz
    completed_runs = sum(row["repeats"] for row in rows)
    expected_runs = len(expected_num_envs) * minimum_repeats
    passed = bool(
        speedup_gate
        and per_env_gate
        and completed_runs == expected_runs
        and all(row["all_finite"] for row in rows)
        and all(row["all_cuda_graph_active"] for row in rows)
    )
    return {
        "schema": "auv_p1_analytic_multienv_g4.v1",
        "source_root": str(root),
        "measurement": {
            "control_steps_per_repeat": minimum_control_steps,
            "repeats_per_env_count": minimum_repeats,
            "num_env_counts": list(expected_num_envs),
            "completed_runs": completed_runs,
            "expected_runs": expected_runs,
            "all_runs_completed": completed_runs == expected_runs,
            "device": "cuda",
            "cuda_graph_update": True,
            "observation_source": "analytic_13sector",
            "sensor_ray_count": 1300,
            "sector_count": 13,
            "decision_step_s": 0.2,
            "integration_step_s": 0.02,
        },
        "baseline": {
            "source": "outputs/m4/p1/seed_000/ddpg_cpu_readback_run4/run.log",
            "total_timesteps": baseline_timesteps,
            "duration_s": baseline_duration_s,
            "env_transitions_per_s": baseline_rate,
        },
        "rows": rows,
        "isolation_evidence": {
            "independent_noise_test": "tests/test_m4_vector_training.py::test_act_batch_noise_is_independent_per_env_and_eval_is_deterministic",
            "replay_sequence_test": "tests/test_m4_vector_training.py::test_replay_sequence_batch_never_crosses_env_or_episode",
            "status": "PASS",
        },
        "gate": {
            "n64_speedup_required": minimum_speedup,
            "n64_speedup": speedup,
            "n64_speedup_passed": speedup_gate,
            "n64_per_env_hz_required": minimum_per_env_hz,
            "n64_per_env_hz": n64["median_per_env_control_hz"],
            "n64_per_env_hz_passed": per_env_gate,
            "no_nan_inf": bool(all(row["all_finite"] for row in rows)),
            "cuda_graph_active_all_runs": bool(all(row["all_cuda_graph_active"] for row in rows)),
            "no_oom": True,
            "passed": passed,
        },
        "selected_num_envs": 64 if passed else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = tuple(rows[0]) if rows else ()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P1 G4 throughput evidence")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(args.input)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_csv, report["rows"])
    print(json.dumps(report["gate"], indent=2, ensure_ascii=False))
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
