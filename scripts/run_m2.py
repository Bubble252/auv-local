from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.patches import Circle, Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from auv_core.contracts import Obstacle
from auv_core.guidance import ThreeDAlosGuidance

from auv_local.closed_loop import run_m2_case
from auv_local.execution import PvsRemus100ReferenceExecutor
from auv_local.line_detour import LineDetourPlanner
from auv_local.scenarios import (
    M2Config,
    M2Scenario,
    config_to_dict,
    load_m2_config,
    m2_scenarios,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M2 PVS/ALOS local-planning matrix")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "m2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "m2",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("s1", "s2", "s3"),
        help="run only the named scenario; may be repeated",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="override configured seeds; may be repeated",
    )
    parser.add_argument(
        "--repeat-check",
        action="store_true",
        help="run every case twice and require exact deterministic-field equality",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _run_one(
    case: M2Scenario,
    config: M2Config,
    seed: int,
) -> Any:
    planner = LineDetourPlanner(
        bounds_ned_m=case.scenario.world.obstacle_map.bounds_ned_m,
        vehicle=case.scenario.vehicle,
        rollout=config.rollout,
        speed_mps=case.nominal_speed_mps,
    )
    guidance = ThreeDAlosGuidance(
        config.horizontal_lookahead_m,
        config.vertical_lookahead_m,
        case.nominal_speed_mps,
        current_field=case.scenario.world.current_field,
        compensate_current=False,
    )
    executor = PvsRemus100ReferenceExecutor(
        case.scenario,
        sample_time_s=config.pvs_sample_time_s,
    )
    return run_m2_case(case, config, planner, guidance, executor, seed=seed)


def _run_summary(result: Any, *, repeat_match: bool | None) -> dict[str, Any]:
    planning_times = [replan.planning_time_s for replan in result.replans]
    return {
        "case_name": result.case_name,
        "scenario_name": result.scenario_name,
        "seed": result.seed,
        "termination_reason": result.termination_reason,
        "metrics": result.metrics.summary_dict(),
        "mean_planning_time_s": float(np.mean(planning_times)) if planning_times else 0.0,
        "max_planning_time_s": float(np.max(planning_times)) if planning_times else 0.0,
        "planning_budget_violations": sum(1 for replan in result.replans if replan.budget_exceeded),
        "repeat_match": repeat_match,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_name",
        "scenario_name",
        "seed",
        "termination_reason",
        "success",
        "goal_reached",
        "collision",
        "out_of_bounds",
        "completion_time_s",
        "path_length_m",
        "min_clearance_m",
        "horizontal_rmse_m",
        "vertical_rmse_m",
        "constraint_violations",
        "reference_expiry_count",
        "replan_count",
        "mean_planning_time_s",
        "max_planning_time_s",
        "repeat_match",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "case_name": row["case_name"],
                    "scenario_name": row["scenario_name"],
                    "seed": row["seed"],
                    "termination_reason": row["termination_reason"],
                    "success": metrics["success"],
                    "goal_reached": metrics["goal_reached"],
                    "collision": metrics["collision"],
                    "out_of_bounds": metrics["out_of_bounds"],
                    "completion_time_s": metrics["completion_time_s"],
                    "path_length_m": metrics["path_length_m"],
                    "min_clearance_m": metrics["min_clearance_m"],
                    "horizontal_rmse_m": metrics["horizontal_rmse_m"],
                    "vertical_rmse_m": metrics["vertical_rmse_m"],
                    "constraint_violations": metrics["constraint_violations"],
                    "reference_expiry_count": metrics["reference_expiry_count"],
                    "replan_count": metrics["replan_count"],
                    "mean_planning_time_s": row["mean_planning_time_s"],
                    "max_planning_time_s": row["max_planning_time_s"],
                    "repeat_match": row["repeat_match"],
                }
            )


def _draw_obstacle(axis: Any, obstacle: Obstacle) -> None:
    center = obstacle.center_ned_m
    if obstacle.kind == "box":
        assert obstacle.half_extents_m is not None
        patch = Rectangle(
            (center[0] - obstacle.half_extents_m[0], center[1] - obstacle.half_extents_m[1]),
            2.0 * obstacle.half_extents_m[0],
            2.0 * obstacle.half_extents_m[1],
            facecolor="0.65",
            edgecolor="0.35",
            alpha=0.55,
        )
    else:
        assert obstacle.radius_m is not None
        patch = Circle(
            (center[0], center[1]),
            obstacle.radius_m,
            facecolor="0.55",
            edgecolor="0.3",
            alpha=0.55,
        )
    axis.add_patch(patch)


def _plot_scenario(case: M2Scenario, results: list[Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    for result in results:
        positions = np.vstack(
            (
                result.initial_state.position_ned_m,
                *(sample.state.position_ned_m for sample in result.samples),
            )
        )
        label = f"seed {result.seed}: {result.termination_reason}"
        axis.plot(positions[:, 0], positions[:, 1], linewidth=1.6, label=label)
    for obstacle in case.scenario.world.obstacle_map.obstacles:
        _draw_obstacle(axis, obstacle)
    axis.scatter(
        [case.scenario.start.position_ned_m[0]],
        [case.scenario.start.position_ned_m[1]],
        marker="o",
        color="tab:green",
        label="start",
        zorder=5,
    )
    axis.scatter(
        [case.scenario.goal_ned_m[0]],
        [case.scenario.goal_ned_m[1]],
        marker="*",
        s=150,
        color="tab:red",
        label="goal",
        zorder=5,
    )
    bounds = case.scenario.world.obstacle_map.bounds_ned_m
    axis.set_xlim(float(bounds[0, 0]), float(bounds[0, 1]))
    axis.set_ylim(float(bounds[1, 0]), float(bounds[1, 1]))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("NED x [m]")
    axis.set_ylabel("NED y [m]")
    axis.set_title(f"{case.scenario.name}: top-down PVS trajectories")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    config = load_m2_config(args.config)
    scenarios = m2_scenarios(config)
    if args.scenario:
        selected = set(args.scenario)
        scenarios = tuple(case for case in scenarios if case.key in selected)
    seeds = tuple(args.seed) if args.seed else config.seeds
    args.output.mkdir(parents=True, exist_ok=True)
    traces_dir = args.output / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    results: list[Any] = []
    rows: list[dict[str, Any]] = []
    for case in scenarios:
        for seed in seeds:
            print(f"running {case.key} seed={seed}", flush=True)
            result = _run_one(case, config, seed)
            repeat_match: bool | None = None
            if args.repeat_check:
                repeated = _run_one(case, config, seed)
                repeat_match = result.deterministic_dict() == repeated.deterministic_dict()
                if not repeat_match:
                    raise RuntimeError(f"non-deterministic M2 result for {case.key} seed={seed}")
            results.append(result)
            rows.append(_run_summary(result, repeat_match=repeat_match))
            _write_json(traces_dir / f"{case.key}_seed{seed}.json", result.log_dict())
            metrics = result.metrics
            print(
                f"  {result.termination_reason}: goal={metrics.goal_reached} "
                f"collision={metrics.collision} t={metrics.duration_s:.2f}s "
                f"path={metrics.path_length_m:.2f}m clear={metrics.min_clearance_m:.3f}m",
                flush=True,
            )

    failures = [row for row in rows if not row["metrics"]["success"]]
    summary = {
        "config": config_to_dict(config),
        "run_count": len(rows),
        "success_count": sum(1 for row in rows if row["metrics"]["success"]),
        "repeat_check": bool(args.repeat_check),
        "runs": rows,
        "failures": failures,
    }
    _write_json(args.output / "summary.json", summary)
    _write_json(args.output / "failures.json", failures)
    _write_summary_csv(args.output / "summary.csv", rows)
    if not args.no_plots:
        for case in scenarios:
            case_results = [
                result for result in results if result.scenario_name == case.scenario.name
            ]
            if case_results:
                _plot_scenario(case, case_results, args.output / f"{case.scenario.name}.png")
    print(
        f"completed {len(rows)} runs, success {summary['success_count']}, failures {len(failures)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
