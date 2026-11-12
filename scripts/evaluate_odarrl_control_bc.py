from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.bc import BCPolicy
from auv_lowlevel.config import load_config
from auv_lowlevel.evaluate import Rollout, run_rollout
from auv_lowlevel.observation_schema import observation_schema_for_dim
from auv_lowlevel.scripts_support import aggregate_episode_summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate I0 pi0 against the online PD/PI teacher")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lowlevel/odarrl_control_i_bc.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--goal-radius", type=float, default=None)
    parser.add_argument("--b0-summary", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _parse_seeds(text: str | None, start: int, count: int) -> tuple[int, ...]:
    if text is None:
        return tuple(range(start, start + count))
    result: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if "-" in token:
            first, last = token.split("-", maxsplit=1)
            result.extend(range(int(first), int(last) + 1))
        elif token:
            result.append(int(token))
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise ValueError("seeds must be non-empty, unique, and non-negative")
    return tuple(result)


def _rollout_arrays(rollout: Rollout) -> dict[str, np.ndarray]:
    return {
        "t_s": rollout.t_s,
        "position_ned_m": rollout.position_ned_m,
        "reference_point_ned_m": rollout.reference_point_ned_m,
        "horizontal_error_m": rollout.horizontal_error_m,
        "vertical_error_m": rollout.vertical_error_m,
        "error_3d_m": rollout.error_3d_m,
        "progress_m": rollout.progress_m,
        "yaw_error_rad": rollout.yaw_error_rad,
        "water_surge_mps": rollout.water_surge_mps,
        "reference_speed_mps": rollout.reference_speed_mps,
        "observation": rollout.observation,
        "base_command": rollout.base_command,
        "limited_total": rollout.limited_total,
        "actual_actuator": rollout.actual_actuator,
        "reward_total": rollout.reward_total,
        "saturated": rollout.saturated,
        "material_saturated": rollout.material_saturated,
        "residual_clipped": rollout.residual_clipped,
        "constraint_violation_count": rollout.constraint_violation_count,
        "control_loop_latency_s": rollout.control_loop_latency_s,
    }


def _run_pair(
    payload: tuple[str, str, int, int | None, float | None, str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path, checkpoint_path, seed, max_steps, goal_radius, output_text, device = payload
    config = load_config(config_path)
    if max_steps is not None or goal_radius is not None:
        config = replace(
            config,
            plant=replace(
                config.plant,
                max_episode_steps=config.plant.max_episode_steps
                if max_steps is None
                else max_steps,
                goal_radius_m=config.plant.goal_radius_m
                if goal_radius is None
                else goal_radius,
            ),
        )
    teacher = run_rollout(config, seed, base_mode="teacher", expose_current=False)
    schema = observation_schema_for_dim(config.observation.dim)
    policy = BCPolicy(
        checkpoint_path,
        device=device,
        expected_observation_dim=schema.dim,
        expected_observation_schema=schema.version,
        expected_speed_feedback=config.observation.speed_feedback,
        expected_current_contract=config.observation.current_contract,
    )
    student = run_rollout(
        config,
        seed,
        base_mode="frozen",
        base_command_fn=policy,
        expose_current=False,
    )
    output = Path(output_text)
    teacher_dir = output / "per_seed" / "teacher"
    student_dir = output / "per_seed" / "student"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    student_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        teacher_dir / f"seed_{seed:05d}.npz",
        **_rollout_arrays(teacher),
    )
    np.savez_compressed(
        student_dir / f"seed_{seed:05d}.npz",
        **_rollout_arrays(student),
    )
    teacher_summary = teacher.summary()
    student_summary = student.summary()
    pair = {"seed": seed, "teacher": teacher_summary, "student": student_summary}
    (output / "per_seed" / f"seed_{seed:05d}.json").write_text(
        json.dumps(pair, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return teacher_summary, student_summary


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    teacher = report["aggregate"]["teacher"]
    student = report["aggregate"]["student"]
    gate = report["gate"]
    lines = [
        "# I0 teacher-student gate",
        "",
        f"- config_sha256: `{report['config_sha256']}`",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- observation: `{report['observation_dim']}D "
        f"({report['observation_schema']})`",
        f"- seeds: `{report['seeds']}`",
        "",
        "| metric | teacher | student | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "success_rate",
        "three_d_rmse_m_mean",
        "horizontal_rmse_m_mean",
        "vertical_rmse_m_mean",
        "saturation_rate_mean",
        "material_saturation_rate_mean",
        "residual_clip_rate_mean",
        "final_goal_distance_m_mean",
        "minimum_goal_distance_m_mean",
        "constraint_violations_mean",
        "control_loop_latency_p95_ms_p95",
    ):
        lines.append(
            f"| `{key}` | {teacher[key]:.6g} | {student[key]:.6g} | "
            f"{student[key] - teacher[key]:.6g} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"- passed: `{gate['passed']}`",
        f"- 3D RMSE gap: `{gate['three_d_rmse_gap_m']:.6g} m` "
        f"(limit `{gate['three_d_rmse_gap_limit_m']}`)",
        f"- success drop: `{gate['success_rate_drop']:.6g}` "
        f"(limit `{gate['success_rate_drop_limit']}`)",
        f"- student material saturation: "
        f"`{gate['student_material_saturation_rate']:.6g}` "
        f"(limit `{gate['student_material_saturation_rate_limit']}`)",
    ]
    if report["b0_comparison"] is not None:
        b0 = report["b0_comparison"]
        lines += [
            "",
            "## B0 comparison",
            "",
            f"- student - B0 3D RMSE: `{b0['three_d_rmse_delta_m']:.6g} m`",
            f"- student - B0 success: `{b0['success_rate_delta']:.6g}`",
            f"- B0 summary: `{b0['summary_path']}`",
        ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_b0_comparison(
    path: Path,
    *,
    seeds: tuple[int, ...],
    student_aggregate: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("--b0-summary must contain an aggregate mapping")
    stored_seeds = payload.get("seeds")
    if stored_seeds is not None and tuple(int(seed) for seed in stored_seeds) != seeds:
        raise ValueError("B0 summary seeds do not match the evaluation seeds")
    b0_rmse = float(aggregate["three_d_rmse_m_mean"])
    b0_success = float(aggregate["success_rate"])
    return {
        "summary_path": str(path.resolve()),
        "three_d_rmse_m": b0_rmse,
        "success_rate": b0_success,
        "three_d_rmse_delta_m": float(
            student_aggregate["three_d_rmse_m_mean"] - b0_rmse
        ),
        "success_rate_delta": float(student_aggregate["success_rate"] - b0_success),
    }


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    schema = observation_schema_for_dim(config.observation.dim)
    checkpoint_policy = BCPolicy(
        args.checkpoint,
        device="cpu",
        expected_observation_dim=schema.dim,
        expected_observation_schema=schema.version,
        expected_speed_feedback=config.observation.speed_feedback,
        expected_current_contract=config.observation.current_contract,
    )
    seeds = _parse_seeds(
        args.seeds,
        config.evaluation.seed_start,
        config.evaluation.seed_count,
    )
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    payloads = [
        (
            str(args.config.resolve()),
            str(args.checkpoint.resolve()),
            seed,
            args.max_steps,
            args.goal_radius,
            str(args.output.resolve()),
            args.device,
        )
        for seed in seeds
    ]
    teacher_episodes: list[dict[str, Any]] = []
    student_episodes: list[dict[str, Any]] = []
    if args.workers == 1:
        for payload in payloads:
            teacher, student = _run_pair(payload)
            teacher_episodes.append(teacher)
            student_episodes.append(student)
    else:
        results: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_pair, payload): payload[2] for payload in payloads}
            for future in as_completed(futures):
                seed = futures[future]
                results[seed] = future.result()
                print(f"finished seed {seed}", flush=True)
        for seed in seeds:
            teacher_episodes.append(results[seed][0])
            student_episodes.append(results[seed][1])

    teacher_aggregate = aggregate_episode_summaries(teacher_episodes)
    student_aggregate = aggregate_episode_summaries(student_episodes)
    three_d_gap = float(
        student_aggregate["three_d_rmse_m_mean"]
        - teacher_aggregate["three_d_rmse_m_mean"]
    )
    success_drop = float(
        teacher_aggregate["success_rate"] - student_aggregate["success_rate"]
    )
    gate = {
        "three_d_rmse_gap_m": three_d_gap,
        "three_d_rmse_gap_limit_m": 0.05,
        "success_rate_drop": success_drop,
        "success_rate_drop_limit": 0.02,
        "student_material_saturation_rate": student_aggregate[
            "material_saturation_rate_mean"
        ],
        "student_material_saturation_rate_limit": 0.05,
    }
    gate["passed"] = bool(
        three_d_gap <= gate["three_d_rmse_gap_limit_m"]
        and success_drop <= gate["success_rate_drop_limit"]
        and gate["student_material_saturation_rate"]
        <= gate["student_material_saturation_rate_limit"]
    )
    b0_comparison = (
        None
        if args.b0_summary is None
        else _load_b0_comparison(
            args.b0_summary,
            seeds=seeds,
            student_aggregate=student_aggregate,
        )
    )
    report = {
        "schema_version": "auv_odarrl_i0_gate.v1",
        "config_path": str(args.config.resolve()),
        "config_sha256": config.config_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_observation_dim": checkpoint_policy.observation_dim,
        "checkpoint_observation_schema": checkpoint_policy.observation_schema.version,
        "git_commit": _git_commit(),
        "observation_dim": schema.dim,
        "observation_schema": schema.version,
        "seeds": list(seeds),
        "max_steps_override": args.max_steps,
        "goal_radius_override_m": args.goal_radius,
        "aggregate": {"teacher": teacher_aggregate, "student": student_aggregate},
        "gate": gate,
        "b0_comparison": b0_comparison,
        "episodes": [
            {"seed": seed, "teacher": teacher, "student": student}
            for seed, teacher, student in zip(
                seeds,
                teacher_episodes,
                student_episodes,
                strict=True,
            )
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(report, args.output)
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
