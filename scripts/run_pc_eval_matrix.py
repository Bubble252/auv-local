#!/usr/bin/env python3
"""Run the versioned PC evaluation matrix without changing evaluator defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auv_lowlevel.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_pc_bc.py"
MATRIX_SCHEMA = "auv_pc_eval_matrix.v1"
ALLOWED_SCENARIOS = frozenset({"default", "strict125"})
ALLOWED_PATH_MODES = frozenset({"normal", "path_zero"})
ALLOWED_TERMINATION_MODES = frozenset({"legacy", "endpoint_stop"})
ALLOWED_PATH_DIMS = frozenset({0, 8, 12})
EXPECTED_GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _contiguous_seed_window(seeds: list[int], *, name: str) -> tuple[int, int]:
    if not seeds:
        raise ValueError(f"{name} must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} contains duplicate seeds")
    ordered = sorted(seeds)
    if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
        raise ValueError(f"{name} must be a contiguous range")
    return ordered[0], len(ordered)


def _seed_set(spec: dict[str, Any], *, name: str) -> set[int]:
    if not isinstance(spec, dict):
        raise ValueError(f"seeds.{name} must be an object")
    try:
        start = int(spec["start"])
        end = int(spec["end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"seeds.{name} must contain integer start/end") from error
    if end < start:
        raise ValueError(f"seeds.{name}.end must be >= start")
    return set(range(start, end + 1))


def validate_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    """Validate matrix structure and return a normalized copy."""

    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError(f"matrix schema must be {MATRIX_SCHEMA!r}")

    configs = matrix.get("base_environment_configs")
    if not isinstance(configs, dict):
        raise ValueError("base_environment_configs must be an object")
    scenarios = matrix.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list")
    if set(scenarios) != ALLOWED_SCENARIOS:
        raise ValueError("scenarios must contain default and strict125")
    for scenario in scenarios:
        if not isinstance(configs.get(scenario), str) or not configs[scenario]:
            raise ValueError(f"missing base_environment_configs.{scenario}")

    seeds = matrix.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("seeds must be an object")
    dev = seeds.get("dev")
    if not isinstance(dev, list) or not all(isinstance(item, int) for item in dev):
        raise ValueError("seeds.dev must be a list of integers")
    dagger_pool = _seed_set(seeds["dagger_pool"], name="dagger_pool")
    held_out = _seed_set(seeds["held_out"], name="held_out")
    dev_set = set(dev)
    if dev_set & held_out:
        raise ValueError("dev seeds overlap held_out")
    if dev_set & dagger_pool:
        raise ValueError("dev seeds overlap dagger_pool")
    if held_out & dagger_pool:
        raise ValueError("held_out overlaps dagger_pool")
    dev_start, dev_count = _contiguous_seed_window(dev, name="seeds.dev")

    path_modes = matrix.get("path_modes")
    if not isinstance(path_modes, list) or not path_modes:
        raise ValueError("path_modes must be a non-empty list")
    if set(path_modes) != ALLOWED_PATH_MODES:
        raise ValueError("path_modes must contain normal and path_zero")
    if matrix.get("termination_mode") not in ALLOWED_TERMINATION_MODES:
        raise ValueError("unsupported termination_mode")
    if matrix.get("path_dim") not in ALLOWED_PATH_DIMS:
        raise ValueError("unsupported path_dim")
    if matrix.get("device") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if matrix.get("dtype") not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")

    output_root = matrix.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root must be a non-empty string")

    normalized = dict(matrix)
    normalized["scenarios"] = ["default", "strict125"]
    normalized["path_modes"] = ["normal", "path_zero"]
    normalized["dev_seed_start"] = dev_start
    normalized["dev_seed_count"] = dev_count
    normalized["dagger_pool_seeds"] = sorted(dagger_pool)
    normalized["held_out_seeds"] = sorted(held_out)
    return normalized


def validate_environment_configs(matrix: dict[str, Any]) -> dict[str, str]:
    sensor_profile = matrix.get("sensor_profile")
    if not isinstance(sensor_profile, str) or not sensor_profile:
        raise ValueError("sensor_profile must be a non-empty string")
    hashes: dict[str, str] = {}
    for scenario in matrix["scenarios"]:
        path = _resolve_repo_path(matrix["base_environment_configs"][scenario])
        if not path.is_file():
            raise FileNotFoundError(f"missing environment config: {path}")
        config = load_config(path)
        if config.sensors is None or config.sensors.profile != sensor_profile:
            raise ValueError(f"{scenario} config does not use sensor profile {sensor_profile}")
        if config.plant.goal_radius_m != EXPECTED_GOAL_RADIUS_M[scenario]:
            raise ValueError(
                f"{scenario} goal_radius_m must be {EXPECTED_GOAL_RADIUS_M[scenario]}"
            )
        hashes[scenario] = _sha256(path)
    return hashes


def build_eval_command(
    *,
    matrix: dict[str, Any],
    scenario: str,
    path_mode: str,
    checkpoint: Path,
    output: Path,
    python: str = sys.executable,
) -> list[str]:
    command = [
        python,
        str(EVALUATOR),
        "--config",
        str(_resolve_repo_path(matrix["base_environment_configs"][scenario])),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--seed-start",
        str(matrix["dev_seed_start"]),
        "--seed-count",
        str(matrix["dev_seed_count"]),
        "--device",
        str(matrix["device"]),
        "--dtype",
        str(matrix["dtype"]),
        "--path-dim",
        str(matrix["path_dim"]),
        "--termination-mode",
        str(matrix["termination_mode"]),
    ]
    if path_mode == "path_zero":
        command.append("--path-zero")
    return command


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "auv_pc_bc_evaluation.v1":
        raise ValueError(f"unexpected evaluation schema: {path}")
    return payload


def _metric_subset(metrics: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {name: metrics.get(name) for name in names}


def _success_only_rmse(episodes: list[dict[str, Any]], key: str) -> float | None:
    rows = [
        episode
        for episode in episodes
        if bool(episode.get("success")) and episode.get(key) is not None
    ]
    if not rows:
        return None
    square_sum = sum(float(episode[key]) ** 2 for episode in rows)
    return (square_sum / len(rows)) ** 0.5


def _derived_metrics(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    student = dict(summary["student"])
    teacher = dict(summary["teacher"])
    student["success_only_3d_rmse_m"] = _success_only_rmse(
        summary["student_episodes"], "path_three_d_rmse_m"
    )
    student["success_only_pre_goal_3d_rmse_m"] = _success_only_rmse(
        summary["student_episodes"], "pre_goal_3d_rmse_m"
    )
    teacher["success_only_3d_rmse_m"] = _success_only_rmse(
        summary["teacher_episodes"], "path_three_d_rmse_m"
    )
    teacher["success_only_pre_goal_3d_rmse_m"] = _success_only_rmse(
        summary["teacher_episodes"], "pre_goal_3d_rmse_m"
    )
    return student, teacher


def _write_comparison(
    *,
    matrix: dict[str, Any],
    matrix_path: Path,
    matrix_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    output_root: Path,
    cells: list[dict[str, Any]],
) -> None:
    by_key = {(cell["scenario"], cell["path_mode"]): cell for cell in cells}
    degradation: dict[str, float | None] = {}
    for scenario in matrix["scenarios"]:
        normal = by_key[(scenario, "normal")]["student"].get("pre_goal_3d_rmse_m")
        zero = by_key[(scenario, "path_zero")]["student"].get("pre_goal_3d_rmse_m")
        degradation[scenario] = (
            None
            if normal is None or zero is None or float(normal) == 0.0
            else (float(zero) - float(normal)) / float(normal)
        )

    payload = {
        "schema_version": "auv_pc_eval_matrix_comparison.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": matrix_sha256,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dev_seeds": {
            "start": matrix["dev_seed_start"],
            "count": matrix["dev_seed_count"],
        },
        "path_zero_pre_goal_rmse_degradation": degradation,
        "cells": cells,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# PC evaluation matrix",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- matrix: `{matrix_path}`",
        f"- dev seeds: `{matrix['dev_seed_start']}`.."
        f"`{matrix['dev_seed_start'] + matrix['dev_seed_count'] - 1}`",
        "",
        "| scenario | path | success | pre-goal 3D RMSE | full 3D RMSE | "
        "plane miss | timeout | safety | saturation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        student = cell["student"]
        lines.append(
            f"| {cell['scenario']} | {cell['path_mode']} | "
            f"{student['success_rate']:.4f} | {student['pre_goal_3d_rmse_m']:.6g} | "
            f"{student['legacy_full_3d_rmse_m']:.6g} | "
            f"{student['goal_plane_miss_rate']:.4f} | "
            f"{student['timeout_rate']:.4f} | {student['safety_rate']:.4f} | "
            f"{student['material_saturation_rate']:.4f} |"
        )
    lines.extend(["", "## Path-zero degradation", "", "| scenario | pre-goal RMSE delta |", "|---|---:|"])
    for scenario, value in degradation.items():
        lines.append(f"| {scenario} | {'N/A' if value is None else f'{value:+.2%}'} |")
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    matrix_path = args.matrix.resolve()
    matrix = validate_matrix(json.loads(matrix_path.read_text(encoding="utf-8")))
    config_hashes = validate_environment_configs(matrix)
    checkpoint = args.checkpoint.resolve()
    if not args.dry_run and not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    output_root = (
        args.output.resolve()
        if args.output is not None
        else _resolve_repo_path(matrix["output_root"]).resolve()
    )
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    metric_names = list(matrix.get("metrics", []))
    if not metric_names:
        raise ValueError("metrics must be a non-empty list")
    matrix_sha256 = _sha256(matrix_path)
    checkpoint_sha256 = _sha256(checkpoint) if checkpoint.is_file() else "dry-run"
    cells: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for scenario in matrix["scenarios"]:
        for path_mode in matrix["path_modes"]:
            cell_output = output_root / scenario / path_mode
            command = build_eval_command(
                matrix=matrix,
                scenario=scenario,
                path_mode=path_mode,
                checkpoint=checkpoint,
                output=cell_output,
            )
            commands.append(command)
            if args.dry_run:
                continue
            cell_output.mkdir(parents=True, exist_ok=True)
            with (cell_output / "run.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    check=True,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            summary_path = cell_output / "summary.json"
            summary = _load_summary(summary_path)
            student_metrics, teacher_metrics = _derived_metrics(summary)
            cells.append(
                {
                    "scenario": scenario,
                    "path_mode": path_mode,
                    "output_dir": str(cell_output),
                    "summary_path": str(summary_path),
                    "summary_sha256": _sha256(summary_path),
                    "student": _metric_subset(student_metrics, metric_names),
                    "teacher": _metric_subset(teacher_metrics, metric_names),
                    "student_episodes": summary["student_episodes"],
                }
            )

    manifest = {
        "schema_version": "auv_pc_eval_matrix_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path),
        "matrix_sha256": matrix_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "environment_config_sha256": config_hashes,
        "dry_run": bool(args.dry_run),
        "commands": commands,
    }
    (output_root / "eval_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.dry_run:
        _write_comparison(
            matrix=matrix,
            matrix_path=matrix_path,
            matrix_sha256=matrix_sha256,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            output_root=output_root,
            cells=cells,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
