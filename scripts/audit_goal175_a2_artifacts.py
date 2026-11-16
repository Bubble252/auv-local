#!/usr/bin/env python3
"""Audit all 72 Goal175 A2 cells for mode, seed, trace, and stability integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
A1_ARMS_PATH = REPO_ROOT / "configs/lowlevel/goal175/a1_arms.json"
DEFAULT_ROOT = REPO_ROOT / "artifacts/lowlevel/goal175_alos_comp_mlp_a2_20261114"
EXPECTED_SEEDS = list(range(20, 60))
REQUIRED_TRACE_FIELDS = (
    "seed",
    "path_error_3d",
    "path_horizontal_error",
    "path_vertical_error",
    "goal_distance_eta_m",
)
STABILITY_METRICS = (
    "success_rate",
    "goal_plane_miss_rate",
    "timeout_rate",
    "safety_termination_rate",
    "path_three_d_rmse_m",
    "path_horizontal_rmse_m",
    "path_vertical_rmse_m",
    "ground_speed_max_mps",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _expected_arms(manifest: dict[str, Any], a1_arms: dict[str, Any]) -> dict[str, dict[str, Any]]:
    zero_by_arm = {str(arm["tag"]): arm for arm in a1_arms["arms"]}
    arms: dict[str, dict[str, Any]] = {}
    for run in manifest["train_runs"]:
        arm_name = str(run["arm"])
        checkpoint = Path(str(run["output"])) / "final.pt"
        arms[arm_name] = {
            "tag": arm_name,
            "role": "policy",
            "action_mode": "policy",
            "alos_current_mode": str(run["alos_current_mode"]),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint": checkpoint,
        }
    for baseline in manifest["zero_baselines"]:
        arm_name = str(baseline["arm"])
        declared = zero_by_arm.get(arm_name)
        if declared is None:
            raise ValueError(f"{arm_name}: missing zero arm declaration")
        arms[arm_name] = {
            "tag": arm_name,
            "role": "zero",
            "action_mode": str(baseline["action_mode"]),
            "alos_current_mode": str(baseline["alos_current_mode"]),
            "checkpoint_sha256": str(a1_arms["checkpoint_sha256"]),
            "checkpoint": _resolve(a1_arms["checkpoint"]),
        }
    if len(arms) != 9:
        raise ValueError(f"A2 requires 9 arms, got {len(arms)}")
    return arms


def _audit_cell(
    *,
    summary_path: Path,
    trace_path: Path,
    arm: dict[str, Any],
    scenario: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not summary_path.is_file():
        return [f"{summary_path}: missing"]
    if not trace_path.is_file():
        return [f"{trace_path}: missing"]
    summary = _load(summary_path)
    if summary.get("seeds") != EXPECTED_SEEDS:
        errors.append(f"{summary_path}: seeds are not 20-59")
    if summary.get("checkpoint_sha256") != arm["checkpoint_sha256"]:
        errors.append(f"{summary_path}: checkpoint SHA drifted")
    if summary.get("action_mode") != arm["action_mode"]:
        errors.append(f"{summary_path}: action_mode drifted")
    if summary.get("curriculum_stage") is not None:
        errors.append(f"{summary_path}: A2 must not activate curriculum sampling")
    if summary.get("curriculum_direct_commands_enabled") is not False:
        errors.append(f"{summary_path}: curriculum direct commands are enabled")
    if summary.get("guidance_config") is not None:
        errors.append(f"{summary_path}: unexpected guidance-config override")
    if summary.get("alos_current_mode_override") is not True:
        errors.append(f"{summary_path}: ALOS mode override was not recorded")

    mode = str(arm["alos_current_mode"])
    env_config = _resolve(str(summary.get("environment_config", "")))
    expected_config = _resolve(scenario["configs"][mode])
    if env_config.resolve() != expected_config.resolve():
        errors.append(f"{summary_path}: environment config does not match the {mode} scenario")
    if _sha256(expected_config) != scenario["sha256"][mode]:
        errors.append(f"{summary_path}: environment SHA drifted from the A2 manifest")

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        errors.append(f"{summary_path}: missing metrics block")
        metrics = {}
    for key in STABILITY_METRICS:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{summary_path}: metrics.{key} is missing or non-finite")
    crab = metrics.get("guidance_crab_deg_abs_mean")
    if not isinstance(crab, (int, float)) or not math.isfinite(float(crab)) or float(crab) > 1e-9:
        errors.append(f"{summary_path}: crab angle is not zero")
    speed = metrics.get("ground_speed_max_mps")
    if isinstance(speed, (int, float)) and float(speed) > 2.5 + 1e-12:
        errors.append(f"{summary_path}: ground speed exceeds 2.5 m/s")

    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        errors.append(f"{summary_path}: expected 40 episode records")
    else:
        for episode in episodes:
            if not isinstance(episode, dict):
                errors.append(f"{summary_path}: episode record is not an object")
                break
            route_rmse = episode.get("path_three_d_rmse_m")
            speed = episode.get("ground_speed_max_mps")
            if (
                not isinstance(route_rmse, (int, float))
                or not math.isfinite(float(route_rmse))
                or not isinstance(speed, (int, float))
                or not math.isfinite(float(speed))
            ):
                errors.append(f"{summary_path}: episode route/speed metrics are non-finite")
                break
            if "timeout" not in episode or "safety" not in episode:
                errors.append(f"{summary_path}: episode is missing timeout/safety outcome")
                break
            if float(speed) > 2.5 + 1e-12:
                errors.append(f"{summary_path}: episode ground speed exceeds 2.5 m/s")
                break

    try:
        trace = np.load(trace_path, allow_pickle=False)
    except Exception as exc:
        errors.append(f"{trace_path}: cannot load trace: {exc}")
        return errors
    missing = [key for key in REQUIRED_TRACE_FIELDS if key not in trace.files]
    if missing:
        errors.append(f"{trace_path}: missing trace fields {missing}")
        return errors
    active = (
        np.asarray(trace["active"], dtype=bool)
        if "active" in trace.files
        else np.ones(np.asarray(trace["seed"]).shape, dtype=bool)
    )
    if not bool(active.any()):
        errors.append(f"{trace_path}: trace has no active rows")
    trace_seeds = set(
        int(value) for value in np.unique(np.asarray(trace["seed"], dtype=np.int64)[active])
    )
    if trace_seeds != set(EXPECTED_SEEDS):
        errors.append(f"{trace_path}: trace seeds do not cover 20-59")
    for key in (
        "path_error_3d",
        "path_horizontal_error",
        "path_vertical_error",
        "goal_distance_eta_m",
    ):
        values = np.asarray(trace[key], dtype=np.float64)
        if not np.isfinite(values).all():
            errors.append(f"{trace_path}: {key} contains NaN/Inf")
            break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = _resolve(args.root)
    manifest = _load(root / "configs/a2_manifest.json")
    a1_arms = _load(A1_ARMS_PATH)
    arms = _expected_arms(manifest, a1_arms)
    expected_cells = len(arms) * len(manifest["eval_scenarios"])
    if expected_cells != 72:
        raise ValueError(f"A2 must audit exactly 72 cells, got {expected_cells}")
    errors: list[str] = []
    present = 0
    for arm in arms.values():
        for scenario in manifest["eval_scenarios"]:
            cell = root / "eval" / str(arm["tag"]) / str(scenario["scenario"])
            if (cell / "summary.json").is_file():
                present += 1
            errors.extend(
                _audit_cell(
                    summary_path=cell / "summary.json",
                    trace_path=cell / "trace.npz",
                    arm=arm,
                    scenario=scenario,
                )
            )
    if present != expected_cells:
        errors.append(f"expected {expected_cells} summaries, found {present}")
    if errors:
        print("Goal175 A2 artifact audit FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Goal175 A2 artifact audit OK: {present}/{expected_cells} cells at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
