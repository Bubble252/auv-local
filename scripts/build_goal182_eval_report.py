#!/usr/bin/env python3
"""Build Goal182 canonical evaluation reports from completed run artifacts.

Non-responsibility: this script does not run evaluations, change evaluator
contracts, select checkpoints, or rank arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from auv_lowlevel.eval_pairing import (
    PAIR_REQUIRED_COMPOSITIONS,
    PAIRED,
    PE2E_CONTROL_CONTRACT,
    UNPAIRED,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "auv_goal182_eval_matrix_manifest.v1"
RATE_EPISODE_KEYS = {
    "success": "success",
    "timeout": "timeout",
    "safety": "safety",
    "plane_miss": "goal_plane_miss",
}
ROUTE_KEYS = (
    "path_three_d_rmse_m",
    "path_horizontal_rmse_m",
    "path_vertical_rmse_m",
)
DEFAULT_RADIUS_M = 2.0  # magic-number-ok: scenario ids encode frozen plant.goal_radius_m
RADIUS_BY_TAG = {
    "default": DEFAULT_RADIUS_M,
    "2p0": DEFAULT_RADIUS_M,
    "2.0": DEFAULT_RADIUS_M,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--goal-id", default="Goal182")
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--training-profile", default="PPO-FLAT-4096x48-E5-MB4")
    parser.add_argument(
        "--reporting-rule",
        default="report every pressure separately; never pool pressure levels before ranking",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean_rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key, False)) for row in rows) / float(len(rows))


def _episode_mean(values: list[Any], *, square: bool = False) -> float | None:
    if any(not _finite(value) for value in values):
        return None
    numeric = [float(value) for value in values]
    if square:
        return math.sqrt(sum(value * value for value in numeric) / len(numeric))
    return sum(numeric) / len(numeric)


def _aggregate_episodes(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    if not rows:
        return metrics, ["INVALID_ARTIFACT_SET:no_episodes"]
    for output_key, episode_key in RATE_EPISODE_KEYS.items():
        if any(episode_key not in row for row in rows):
            failures.append(f"INVALID_REPORT_PROVENANCE:missing_{episode_key}")
            metrics[output_key] = None
        else:
            metrics[output_key] = _mean_rate(rows, episode_key)
    for key in ROUTE_KEYS:
        metrics[key] = _episode_mean([row.get(key) for row in rows], square=True)
        if metrics[key] is None:
            failures.append(f"INVALID_REPORT_PROVENANCE:{key}")
    metrics["path_completion_ratio"] = _episode_mean(
        [row.get("path_completion_ratio") for row in rows]
    )
    if metrics["path_completion_ratio"] is None:
        failures.append("INVALID_REPORT_PROVENANCE:path_completion_ratio")
    metrics["material_saturation_rate"] = _episode_mean(
        [row.get("material_saturation_rate") for row in rows]
    )
    if metrics["material_saturation_rate"] is None:
        failures.append("INVALID_REPORT_PROVENANCE:material_saturation_rate")
    weights = [int(row.get("pre_goal_path_steps", 0)) for row in rows]
    values = [row.get("pre_goal_3d_rmse_m") for row in rows]
    if sum(weights) > 0 and all(_finite(value) for value in values):
        metrics["pre_goal_3d_rmse_m"] = math.sqrt(
            sum(weight * float(value) ** 2 for weight, value in zip(weights, values, strict=True))
            / sum(weights)
        )
    else:
        metrics["pre_goal_3d_rmse_m"] = None
        failures.append("INVALID_REPORT_PROVENANCE:pre_goal_3d_rmse_m")
    return metrics, failures


def _parse_scenario_id(scenario_id: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    pressure = re.search(r"(?:^|__)p-([^_]+(?:10|05|07|15|20|_n2)?)(?:__|$)", scenario_id)
    route = re.search(r"(?:^|__)r-(.+?)(?=__|$)", scenario_id)
    depth = re.search(r"(?:^|__)d([0-9p.]+)(?:__|$)", scenario_id)
    length = re.search(r"(?:^|__)l([0-9p.]+)(?:__|$)", scenario_id)
    speed = re.search(r"(?:^|__)v([0-9p.]+)(?:__|$)", scenario_id)
    radius = re.search(r"(?:^|__)g-([^_]+)(?:__|$)", scenario_id)
    if pressure:
        values["pressure_profile"] = pressure.group(1)
    if route:
        values["route_profile"] = route.group(1)
    if depth:
        values["depth_m"] = float(depth.group(1).replace("p", "."))
    if length:
        values["length_m"] = float(length.group(1).replace("p", "."))
    if speed:
        values["speed_mps"] = float(speed.group(1).replace("p", "."))
    if radius:
        values["radius_m"] = RADIUS_BY_TAG.get(radius.group(1), radius.group(1))
    return values


def _environment_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("environment_config")
    if not isinstance(value, str) or not value:
        return {}
    path = _resolve(value)
    if not path.is_file():
        return {}
    config = _load_json(path)
    reference = config.get("reference") if isinstance(config.get("reference"), dict) else {}
    plant = config.get("plant") if isinstance(config.get("plant"), dict) else {}
    curriculum = config.get("curriculum") if isinstance(config.get("curriculum"), dict) else {}
    return {
        "path_family": reference.get("path_family"),
        "length_m": reference.get("length_m"),
        "speed_mps": reference.get("speed_mps"),
        "depth_m": reference.get("depth_start_m"),
        "depth_band_m": reference.get("depth_start_m"),
        "vertical_relief_m": reference.get("depth_amplitude_m"),
        "radius_m": plant.get("goal_radius_m"),
        "stage": curriculum.get("initial_stage"),
        "environment_sha256": _sha256(path),
    }


def _cell_metadata(job: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    scenario = job.get("scenario_metadata")
    if not isinstance(scenario, dict):
        scenario = {}
    parsed = _parse_scenario_id(str(job.get("scenario", "")))
    environment = _environment_metadata(summary)
    merged = {**parsed, **environment, **scenario}
    return {
        "stage": summary.get("curriculum_stage") or merged.get("stage"),
        "route_profile": merged.get("route_profile"),
        "path_family": merged.get("path_family", merged.get("route_profile")),
        "depth_m": merged.get("depth_m"),
        "depth_band_m": merged.get("depth_band_m"),
        "vertical_relief_m": merged.get("vertical_relief_m"),
        "length_m": merged.get("length_m"),
        "speed_mps": merged.get("speed_mps"),
        "pressure_profile": merged.get("pressure_profile", "unknown"),
        "radius_m": merged.get("radius_m"),
    }


def _training_seed(job: dict[str, Any]) -> int | None:
    value = job.get("training_seed")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(r"_s(\d+)$", str(job.get("run_tag", "")))
    return int(match.group(1)) if match else None


def _run_row(
    job: dict[str, Any],
    *,
    goal_id: str,
    tier: str,
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = _resolve(str(job["output"]))
    summary_path = run_dir / "summary.json"
    episodes_path = run_dir / "episodes.jsonl"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    episodes = _load_jsonl(episodes_path) if episodes_path.is_file() else []
    metrics, failures = _aggregate_episodes(episodes)
    if not summary_path.is_file():
        failures.append("INVALID_ARTIFACT_SET:missing_summary")
    if not episodes_path.is_file():
        failures.append("INVALID_ARTIFACT_SET:missing_episodes")
    meta = _cell_metadata(job, summary)
    episode_routes = {
        str(episode["route_profile"])
        for episode in episodes
        if episode.get("route_profile") is not None
    }
    episode_depths = {
        float(episode["depth_m"]) for episode in episodes if episode.get("depth_m") is not None
    }
    if len(episode_routes) > 1:
        failures.append("INVALID_ROUTE_AXIS_UNCONTROLLED")
    if len(episode_depths) > 1:
        failures.append("INVALID_GOAL182_DEPTH")
    if meta.get("route_profile") is None and len(episode_routes) == 1:
        meta["route_profile"] = next(iter(episode_routes))
        meta["path_family"] = meta.get("path_family") or meta["route_profile"]
    if meta.get("depth_m") is None and len(episode_depths) == 1:
        meta["depth_m"] = next(iter(episode_depths))
    is_zero_baseline = bool(job.get("is_zero_baseline", False))
    zero_status = job.get("zero_baseline_status")
    composition = job.get("learning_composition")
    control_contract = str(job.get("control_contract", "") or "")
    if not is_zero_baseline:
        if control_contract == PE2E_CONTROL_CONTRACT:
            if zero_status != UNPAIRED:
                failures.append("INVALID_ZERO_BASELINE_PROTOCOL")
        elif composition in PAIR_REQUIRED_COMPOSITIONS and zero_status != PAIRED:
            failures.append("INVALID_INCOMPLETE_EVAL")
    if tier == "DEPTH_GENERALIZATION":
        depth = meta.get("depth_m")
        if depth is None:
            failures.append("INVALID_GOAL182_DEPTH")
        else:
            try:
                depth_m = float(depth)
            except (TypeError, ValueError):
                failures.append("INVALID_GOAL182_DEPTH")
            else:
                if not math.isfinite(depth_m) or depth_m <= 0.0:
                    failures.append("INVALID_GOAL182_DEPTH")
        if meta.get("route_profile") is None:
            failures.append("INVALID_GOAL182_ROUTE")
    radius = meta.get("radius_m")
    if radius is None:
        failures.append("INVALID_GOAL182_RADIUS")
    else:
        try:
            radius_m = float(radius)
        except (TypeError, ValueError):
            failures.append("INVALID_GOAL182_RADIUS")
        else:
            if not math.isclose(radius_m, DEFAULT_RADIUS_M, abs_tol=1e-9):
                failures.append("INVALID_GOAL182_RADIUS")
    arm_id = str(job.get("arm_id", job.get("arm_tag", job.get("arm", "unknown"))))
    row = {
        "goal_id": goal_id,
        "tier": tier,
        "phase": phase,
        "arm_id": arm_id,
        "role": job.get("role", job.get("paradigm")),
        "learning_composition": composition,
        "is_zero_baseline": is_zero_baseline,
        "zero_baseline_status": zero_status,
        "paired_zero_tag": job.get("paired_zero_tag"),
        "evaluation_set_id": job.get("evaluation_set_id"),
        "scenario": job.get("scenario"),
        "training_seed": _training_seed(job),
        "checkpoint_sha256": summary.get("checkpoint_sha256", job.get("checkpoint_sha256")),
        **meta,
        "eval_seeds": sorted({int(episode["seed"]) for episode in episodes if "seed" in episode}),
        "n_eval_episodes": len(episodes),
        "success": metrics.get("success"),
        "timeout": metrics.get("timeout"),
        "safety": metrics.get("safety"),
        "plane_miss": metrics.get("plane_miss"),
        "path_three_d_rmse_m": metrics.get("path_three_d_rmse_m"),
        "path_horizontal_rmse_m": metrics.get("path_horizontal_rmse_m"),
        "path_vertical_rmse_m": metrics.get("path_vertical_rmse_m"),
        "pre_goal_3d_rmse_m": metrics.get("pre_goal_3d_rmse_m"),
        "path_completion_ratio": metrics.get("path_completion_ratio"),
        "material_saturation_rate": metrics.get("material_saturation_rate"),
        "valid": not failures,
        "failure_codes": failures,
        "run_dir": _relative(run_dir),
    }
    run = {
        "arm_id": arm_id,
        "is_zero_baseline": is_zero_baseline,
        "zero_baseline_status": zero_status,
        "training_seed": row["training_seed"],
        "scenario": job.get("scenario"),
        "run_dir": row["run_dir"],
        "summary_sha256": _sha256(summary_path),
        "episodes_sha256": _sha256(episodes_path),
        "trace_sha256": _sha256(run_dir / "trace.npz"),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "environment_sha256": _environment_metadata(summary).get("environment_sha256"),
        "valid": row["valid"],
        "failure_codes": failures,
    }
    return row, run


def _add_failure(row: dict[str, Any], code: str) -> None:
    failures = row["failure_codes"]
    if code not in failures:
        failures.append(code)
    row["valid"] = False


def _enforce_paired_zero_seeds(
    policy_rows: list[dict[str, Any]], zero_rows: list[dict[str, Any]]
) -> None:
    """Require each pair-required policy cell to have the same zero seeds."""

    zeros_by_key = {(str(row.get("arm_id")), str(row.get("scenario"))): row for row in zero_rows}
    for row in policy_rows:
        if row.get("learning_composition") not in PAIR_REQUIRED_COMPOSITIONS:
            continue
        if row.get("zero_baseline_status") != PAIRED:
            continue
        zero_tag = row.get("paired_zero_tag")
        zero = zeros_by_key.get((str(zero_tag), str(row.get("scenario"))))
        if zero is None:
            _add_failure(row, "INVALID_INCOMPLETE_EVAL:paired_zero_missing")
            continue
        if not zero.get("valid", False):
            _add_failure(row, "INVALID_INCOMPLETE_EVAL:paired_zero_invalid")
            continue
        policy_seeds = {int(seed) for seed in row.get("eval_seeds", [])}
        zero_seeds = {int(seed) for seed in zero.get("eval_seeds", [])}
        if not policy_seeds or policy_seeds != zero_seeds:
            _add_failure(row, "INVALID_INCOMPLETE_EVAL:paired_zero_seed_mismatch")


def build_report(
    *,
    run_manifest_path: Path,
    goal_id: str,
    matrix_id: str,
    tier: str,
    phase: str,
    training_profile: str,
    reporting_rule: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    payload = _load_json(run_manifest_path)
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("run manifest must contain a non-empty jobs list")
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("every run-manifest job must be an object")
        row, run = _run_row(job, goal_id=goal_id, tier=tier, phase=phase)
        rows.append(row)
        runs.append(run)
    policy_rows = [row for row in rows if not row["is_zero_baseline"]]
    zero_rows = [row for row in rows if row["is_zero_baseline"]]
    _enforce_paired_zero_seeds(policy_rows, zero_rows)
    valid_cells = sum(bool(row["valid"]) for row in policy_rows)
    valid_zero_cells = sum(bool(row["valid"]) for row in zero_rows)
    paired_zero_cells = sum(row["zero_baseline_status"] == PAIRED for row in policy_rows)
    unpaired_zero_cells = sum(row["zero_baseline_status"] == UNPAIRED for row in policy_rows)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "goal_id": goal_id,
        "matrix_id": matrix_id,
        "tier": tier,
        "phase": phase,
        "training_profile": training_profile,
        "heldout": {"enabled": False, "seeds": []},
        "reporting_rule": reporting_rule,
        "run_manifest": _relative(run_manifest_path),
        "run_manifest_sha256": _sha256(run_manifest_path),
        "expected_cells": len(policy_rows),
        "observed_cells": sum(
            bool(run["summary_sha256"]) for run in runs if not run["is_zero_baseline"]
        ),
        "valid_cells": valid_cells,
        "invalid_cells": len(policy_rows) - valid_cells,
        "zero_cells": len(zero_rows),
        "observed_zero_cells": sum(
            bool(run["summary_sha256"]) for run in runs if run["is_zero_baseline"]
        ),
        "valid_zero_cells": valid_zero_cells,
        "invalid_zero_cells": len(zero_rows) - valid_zero_cells,
        "paired_zero_cells": paired_zero_cells,
        "unpaired_zero_cells": unpaired_zero_cells,
        "runs": runs,
    }
    return manifest, rows, _markdown(manifest, rows)


def _format(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _markdown(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {manifest['goal_id']} evaluation report",
        "",
        f"- matrix: `{manifest['matrix_id']}`",
        f"- tier: `{manifest['tier']}`",
        f"- phase: `{manifest['phase']}`",
        f"- expected cells: `{manifest['expected_cells']}`",
        f"- observed cells: `{manifest['observed_cells']}`",
        f"- valid cells: `{manifest['valid_cells']}`",
        f"- invalid cells: `{manifest['invalid_cells']}`",
        f"- zero cells: `{manifest['zero_cells']}`",
        f"- paired-zero cells: `{manifest['paired_zero_cells']}`",
        f"- unpaired-zero cells: `{manifest['unpaired_zero_cells']}`",
        f"- reporting rule: {manifest['reporting_rule']}",
        "",
    ]
    by_pressure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pressure[str(row.get("pressure_profile", "unknown"))].append(row)
    for pressure in sorted(by_pressure):
        lines.extend(
            [
                f"## Pressure: {pressure}",
                "",
                "| arm | seed | stage | route | depth | speed | radius | zero | n | success | timeout | safety | plane miss | 3D RMSE | cross | vertical | pre-goal 3D | completion | saturation | valid |",
                "|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in by_pressure[pressure]:
            lines.append(
                "| {arm} | {seed} | {stage} | {route} | {depth} | {speed} | {radius} | {zero} | {n} | "
                "{success} | {timeout} | {safety} | {miss} | {rmse3d} | {cross} | "
                "{vertical} | {pregoal} | {completion} | {saturation} | {valid} |".format(
                    arm=row["arm_id"],
                    seed=_format(row["training_seed"]),
                    stage=_format(row["stage"]),
                    route=_format(row["route_profile"]),
                    depth=_format(row.get("depth_m")),
                    speed=_format(row["speed_mps"]),
                    radius=_format(row["radius_m"]),
                    zero=_format(row.get("zero_baseline_status")),
                    n=row["n_eval_episodes"],
                    success=_format(row["success"]),
                    timeout=_format(row["timeout"]),
                    safety=_format(row["safety"]),
                    miss=_format(row["plane_miss"]),
                    rmse3d=_format(row["path_three_d_rmse_m"]),
                    cross=_format(row["path_horizontal_rmse_m"]),
                    vertical=_format(row["path_vertical_rmse_m"]),
                    pregoal=_format(row["pre_goal_3d_rmse_m"]),
                    completion=_format(row["path_completion_ratio"]),
                    saturation=_format(row["material_saturation_rate"]),
                    valid=_format(row["valid"]),
                )
            )
        lines.append("")
    invalid = [row for row in rows if not row["valid"]]
    if invalid:
        lines.extend(["## Invalid cells", ""])
        for row in invalid:
            lines.append(
                f"- `{row['arm_id']}` / `{row.get('stage')}` / "
                f"`{row.get('pressure_profile')}`: `{', '.join(row['failure_codes'])}`"
            )
        lines.append("")
    return "\n".join(lines)


def _write_outputs(
    output_root: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    markdown: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_root / "report_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_root / "report.md").write_text(markdown + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_manifest_path = _resolve(args.run_manifest)
    output_root = (
        run_manifest_path.parent if args.output_root is None else _resolve(args.output_root)
    )
    manifest, rows, markdown = build_report(
        run_manifest_path=run_manifest_path,
        goal_id=args.goal_id,
        matrix_id=args.matrix_id,
        tier=args.tier,
        phase=args.phase,
        training_profile=args.training_profile,
        reporting_rule=args.reporting_rule,
    )
    _write_outputs(output_root, manifest, rows, markdown)
    if manifest["invalid_cells"]:
        print(
            f"Goal182 report built with invalid cells: {manifest['invalid_cells']}",
            file=sys.stderr,
        )
        return 1
    print(
        f"Goal182 report OK: {manifest['valid_cells']}/{manifest['expected_cells']} cells "
        f"at {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
