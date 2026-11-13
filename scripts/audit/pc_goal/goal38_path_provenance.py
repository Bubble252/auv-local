#!/usr/bin/env python3
"""Audit Goal38 path provenance and control invariance.

The script is read-only. It reconstructs the deterministic path bank from each
block's environment config, checks every trace row against
``path_bank[seed % path_bank_size]``, and optionally compares control fields
with the original Goal31 traces keyed by ``(episode_seed, step)``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.path import BatchedReferencePathBank
from auv_lowlevel.config import load_config
from auv_lowlevel.path import ReferencePath, generate_sinusoidal_3d_path_bank

SCHEMA_VERSION = "auv_pc_goal38_path_provenance.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
PATH_BANK_SIZE = 32
PATH_BANK_SEED = 41
GOAL_TOLERANCE_M = 1e-8
TANGENT_TOLERANCE = 1e-8
ROUTE_TOLERANCE = 1e-6
PROGRESS_TOLERANCE_M = 1e-6
CONTROL_TOLERANCE = 1e-8
CONTROL_FIELDS = ("actual_actuator", "eta", "nu", "requested_action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seed-start", type=int, default=20)
    parser.add_argument("--expected-seed-count", type=int, default=40)
    parser.add_argument(
        "--skip-control-invariance",
        action="store_true",
        help="skip comparison with the original Goal31 traces (for smoke batch sizes)",
    )
    return parser.parse_args()


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("cannot normalize a zero or non-finite vector")
    return value / norm


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_blocks(root: Path) -> dict[tuple[str, str], Path]:
    blocks: dict[tuple[str, str], Path] = {}
    for scenario in SCENARIOS:
        for mode in MODES:
            directory = root / scenario / mode
            if (directory / "trace.npz").is_file() and (directory / "summary.json").is_file():
                blocks[(scenario, mode)] = directory
    return blocks


def _path_bank_for(config_path: Path) -> tuple[tuple[ReferencePath, ...], float]:
    config = load_config(config_path)
    if int(config.reference.path_bank_size) != PATH_BANK_SIZE:
        raise ValueError(
            f"{config_path}: expected path_bank_size={PATH_BANK_SIZE}, "
            f"got {config.reference.path_bank_size}"
        )
    if int(config.reference.path_bank_seed) != PATH_BANK_SEED:
        raise ValueError(
            f"{config_path}: expected path_bank_seed={PATH_BANK_SEED}, "
            f"got {config.reference.path_bank_seed}"
        )
    bank = tuple(generate_sinusoidal_3d_path_bank(config.reference))
    if len(bank) != PATH_BANK_SIZE:
        raise ValueError(f"{config_path}: expected {PATH_BANK_SIZE} paths, got {len(bank)}")
    max_start_jitter_m = math.sqrt(
        2.0 * float(config.reference.start_jitter_xy_m) ** 2
        + float(config.reference.start_jitter_z_m) ** 2
    )
    return bank, max_start_jitter_m


def _require_trace_arrays(trace: dict[str, np.ndarray]) -> None:
    required = (
        "episode_seed",
        "step",
        "eta",
        "goal_ned_m",
        "goal_tangent_ned",
        "route_tangent_ned",
        "path_progress_m",
    )
    missing = [name for name in required if name not in trace]
    if missing:
        raise ValueError(f"trace is missing arrays: {missing}")
    sizes = {name: int(np.asarray(trace[name]).shape[0]) for name in required}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"trace arrays have inconsistent lengths: {sizes}")
    if sizes["episode_seed"] == 0:
        raise ValueError("trace contains no rows")


def _trace_key_index(trace: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    steps = np.asarray(trace["step"], dtype=np.int64)
    if seeds.shape != steps.shape:
        raise ValueError("episode_seed and step shapes differ")
    keys: dict[tuple[int, int], int] = {}
    for index, (seed, step) in enumerate(zip(seeds.tolist(), steps.tolist(), strict=True)):
        key = (int(seed), int(step))
        if key in keys:
            raise ValueError(f"duplicate trace key {key}")
        keys[key] = index
    return keys


def _audit_provenance(
    trace: dict[str, np.ndarray],
    path_bank: tuple[ReferencePath, ...],
    expected_seeds: set[int],
    max_start_jitter_m: float,
) -> dict[str, Any]:
    _require_trace_arrays(trace)
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    steps = np.asarray(trace["step"], dtype=np.int64)
    eta = np.asarray(trace["eta"], dtype=np.float64)
    goals = np.asarray(trace["goal_ned_m"], dtype=np.float64)
    goal_tangents = np.asarray(trace["goal_tangent_ned"], dtype=np.float64)
    route_tangents = np.asarray(trace["route_tangent_ned"], dtype=np.float64)
    progress = np.asarray(trace["path_progress_m"], dtype=np.float64).reshape(-1)
    if eta.shape[1:] != (6,) or goals.shape[1:] != (3,) or goal_tangents.shape[1:] != (3,):
        raise ValueError("trace geometry has unexpected shape")
    if route_tangents.shape != goals.shape:
        raise ValueError("route_tangent_ned shape does not match goal_ned_m")
    for name, array in (
        ("eta", eta),
        ("goal_ned_m", goals),
        ("goal_tangent_ned", goal_tangents),
        ("route_tangent_ned", route_tangents),
        ("path_progress_m", progress),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")

    observed_seeds = {int(seed) for seed in seeds.tolist()}
    if observed_seeds != expected_seeds:
        missing = sorted(expected_seeds - observed_seeds)
        extra = sorted(observed_seeds - expected_seeds)
        raise ValueError(f"seed set mismatch: missing={missing}, extra={extra}")

    seed_checks: dict[int, dict[str, bool]] = {
        seed: {
            "goal": True,
            "goal_tangent": True,
            "route_tangent": True,
            "path_progress": True,
            "start_within_jitter": False,
            "steps_start_at_zero": False,
        }
        for seed in expected_seeds
    }
    max_goal_error = 0.0
    max_goal_tangent_error = 0.0
    max_route_error = 0.0
    max_progress_error = 0.0
    max_start_distance = 0.0
    max_start_jitter = float(max_start_jitter_m)
    batched_bank = BatchedReferencePathBank(path_bank, device="cpu", dtype=torch.float64)
    search_radius = max(8, batched_bank.required_search_radius(max_start_jitter_m))

    for seed in sorted(expected_seeds):
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            continue
        path_index = seed % len(path_bank)
        path = path_bank[path_index]
        seed_steps = steps[indices]
        if int(seed_steps[0]) == 0:
            seed_checks[seed]["steps_start_at_zero"] = True
        expected_goal = path.goal_ned_m
        expected_goal_tangent = path.tangents_ned[-1]
        start_distance = float(np.linalg.norm(eta[indices[0], :3] - path.start_ned_m))
        path_set = batched_bank.select(torch.tensor([path_index], dtype=torch.long))
        hint = torch.zeros(1, dtype=torch.long)

        ordered = indices[np.argsort(steps[indices], kind="stable")]
        for index in ordered.tolist():
            goal_error = float(np.linalg.norm(goals[index] - expected_goal))
            tangent_error = float(
                np.linalg.norm(_unit(goal_tangents[index]) - _unit(expected_goal_tangent))
            )
            position = torch.as_tensor(eta[index, :3], dtype=torch.float64)[None, :]
            projection = path_set.project(
                position,
                segment_hint=hint,
                search_radius=search_radius,
            )
            hint = projection.segment_index
            projection_tangent = projection.tangent_ned[0].detach().cpu().numpy()
            projection_progress = float(projection.progress_m[0].item())
            route_error = float(
                np.linalg.norm(_unit(route_tangents[index]) - _unit(projection_tangent))
            )
            progress_error = abs(float(progress[index]) - projection_progress)
            max_goal_error = max(max_goal_error, goal_error)
            max_goal_tangent_error = max(max_goal_tangent_error, tangent_error)
            max_route_error = max(max_route_error, route_error)
            max_progress_error = max(max_progress_error, progress_error)
            seed_checks[seed]["goal"] &= goal_error <= GOAL_TOLERANCE_M
            seed_checks[seed]["goal_tangent"] &= tangent_error <= TANGENT_TOLERANCE
            seed_checks[seed]["route_tangent"] &= route_error <= ROUTE_TOLERANCE
            seed_checks[seed]["path_progress"] &= progress_error <= PROGRESS_TOLERANCE_M

        max_start_distance = max(max_start_distance, start_distance)
        seed_checks[seed]["start_within_jitter"] = (
            start_distance <= max_start_jitter_m + 1e-6
        )

    counts = {
        key: int(sum(bool(values[key]) for values in seed_checks.values()))
        for key in (
            "goal",
            "goal_tangent",
            "route_tangent",
            "path_progress",
            "start_within_jitter",
            "steps_start_at_zero",
        )
    }
    valid = all(count == len(expected_seeds) for count in counts.values())
    return {
        "valid": valid,
        "seed_count": len(expected_seeds),
        "counts": counts,
        "max_goal_error_m": max_goal_error,
        "max_goal_tangent_error": max_goal_tangent_error,
        "max_route_tangent_error": max_route_error,
        "max_path_progress_error_m": max_progress_error,
        "max_start_distance_m": max_start_distance,
        "max_start_jitter_allowance_m": max_start_jitter,
        "mismatch_seeds": {
            key: [seed for seed, values in seed_checks.items() if not values[key]]
            for key in counts
        },
    }


def _compare_control_fields(
    fresh: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    fresh_keys = _trace_key_index(fresh)
    reference_keys = _trace_key_index(reference)
    common = sorted(set(fresh_keys) & set(reference_keys))
    missing_fresh = sorted(set(reference_keys) - set(fresh_keys))
    missing_reference = sorted(set(fresh_keys) - set(reference_keys))
    result: dict[str, Any] = {
        "common_rows": len(common),
        "missing_fresh_rows": len(missing_fresh),
        "missing_reference_rows": len(missing_reference),
        "field_max_abs_diff": {},
        "terminal_kind_equal": None,
        "valid": not missing_fresh and not missing_reference and bool(common),
    }
    if not common:
        result["valid"] = False
        return result
    fresh_indices = np.asarray([fresh_keys[key] for key in common], dtype=np.int64)
    reference_indices = np.asarray([reference_keys[key] for key in common], dtype=np.int64)
    for field in CONTROL_FIELDS:
        if field not in fresh or field not in reference:
            result["field_max_abs_diff"][field] = None
            result["valid"] = False
            continue
        left = np.asarray(fresh[field], dtype=np.float64)[fresh_indices]
        right = np.asarray(reference[field], dtype=np.float64)[reference_indices]
        if left.shape != right.shape:
            result["field_max_abs_diff"][field] = None
            result["valid"] = False
            continue
        difference = float(np.max(np.abs(left - right))) if left.size else 0.0
        result["field_max_abs_diff"][field] = difference
        result["valid"] &= difference <= CONTROL_TOLERANCE
    if "terminal_kind" in fresh and "terminal_kind" in reference:
        left_kind = np.asarray(fresh["terminal_kind"])[fresh_indices]
        right_kind = np.asarray(reference["terminal_kind"])[reference_indices]
        equal = bool(np.array_equal(left_kind, right_kind))
        result["terminal_kind_equal"] = equal
        result["valid"] &= equal
    else:
        result["valid"] = False
    return result


def analyze(
    trace_root: Path,
    reference_root: Path,
    expected_seed_start: int,
    expected_seed_count: int,
    check_control_invariance: bool,
) -> dict[str, Any]:
    expected_seeds = set(range(expected_seed_start, expected_seed_start + expected_seed_count))
    blocks = _discover_blocks(trace_root)
    reference_blocks = _discover_blocks(reference_root)
    missing_blocks = [
        f"{scenario}/{mode}"
        for scenario in SCENARIOS
        for mode in MODES
        if (scenario, mode) not in blocks
    ]
    path_bank_cache: dict[str, tuple[tuple[ReferencePath, ...], float]] = {}
    scenario_results: dict[str, dict[str, Any]] = {scenario: {} for scenario in SCENARIOS}
    invalid_reasons: list[str] = []

    if missing_blocks:
        invalid_reasons.append(f"missing trace blocks: {missing_blocks}")

    for scenario in SCENARIOS:
        for mode in MODES:
            directory = blocks.get((scenario, mode))
            if directory is None:
                continue
            summary = _load_json(directory / "summary.json")
            config_path = Path(str(summary.get("environment_config_override") or summary.get("environment_config")))
            if not config_path.is_absolute():
                config_path = directory / config_path
            config_key = str(config_path)
            if config_key not in path_bank_cache:
                path_bank_cache[config_key] = _path_bank_for(config_path)
            path_bank, max_start_jitter_m = path_bank_cache[config_key]
            with np.load(directory / "trace.npz") as loaded:
                fresh_trace = {name: np.asarray(loaded[name]) for name in loaded.files}
            provenance = _audit_provenance(
                fresh_trace,
                path_bank,
                expected_seeds,
                max_start_jitter_m,
            )
            control_comparison: dict[str, Any] | None = None
            if check_control_invariance:
                reference_dir = reference_blocks.get((scenario, mode))
                if reference_dir is not None:
                    with np.load(reference_dir / "trace.npz") as loaded:
                        reference_trace = {
                            name: np.asarray(loaded[name]) for name in loaded.files
                        }
                    control_comparison = _compare_control_fields(fresh_trace, reference_trace)
            block_valid = bool(provenance["valid"]) and (
                not check_control_invariance
                or (control_comparison is not None and bool(control_comparison["valid"]))
            )
            if not provenance["valid"]:
                invalid_reasons.append(f"{scenario}/{mode}: provenance mismatch")
            if check_control_invariance:
                if control_comparison is None:
                    invalid_reasons.append(f"{scenario}/{mode}: missing reference trace")
                elif not control_comparison["valid"]:
                    invalid_reasons.append(f"{scenario}/{mode}: control invariance failed")
            scenario_results[scenario][mode] = {
                "directory": str(directory),
                "environment_config": config_key,
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "provenance": provenance,
                "control_comparison": control_comparison,
                "valid": block_valid,
            }

    overall_valid = not invalid_reasons and all(
        scenario_results[scenario][mode]["valid"]
        for scenario in SCENARIOS
        for mode in MODES
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_root": str(trace_root),
        "reference_root": str(reference_root),
        "expected_seed_start": expected_seed_start,
        "expected_seed_count": expected_seed_count,
        "expected_seeds": sorted(expected_seeds),
        "heldout_opened": False,
        "control_invariance_checked": check_control_invariance,
        "path_bank_size": PATH_BANK_SIZE,
        "path_bank_seed": PATH_BANK_SEED,
        "tolerances": {
            "goal_m": GOAL_TOLERANCE_M,
            "tangent": TANGENT_TOLERANCE,
            "route_tangent": ROUTE_TOLERANCE,
            "path_progress_m": PROGRESS_TOLERANCE_M,
            "control": CONTROL_TOLERANCE,
        },
        "blocks": scenario_results,
        "invalid_reasons": invalid_reasons,
        "valid": overall_valid,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal38 Path Provenance Audit",
        "",
        f"- valid: `{result['valid']}`",
        f"- trace root: `{result['trace_root']}`",
        f"- reference root: `{result['reference_root']}`",
        f"- seeds: `{result['expected_seed_start']}-{result['expected_seed_start'] + result['expected_seed_count'] - 1}`",
        f"- held-out opened: `{result['heldout_opened']}`",
        "",
        "| scenario | mode | goal | goal tan | route tan | progress | start | controls | valid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["blocks"].get(scenario, {}).get(mode)
            if block is None:
                lines.append(f"| {scenario} | {mode} | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |")
                continue
            provenance = block["provenance"]
            counts = provenance["counts"]
            total = provenance["seed_count"]
            control = block["control_comparison"]
            control_ok = None if control is None else bool(control["valid"])
            lines.append(
                f"| {scenario} | {mode} | {counts['goal']}/{total} | "
                f"{counts['goal_tangent']}/{total} | {counts['route_tangent']}/{total} | "
                f"{counts['path_progress']}/{total} | {counts['start_within_jitter']}/{total} | "
                f"{'N/A' if control_ok is None else ('PASS' if control_ok else 'FAIL')} | "
                f"{'PASS' if block['valid'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Control maxima",
            "",
        ]
    )
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["blocks"].get(scenario, {}).get(mode)
            if block is None or block["control_comparison"] is None:
                continue
            control = block["control_comparison"]
            values = ", ".join(
                f"{field}={value:.3e}" if value is not None else f"{field}=MISSING"
                for field, value in control["field_max_abs_diff"].items()
            )
            lines.append(f"- {scenario}/{mode}: {values}")
    lines.extend(["", "## Invalid reasons", ""])
    lines.extend(
        [f"- {reason}" for reason in result["invalid_reasons"]]
        if result["invalid_reasons"]
        else ["- none"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result = analyze(
        args.trace_root,
        args.reference_root,
        args.expected_seed_start,
        args.expected_seed_count,
        not args.skip_control_invariance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "invalid_reasons": result["invalid_reasons"]}, indent=2))


if __name__ == "__main__":
    main()
