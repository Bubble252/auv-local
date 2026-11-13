#!/usr/bin/env python3
"""Goal60 read-only audit of action saturation, reward direction and capture.

Consumes traces produced by ``evaluate_pc_command_residual_offpolicy.py
--trace-output``.  It never loads a policy or steps an environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal60_trace_audit.v1"
SCENARIOS = ("default", "strict125")
POLICY_SEEDS = (240, 752, 880, 1008)
DEV_SEEDS = tuple(range(20, 60))
HELDOUT_SEEDS = tuple(range(100, 150))
TERMINAL_WINDOW_STEPS = 100
SATURATION_FRACTION = 0.9
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_261_110
RHO_MIN = 0.30
SAT_MEDIAN_DIFF_MIN = 0.05
CAPTURE_DISCORDANCE_MIN = 0.10
CONFLICT_FRACTION_MIN = 0.20
PATH_COMPONENTS = (
    "reward_path_progress",
    "reward_path_cross",
    "reward_path_vertical",
    "reward_path_heading",
)
COMMAND_COMPONENTS = (
    "reward_command_yaw",
    "reward_command_pitch",
    "reward_command_speed",
)
TERMINAL_COMPONENTS = ("reward_terminal", "reward_terminal_progress")
REQUIRED_TRACE_FIELDS = {
    "step",
    "env_index",
    "episode_id",
    "seed",
    "action",
    "residual",
    "base_command",
    "gate",
    "reward",
    "goal_distance",
    "path_error_3d",
    "eta_ned",
    "p_nav_ned",
    "goal_ned",
    "goal_tangent_ned",
    "done",
    "terminal_kind",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=Path("artifacts/traces"),
        help="Directory containing <scenario>_policy_s<seed> and <scenario>_zero",
    )
    parser.add_argument(
        "--goal59-root",
        type=Path,
        default=Path(
            "/data/yhx/underwater-robotics/src/plan_p1/"
            "goal59_bounded_terminal_residual_20261109"
        ),
        help="Goal59 root for training_curve.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/traces/analysis/goal60_trace_audit.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/traces/analysis/goal60_trace_audit.md"),
    )
    return parser.parse_args(argv)


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= 1.0e-15:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("Spearman inputs must have the same shape")
    if left_array.size < 2 or not np.all(np.isfinite(left_array)):
        return None
    if not np.all(np.isfinite(right_array)):
        return None
    return _pearson(_rankdata(left_array), _rankdata(right_array))


def _bootstrap_spearman(
    left: np.ndarray, right: np.ndarray, *, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, left.size, left.size)
        rho = _spearman(left[indices], right[indices])
        if rho is not None:
            values.append(rho)
    if not values:
        return {"valid_draws": 0, "ci_low": None, "ci_high": None}
    return {
        "valid_draws": len(values),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def _bootstrap_median_difference(
    left: np.ndarray, right: np.ndarray, *, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        left_sample = left[rng.integers(0, left.size, left.size)]
        right_sample = right[rng.integers(0, right.size, right.size)]
        values.append(float(np.median(left_sample) - np.median(right_sample)))
    return {
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def _bootstrap_proportion(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = values[rng.integers(0, values.size, values.size)]
        draws.append(float(np.mean(sample)))
    return {
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _episode_value(episodes: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    matches = [row for row in episodes if int(row.get("seed", -1)) == seed]
    if len(matches) != 1:
        raise ValueError(f"expected one episode for seed {seed}, got {len(matches)}")
    return matches[0]


def _resolve_config_path(path_text: str, trace_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    for root in (Path.cwd(), trace_dir, trace_dir.parents[1] if len(trace_dir.parents) > 1 else trace_dir):
        candidate = root / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve training config {path_text!r}")


def _actor_output_scale(trace_dir: Path) -> float:
    meta = json.loads((trace_dir / "trace_meta.json").read_text(encoding="utf-8"))
    config_path = _resolve_config_path(str(meta["training_config"]), trace_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return float(config["training"]["actor_output_scale"])


def _validate_run(trace_dir: Path, *, require_checkpoint: bool) -> dict[str, Any]:
    summary_path = trace_dir / "summary.json"
    episodes_path = trace_dir / "episodes.json"
    trace_path = trace_dir / "trace.npz"
    meta_path = trace_dir / "trace_meta.json"
    for path in (summary_path, episodes_path, trace_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != "auv_pc_command_residual_offpolicy_trace.v1":
        raise ValueError(f"{trace_dir}: unexpected trace schema")
    if meta.get("finite") is not True:
        raise ValueError(f"{trace_dir}: trace meta is not finite")
    if int(meta.get("seed_start", -1)) != DEV_SEEDS[0]:
        raise ValueError(f"{trace_dir}: seed_start mismatch")
    if int(meta.get("seed_count", -1)) != len(DEV_SEEDS):
        raise ValueError(f"{trace_dir}: seed_count mismatch")
    if not isinstance(episodes, list) or len(episodes) != len(DEV_SEEDS):
        raise ValueError(f"{trace_dir}: expected 40 episode records")
    episode_seeds = [int(row["seed"]) for row in episodes]
    if episode_seeds != list(DEV_SEEDS):
        raise ValueError(f"{trace_dir}: episode seeds do not match 20-59")
    if any(seed in HELDOUT_SEEDS for seed in episode_seeds):
        raise ValueError(f"{trace_dir}: held-out seed found")
    if require_checkpoint:
        if not summary_payload.get("checkpoint_sha256"):
            raise ValueError(f"{trace_dir}: missing checkpoint SHA256")
        if summary_payload.get("checkpoint_sha256") != meta.get("checkpoint_sha256"):
            raise ValueError(f"{trace_dir}: checkpoint SHA256 mismatch")
    with np.load(trace_path, allow_pickle=False) as trace:
        missing = REQUIRED_TRACE_FIELDS - set(trace.files)
        if missing:
            raise ValueError(f"{trace_dir}: missing trace fields {sorted(missing)}")
        seeds = sorted(int(value) for value in np.unique(trace["seed"]))
        if seeds != list(DEV_SEEDS):
            raise ValueError(f"{trace_dir}: trace seeds do not match 20-59")
        if any(seed in HELDOUT_SEEDS for seed in seeds):
            raise ValueError(f"{trace_dir}: held-out seed found in trace")
        done_count = int(np.sum(trace["done"].astype(bool)))
        if done_count != len(DEV_SEEDS):
            raise ValueError(f"{trace_dir}: expected 40 terminal rows, got {done_count}")
        finite = all(
            bool(np.all(np.isfinite(trace[name])))
            for name in trace.files
            if np.issubdtype(trace[name].dtype, np.floating)
        )
        if not finite:
            raise ValueError(f"{trace_dir}: non-finite trace array")
    return {
        "summary": summary_payload,
        "episodes": episodes,
        "meta": meta,
        "done_count": done_count,
        "trace_path": trace_path,
    }


def _window_sum(trace: np.lib.npyio.NpzFile, name: str, rows: np.ndarray) -> float:
    if name not in trace.files:
        return 0.0
    values = np.asarray(trace[name][rows], dtype=np.float64)
    return float(np.sum(values))


def _episode_metrics(
    trace_path: Path,
    episodes: list[dict[str, Any]],
    *,
    actor_output_scale: float,
    checkpoint_label: str,
    scenario: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    with np.load(trace_path, allow_pickle=False) as trace:
        for seed in DEV_SEEDS:
            indices = np.flatnonzero(trace["seed"] == seed)
            if indices.size == 0:
                raise ValueError(f"{trace_path}: no rows for seed {seed}")
            episode = _episode_value(episodes, seed)
            actions = np.asarray(trace["action"][indices], dtype=np.float64)
            residual = np.asarray(trace["residual"][indices], dtype=np.float64)
            base = np.asarray(trace["base_command"][indices], dtype=np.float64)
            terminal_window = indices[-min(TERMINAL_WINDOW_STEPS, indices.size) :]
            path_command = sum(
                _window_sum(trace, name, terminal_window)
                for name in PATH_COMPONENTS + COMMAND_COMPONENTS
            )
            terminal_reward = sum(
                _window_sum(trace, name, terminal_window)
                for name in TERMINAL_COMPONENTS
            )
            total_reward = _window_sum(trace, "reward_total", terminal_window)
            residual_norm = np.linalg.norm(residual, axis=1)
            base_norm = np.linalg.norm(base, axis=1)
            rows_out.append(
                {
                    "scenario": scenario,
                    "mode": mode,
                    "checkpoint": checkpoint_label,
                    "seed": int(seed),
                    "success": bool(episode["success"]),
                    "capture": bool(episode["terminal_eta_capture"]),
                    "goal_plane_miss": bool(episode["goal_plane_miss"]),
                    "timeout": bool(episode["timeout"]),
                    "safety": bool(episode["safety"]),
                    "terminal_kind": int(episode["terminal_kind"]),
                    "terminal_distance_m": _finite_float(
                        episode["terminal_eta_distance_m"]
                    ),
                    "terminal_signed_offset_m": _finite_float(
                        episode["terminal_signed_goal_offset_m"]
                    ),
                    "terminal_vertical_offset_m": _finite_float(
                        episode["terminal_vertical_offset_m"]
                    ),
                    "terminal_horizontal_offset_m": _finite_float(
                        episode["terminal_horizontal_offset_m"]
                    ),
                    "action_saturation_fraction": float(
                        np.mean(np.abs(actions) >= SATURATION_FRACTION * actor_output_scale)
                    ),
                    "residual_l2_mean": float(np.mean(residual_norm)),
                    "residual_to_base_ratio": float(
                        np.mean(residual_norm / np.maximum(base_norm, 1.0e-12))
                    ),
                    "path_command_reward_terminal_window": float(path_command),
                    "path_reward_terminal_window": float(
                        sum(
                            _window_sum(trace, name, terminal_window)
                            for name in PATH_COMPONENTS
                        )
                    ),
                    "command_reward_terminal_window": float(
                        sum(
                            _window_sum(trace, name, terminal_window)
                            for name in COMMAND_COMPONENTS
                        )
                    ),
                    "terminal_reward_terminal_window": float(terminal_reward),
                    "total_reward_terminal_window": float(total_reward),
                }
            )
    return rows_out


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    capture = np.asarray([row["capture"] for row in rows], dtype=bool)
    sat = np.asarray(
        [row["action_saturation_fraction"] for row in rows], dtype=np.float64
    )
    residual = np.asarray([row["residual_l2_mean"] for row in rows], dtype=np.float64)
    distance = np.asarray([row["terminal_distance_m"] for row in rows], dtype=np.float64)
    path_reward = np.asarray(
        [row["path_command_reward_terminal_window"] for row in rows], dtype=np.float64
    )
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean(success)),
        "capture_rate": float(np.mean(capture)),
        "action_saturation_mean": float(np.mean(sat)),
        "residual_l2_mean": float(np.mean(residual)),
        "terminal_distance_mean_m": float(np.mean(distance)),
        "path_command_reward_window_mean": float(np.mean(path_reward)),
    }


def _h_a(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    sat = np.asarray(
        [row["action_saturation_fraction"] for row in rows], dtype=np.float64
    )
    residual = np.asarray([row["residual_l2_mean"] for row in rows], dtype=np.float64)
    distance = np.asarray([row["terminal_distance_m"] for row in rows], dtype=np.float64)
    rho_sat = _spearman(sat, distance)
    rho_residual = _spearman(residual, distance)
    ci_sat = _bootstrap_spearman(sat, distance, seed=seed)
    ci_residual = _bootstrap_spearman(residual, distance, seed=seed + 1)
    failed = ~success
    median_diff_sat = None
    median_diff_residual = None
    ci_diff_sat = None
    ci_diff_residual = None
    if failed.any() and success.any():
        median_diff_sat = float(np.median(sat[failed]) - np.median(sat[success]))
        median_diff_residual = float(
            np.median(residual[failed]) - np.median(residual[success])
        )
        ci_diff_sat = _bootstrap_median_difference(sat[failed], sat[success], seed=seed + 2)
        ci_diff_residual = _bootstrap_median_difference(
            residual[failed], residual[success], seed=seed + 3
        )
    sat_supported = bool(
        rho_sat is not None
        and rho_sat >= RHO_MIN
        and ci_sat["ci_low"] is not None
        and ci_sat["ci_low"] > 0.0
        and median_diff_sat is not None
        and median_diff_sat > SAT_MEDIAN_DIFF_MIN
    )
    residual_supported = bool(
        rho_residual is not None
        and rho_residual >= RHO_MIN
        and ci_residual["ci_low"] is not None
        and ci_residual["ci_low"] > 0.0
        and median_diff_residual is not None
        and median_diff_residual > SAT_MEDIAN_DIFF_MIN
    )
    return {
        "rho_saturation_vs_terminal_distance": rho_sat,
        "saturation_ci": ci_sat,
        "rho_residual_vs_terminal_distance": rho_residual,
        "residual_ci": ci_residual,
        "failed_minus_success_saturation_median": median_diff_sat,
        "saturation_diff_ci": ci_diff_sat,
        "failed_minus_success_residual_median": median_diff_residual,
        "residual_diff_ci": ci_diff_residual,
        "saturation_supported": sat_supported,
        "residual_supported": residual_supported,
        "supported": bool(sat_supported or residual_supported),
    }


def _h_b(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    capture = np.asarray([row["capture"] for row in rows], dtype=bool)
    path_reward = np.asarray(
        [row["path_command_reward_terminal_window"] for row in rows], dtype=np.float64
    )
    path_only = np.asarray(
        [row["path_reward_terminal_window"] for row in rows], dtype=np.float64
    )
    command_only = np.asarray(
        [row["command_reward_terminal_window"] for row in rows], dtype=np.float64
    )
    total_reward = np.asarray(
        [row["total_reward_terminal_window"] for row in rows], dtype=np.float64
    )
    distance = np.asarray([row["terminal_distance_m"] for row in rows], dtype=np.float64)
    failed = ~success
    conflict_failed = failed & (path_reward > 0.0) & (~capture)
    conflict_all = (path_reward > 0.0) & (~capture)
    conflict_fraction_failed = float(np.mean(conflict_failed[failed])) if failed.any() else 0.0
    conflict_fraction_all = float(np.mean(conflict_all))
    ci_failed = _bootstrap_proportion(conflict_failed[failed], seed=seed + 4) if failed.any() else None
    rho = _spearman(path_reward, distance)
    ci = _bootstrap_spearman(path_reward, distance, seed=seed + 5)
    supported = bool(
        conflict_fraction_failed >= CONFLICT_FRACTION_MIN
        or (
            rho is not None
            and rho >= RHO_MIN
            and ci["ci_low"] is not None
            and ci["ci_low"] > 0.0
        )
    )
    failed_count = max(int(np.sum(failed)), 1)
    return {
        "failed_episodes": int(np.sum(failed)),
        "failed_path_positive_capture_negative_fraction": conflict_fraction_failed,
        "failed_conflict_ci": ci_failed,
        "all_path_positive_capture_negative_fraction": conflict_fraction_all,
        "failed_path_reward_mean": float(np.mean(path_only[failed])) if failed.any() else None,
        "failed_command_reward_mean": float(np.mean(command_only[failed])) if failed.any() else None,
        "failed_total_reward_mean": float(np.mean(total_reward[failed])) if failed.any() else None,
        "failed_path_positive_fraction": float(np.mean(path_only[failed] > 0.0)) if failed.any() else None,
        "failed_command_positive_fraction": float(np.mean(command_only[failed] > 0.0)) if failed.any() else None,
        "failed_total_positive_fraction": float(np.mean(total_reward[failed] > 0.0)) if failed.any() else None,
        "rho_path_reward_vs_terminal_distance": rho,
        "path_reward_ci": ci,
        "supported": supported,
        "failed_count_used": failed_count,
    }


def _h_c(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    capture = np.asarray([row["capture"] for row in rows], dtype=bool)
    terminal_kind = np.asarray([row["terminal_kind"] for row in rows], dtype=np.int64)
    kind_goal = terminal_kind == 1
    discordance = success != capture
    kind_discordance = kind_goal != capture
    discordance_rate = float(np.mean(discordance))
    kind_discordance_rate = float(np.mean(kind_discordance))
    distances = np.asarray([row["terminal_distance_m"] for row in rows], dtype=np.float64)
    signed = np.asarray(
        [row["terminal_signed_offset_m"] for row in rows], dtype=np.float64
    )
    group_masks = {
        "success_capture": success & capture,
        "success_no_capture": success & ~capture,
        "no_success_capture": ~success & capture,
        "no_success_no_capture": ~success & ~capture,
    }
    group_stats = {}
    for name, mask in group_masks.items():
        group_stats[name] = {
            "count": int(np.sum(mask)),
            "terminal_distance_mean_m": float(np.mean(distances[mask])) if mask.any() else None,
            "terminal_distance_median_m": float(np.median(distances[mask])) if mask.any() else None,
            "signed_offset_mean_m": float(np.mean(signed[mask])) if mask.any() else None,
        }
    return {
        "success_capture_2x2": {
            "success_capture": int(np.sum(success & capture)),
            "success_no_capture": int(np.sum(success & ~capture)),
            "no_success_capture": int(np.sum(~success & capture)),
            "no_success_no_capture": int(np.sum(~success & ~capture)),
        },
        "groups": group_stats,
        "success_capture_discordance_rate": discordance_rate,
        "success_capture_discordance_ci": _bootstrap_proportion(discordance, seed=seed + 6),
        "terminal_kind_goal_vs_capture_discordance_rate": kind_discordance_rate,
        "terminal_kind_goal_vs_capture_discordance_ci": _bootstrap_proportion(
            kind_discordance, seed=seed + 7
        ),
        "supported": bool(
            discordance_rate >= CAPTURE_DISCORDANCE_MIN
            or kind_discordance_rate >= CAPTURE_DISCORDANCE_MIN
        ),
    }


def _load_training_curve(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    values: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    if not values:
        return None
    ratios = np.asarray(
        [float(row["residual_l2_ratio"]) for row in values], dtype=np.float64
    )
    if ratios.size < 10 or not np.all(np.isfinite(ratios)):
        return None
    block = max(1, ratios.size // 10)
    first = float(np.mean(ratios[:block]))
    previous = float(np.mean(ratios[-2 * block : -block]))
    last = float(np.mean(ratios[-block:]))
    return {
        "start_residual_l2_ratio": first,
        "previous10_residual_l2_ratio": previous,
        "last10_residual_l2_ratio": last,
        "last10_minus_previous10": last - previous,
        "end_minus_start": last - first,
    }


def _h_d_e(
    policy_rows: list[dict[str, Any]],
    *,
    goal59_root: Path,
    scenario: str,
) -> dict[str, Any]:
    metrics = _metrics(policy_rows)
    curves = {
        str(seed): _load_training_curve(
            goal59_root / "artifacts" / f"formal_s{seed}" / "training_curve.csv"
        )
        for seed in POLICY_SEEDS
    }
    valid_curves = {key: value for key, value in curves.items() if value is not None}
    last10 = [
        value["last10_minus_previous10"] for value in valid_curves.values()
    ]
    h_d = {
        "action_saturation_mean": metrics["action_saturation_mean"],
        "residual_l2_mean": metrics["residual_l2_mean"],
        "residual_to_base_ratio_mean": float(
            np.mean([row["residual_to_base_ratio"] for row in policy_rows])
        ),
        "supported": bool(
            metrics["action_saturation_mean"] < 0.50
            and float(np.mean([row["residual_to_base_ratio"] for row in policy_rows]))
            < 0.05
            and bool(last10)
            and float(np.mean(last10)) > 0.01
        ),
    }
    h_e = {
        "training_curves": curves,
        "mean_last10_minus_previous10": float(np.mean(last10)) if last10 else None,
        "supported": bool(last10 and float(np.mean(last10)) > 0.02),
        "note": "Read-only trend; no new training or UTD comparison was run.",
    }
    return {"H_D": h_d, "H_E": h_e}


def _analyze_scenario(
    trace_root: Path,
    scenario: str,
    *,
    goal59_root: Path,
    seed: int,
) -> dict[str, Any]:
    policy_rows: list[dict[str, Any]] = []
    checkpoint_metrics: dict[str, dict[str, Any]] = {}
    checkpoint_rows: dict[str, list[dict[str, Any]]] = {}
    data_integrity: dict[str, Any] = {"runs": {}}
    for checkpoint in POLICY_SEEDS:
        trace_dir = trace_root / f"{scenario}_policy_s{checkpoint}"
        validated = _validate_run(trace_dir, require_checkpoint=True)
        scale = _actor_output_scale(trace_dir)
        rows = _episode_metrics(
            validated["trace_path"],
            validated["episodes"],
            actor_output_scale=scale,
            checkpoint_label=str(checkpoint),
            scenario=scenario,
            mode="policy",
        )
        policy_rows.extend(rows)
        checkpoint_rows[str(checkpoint)] = rows
        checkpoint_metrics[str(checkpoint)] = _metrics(rows)
        data_integrity["runs"][trace_dir.name] = {
            "done_count": validated["done_count"],
            "checkpoint_sha256": validated["meta"]["checkpoint_sha256"],
            "episodes": len(validated["episodes"]),
        }
    zero_dir = trace_root / f"{scenario}_zero"
    zero_validated = _validate_run(zero_dir, require_checkpoint=False)
    zero_scale = _actor_output_scale(zero_dir)
    zero_rows = _episode_metrics(
        zero_validated["trace_path"],
        zero_validated["episodes"],
        actor_output_scale=zero_scale,
        checkpoint_label="zero",
        scenario=scenario,
        mode="zero-residual",
    )
    data_integrity["runs"][zero_dir.name] = {
        "done_count": zero_validated["done_count"],
        "checkpoint_sha256": zero_validated["meta"].get("checkpoint_sha256"),
        "episodes": len(zero_validated["episodes"]),
    }
    data_integrity["all_valid"] = True
    checkpoint_diagnostics = {
        checkpoint: {
            "H_A": _h_a(rows, seed=seed + 300 + index * 10),
            "H_B": _h_b(rows, seed=seed + 400 + index * 10),
            "H_C": _h_c(rows, seed=seed + 500 + index * 10),
        }
        for index, (checkpoint, rows) in enumerate(checkpoint_rows.items())
    }
    return {
        "scenario": scenario,
        "policy_checkpoints": checkpoint_metrics,
        "checkpoint_diagnostics": checkpoint_diagnostics,
        "policy_pooled": _metrics(policy_rows),
        "zero_residual": _metrics(zero_rows),
        "H_A": _h_a(policy_rows, seed=seed),
        "H_B": _h_b(policy_rows, seed=seed + 100),
        "H_C": _h_c(policy_rows, seed=seed + 200),
        **_h_d_e(policy_rows, goal59_root=goal59_root, scenario=scenario),
        "data_integrity": data_integrity,
        "episodes": {
            "policy": policy_rows,
            "zero": zero_rows,
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal60 只读 trace 审计",
        "",
        "日期：2026-11-10<br>",
        "协议：Goal59 `final.pt`，dev 20–59，default/strict125，CPU/float64，"
        "`endpoint_stop`；不做训练，不打开 held-out。",
        "",
        "## 数据完整性",
        "",
        "| scenario | run | episodes | done | checkpoint SHA |",
        "|---|---|---:|---:|---|",
    ]
    for scenario, result in payload["scenarios"].items():
        for name, info in result["data_integrity"]["runs"].items():
            sha = info["checkpoint_sha256"]
            lines.append(
                f"| {scenario} | {name} | {info['episodes']} | {info['done_count']} | "
                f"`{str(sha)[:12] if sha else 'zero'}` |"
            )
    lines.extend(["", "## 汇总", ""])
    for scenario, result in payload["scenarios"].items():
        lines.extend(
            [
                f"### {scenario}",
                "",
                "| checkpoint | success | capture | sat | residual L2 | conflict |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for checkpoint, metrics in result["policy_checkpoints"].items():
            lines.append(
                f"| {checkpoint} | {_fmt(metrics['success_rate'])} | "
                f"{_fmt(metrics['capture_rate'])} | "
                f"{_fmt(metrics['action_saturation_mean'])} | "
                f"{_fmt(metrics['residual_l2_mean'])} | "
                f"{_fmt(result['H_B']['failed_path_positive_capture_negative_fraction'])} |"
            )
        lines.extend(
            [
                "",
                "| checkpoint | H-A rho | H-A | H-B conflict | H-B | H-C discord | H-C |",
                "|---|---:|---|---:|---|---:|---|",
            ]
        )
        for checkpoint, diag in result["checkpoint_diagnostics"].items():
            lines.append(
                f"| {checkpoint} | {_fmt(diag['H_A']['rho_saturation_vs_terminal_distance'])} | "
                f"{_fmt(diag['H_A']['supported'])} | "
                f"{_fmt(diag['H_B']['failed_path_positive_capture_negative_fraction'])} | "
                f"{_fmt(diag['H_B']['supported'])} | "
                f"{_fmt(diag['H_C']['success_capture_discordance_rate'])} | "
                f"{_fmt(diag['H_C']['supported'])} |"
            )
        lines.extend(
            [
                "",
                f"- H-A pooled: rho_sat={_fmt(result['H_A']['rho_saturation_vs_terminal_distance'])}, "
                f"CI=[{_fmt(result['H_A']['saturation_ci']['ci_low'])}, "
                f"{_fmt(result['H_A']['saturation_ci']['ci_high'])}], "
                f"failed-success median={_fmt(result['H_A']['failed_minus_success_saturation_median'])}, "
                f"supported={_fmt(result['H_A']['supported'])}.",
                f"- H-B pooled: conflict={_fmt(result['H_B']['failed_path_positive_capture_negative_fraction'])}, "
                f"failed path/command/total mean=({_fmt(result['H_B']['failed_path_reward_mean'])}, "
                f"{_fmt(result['H_B']['failed_command_reward_mean'])}, "
                f"{_fmt(result['H_B']['failed_total_reward_mean'])}), "
                f"rho_path_distance={_fmt(result['H_B']['rho_path_reward_vs_terminal_distance'])}, "
                f"CI=[{_fmt(result['H_B']['path_reward_ci']['ci_low'])}, "
                f"{_fmt(result['H_B']['path_reward_ci']['ci_high'])}], "
                f"supported={_fmt(result['H_B']['supported'])}.",
                f"- H-C pooled: success/capture discordance={_fmt(result['H_C']['success_capture_discordance_rate'])}, "
                f"kind/capture discordance={_fmt(result['H_C']['terminal_kind_goal_vs_capture_discordance_rate'])}, "
                f"success_no_capture={result['H_C']['groups']['success_no_capture']['count']} "
                f"(distance mean={_fmt(result['H_C']['groups']['success_no_capture']['terminal_distance_mean_m'])} m), "
                f"supported={_fmt(result['H_C']['supported'])}.",
                f"- H-D: supported={_fmt(result['H_D']['supported'])}; "
                f"H-E: supported={_fmt(result['H_E']['supported'])}; "
                f"H-E curve last10 delta={_fmt(result['H_E'].get('mean_last10_minus_previous10'))}.",
                "",
            ]
        )
    lines.extend(
        [
            "## 判读",
            "",
            "H-A/H-B/H-C 的 supported 只是预注册诊断条件，不是性能 PASS。"
            "只有所有数据完整性检查通过，且至少一个假设被支持，才进入对应的修复 Goal；"
            "都不支持时写 blocker 和新的可证伪 hypothesis，不启动 SAC/REDQ/DroQ。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios: dict[str, Any] = {}
    for index, scenario in enumerate(SCENARIOS):
        scenarios[scenario] = _analyze_scenario(
            args.trace_root,
            scenario,
            goal59_root=args.goal59_root,
            seed=BOOTSTRAP_SEED + index * 1000,
        )
    overall = {
        "H_A_supported_scenarios": [
            name for name, result in scenarios.items() if result["H_A"]["supported"]
        ],
        "H_B_supported_scenarios": [
            name for name, result in scenarios.items() if result["H_B"]["supported"]
        ],
        "H_C_supported_scenarios": [
            name for name, result in scenarios.items() if result["H_C"]["supported"]
        ],
        "H_D_supported_scenarios": [
            name for name, result in scenarios.items() if result["H_D"]["supported"]
        ],
        "H_E_supported_scenarios": [
            name for name, result in scenarios.items() if result["H_E"]["supported"]
        ],
        "heldout_opened": False,
        "training_started": False,
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "scenarios": scenarios,
        "overall": overall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
