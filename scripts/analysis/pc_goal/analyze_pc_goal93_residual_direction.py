#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EPS = 1e-12
GATE_ACTIVE_THRESHOLD = 0.05
TERMINAL_WINDOW_S = 1.0
SUCCESS_KIND = 1
PLANE_MISS_KIND = 9
TIMEOUT_KIND = 2
SCENARIO_GOAL_RADIUS = {"default": 2.0, "strict125": 1.25}
SCALE_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Goal93 residual direction and actuator authority"
    )
    parser.add_argument("--scale-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=93)
    return parser.parse_args(argv)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, EPS)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("cosine_rows expects two (N, 3) arrays")
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > EPS
    result[valid] = numerator[valid] / denominator[valid]
    return result


def survival_rows(applied: np.ndarray, effective: np.ndarray) -> np.ndarray:
    applied = np.asarray(applied, dtype=np.float64)
    effective = np.asarray(effective, dtype=np.float64)
    if applied.shape != effective.shape or applied.ndim != 2 or applied.shape[1] != 3:
        raise ValueError("survival_rows expects two (N, 3) arrays")
    numerator = np.sum(applied * effective, axis=1)
    denominator = np.sum(applied * applied, axis=1)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > EPS
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _nanmedian(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def split_episode_trace(trace: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    if "episode_seed" not in trace or "step" not in trace:
        raise ValueError("trace must contain episode_seed and step")
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64).reshape(-1)
    steps = np.asarray(trace["step"], dtype=np.int64).reshape(-1)
    episodes: dict[int, dict[str, np.ndarray]] = {}
    for seed in np.unique(seeds):
        index = np.flatnonzero(seeds == seed)
        order = index[np.argsort(steps[index], kind="stable")]
        episodes[int(seed)] = {
            name: np.asarray(value)[order]
            for name, value in trace.items()
            if np.asarray(value).shape[0] == seeds.shape[0]
        }
    return episodes


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _channel_survival(applied: np.ndarray, effective: np.ndarray) -> list[float]:
    result: list[float] = []
    for channel in range(3):
        values = applied[:, channel]
        valid = np.abs(values) > EPS
        if not np.any(valid):
            result.append(float("nan"))
            continue
        ratio = effective[valid, channel] / values[valid]
        result.append(_nanmedian(ratio))
    return result


def _channel_clip_fraction(applied: np.ndarray, effective: np.ndarray) -> list[float]:
    tolerance = np.array([1e-6, 1e-6, 1e-3])
    clipped = np.abs(effective - applied) > tolerance
    return [float(np.mean(clipped[:, channel])) for channel in range(3)]


def episode_summary(
    episode: dict[str, np.ndarray],
    goal_radius: float,
    action_full_scale: np.ndarray,
    residual_scale: float,
) -> dict[str, Any]:
    gate = np.asarray(episode["frozen_residual_gate"], dtype=np.float64).reshape(-1)
    mask = gate > GATE_ACTIVE_THRESHOLD
    r_ppo = np.asarray(episode["frozen_residual_action"], dtype=np.float64)
    r_bc = np.asarray(episode["frozen_residual_base_action"], dtype=np.float64)
    applied = np.asarray(episode["actuator_residual_applied"], dtype=np.float64)
    effective = np.asarray(episode["actuator_residual_effective"], dtype=np.float64)
    terminal_kind = int(np.asarray(episode["terminal_kind"]).reshape(-1)[-1])
    time_s = np.asarray(episode["time_s"], dtype=np.float64).reshape(-1)
    terminal_mask = time_s >= float(time_s[-1]) - TERMINAL_WINDOW_S
    distance = np.asarray(episode["goal_distance_eta_m"], dtype=np.float64).reshape(-1)
    cos_all = cosine_rows(r_ppo, r_bc)
    survival_all = survival_rows(applied, effective)
    ppo_norm = np.linalg.norm(r_ppo, axis=1)
    bc_norm = np.linalg.norm(r_bc, axis=1)
    ppo_physical = r_ppo * gate[:, None] * residual_scale * action_full_scale[None, :]
    bc_physical = r_bc * residual_scale * action_full_scale[None, :]
    ppo_physical_norm = np.linalg.norm(ppo_physical, axis=1)
    ppo_fin_physical = np.linalg.norm(ppo_physical[:, :2], axis=1)
    bc_fin_physical = np.linalg.norm(bc_physical[:, :2], axis=1)
    applied_norm = np.linalg.norm(applied, axis=1)
    effective_norm = np.linalg.norm(effective, axis=1)
    survival_channels = _channel_survival(applied, effective)
    clip_channels = _channel_clip_fraction(applied, effective)
    fin_clip = np.any(
        np.abs(effective[:, :2] - applied[:, :2]) > np.array([1e-6, 1e-6]),
        axis=1,
    )
    ppo_share = np.abs(r_ppo) / np.maximum(np.abs(r_ppo) + np.abs(r_bc), EPS)
    return {
        "seed": int(np.asarray(episode["episode_seed"]).reshape(-1)[0]),
        "terminal_kind": terminal_kind,
        "success": terminal_kind == SUCCESS_KIND,
        "plane_miss": terminal_kind == PLANE_MISS_KIND,
        "timeout": terminal_kind == TIMEOUT_KIND,
        "gate_active_fraction": float(np.mean(mask)),
        "cos_gate_median": _nanmedian(cos_all[mask]),
        "cos_terminal_median": _nanmedian(cos_all[terminal_mask]),
        "survival_gate_median": _nanmedian(survival_all[mask]),
        "survival_gate_mean": _nanmean(survival_all[mask]),
        "survival_rudder_median": survival_channels[0],
        "survival_stern_median": survival_channels[1],
        "survival_rpm_median": survival_channels[2],
        "clip_fraction_rudder": clip_channels[0],
        "clip_fraction_stern": clip_channels[1],
        "clip_fraction_rpm": clip_channels[2],
        "fin_clip_step_fraction": float(np.mean(fin_clip)),
        "applied_l2_gate_mean": _nanmean(applied_norm[mask]),
        "effective_l2_gate_mean": _nanmean(effective_norm[mask]),
        "ppo_l2_gate_mean": _nanmean(ppo_norm[mask]),
        "ppo_physical_l2_gate_mean": _nanmean(ppo_physical_norm[mask]),
        "ppo_fin_physical_l2_gate_mean": _nanmean(ppo_fin_physical[mask]),
        "ppo_rpm_physical_abs_gate_mean": _nanmean(np.abs(ppo_physical[mask, 2])),
        "bc_fin_physical_l2_gate_mean": _nanmean(bc_fin_physical[mask]),
        "bc_rpm_physical_abs_gate_mean": _nanmean(np.abs(bc_physical[mask, 2])),
        "ppo_share_rudder_median": _nanmedian(ppo_share[mask, 0]),
        "ppo_share_stern_median": _nanmedian(ppo_share[mask, 1]),
        "ppo_share_rpm_median": _nanmedian(ppo_share[mask, 2]),
        "bc_l2_gate_mean": _nanmean(bc_norm[mask]),
        "clip_step_fraction": float(np.mean(np.any(np.abs(effective - applied) > np.array([1e-6, 1e-6, 1e-3]), axis=1))),
        "final_goal_distance_m": float(distance[-1]),
        "success_margin_m": float(goal_radius - distance[-1]),
        "final_signed_tangent_offset_m": float(
            np.asarray(episode["signed_goal_tangent_offset_eta_m"]).reshape(-1)[-1]
        ),
    }


def summarize_trace(
    path: Path,
    goal_radius: float,
    action_full_scale: np.ndarray,
    residual_scale: float,
) -> dict[str, Any]:
    episodes = split_episode_trace(load_trace(path))
    summaries = [
        episode_summary(episode, goal_radius, action_full_scale, residual_scale)
        for episode in episodes.values()
    ]
    successes = [row for row in summaries if row["success"]]
    failures = [row for row in summaries if row["plane_miss"]]
    return {
        "count": len(summaries),
        "success": len(successes),
        "plane_miss": len(failures),
        "timeout": sum(1 for row in summaries if row["timeout"]),
        "episodes": summaries,
        "all": _aggregate(summaries),
        "success_group": _aggregate(successes),
        "failure_group": _aggregate(failures),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "cos_gate_median",
        "cos_terminal_median",
        "survival_gate_median",
        "survival_gate_mean",
        "survival_rudder_median",
        "survival_stern_median",
        "survival_rpm_median",
        "clip_fraction_rudder",
        "clip_fraction_stern",
        "clip_fraction_rpm",
        "fin_clip_step_fraction",
        "applied_l2_gate_mean",
        "effective_l2_gate_mean",
        "ppo_l2_gate_mean",
        "ppo_physical_l2_gate_mean",
        "ppo_fin_physical_l2_gate_mean",
        "ppo_rpm_physical_abs_gate_mean",
        "bc_fin_physical_l2_gate_mean",
        "bc_rpm_physical_abs_gate_mean",
        "ppo_share_rudder_median",
        "ppo_share_stern_median",
        "ppo_share_rpm_median",
        "bc_l2_gate_mean",
        "clip_step_fraction",
        "final_goal_distance_m",
        "success_margin_m",
        "final_signed_tangent_offset_m",
    )
    return {
        name: _nanmedian(np.asarray([row[name] for row in rows], dtype=np.float64))
        for name in names
    }


def cluster_bootstrap_median_difference(
    rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
    *,
    left_flag: str = "success",
    right_flag: str = "plane_miss",
    samples: int = 2000,
    seed: int = 93,
) -> dict[str, Any]:
    left = np.asarray(
        [row[left_key] for row in rows if row.get(left_flag)], dtype=np.float64
    )
    right = np.asarray(
        [row[right_key] for row in rows if row.get(right_flag)], dtype=np.float64
    )
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size < 5 or right.size < 5:
        return {
            "left_n": int(left.size),
            "right_n": int(right.size),
            "difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "supported": False,
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        left_sample = rng.choice(left, size=left.size, replace=True)
        right_sample = rng.choice(right, size=right.size, replace=True)
        draws[index] = np.median(left_sample) - np.median(right_sample)
    return {
        "left_n": int(left.size),
        "right_n": int(right.size),
        "difference": float(np.median(left) - np.median(right)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "supported": bool(np.percentile(draws, 2.5) > 0.0),
    }


def _run_identity(summary: dict[str, Any], path: Path) -> tuple[str, str, float, str]:
    training_config = Path(str(summary.get("training_config", ""))).name
    arm = Path(training_config).stem
    environment = str(summary.get("environment_config", ""))
    scenario = "strict125" if "strict125" in environment else "default"
    scale = _safe_float(summary.get("actuator_residual_scale_override"))
    if not math.isfinite(scale):
        raise ValueError(f"{path}: missing actuator_residual_scale_override")
    mode = str(summary.get("action_mode", ""))
    return arm, scenario, scale, mode


def load_scale_root(root: Path) -> dict[tuple[str, str, float], dict[str, Any]]:
    result: dict[tuple[str, str, float], dict[str, Any]] = {}
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text())
        arm, scenario, scale, mode = _run_identity(summary, path)
        if mode != "policy":
            continue
        metrics = summary.get("metrics", {})
        episodes = summary.get("episodes", [])
        per_seed = {
            int(row["seed"]): bool(row.get("success_eta_tight_at_done"))
            for row in episodes
        }
        result[(arm, scenario, scale)] = {
            "summary_path": str(path),
            "success": int(metrics.get("success_count_eta_tight_at_done", 0)),
            "n": len(episodes),
            "success_rate": _safe_float(metrics.get("success_rate_eta_tight_at_done")),
            "plane_miss_rate": _safe_float(metrics.get("goal_plane_miss_rate")),
            "timeout_rate": _safe_float(metrics.get("timeout_rate")),
            "path_three_d_rmse_m": _safe_float(metrics.get("path_three_d_rmse_m")),
            "pre_goal_3d_rmse_m": _safe_float(metrics.get("pre_goal_3d_rmse_m")),
            "magnitude_clip_rate": _safe_float(
                metrics.get("actuator_residual_magnitude_clip_rate")
            ),
            "rate_clip_rate": _safe_float(
                metrics.get("actuator_residual_rate_clip_rate")
            ),
            "residual_action_abs_mean": _safe_float(
                metrics.get("residual_action_abs_mean")
            ),
            "residual_to_base_ratio": _safe_float(
                metrics.get("residual_to_base_l2_ratio_mean")
            ),
            "eta_done_distance_m": _safe_float(
                metrics.get("eta_done_distance_m_mean")
            ),
            "success_margin_m": _safe_float(metrics.get("success_margin_m_mean")),
            "per_seed": per_seed,
        }
    return result


def load_trace_root(root: Path) -> dict[tuple[str, str, float, str], dict[str, Any]]:
    result: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text())
        arm, scenario, scale, mode = _run_identity(summary, path)
        trace_path = path.parent / "trace.npz"
        if not trace_path.is_file():
            continue
        metadata = summary.get("checkpoint_metadata", {})
        full_scale = np.asarray(
            metadata.get("action_full_scale", [1.0, 1.0, 1.0]), dtype=np.float64
        )
        if full_scale.shape != (3,):
            raise ValueError(f"{path}: invalid action_full_scale")
        result[(arm, scenario, scale, mode)] = summarize_trace(
            trace_path,
            SCENARIO_GOAL_RADIUS[scenario],
            full_scale,
            scale,
        )
    return result


def paired_gain_loss(
    baseline: dict[int, bool], candidate: dict[int, bool]
) -> dict[str, int]:
    gain = sum(
        1
        for seed, value in candidate.items()
        if value and not baseline.get(seed, False)
    )
    loss = sum(
        1
        for seed, value in baseline.items()
        if value and not candidate.get(seed, False)
    )
    return {"gain": int(gain), "loss": int(loss), "net": int(gain - loss)}


def _scale_table(
    scale_data: dict[tuple[str, str, float], dict[str, Any]],
    arm: str,
    scenario: str,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    baseline = scale_data.get((arm, scenario, 1.0))
    for scale in SCALE_VALUES:
        row = scale_data.get((arm, scenario, scale))
        if row is None:
            continue
        previous = scale_data.get((arm, scenario, 1.25))
        paired = (
            paired_gain_loss(baseline["per_seed"], row["per_seed"])
            if baseline is not None
            else {"gain": 0, "loss": 0, "net": 0}
        )
        rows[f"{scale:.2f}"] = {
            **{key: value for key, value in row.items() if key != "per_seed"},
            "paired_vs_scale1p00": paired,
            "clip_delta_vs_scale1p25": (
                row["magnitude_clip_rate"] - previous["magnitude_clip_rate"]
                if previous is not None
                else float("nan")
            ),
        }
    return rows


def _direction_table(
    trace_data: dict[tuple[str, str, float, str], dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for (arm, scenario, scale, mode), summary in sorted(trace_data.items()):
        if mode != "policy":
            continue
        if scale not in (1.0, 1.25):
            continue
        key = f"{arm}/{scenario}/scale={scale:.2f}"
        gate_cos = cluster_bootstrap_median_difference(
            summary["episodes"],
            "cos_gate_median",
            "cos_gate_median",
            samples=samples,
            seed=seed,
        )
        terminal_cos = cluster_bootstrap_median_difference(
            summary["episodes"],
            "cos_terminal_median",
            "cos_terminal_median",
            samples=samples,
            seed=seed + 1,
        )
        survival = cluster_bootstrap_median_difference(
            summary["episodes"],
            "survival_gate_median",
            "survival_gate_median",
            samples=samples,
            seed=seed + 2,
        )
        rows[key] = {
            "count": summary["count"],
            "success": summary["success"],
            "plane_miss": summary["plane_miss"],
            "all": summary["all"],
            "success_group": summary["success_group"],
            "failure_group": summary["failure_group"],
            "cos_difference": gate_cos,
            "terminal_cos_difference": terminal_cos,
            "survival_difference": survival,
        }
    return rows


def _terminal_comparison(
    trace_data: dict[tuple[str, str, float, str], dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scenario in ("default", "strict125"):
        policy = trace_data.get(("pg10_3", scenario, 1.25, "policy"))
        frozen = trace_data.get(("pg10_3", scenario, 1.25, "frozen-base"))
        if policy is None or frozen is None:
            continue
        policy_by_seed = {row["seed"]: row for row in policy["episodes"]}
        frozen_by_seed = {row["seed"]: row for row in frozen["episodes"]}
        shared = sorted(set(policy_by_seed) & set(frozen_by_seed))
        differences = np.asarray(
            [
                policy_by_seed[seed]["final_goal_distance_m"]
                - frozen_by_seed[seed]["final_goal_distance_m"]
                for seed in shared
            ],
            dtype=np.float64,
        )
        flips = [
            seed
            for seed in shared
            if policy_by_seed[seed]["success"] != frozen_by_seed[seed]["success"]
        ]
        radius = SCENARIO_GOAL_RADIUS[scenario]
        flip_margins = [
            abs(radius - frozen_by_seed[seed]["final_goal_distance_m"])
            for seed in flips
        ] + [
            abs(radius - policy_by_seed[seed]["final_goal_distance_m"])
            for seed in flips
        ]
        success_margins = [
            row["success_margin_m"]
            for row in policy["episodes"]
            if row["success"]
        ]
        result[scenario] = {
            "paired_episodes": len(shared),
            "policy_success": policy["success"],
            "frozen_success": frozen["success"],
            "flips": int(len(flips)),
            "done_distance_difference_median_m": _nanmedian(differences),
            "done_distance_difference_mean_m": _nanmean(differences),
            "success_margin_median_m": _nanmedian(
                np.asarray(success_margins, dtype=np.float64)
            ),
            "flip_shell_margin_max_m": (
                float(np.max(flip_margins)) if flip_margins else float("nan")
            ),
            "flip_shell_within_5cm": bool(
                not flip_margins or float(np.max(flip_margins)) <= 0.05
            ),
        }
    return result


def _has_direction_support(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    gate = row["cos_difference"]
    terminal = row["terminal_cos_difference"]
    return bool(
        (
            row["all"].get("cos_gate_median", float("nan")) >= 0.2
            and gate.get("supported", False)
            and gate.get("difference", 0.0) >= 0.1
        )
        or (
            terminal.get("supported", False)
            and terminal.get("difference", 0.0) >= 0.1
        )
    )


def decide(
    scale_data: dict[tuple[str, str, float], dict[str, Any]],
    trace_data: dict[tuple[str, str, float, str], dict[str, Any]],
    direction_table: dict[str, Any],
    terminal_table: dict[str, Any],
) -> dict[str, Any]:
    pg = scale_data.get(("pg10_3", "strict125", 1.0))
    pg125 = scale_data.get(("pg10_3", "strict125", 1.25))
    pg15 = scale_data.get(("pg10_3", "strict125", 1.5))
    pg_default_125 = scale_data.get(("pg10_3", "default", 1.25))
    pg_default_15 = scale_data.get(("pg10_3", "default", 1.5))

    authority_not_binding = False
    authority_binding_beneficial = False
    if pg is not None and pg125 is not None:
        clip_drop = pg["magnitude_clip_rate"] - pg125["magnitude_clip_rate"]
        if clip_drop <= -0.30 and pg["success"] >= pg125["success"] - 1:
            authority_not_binding = True
        paired_125 = paired_gain_loss(pg["per_seed"], pg125["per_seed"])
        paired_15 = (
            paired_gain_loss(pg["per_seed"], pg15["per_seed"])
            if pg15 is not None
            else {"net": -999}
        )
        if paired_125["net"] <= 0 and paired_15["net"] <= 0:
            authority_not_binding = True
    if pg is not None and pg125 is not None:
        for scale_row, default_row in ((pg125, pg_default_125), (pg15, pg_default_15)):
            if scale_row is None:
                continue
            paired = paired_gain_loss(pg["per_seed"], scale_row["per_seed"])
            default_success = default_row["success"] if default_row is not None else 0
            if (
                paired["net"] >= 2
                and default_success >= 31
                and scale_row["magnitude_clip_rate"]
                <= pg125["magnitude_clip_rate"] + 0.05
                and scale_row["timeout_rate"] <= pg125["timeout_rate"] + 1e-12
            ):
                authority_binding_beneficial = True

    ppo_effective_survival_low = False
    pg_dir = direction_table.get("pg10_3/strict125/scale=1.25")
    c10_dir = direction_table.get("c10/strict125/scale=1.25")
    if pg_dir is not None:
        fin_survival = min(
            pg_dir["all"].get("survival_rudder_median", float("nan")),
            pg_dir["all"].get("survival_stern_median", float("nan")),
        )
        if (
            pg_dir["all"].get("fin_clip_step_fraction", 0.0) >= 0.5
            or fin_survival < 0.5
        ):
            ppo_effective_survival_low = True
    if pg_dir is not None and c10_dir is not None:
        pg_clip = pg_dir["all"].get("fin_clip_step_fraction", float("nan"))
        c10_clip = c10_dir["all"].get("fin_clip_step_fraction", float("nan"))
        pg_ppo = pg_dir["all"].get("ppo_l2_gate_mean", float("nan"))
        c10_ppo = c10_dir["all"].get("ppo_l2_gate_mean", float("nan"))
        if (
            min(
                c10_dir["all"].get("survival_rudder_median", 1.0),
                c10_dir["all"].get("survival_stern_median", 1.0),
            )
            <= min(
                pg_dir["all"].get("survival_rudder_median", 0.0),
                pg_dir["all"].get("survival_stern_median", 0.0),
            )
            and math.isfinite(c10_ppo)
            and math.isfinite(pg_ppo)
            and pg_ppo > EPS
            and (c10_ppo - pg_ppo) >= 0.25 * pg_ppo
            and abs(c10_clip - pg_clip) < 0.01
        ):
            ppo_effective_survival_low = True

    direction_positive = _has_direction_support(pg_dir) or _has_direction_support(c10_dir)

    terminal_shell_sensitive = False
    terminal_default = terminal_table.get("default")
    terminal_strict = terminal_table.get("strict125")
    if terminal_default is not None and terminal_strict is not None:
        terminal_shell_sensitive = bool(
            terminal_default["success_margin_median_m"] < 0.05
            and terminal_strict["success_margin_median_m"] < 0.05
            and abs(terminal_default["done_distance_difference_median_m"]) < 0.01
            and abs(terminal_strict["done_distance_difference_median_m"]) < 0.01
            and terminal_default.get("flip_shell_within_5cm", False)
            and terminal_strict.get("flip_shell_within_5cm", False)
        )

    if authority_not_binding or ppo_effective_survival_low:
        route = "ACTION_PARAMETERIZATION_CLIPPING"
    elif terminal_shell_sensitive:
        route = "TERMINAL_CAPTURE"
    elif not direction_positive:
        route = "DIRECTION_WRONG"
    elif authority_binding_beneficial:
        route = "ACTION_AUTHORITY_GAIN"
    else:
        route = "CREDIT_ASSIGNMENT_SAMPLE_INEFFICIENT"

    return {
        "AUTHORITY_NOT_BINDING": authority_not_binding,
        "AUTHORITY_BINDING_BENEFICIAL": authority_binding_beneficial,
        "PPO_EFFECTIVE_SURVIVAL_LOW": ppo_effective_survival_low,
        "DIRECTION_POSITIVE": direction_positive,
        "TERMINAL_SHELL_SENSITIVE": terminal_shell_sensitive,
        "route": route,
    }


def _format_float(value: Any) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:.1f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    return f"{number:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    decision = report["decision"]
    lines.append("# Goal93 结果：残差方向、动作权限与终端捕获")
    lines.append("")
    lines.append(f"**决策路线**：`{decision['route']}`")
    lines.append("")
    lines.append("## 判据")
    lines.append("")
    lines.append("| 判据 | 结果 |")
    lines.append("|---|---|")
    for key in (
        "AUTHORITY_NOT_BINDING",
        "AUTHORITY_BINDING_BENEFICIAL",
        "PPO_EFFECTIVE_SURVIVAL_LOW",
        "DIRECTION_POSITIVE",
        "TERMINAL_SHELL_SENSITIVE",
    ):
        lines.append(f"| `{key}` | `{decision[key]}` |")
    lines.append("")
    lines.append("## Scale sweep")
    lines.append("")
    for arm in ("pg10_3", "c10"):
        for scenario in ("default", "strict125"):
            table = report["scale_sweep"].get(arm, {}).get(scenario, {})
            if not table:
                continue
            lines.append(f"### {arm} / {scenario}")
            lines.append("")
            lines.append(
                "| scale | success | plane-miss | timeout | path 3D | pre-goal 3D | "
                "mag clip | rate clip | residual | paired vs 1.0 |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for scale, row in table.items():
                paired = row["paired_vs_scale1p00"]
                lines.append(
                    f"| {scale} | {row['success']}/{row['n']} | "
                    f"{_format_float(row['plane_miss_rate'])} | "
                    f"{_format_float(row['timeout_rate'])} | "
                    f"{_format_float(row['path_three_d_rmse_m'])} | "
                    f"{_format_float(row['pre_goal_3d_rmse_m'])} | "
                    f"{_format_float(row['magnitude_clip_rate'])} | "
                    f"{_format_float(row['rate_clip_rate'])} | "
                    f"{_format_float(row['residual_action_abs_mean'])} | "
                    f"+{paired['gain']}/-{paired['loss']} |"
                )
            lines.append("")
    lines.append("## 方向与动作权限")
    lines.append("")
    lines.append(
        "| arm / scenario / scale | success | cos median | success cos | failure cos | "
        "cos diff [95% CI] | survival total | survival rudder/stern/rpm | fin clip | "
        "applied L2 | effective L2 | PPO physical L2 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, row in sorted(report["direction"].items()):
        diff = row["cos_difference"]
        lines.append(
            f"| {key} | {row['success']}/{row['count']} | "
            f"{_format_float(row['all'].get('cos_gate_median'))} | "
            f"{_format_float(row['success_group'].get('cos_gate_median'))} | "
            f"{_format_float(row['failure_group'].get('cos_gate_median'))} | "
            f"{_format_float(diff.get('difference'))} "
            f"[{_format_float(diff.get('ci_low'))}, {_format_float(diff.get('ci_high'))}] | "
            f"{_format_float(row['all'].get('survival_gate_median'))} | "
            f"{_format_float(row['all'].get('survival_rudder_median'))}/"
            f"{_format_float(row['all'].get('survival_stern_median'))}/"
            f"{_format_float(row['all'].get('survival_rpm_median'))} | "
            f"{_format_float(row['all'].get('fin_clip_step_fraction'))} | "
            f"{_format_float(row['all'].get('applied_l2_gate_mean'))} | "
            f"{_format_float(row['all'].get('effective_l2_gate_mean'))} | "
            f"{_format_float(row['all'].get('ppo_physical_l2_gate_mean'))} |"
        )
    lines.append("")
    lines.append("### 物理权限分解")
    lines.append("")
    lines.append(
        "| arm / scenario / scale | PPO fin L2 | BC fin L2 | PPO fin / BC fin | "
        "PPO rpm | BC rpm |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, row in sorted(report["direction"].items()):
        ppo_fin = row["all"].get("ppo_fin_physical_l2_gate_mean", float("nan"))
        bc_fin = row["all"].get("bc_fin_physical_l2_gate_mean", float("nan"))
        ratio = ppo_fin / max(bc_fin, EPS) if math.isfinite(bc_fin) else float("nan")
        lines.append(
            f"| {key} | {_format_float(ppo_fin)} | {_format_float(bc_fin)} | "
            f"{_format_float(ratio)} | "
            f"{_format_float(row['all'].get('ppo_rpm_physical_abs_gate_mean'))} | "
            f"{_format_float(row['all'].get('bc_rpm_physical_abs_gate_mean'))} |"
        )
    lines.append("")
    lines.append("## 终端捕获")
    lines.append("")
    lines.append("| scenario | policy success | frozen success | flips | policy-frozen done Δ | success margin | flip shell margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for scenario, row in sorted(report["terminal"].items()):
        lines.append(
            f"| {scenario} | {row['policy_success']}/{row['paired_episodes']} | "
            f"{row['frozen_success']}/{row['paired_episodes']} | {row['flips']} | "
            f"{_format_float(row['done_distance_difference_median_m'])} | "
            f"{_format_float(row['success_margin_median_m'])} | "
            f"{_format_float(row['flip_shell_margin_max_m'])} |"
        )
    lines.append("")
    lines.append("## 解释边界")
    lines.append("")
    lines.append(
        "- `cos(r_PPO, r_BC)` 是方向代理，不是因果最优动作；BC 在同一契约下是强参照，"
        "但不是 oracle。"
    )
    lines.append(
        "- `survival` 衡量 requested residual 经过包线/速率限制后剩多少，"
        "不等于该增量一定对 terminal 有用。"
    )
    lines.append(
        "- 本审计不训练、不改 reward/terminal/success/goal radius、不打开 held-out。"
    )
    return "\n".join(lines) + "\n"


def build_report(
    scale_root: Path,
    trace_root: Path,
    *,
    bootstrap: int = 2000,
    bootstrap_seed: int = 93,
) -> dict[str, Any]:
    scale_data = load_scale_root(scale_root)
    trace_data = load_trace_root(trace_root)
    scale_sweep = {
        arm: {
            scenario: _scale_table(scale_data, arm, scenario)
            for scenario in ("default", "strict125")
        }
        for arm in ("pg10_3", "c10")
    }
    direction = _direction_table(
        trace_data, samples=bootstrap, seed=bootstrap_seed
    )
    terminal = _terminal_comparison(trace_data)
    decision = decide(scale_data, trace_data, direction, terminal)
    return {
        "scale_sweep": scale_sweep,
        "direction": direction,
        "terminal": terminal,
        "decision": decision,
        "inputs": {
            "scale_root": str(scale_root),
            "trace_root": str(trace_root),
        },
    }


def main() -> int:
    args = _parse_args()
    report = build_report(
        args.scale_root,
        args.trace_root,
        bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(render_markdown(report))
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
