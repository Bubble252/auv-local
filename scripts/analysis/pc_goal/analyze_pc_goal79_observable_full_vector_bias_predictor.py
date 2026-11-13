#!/usr/bin/env python3
"""Read-only Goal79 observable full-vector terminal bias predictor audit."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal79_observable_full_vector_bias_predictor.v1"
DEADBAND_M = 0.25
RIDGE_ALPHA = 1.0
FALSE_REDUCTION_MIN = 0.50
TRUE_RETENTION_MIN = 0.95
MIN_TRAIN_SEEDS = 30
FORBIDDEN_FEATURE_TOKENS = (
    "eta",
    "p_nav_eta",
    "p_nav_ned",
    "eta_ned",
    "goal_ned",
)
FEATURE_NAMES = (
    "signed_goal_m",
    "lateral_goal_m",
    "vertical_goal_m",
    "goal_distance_m",
    "path_error_3d_m",
    "path_horizontal_error_m",
    "path_vertical_error_m",
    "base_command_l2",
    "residual_l2",
    "action_l2",
    "residual_gate",
    "steps",
    "time_s",
    "base_command_x",
    "base_command_y",
    "base_command_z",
    "residual_x",
    "residual_y",
    "residual_z",
    "action_x",
    "action_y",
    "action_z",
)


def _load_module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G78 = _load_module(
    "goal78_analysis_for_goal79", "analyze_pc_goal78_full_vector_bias_correction.py"
)
G77 = G78.G77
G76 = G78.G76
G73 = G78.G73
DiagnosticError = G78.DiagnosticError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("artifacts/goal71_command_rate_limit"),
    )
    parser.add_argument("--candidate-suffix", default="_dwell50_m025_cmdlimit")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/goal79_observable_bias_predictor/analysis/goal79.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/goal79_observable_bias_predictor/analysis/goal79.md"),
    )
    return parser.parse_args(argv)


def _goal_axes(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangent = np.asarray(tangent, dtype=np.float64)
    if tangent.shape != (3,):
        raise DiagnosticError("goal tangent must have shape (3,)")
    tangent_norm = float(np.linalg.norm(tangent))
    if not math.isfinite(tangent_norm) or tangent_norm <= 1e-12:
        raise DiagnosticError("goal tangent is zero or non-finite")
    t_axis = tangent / tangent_norm
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    v_axis = up - float(np.dot(up, t_axis)) * t_axis
    v_norm = float(np.linalg.norm(v_axis))
    if not math.isfinite(v_norm) or v_norm <= 1e-8:
        raise DiagnosticError("goal tangent is parallel to world vertical")
    v_axis = v_axis / v_norm
    h_axis = np.cross(t_axis, v_axis)
    h_norm = float(np.linalg.norm(h_axis))
    if not math.isfinite(h_norm) or h_norm <= 1e-12:
        raise DiagnosticError("failed to construct goal-frame lateral axis")
    return t_axis, h_axis / h_norm, v_axis


def _goal_frame_values(delta: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    offsets = G77._goal_frame_offsets(delta, tangent)
    return np.asarray(
        [offsets["signed_m"], offsets["lateral_m"], offsets["vertical_m"]],
        dtype=np.float64,
    )


def _goal_frame_vector(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,):
        raise DiagnosticError("goal-frame values must have shape (3,)")
    t_axis, h_axis, v_axis = _goal_axes(tangent)
    return values[0] * t_axis + values[1] * h_axis + values[2] * v_axis


def _last_trace_value(trace: Any, seed: int, name: str) -> Any:
    if name not in trace:
        raise DiagnosticError(f"trace missing field {name}")
    selected = np.flatnonzero(np.asarray(trace["seed"]) == seed)
    selected = selected[np.argsort(np.asarray(trace["step"])[selected])]
    values = np.asarray(trace[name])[selected]
    if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
        raise DiagnosticError(f"seed {seed}: non-finite trace field {name}")
    return values[-1]


def _feature_vector(
    *,
    p_nav: np.ndarray,
    goal: np.ndarray,
    tangent: np.ndarray,
    path_error_3d: float,
    path_horizontal_error: float,
    path_vertical_error: float,
    base_command: np.ndarray,
    residual: np.ndarray,
    action: np.ndarray,
    residual_gate: float,
    steps: int,
    time_s: float,
) -> np.ndarray:
    """Build the fixed observable input; this signature deliberately has no eta."""

    p_nav = np.asarray(p_nav, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    base_command = np.asarray(base_command, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if (
        p_nav.shape != (3,)
        or goal.shape != (3,)
        or base_command.shape != (3,)
        or residual.shape != (3,)
        or action.shape != (3,)
    ):
        raise DiagnosticError("observable 3-vectors must have shape (3,)")
    geometry = _goal_frame_values(p_nav - goal, tangent)
    values = np.asarray(
        [
            geometry[0],
            geometry[1],
            geometry[2],
            float(np.linalg.norm(p_nav - goal)),
            float(path_error_3d),
            float(path_horizontal_error),
            float(path_vertical_error),
            float(np.linalg.norm(base_command)),
            float(np.linalg.norm(residual)),
            float(np.linalg.norm(action)),
            float(residual_gate),
            float(steps),
            float(time_s),
            *base_command.tolist(),
            *residual.tolist(),
            *action.tolist(),
        ],
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),):
        raise DiagnosticError("feature vector length does not match whitelist")
    if not np.all(np.isfinite(values)):
        raise DiagnosticError("non-finite observable feature vector")
    return values


def _feature_audit() -> dict[str, Any]:
    lowered = [name.lower() for name in FEATURE_NAMES]
    forbidden = [
        name
        for name in lowered
        if any(token in name for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    parameters = list(inspect.signature(_feature_vector).parameters)
    forbidden_parameters = [
        name
        for name in parameters
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    duplicates = sorted({name for name in lowered if lowered.count(name) > 1})
    supported = bool(not forbidden and not forbidden_parameters and not duplicates)
    return {
        "supported": supported,
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "forbidden_feature_names": forbidden,
        "feature_function_parameters": parameters,
        "forbidden_parameters": forbidden_parameters,
        "duplicate_feature_names": duplicates,
        "uses_eta_or_oracle_label_as_input": False,
    }


def _predicted_position(p_nav: np.ndarray, predicted_error: np.ndarray) -> dict[str, float]:
    p_nav = np.asarray(p_nav, dtype=np.float64)
    predicted_error = np.asarray(predicted_error, dtype=np.float64)
    if p_nav.shape != (3,) or predicted_error.shape != (3,):
        raise DiagnosticError("p_nav and predicted_error must have shape (3,)")
    error_norm = float(np.linalg.norm(predicted_error))
    if not math.isfinite(error_norm):
        raise DiagnosticError("non-finite predicted error norm")
    correction = max(error_norm - DEADBAND_M, 0.0)
    if error_norm <= 1e-12:
        unit = np.zeros(3, dtype=np.float64)
    else:
        unit = predicted_error / error_norm
    corrected = p_nav - correction * unit
    if not np.all(np.isfinite(corrected)):
        raise DiagnosticError("non-finite predicted corrected position")
    return {
        "predicted_error_norm": error_norm,
        "predicted_correction": float(correction),
        "p_nav": corrected,
    }


def _task(task: tuple[str, str, str, str, str]) -> list[dict[str, Any]]:
    scenario, key, on_name, suffix, action_mode = task
    on_dir = Path(on_name)
    episodes = {int(row["seed"]): row for row in G73._read_episodes(on_dir)}
    summary = json.loads((on_dir / "summary.json").read_text(encoding="utf-8"))
    tight_radius, protocol_radius = G76.G75.G74._radii(on_dir, summary)
    records: list[dict[str, Any]] = []
    with np.load(on_dir / "trace.npz", allow_pickle=False) as trace:
        for seed in G73.DEV_SEEDS:
            episode = episodes[seed]
            rows = G73._seed_rows(trace, seed)
            eta = np.asarray(rows["eta_ned"][-1], dtype=np.float64)
            p_nav = np.asarray(rows["p_nav_ned"][-1], dtype=np.float64)
            goal = np.asarray(rows["goal_ned"][-1], dtype=np.float64)
            tangent = np.asarray(
                _last_trace_value(trace, seed, "goal_tangent_ned"), dtype=np.float64
            )
            d_p = float(np.linalg.norm(p_nav - goal))
            d_e = float(np.linalg.norm(eta - goal))
            if not math.isclose(
                d_p, float(episode["terminal_p_nav_distance_m"]), abs_tol=1e-6
            ):
                raise DiagnosticError(f"{on_dir} seed={seed}: p_nav distance mismatch")
            if not math.isclose(
                d_e, float(episode["terminal_eta_distance_m"]), abs_tol=1e-6
            ):
                raise DiagnosticError(f"{on_dir} seed={seed}: eta distance mismatch")
            success_none = bool(d_p <= tight_radius + 1e-12)
            mismatch = bool(success_none) != bool(episode["success"])
            tight_capture = bool(d_e <= tight_radius + 1e-12)
            protocol_capture = bool(d_e <= protocol_radius + 1e-12)
            label = _goal_frame_values(p_nav - eta, tangent)
            features = _feature_vector(
                p_nav=p_nav,
                goal=goal,
                tangent=tangent,
                path_error_3d=float(_last_trace_value(trace, seed, "path_error_3d")),
                path_horizontal_error=float(
                    _last_trace_value(trace, seed, "path_horizontal_error")
                ),
                path_vertical_error=float(
                    _last_trace_value(trace, seed, "path_vertical_error")
                ),
                base_command=np.asarray(
                    _last_trace_value(trace, seed, "base_command"), dtype=np.float64
                ),
                residual=np.asarray(
                    _last_trace_value(trace, seed, "residual"), dtype=np.float64
                ),
                action=np.asarray(
                    _last_trace_value(trace, seed, "action"), dtype=np.float64
                ),
                residual_gate=float(_last_trace_value(trace, seed, "gate")),
                steps=int(rows["step"][-1]),
                time_s=float(rows["time_s"][-1]),
            )
            records.append(
                {
                    "scenario": scenario,
                    "run_key": key,
                    "action_mode": action_mode,
                    "seed": int(seed),
                    "suffix": suffix,
                    "features": features,
                    "label": label,
                    "p_nav": p_nav,
                    "goal": goal,
                    "tangent": tangent,
                    "tight_radius_m": float(tight_radius),
                    "protocol_radius_m": float(protocol_radius),
                    "terminal_eta_distance_m": d_e,
                    "oracle_error_norm": float(np.linalg.norm(p_nav - eta)),
                    "success_A0": success_none,
                    "tight_capture": tight_capture,
                    "protocol_capture": protocol_capture,
                    "tight_true_A0": bool(success_none and tight_capture),
                    "tight_false_A0": bool(success_none and not tight_capture),
                    "mismatch": mismatch,
                }
            )
    return records


def _tasks(args: argparse.Namespace) -> list[tuple[str, str, str, str, str]]:
    tasks: list[tuple[str, str, str, str, str]] = []
    for scenario in G73.SCENARIOS:
        for checkpoint in G73.POLICY_SEEDS:
            key = str(checkpoint)
            tasks.append(
                (
                    scenario,
                    key,
                    str(args.candidate_root / G73._run_name(scenario, key, args.candidate_suffix)),
                    args.candidate_suffix,
                    "policy",
                )
            )
        tasks.append(
            (
                scenario,
                "zero",
                str(args.candidate_root / G73._run_name(scenario, "zero", args.candidate_suffix)),
                args.candidate_suffix,
                "zero-residual",
            )
        )
    return tasks


def _ridge_fit(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or labels.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise DiagnosticError("ridge features/labels shape mismatch")
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    label_mean = np.mean(labels, axis=0)
    centered = labels - label_mean
    gram = standardized.T @ standardized + RIDGE_ALPHA * np.eye(features.shape[1])
    weights = np.linalg.solve(gram, standardized.T @ centered)
    return weights, label_mean, mean, scale


def _ridge_predict(
    features: np.ndarray,
    weights: np.ndarray,
    label_mean: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (np.asarray(features, dtype=np.float64) - mean) / scale
    return standardized @ weights + label_mean


def _seed_grouped_splits(records: list[dict[str, Any]]) -> list[tuple[np.ndarray, np.ndarray]]:
    seeds = np.asarray([int(record["seed"]) for record in records], dtype=np.int64)
    unique_seeds = sorted(int(seed) for seed in np.unique(seeds))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for held_seed in unique_seeds:
        train = np.flatnonzero(seeds != held_seed)
        test = np.flatnonzero(seeds == held_seed)
        if train.size == 0 or test.size == 0:
            raise DiagnosticError(f"empty seed-grouped fold for seed={held_seed}")
        if np.intersect1d(seeds[train], seeds[test]).size:
            raise DiagnosticError(f"seed leakage in fold seed={held_seed}")
        splits.append((train, test))
    return splits


def _seed_grouped_oof(records: list[dict[str, Any]]) -> np.ndarray:
    features = np.stack([record["features"] for record in records]).astype(np.float64)
    labels = np.stack([record["label"] for record in records]).astype(np.float64)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(labels)):
        raise DiagnosticError("non-finite predictor input")
    predictions = np.full_like(labels, np.nan)
    for train, test in _seed_grouped_splits(records):
        if np.unique([records[index]["seed"] for index in train]).size < MIN_TRAIN_SEEDS:
            raise DiagnosticError("too few training seeds in LOOCV fold")
        weights, label_mean, mean, scale = _ridge_fit(features[train], labels[train])
        predictions[test] = _ridge_predict(features[test], weights, label_mean, mean, scale)
    if not np.all(np.isfinite(predictions)):
        raise DiagnosticError("non-finite out-of-fold prediction")
    return predictions


def _count(records: list[dict[str, Any]], field: str) -> int:
    return int(sum(bool(record[field]) for record in records))


def _median(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None and math.isfinite(float(record[field]))
    ]
    return None if not values else float(np.median(values))


def _rmse(labels: np.ndarray, predictions: np.ndarray, column: int) -> float:
    residual = labels[:, column] - predictions[:, column]
    return float(np.sqrt(np.mean(residual**2)))


def _r2(labels: np.ndarray, predictions: np.ndarray, column: int) -> float | None:
    residual = labels[:, column] - predictions[:, column]
    denominator = float(np.sum((labels[:, column] - np.mean(labels[:, column])) ** 2))
    if denominator <= 1e-12:
        return None
    return float(1.0 - np.sum(residual**2) / denominator)


def _arm_stats(records: list[dict[str, Any]], base: dict[str, int]) -> dict[str, Any]:
    tight_true = _count(records, "tight_true_pred")
    tight_false = _count(records, "tight_false_pred")
    false_reduction = (
        None if base["tight_false"] <= 0 else (base["tight_false"] - tight_false) / base["tight_false"]
    )
    true_retention = (
        None if base["tight_true"] <= 0 else tight_true / base["tight_true"]
    )
    return {
        "success": _count(records, "success_pred"),
        "tight_true": tight_true,
        "tight_false": tight_false,
        "protocol_true": _count(records, "protocol_true_pred"),
        "tight_false_reduction_fraction": false_reduction,
        "tight_true_retention": true_retention,
    }


def _attach_predictions(records: list[dict[str, Any]], predictions: np.ndarray) -> None:
    for record, predicted_components in zip(records, predictions, strict=True):
        predicted_error = _goal_frame_vector(predicted_components, record["tangent"])
        correction = _predicted_position(record["p_nav"], predicted_error)
        p_pred = np.asarray(correction["p_nav"], dtype=np.float64)
        goal = np.asarray(record["goal"], dtype=np.float64)
        success = bool(
            np.linalg.norm(p_pred - goal) <= float(record["tight_radius_m"]) + 1e-12
        )
        record["predicted_error_norm"] = float(correction["predicted_error_norm"])
        record["predicted_correction"] = float(correction["predicted_correction"])
        record["success_pred"] = success
        record["tight_true_pred"] = bool(success and record["tight_capture"])
        record["tight_false_pred"] = bool(success and not record["tight_capture"])
        record["protocol_true_pred"] = bool(success and record["protocol_capture"])


def _aggregate(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    labels = np.stack([record["label"] for record in records]).astype(np.float64)
    predictions = np.stack([record["predicted_label"] for record in records]).astype(np.float64)
    base = {
        "success": _count(records, "success_A0"),
        "tight_true": _count(records, "tight_true_A0"),
        "tight_false": _count(records, "tight_false_A0"),
        "protocol_true": int(
            sum(record["success_A0"] and record["protocol_capture"] for record in records)
        ),
    }
    arm = _arm_stats(records, base)
    h79a_supported = bool(
        arm["tight_false_reduction_fraction"] is not None
        and arm["tight_true_retention"] is not None
        and arm["tight_false_reduction_fraction"] >= FALSE_REDUCTION_MIN
        and arm["tight_true_retention"] >= TRUE_RETENTION_MIN
    )
    return {
        "scenario": scenario,
        "n": len(records),
        "base": base,
        "prediction_arm": arm,
        "gates": {"H79-A": {"supported": h79a_supported}},
        "mismatch": _count(records, "mismatch"),
        "prediction_finite": int(
            sum(
                math.isfinite(float(record["predicted_error_norm"]))
                and math.isfinite(float(record["predicted_correction"]))
                for record in records
            )
        ),
        "prediction_metrics": {
            "rmse_signed_m": _rmse(labels, predictions, 0),
            "rmse_lateral_m": _rmse(labels, predictions, 1),
            "rmse_vertical_m": _rmse(labels, predictions, 2),
            "r2_signed": _r2(labels, predictions, 0),
            "r2_lateral": _r2(labels, predictions, 1),
            "r2_vertical": _r2(labels, predictions, 2),
            "predicted_error_norm_median_m": _median(records, "predicted_error_norm"),
            "oracle_error_norm_median_m": _median(records, "oracle_error_norm"),
            "predicted_correction_median_m": _median(records, "predicted_correction"),
        },
        "by_checkpoint": {
            key: {
                "n": sum(record["run_key"] == key for record in records),
                "tight_false_A0": sum(
                    record["run_key"] == key and record["tight_false_A0"] for record in records
                ),
                "tight_false_pred": sum(
                    record["run_key"] == key and record["tight_false_pred"] for record in records
                ),
            }
            for key in ("240", "752", "880", "1008", "zero")
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal79：observable full-vector terminal bias predictor",
        "",
        f"状态：`{payload['overall']['status']}`",
        "",
        "## 预测 correction 结果",
        "",
        "| scenario | base success/true/false | pred success/true/false | true 保留率 | false 降幅 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario, result in payload["scenarios"].items():
        base = result["base"]
        arm = result["prediction_arm"]
        lines.append(
            f"| {scenario} | {base['success']}/{base['tight_true']}/{base['tight_false']} | "
            f"{arm['success']}/{arm['tight_true']}/{arm['tight_false']} | "
            f"{_fmt(arm['tight_true_retention'])} | "
            f"{_fmt(arm['tight_false_reduction_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## 预测误差",
            "",
            "| scenario | RMSE signed/lateral/vertical (m) | R2 signed/lateral/vertical | pred norm 中位 | oracle norm 中位 | correction 中位 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, result in payload["scenarios"].items():
        metrics = result["prediction_metrics"]
        lines.append(
            f"| {scenario} | {metrics['rmse_signed_m']:.4f}/"
            f"{metrics['rmse_lateral_m']:.4f}/{metrics['rmse_vertical_m']:.4f} | "
            f"{_fmt(metrics['r2_signed'])}/{_fmt(metrics['r2_lateral'])}/"
            f"{_fmt(metrics['r2_vertical'])} | "
            f"{_fmt(metrics['predicted_error_norm_median_m'])} | "
            f"{_fmt(metrics['oracle_error_norm_median_m'])} | "
            f"{_fmt(metrics['predicted_correction_median_m'])} |"
        )
    lines.extend(["", "## Gate 判读", ""])
    for scenario, result in payload["scenarios"].items():
        lines.append(
            f"- `{scenario}`：H79-A={_fmt(result['gates']['H79-A']['supported'])}，"
            f"mismatch={result['mismatch']}/{result['n']}，"
            f"finite={result['prediction_finite']}/{result['n']}"
        )
    audit = payload["feature_audit"]
    lines.extend(
        [
            "",
            "## 特征审计",
            "",
            f"- 特征数：`{audit['n_features']}`",
            f"- forbidden feature names：`{audit['forbidden_feature_names']}`",
            f"- forbidden parameters：`{audit['forbidden_parameters']}`",
            f"- H79-B supported：`{_fmt(audit['supported'])}`",
            "",
            "## 隔离",
            "",
            f"- held-out opened: `{_fmt(payload['overall']['heldout_opened'])}`",
            f"- training started: `{_fmt(payload['overall']['training_started'])}`",
            f"- new rollouts: `{_fmt(payload['overall']['new_rollouts'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    audit = _feature_audit()
    try:
        tasks = _tasks(args)
        if args.jobs == 1:
            task_results = [_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
                task_results = list(pool.map(_task, tasks))
        records = [record for rows in task_results for record in rows]
        for record in records:
            record["predicted_label"] = None
        scenario_records = {
            scenario: [record for record in records if record["scenario"] == scenario]
            for scenario in G73.SCENARIOS
        }
        for _scenario, rows in scenario_records.items():
            predictions = _seed_grouped_oof(rows)
            for record, predicted in zip(rows, predictions, strict=True):
                record["predicted_label"] = predicted
        for rows in scenario_records.values():
            _attach_predictions(rows, np.stack([row["predicted_label"] for row in rows]))
        scenarios = {
            scenario: _aggregate(rows, scenario)
            for scenario, rows in scenario_records.items()
        }
        h79a = all(result["gates"]["H79-A"]["supported"] for result in scenarios.values())
        mismatch = sum(result["mismatch"] for result in scenarios.values())
        nonfinite = sum(
            result["n"] - result["prediction_finite"] for result in scenarios.values()
        )
        h79b = bool(audit["supported"])
        if mismatch > 0 or nonfinite > 0 or not h79b:
            status = "INVALID"
        elif h79a:
            status = "OBSERVABLE_FULL_VECTOR_PREDICTOR_SUPPORTED"
        else:
            status = "OBSERVABLE_FULL_VECTOR_PREDICTOR_NOT_SUPPORTED"
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": scenarios,
            "feature_audit": audit,
            "overall": {
                "status": status,
                "H79-A supported": h79a,
                "H79-B supported": h79b,
                "mismatch": mismatch,
                "prediction_nonfinite": nonfinite,
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": False,
                "seed_grouped_loocv": True,
                "ridge_alpha": RIDGE_ALPHA,
                "deadband_m": DEADBAND_M,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        args.markdown.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
        return 0 if status != "INVALID" else 2
    except (DiagnosticError, FileNotFoundError, KeyError, ValueError) as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA_VERSION,
            "scenarios": {},
            "feature_audit": audit,
            "overall": {
                "status": "INVALID",
                "error": str(error),
                "heldout_opened": False,
                "training_started": False,
                "new_rollouts": False,
            },
        }
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
