#!/usr/bin/env python3
"""Summarize PC eval grids without reducing them to a single success number.

Layout: <root>/<tag>/<scenario>/{summary.json,episodes.json,trace.npz}
The zero tag is optional; when present, paired deltas are computed per episode.
Scenario directories are discovered automatically, so the same analyzer works
for nominal/strict, pressure ladders, route families, lengths and speeds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts/analyze_pc_goal145_capture_path_audit.py"
DEFAULT_SCENARIO_ORDER = ("default", "strict125")


def _helpers() -> Any:
    spec = importlib.util.spec_from_file_location("goal145_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _helpers()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="optional scenario filter; default discovers every <tag>/<scenario>/summary.json",
    )
    parser.add_argument("--zero-tag", default="zero")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=153)
    return parser.parse_args(argv)


def _discover_tags(root: Path) -> list[str]:
    tags = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (
            any(path.glob("*/summary.json")) or any(path.glob("*/*/summary.json"))
        ):
            tags.append(path.name)
    return tags


def _discover_scenarios(root: Path, tags: list[str]) -> list[str]:
    found = {
        summary.parent.name
        for tag in tags
        for summary in list((root / tag).glob("*/summary.json"))
        + list((root / tag).glob("*/*/summary.json"))
        if summary.parent.is_dir()
    }
    ordered = [name for name in DEFAULT_SCENARIO_ORDER if name in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def _distance(trace: np.lib.npyio.NpzFile) -> np.ndarray | None:
    for key in ("goal_distance_eta", "goal_distance_eta_m"):
        if key in trace.files:
            return np.asarray(trace[key], dtype=np.float64)
    return None


def _merge_arrays(run_dirs: list[Path], key: str) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    for run_dir in run_dirs:
        path = run_dir / "trace.npz"
        if not path.is_file():
            continue
        trace = np.load(path, allow_pickle=True)
        if key not in trace.files:
            continue
        chunks.append(np.asarray(trace[key], dtype=np.float64))
    return chunks


def _residual_bins(run_dirs: list[Path]) -> dict[str, Any]:
    distance_chunks: list[np.ndarray] = []
    residual_chunks: list[np.ndarray] = []
    gate_chunks: list[np.ndarray] = []
    for run_dir in run_dirs:
        path = run_dir / "trace.npz"
        if not path.is_file():
            continue
        trace = np.load(path, allow_pickle=True)
        distance = _distance(trace)
        if distance is None:
            continue
        residual_key = None
        for key in ("learned_residual_action", "residual_action", "residual"):
            if key in trace.files:
                residual_key = key
                break
        if residual_key is None:
            continue
        residual = np.asarray(trace[residual_key], dtype=np.float64)
        norm = np.linalg.norm(residual, axis=1)
        distance_chunks.append(distance)
        residual_chunks.append(norm)
        if "gate" in trace.files:
            gate_chunks.append(np.asarray(trace["gate"], dtype=np.float64))
    if not distance_chunks:
        return {}
    distance = np.concatenate(distance_chunks)
    norm = np.concatenate(residual_chunks)
    finite = np.isfinite(distance) & np.isfinite(norm)
    result: dict[str, Any] = {}
    for name, mask in (
        ("le2m", finite & (distance <= 2.0)),
        ("2to10m", finite & (distance > 2.0) & (distance <= 10.0)),
        ("gt10m", finite & (distance > 10.0)),
    ):
        values = norm[mask]
        result[name] = {
            "steps": int(values.size),
            "rms": None if values.size == 0 else float(np.sqrt(np.mean(values**2))),
            "mean": None if values.size == 0 else float(values.mean()),
        }
    if gate_chunks:
        gate = np.concatenate(gate_chunks)
        result["gate"] = {
            "mean": float(np.nanmean(gate)),
            "min": float(np.nanmin(gate)),
            "max": float(np.nanmax(gate)),
        }
    return result


def _reward_components(run_dirs: list[Path]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for run_dir in run_dirs:
        path = run_dir / "trace.npz"
        if not path.is_file():
            continue
        trace = np.load(path, allow_pickle=True)
        for key in trace.files:
            if not key.startswith("reward_"):
                continue
            values = np.asarray(trace[key], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                sums[key] = sums.get(key, 0.0) + float(values.sum())
                counts[key] = counts.get(key, 0) + int(values.size)
    return {key: sums[key] / counts[key] for key in sums}


def _trace_bundles(run_dirs: list[Path]) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = run_dir / "trace.npz"
        if not path.is_file():
            continue
        trace = np.load(path, allow_pickle=True)
        active = (
            np.asarray(trace["active"], dtype=bool)
            if "active" in trace.files
            else np.ones(len(trace["step"]), dtype=bool)
        )
        bundles.append({"dir": run_dir, "trace": trace, "active": active})
    return bundles


def _quantiles(values: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in qs}


def _route_health(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = {"3d": [], "vertical": [], "horizontal": []}
    distance_chunks: list[np.ndarray] = []
    for bundle in bundles:
        trace = bundle["trace"]
        active = bundle["active"]
        if not all(key in trace.files for key in ("path_error_3d", "path_vertical_error", "path_horizontal_error", "goal_distance_eta")):
            continue
        arrays["3d"].append(np.asarray(trace["path_error_3d"], dtype=np.float64)[active])
        arrays["vertical"].append(np.asarray(trace["path_vertical_error"], dtype=np.float64)[active])
        arrays["horizontal"].append(np.asarray(trace["path_horizontal_error"], dtype=np.float64)[active])
        distance_chunks.append(np.asarray(trace["goal_distance_eta"], dtype=np.float64)[active])
    if not arrays["3d"]:
        return {}
    data = {key: np.concatenate(chunks) for key, chunks in arrays.items()}
    distance = np.concatenate(distance_chunks)
    result: dict[str, Any] = {}
    for key, values in data.items():
        finite = values[np.isfinite(values)]
        result[key] = {
            "steps": int(finite.size),
            "rmse": float(np.sqrt(np.mean(finite**2))) if finite.size else None,
            **_quantiles(finite, (0.5, 0.9, 0.95, 0.99)),
            "max": float(np.max(finite)) if finite.size else None,
            "frac_gt_0p5": float(np.mean(finite > 0.5)) if finite.size else None,
            "frac_gt_1p0": float(np.mean(finite > 1.0)) if finite.size else None,
            "frac_gt_2p0": float(np.mean(finite > 2.0)) if finite.size else None,
        }
    result["by_distance"] = {}
    for name, mask in (
        ("le2m", distance <= 2.0),
        ("2to10m", (distance > 2.0) & (distance <= 10.0)),
        ("gt10m", distance > 10.0),
    ):
        block: dict[str, Any] = {"steps": int(mask.sum())}
        for key, values in data.items():
            selected = values[mask]
            selected = selected[np.isfinite(selected)]
            block[key] = {
                "rmse": float(np.sqrt(np.mean(selected**2))) if selected.size else None,
                "p95": float(np.quantile(selected, 0.95)) if selected.size else None,
            }
        result["by_distance"][name] = block
    return result


def _terminal_distribution(
    bundles: list[dict[str, Any]], *, path_length_m: float | None = None
) -> dict[str, Any]:
    final_distance: list[float] = []
    final_vertical: list[float] = []
    final_vertical_signed: list[float] = []
    final_horizontal: list[float] = []
    final_miss_n: list[float] = []
    final_miss_e: list[float] = []
    final_progress: list[float] = []
    entry_vertical: list[float] = []
    entry_horizontal: list[float] = []
    closest: list[float] = []
    steps_to_closest: list[float] = []
    near3_fraction: list[float] = []
    near_radius_fraction: dict[str, list[float]] = {
        "le1p25": [],
        "le1p5": [],
        "le2": [],
        "le3": [],
    }
    terminal_phase_fraction: list[float] = []
    for bundle in bundles:
        trace = bundle["trace"]
        active = bundle["active"]
        if "seed" not in trace.files or "goal_distance_eta" not in trace.files:
            continue
        seeds = np.unique(trace["seed"][active])
        for seed in seeds:
            mask = active & (trace["seed"] == seed)
            if not mask.any():
                continue
            index = np.flatnonzero(mask)
            distance = np.asarray(trace["goal_distance_eta"], dtype=np.float64)[index]
            vertical = np.asarray(trace["path_vertical_error"], dtype=np.float64)[index]
            horizontal = np.asarray(trace["path_horizontal_error"], dtype=np.float64)[index]
            finite = np.isfinite(distance) & np.isfinite(vertical) & np.isfinite(horizontal)
            if not bool(finite.any()):
                continue
            last = int(np.flatnonzero(finite)[-1])
            final_distance.append(float(distance[last]))
            final_vertical.append(float(abs(vertical[last])))
            final_vertical_signed.append(float(vertical[last]))
            final_horizontal.append(float(horizontal[last]))
            if "eta_ned" in trace.files and "goal_ned" in trace.files:
                eta = np.asarray(trace["eta_ned"], dtype=np.float64)[index, :3]
                goal = np.asarray(trace["goal_ned"], dtype=np.float64)[index, :3]
                miss = eta - goal
                final_miss_n.append(float(miss[last, 0]))
                final_miss_e.append(float(miss[last, 1]))
            if "path_progress" in trace.files:
                progress = np.asarray(trace["path_progress"], dtype=np.float64)[index]
                final_progress.append(float(progress[last]))
            entry = np.flatnonzero((distance <= 2.0) & finite)
            if entry.size:
                entry_vertical.append(float(abs(vertical[entry[0]])))
                entry_horizontal.append(float(horizontal[entry[0]]))
            closest_index = int(np.nanargmin(np.where(finite, distance, np.nan)))
            closest.append(float(distance[closest_index]))
            steps_to_closest.append(float(closest_index + 1))
            near3_fraction.append(float(np.mean((distance <= 3.0) & finite)))
            for name, radius in (
                ("le1p25", 1.25),
                ("le1p5", 1.5),
                ("le2", 2.0),
                ("le3", 3.0),
            ):
                near_radius_fraction[name].append(float(np.mean((distance <= radius) & finite)))
            if "guidance_terminal_phase" in trace.files:
                phase = np.asarray(trace["guidance_terminal_phase"], dtype=np.float64)[index]
                terminal_phase_fraction.append(float(np.mean(phase >= 0.5)))
    if not final_distance:
        return {}
    bins = {"le1p25": 0, "1p25to1p5": 0, "1p5to2": 0, "2to3": 0, "gt3": 0}
    for value in final_distance:
        if value <= 1.25:
            bins["le1p25"] += 1
        elif value <= 1.5:
            bins["1p25to1p5"] += 1
        elif value <= 2.0:
            bins["1p5to2"] += 1
        elif value <= 3.0:
            bins["2to3"] += 1
        else:
            bins["gt3"] += 1
    return {
        "n": len(final_distance),
        "final_goal_distance_m": {
            "mean": float(np.mean(final_distance)),
            **_quantiles(np.asarray(final_distance), (0.5, 0.9, 0.95)),
            "max": float(np.max(final_distance)),
        },
        "final_distance_bins": bins,
        "final_vertical_abs_m": {
            "mean": float(np.mean(final_vertical)),
            **_quantiles(np.asarray(final_vertical), (0.5, 0.9, 0.95)),
        },
        "final_vertical_signed_m": {
            "mean": float(np.mean(final_vertical_signed)),
            **_quantiles(np.asarray(final_vertical_signed), (0.5, 0.9)),
        },
        "final_horizontal_m": {
            "mean": float(np.mean(final_horizontal)),
            **_quantiles(np.asarray(final_horizontal), (0.5, 0.9, 0.95)),
        },
        "final_miss_n_m": {
            "mean": _mean(final_miss_n),
            "above_gt_0p1m": int(np.sum(np.asarray(final_miss_n) > 0.1)) if final_miss_n else 0,
            "within_0p1m": int(np.sum(np.abs(np.asarray(final_miss_n)) <= 0.1)) if final_miss_n else 0,
            "below_lt_minus_0p1m": int(np.sum(np.asarray(final_miss_n) < -0.1)) if final_miss_n else 0,
        },
        "final_miss_e_m": {
            "mean": _mean(final_miss_e),
            "above_gt_0p1m": int(np.sum(np.asarray(final_miss_e) > 0.1)) if final_miss_e else 0,
            "within_0p1m": int(np.sum(np.abs(np.asarray(final_miss_e)) <= 0.1)) if final_miss_e else 0,
            "below_lt_minus_0p1m": int(np.sum(np.asarray(final_miss_e) < -0.1)) if final_miss_e else 0,
        },
        "entry_2m_vertical_abs_m": {
            "mean": float(np.mean(entry_vertical)) if entry_vertical else None,
            **_quantiles(np.asarray(entry_vertical), (0.5, 0.9, 0.95)),
        },
        "entry_2m_horizontal_m": {
            "mean": float(np.mean(entry_horizontal)) if entry_horizontal else None,
            **_quantiles(np.asarray(entry_horizontal), (0.5, 0.9, 0.95)),
        },
        "closest_approach_m": {
            "mean": float(np.mean(closest)),
            **_quantiles(np.asarray(closest), (0.5, 0.9)),
        },
        "steps_to_closest": float(np.mean(steps_to_closest)),
        "near3_fraction": float(np.mean(near3_fraction)),
        "near_radius_fraction": {
            name: float(np.mean(values)) if values else None
            for name, values in near_radius_fraction.items()
        },
        "final_path_progress_m": _mean(final_progress),
        "final_path_progress_fraction": (
            None
            if path_length_m is None or path_length_m <= 0.0 or not final_progress
            else float(np.mean(final_progress) / path_length_m)
        ),
        "terminal_phase_fraction": (
            float(np.mean(terminal_phase_fraction)) if terminal_phase_fraction else None
        ),
    }


def _summary_terminal_kind_counts(run_dirs: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = payload.get("summary", payload)
        terminal_counts = summary.get("terminal_kind_counts")
        if not isinstance(terminal_counts, dict):
            continue
        for name, value in terminal_counts.items():
            counts[str(name)] = counts.get(str(name), 0) + int(value)
    return counts


def _direction_health(
    runs: list[dict[str, Any]], run_dirs: list[Path]
) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    groups: dict[str, list[dict[str, Any]]] = {"success": [], "failure": []}
    for run in runs:
        for seed, episode in run["episodes"].items():
            trace = run["trace"].get(seed)
            if trace is not None:
                groups["success" if H._success(episode) else "failure"].append(trace)
    kinds = _summary_terminal_kind_counts(run_dirs)
    if not kinds:
        for run in runs:
            for episode in run["episodes"].values():
                raw = episode.get("terminal_kind", "UNKNOWN")
                kind = f"KIND_{raw}" if isinstance(raw, int) else str(raw)
                kinds[kind] = kinds.get(kind, 0) + 1
    result: dict[str, Any] = {"terminal_kind_counts": kinds, "groups": {}}
    for name, rows in groups.items():
        result["groups"][name] = {
            "n": len(rows),
            "final_goal_distance_m": _mean([float(t.get("final_goal_distance_m", np.nan)) for t in rows]),
            "final_vertical_abs_m": _mean([float(t.get("final_vertical_abs_m", np.nan)) for t in rows]),
            "final_horizontal_m": _mean([float(t.get("final_horizontal_m", np.nan)) for t in rows]),
            "near_goal_vertical_abs_mean_m": _mean(
                [float(t.get("near_goal_vertical_abs_mean_m", np.nan)) for t in rows]
            ),
            "near_goal_horizontal_mean_m": _mean(
                [float(t.get("near_goal_horizontal_mean_m", np.nan)) for t in rows]
            ),
        }
    final_d = np.asarray(
        [
            float(trace.get("final_miss_d_m", np.nan))
            for run in runs
            for trace in run["trace"].values()
        ],
        dtype=np.float64,
    )
    final_d = final_d[np.isfinite(final_d)]
    result["final_vertical_sign"] = {
        "above_gt_0p1m": int(np.sum(final_d > 0.1)),
        "within_0p1m": int(np.sum(np.abs(final_d) <= 0.1)),
        "below_lt_minus_0p1m": int(np.sum(final_d < -0.1)),
    }
    return result


def _action_health(bundles: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    scale_clip: list[float] = []
    rate_clip: list[float] = []
    material_sat: list[float] = []
    policy_head_sat: list[float] = []
    effective_action_sat: list[float] = []
    action_abs_max: list[float] = []
    for bundle in bundles:
        trace = bundle["trace"]
        active = bundle["active"]
        if "scale_clipped" in trace.files:
            scale_clip.append(float(np.mean(np.asarray(trace["scale_clipped"], dtype=bool)[active])))
        if "rate_clipped" in trace.files:
            rate_clip.append(float(np.mean(np.asarray(trace["rate_clipped"], dtype=bool)[active])))
        if "material_saturated" in trace.files:
            material_sat.append(
                float(np.mean(np.asarray(trace["material_saturated"], dtype=bool)[active]))
            )
        elif "actuator_saturated" in trace.files:
            material_sat.append(
                float(np.mean(np.asarray(trace["actuator_saturated"], dtype=bool)[active]))
            )
        if "policy_action_raw" in trace.files:
            raw_action = np.asarray(trace["policy_action_raw"], dtype=np.float64)[active]
            policy_head_sat.append(float(np.mean(np.any(np.abs(raw_action) > 0.95, axis=1))))
        if "action" in trace.files:
            effective_action = np.asarray(trace["action"], dtype=np.float64)[active]
            effective_action_sat.append(
                float(np.mean(np.any(np.abs(effective_action) > 0.95, axis=1)))
            )
        if "actual_actuator" in trace.files:
            actuator = np.asarray(trace["actual_actuator"], dtype=np.float64)[active]
            action_abs_max.append(float(np.max(np.abs(actuator))) if actuator.size else float("nan"))
    return {
        "episode_action_saturation_fraction": _mean(
            [float(e.get("action_saturation_fraction", np.nan)) for e in episodes]
        ),
        "episode_policy_action_saturation_fraction": _mean(
            [float(e.get("policy_action_saturation_fraction", np.nan)) for e in episodes]
        ),
        "episode_material_saturation_fraction": _mean(
            [float(e.get("material_saturation_fraction", np.nan)) for e in episodes]
        ),
        "episode_scale_clipped_fraction": _mean(
            [float(e.get("scale_clipped_fraction", np.nan)) for e in episodes]
        ),
        "episode_rate_clipped_fraction": _mean(
            [float(e.get("rate_clipped_fraction", np.nan)) for e in episodes]
        ),
        "episode_action_abs_max": _mean(
            [float(e.get("action_abs_max", np.nan)) for e in episodes]
        ),
        "trace_scale_clipped_fraction": _mean(scale_clip),
        "trace_rate_clipped_fraction": _mean(rate_clip),
        "trace_material_saturation_fraction": _mean(material_sat),
        "trace_policy_head_saturation_fraction": _mean(policy_head_sat),
        "trace_effective_action_saturation_fraction": _mean(effective_action_sat),
        "trace_actual_actuator_abs_max": _mean(action_abs_max),
        "episode_residual_l2_ratio": _mean(
            [float(e.get("residual_l2_ratio", np.nan)) for e in episodes]
        ),
        "episode_residual_l2_ratio_gate_active": _mean(
            [float(e.get("residual_l2_ratio_gate_active", np.nan)) for e in episodes]
        ),
    }


def _resolve_environment_config(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_file():
        return path
    marker = "configs/"
    index = value.find(marker)
    if index >= 0:
        candidate = REPO_ROOT / value[index:]
        if candidate.is_file():
            return candidate
    return None


def _scenario_profile(run_dirs: list[Path]) -> dict[str, Any]:
    """Return auditable scenario axes, not just an opaque scenario directory name."""

    profile: dict[str, Any] = {
        "environment_config_sha256": None,
        "environment_config": None,
        "curriculum_stage": None,
        "curriculum_direct_commands_enabled": None,
        "learner_terminal_gate_mode": None,
        "nominal_effective_actuator_residual_scale": None,
        "goal_radius_m": None,
        "max_episode_steps": None,
        "route_family": None,
        "route_family_mix": None,
        "path_bank_size": None,
        "reference_length_m": None,
        "reference_speed_mps": None,
        "lateral_amplitude_m": None,
        "depth_amplitude_m": None,
        "start_jitter_xy_m": None,
        "start_jitter_z_m": None,
        "start_jitter_yaw_rad": None,
        "current_speed_range_mps": None,
        "current_type": None,
        "sensor_profile": None,
        "noise_scale": None,
        "noise_scale_range": None,
        "plant_randomization_enabled": None,
        "plant_scale_ranges": None,
        "curriculum_stage_mix": None,
        "curriculum_current_speed_override_mps": None,
        "curriculum_shear_scale_override": None,
    }
    hashes: set[str] = set()
    config_paths: set[str] = set()
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("environment_config_sha256"):
            hashes.add(str(payload["environment_config_sha256"]))
        for key in (
            "curriculum_stage",
            "curriculum_direct_commands_enabled",
            "learner_terminal_gate_mode",
            "nominal_effective_actuator_residual_scale",
        ):
            if profile.get(key) is None and payload.get(key) is not None:
                profile[key] = payload[key]
        env_value = payload.get("environment_config")
        env_path = _resolve_environment_config(env_value)
        if env_path is not None:
            config_paths.add(str(env_path))
            if profile["environment_config"] is None:
                profile["environment_config"] = str(env_path)
            if profile["goal_radius_m"] is None:
                config = json.loads(env_path.read_text(encoding="utf-8"))
                plant = config.get("plant") or {}
                reference = config.get("reference") or {}
                current = config.get("current") or {}
                sensors = config.get("sensors") or {}
                randomization = config.get("plant_randomization") or {}
                curriculum = config.get("curriculum") or {}
                profile.update(
                    {
                        "goal_radius_m": plant.get("goal_radius_m"),
                        "max_episode_steps": plant.get("max_episode_steps"),
                        "route_family": reference.get("path_family"),
                        "route_family_mix": reference.get("path_family_mix"),
                        "path_bank_size": reference.get("path_bank_size"),
                        "reference_length_m": reference.get("length_m"),
                        "reference_speed_mps": reference.get("speed_mps"),
                        "lateral_amplitude_m": reference.get("lateral_amplitude_m"),
                        "depth_amplitude_m": reference.get("depth_amplitude_m"),
                        "start_jitter_xy_m": reference.get("start_jitter_xy_m"),
                        "start_jitter_z_m": reference.get("start_jitter_z_m"),
                        "start_jitter_yaw_rad": reference.get("start_jitter_yaw_rad"),
                        "current_speed_range_mps": current.get("speed_mps"),
                        "current_type": current.get("type"),
                        "sensor_profile": sensors.get("profile"),
                        "noise_scale": sensors.get("noise_scale"),
                        "noise_scale_range": sensors.get("noise_scale_range"),
                        "plant_randomization_enabled": randomization.get("enabled"),
                        "plant_scale_ranges": {
                            key: randomization.get(key)
                            for key in (
                                "drag_scale",
                                "thrust_scale",
                                "fin_scale",
                                "added_mass_scale",
                                "actuator_time_scale",
                            )
                            if randomization.get(key) is not None
                        }
                        or None,
                        "curriculum_stage_mix": curriculum.get("stage_mix"),
                        "curriculum_current_speed_override_mps": curriculum.get(
                            "current_speed_range_override_mps"
                        ),
                        "curriculum_shear_scale_override": curriculum.get(
                            "shear_scale_override"
                        ),
                    }
                )
    if hashes:
        profile["environment_config_sha256"] = sorted(hashes)
    if config_paths:
        profile["environment_config"] = sorted(config_paths)[0] if len(config_paths) == 1 else sorted(config_paths)
    return profile


def _scenario_run_dirs(tag_dir: Path, scenario: str) -> list[Path]:
    """Support both tag/scenario and tag/<seed_tag>/scenario layouts."""

    direct = tag_dir / scenario
    if (direct / "summary.json").is_file():
        return [direct]
    return sorted(
        path.parent
        for path in tag_dir.glob(f"*/{scenario}/summary.json")
        if path.parent.is_dir()
    )


def _pool_paired(
    runs: list[dict[str, Any]],
    zero: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        paired = H._paired(run, zero, samples=samples, seed=seed + index)
        rows.extend(paired["rows"])
    result: dict[str, Any] = {
        "n_pairs": len(rows),
        "policy_success_count": sum(bool(row["policy_success"]) for row in rows),
        "zero_success_count": sum(bool(row["zero_success"]) for row in rows),
        "policy_plane_miss_count": sum(bool(row["policy_plane_miss"]) for row in rows),
        "zero_plane_miss_count": sum(bool(row["zero_plane_miss"]) for row in rows),
        "policy_timeout_count": sum(bool(row["policy_timeout"]) for row in rows),
        "zero_timeout_count": sum(bool(row["zero_timeout"]) for row in rows),
        "policy_safety_count": sum(bool(row["policy_safety"]) for row in rows),
        "zero_safety_count": sum(bool(row["zero_safety"]) for row in rows),
        "deltas": {},
        "rows": rows,
    }
    for key in (
        "path_vertical_rmse_m",
        "path_horizontal_rmse_m",
        "path_three_d_rmse_m",
        "path_vertical_mse_m2",
        "path_horizontal_mse_m2",
        "final_goal_distance_m",
        "final_vertical_abs_m",
        "final_horizontal_m",
        "near_goal_vertical_abs_mean_m",
        "near_goal_horizontal_mean_m",
        "terminal_phase_rate",
        "actuator_saturation_fraction",
        "material_saturation_fraction",
        "command_residual_abs_mean",
        "residual_abs_mean",
    ):
        values = [row[f"delta_{key}"] for row in rows if f"delta_{key}" in row]
        result["deltas"][key] = H._summarize_delta(
            key, values, samples=samples, seed=seed
        )
    result["deltas"]["success"] = H._summarize_delta(
        "success",
        [float(row["policy_success"]) - float(row["zero_success"]) for row in rows],
        samples=samples,
        seed=seed,
    )
    result["deltas"]["plane_miss"] = H._summarize_delta(
        "plane_miss",
        [float(row["policy_plane_miss"]) - float(row["zero_plane_miss"]) for row in rows],
        samples=samples,
        seed=seed,
    )
    return result


def _mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(array.mean())


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _tag_block(
    root: Path,
    tag: str,
    scenario: str,
    zero: dict[str, Any] | None,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    run_dirs = _scenario_run_dirs(root / tag, scenario)
    if not run_dirs:
        return {}
    runs = [H._load_run(path, interface="t3") for path in run_dirs]
    episodes = [episode for run in runs for episode in run["episodes"].values()]
    traces = [trace for run in runs for trace in run["trace"].values()]
    profile = _scenario_profile(run_dirs)
    block: dict[str, Any] = {
        "tag": tag,
        "scenario": scenario,
        "n_episodes": len(episodes),
        "n_checkpoints": len(runs),
        "success": sum(bool(H._success(ep)) for ep in episodes),
        "plane_miss": sum(bool(ep.get("goal_plane_miss", False)) for ep in episodes),
        "timeout": sum(bool(ep.get("timeout", False)) for ep in episodes),
        "safety": sum(bool(ep.get("safety", False)) for ep in episodes),
        "path_three_d_rmse_m": _mean(
            [float(t.get("path_three_d_rmse_m", np.nan)) for t in traces]
        ),
        "path_vertical_rmse_m": _mean(
            [float(t.get("path_vertical_rmse_m", np.nan)) for t in traces]
        ),
        "path_horizontal_rmse_m": _mean(
            [float(t.get("path_horizontal_rmse_m", np.nan)) for t in traces]
        ),
        "final_goal_distance_m": _mean(
            [float(t.get("final_goal_distance_m", np.nan)) for t in traces]
        ),
        "final_vertical_abs_m": _mean(
            [float(t.get("final_vertical_abs_m", np.nan)) for t in traces]
        ),
        "final_horizontal_m": _mean(
            [float(t.get("final_horizontal_m", np.nan)) for t in traces]
        ),
        "action_saturation_fraction": _mean(
            [float(t.get("actuator_saturation_fraction", np.nan)) for t in traces]
        ),
        "residual_bins": _residual_bins(run_dirs),
        "reward_components": _reward_components(run_dirs),
        "scenario_profile": profile,
    }
    bundles = _trace_bundles(run_dirs)
    block["route_health"] = _route_health(bundles)
    path_length = profile.get("reference_length_m")
    block["terminal_distribution"] = _terminal_distribution(
        bundles,
        path_length_m=float(path_length) if path_length is not None else None,
    )
    block["direction_health"] = _direction_health(runs, run_dirs)
    block["action_health"] = _action_health(bundles, episodes)
    if zero is not None:
        paired = _pool_paired(runs, zero, samples=samples, seed=seed)
        block["paired"] = paired
    return block


def _markdown(summary: dict[str, Any]) -> str:
    scenarios = summary["scenarios"]
    lines = [
        "# PC standing eval grid",
        "",
        f"Root: `{summary['root']}`",
        "",
        "All `path3d/vertical/horizontal` columns below are **episode-mean metrics**; the route-health section uses pooled active-step RMSE and is not directly interchangeable.",
        "",
        "## Scenario axes",
        "",
        "| scenario | goal r (m) | route | length (m) | speed (m/s) | current (m/s) | noise | plant | curriculum/stage | env SHA |",
        "|---|---:|---|---:|---:|---|---|---|---|---|",
    ]
    for scenario in scenarios:
        profile = next(
            (
                summary["blocks"][tag][scenario].get("scenario_profile", {})
                for tag in summary["tags"]
                if summary["blocks"].get(tag, {}).get(scenario)
            ),
            {},
        )
        route = profile.get("route_family")
        if route is None and profile.get("route_family_mix"):
            route = profile["route_family_mix"]
        noise = profile.get("noise_scale_range")
        if noise is None:
            noise = profile.get("noise_scale")
        plant = "on" if profile.get("plant_randomization_enabled") else "off"
        if profile.get("plant_scale_ranges"):
            plant = "±20%" if plant == "on" else plant
        curriculum = profile.get("curriculum_stage") or profile.get("curriculum_stage_mix") or "fixed"
        if profile.get("curriculum_direct_commands_enabled") is not None:
            curriculum = f"{curriculum}/direct={profile['curriculum_direct_commands_enabled']}"
        env_sha = profile.get("environment_config_sha256")
        if isinstance(env_sha, list):
            env_sha = ",".join(value[:8] for value in env_sha)
        elif isinstance(env_sha, str):
            env_sha = env_sha[:8]
        lines.append(
            "| {scenario} | {radius} | {route} | {length} | {speed} | {current} | {noise} | {plant} | {curriculum} | {sha} |".format(
                scenario=scenario,
                radius=_fmt(profile.get("goal_radius_m"), 2),
                route=route or "legacy_sine",
                length=_fmt(profile.get("reference_length_m"), 1),
                speed=_fmt(profile.get("reference_speed_mps"), 2),
                current=profile.get("current_speed_range_mps") or "NA",
                noise=noise if noise is not None else "default",
                plant=plant,
                curriculum=curriculum,
                sha=env_sha or "NA",
            )
        )
    lines.extend(
        [
            "",
            "| tag | scenario | success | plane miss | timeout | safety | path3d (ep mean) | vertical | horizontal | final d | res<=2 | res2-10 | res>10 | gate mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenarios:
        for tag in summary["tags"]:
            block = summary["blocks"].get(tag, {}).get(scenario)
            if not block:
                continue
            bins = block.get("residual_bins", {})
            gate = bins.get("gate", {})
            lines.append(
                "| {tag} | {scenario} | {success}/{n} | {miss} | {timeout} | {safety} | {path} | {vert} | {horiz} | {fd} | {r2} | {r10} | {rgt} | {gate} |".format(
                    tag=tag,
                    scenario=scenario,
                    success=block["success"],
                    n=block["n_episodes"],
                    miss=block["plane_miss"],
                    timeout=block["timeout"],
                    safety=block["safety"],
                    path=_fmt(block["path_three_d_rmse_m"]),
                    vert=_fmt(block["path_vertical_rmse_m"]),
                    horiz=_fmt(block["path_horizontal_rmse_m"]),
                    fd=_fmt(block["final_goal_distance_m"]),
                    r2=_fmt(bins.get("le2m", {}).get("rms")),
                    r10=_fmt(bins.get("2to10m", {}).get("rms")),
                    rgt=_fmt(bins.get("gt10m", {}).get("rms")),
                    gate=_fmt(gate.get("mean")),
                )
            )
    lines.extend(
        [
            "",
            "## Paired deltas vs zero",
            "",
            "| tag | scenario | Δsuccess | Δpath3d | Δvertical | Δhorizontal |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenarios:
        for tag in summary["tags"]:
            block = summary["blocks"].get(tag, {}).get(scenario)
            if not block or "paired" not in block:
                continue
            paired = block["paired"]
            deltas = paired.get("deltas", {})
            lines.append(
                "| {tag} | {scenario} | {ds} | {dp} | {dv} | {dh} |".format(
                    tag=tag,
                    scenario=scenario,
                    ds=_fmt(deltas.get("success", {}).get("mean")),
                    dp=_fmt(deltas.get("path_three_d_rmse_m", {}).get("mean")),
                    dv=_fmt(deltas.get("path_vertical_rmse_m", {}).get("mean")),
                    dh=_fmt(deltas.get("path_horizontal_rmse_m", {}).get("mean")),
                )
            )
    lines.extend(
        [
            "",
            "## Route health (pooled active steps)",
            "",
            "| tag | scenario | 3D RMSE | 3D p95 | 3D max | 3D >1m | RMSE <=2m | RMSE 2-10m | RMSE >10m | vertical p95 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenarios:
        for tag in summary["tags"]:
            block = summary["blocks"].get(tag, {}).get(scenario)
            if not block:
                continue
            route = block.get("route_health", {})
            three = route.get("3d", {})
            bins = route.get("by_distance", {})
            lines.append(
                "| {tag} | {scenario} | {rmse} | {p95} | {mx} | {gt1} | {le2} | {mid} | {gt10} | {vp95} |".format(
                    tag=tag,
                    scenario=scenario,
                    rmse=_fmt(three.get("rmse")),
                    p95=_fmt(three.get("p95")),
                    mx=_fmt(three.get("max")),
                    gt1=_fmt(None if three.get("frac_gt_1p0") is None else 100.0 * three["frac_gt_1p0"], 2),
                    le2=_fmt(bins.get("le2m", {}).get("3d", {}).get("rmse")),
                    mid=_fmt(bins.get("2to10m", {}).get("3d", {}).get("rmse")),
                    gt10=_fmt(bins.get("gt10m", {}).get("3d", {}).get("rmse")),
                    vp95=_fmt(route.get("vertical", {}).get("p95")),
                )
            )
    lines.extend(
        [
            "",
            "## Terminal-near distribution",
            "",
            "| tag | scenario | final d p50 | p90 | <=1.25 | 1.25-1.5 | 1.5-2 | 2-3 | >3 | final dv signed | entry2m vert p90 | entry2m horiz p90 | closest p50 | final progress | near<=r frac |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenarios:
        for tag in summary["tags"]:
            block = summary["blocks"].get(tag, {}).get(scenario)
            if not block:
                continue
            term = block.get("terminal_distribution", {})
            bins = term.get("final_distance_bins", {})
            lines.append(
                "| {tag} | {scenario} | {p50} | {p90} | {b1} | {b2} | {b3} | {b4} | {b5} | {dv} | {ev} | {eh} | {cl} | {progress} | {near} |".format(
                    tag=tag,
                    scenario=scenario,
                    p50=_fmt(term.get("final_goal_distance_m", {}).get("p50")),
                    p90=_fmt(term.get("final_goal_distance_m", {}).get("p90")),
                    b1=bins.get("le1p25", 0),
                    b2=bins.get("1p25to1p5", 0),
                    b3=bins.get("1p5to2", 0),
                    b4=bins.get("2to3", 0),
                    b5=bins.get("gt3", 0),
                    dv=_fmt(term.get("final_vertical_signed_m", {}).get("mean")),
                    ev=_fmt(term.get("entry_2m_vertical_abs_m", {}).get("p90")),
                    eh=_fmt(term.get("entry_2m_horizontal_m", {}).get("p90")),
                    cl=_fmt(term.get("closest_approach_m", {}).get("p50")),
                    progress=_fmt(term.get("final_path_progress_fraction"), 3),
                    near=_fmt(term.get("near_radius_fraction", {}).get("le2")),
                )
            )
    lines.extend(
        [
            "",
            "## Directional / failure decomposition",
            "",
            "| tag | scenario | GOAL | PLANE_MISS | TIME_LIMIT | SAFETY | success final d | failure final d | failure vert | failure horiz | above | within | below |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenarios:
        for tag in summary["tags"]:
            block = summary["blocks"].get(tag, {}).get(scenario)
            if not block:
                continue
            direction = block.get("direction_health", {})
            kinds = direction.get("terminal_kind_counts", {})
            groups = direction.get("groups", {})
            sign = direction.get("final_vertical_sign", {})
            safety_kinds = {
                "OUT_OF_BOUNDS",
                "NAN_OR_INF",
                "INVALID_STATE",
                "DEPTH_LIMIT",
                "SAFETY_ROLL_ANGLE",
                "SAFETY_ROLL_RATE",
            }
            lines.append(
                "| {tag} | {scenario} | {goal} | {miss} | {timeout} | {safety} | {sd} | {fd} | {fv} | {fh} | {above} | {within} | {below} |".format(
                    tag=tag,
                    scenario=scenario,
                    goal=kinds.get("GOAL", 0),
                    miss=kinds.get("GOAL_PLANE_MISS", 0),
                    timeout=kinds.get("TIME_LIMIT", 0),
                    safety=sum(value for name, value in kinds.items() if name in safety_kinds),
                    sd=_fmt(groups.get("success", {}).get("final_goal_distance_m")),
                    fd=_fmt(groups.get("failure", {}).get("final_goal_distance_m")),
                    fv=_fmt(groups.get("failure", {}).get("final_vertical_abs_m")),
                    fh=_fmt(groups.get("failure", {}).get("final_horizontal_m")),
                    above=sign.get("above_gt_0p1m", 0),
                    within=sign.get("within_0p1m", 0),
                    below=sign.get("below_lt_minus_0p1m", 0),
                )
            )
    lines.extend(
        [
            "",
            "## Action / residual health",
            "",
            "| tag | scenario | policy-head sat pre-scale | residual sat post-scale | scale clip | rate clip | actuator/material sat | residual L2 ratio | gate mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenarios:
        for tag in summary["tags"]:
            block = summary["blocks"].get(tag, {}).get(scenario)
            if not block:
                continue
            action = block.get("action_health", {})
            gate = block.get("residual_bins", {}).get("gate", {})
            lines.append(
                "| {tag} | {scenario} | {raw_sat} | {sat} | {scale} | {rate} | {mat} | {res} | {gate} |".format(
                    tag=tag,
                    scenario=scenario,
                    raw_sat=_fmt(
                        None
                        if action.get("trace_policy_head_saturation_fraction") is None
                        else 100.0 * action["trace_policy_head_saturation_fraction"],
                        2,
                    ),
                    sat=_fmt(
                        None
                        if action.get("trace_effective_action_saturation_fraction") is None
                        else 100.0 * action["trace_effective_action_saturation_fraction"],
                        2,
                    ),
                    scale=_fmt(None if action.get("trace_scale_clipped_fraction") is None else 100.0 * action["trace_scale_clipped_fraction"], 2),
                    rate=_fmt(None if action.get("trace_rate_clipped_fraction") is None else 100.0 * action["trace_rate_clipped_fraction"], 2),
                    mat=_fmt(None if action.get("trace_material_saturation_fraction") is None else 100.0 * action["trace_material_saturation_fraction"], 2),
                    res=_fmt(action.get("episode_residual_l2_ratio")),
                    gate=_fmt(gate.get("mean")),
                )
            )
    lines.extend(
        [
            "",
            "Note: `policy-head sat pre-scale` requires trace key `policy_action_raw` (new evaluator). "
            "`residual sat post-scale` uses the effective action after the global residual multiplier. "
            "`actuator/material sat` requires saturated masks in the trace; older artifacts show `NA` rather than a guessed value.",
        ]
    )
    lines.extend(["", "## Reward components (per-step mean)", ""])
    for scenario in scenarios:
        for tag in summary["tags"]:
            block = summary["blocks"].get(tag, {}).get(scenario)
            if not block:
                continue
            comps = block.get("reward_components", {})
            selected = {
                key: value
                for key, value in comps.items()
                if key
                in {
                    "reward_vertical_potential",
                    "reward_heave_potential",
                    "reward_path_vertical",
                    "reward_path_cross",
                    "reward_goal_plane_miss",
                    "reward_total",
                }
            }
            if selected:
                lines.append(
                    f"- `{tag}/{scenario}`: "
                    + ", ".join(f"{k.replace('reward_', '')}={v:.5f}" for k, v in sorted(selected.items()))
                )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve()
    tags = args.tags or [tag for tag in _discover_tags(root) if tag != args.zero_tag]
    scenarios = args.scenarios or _discover_scenarios(root, tags)
    if not scenarios:
        raise ValueError(f"no scenario summaries found under {root}")
    zero_runs: dict[str, dict[str, Any] | None] = {}
    for scenario in scenarios:
        zero_dirs = _scenario_run_dirs(root / args.zero_tag, scenario)
        zero_runs[scenario] = (
            H._load_run(zero_dirs[0], interface="t3") if zero_dirs else None
        )
    blocks: dict[str, dict[str, Any]] = {}
    for tag in tags:
        blocks[tag] = {}
        for scenario in scenarios:
            run_dirs = _scenario_run_dirs(root / tag, scenario)
            if not run_dirs:
                continue
            blocks[tag][scenario] = _tag_block(
                root,
                tag,
                scenario,
                zero_runs[scenario],
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
    summary = {
        "root": str(root),
        "tags": tags,
        "scenarios": scenarios,
        "blocks": blocks,
    }
    output_json = args.output_json or (root / "grid_summary.json")
    output_md = args.output_md or (root / "grid_summary.md")
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(summary) + "\n", encoding="utf-8")
    print(_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
