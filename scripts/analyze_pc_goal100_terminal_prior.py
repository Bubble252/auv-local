#!/usr/bin/env python3
"""Audit Goal100's terminal-prior-preservation screen.

The script is read-only: it consumes Goal95/Goal98 reference summaries and
Goal100 B20/A3/A2/A125 summaries/traces. It never loads a policy, steps an
environment, trains, changes success semantics, or reads held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal100_terminal_prior.v1"
ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")
PROFILES = {
    "B20": {
        "action_mode": "frozen-base",
        "attenuation_full_m": None,
        "attenuation_enter_m": None,
    },
    "A3": {
        "action_mode": "attenuated-residual",
        "attenuation_full_m": 3.0,
        "attenuation_enter_m": 5.0,
    },
    "A2": {
        "action_mode": "attenuated-residual",
        "attenuation_full_m": 2.0,
        "attenuation_enter_m": 4.0,
    },
    "A125": {
        "action_mode": "attenuated-residual",
        "attenuation_full_m": 1.25,
        "attenuation_enter_m": 3.25,
    },
}
FROZEN_SUCCESS = {"default": 31, "strict125": 20}
EXPECTED_REFERENCE_SUCCESS = {
    ("B15", "P_D3", "default"): 31,
    ("B15", "P_BC15", "default"): 31,
    ("B15", "P_D3", "strict125"): 20,
    ("B15", "P_BC15", "strict125"): 20,
    ("P0_15", "P_D3", "default"): 30,
    ("P0_15", "P_BC15", "default"): 30,
    ("P0_15", "P_D3", "strict125"): 19,
    ("P0_15", "P_BC15", "strict125"): 19,
    ("P0_20", "P_D3", "default"): 31,
    ("P0_20", "P_BC15", "default"): 32,
    ("P0_20", "P_D3", "strict125"): 18,
    ("P0_20", "P_BC15", "strict125"): 18,
}
EXPECTED_CHECKPOINT_SHA = {
    "P_D3": "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616",
    "P_BC15": "1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43",
}


class AuditError(ValueError):
    """Raised when a run is incomplete or violates the frozen protocol."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal100_terminal_prior_20261111"),
    )
    parser.add_argument(
        "--goal95-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40"),
    )
    parser.add_argument(
        "--goal98-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal98_residual_scale_grid_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal100_terminal_prior_20261111/phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal100_terminal_prior_20261111/phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _count_success(episodes: list[dict[str, Any]], key: str = "success") -> int:
    return sum(1 for row in episodes if bool(row.get(key, False)))


def _episode_outcomes(
    path: Path, episodes: list[dict[str, Any]], *, expected_seeds: set[int]
) -> dict[str, Any]:
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise AuditError(f"{path}: expected 40 dev episodes")
    by_seed: dict[int, bool] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{path}: malformed episode row")
        by_seed[int(row["seed"])] = bool(row["success"])
    if set(by_seed) != expected_seeds:
        raise AuditError(f"{path}: episodes do not cover dev seeds 20-59")
    return {
        "success": _count_success(episodes),
        "success_by_seed": by_seed,
        "success_pnav": _count_success(episodes, "success_pnav_tight"),
        "success_eta_ever": _count_success(episodes, "success_eta_tight_ever"),
        "success_eta_done": _count_success(episodes, "success_eta_tight_at_done"),
        "timeout": sum(1 for row in episodes if bool(row.get("timeout", False))),
        "safety": sum(1 for row in episodes if bool(row.get("safety", False))),
    }


def _validate_common(
    path: Path, summary: dict[str, Any], *, arm: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_sha = EXPECTED_CHECKPOINT_SHA[arm]
    actual_sha = summary.get("checkpoint_sha256")
    if actual_sha != expected_sha:
        raise AuditError(f"{path}: checkpoint SHA {actual_sha!r} != {expected_sha}")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        raise AuditError(f"{path}: missing episode list")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise AuditError(f"{path}: missing metrics object")
    return episodes, metrics


def _record_from_summary(
    path: Path,
    *,
    arm: str,
    scenario: str,
    profile: str,
    action_mode: str,
    expected_seeds: set[int],
) -> dict[str, Any]:
    summary = _read_json(path)
    if summary.get("action_mode") != action_mode:
        raise AuditError(
            f"{path}: action_mode {summary.get('action_mode')!r} != {action_mode!r}"
        )
    episodes, metrics = _validate_common(path, summary, arm=arm)
    record = _episode_outcomes(path, episodes, expected_seeds=expected_seeds)
    record.update(
        {
            "run": path.parent.name,
            "arm": arm,
            "scenario": scenario,
            "profile": profile,
            "path_three_d_rmse_m": float(metrics.get("path_three_d_rmse_m", math.nan)),
            "goal_plane_miss_rate": float(metrics.get("goal_plane_miss_rate", math.nan)),
            "residual_magnitude_clip_rate": float(
                metrics.get("actuator_residual_magnitude_clip_rate", math.nan)
            ),
            "residual_rate_clip_rate": float(
                metrics.get("actuator_residual_rate_clip_rate", math.nan)
            ),
            "ppo_residual_attenuation_steps": int(
                metrics.get("ppo_residual_attenuation_steps", 0)
            ),
            "ppo_residual_attenuation_mean": metrics.get(
                "ppo_residual_attenuation_mean"
            ),
            "ppo_residual_attenuation_zero_fraction": metrics.get(
                "ppo_residual_attenuation_zero_fraction"
            ),
        }
    )
    return record


def _load_references(
    goal95_root: Path,
    goal98_root: Path,
    *,
    expected_seeds: set[int],
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for scenario in SCENARIOS:
            for profile, mode in (("P0_15", "policy"), ("B15", "frozen-base")):
                path = (
                    goal95_root
                    / f"{arm}_s86201_{scenario}_{mode}"
                    / "summary.json"
                )
                if not path.is_file():
                    raise AuditError(f"{path}: missing Goal95 reference summary")
                record = _record_from_summary(
                    path,
                    arm=arm,
                    scenario=scenario,
                    profile=profile,
                    action_mode=mode,
                    expected_seeds=expected_seeds,
                )
                expected = EXPECTED_REFERENCE_SUCCESS[(profile, arm, scenario)]
                if record["success"] != expected:
                    raise AuditError(
                        f"{path}: success {record['success']} != frozen reference {expected}"
                    )
                references[f"{profile}_{arm}_{scenario}"] = record
            path = goal98_root / f"{arm}_{scenario}_s200" / "summary.json"
            if not path.is_file():
                raise AuditError(f"{path}: missing Goal98 scale-2.00 summary")
            record = _record_from_summary(
                path,
                arm=arm,
                scenario=scenario,
                profile="P0_20",
                action_mode="policy",
                expected_seeds=expected_seeds,
            )
            expected = EXPECTED_REFERENCE_SUCCESS[("P0_20", arm, scenario)]
            if record["success"] != expected:
                raise AuditError(
                    f"{path}: success {record['success']} != frozen reference {expected}"
                )
            references[f"P0_20_{arm}_{scenario}"] = record
    return references


def _smooth_terminal_gate(
    distance_m: np.ndarray, *, enter_m: float, full_m: float
) -> np.ndarray:
    progress = np.clip((enter_m - distance_m) / (enter_m - full_m), 0.0, 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def _validate_attenuation_trace(path: Path, spec: dict[str, float | None]) -> None:
    full_m = spec["attenuation_full_m"]
    enter_m = spec["attenuation_enter_m"]
    if full_m is None or enter_m is None:
        return
    if not path.is_file():
        raise AuditError(f"{path}: missing attenuation trace")
    required = (
        "ppo_residual_attenuation",
        "ppo_residual_attenuation_distance_m",
        "goal_distance_eta_m",
        "episode_seed",
        "step",
    )
    with np.load(path) as data:
        missing = [name for name in required if name not in data]
        if missing:
            raise AuditError(f"{path}: missing trace fields {missing}")
        attenuation = np.asarray(data["ppo_residual_attenuation"], dtype=np.float64)
        distance = np.asarray(
            data["ppo_residual_attenuation_distance_m"], dtype=np.float64
        )
        if attenuation.shape != distance.shape or attenuation.size == 0:
            raise AuditError(f"{path}: attenuation/distance shape mismatch")
        if not np.all(np.isfinite(attenuation)) or not np.all(np.isfinite(distance)):
            raise AuditError(f"{path}: non-finite attenuation trace")
        expected = 1.0 - _smooth_terminal_gate(
            distance, enter_m=float(enter_m), full_m=float(full_m)
        )
        if not np.allclose(attenuation, expected, rtol=0.0, atol=1e-5):
            raise AuditError(f"{path}: attenuation does not match profile boundary")


def _load_goal100(
    root: Path,
    *,
    expected_seeds: set[int],
) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        raise AuditError(f"{root}: missing Goal100 root")
    records: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for scenario in SCENARIOS:
            for profile, spec in PROFILES.items():
                run_dir = root / f"{arm}_{scenario}_{profile}"
                path = run_dir / "summary.json"
                if not path.is_file():
                    raise AuditError(f"{path}: missing summary")
                record = _record_from_summary(
                    path,
                    arm=arm,
                    scenario=scenario,
                    profile=profile,
                    action_mode=str(spec["action_mode"]),
                    expected_seeds=expected_seeds,
                )
                summary = _read_json(path)
                recorded_scale = summary.get("actuator_residual_scale_override")
                if recorded_scale is None or abs(float(recorded_scale) - 2.0) > 1e-12:
                    raise AuditError(f"{path}: scale override {recorded_scale!r} != 2.0")
                enabled = summary.get("ppo_residual_attenuation_enabled")
                if enabled is not (profile != "B20"):
                    raise AuditError(f"{path}: attenuation enabled flag mismatch")
                if profile != "B20":
                    for name, expected in (
                        ("ppo_residual_attenuation_full_m", spec["attenuation_full_m"]),
                        ("ppo_residual_attenuation_enter_m", spec["attenuation_enter_m"]),
                    ):
                        recorded = summary.get(name)
                        if recorded is None or abs(float(recorded) - float(expected)) > 1e-12:
                            raise AuditError(f"{path}: {name} {recorded!r} != {expected}")
                _validate_attenuation_trace(run_dir / "trace.npz", spec)
                records[f"{profile}_{arm}_{scenario}"] = record
    return records


def _transition(
    baseline: dict[int, bool], other: dict[int, bool]
) -> dict[str, Any]:
    gained = sorted(seed for seed, value in other.items() if value and not baseline[seed])
    lost = sorted(seed for seed in baseline if baseline[seed] and not other[seed])
    return {"gained": gained, "lost": lost}


def _profile_summary(
    records: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    profile: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"profile": profile, "groups": {}}
    ratios: list[float] = []
    for arm in ARMS:
        for scenario in SCENARIOS:
            key = f"{profile}_{arm}_{scenario}"
            record = records[key]
            baseline = references[f"B15_{arm}_{scenario}"]
            ratio = record["path_three_d_rmse_m"] / baseline["path_three_d_rmse_m"]
            ratios.append(ratio)
            result["groups"][f"{arm}_{scenario}"] = {
                **record,
                "path_ratio_vs_B15": ratio,
                "transition_vs_B15": _transition(
                    baseline["success_by_seed"], record["success_by_seed"]
                ),
            }
    result["mean_path_ratio_vs_B15"] = float(np.mean(ratios))
    return result


def _profile_qualifies(profile: dict[str, Any], *, require_attenuation: bool) -> bool:
    for arm in ARMS:
        for scenario in SCENARIOS:
            group = profile["groups"][f"{arm}_{scenario}"]
            if group["success"] < FROZEN_SUCCESS[scenario]:
                return False
            if group["timeout"] or group["safety"]:
                return False
            if group["path_ratio_vs_B15"] > 1.05:
                return False
            if require_attenuation and profile["profile"] != "B20":
                if group.get("ppo_residual_attenuation_steps", 0) <= 0:
                    return False
    return True


def _gate(
    goal100_root: Path,
    goal95_root: Path,
    goal98_root: Path,
    *,
    expected_seeds: set[int],
) -> dict[str, Any]:
    references = _load_references(
        goal95_root, goal98_root, expected_seeds=expected_seeds
    )
    records = _load_goal100(goal100_root, expected_seeds=expected_seeds)
    profiles = {
        profile: _profile_summary(records, references, profile=profile)
        for profile in PROFILES
    }
    b20 = profiles["B20"]
    attenuation_profiles = [profiles[name] for name in ("A3", "A2", "A125")]
    b20_ok = _profile_qualifies(b20, require_attenuation=False)
    attenuation_ok = [
        profile
        for profile in attenuation_profiles
        if _profile_qualifies(profile, require_attenuation=True)
    ]
    if b20_ok:
        status = "BC_SCALE_ONLY_SUPPORTED"
    elif attenuation_ok:
        status = "ATTENUATION_SUPPORTED"
    else:
        strict_touch = any(
            group["success"] >= FROZEN_SUCCESS["strict125"]
            for profile in profiles.values()
            for name, group in profile["groups"].items()
            if name.endswith("strict125")
        )
        default_touch = any(
            group["success"] >= FROZEN_SUCCESS["default"]
            for profile in profiles.values()
            for name, group in profile["groups"].items()
            if name.endswith("default")
        )
        status = (
            "PRIOR_PRESERVATION_PARTIAL"
            if strict_touch or default_touch
            else "PRIOR_PRESERVATION_NOT_SUPPORTED"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "b20_qualifies": b20_ok,
        "qualifying_attenuation_profiles": [
            profile["profile"] for profile in attenuation_ok
        ],
        "profiles": profiles,
        "references": references,
        "heldout_opened": False,
        "training_started": False,
        "algorithm_switched": False,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal100 terminal prior-preservation audit",
        "",
        f"- status: `{payload['status']}`",
        f"- B20 qualifies: `{payload['b20_qualifies']}`",
        "- qualifying attenuation: "
        + (", ".join(payload["qualifying_attenuation_profiles"]) or "none"),
        "",
        "| profile | arm | scenario | success | timeout | safety | path ratio |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for profile_name, profile in payload["profiles"].items():
        for group_name, group in profile["groups"].items():
            arm, scenario = group_name.rsplit("_", 1)
            lines.append(
                f"| {profile_name} | {arm} | {scenario} | {group['success']} | "
                f"{group['timeout']} | {group['safety']} | "
                f"{group['path_ratio_vs_B15']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_seeds = set(range(20, 60))
    payload = _gate(
        args.root,
        args.goal95_root,
        args.goal98_root,
        expected_seeds=expected_seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(f"Goal100 status: {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
