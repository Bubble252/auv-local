#!/usr/bin/env python3
"""Audit Goal104's pre-registered per-channel actuator gain matrix.

The script reads only dev40 evaluation summaries and traces.  It does not load
or step a policy, train, alter success semantics, or read held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal104_channel_parameterization.v1"
ARM = "P_D3"
SCENARIOS = ("default", "strict125")
PROFILES: dict[str, tuple[float, float, float]] = {
    "C0": (1.00, 1.00, 1.00),
    "R075": (0.75, 1.00, 1.00),
    "R125": (1.25, 1.00, 1.00),
    "R050": (0.50, 1.00, 1.00),
    "R150": (1.50, 1.00, 1.00),
    "S075": (1.00, 0.75, 1.00),
    "S125": (1.00, 1.25, 1.00),
    "S050": (1.00, 0.50, 1.00),
    "S150": (1.00, 1.50, 1.00),
    "U075": (1.00, 1.00, 0.75),
    "U125": (1.00, 1.00, 1.25),
    "U050": (1.00, 1.00, 0.50),
    "U150": (1.00, 1.00, 1.50),
}
CHANNELS: dict[str, tuple[str, ...]] = {
    "rudder": ("R050", "R075", "R125", "R150"),
    "stern": ("S050", "S075", "S125", "S150"),
    "rpm": ("U050", "U075", "U125", "U150"),
}
BASELINE_PROFILE = "C0"
EVAL_SCALE = 2.0
TRAINING_SCALE = 1.5
EXPECTED_SUCCESS = {"default": 33, "strict125": 20}
GATE_SUCCESS = {"default": 33, "strict125": 21}
EXPECTED_SEEDS = set(range(20, 60))
EXPECTED_CHECKPOINT_SHA = (
    "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616"
)
PATH_RATIO_LIMIT = 1.05
EXPECTED_RUNS = {
    f"{ARM}_{scenario}_{profile}"
    for scenario in SCENARIOS
    for profile in PROFILES
}
TRACE_FIELDS = (
    "step",
    "episode_seed",
    "goal_distance_eta_m",
    "terminal_kind",
    "actuator_residual_applied",
)


class AuditError(ValueError):
    """Raised when the channel-gain matrix is incomplete or malformed."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal104_channel_parameterization_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal104_channel_parameterization_20261111/"
            "phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal104_channel_parameterization_20261111/"
            "phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _validate_trace(path: Path) -> None:
    if not path.is_file():
        raise AuditError(f"{path}: missing trace")
    with np.load(path) as data:
        missing = [name for name in TRACE_FIELDS if name not in data]
        if missing:
            raise AuditError(f"{path}: missing trace fields {missing}")
        for name in TRACE_FIELDS:
            array = np.asarray(data[name])
            if array.size == 0:
                raise AuditError(f"{path}: empty trace field {name}")
            if np.issubdtype(array.dtype, np.number) and not np.all(
                np.isfinite(array)
            ):
                raise AuditError(f"{path}: non-finite trace field {name}")
        residual_shape = np.asarray(data["actuator_residual_applied"]).shape
        if len(residual_shape) != 2 or residual_shape[1] != 3:
            raise AuditError(
                f"{path}: actuator_residual_applied must have shape (N, 3)"
            )
        trace_seeds = np.asarray(data["episode_seed"], dtype=np.int64)
        if set(np.unique(trace_seeds).tolist()) != EXPECTED_SEEDS:
            raise AuditError(f"{path}: trace seeds do not cover 20-59")


def _success_by_seed(episodes: list[dict[str, Any]], path: Path) -> dict[int, bool]:
    by_seed: dict[int, bool] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{path}: malformed episode row")
        seed = int(row["seed"])
        if seed in by_seed:
            raise AuditError(f"{path}: duplicate seed {seed}")
        by_seed[seed] = bool(row["success"])
    if set(by_seed) != EXPECTED_SEEDS:
        raise AuditError(f"{path}: seeds do not cover 20-59")
    return by_seed


def _run_record(run_dir: Path, *, scenario: str, profile: str) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise AuditError(f"{summary_path}: expected 40 dev episodes")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise AuditError(f"{summary_path}: checkpoint SHA mismatch")
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - EVAL_SCALE) > 1e-12:
        raise AuditError(
            f"{summary_path}: scale override {recorded_scale!r} != {EVAL_SCALE}"
        )
    expected_gains = PROFILES[profile]
    recorded_gains = summary.get("actuator_residual_channel_gains")
    if (
        not isinstance(recorded_gains, list)
        or len(recorded_gains) != 3
        or any(
            abs(float(actual) - expected) > 1e-12
            for actual, expected in zip(recorded_gains, expected_gains, strict=True)
        )
    ):
        raise AuditError(f"{summary_path}: channel gains mismatch")
    _validate_trace(run_dir / "trace.npz")
    by_seed = _success_by_seed(episodes, summary_path)
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise AuditError(f"{summary_path}: missing metrics object")
    path_rmse = metrics.get("path_three_d_rmse_m")
    if not isinstance(path_rmse, (int, float)) or not math.isfinite(float(path_rmse)):
        raise AuditError(f"{summary_path}: invalid path_three_d_rmse_m")
    return {
        "run": run_dir.name,
        "scenario": scenario,
        "profile": profile,
        "channel_gains": list(expected_gains),
        "episodes": len(episodes),
        "success": sum(by_seed.values()),
        "success_by_seed": by_seed,
        "success_pnav": sum(bool(row.get("success_pnav_tight", False)) for row in episodes),
        "success_eta_ever": sum(
            bool(row.get("success_eta_tight_ever", False)) for row in episodes
        ),
        "success_eta_done": sum(
            bool(row.get("success_eta_tight_at_done", False)) for row in episodes
        ),
        "timeout": sum(bool(row.get("timeout", False)) for row in episodes),
        "safety": sum(bool(row.get("safety", False)) for row in episodes),
        "path_three_d_rmse_m": float(path_rmse),
        "residual_magnitude_clip_rate": float(
            metrics.get("actuator_residual_magnitude_clip_rate", math.nan)
        ),
        "residual_rate_clip_rate": float(
            metrics.get("actuator_residual_rate_clip_rate", math.nan)
        ),
    }


def _scan(root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    if not root.is_dir():
        raise AuditError(f"{root}: missing channel-gain root")
    groups: dict[str, dict[str, dict[str, Path]]] = {
        scenario: {profile: {} for profile in PROFILES} for scenario in SCENARIOS
    }
    found: set[str] = set()
    prefix = f"{ARM}_"
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not run_dir.name.startswith(prefix):
            continue
        parts = run_dir.name[len(prefix) :].split("_", 1)
        if len(parts) != 2:
            continue
        scenario, profile = parts
        if scenario not in SCENARIOS or profile not in PROFILES:
            continue
        if profile in groups[scenario]:
            groups[scenario][profile] = {"path": run_dir}
            found.add(run_dir.name)
    missing = sorted(EXPECTED_RUNS - found)
    if missing:
        raise AuditError(f"missing runs: {missing}")
    return groups


def _mcnemar_exact_p(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = min(gained, lost)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _transition(
    baseline: dict[int, bool], other: dict[int, bool]
) -> dict[str, Any]:
    gained = sorted(seed for seed, value in other.items() if value and not baseline[seed])
    lost = sorted(seed for seed in baseline if baseline[seed] and not other[seed])
    return {
        "gained": gained,
        "lost": lost,
        "net": len(gained) - len(lost),
        "mcnemar_exact_p": _mcnemar_exact_p(len(gained), len(lost)),
    }


def _summarize(
    groups: dict[str, dict[str, dict[str, Path]]]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        records[scenario] = {
            profile: _run_record(
                groups[scenario][profile]["path"],
                scenario=scenario,
                profile=profile,
            )
            for profile in PROFILES
        }
    baseline = {
        scenario: records[scenario][BASELINE_PROFILE] for scenario in SCENARIOS
    }
    summary: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for profile in PROFILES:
            record = records[scenario][profile]
            path_ratio = (
                record["path_three_d_rmse_m"]
                / baseline[scenario]["path_three_d_rmse_m"]
                if baseline[scenario]["path_three_d_rmse_m"] > 0.0
                else None
            )
            summary[scenario][profile] = {
                "profile": profile,
                "channel_gains": record["channel_gains"],
                "success": record["success"],
                "success_pnav": record["success_pnav"],
                "success_eta_ever": record["success_eta_ever"],
                "success_eta_done": record["success_eta_done"],
                "timeout": record["timeout"],
                "safety": record["safety"],
                "path_three_d_rmse_m": record["path_three_d_rmse_m"],
                "path_ratio_vs_C0": path_ratio,
                "residual_magnitude_clip_rate": record["residual_magnitude_clip_rate"],
                "residual_rate_clip_rate": record["residual_rate_clip_rate"],
                "transition_vs_C0": _transition(
                    baseline[scenario]["success_by_seed"], record["success_by_seed"]
                ),
                "success_by_seed": record["success_by_seed"],
            }
    return summary


def _channel_direction(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = summary["strict125"][BASELINE_PROFILE]
    channels: dict[str, Any] = {}
    for channel, profiles in CHANNELS.items():
        positive = [
            profile
            for profile in profiles
            if summary["strict125"][profile]["success"] > baseline["success"]
        ]
        negative = [
            profile
            for profile in profiles
            if summary["strict125"][profile]["success"] < baseline["success"]
        ]
        pairs: list[dict[str, Any]] = []
        for pos in positive:
            for neg in negative:
                pos_success = summary["strict125"][pos]["success_by_seed"]
                neg_success = summary["strict125"][neg]["success_by_seed"]
                positive_only = sorted(
                    seed
                    for seed, value in pos_success.items()
                    if value and not neg_success[seed]
                )
                negative_only = sorted(
                    seed
                    for seed, value in neg_success.items()
                    if value and not pos_success[seed]
                )
                if positive_only:
                    pairs.append(
                        {
                            "positive_profile": pos,
                            "negative_profile": neg,
                            "same_seed_opposite_flips": positive_only,
                            "reverse_only_seeds": negative_only,
                        }
                    )
        channels[channel] = {
            "positive_strict_profiles": positive,
            "negative_strict_profiles": negative,
            "opposite_seed_pairs": pairs,
            "directional": bool(positive and negative and pairs),
            "profiles": {
                profile: {
                    "strict_delta": (
                        summary["strict125"][profile]["success"]
                        - baseline["success"]
                    ),
                    "default_delta": (
                        summary["default"][profile]["success"]
                        - summary["default"][BASELINE_PROFILE]["success"]
                    ),
                }
                for profile in profiles
            },
        }
    return channels


def _gate(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_failures: list[str] = []
    for scenario in SCENARIOS:
        actual = summary[scenario][BASELINE_PROFILE]["success"]
        expected = EXPECTED_SUCCESS[scenario]
        if actual != expected:
            baseline_failures.append(
                f"C0 {scenario}: {actual}/40 != expected {expected}/40"
            )
    if baseline_failures:
        return {
            "status": "INVALID",
            "reason": "C0_does_not_reproduce_expected_success",
            "baseline_failures": baseline_failures,
            "qualifying_profiles": [],
            "channel_direction": {},
        }

    candidates: list[dict[str, Any]] = []
    for profile in PROFILES:
        if profile == BASELINE_PROFILE:
            continue
        default = summary["default"][profile]
        strict = summary["strict125"][profile]
        path_ratios = [
            default["path_ratio_vs_C0"],
            strict["path_ratio_vs_C0"],
        ]
        success_ok = (
            default["success"] >= GATE_SUCCESS["default"]
            and strict["success"] >= GATE_SUCCESS["strict125"]
        )
        path_ok = all(
            ratio is not None and ratio <= PATH_RATIO_LIMIT for ratio in path_ratios
        )
        risk_ok = (
            default["timeout"] == 0
            and strict["timeout"] == 0
            and default["safety"] == 0
            and strict["safety"] == 0
        )
        if success_ok and path_ok and risk_ok:
            candidates.append(
                {
                    "profile": profile,
                    "channel_gains": list(PROFILES[profile]),
                    "default_success": default["success"],
                    "strict125_success": strict["success"],
                    "default_path_ratio_vs_C0": default["path_ratio_vs_C0"],
                    "strict125_path_ratio_vs_C0": strict["path_ratio_vs_C0"],
                }
            )
    channel_direction = _channel_direction(summary)
    if candidates:
        return {
            "status": "CHANNEL_SIGNAL",
            "reason": "profile_meets_success_path_and_risk_contract",
            "baseline_failures": [],
            "qualifying_profiles": candidates,
            "channel_direction": channel_direction,
        }
    directional_channels = [
        channel
        for channel, result in channel_direction.items()
        if bool(result["directional"])
    ]
    if directional_channels:
        return {
            "status": "SINGLE_CHANNEL_DIRECTION",
            "reason": "opposite_strict_directions_with_same_seed_flips",
            "baseline_failures": [],
            "qualifying_profiles": [],
            "directional_channels": directional_channels,
            "channel_direction": channel_direction,
        }
    return {
        "status": "NO_CHANNEL_SIGNAL",
        "reason": "no_profile_meets_gate_and_no_same_seed_directional_pair",
        "baseline_failures": [],
        "qualifying_profiles": [],
        "channel_direction": channel_direction,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    gate = payload["gate"]
    lines = [
        "# Goal104 Phase 0：B20 分通道动作参数化审计",
        "",
        "本文件由 `scripts/analyze_pc_goal104_channel_parameterization.py` 生成；只读 dev40 summary/trace，不训练。",
        "",
        f"- Gate: `{gate['status']}`",
        f"- reason: `{gate['reason']}`",
        f"- profile 数: `{len(PROFILES)}`，scenario 数: `{len(SCENARIOS)}`",
        "",
    ]
    if gate.get("baseline_failures"):
        lines.extend(["## INVALID 原因", ""])
        lines.extend(f"- {failure}" for failure in gate["baseline_failures"])
        lines.append("")
    if gate.get("qualifying_profiles"):
        lines.extend(["## CHANNEL_SIGNAL profiles", ""])
        lines.append("| profile | default | strict125 | default path ratio | strict path ratio |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in gate["qualifying_profiles"]:
            lines.append(
                f"| {row['profile']} | {row['default_success']} | "
                f"{row['strict125_success']} | "
                f"{_fmt(row['default_path_ratio_vs_C0'])} | "
                f"{_fmt(row['strict125_path_ratio_vs_C0'])} |"
            )
        lines.append("")
    for scenario in SCENARIOS:
        lines.extend(
            [
                f"## {scenario}",
                "",
                "| profile | gains (r,s,u) | success | pnav | eta-ever | path ratio | timeout | safety | gained | lost | McNemar p |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
            ]
        )
        scenario_summary = payload.get("summary", {}).get(scenario, {})
        if not scenario_summary:
            lines.append("| — | — | — | — | — | — | — | — | — | — |")
        else:
            for profile in PROFILES:
                row = scenario_summary[profile]
                lines.append(
                    f"| {profile} | {row['channel_gains']} | {row['success']}/40 | "
                    f"{row['success_pnav']} | {row['success_eta_ever']} | "
                    f"{_fmt(row['path_ratio_vs_C0'])} | {row['timeout']} | "
                    f"{row['safety']} | {row['transition_vs_C0']['gained']} | "
                    f"{row['transition_vs_C0']['lost']} | "
                    f"{row['transition_vs_C0']['mcnemar_exact_p']:.4f} |"
                )
        lines.append("")
    lines.extend(["## 分通道方向", ""])
    for channel, result in gate["channel_direction"].items():
        lines.append(f"### {channel}")
        lines.append("")
        lines.append(
            f"- positive strict profiles: `{result['positive_strict_profiles']}`"
        )
        lines.append(
            f"- negative strict profiles: `{result['negative_strict_profiles']}`"
        )
        lines.append(f"- directional: `{result['directional']}`")
        for pair in result["opposite_seed_pairs"]:
            lines.append(
                f"- {pair['positive_profile']} vs {pair['negative_profile']}: "
                f"same-seed flips `{pair['same_seed_opposite_flips']}`"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = _summarize(_scan(args.root))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "arm": ARM,
            "profiles": {name: list(gains) for name, gains in PROFILES.items()},
            "scenarios": list(SCENARIOS),
            "eval_scale": EVAL_SCALE,
            "training_scale": TRAINING_SCALE,
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "summary": summary,
            "gate": _gate(summary),
        }
    except AuditError as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "arm": ARM,
            "profiles": {name: list(gains) for name, gains in PROFILES.items()},
            "scenarios": list(SCENARIOS),
            "eval_scale": EVAL_SCALE,
            "training_scale": TRAINING_SCALE,
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "summary": {},
            "gate": {
                "status": "INVALID",
                "reason": str(exc),
                "baseline_failures": [],
                "qualifying_profiles": [],
                "channel_direction": {},
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(payload, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print(f"gate={payload['gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
