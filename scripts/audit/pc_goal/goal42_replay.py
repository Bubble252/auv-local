#!/usr/bin/env python3
"""Validate Goal42 instrumentation replay against frozen Goal38 traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal42_replay_gate.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
CONTROL_FIELDS = ("eta", "nu", "actual_actuator", "requested_action", "terminal_kind")
EXTRA_FIELDS = (
    "nav_age_s",
    "usbl_age_s",
    "dvl_bottom_age_s",
    "usbl_valid",
    "usbl_outlier",
)
AGE_FIELDS = ("nav_age_s", "usbl_age_s", "dvl_bottom_age_s")
SUMMARY_FLAGS = ("success", "goal_capture_eta", "goal_plane_miss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _bitwise_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _max_abs_diff(left: Any, right: Any) -> float | None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return None
    if not np.issubdtype(left_array.dtype, np.number) or not np.issubdtype(
        right_array.dtype, np.number
    ):
        return None
    difference = np.abs(
        left_array.astype(np.float64) - right_array.astype(np.float64)
    )
    if difference.size == 0:
        return 0.0
    return float(np.max(difference))


def _validate_extra_fields(trace: Any) -> dict[str, Any]:
    missing = [name for name in EXTRA_FIELDS if name not in trace.files]
    if missing:
        raise ValueError(f"replay trace is missing sensor-age fields: {missing}")
    finite: dict[str, bool] = {}
    nonnegative: dict[str, bool] = {}
    unit_interval: dict[str, bool] = {}
    for name in EXTRA_FIELDS:
        values = np.asarray(trace[name], dtype=np.float64)
        finite[name] = bool(np.all(np.isfinite(values)))
        if not finite[name]:
            raise ValueError(f"replay trace has non-finite {name}")
        if name in AGE_FIELDS:
            nonnegative[name] = bool(np.all(values >= 0.0))
            if not nonnegative[name]:
                raise ValueError(f"replay trace has negative {name}")
        if name in ("usbl_valid", "usbl_outlier"):
            unit_interval[name] = bool(np.all((values >= 0.0) & (values <= 1.0)))
            if not unit_interval[name]:
                raise ValueError(f"replay trace has out-of-range {name}")
    return {
        "fields": list(EXTRA_FIELDS),
        "all_finite": all(finite.values()),
        "nonnegative": nonnegative,
        "unit_interval": unit_interval,
    }


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    seeds = sorted(int(seed) for seed in summary["seeds"])
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"{path}: seeds mismatch")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{path}: checkpoint SHA256 mismatch")
    episodes = {int(item["seed"]): item for item in summary["episodes"]}
    if sorted(episodes) != list(EXPECTED_SEEDS):
        raise ValueError(f"{path}: episode seeds mismatch")
    for item in episodes.values():
        for name in SUMMARY_FLAGS:
            if not isinstance(item.get(name), bool):
                raise ValueError(f"{path}: episode has invalid {name}")
    return {"summary": summary, "episodes": episodes}


def _compare_summaries(
    reference: dict[str, Any], replay: dict[str, Any]
) -> dict[str, Any]:
    mismatches: list[str] = []
    for seed in EXPECTED_SEEDS:
        ref_episode = reference["episodes"][seed]
        replay_episode = replay["episodes"][seed]
        for name in SUMMARY_FLAGS:
            if ref_episode[name] != replay_episode[name]:
                mismatches.append(f"seed {seed}: {name}")
    return {
        "consistent": not mismatches,
        "mismatches": mismatches,
        "compared_fields": list(SUMMARY_FLAGS),
    }


def _audit_block(
    reference_root: Path, replay_root: Path, scenario: str, mode: str
) -> dict[str, Any]:
    reference_dir = reference_root / scenario / mode
    replay_dir = replay_root / scenario / mode
    reference = _load_summary(reference_dir / "summary.json")
    replay = _load_summary(replay_dir / "summary.json")
    summary_check = _compare_summaries(reference, replay)

    with np.load(reference_dir / "trace.npz", allow_pickle=False) as reference_trace:
        with np.load(replay_dir / "trace.npz", allow_pickle=False) as replay_trace:
            missing_extra = _validate_extra_fields(replay_trace)
            reference_fields = tuple(reference_trace.files)
            missing_reference_fields = [
                name for name in reference_fields if name not in replay_trace.files
            ]
            if missing_reference_fields:
                raise ValueError(
                    f"replay trace is missing Goal38 fields: {missing_reference_fields}"
                )

            seed_step_equal = _bitwise_equal(
                reference_trace["episode_seed"], replay_trace["episode_seed"]
            ) and _bitwise_equal(reference_trace["step"], replay_trace["step"])
            if not seed_step_equal:
                raise ValueError("replay seed/step sequence differs from Goal38")

            control_checks: dict[str, Any] = {}
            for name in CONTROL_FIELDS:
                control_checks[name] = {
                    "bitwise_equal": _bitwise_equal(
                        reference_trace[name], replay_trace[name]
                    ),
                    "max_abs_diff": _max_abs_diff(
                        reference_trace[name], replay_trace[name]
                    ),
                }

            mismatched_reference_fields = [
                name
                for name in reference_fields
                if not _bitwise_equal(reference_trace[name], replay_trace[name])
            ]
            row_count = int(np.asarray(replay_trace["episode_seed"]).shape[0])
            seeds = sorted(
                int(seed) for seed in np.unique(replay_trace["episode_seed"])
            )
            if seeds != list(EXPECTED_SEEDS):
                raise ValueError("replay trace seed set differs from dev seeds")

    control_passed = all(item["bitwise_equal"] for item in control_checks.values())
    reference_passed = not mismatched_reference_fields
    passed = (
        summary_check["consistent"]
        and seed_step_equal
        and control_passed
        and reference_passed
        and missing_extra["all_finite"]
    )
    return {
        "scenario": scenario,
        "mode": mode,
        "reference": str(reference_dir),
        "replay": str(replay_dir),
        "row_count": row_count,
        "seeds": seeds,
        "seed_step_sequence_bitwise_equal": seed_step_equal,
        "summary": summary_check,
        "control_fields": control_checks,
        "control_fields_bitwise_equal": control_passed,
        "reference_field_count": len(reference_fields),
        "mismatched_reference_fields": mismatched_reference_fields,
        "all_reference_fields_bitwise_equal": reference_passed,
        "instrumentation": missing_extra,
        "pass": passed,
    }


def analyze(reference_root: Path, replay_root: Path) -> dict[str, Any]:
    blocks: dict[str, Any] = {}
    invalid_reasons: list[str] = []
    for scenario in SCENARIOS:
        for mode in MODES:
            key = f"{scenario}/{mode}"
            try:
                block = _audit_block(reference_root, replay_root, scenario, mode)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                invalid_reasons.append(f"{key}: {exc}")
                continue
            blocks[key] = block
            if not block["pass"]:
                invalid_reasons.append(f"{key}: replay gate failed")

    expected_block_count = len(SCENARIOS) * len(MODES)
    valid = not invalid_reasons and len(blocks) == expected_block_count
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_root": str(reference_root),
        "replay_root": str(replay_root),
        "validation": {
            "gate": "PASS" if valid else "INVALID_REPLAY",
            "valid": valid,
            "invalid_reasons": invalid_reasons,
            "block_count": len(blocks),
            "expected_block_count": expected_block_count,
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "heldout_opened": False,
        },
        "blocks": blocks,
    }


def main() -> int:
    args = parse_args()
    result = analyze(args.reference_root, args.formal_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "gate": result["validation"]["gate"],
                "valid": result["validation"]["valid"],
                "blocks": result["validation"]["block_count"],
                "invalid_reasons": result["validation"]["invalid_reasons"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
