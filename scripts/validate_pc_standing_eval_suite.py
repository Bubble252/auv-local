#!/usr/bin/env python3
"""Validate the frozen standing evaluation contract before a training Goal starts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO_ROOT / "configs/lowlevel/eval/pc_standing_eval_suite_v1.json"
SCHEMA = "auv_pc_standing_eval_suite.v1"

REQUIRED_PARADIGMS = {
    "T0",
    "T0-NONE",
    "BC",
    "BC-PPO",
    "BC-SAC",
    "ALOS-PPO",
    "ALOS-SAC",
}
REQUIRED_ALOS_CURRENT_MODES = {"ALOS-NONE", "ALOS-COMP"}
EXPECTED_ALOS_CURRENT_MODES = {
    "ALOS-NONE": {
        "compensate_current": False,
        "current_crab_enabled": False,
        "current_compensation_source": "none",
    },
    "ALOS-COMP": {
        "compensate_current": True,
        "current_crab_enabled": False,
        "current_compensation_source": "estimated",
    },
}
REQUIRED_CURRICULA = {"C-fixed", "C-full", "C-hard", "C-no-current"}
REQUIRED_STAGES = {f"C{index}" for index in range(7)}
REQUIRED_PRESSURE = {
    "nominal",
    "current05",
    "current07",
    "current10",
    "current15",
    "noise_n2",
    "plant20",
    "combined20",
}
REQUIRED_ROUTES = {"legacy_sine", "P0", "P1", "P2", "P3", "P4"}
REQUIRED_LENGTHS = {30, 35, 45}
REQUIRED_SPEEDS = {0.9, 1.2, 1.4, 1.8, 2.2, 2.5}
REQUIRED_MAX_SPEED_MPS = 2.5
REQUIRED_METRIC_GROUPS = {
    "task_outcome",
    "route_health",
    "terminal_capture",
    "direction_and_failure",
    "control_health",
    "integrity",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    return parser.parse_args(argv)


def validate(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA:
        errors.append(f"schema_version must be {SCHEMA!r}")

    success = payload.get("success_contract") or {}
    if success.get("primary") != "success_eta_tight_at_done":
        errors.append("success_contract.primary must be success_eta_tight_at_done")
    if bool(success.get("p_nav_used_as_performance_axis", True)):
        errors.append("success_contract.p_nav_used_as_performance_axis must be false")
    if bool(success.get("extra_position_offset", True)):
        errors.append("success_contract.extra_position_offset must be false")
    if bool(success.get("extra_latency", True)):
        errors.append("success_contract.extra_latency must be false")
    if set((success.get("goal_radii_m") or {}).keys()) != {"default", "strict125"}:
        errors.append("success_contract.goal_radii_m must contain default and strict125")

    heldout = payload.get("heldout_seeds") or {}
    if heldout.get("status") != "closed":
        errors.append("heldout_seeds.status must be closed")
    if (heldout.get("start"), heldout.get("end")) != (100, 149):
        errors.append("heldout_seeds must be exactly 100-149 and closed")

    alos_modes = {
        item.get("id"): item
        for item in payload.get("alos_current_modes", [])
        if isinstance(item, dict)
    }
    if set(alos_modes) != REQUIRED_ALOS_CURRENT_MODES:
        errors.append(
            "alos_current_modes must contain exactly "
            f"{sorted(REQUIRED_ALOS_CURRENT_MODES)}"
        )
    for mode_id, expected in EXPECTED_ALOS_CURRENT_MODES.items():
        mode = alos_modes.get(mode_id) or {}
        for field, value in expected.items():
            if mode.get(field) != value:
                errors.append(f"alos_current_modes.{mode_id}.{field} must be {value!r}")
    if payload.get("default_alos_current_mode") != "ALOS-COMP":
        errors.append("default_alos_current_mode must be 'ALOS-COMP'")

    paradigm_items = {
        item.get("id"): item
        for item in payload.get("paradigms", [])
        if isinstance(item, dict)
    }
    missing = REQUIRED_PARADIGMS - set(paradigm_items)
    if missing:
        errors.append(f"missing paradigms: {sorted(missing)}")
    for paradigm_id, paradigm in paradigm_items.items():
        mode_id = paradigm.get("alos_current_mode")
        if mode_id not in REQUIRED_ALOS_CURRENT_MODES:
            errors.append(
                f"paradigms.{paradigm_id}.alos_current_mode must be one of "
                f"{sorted(REQUIRED_ALOS_CURRENT_MODES)}"
            )
    if (paradigm_items.get("T0") or {}).get("alos_current_mode") != "ALOS-COMP":
        errors.append("paradigm T0 must use ALOS-COMP")
    if (paradigm_items.get("T0-NONE") or {}).get("alos_current_mode") != "ALOS-NONE":
        errors.append("paradigm T0-NONE must use ALOS-NONE")

    curricula = {
        item.get("id"): item for item in payload.get("curricula", []) if isinstance(item, dict)
    }
    missing = REQUIRED_CURRICULA - set(curricula)
    if missing:
        errors.append(f"missing curricula: {sorted(missing)}")
    for name in ("C-fixed", "C-full", "C-hard", "C-no-current"):
        curriculum = curricula.get(name) or {}
        stages = set(curriculum.get("stages", []))
        if not stages:
            errors.append(f"curricula.{name}.stages must not be empty")
        if curriculum.get("mode") not in {"fixed", "promotion"}:
            errors.append(f"curricula.{name}.mode must be fixed or promotion")
        eval_stage = curriculum.get("eval_stage")
        final_stage = curriculum.get("final_stage")
        if eval_stage not in REQUIRED_STAGES:
            errors.append(f"curricula.{name}.eval_stage must be C0-C6")
        if final_stage not in REQUIRED_STAGES:
            errors.append(f"curricula.{name}.final_stage must be C0-C6")
        if eval_stage != final_stage:
            errors.append(f"curricula.{name}.eval_stage must equal final_stage")
    full_stages = set((curricula.get("C-full") or {}).get("stages", []))
    if not REQUIRED_STAGES.issubset(full_stages):
        errors.append(f"C-full must contain {sorted(REQUIRED_STAGES - full_stages)}")
    full_final = (curricula.get("C-full") or {}).get("final_stage")
    hard_final = (curricula.get("C-hard") or {}).get("final_stage")
    if full_final != hard_final:
        errors.append("C-full and C-hard must share the same final_stage")
    if (curricula.get("C-fixed") or {}).get("mode") != "fixed":
        errors.append("C-fixed.mode must be fixed")
    if (curricula.get("C-full") or {}).get("mode") != "promotion":
        errors.append("C-full.mode must be promotion")

    pressure = {
        item.get("id") for item in payload.get("pressure_profiles", []) if isinstance(item, dict)
    }
    missing = REQUIRED_PRESSURE - pressure
    if missing:
        errors.append(f"missing pressure profiles: {sorted(missing)}")
    routes = set(payload.get("route_profiles", []))
    missing = REQUIRED_ROUTES - routes
    if missing:
        errors.append(f"missing route profiles: {sorted(missing)}")
    if set(payload.get("length_profiles_m", [])) != REQUIRED_LENGTHS:
        errors.append(f"length_profiles_m must be {sorted(REQUIRED_LENGTHS)}")
    if set(payload.get("speed_profiles_mps", [])) != REQUIRED_SPEEDS:
        errors.append(f"speed_profiles_mps must be {sorted(REQUIRED_SPEEDS)}")
    if float(payload.get("speed_max_mps", 0.0)) != REQUIRED_MAX_SPEED_MPS:
        errors.append(f"speed_max_mps must be {REQUIRED_MAX_SPEED_MPS}")
    for speed in payload.get("speed_profiles_mps", []):
        if float(speed) > REQUIRED_MAX_SPEED_MPS + 1e-12:
            errors.append(
                f"speed_profiles_mps.{speed} exceeds the REMUS100 envelope "
                f"({REQUIRED_MAX_SPEED_MPS} m/s)"
            )

    speed_curriculum = payload.get("speed_curriculum") or {}
    stage_speeds = speed_curriculum.get("stages_mps") or {}
    if set(stage_speeds) != REQUIRED_STAGES:
        errors.append("speed_curriculum.stages_mps must contain C0-C6")
    upper_bounds: list[float] = []
    for stage in sorted(REQUIRED_STAGES, key=lambda value: int(value[1:])):
        values = stage_speeds.get(stage)
        if not isinstance(values, list) or len(values) != 2:
            errors.append(f"speed_curriculum.stages_mps.{stage} must be a 2-item list")
            continue
        lower, upper = float(values[0]), float(values[1])
        if lower < 0.0 or upper < lower:
            errors.append(f"speed_curriculum.stages_mps.{stage} must satisfy 0 <= min <= max")
        if upper > REQUIRED_MAX_SPEED_MPS + 1e-12:
            errors.append(
                f"speed_curriculum.stages_mps.{stage} exceeds the REMUS100 envelope "
                f"({REQUIRED_MAX_SPEED_MPS} m/s)"
            )
        upper_bounds.append(upper)
    if upper_bounds and any(
        left > right + 1e-12
        for left, right in zip(upper_bounds, upper_bounds[1:], strict=False)
    ):
        errors.append("speed_curriculum upper bounds must be non-decreasing from C0 to C6")
    c6 = stage_speeds.get("C6")
    if isinstance(c6, list) and len(c6) == 2:
        if float(c6[0]) > min(REQUIRED_SPEEDS) or float(c6[1]) < max(REQUIRED_SPEEDS):
            errors.append("speed_curriculum.C6 must cover the full speed_profiles_mps axis")

    metrics = payload.get("required_metrics") or {}
    missing_groups = REQUIRED_METRIC_GROUPS - set(metrics)
    if missing_groups:
        errors.append(f"missing metric groups: {sorted(missing_groups)}")
    for group in REQUIRED_METRIC_GROUPS & set(metrics):
        if not isinstance(metrics[group], list) or not metrics[group]:
            errors.append(f"required_metrics.{group} must be a non-empty list")

    tiers = payload.get("tiers") or []
    if not tiers:
        errors.append("tiers must not be empty")
    tier_ids = {item.get("id") for item in tiers if isinstance(item, dict)}
    for required in {"A_screen", "B_robustness", "C_full_geometry", "D_formal"}:
        if required not in tier_ids:
            errors.append(f"missing tier {required!r}")
    for tier in tiers:
        if not isinstance(tier, dict):
            errors.append("every tier must be an object")
            continue
        if int(tier.get("training_seeds", 0)) <= 0:
            errors.append(f"tier {tier.get('id')!r} training_seeds must be positive")
        if int(tier.get("eval_seeds", 0)) <= 0:
            errors.append(f"tier {tier.get('id')!r} eval_seeds must be positive")
        tier_speeds = tier.get("speed_profiles_mps")
        if not isinstance(tier_speeds, list) or not tier_speeds:
            errors.append(f"tier {tier.get('id')!r} speed_profiles_mps must be non-empty")
        elif not set(float(value) for value in tier_speeds).issubset(REQUIRED_SPEEDS):
            errors.append(f"tier {tier.get('id')!r} contains an undeclared speed profile")
        if tier.get("id") in {"B_robustness", "C_full_geometry", "D_formal"}:
            if REQUIRED_MAX_SPEED_MPS not in set(float(value) for value in (tier_speeds or [])):
                errors.append(f"tier {tier.get('id')!r} must include {REQUIRED_MAX_SPEED_MPS} m/s")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(args.suite.read_text(encoding="utf-8"))
    errors = validate(payload)
    report = {
        "suite": str(args.suite),
        "schema_version": payload.get("schema_version"),
        "valid": not errors,
        "errors": errors,
        "paradigm_count": len(payload.get("paradigms", [])),
        "pressure_count": len(payload.get("pressure_profiles", [])),
        "route_count": len(payload.get("route_profiles", [])),
        "tier_count": len(payload.get("tiers", [])),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif errors:
        print("INVALID standing eval suite:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print(
            "OK standing eval suite: "
            f"{report['paradigm_count']} paradigms, {report['pressure_count']} pressure profiles, "
            f"{report['route_count']} route profiles, {report['tier_count']} tiers"
        )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
