#!/usr/bin/env python3
"""Validate Goal182 evaluation matrix arithmetic and pressure reporting rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "governance" / "goal182_eval_matrix.json"
DEFAULT_FREEZE_MANIFEST = ROOT / "configs" / "governance" / "goal182_matrix_freeze.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--require-frozen", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tier_errors(tier: dict[str, Any], top: dict[str, Any], errors: list[str]) -> None:
    label = f"tier:{tier.get('tier_id')}"
    tier_id = tier.get("tier_id")
    if tier_id == "EXTERNAL_GEOMETRY_SUITE":
        if tier.get("may_select_winner") is not False or tier.get("may_select_checkpoint") is not False:
            errors.append(f"{label}: external suite must not select winner/checkpoint")
        return
    required = {
        "stage_levels",
        "pressure_levels",
        "radii_m",
        "train_seed_count",
        "eval_seed_count",
        "expected_cells",
        "expected_episodes_per_arm",
    }
    missing = sorted(required - set(tier))
    if missing:
        errors.append(f"{label}: missing {missing}")
        return
    if tier["train_seed_count"] != len(top["train_seeds"]):
        errors.append(f"{label}: train_seed_count mismatch")
    if tier["eval_seed_count"] != len(top["eval_seeds"]):
        errors.append(f"{label}: eval_seed_count mismatch")
    if tier["pressure_levels"] != top["pressure_levels"] and tier["pressure_levels"] != ["nominal"]:
        errors.append(f"{label}: pressure levels must match top-level list or be nominal-only")
    if tier["radii_m"] != top["radii_m"]:
        errors.append(f"{label}: radii mismatch")
    stage_count = len(tier["stage_levels"])
    pressure_count = len(tier["pressure_levels"])
    radius_count = len(tier["radii_m"])
    if tier_id == "MAIN_RANKING":
        route_count = 1
        cells = stage_count * pressure_count * radius_count * route_count * tier["train_seed_count"]
    elif tier_id == "SPEED_ENVELOPE":
        route_count = 1
        speed_count = len(tier.get("fixed_speeds_mps", []))
        cells = stage_count * pressure_count * speed_count * radius_count * route_count * tier["train_seed_count"]
    elif tier_id == "ROUTE_GENERALIZATION":
        route_count = int(tier.get("route_profile_count", 0))
        cells = stage_count * pressure_count * route_count * radius_count * tier["train_seed_count"]
    elif tier_id == "DEPTH_GENERALIZATION":
        route_count = len(tier.get("route_profiles", []))
        depth_count = len(tier.get("depths_m", []))
        cells = stage_count * pressure_count * depth_count * route_count * radius_count * tier["train_seed_count"]
    else:
        errors.append(f"{label}: unknown tier")
        return
    if cells != tier["expected_cells"]:
        errors.append(f"{label}: expected_cells {tier['expected_cells']} != computed {cells}")
    episodes = cells * tier["eval_seed_count"]
    if episodes != tier["expected_episodes_per_arm"]:
        errors.append(
            f"{label}: expected_episodes_per_arm {tier['expected_episodes_per_arm']} != computed {episodes}"
        )


def main() -> int:
    args = parse_args()
    if args.schema_only:
        print("PASS goal182 eval matrix schema")
        return 0
    matrix = _load(args.matrix)
    errors: list[str] = []
    if matrix.get("schema_version") != "auv_goal_matrix.v2":
        errors.append("wrong schema_version")
    if matrix.get("goal_id") != "Goal182" or matrix.get("matrix_id") != "G182-EVAL":
        errors.append("goal/matrix id mismatch")
    if matrix.get("training_profile") != "PPO-FLAT-4096x48-E5-MB4":
        errors.append("training_profile mismatch")
    heldout = matrix.get("heldout", {})
    if heldout.get("enabled") is not False:
        errors.append("held-out must remain closed")
    if set(heldout.get("seeds", [])) & set(range(100, 150)):
        errors.append("held-out seeds 100-149 must not be registered")
    if "never pool pressure" not in str(matrix.get("reporting_rule", "")):
        errors.append("reporting_rule must forbid pressure pooling")
    tiers = matrix.get("tiers", [])
    ids = [str(tier.get("tier_id")) for tier in tiers]
    expected_ids = [
        "MAIN_RANKING",
        "SPEED_ENVELOPE",
        "ROUTE_GENERALIZATION",
        "DEPTH_GENERALIZATION",
        "EXTERNAL_GEOMETRY_SUITE",
    ]
    if ids != expected_ids:
        errors.append(f"tier order/ids mismatch: {ids}")
    for tier in tiers:
        _tier_errors(tier, matrix, errors)
    if args.require_frozen:
        if matrix.get("matrix_status") != "FROZEN":
            errors.append("formal evaluation requires matrix_status=FROZEN")
        frozen_offenders = [
            str(tier.get("tier_id"))
            for tier in tiers
            if tier.get("status") != "FROZEN"
        ]
        if frozen_offenders:
            errors.append(f"formal evaluation requires FROZEN tiers; got {frozen_offenders}")
        if not args.freeze_manifest.is_file():
            errors.append(f"freeze manifest not found: {args.freeze_manifest}")
        else:
            manifest = _load(args.freeze_manifest)
            relative = str(args.matrix.resolve().relative_to(ROOT))
            entries = {
                str(entry.get("path")): entry
                for entry in manifest.get("matrices", [])
                if isinstance(entry, dict)
            }
            entry = entries.get(relative)
            if entry is None:
                errors.append(f"freeze manifest: missing {relative}")
            elif entry.get("sha256") != _sha256(args.matrix):
                errors.append(f"freeze manifest: SHA mismatch for {relative}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    suffix = " frozen" if args.require_frozen else ""
    print(f"PASS goal182 eval matrix{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
