#!/usr/bin/env python3
"""Freeze Goal182 core/evaluation matrices and write the SHA-256 manifest.

Freezing is a protocol action: it changes only ``matrix_status`` / tier
``status`` fields and records the resulting file digests.  Arm execution status
stays untouched.  Re-running the script is idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE = REPO_ROOT / "configs" / "governance"
CORE_MATRICES = (
    GOVERNANCE / "goal182_core_pe2e_matrix.json",
    GOVERNANCE / "goal182_core_alosb_matrix.json",
)
EVAL_MATRIX = GOVERNANCE / "goal182_eval_matrix.json"
MANIFEST = GOVERNANCE / "goal182_matrix_freeze.json"
MANIFEST_SCHEMA = "auv_goal182_matrix_freeze.v1"
TRAINING_PROFILE = "PPO-FLAT-4096x48-E5-MB4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--frozen-at", default=date.today().isoformat())
    parser.add_argument(
        "--frozen-by",
        default="user-confirmed D13.6 freeze and A0/A3 3D priority (2026-11-17)",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_core(path: Path) -> dict[str, Any]:
    payload = _load(path)
    payload["matrix_status"] = "FROZEN"
    _write(path, payload)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "matrix_id": payload["matrix_id"],
        "matrix_status": "FROZEN",
        "arm_count": payload["arm_count"],
        "sha256": _sha256(path),
    }


def _freeze_eval(path: Path) -> dict[str, Any]:
    payload = _load(path)
    payload["matrix_status"] = "FROZEN"
    for tier in payload.get("tiers", []):
        tier["status"] = "FROZEN"
    _write(path, payload)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "matrix_id": payload["matrix_id"],
        "matrix_status": "FROZEN",
        "tiers": {
            tier["tier_id"]: "FROZEN"
            for tier in payload.get("tiers", [])
        },
        "sha256": _sha256(path),
    }


def _manifest_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("freeze manifest: wrong schema_version")
    if manifest.get("goal_id") != "Goal182":
        errors.append("freeze manifest: wrong goal_id")
    if manifest.get("training_profile") != TRAINING_PROFILE:
        errors.append("freeze manifest: wrong training_profile")
    entries = {
        entry.get("path"): entry
        for entry in manifest.get("matrices", [])
        if isinstance(entry, dict)
    }
    for path in (*CORE_MATRICES, EVAL_MATRIX):
        relative = str(path.relative_to(REPO_ROOT))
        entry = entries.get(relative)
        if entry is None:
            errors.append(f"freeze manifest: missing {relative}")
            continue
        if not path.is_file():
            errors.append(f"freeze manifest: missing matrix file {relative}")
            continue
        if entry.get("sha256") != _sha256(path):
            errors.append(f"freeze manifest: SHA mismatch for {relative}")
        payload = _load(path)
        if payload.get("matrix_status") != "FROZEN":
            errors.append(f"freeze manifest: {relative} is not FROZEN")
    return errors


def main() -> int:
    args = parse_args()
    if not args.check:
        entries = [_freeze_core(path) for path in CORE_MATRICES]
        entries.append(_freeze_eval(EVAL_MATRIX))
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "goal_id": "Goal182",
            "training_profile": TRAINING_PROFILE,
            "frozen_at": args.frozen_at,
            "frozen_by": args.frozen_by,
            "matrices": entries,
        }
        _write(args.manifest, manifest)
    manifest = _load(args.manifest)
    errors = _manifest_errors(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"PASS goal182 matrix freeze matrices={len(manifest['matrices'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
