#!/usr/bin/env python3
"""Build frozen Goal151 T2-SAC terminal-gate evaluation configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json"
OUTPUT_DIR = REPO_ROOT / "configs/lowlevel/goal151"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant(source: dict[str, Any], name: str, *, full_m: float) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["experiment"]["name"] = f"pc_goal151_{name}_20261111"
    payload["experiment"]["output"] = f"artifacts/lowlevel/goal151_{name}"
    payload["residual_terminal_gate_enabled"] = True
    payload["residual_terminal_gate_enter_m"] = 10.0
    payload["residual_terminal_gate_full_m"] = full_m
    return payload


def build(output_dir: Path, *, force: bool = False) -> dict[str, Any]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    variants = {
        "near10_2": _variant(source, "near10_2", full_m=2.0),
        "shell10_3": _variant(source, "shell10_3", full_m=3.0),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, Any] = {}
    for name, payload in variants.items():
        path = output_dir / f"{name}.json"
        if path.exists() and not force:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current != payload:
                raise FileExistsError(f"{path} exists with different content; use --force")
        else:
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        profiles[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "residual_terminal_gate_enabled": True,
            "residual_terminal_gate_enter_m": 10.0,
            "residual_terminal_gate_full_m": float(payload["residual_terminal_gate_full_m"]),
            "learner_terminal_gate_mode": "shell" if name == "shell10_3" else "none",
        }
    manifest = {
        "schema_version": "auv_goal151_t2_terminal_gate_profiles.v1",
        "source_config": str(SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(SOURCE),
        "frozen_before_measurement": True,
        "variants": profiles,
    }
    (output_dir / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build(args.output_dir, force=args.force)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
