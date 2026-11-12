#!/usr/bin/env python3
"""Build frozen Goal147 terminal pitch-authority evaluation profiles."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/lowlevel/goal136/curriculum/C0.json"
OUTPUT_DIR = REPO_ROOT / "configs/lowlevel/goal147"

GUIDANCE_ARMS = {
    "z": {},
    "b85": {
        "terminal_braking_enabled": True,
        "terminal_speed_ratio": 0.85,
    },
    "b75": {
        "terminal_braking_enabled": True,
        "terminal_speed_ratio": 0.75,
    },
}

CONTROLLER_ARMS = {
    "k4": {"pitch_gain": 4.0},
    "r2": {"pitch_rate_gain": 2.0},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_variant(
    name: str,
    base: dict[str, object],
    section: str,
    changes: dict[str, object],
    *,
    prefix: str | None = None,
) -> dict[str, object]:
    payload = deepcopy(base)
    section_values = payload[section]
    if not isinstance(section_values, dict):
        raise TypeError(f"{section} is not a mapping")
    unknown = set(changes) - set(section_values)
    if unknown:
        raise ValueError(f"unknown {section} keys for {name}: {sorted(unknown)}")
    section_values.update(changes)
    path = OUTPUT_DIR / f"{prefix or section}_{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256(path),
        "changes": changes,
        "section": section,
    }


def main() -> int:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "auv_goal147_terminal_pitch_profiles.v1",
        "base_config": str(BASE_CONFIG.relative_to(REPO_ROOT)),
        "base_sha256": _sha256(BASE_CONFIG),
        "frozen_before_measurement": True,
        "arms": {},
    }
    for name, changes in GUIDANCE_ARMS.items():
        manifest["arms"][name] = {
            "guidance": _write_variant(name, base, "guidance", changes),
            "controller": None,
        }
    for name, changes in CONTROLLER_ARMS.items():
        manifest["arms"][name] = {
            "guidance": _write_variant(name, base, "guidance", {}),
            "controller": _write_variant(
                name, base, "base_controller", changes, prefix="controller"
            ),
        }
    manifest_path = OUTPUT_DIR / "profile_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    for name, entry in manifest["arms"].items():
        print(name, entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
