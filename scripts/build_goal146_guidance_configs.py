#!/usr/bin/env python3
"""Build the frozen Goal146 guidance-only evaluation profiles."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/lowlevel/goal136/curriculum/C0.json"
OUTPUT_DIR = REPO_ROOT / "configs/lowlevel/goal146"

ARMS = {
    "z": {},
    "p1_pitch20deg": {"pitch_rate_limit_rad_s": 0.3490658504},
    "p2_pitch30deg": {"pitch_rate_limit_rad_s": 0.5235987756},
    "s1_speed035": {"speed_rate_limit_mps2": 0.35},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base_guidance = base["guidance"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "auv_goal146_guidance_profiles.v1",
        "base_config": str(BASE_CONFIG.relative_to(REPO_ROOT)),
        "base_sha256": _sha256(BASE_CONFIG),
        "frozen_before_measurement": True,
        "arms": {},
    }

    for name, changes in ARMS.items():
        unknown = set(changes) - set(base_guidance)
        if unknown:
            raise ValueError(f"unknown guidance keys for {name}: {sorted(unknown)}")
        payload = deepcopy(base)
        for key, value in changes.items():
            payload["guidance"][key] = value
        output = OUTPUT_DIR / f"guidance_{name}.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["arms"][name] = {
            "path": str(output.relative_to(REPO_ROOT)),
            "sha256": _sha256(output),
            "changes": changes,
            "guidance": payload["guidance"],
        }

    manifest_path = OUTPUT_DIR / "guidance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    for name, entry in manifest["arms"].items():
        print(f"{name}: {entry['path']} {entry['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
