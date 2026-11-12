#!/usr/bin/env python3
"""Build frozen Goal148 K4 stress profiles from the strict Goal134 stressors."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/lowlevel/goal136/curriculum/C0.json"
STRESS_ROOT = REPO_ROOT / "configs/lowlevel/goal134"
OUTPUT_DIR = REPO_ROOT / "configs/lowlevel/goal148"

LEVELS = ("nominal", "current15", "current20", "plant20", "combined20")
PLANT_LEVELS = {"plant20", "combined20"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build(level: str, base: dict[str, object]) -> dict[str, object]:
    payload = deepcopy(base)
    if level != "nominal":
        stress = _load(STRESS_ROOT / f"pc_goal134_{level}_strict_20261111.json")
        payload["current"] = deepcopy(stress["current"])
        if level in PLANT_LEVELS:
            if "plant_randomization" not in stress:
                raise ValueError(f"{level} stress config has no plant_randomization")
            payload["plant_randomization"] = deepcopy(stress["plant_randomization"])

    payload["experiment"] = deepcopy(payload["experiment"])
    experiment = payload["experiment"]
    if not isinstance(experiment, dict):
        raise TypeError("experiment must be a mapping")
    experiment["name"] = f"pc_goal148_k4_{level}_strict_20261111"
    experiment["output"] = (
        f"artifacts/lowlevel/goal148_k4_terminal_pitch_stress_20261111/{level}"
    )

    _validate(level, payload, base)
    return payload


def _validate(level: str, payload: dict[str, object], base: dict[str, object]) -> None:
    plant = payload["plant"]
    if not isinstance(plant, dict) or float(plant["goal_radius_m"]) != 1.25:
        raise ValueError(f"{level}: goal_radius must stay 1.25")
    for section in ("reward", "termination", "guidance", "base_controller"):
        if payload[section] != base[section]:
            raise ValueError(f"{level}: {section} must be unchanged from C0")
    if payload["curriculum"] != base["curriculum"]:
        raise ValueError(f"{level}: curriculum must be unchanged from C0")

    current = payload["current"]
    if not isinstance(current, dict):
        raise TypeError(f"{level}: current must be a mapping")
    speed = current.get("speed_mps")
    if not (isinstance(speed, list) and len(speed) == 2):
        raise ValueError(f"{level}: current.speed_mps must be a 2-item list")
    expected = {
        "nominal": [0.25, 0.45],
        "current15": [0.375, 0.675],
        "current20": [0.5, 0.9],
        "plant20": [0.25, 0.45],
        "combined20": [0.5, 0.9],
    }[level]
    if [float(v) for v in speed] != expected:
        raise ValueError(f"{level}: unexpected current.speed_mps {speed}")
    if level in PLANT_LEVELS:
        if "plant_randomization" not in payload:
            raise ValueError(f"{level}: missing plant_randomization")
        plant_randomization = payload["plant_randomization"]
        if not isinstance(plant_randomization, dict) or not plant_randomization.get("enabled"):
            raise ValueError(f"{level}: plant_randomization must be enabled")
    elif "plant_randomization" in payload:
        raise ValueError(f"{level}: plant_randomization must be absent")


def main() -> int:
    base = _load(BASE_CONFIG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "auv_goal148_k4_stress_profiles.v1",
        "base_config": str(BASE_CONFIG.relative_to(REPO_ROOT)),
        "base_sha256": _sha256(BASE_CONFIG),
        "frozen_before_measurement": True,
        "goal_radius_m": 1.25,
        "levels": {},
    }
    for level in LEVELS:
        payload = _build(level, base)
        path = OUTPUT_DIR / f"env_{level}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["levels"][level] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "current_speed_mps": payload["current"]["speed_mps"],
            "plant_randomization": payload.get("plant_randomization"),
        }
    manifest_path = OUTPUT_DIR / "profile_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    for level, entry in manifest["levels"].items():
        print(level, entry["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
