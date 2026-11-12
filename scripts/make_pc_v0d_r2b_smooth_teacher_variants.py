#!/usr/bin/env python3
"""Generate non-destructive V-0D R2B smooth-teacher variants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

BASE_CONFIG = Path(
    "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json"
)
OUTPUT_DIR = Path("configs/lowlevel")
SUFFIX = "20261107"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant(
    base: dict[str, Any],
    *,
    tag: str,
    guidance: dict[str, Any],
    goal_radius_m: float,
) -> dict[str, Any]:
    data = copy.deepcopy(base)
    name = f"pc_c3_random_realistic_v0d_teacher_r2b_smooth_{tag}_{SUFFIX}"
    if goal_radius_m == 1.25:
        name += "_strict125"
    data["experiment"]["name"] = name
    data["experiment"]["output"] = f"artifacts/lowlevel/{name}"
    data["plant"]["goal_radius_m"] = float(goal_radius_m)
    data["guidance"].update(guidance)
    if data["guidance"].get("compensate_current") is not False:
        raise ValueError("smooth teacher must keep guidance.compensate_current=false")
    return data


def _write(data: dict[str, Any], path: Path, *, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _guidance_variants() -> dict[str, dict[str, Any]]:
    return {
        "h15": {"horizontal_gain": 1.5, "vertical_gain": 1.0},
        "h20": {"horizontal_gain": 2.0, "vertical_gain": 0.8},
        "rate": {
            "heading_rate_limit_rad_s": 0.35,
            "pitch_rate_limit_rad_s": 0.18,
        },
        "gov": {
            "saturation_governor_enabled": True,
            "saturation_governor_start": 0.25,
            "saturation_governor_span": 0.3,
            "saturation_governor_max_reduction": 0.12,
            "saturation_governor_min_scale": 0.88,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = _load(args.base)
    manifest: dict[str, Any] = {
        "schema_version": "auv_pc_v0d_r2b_smooth_teacher_variants.v1",
        "base_config": str(args.base.resolve()),
        "base_config_sha256": _sha256(args.base),
        "suffix": SUFFIX,
        "variants": {},
    }
    for tag, overlay in _guidance_variants().items():
        for goal_radius_m, radius_tag in ((2.0, "default"), (1.25, "strict125")):
            filename = (
                f"odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_"
                f"smooth_{tag}_{SUFFIX}"
            )
            if goal_radius_m == 1.25:
                filename += "_strict125"
            path = args.output_dir / f"{filename}.json"
            data = _variant(
                base,
                tag=tag,
                guidance=overlay,
                goal_radius_m=goal_radius_m,
            )
            digest = _write(data, path, overwrite=args.overwrite)
            manifest["variants"][f"{tag}_{radius_tag}"] = {
                "path": str(path.resolve()),
                "sha256": digest,
                "guidance_overrides": overlay,
                "goal_radius_m": goal_radius_m,
            }
            print(f"wrote {path} sha256={digest}")
    manifest_path = args.output_dir / f"pc_v0d_r2b_smooth_teacher_variants_{SUFFIX}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
