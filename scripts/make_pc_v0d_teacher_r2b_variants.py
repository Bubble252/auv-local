from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

V0D_CONFIG = Path(
    "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json"
)
R2B_CONFIG = Path("configs/lowlevel/odarrl_control_alos_r2b_crab_floor_init100.json")
OUTPUT_DIR = Path("configs/lowlevel")

def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _teacher_variant(
    r2b: dict[str, Any],
    *,
    name: str,
    output: str,
    goal_radius_m: float = 2.0,
    current_crab_enabled: bool = True,
    route_speed_floor_enabled: bool = True,
    robust_terminal_enabled: bool = True,
) -> dict[str, Any]:
    data = copy.deepcopy(r2b)
    guidance = data["guidance"]
    guidance["current_crab_enabled"] = bool(current_crab_enabled)
    guidance["route_speed_floor_enabled"] = bool(route_speed_floor_enabled)
    guidance["robust_terminal_enabled"] = bool(robust_terminal_enabled)
    guidance["compensate_current"] = False

    # The ALOS teacher must use the ALOS command path.  V0-D's path
    # command path (_path_command) deliberately ignores all ALOS guidance
    # fields, so combining it with R2B guidance would silently reproduce R0.
    data["experiment"]["pc_control_mode"] = "alos"
    data["plant"]["goal_radius_m"] = float(goal_radius_m)
    data["experiment"]["name"] = name
    data["experiment"]["output"] = output
    return data


def _r0_strict(base: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(base)
    data["plant"]["goal_radius_m"] = 1.25
    data["experiment"]["name"] = "pc_c3_random_realistic_v0d_r0_current_strict125"
    data["experiment"]["output"] = "artifacts/lowlevel/pc_v0d_r0_current_strict125"
    return data


def _write(data: dict[str, Any], path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path} sha256={_sha256(path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite files produced by this generator; never use for unrelated configs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = _load(V0D_CONFIG)
    r2b = _load(R2B_CONFIG)

    variants = {
        "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json": _teacher_variant(
            r2b,
            name="pc_c3_random_realistic_v0d_teacher_r2b_current",
            output="artifacts/lowlevel/pc_v0d_teacher_r2b_current",
        ),
        "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_strict125.json": _teacher_variant(
            r2b,
            name="pc_c3_random_realistic_v0d_teacher_r2b_current_strict125",
            output="artifacts/lowlevel/pc_v0d_teacher_r2b_current_strict125",
            goal_radius_m=1.25,
        ),
        "odarrl_control_pc_c3_random_realistic_v0d_r0_current_strict125.json": _r0_strict(
            base
        ),
        "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_nocrab.json": _teacher_variant(
            r2b,
            name="pc_c3_random_realistic_v0d_teacher_r2b_current_nocrab",
            output="artifacts/lowlevel/pc_v0d_teacher_r2b_current_nocrab",
            current_crab_enabled=False,
        ),
        "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_nofloor.json": _teacher_variant(
            r2b,
            name="pc_c3_random_realistic_v0d_teacher_r2b_current_nofloor",
            output="artifacts/lowlevel/pc_v0d_teacher_r2b_current_nofloor",
            route_speed_floor_enabled=False,
        ),
        "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_noboth.json": _teacher_variant(
            r2b,
            name="pc_c3_random_realistic_v0d_teacher_r2b_current_noboth",
            output="artifacts/lowlevel/pc_v0d_teacher_r2b_current_noboth",
            current_crab_enabled=False,
            route_speed_floor_enabled=False,
        ),
        "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_noterminal.json": _teacher_variant(
            r2b,
            name="pc_c3_random_realistic_v0d_teacher_r2b_current_noterminal",
            output="artifacts/lowlevel/pc_v0d_teacher_r2b_current_noterminal",
            robust_terminal_enabled=False,
        ),
        "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_alos_plain.json": _teacher_variant(
            r2b,
            name="pc_c3_random_realistic_v0d_teacher_r2b_current_alos_plain",
            output="artifacts/lowlevel/pc_v0d_teacher_r2b_current_alos_plain",
            current_crab_enabled=False,
            route_speed_floor_enabled=False,
            robust_terminal_enabled=False,
        ),
    }

    for filename, data in variants.items():
        _write(data, args.output_dir / filename, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
