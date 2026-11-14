#!/usr/bin/env python3
"""Fail closed when Goal175 A1 changes its frozen actor, route, or ALOS contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from auv_lowlevel.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "configs/governance/goal175_matrix.json"
ARMS_PATH = REPO_ROOT / "configs/lowlevel/goal175/a1_arms.json"
DEFAULT_CONFIG_ROOT = (
    REPO_ROOT / "artifacts/lowlevel/goal175_alos_comp_mlp_20261114/configs"
)
EXPECTED_PRESSURES = {"nominal", "current10", "noise_n2", "plant20"}
EXPECTED_RADII = {"default", "strict125"}
EXPECTED_MODES = {"ALOS-COMP", "ALOS-NONE"}
EXPECTED_ARM_MODES = {
    "P_RUD_075_COMP": "ALOS-COMP",
    "P_RUD_075_NONE": "ALOS-NONE",
    "zero_bc50": "ALOS-COMP",
    "zero_alos_comp": "ALOS-COMP",
    "zero_alos_none": "ALOS-NONE",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("goal_id") != "Goal175":
        raise ValueError("goal175 matrix has the wrong goal_id")
    arms = matrix.get("arms", [])
    levels = matrix.get("levels", [])
    radii = matrix.get("radii", {})
    baselines = matrix.get("baselines", [])
    expected = {
        "policy": len(arms) * len(matrix.get("checkpoint_seeds", [])) * len(levels) * len(radii),
        "zero": len(baselines) * len(levels) * len(radii),
    }
    expected["total"] = expected["policy"] + expected["zero"]
    if expected != matrix.get("expected_cells"):
        raise ValueError(f"Goal175 matrix cell arithmetic drifted: {expected!r}")
    if expected != {"policy": 48, "zero": 24, "total": 72}:
        raise ValueError(f"Goal175 formal matrix must stay at 72 cells: {expected!r}")
    if set(levels) != EXPECTED_PRESSURES or set(radii) != EXPECTED_RADII:
        raise ValueError("Goal175 level/radius axes drifted")
    if matrix.get("heldout", {}).get("enabled") is not False:
        raise ValueError("Goal175 held-out must remain closed")


def _check_reference(payload: dict[str, Any], *, source: Path) -> None:
    reference = payload.get("reference", {})
    frozen = {
        "path_family": None,
        "path_family_mix": None,
        "generator": "sinusoidal_3d_arclength",
        "length_m": 35.0,
        "speed_mps": 1.2,
        "path_bank_seed": 41,
        "path_bank_size": 32,
    }
    for key, value in frozen.items():
        if reference.get(key) != value:
            raise ValueError(f"{source}: reference.{key} drifted: {reference.get(key)!r} != {value!r}")
    guidance = payload.get("guidance", {})
    if guidance.get("terminal_alignment_min_m") != 50.0:
        raise ValueError(f"{source}: terminal_alignment_min_m must remain 50.0")
    if guidance.get("terminal_alignment_max_m") != 50.0:
        raise ValueError(f"{source}: terminal_alignment_max_m must remain 50.0")


def _check_mode(payload: dict[str, Any], mode: str, *, source: Path) -> None:
    guidance = payload.get("guidance", {})
    observed = (
        guidance.get("compensate_current"),
        guidance.get("current_crab_enabled"),
        guidance.get("current_compensation_source"),
    )
    expected = {
        "ALOS-COMP": (True, False, "estimated"),
        "ALOS-NONE": (False, False, "none"),
    }[mode]
    if observed != expected:
        raise ValueError(f"{source}: ALOS mode drifted: {observed!r} != {expected!r}")
    observation = payload.get("observation", {})
    if observation.get("current_contract") != "hidden" or observation.get("expose_current"):
        raise ValueError(f"{source}: true-current observation leak")
    config = load_config(source)
    if config.guidance.compensate_current and config.guidance.current_compensation_source != "estimated":
        raise ValueError(f"{source}: COMP is not using estimated current")


def _check_a1_configs(config_root: Path) -> dict[str, Any]:
    manifest_path = config_root / "a1_manifest.json"
    manifest = _load(manifest_path)
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 8:
        raise ValueError(f"{manifest_path}: expected 8 pressure x radius scenarios")
    seen: set[tuple[str, str]] = set()
    for scenario in scenarios:
        key = (str(scenario["pressure"]), str(scenario["radius"]))
        if key in seen:
            raise ValueError(f"{manifest_path}: duplicate scenario axis {key}")
        seen.add(key)
        configs = scenario.get("configs", {})
        if set(configs) != EXPECTED_MODES:
            raise ValueError(f"{manifest_path}: {scenario['scenario']} is missing a mode config")
        for mode in sorted(EXPECTED_MODES):
            path = _resolve(configs[mode])
            if not path.is_file():
                raise ValueError(f"{path}: missing ALOS mode config")
            if _sha256(path) != scenario["sha256"][mode]:
                raise ValueError(f"{path}: config SHA drifted from the A1 manifest")
            payload = _load(path)
            _check_reference(payload, source=path)
            _check_mode(payload, mode, source=path)
    if seen != {(pressure, radius) for pressure in EXPECTED_PRESSURES for radius in EXPECTED_RADII}:
        raise ValueError(f"{manifest_path}: A1 scenario grid is incomplete: {sorted(seen)}")
    axes = manifest.get("axes", {})
    if set(axes.get("alos_current_modes", [])) != EXPECTED_MODES:
        raise ValueError(f"{manifest_path}: ALOS mode axis drifted")
    if axes.get("terminal_alignment_m") != 50.0:
        raise ValueError(f"{manifest_path}: terminal alignment must remain 50 m")
    return manifest


def _check_a1_arms(
    arms_path: Path,
    *,
    allow_missing_checkpoint: bool,
    checkpoint_override: Path | None = None,
    training_config_override: Path | None = None,
) -> None:
    payload = _load(arms_path)
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != 5:
        raise ValueError(f"{arms_path}: A1 requires exactly five arms")
    observed = {str(arm.get("tag")): str(arm.get("alos_current_mode")) for arm in arms}
    if observed != EXPECTED_ARM_MODES:
        raise ValueError(f"{arms_path}: arm/mode matrix drifted: {observed!r}")
    training_config = (
        _resolve(payload["training_config"])
        if training_config_override is None
        else _resolve(training_config_override)
    )
    checkpoint = (
        _resolve(payload["checkpoint"])
        if checkpoint_override is None
        else _resolve(checkpoint_override)
    )
    if not training_config.is_file():
        raise ValueError(f"{training_config}: missing training config")
    if not checkpoint.is_file():
        if allow_missing_checkpoint:
            print(f"WARNING: checkpoint is not present locally: {checkpoint}")
            return
        raise ValueError(f"{checkpoint}: missing Goal173 checkpoint")
    if _sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError(f"{checkpoint}: checkpoint SHA drifted")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from auv_lowlevel.batch.checkpoint import read_pc_checkpoint_metadata

    metadata_hash = read_pc_checkpoint_metadata(state)["training_config_sha256"]
    if metadata_hash != _sha256(training_config):
        raise ValueError(f"{checkpoint}: training-config hash does not match {training_config}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a1",), default="a1")
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--arms", type=Path, default=ARMS_PATH)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument("--allow-missing-checkpoint", action="store_true")
    args = parser.parse_args()
    matrix = _resolve(args.matrix)
    arms = _resolve(args.arms)
    config_root = _resolve(args.config_root)
    _check_matrix(_load(matrix))
    manifest = _check_a1_configs(config_root)
    _check_a1_arms(
        arms,
        allow_missing_checkpoint=args.allow_missing_checkpoint,
        checkpoint_override=args.checkpoint,
        training_config_override=args.training_config,
    )
    print(
        "Goal175 A1 protocol OK: "
        f"{manifest['scenario_count']} scenarios x 5 arms = 40 cells; "
        "Goal173 checkpoint, ALOS-COMP/NONE, 50 m guidance, and held-out lock frozen"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
