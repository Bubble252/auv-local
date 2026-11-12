#!/usr/bin/env python3
"""Audit teacher actuator saturation on recorded PC demonstrations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.actions import ActuatorLimits
from auv_lowlevel.batch.actions import BatchedPureActionPipeline
from auv_lowlevel.batch.pc_bc import load_pc_demonstration_dataset
from auv_lowlevel.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="NAME=CONFIG=DEMOS",
        help="named environment config and demonstration dataset; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-threshold", type=float, default=0.99)
    return parser.parse_args()


def _parse_cell(value: str) -> tuple[str, Path, Path]:
    fields = value.split("=", 2)
    if len(fields) != 3 or not all(fields):
        raise ValueError(f"invalid --cell value {value!r}; expected NAME=CONFIG=DEMOS")
    name, config, demos = fields
    return name, Path(config), Path(demos)


def _finite_mean(values: torch.Tensor) -> float:
    number = float(values.double().mean().item())
    if not math.isfinite(number):
        raise ValueError("saturation audit produced a non-finite value")
    return number


def _audit_cell(
    name: str,
    config_path: Path,
    demos_path: Path,
    *,
    action_threshold: float,
) -> dict[str, Any]:
    config = load_config(config_path)
    limits = ActuatorLimits(
        max_rudder_angle_rad=config.actuator.policy_max_rudder_angle_rad,
        max_stern_plane_angle_rad=config.actuator.policy_max_stern_plane_angle_rad,
        max_rudder_rate_rad_s=config.actuator.policy_max_rudder_rate_rad_s,
        max_stern_plane_rate_rad_s=config.actuator.policy_max_stern_plane_rate_rad_s,
        max_propeller_rpm=config.actuator.policy_max_propeller_rpm,
        material_tolerance_fraction=config.actuator.material_tolerance_fraction,
    )
    pipeline = BatchedPureActionPipeline(limits, dtype=torch.float64, validate_finite=False)
    dataset, metadata = load_pc_demonstration_dataset(demos_path)
    order = np.lexsort((dataset.step, dataset.episode_id))
    inverse = np.argsort(order)
    episode = dataset.episode_id[order]
    action = torch.as_tensor(dataset.action[order], dtype=torch.float64)
    actuator = torch.as_tensor(dataset.actual_actuator[order], dtype=torch.float64)
    previous = torch.cat([torch.zeros_like(actuator[:1]), actuator[:-1]], dim=0)
    reset = torch.as_tensor(episode[1:] != episode[:-1])
    previous[1:][reset] = 0.0
    with torch.inference_mode():
        applied = pipeline.apply(action, previous, config.plant.dt_s)
    material = applied.material_saturated.double().numpy()[inverse]
    saturated = applied.saturated.double().numpy()[inverse]
    material_magnitude = (
        applied.material_magnitude_clipped.any(dim=1).double().numpy()[inverse]
    )
    material_rate = applied.material_rate_clipped.any(dim=1).double().numpy()[inverse]
    return {
        "config": str(config_path),
        "demos": str(demos_path),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "sample_count": dataset.sample_count,
        "episode_count": dataset.episode_count,
        "material_saturation_rate": float(material.mean()),
        "full_saturation_rate": float(saturated.mean()),
        "material_magnitude_fraction": float(material_magnitude.mean()),
        "material_rate_fraction": float(material_rate.mean()),
        "channel_material_magnitude_fraction": (
            applied.material_magnitude_clipped.double().mean(dim=0).numpy().tolist()
        ),
        "action_abs_mean": action.abs().mean(dim=0).numpy().tolist(),
        "action_abs_p99": action.abs().quantile(0.99, dim=0).numpy().tolist(),
        "actual_abs_mean": actuator.abs().mean(dim=0).numpy().tolist(),
        "action_threshold": float(action_threshold),
        "action_channel_saturation_rate": (
            action.abs() >= action_threshold
        ).double().mean(dim=0).numpy().tolist(),
        "action_any_surface_saturation_rate": float(
            (action[:, :2].abs() >= action_threshold).any(dim=1).double().mean().item()
        ),
    }


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.action_threshold <= 1.0:
        raise ValueError("--action-threshold must lie in (0, 1]")
    cells = [_parse_cell(value) for value in args.cell]
    names = [name for name, _, _ in cells]
    if len(names) != len(set(names)):
        raise ValueError("cell names must be unique")
    missing = [str(path) for _, config, demos in cells for path in (config, demos) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input files: {missing}")
    payload = {
        "schema_version": "auv_pc_teacher_saturation_audit.v1",
        "cells": {
            name: _audit_cell(
                name,
                config,
                demos,
                action_threshold=args.action_threshold,
            )
            for name, config, demos in cells
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
