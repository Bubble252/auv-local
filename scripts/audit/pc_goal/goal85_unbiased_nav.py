#!/usr/bin/env python3
"""Audit the Goal85 unbiased-noise sensor contract without stepping a plant.

The suite is driven with a fixed true state so any systematic ``p_nav - eta``
mean, trend or delay is attributable to the sensor chain, not to control.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from auv_lowlevel.batch.sensors import PROFILES, SyntheticSensorSuite
from auv_lowlevel.config import load_config

SCHEMA_VERSION = "auv_pc_goal85_unbiased_nav_audit.v1"
SEGMENTS = 10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"
        ),
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _profile_audit(profile_name: str) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    checks = {
        "nav_bias_zero": profile.nav_bias_m == (0.0, 0.0),
        "nav_drift_zero": profile.nav_drift_sigma_m == (0.0, 0.0),
        "nav_usbl_gain_zero": profile.nav_usbl_gain == 0.0,
        "nav_outage_drift_zero": profile.nav_dvl_outage_drift_mps == (0.0, 0.0),
        "nav_delay_zero": profile.nav_delay_s == (0.0, 0.0),
        "calibration_zero": profile.nav_bias_calibration_fraction == 0.0,
        "nav_noise_positive": profile.nav_noise_m[0] > 0.0,
    }
    delay_fields = (
        "dvl_bottom_delay_s",
        "dvl_water_delay_s",
        "imu_att_delay_s",
        "imu_gyro_delay_s",
        "usbl_delay_s",
        "current_est_delay_s",
    )
    for field in delay_fields:
        checks[field] = getattr(profile, field) == (0.0, 0.0)
    return {
        "profile": profile_name,
        "nav_noise_m": list(profile.nav_noise_m),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_audit(
    *,
    config_path: Path,
    steps: int,
    envs: int,
    seed_start: int,
) -> dict[str, Any]:
    if steps <= 0 or envs <= 0:
        raise ValueError("steps and envs must be positive")
    config = load_config(config_path)
    if config.sensors is None:
        raise ValueError("config has no sensors block")
    profile_name = config.sensors.profile
    profile_audit = _profile_audit(profile_name)
    suite = SyntheticSensorSuite(config, num_envs=envs, dtype=torch.float64)
    seeds = [seed_start + index for index in range(envs)]
    suite.reset(seeds)
    eta = torch.zeros((envs, 6), dtype=torch.float64)
    eta[:, :3] = torch.tensor([2.0, -1.0, 10.0], dtype=torch.float64)
    eta[:, 5] = 0.3
    nu = torch.tensor([1.0, 0.05, -0.02, 0.01, -0.005, 0.02], dtype=torch.float64)
    nu = nu.expand(envs, 6).clone()
    current = torch.zeros((envs, 3), dtype=torch.float64)

    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    segment = torch.zeros((SEGMENTS, 3), dtype=torch.float64)
    segment_count = torch.zeros(SEGMENTS, dtype=torch.float64)
    max_abs = torch.zeros(3, dtype=torch.float64)
    nav_age_max = 0.0
    for step in range(steps):
        output = suite.step(
            eta_true=eta, nu_true=nu, current_true_ned_mps=current
        )
        error = output.p_nav - eta[:, :3]
        total += error.sum(dim=0)
        total_sq += (error * error).sum(dim=0)
        max_abs = torch.maximum(max_abs, error.abs().amax(dim=0))
        index = min(SEGMENTS - 1, step * SEGMENTS // steps)
        segment[index] += error.sum(dim=0)
        segment_count[index] += envs
        nav_age_max = max(nav_age_max, float(output.nav_age_s.max()))
    count = float(steps * envs)
    mean = total / count
    variance = torch.clamp(total_sq / count - mean * mean, min=0.0)
    std = torch.sqrt(variance)
    segment_mean = segment / segment_count[:, None]
    drift = segment_mean[-1] - segment_mean[0]
    passed = bool(
        float(mean.abs().max()) < 0.05
        and float(drift.abs().max()) < 0.02
        and nav_age_max <= 1e-9
        and profile_audit["passed"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "config": str(config_path),
        "sensor_profile": profile_name,
        "steps": steps,
        "envs": envs,
        "samples": int(steps * envs),
        "mean_m": [float(value) for value in mean],
        "std_m": [float(value) for value in std],
        "max_abs_m": [float(value) for value in max_abs],
        "max_abs_mean_m": float(mean.abs().max()),
        "drift_last_minus_first_m": [float(value) for value in drift],
        "max_abs_drift_m": float(drift.abs().max()),
        "nav_age_max_s": nav_age_max,
        "profile_audit": profile_audit,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_audit(
        config_path=args.config,
        steps=args.steps,
        envs=args.envs,
        seed_start=args.seed_start,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("sensor_profile", "passed")}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
