"""Shared constants and path resolution for the PC PPO evaluator."""

from __future__ import annotations

from pathlib import Path

from auv_lowlevel._paths import resolve_from_root

REPO_ROOT = Path(__file__).resolve().parents[4]
TURN_CURVATURE_THRESHOLD = 0.02
DIVERSITY_REFERENCE_OVERRIDE_FIELDS = (
    "path_bank_size",
    "path_bank_seed",
    "lateral_amplitude_m",
    "lateral_cycles",
    "depth_start_m",
    "depth_slope_m",
    "depth_amplitude_m",
    "depth_cycles",
    "x_span_m",
    "start_jitter_xy_m",
    "start_jitter_z_m",
    "start_jitter_yaw_rad",
)


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)
