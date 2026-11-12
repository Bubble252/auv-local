#!/usr/bin/env python3
"""Frozen Goal172 diagnostic matrix and artifact paths.

The matrix is intentionally code-local rather than an optional CLI.  Goal172 is
eval-only, but it still has to cover every current level and both radii.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LEVELS = ("nominal", "current07", "current10", "current15")
RADII = ("default", "strict125")
RADIUS_VALUES = {"default": 2.0, "strict125": 1.25}
SEED_START = 20
SEED_COUNT = 40
POLICY_SEED = 86201
DEFAULT_FORMAL_ROOT = Path(
    "artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112"
)
DEFAULT_DIAGNOSTIC_ROOT = Path(
    "artifacts/lowlevel/goal172_residual_attribution_full_20261112"
)

PPO_SCALE_ARMS = ("R1_BC_PPO", "R1_ALOS_PPO")
SAC_SCALE_ARMS = ("R1_BC_SAC", "R1_ALOS_SAC")
PPO_CHANNEL_ARMS = ("R1_BC_PPO", "R1_ALOS_PPO")
CHANNEL_GAINS = {
    "rudder_only": "1.0,0.000001,0.000001",
    "stern_only": "0.000001,1.0,0.000001",
    "rpm_only": "0.000001,0.000001,1.0",
}
REWARD_ENVIRONMENTS = ("old_route", "route_health")


@dataclass(frozen=True)
class DiagnosticJob:
    kind: str
    arm: str
    radius: str
    tag: str
    value: str
    level: str

    @property
    def is_reward(self) -> bool:
        return self.kind == "reward"

    def output_path(self, root: Path) -> Path:
        if self.is_reward:
            return (
                root
                / "reward_attribution"
                / self.arm
                / self.radius
                / self.level
                / f"zero_seed{POLICY_SEED}"
            )
        prefix = {
            "ppo": "screen",
            "sac": "screen",
            "channel": "screen_channel",
            "rudder_scale": "screen_rudder",
        }[self.kind]
        return (
            root
            / prefix
            / self.arm
            / self.radius
            / self.tag
            / self.level
            / f"policy_seed{POLICY_SEED}"
        )

    def as_tsv(self) -> str:
        return "\t".join(
            (self.kind, self.arm, self.radius, self.tag, self.value, self.level)
        )


def diagnostic_jobs() -> list[DiagnosticJob]:
    jobs: list[DiagnosticJob] = []
    for arm in PPO_SCALE_ARMS:
        for radius in RADII:
            for scale in ("0.50", "1.25", "2.50"):
                tag = f"x{scale.replace('.', 'p')}"
                for level in LEVELS:
                    jobs.append(DiagnosticJob("ppo", arm, radius, tag, scale, level))
    for arm in SAC_SCALE_ARMS:
        for radius in RADII:
            for multiplier in ("2", "5", "10"):
                tag = f"x{multiplier}"
                for level in LEVELS:
                    jobs.append(
                        DiagnosticJob("sac", arm, radius, tag, multiplier, level)
                    )
    for arm in PPO_CHANNEL_ARMS:
        for radius in RADII:
            for channel, gains in CHANNEL_GAINS.items():
                for level in LEVELS:
                    jobs.append(
                        DiagnosticJob("channel", arm, radius, channel, gains, level)
                    )
    for radius in RADII:
        for scale in ("0.375", "0.75", "1.0"):
            tag = f"x{scale.replace('.', 'p')}"
            for level in LEVELS:
                jobs.append(
                    DiagnosticJob(
                        "rudder_scale", "R1_BC_PPO", radius, tag, scale, level
                    )
                )
    for environment in REWARD_ENVIRONMENTS:
        for radius in RADII:
            for level in LEVELS:
                jobs.append(
                    DiagnosticJob("reward", environment, radius, "zero", "-", level)
                )
    expected = 96 + 48 + 24 + 16
    if len(jobs) != expected:
        raise AssertionError(f"Goal172 job matrix has {len(jobs)} jobs, expected {expected}")
    return jobs


def formal_baseline_path(
    formal_root: Path, arm: str, radius: str, level: str
) -> Path:
    return (
        formal_root
        / "eval"
        / arm
        / radius
        / level
        / f"policy_seed{POLICY_SEED}"
    )


def formal_zero_path(formal_root: Path, zero_arm: str, radius: str, level: str) -> Path:
    return formal_root / "eval" / zero_arm / radius / level / f"zero_seed{POLICY_SEED}"

