"""Command-line parsing for the PC PPO evaluator."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from auv_lowlevel.batch.pc_history_ablation import (
    PC_HISTORY_ABLATION_MODES,
    PC_PATH_ABLATION_MODES,
)
from auv_lowlevel.outcome_contract import DEFAULT_SUCCESS_FRAME, SUCCESS_FRAMES


def _parse_residual_channel_gains(value: str) -> tuple[float, float, float]:
    """Parse an eval-only comma-separated (rudder, stern, rpm) gain triple."""

    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "residual channel gains must contain exactly three comma-separated values"
        )
    try:
        gains = tuple(float(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("residual channel gains must be floats") from exc
    if any(not np.isfinite(gain) or gain <= 0.0 for gain in gains):
        raise argparse.ArgumentTypeError("residual channel gains must be finite and positive")
    return gains


def _parse_injected_residual_profile(value: str) -> tuple[str, tuple[float, float, float] | None]:
    """Parse an eval-only residual injection profile."""

    name = value.strip().lower()
    if name in {"zero", "ppo", "neg-ppo"}:
        return name, None
    gain_match = re.fullmatch(r"ppo-(\d+(?:\.\d+)?)x", name)
    if gain_match is not None:
        factor = float(gain_match.group(1))
        if not np.isfinite(factor) or factor <= 0.0:
            raise argparse.ArgumentTypeError("ppo-<N>x gain must be finite and positive")
        return "ppo-gain", (factor, 0.0, 0.0)
    if name.startswith("const:"):
        parts = [item.strip() for item in value.split(":", 1)[1].split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "const injected residual must be const:<rudder>,<stern>,<rpm>"
            )
        try:
            values = tuple(float(item) for item in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "const injected residual values must be floats"
            ) from exc
        if any(not np.isfinite(item) for item in values):
            raise argparse.ArgumentTypeError("const injected residual values must be finite")
        return "const", values
    raise argparse.ArgumentTypeError(
        "injected residual profile must be zero, ppo, neg-ppo, ppo-2x, ppo-4x, "
        "or const:<rudder>,<stern>,<rpm>"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PC20 PPO actor in closed loop")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=None,
        help=(
            "optional PC environment override for clean/realistic paired evaluation; "
            "network, normalization, stage and actuator contracts are still checked"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument(
        "--curriculum-stage",
        choices=("C0", "C1", "C2", "C3", "C4", "C5", "C6"),
        default=None,
        help="evaluate a fixed curriculum stage by sampling its episode parameters",
    )
    parser.add_argument(
        "--disable-curriculum-direct-command",
        action="store_true",
        help=(
            "diagnostic-only: keep the curriculum episode parameters but use ALOS "
            "instead of C0/C1 random direct commands"
        ),
    )
    parser.add_argument(
        "--curriculum-current-speed-range-mps",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help=(
            "eval-only stress override for the sampled curriculum current speed; "
            "requires --curriculum-stage and keeps depth/path/noise/speed unchanged"
        ),
    )
    parser.add_argument(
        "--curriculum-shear-scale",
        type=float,
        default=None,
        help=(
            "eval-only stress override for the sampled curriculum shear scale; "
            "requires --curriculum-stage and keeps depth/path/noise/speed unchanged"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--history-ablation",
        choices=PC_HISTORY_ABLATION_MODES,
        default="none",
        help="evaluation-only perturbation of the actor history layout",
    )
    parser.add_argument(
        "--path-ablation",
        choices=PC_PATH_ABLATION_MODES,
        default="none",
        help="evaluation-only perturbation of the actor path block before its encoder",
    )
    parser.add_argument(
        "--action-mode",
        choices=(
            "policy",
            "frozen-base",
            "zero-command-residual",
            "zero-actuator-residual",
            "zero-residual",
            "gated-residual",
            "attenuated-residual",
            "injected-residual",
            "alos",
        ),
        default="policy",
        help=(
            "policy uses the full actor; frozen-base uses only the frozen BC "
            "action for a paired baseline; zero-command-residual feeds a zero "
            "residual to the C1 ALOS+PD/PI wrapper (B0 baseline); "
            "zero-actuator-residual does the same for actuator-space arms"
        ),
    )
    parser.add_argument(
        "--injected-residual-profile",
        type=_parse_injected_residual_profile,
        default=("zero", None),
        help=(
            "eval-only residual injection: zero, ppo, neg-ppo, ppo-<N>x, "
            "or const:<rudder>,<stern>,<rpm>"
        ),
    )
    parser.add_argument(
        "--success-frame",
        choices=SUCCESS_FRAMES,
        default=DEFAULT_SUCCESS_FRAME,
        help=(
            "Eval-only success reporting frame; eta uses tight-radius ever-capture "
            "and never changes rollout, reward, or termination"
        ),
    )
    parser.add_argument(
        "--terminal-gate-enter-m",
        type=float,
        default=20.0,
        help="eval-only gate: residual is zero beyond this distance",
    )
    parser.add_argument(
        "--terminal-gate-full-m",
        type=float,
        default=5.0,
        help="eval-only gate: residual is fully open inside this distance",
    )
    parser.add_argument(
        "--allow-stage-override",
        action="store_true",
        help="allow an explicit C3<->C4 cross-stage evaluation",
    )
    parser.add_argument(
        "--allow-guidance-rate-limit-override",
        action="store_true",
        help=(
            "allow an explicit evaluation-only change to guidance "
            "command_rate_limit_enabled and pitch_rate_limit_rad_s only"
        ),
    )
    parser.add_argument(
        "--allow-sensor-override",
        action="store_true",
        help=(
            "allow an explicit evaluation-only sensor-profile override; "
            "the report records both base and override sensor profiles"
        ),
    )
    parser.add_argument(
        "--allow-diversity-environment-override",
        action="store_true",
        help=(
            "allow evaluation-only reference geometry/jitter changes used by "
            "Goal136 D-line arms; reward, termination, guidance, observation, "
            "normalization, actuator, residual, domain, plant and sensor "
            "contracts remain locked"
        ),
    )
    parser.add_argument(
        "--allow-start-progress-training-override",
        action="store_true",
        help=(
            "allow a training-config reference.start_progress_m > 0 to be "
            "evaluated from the full path start (override start_progress_m=0); "
            "all other reference fields remain locked"
        ),
    )
    parser.add_argument(
        "--allow-actor-observation-override",
        action="store_true",
        help=(
            "allow switching only observation.ideal_actor_observations for paired "
            "ideal/nominal evaluation of the same policy"
        ),
    )
    parser.add_argument(
        "--guidance-config",
        type=Path,
        default=None,
        help=(
            "optional ALOS-R guidance profile whose guidance block replaces the "
            "environment config guidance while retaining the environment/stage"
        ),
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=None,
        help=(
            "optional eval-only controller profile; only pitch_gain and "
            "pitch_rate_gain may differ from the environment config"
        ),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help=(
            "optional .npz path for per-step closed-loop diagnostics; use a single "
            "evaluation seed to keep the trace small"
        ),
    )
    parser.add_argument(
        "--trace-sensor-ages",
        action="store_true",
        help=(
            "with --trace-output, also record nav/usbl/dvl sensor ages and USBL "
            "validity; default trace fields remain unchanged"
        ),
    )
    parser.add_argument(
        "--trace-estimator-state",
        action="store_true",
        help=(
            "with --trace-output, record realized nav OU/event, USBL innovations "
            "and DVL hold drift; default trace fields remain unchanged"
        ),
    )
    parser.add_argument(
        "--trace-reward-breakdown",
        action="store_true",
        help=(
            "with --trace-output, record every PC reward component as reward_<name>; "
            "default trace fields remain unchanged"
        ),
    )
    parser.add_argument(
        "--actuator-residual-scale-override",
        type=float,
        default=None,
        help=(
            "evaluation-only multiplier for the actuator residual; does not alter "
            "training, checkpoint weights, or the recorded training-config hash"
        ),
    )
    parser.add_argument(
        "--actuator-residual-channel-gains",
        type=_parse_residual_channel_gains,
        default=None,
        help=(
            "evaluation-only per-channel multipliers for actuator residual "
            "delta, ordered rudder,stern,rpm; default is 1,1,1"
        ),
    )
    parser.add_argument(
        "--actuator-residual-scale-near",
        type=float,
        default=None,
        help=("evaluation-only residual scale inside --actuator-residual-scale-switch-distance-m"),
    )
    parser.add_argument(
        "--actuator-residual-scale-far",
        type=float,
        default=None,
        help=("evaluation-only residual scale outside --actuator-residual-scale-switch-distance-m"),
    )
    parser.add_argument(
        "--actuator-residual-scale-switch-distance-m",
        type=float,
        default=None,
        help="evaluation-only distance threshold for the near/far residual schedule",
    )
    parser.add_argument(
        "--actuator-residual-near-channel-gains",
        type=_parse_residual_channel_gains,
        default=None,
        help=(
            "evaluation-only per-channel gains inside the near channel-gain "
            "distance, ordered rudder,stern,rpm"
        ),
    )
    parser.add_argument(
        "--actuator-residual-channel-gain-near-distance-m",
        type=float,
        default=None,
        help="evaluation-only distance at or below which near channel gains apply",
    )
    parser.add_argument(
        "--actuator-residual-channel-gain-far-distance-m",
        type=float,
        default=None,
        help="evaluation-only distance at or above which far channel gains apply",
    )
    parser.add_argument(
        "--ppo-residual-attenuation-full-m",
        type=float,
        default=1.25,
        help=(
            "with --action-mode attenuated-residual, fully disable the PPO "
            "increment at or below this distance"
        ),
    )
    parser.add_argument(
        "--ppo-residual-attenuation-enter-m",
        type=float,
        default=3.0,
        help=(
            "with --action-mode attenuated-residual, fully restore the PPO "
            "increment at or above this distance"
        ),
    )
    return parser.parse_args(argv)
