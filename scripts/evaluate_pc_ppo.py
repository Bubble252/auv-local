from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.batch.checkpoint import validate_pc_checkpoint_metadata
from auv_lowlevel.batch.geometry import (
    body_to_world,
    ground_velocity_body_for_contract,
    quaternion_from_euler,
)
from auv_lowlevel.batch.pc import PCTerminalKind
from auv_lowlevel.batch.pc_actuator_residual import BatchedPCActuatorResidualEnv
from auv_lowlevel.batch.pc_bc import (
    restore_pc_bc_frozen_current_encoder_alpha,
    restore_pc_bc_frozen_path_encoder_alpha,
)
from auv_lowlevel.batch.pc_command_residual import (
    BatchedPCCommandResidualEnv,
    smooth_terminal_gate,
)
from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.batch.pc_frozen_residual_vec_env import (
    BatchTorchPCFrozenBCResidualVecEnv,
)
from auv_lowlevel.batch.pc_history_ablation import (
    PC_HISTORY_ABLATION_MODES,
    PC_PATH_ABLATION_MODES,
    ablate_pc_history,
    ablate_pc_path,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import PC_PATH_DEFAULT_BLOCKS, PC_PATH_DIM
from auv_lowlevel.batch.pc_policy import PCActorCritic, pc_schema_hash
from auv_lowlevel.batch.pc_vec_env import BatchTorchPCVecEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import (
    CurriculumStageSampler,
    curriculum_episode_metadata,
    default_curriculum_stages,
)
from auv_lowlevel.outcome_contract import (
    DEFAULT_SUCCESS_FRAME,
    SUCCESS_FRAMES,
    score_success_contract,
    summarize_success_contract,
    validate_success_frame,
)
from auv_lowlevel.ppo_config import load_training_config, resolve_pc_action_interface

REPO_ROOT = Path(__file__).resolve().parents[1]
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
        raise argparse.ArgumentTypeError(
            "residual channel gains must be floats"
        ) from exc
    if any(not np.isfinite(gain) or gain <= 0.0 for gain in gains):
        raise argparse.ArgumentTypeError(
            "residual channel gains must be finite and positive"
        )
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
            raise argparse.ArgumentTypeError(
                "const injected residual values must be finite"
            )
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
        help=(
            "evaluation-only residual scale inside "
            "--actuator-residual-scale-switch-distance-m"
        ),
    )
    parser.add_argument(
        "--actuator-residual-scale-far",
        type=float,
        default=None,
        help=(
            "evaluation-only residual scale outside "
            "--actuator-residual-scale-switch-distance-m"
        ),
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


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_trace_sensor_ages(
    trace_output: Path | None, trace_sensor_ages: bool
) -> None:
    if trace_sensor_ages and trace_output is None:
        raise ValueError("--trace-sensor-ages requires --trace-output")


def _validate_trace_estimator_state(
    trace_output: Path | None, trace_estimator_state: bool
) -> None:
    if trace_estimator_state and trace_output is None:
        raise ValueError("--trace-estimator-state requires --trace-output")


def _validate_trace_reward_breakdown(
    trace_output: Path | None, trace_reward_breakdown: bool
) -> None:
    if trace_reward_breakdown and trace_output is None:
        raise ValueError("--trace-reward-breakdown requires --trace-output")


def _validate_actuator_residual_scale_schedule(
    *,
    near: float | None,
    far: float | None,
    switch_distance_m: float | None,
    constant_override: float | None,
) -> None:
    values = (near, far, switch_distance_m)
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise ValueError(
            "actuator residual schedule requires near, far, and switch distance"
        )
    if near is None:
        return
    assert far is not None
    assert switch_distance_m is not None
    if constant_override is not None:
        raise ValueError(
            "actuator residual schedule and constant scale override are mutually exclusive"
        )
    for name, value in (
        ("near", near),
        ("far", far),
        ("switch_distance_m", switch_distance_m),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"actuator residual {name} scale must be finite and positive")


def _validate_actuator_channel_gain_schedule(
    *,
    near_gains: tuple[float, float, float] | None,
    near_distance_m: float | None,
    far_distance_m: float | None,
) -> None:
    values = (near_gains, near_distance_m, far_distance_m)
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise ValueError(
            "actuator channel gain schedule requires near gains, near distance, "
            "and far distance"
        )
    if near_gains is None:
        return
    assert near_distance_m is not None
    assert far_distance_m is not None
    if not np.isfinite(near_distance_m) or not np.isfinite(far_distance_m):
        raise ValueError("actuator channel gain distances must be finite")
    if near_distance_m <= 0.0 or far_distance_m <= near_distance_m:
        raise ValueError(
            "actuator channel gain distances require 0 < near_distance < far_distance"
        )


def _ppo_residual_attenuation(
    distance_m: torch.Tensor,
    *,
    full_m: float,
    enter_m: float,
) -> torch.Tensor:
    """Return 0 at/below ``full_m`` and 1 at/above ``enter_m``.

    This is the inverse of the training terminal gate: near the goal the PPO
    increment is suppressed, while the frozen BC residual remains intact.
    """

    if (
        not np.isfinite(full_m)
        or full_m <= 0.0
        or not np.isfinite(enter_m)
        or enter_m <= full_m
    ):
        raise ValueError(
            "PPO attenuation requires finite positive full_m < enter_m"
        )
    return 1.0 - smooth_terminal_gate(
        distance_m,
        enter_m=enter_m,
        full_m=full_m,
    )


def _reward_breakdown_trace_fields(infos: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in infos.items()
        if name.startswith("reward_") and torch.is_tensor(value)
    }


def _trace_sensor_age_fields(sensor: Any) -> dict[str, Any]:
    return {
        "nav_age_s": sensor.nav_age_s,
        "usbl_age_s": sensor.usbl_age_s,
        "dvl_bottom_age_s": sensor.dvl_bottom_age_s,
        "usbl_valid": sensor.usbl_valid,
        "usbl_outlier": sensor.usbl_outlier,
    }


def _record_trace_sensor_ages(
    record_trace: Any, sensor: Any, *, enabled: bool
) -> None:
    if not enabled:
        return
    for name, value in _trace_sensor_age_fields(sensor).items():
        record_trace(name, value)


def _trace_estimator_state_fields(sensor: Any) -> dict[str, Any]:
    return {
        "nav_ou": sensor.nav_ou,
        "nav_event": sensor.nav_event,
        "nav_event_usbl": sensor.nav_event_usbl,
        "nav_event_dvl": sensor.nav_event_dvl,
        "nav_bias": sensor.nav_bias,
        "usbl_due": sensor.usbl_due,
        "usbl_accepted": sensor.usbl_accepted,
        "usbl_rejected": sensor.usbl_rejected,
        "usbl_innovation": sensor.usbl_innovation,
        "usbl_innovation_applied": sensor.usbl_innovation_applied,
        "usbl_raw": sensor.usbl_raw,
        "usbl_predicted": sensor.usbl_predicted,
        "dvl_bottom_outage": sensor.dvl_bottom_outage,
        "nav_dvl_outage_delta": sensor.nav_dvl_outage_delta,
    }


def _record_trace_estimator_state(
    record_trace: Any, sensor: Any, *, enabled: bool
) -> None:
    if not enabled:
        return
    for name, value in _trace_estimator_state_fields(sensor).items():
        record_trace(name, value)


def _stage_override_allowed(
    checkpoint_stage: str,
    target_stage: str,
    *,
    allow_stage_override: bool,
) -> bool:
    """Return whether an explicit C3/C4 transfer was requested."""

    return (
        allow_stage_override
        and checkpoint_stage != target_stage
        and {checkpoint_stage, target_stage} == {"C3", "C4"}
    )


def _validate_environment_override(
    base: Any,
    override: Any,
    *,
    allow_sensor_override: bool = False,
    allow_actor_observation_override: bool = False,
    allow_guidance_rate_limit_override: bool = False,
    allow_diversity_environment_override: bool = False,
    allow_start_progress_training_override: bool = False,
) -> None:
    """Reject evaluation overrides that change the trained control semantics."""

    if override.pc_control_mode != base.pc_control_mode:
        raise ValueError(
            "environment override changes pc_control_mode: "
            f"{base.pc_control_mode!r} -> {override.pc_control_mode!r}"
        )
    if override.reward != base.reward:
        raise ValueError("environment override changes the PC reward contract")
    guidance_allowed = base.guidance
    if allow_guidance_rate_limit_override:
        guidance_allowed = replace(
            base.guidance,
            command_rate_limit_enabled=override.guidance.command_rate_limit_enabled,
            pitch_rate_limit_rad_s=override.guidance.pitch_rate_limit_rad_s,
        )
    if override.guidance != guidance_allowed:
        raise ValueError("environment override changes the guidance contract")
    observation_allowed = base.observation
    if allow_actor_observation_override:
        observation_allowed = replace(
            base.observation,
            ideal_actor_observations=override.observation.ideal_actor_observations,
        )
    if override.observation != observation_allowed:
        raise ValueError("environment override changes the observation contract")
    if override.normalization != base.normalization:
        raise ValueError("environment override changes normalization scales")
    if override.actuator != base.actuator:
        raise ValueError("environment override changes the actuator contract")
    if override.residual != base.residual:
        raise ValueError("environment override changes the residual contract")
    if override.termination != base.termination:
        raise ValueError("environment override changes the termination contract")
    if override.sensors != base.sensors and not allow_sensor_override:
        raise ValueError("environment override changes the sensor contract")
    if override.domain != base.domain:
        raise ValueError("environment override changes the domain curriculum")
    plant_allowed = replace(base.plant, goal_radius_m=override.plant.goal_radius_m)
    if _dataclass_differs(override.plant, plant_allowed):
        raise ValueError(
            "environment override may change only plant.goal_radius_m"
        )
    reference_allowed = replace(
        base.reference,
        command_speed_range_mps=override.reference.command_speed_range_mps,
    )
    if allow_diversity_environment_override:
        reference_allowed = replace(
            reference_allowed,
            **{
                field: getattr(override.reference, field)
                for field in DIVERSITY_REFERENCE_OVERRIDE_FIELDS
            },
        )
    if (
        not allow_start_progress_training_override
        and (
            base.reference.start_progress_m != override.reference.start_progress_m
            or base.reference.start_progress_full_fraction
            != override.reference.start_progress_full_fraction
        )
    ):
        raise ValueError(
            "environment override changes reference.start_progress_m or "
            "reference.start_progress_full_fraction; pass "
            "--allow-start-progress-training-override only for a training-config "
            "start with an evaluation start of 0"
        )
    if allow_start_progress_training_override:
        if (
            override.reference.start_progress_m != 0.0
            or override.reference.start_progress_full_fraction != 0.0
        ):
            raise ValueError(
                "evaluation override must keep reference.start_progress_m=0 and "
                "reference.start_progress_full_fraction=0; only the training "
                "config may mix in near-goal starts"
            )
        reference_allowed = replace(
            reference_allowed,
            start_progress_m=0.0,
            start_progress_full_fraction=0.0,
        )
    if override.reference != reference_allowed:
        if allow_diversity_environment_override:
            raise ValueError(
                "environment override may change only "
                "reference.command_speed_range_mps and the explicit diversity "
                "reference geometry/jitter fields"
            )
        if allow_start_progress_training_override:
            raise ValueError(
                "environment override may change only the training-config "
                "start-progress fields; all other reference fields are locked"
            )
        raise ValueError(
            "environment override may change only reference.command_speed_range_mps"
        )


def _validate_controller_pitch_override(base: Any, override: Any) -> None:
    """Allow only pitch-loop gains to differ for an eval-only authority probe."""

    allowed = replace(
        base,
        pitch_gain=override.pitch_gain,
        pitch_rate_gain=override.pitch_rate_gain,
    )
    if _dataclass_differs(override, allowed):
        raise ValueError(
            "controller override may change only pitch_gain and pitch_rate_gain"
        )


def _dataclass_differs(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return True
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            if not np.array_equal(left_value, right_value):
                return True
        elif left_value != right_value:
            return True
    return False


def _final_path_blocks(training: Any) -> tuple[str, ...]:
    if training.pc is None or training.pc.path_dim == 0:
        return ()
    if not training.pc.path_curriculum:
        return tuple(PC_PATH_DEFAULT_BLOCKS)
    return tuple(
        block for block in training.pc.path_curriculum[-1][1] if block != "state_base"
    )


def _uses_frozen_residual_wrapper(
    *, base_mode: str, base_output_mode: str, action_mode: str
) -> bool:
    """Return whether evaluation must compose frozen BC with the PPO increment.

    ``alos`` is the unambiguous traditional-controller baseline: it bypasses
    the frozen-BC wrapper entirely and sends an all-zero residual to the
    actuator wrapper, whose base action is ALOS + PD/PI.
    """

    if action_mode == "alos":
        return False
    return (
        base_mode == "frozen_pc_bc_residual"
        and base_output_mode == "residual"
    )


def _new_episode_metrics(count: int) -> dict[str, Any]:
    return {
        "active_steps": np.zeros(count, dtype=np.int64),
        "reward_sum": np.zeros(count, dtype=np.float64),
        "abs_yaw_error_sum": np.zeros(count, dtype=np.float64),
        "abs_pitch_error_sum": np.zeros(count, dtype=np.float64),
        "abs_speed_error_sum": np.zeros(count, dtype=np.float64),
        "yaw_error_sq_sum": np.zeros(count, dtype=np.float64),
        "pitch_error_sq_sum": np.zeros(count, dtype=np.float64),
        "speed_error_sq_sum": np.zeros(count, dtype=np.float64),
        "abs_roll_max": np.zeros(count, dtype=np.float64),
        "abs_roll_sum": np.zeros(count, dtype=np.float64),
        "roll_abs_values": [],
        "roll_rate_abs_values": [],
        "path_steps": np.zeros(count, dtype=np.int64),
        "horizontal_error_sq_sum": np.zeros(count, dtype=np.float64),
        "vertical_error_sq_sum": np.zeros(count, dtype=np.float64),
        "three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "pre_goal_path_steps": np.zeros(count, dtype=np.int64),
        "pre_goal_three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "final_progress_m": np.zeros(count, dtype=np.float64),
        "alos_horizontal_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_vertical_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_three_d_error_sq_sum": np.zeros(count, dtype=np.float64),
        "alos_final_progress_m": np.zeros(count, dtype=np.float64),
        "route_ground_speed_ratio_min": np.full(count, np.inf, dtype=np.float64),
        "route_ground_speed_ratio_min_time_s": np.full(
            count, np.inf, dtype=np.float64
        ),
        "route_ground_speed_ratio_sum": np.zeros(count, dtype=np.float64),
        "route_ground_speed_steps": np.zeros(count, dtype=np.int64),
        "ground_speed_ratio_min": np.full(count, np.inf, dtype=np.float64),
        "ground_speed_ratio_min_time_s": np.full(count, np.inf, dtype=np.float64),
        "ground_speed_ratio_sum": np.zeros(count, dtype=np.float64),
        "ground_speed_steps": np.zeros(count, dtype=np.int64),
        "material_saturation_steps": np.zeros(count, dtype=np.int64),
        "safety_terminations": np.zeros(count, dtype=np.int64),
        "goal_terminations": np.zeros(count, dtype=np.int64),
        "goal_plane_miss_terminations": np.zeros(count, dtype=np.int64),
        "timeout_terminations": np.zeros(count, dtype=np.int64),
        "pnav_goal_capture_seen": np.zeros(count, dtype=bool),
        "eta_goal_capture_seen": np.zeros(count, dtype=bool),
        "eta_goal_capture_at_done": np.zeros(count, dtype=bool),
        "miss_distance_at_plane_m": np.full(count, np.nan, dtype=np.float64),
        "endpoint_entry_reached": np.zeros(count, dtype=bool),
        "post_goal_overshoot_m": np.full(count, np.nan, dtype=np.float64),
        "terminal_longitudinal_steps": np.zeros(count, dtype=np.int64),
        "terminal_longitudinal_sq_sum": np.zeros(count, dtype=np.float64),
        "max_hold_fraction": np.zeros(count, dtype=np.float64),
        "final_hold_fraction": np.zeros(count, dtype=np.float64),
        "sensor_p_nav_error_m": [],
        "sensor_dvl_bottom_valid": [],
        "sensor_current_estimate_valid": [],
        "sensor_usbl_valid": [],
        "sensor_usbl_outlier": [],
        "sensor_nav_age_s": [],
        "sensor_dvl_bottom_age_s": [],
        "sensor_current_estimate_age_s": [],
        "guidance_current_valid": [],
        "guidance_current_age_s": [],
        "guidance_crab_angle_deg": [],
        "guidance_speed_scale": [],
        "guidance_terminal_phase": [],
        "guidance_terminal_speed_ratio": [],
        "guidance_terminal_speed_ratio_values": [],
        "guidance_saturation_ema": [],
        "guidance_saturation_emergency": [],
        "guidance_saturation_emergency_steps": np.zeros(count, dtype=np.int64),
        "guidance_route_floor_active": [],
        "guidance_route_floor_launch_active": [],
        "guidance_route_floor_valid": [],
        "guidance_route_floor_boost_mps": [],
        "guidance_route_floor_ratio": [],
        "residual_action_values": [],
        "base_action_l2_values": [],
        "residual_action_l2_values": [],
        "residual_to_base_ratio_values": [],
        "residual_to_base_ratio_indices": [],
        "command_residual_applied_values": [],
        "command_residual_applied_indices": [],
        "command_residual_scale_clip_steps": np.zeros(count, dtype=np.int64),
        "command_residual_rate_clip_steps": np.zeros(count, dtype=np.int64),
        "command_residual_steps": np.zeros(count, dtype=np.int64),
        "actuator_residual_applied_values": [],
        "actuator_residual_applied_indices": [],
        "actuator_residual_gate_values": [],
        "actuator_residual_gate_indices": [],
        "actuator_residual_scale_clip_steps": np.zeros(count, dtype=np.int64),
        "actuator_residual_magnitude_clip_steps": np.zeros(count, dtype=np.int64),
        "actuator_residual_rate_clip_steps": np.zeros(count, dtype=np.int64),
        "actuator_residual_steps": np.zeros(count, dtype=np.int64),
        "frozen_residual_gate_values": [],
        "frozen_residual_gate_indices": [],
        "frozen_residual_clip_steps": np.zeros(count, dtype=np.int64),
        "frozen_residual_steps": np.zeros(count, dtype=np.int64),
        "ppo_residual_attenuation_values": [],
        "ppo_residual_attenuation_distance_values": [],
        "ppo_residual_attenuation_indices": [],
        "path_action_delta_l2_sum": np.zeros(count, dtype=np.float64),
        "path_action_delta_l2_max": np.zeros(count, dtype=np.float64),
        "path_action_delta_abs_sum": np.zeros((count, 3), dtype=np.float64),
        "path_action_steps": np.zeros(count, dtype=np.int64),
        "history_action_delta_l2_sum": np.zeros(count, dtype=np.float64),
        "history_action_delta_l2_max": np.zeros(count, dtype=np.float64),
        "history_action_delta_abs_sum": np.zeros((count, 3), dtype=np.float64),
        "history_action_steps": np.zeros(count, dtype=np.int64),
        "ground_surge_sum": np.zeros(count, dtype=np.float64),
        "ground_surge_sq_sum": np.zeros(count, dtype=np.float64),
        "ground_speed_sum": np.zeros(count, dtype=np.float64),
        "ground_speed_sq_sum": np.zeros(count, dtype=np.float64),
        "rpm_sum": np.zeros(count, dtype=np.float64),
        "rpm_sq_sum": np.zeros(count, dtype=np.float64),
        "action_delta_abs_sum": np.zeros((count, 3), dtype=np.float64),
        "action_delta_l2_sum": np.zeros(count, dtype=np.float64),
        "action_delta_steps": np.zeros(count, dtype=np.int64),
        "turn_speed_sum": np.zeros(count, dtype=np.float64),
        "turn_steps": np.zeros(count, dtype=np.int64),
        "straight_speed_sum": np.zeros(count, dtype=np.float64),
        "straight_steps": np.zeros(count, dtype=np.int64),
        "final_goal_distance_m": np.zeros(count, dtype=np.float64),
        "final_goal_distance_set": np.zeros(count, dtype=bool),
        "pnav_final_goal_distance_m": np.zeros(count, dtype=np.float64),
        "path_overshoot_m": np.zeros(count, dtype=np.float64),
        "previous_ground_surge_for_delta": np.full(count, np.nan, dtype=np.float64),
        "previous_action_for_delta": np.zeros((count, 3), dtype=np.float64),
        "has_previous_action": np.zeros(count, dtype=bool),
        "ground_surge_values": [],
        "ground_surge_indices": [],
        "ground_surge_delta_values": [],
        "ground_surge_delta_indices": [],
        "ground_speed_values": [],
        "ground_speed_indices": [],
        "ground_speed_below_min_steps": np.zeros(count, dtype=np.int64),
        "ground_speed_above_max_steps": np.zeros(count, dtype=np.int64),
        "ground_speed_envelope_steps": np.zeros(count, dtype=np.int64),
        "initial_speed_mps": np.zeros(count, dtype=np.float64),
        "initial_propeller_rpm": np.zeros(count, dtype=np.float64),
        "initial_speed_ratio": np.zeros(count, dtype=np.float64),
        "initial_propeller_rpm_ratio": np.zeros(count, dtype=np.float64),
        "rpm_values": [],
        "rpm_indices": [],
        "path_curvature_values": [],
        "path_curvature_indices": [],
        "done": np.zeros(count, dtype=bool),
    }


def _concat_series(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)


def _episode_series(
    metrics: dict[str, Any],
    values_key: str,
    indices_key: str,
    index: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for values, indices in zip(
        metrics.get(values_key, []), metrics.get(indices_key, []), strict=False
    ):
        mask = indices == index
        if bool(mask.any()):
            chunks.append(values[mask])
    return _concat_series(chunks)


def _residual_action_summary(
    metrics: dict[str, Any],
    action_full_scale: Sequence[float],
) -> dict[str, float]:
    values = metrics.get("residual_action_values", [])
    if not values:
        return {}
    residual_abs = np.concatenate(values, axis=0)
    base_l2 = np.concatenate(metrics["base_action_l2_values"], axis=0)
    residual_l2 = np.concatenate(metrics["residual_action_l2_values"], axis=0)
    scale = np.asarray(action_full_scale, dtype=np.float64)
    if residual_abs.ndim != 2 or residual_abs.shape[1] != scale.size:
        raise ValueError("residual action shape does not match action_full_scale")
    physical_abs = residual_abs * scale[None, :]
    ratios = residual_l2 / np.maximum(base_l2, 1e-12)
    return {
        "residual_action_abs_mean": float(np.mean(residual_abs)),
        "residual_action_abs_p95": float(np.percentile(residual_abs, 95)),
        "residual_action_abs_max": float(np.max(residual_abs)),
        "residual_action_l2_mean": float(np.mean(residual_l2)),
        "base_action_l2_mean": float(np.mean(base_l2)),
        "residual_to_base_l2_ratio_mean": float(np.mean(ratios)),
        "residual_to_base_l2_ratio_p95": float(np.percentile(ratios, 95)),
        "residual_rudder_abs_mean_rad": float(np.mean(physical_abs[:, 0])),
        "residual_stern_abs_mean_rad": float(np.mean(physical_abs[:, 1])),
        "residual_rpm_abs_mean_rpm": float(np.mean(physical_abs[:, 2])),
    }


def _command_residual_summary(metrics: dict[str, Any]) -> dict[str, float]:
    values = metrics.get("command_residual_applied_values", [])
    if not values:
        return {}
    applied = np.concatenate(values, axis=0)
    if applied.ndim != 2 or applied.shape[1] != 3:
        raise ValueError("command residual values must have shape (N, 3)")
    steps = max(int(np.sum(metrics.get("command_residual_steps", []))), 1)
    absolute = np.abs(applied)
    result = {
        "command_residual_abs_mean": float(np.mean(absolute)),
        "command_residual_abs_p95": float(np.percentile(absolute, 95)),
        "command_residual_abs_max": float(np.max(absolute)),
        "command_residual_heading_abs_mean_rad": float(np.mean(absolute[:, 0])),
        "command_residual_pitch_abs_mean_rad": float(np.mean(absolute[:, 1])),
        "command_residual_speed_abs_mean_mps": float(np.mean(absolute[:, 2])),
        "command_residual_scale_clip_rate": float(
            np.sum(metrics["command_residual_scale_clip_steps"]) / steps
        ),
        "command_residual_rate_clip_rate": float(
            np.sum(metrics["command_residual_rate_clip_steps"]) / steps
        ),
    }
    return result


def _actuator_residual_summary(metrics: dict[str, Any]) -> dict[str, float]:
    values = metrics.get("actuator_residual_applied_values", [])
    if not values:
        return {}
    applied = np.concatenate(values, axis=0)
    if applied.ndim != 2 or applied.shape[1] != 3:
        raise ValueError("actuator residual values must have shape (N, 3)")
    steps = max(int(np.sum(metrics.get("actuator_residual_steps", []))), 1)
    absolute = np.abs(applied)
    result = {
        "actuator_residual_abs_mean": float(np.mean(absolute)),
        "actuator_residual_abs_p95": float(np.percentile(absolute, 95)),
        "actuator_residual_abs_max": float(np.max(absolute)),
        "actuator_residual_rudder_abs_mean_rad": float(np.mean(absolute[:, 0])),
        "actuator_residual_stern_abs_mean_rad": float(np.mean(absolute[:, 1])),
        "actuator_residual_rpm_abs_mean_rpm": float(np.mean(absolute[:, 2])),
        "actuator_residual_scale_clip_rate": float(
            np.sum(metrics["actuator_residual_scale_clip_steps"]) / steps
        ),
        "actuator_residual_magnitude_clip_rate": float(
            np.sum(metrics["actuator_residual_magnitude_clip_steps"]) / steps
        ),
        "actuator_residual_rate_clip_rate": float(
            np.sum(metrics["actuator_residual_rate_clip_steps"]) / steps
        ),
    }
    gate_values = metrics.get("actuator_residual_gate_values", [])
    if gate_values:
        gate = np.concatenate(gate_values, axis=0)
        result.update(
            {
                "actuator_residual_gate_mean": float(np.mean(gate)),
                "actuator_residual_gate_zero_fraction": float(np.mean(gate <= 1e-9)),
                "actuator_residual_gate_full_fraction": float(
                    np.mean(gate >= 1.0 - 1e-9)
                ),
            }
        )
    return result


def _frozen_residual_summary(metrics: dict[str, Any]) -> dict[str, float]:
    values = metrics.get("frozen_residual_gate_values", [])
    if not values:
        return {}
    gate = np.concatenate(values, axis=0)
    steps = max(int(np.sum(metrics.get("frozen_residual_steps", []))), 1)
    return {
        "frozen_residual_gate_mean": float(np.mean(gate)),
        "frozen_residual_gate_zero_fraction": float(np.mean(gate <= 1e-9)),
        "frozen_residual_gate_full_fraction": float(np.mean(gate >= 1.0 - 1e-9)),
        "frozen_residual_clip_rate": float(
            np.sum(metrics["frozen_residual_clip_steps"]) / steps
        ),
    }


def _ppo_residual_attenuation_summary(metrics: dict[str, Any]) -> dict[str, float]:
    values = metrics.get("ppo_residual_attenuation_values", [])
    if not values:
        return {}
    attenuation = np.concatenate(values, axis=0)
    return {
        "ppo_residual_attenuation_mean": float(np.mean(attenuation)),
        "ppo_residual_attenuation_zero_fraction": float(
            np.mean(attenuation <= 1e-9)
        ),
        "ppo_residual_attenuation_full_fraction": float(
            np.mean(attenuation >= 1.0 - 1e-9)
        ),
        "ppo_residual_attenuation_steps": int(attenuation.size),
    }


def _summary(
    metrics: dict[str, Any],
    *,
    path_enabled: bool,
    path_length_m: float,
    action_full_scale: Sequence[float],
    speed_envelope_min_mps: float,
    speed_envelope_max_mps: float,
    control_dt_s: float = 0.02,
    endpoint_stop_enabled: bool = False,
    goal_radius_m: float = 2.0,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
) -> dict[str, float]:
    steps = np.maximum(metrics["active_steps"], 1)
    roll_abs = np.concatenate(metrics["roll_abs_values"])
    roll_rate_abs = np.concatenate(metrics["roll_rate_abs_values"])
    p_nav_error = np.concatenate(metrics["sensor_p_nav_error_m"])
    nav_age = np.concatenate(metrics["sensor_nav_age_s"])
    dvl_age = np.concatenate(metrics["sensor_dvl_bottom_age_s"])
    current_age = np.concatenate(metrics["sensor_current_estimate_age_s"])
    guidance_current_valid = np.concatenate(metrics["guidance_current_valid"])
    guidance_current_age = np.concatenate(metrics["guidance_current_age_s"])
    guidance_crab_deg = np.concatenate(metrics["guidance_crab_angle_deg"])
    guidance_speed_scale = np.concatenate(metrics["guidance_speed_scale"])
    guidance_terminal_phase = np.concatenate(metrics["guidance_terminal_phase"])
    guidance_terminal_ratios = np.concatenate(metrics["guidance_terminal_speed_ratio"])
    guidance_terminal_ratio_values = metrics["guidance_terminal_speed_ratio_values"]
    if guidance_terminal_ratio_values:
        guidance_terminal_ratio_values = np.concatenate(guidance_terminal_ratio_values)
    else:
        guidance_terminal_ratio_values = np.zeros(0, dtype=np.float64)
    guidance_saturation_ema = np.concatenate(metrics["guidance_saturation_ema"])
    guidance_saturation_emergency = np.concatenate(
        metrics["guidance_saturation_emergency"]
    )
    summary = {
        "success_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "command_hold_success_rate": float(
            np.mean(metrics["max_hold_fraction"] >= 1.0 - 1e-9)
        ),
        "mean_abs_yaw_error_rad": float(np.mean(metrics["abs_yaw_error_sum"] / steps)),
        "mean_abs_pitch_error_rad": float(
            np.mean(metrics["abs_pitch_error_sum"] / steps)
        ),
        "mean_abs_speed_error_mps": float(
            np.mean(metrics["abs_speed_error_sum"] / steps)
        ),
        "yaw_rmse_rad": float(np.sqrt(np.mean(metrics["yaw_error_sq_sum"] / steps))),
        "pitch_rmse_rad": float(
            np.sqrt(np.mean(metrics["pitch_error_sq_sum"] / steps))
        ),
        "speed_rmse_mps": float(
            np.sqrt(np.mean(metrics["speed_error_sq_sum"] / steps))
        ),
        "roll_abs_mean_rad": float(np.mean(metrics["abs_roll_sum"] / steps)),
        "roll_abs_p95_rad": float(np.percentile(roll_abs, 95)),
        "roll_abs_max_rad": float(np.max(metrics["abs_roll_max"])),
        "roll_rate_abs_p95_rad_s": float(np.percentile(roll_rate_abs, 95)),
        "material_saturation_rate": float(
            np.sum(metrics["material_saturation_steps"]) / np.sum(steps)
        ),
        "safety_termination_rate": float(
            np.mean(metrics["safety_terminations"] > 0)
        ),
        "goal_termination_rate": float(np.mean(metrics["goal_terminations"] > 0)),
        "goal_plane_miss_rate": float(
            np.mean(metrics["goal_plane_miss_terminations"] > 0)
        ),
        "timeout_rate": float(np.mean(metrics["timeout_terminations"] > 0)),
        "goal_capture_pnav_rate": float(
            np.mean(metrics["pnav_goal_capture_seen"])
        ),
        "goal_capture_eta_rate": float(np.mean(metrics["eta_goal_capture_seen"])),
        "mean_reward": float(np.mean(metrics["reward_sum"] / steps)),
        "mean_episode_reward": float(np.mean(metrics["reward_sum"])),
        "final_hold_fraction_mean": float(np.mean(metrics["final_hold_fraction"])),
        "nav_position_error_mean_m": float(np.mean(p_nav_error)),
        "nav_position_error_rmse_m": float(np.sqrt(np.mean(np.square(p_nav_error)))),
        "nav_position_error_p95_m": float(np.percentile(p_nav_error, 95)),
        "dvl_bottom_valid_rate": float(
            np.mean(np.concatenate(metrics["sensor_dvl_bottom_valid"]))
        ),
        "dvl_bottom_outage_rate": float(
            1.0 - np.mean(np.concatenate(metrics["sensor_dvl_bottom_valid"]))
        ),
        "current_estimate_valid_rate": float(
            np.mean(np.concatenate(metrics["sensor_current_estimate_valid"]))
        ),
        "usbl_valid_rate": float(
            np.mean(np.concatenate(metrics["sensor_usbl_valid"]))
        ),
        "usbl_outlier_rate": float(
            np.mean(np.concatenate(metrics["sensor_usbl_outlier"]))
        ),
        "nav_age_s_mean": float(np.mean(nav_age)),
        "nav_age_s_p95": float(np.percentile(nav_age, 95)),
        "dvl_bottom_age_s_mean": float(np.mean(dvl_age)),
        "dvl_bottom_age_s_p95": float(np.percentile(dvl_age, 95)),
        "current_estimate_age_s_mean": float(np.mean(current_age)),
        "current_estimate_age_s_p95": float(np.percentile(current_age, 95)),
        "guidance_current_valid_rate": float(np.mean(guidance_current_valid)),
        "guidance_current_age_s_mean": float(np.mean(guidance_current_age)),
        "guidance_crab_deg_mean": float(np.mean(guidance_crab_deg)),
        "guidance_crab_deg_abs_mean": float(np.mean(np.abs(guidance_crab_deg))),
        "guidance_speed_scale_mean": float(np.mean(guidance_speed_scale)),
        "guidance_terminal_phase_rate": float(np.mean(guidance_terminal_phase)),
        "terminal_speed_ratio_mean": float(
            np.mean(guidance_terminal_ratio_values)
            if guidance_terminal_ratio_values.size
            else 0.0
        ),
        "guidance_terminal_speed_ratio_mean": float(
            np.mean(guidance_terminal_ratios)
        ),
        "guidance_speed_ratio_mean": float(np.mean(guidance_terminal_ratios)),
        "guidance_route_speed_ratio_min": float(np.min(guidance_terminal_ratios)),
        "ground_route_speed_ratio_min": float(
            np.min(
                metrics["route_ground_speed_ratio_min"][
                    metrics["route_ground_speed_steps"] > 0
                ]
            )
            if np.any(metrics["route_ground_speed_steps"] > 0)
            else np.nan
        ),
        "ground_route_speed_ratio_episode_min_mean": float(
            np.mean(
                metrics["route_ground_speed_ratio_min"][
                    metrics["route_ground_speed_steps"] > 0
                ]
            )
            if np.any(metrics["route_ground_speed_steps"] > 0)
            else np.nan
        ),
        "ground_route_speed_ratio_mean": float(
            np.mean(
                metrics["route_ground_speed_ratio_sum"]
                / np.maximum(metrics["route_ground_speed_steps"], 1)
            )
        ),
        "ground_route_speed_ratio_min_time_s": float(
            np.mean(metrics["route_ground_speed_ratio_min_time_s"])
        ),
        "ground_speed_ratio_min": float(
            np.min(
                metrics["ground_speed_ratio_min"][
                    metrics["ground_speed_steps"] > 0
                ]
            )
            if np.any(metrics["ground_speed_steps"] > 0)
            else np.nan
        ),
        "ground_speed_ratio_episode_min_mean": float(
            np.mean(
                metrics["ground_speed_ratio_min"][
                    metrics["ground_speed_steps"] > 0
                ]
            )
            if np.any(metrics["ground_speed_steps"] > 0)
            else np.nan
        ),
        "ground_speed_ratio_mean": float(
            np.mean(
                metrics["ground_speed_ratio_sum"]
                / np.maximum(metrics["ground_speed_steps"], 1)
            )
        ),
        "ground_speed_ratio_min_time_s": float(
            np.mean(metrics["ground_speed_ratio_min_time_s"])
        ),
        "guidance_saturation_ema_mean": float(np.mean(guidance_saturation_ema)),
        "guidance_saturation_emergency_rate": float(
            np.mean(guidance_saturation_emergency)
        ),
        "guidance_saturation_emergency_episode_rate": float(
            np.mean(metrics["guidance_saturation_emergency_steps"] > 0)
        ),
        "guidance_route_floor_active_rate": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_active"]))
        ),
        "guidance_route_floor_launch_active_rate": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_launch_active"]))
        ),
        "guidance_route_floor_valid_rate": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_valid"]))
        ),
        "guidance_route_floor_boost_mps_mean": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_boost_mps"]))
        ),
        "guidance_route_floor_ratio_min": float(
            np.min(np.concatenate(metrics["guidance_route_floor_ratio"]))
        ),
        "guidance_route_floor_ratio_mean": float(
            np.mean(np.concatenate(metrics["guidance_route_floor_ratio"]))
        ),
    }
    summary.update(_residual_action_summary(metrics, action_full_scale))
    summary.update(_command_residual_summary(metrics))
    summary.update(_actuator_residual_summary(metrics))
    summary.update(_frozen_residual_summary(metrics))
    summary.update(_ppo_residual_attenuation_summary(metrics))
    if path_enabled:
        path_steps = np.maximum(metrics["path_steps"], 1)
        pre_goal_steps_total = int(np.sum(metrics["pre_goal_path_steps"]))
        pre_goal_rmse = (
            None
            if pre_goal_steps_total <= 0
            else float(
                np.sqrt(
                    np.sum(metrics["pre_goal_three_d_error_sq_sum"])
                    / pre_goal_steps_total
                )
            )
        )
        miss_values = metrics["miss_distance_at_plane_m"][
            np.isfinite(metrics["miss_distance_at_plane_m"])
        ]
        entry_mask = metrics["endpoint_entry_reached"]
        overshoot_values = metrics["post_goal_overshoot_m"][
            entry_mask & np.isfinite(metrics["post_goal_overshoot_m"])
        ]
        terminal_steps = int(np.sum(metrics["terminal_longitudinal_steps"]))
        terminal_longitudinal_rmse = (
            None
            if endpoint_stop_enabled or terminal_steps <= 0
            else float(
                np.sqrt(
                    np.sum(metrics["terminal_longitudinal_sq_sum"])
                    / terminal_steps
                )
            )
        )
        summary.update(
            {
                "path_horizontal_rmse_m": float(
                    np.sqrt(np.mean(metrics["horizontal_error_sq_sum"] / path_steps))
                ),
                "path_vertical_rmse_m": float(
                    np.sqrt(np.mean(metrics["vertical_error_sq_sum"] / path_steps))
                ),
                "path_three_d_rmse_m": float(
                    np.sqrt(np.mean(metrics["three_d_error_sq_sum"] / path_steps))
                ),
                "pre_goal_3d_rmse_m": pre_goal_rmse,
                "final_path_progress_m_mean": float(
                    np.mean(metrics["final_progress_m"])
                ),
                "miss_distance_at_plane_m": (
                    None if miss_values.size == 0 else float(np.mean(miss_values))
                ),
                "miss_distance_at_plane_max_m": (
                    None if miss_values.size == 0 else float(np.max(miss_values))
                ),
                "post_goal_overshoot_m": (
                    None
                    if endpoint_stop_enabled or overshoot_values.size == 0
                    else float(np.mean(overshoot_values))
                ),
                "terminal_longitudinal_rmse_m": terminal_longitudinal_rmse,
                "path_completion_ratio_mean": float(
                    np.mean(
                        np.clip(
                            metrics["final_progress_m"] / max(path_length_m, 1e-12),
                            0.0,
                            1.0,
                        )
                    )
                ),
                "alos_projection_horizontal_rmse_m": float(
                    np.sqrt(
                        np.mean(
                            metrics["alos_horizontal_error_sq_sum"] / path_steps
                        )
                    )
                ),
                "alos_projection_vertical_rmse_m": float(
                    np.sqrt(
                        np.mean(metrics["alos_vertical_error_sq_sum"] / path_steps)
                    )
                ),
                "alos_projection_three_d_rmse_m": float(
                    np.sqrt(
                        np.mean(metrics["alos_three_d_error_sq_sum"] / path_steps)
                    )
                ),
                "alos_projection_final_progress_m_mean": float(
                    np.mean(metrics["alos_final_progress_m"])
                ),
            }
        )
    path_action_steps = int(np.sum(metrics["path_action_steps"]))
    if path_action_steps > 0:
        scale = np.asarray(action_full_scale, dtype=np.float64)
        action_abs_sum = np.sum(metrics["path_action_delta_abs_sum"], axis=0)
        summary.update(
            {
                "path_action_delta_l2_mean": float(
                    np.sum(metrics["path_action_delta_l2_sum"]) / path_action_steps
                ),
                "path_action_delta_l2_max": float(
                    np.max(metrics["path_action_delta_l2_max"])
                ),
                "path_action_delta_abs_mean": float(
                    np.sum(action_abs_sum) / (path_action_steps * scale.size)
                ),
                "path_action_delta_rudder_abs_mean_rad": float(
                    action_abs_sum[0] / path_action_steps * scale[0]
                ),
                "path_action_delta_stern_abs_mean_rad": float(
                    action_abs_sum[1] / path_action_steps * scale[1]
                ),
                "path_action_delta_rpm_abs_mean_rpm": float(
                    action_abs_sum[2] / path_action_steps * scale[2]
                ),
            }
        )
    history_action_steps = int(np.sum(metrics["history_action_steps"]))
    if history_action_steps > 0:
        scale = np.asarray(action_full_scale, dtype=np.float64)
        action_abs_sum = np.sum(metrics["history_action_delta_abs_sum"], axis=0)
        summary.update(
            {
                "history_action_delta_l2_mean": float(
                    np.sum(metrics["history_action_delta_l2_sum"])
                    / history_action_steps
                ),
                "history_action_delta_l2_max": float(
                    np.max(metrics["history_action_delta_l2_max"])
                ),
                "history_action_delta_abs_mean": float(
                    np.sum(action_abs_sum) / (history_action_steps * scale.size)
                ),
                "history_action_delta_rudder_abs_mean_rad": float(
                    action_abs_sum[0] / history_action_steps * scale[0]
                ),
                "history_action_delta_stern_abs_mean_rad": float(
                    action_abs_sum[1] / history_action_steps * scale[1]
                ),
                "history_action_delta_rpm_abs_mean_rpm": float(
                    action_abs_sum[2] / history_action_steps * scale[2]
                ),
            }
        )
    ground_surge_values = _concat_series(metrics.get("ground_surge_values", []))
    ground_speed_values = _concat_series(metrics.get("ground_speed_values", []))
    ground_surge_delta_values = _concat_series(
        metrics.get("ground_surge_delta_values", [])
    )
    rpm_values = _concat_series(metrics.get("rpm_values", []))
    curvature_values = _concat_series(metrics.get("path_curvature_values", []))
    if ground_surge_values.size:
        summary["ground_surge_mean_mps"] = float(np.mean(ground_surge_values))
        summary["ground_surge_std_mps"] = float(np.std(ground_surge_values))
    if ground_speed_values.size:
        summary["ground_speed_mean_mps"] = float(np.mean(ground_speed_values))
        summary["ground_speed_std_mps"] = float(np.std(ground_speed_values))
        summary["ground_speed_min_mps"] = float(np.min(ground_speed_values))
        summary["ground_speed_max_mps"] = float(np.max(ground_speed_values))
    envelope_steps = int(np.sum(metrics["ground_speed_envelope_steps"]))
    if envelope_steps > 0:
        summary.update(
            {
                "speed_envelope_min_mps": float(speed_envelope_min_mps),
                "speed_envelope_max_mps": float(speed_envelope_max_mps),
                "ground_speed_below_min_rate": float(
                    np.sum(metrics["ground_speed_below_min_steps"]) / envelope_steps
                ),
                "ground_speed_above_max_rate": float(
                    np.sum(metrics["ground_speed_above_max_steps"]) / envelope_steps
                ),
            }
        )
    summary.update(
        {
            "initial_speed_mean_mps": float(np.mean(metrics["initial_speed_mps"])),
            "initial_speed_min_mps": float(np.min(metrics["initial_speed_mps"])),
            "initial_speed_max_mps": float(np.max(metrics["initial_speed_mps"])),
            "initial_speed_nonzero_fraction": float(
                np.mean(metrics["initial_speed_mps"] > 1e-9)
            ),
            "initial_propeller_rpm_mean": float(
                np.mean(metrics["initial_propeller_rpm"])
            ),
            "initial_propeller_rpm_min": float(
                np.min(metrics["initial_propeller_rpm"])
            ),
            "initial_propeller_rpm_max": float(
                np.max(metrics["initial_propeller_rpm"])
            ),
            "initial_propeller_rpm_nonzero_fraction": float(
                np.mean(metrics["initial_propeller_rpm"] > 1e-9)
            ),
        }
    )
    if ground_surge_delta_values.size:
        summary["ground_surge_delta_abs_p95_mps"] = float(
            np.percentile(np.abs(ground_surge_delta_values), 95)
        )
    if rpm_values.size:
        summary["rpm_sq_mean_rpm2"] = float(np.mean(np.square(rpm_values)))
        summary["rpm_sq_integral_proxy_rpm2_s"] = float(
            np.sum(np.square(rpm_values)) * control_dt_s
        )
    if curvature_values.size:
        summary["abs_curvature_mean_1_per_m"] = float(
            np.mean(np.abs(curvature_values))
        )
        summary["turn_step_rate"] = float(
            np.mean(np.abs(curvature_values) >= TURN_CURVATURE_THRESHOLD)
        )
    action_delta_steps = int(np.sum(metrics["action_delta_steps"]))
    if action_delta_steps > 0:
        scale = np.asarray(action_full_scale, dtype=np.float64)
        action_abs_sum = np.sum(metrics["action_delta_abs_sum"], axis=0)
        summary["action_delta_abs_mean"] = float(
            np.sum(action_abs_sum) / (action_delta_steps * scale.size)
        )
        summary["action_delta_l2_mean"] = float(
            np.sum(metrics["action_delta_l2_sum"]) / action_delta_steps
        )
        summary["action_delta_rudder_abs_mean_rad"] = float(
            action_abs_sum[0] / action_delta_steps * scale[0]
        )
        summary["action_delta_stern_abs_mean_rad"] = float(
            action_abs_sum[1] / action_delta_steps * scale[1]
        )
        summary["action_delta_rpm_abs_mean_rpm"] = float(
            action_abs_sum[2] / action_delta_steps * scale[2]
        )
    turn_steps = int(np.sum(metrics["turn_steps"]))
    straight_steps = int(np.sum(metrics["straight_steps"]))
    if turn_steps > 0:
        summary["turn_speed_mean_mps"] = float(
            np.sum(metrics["turn_speed_sum"]) / turn_steps
        )
    if straight_steps > 0:
        summary["straight_speed_mean_mps"] = float(
            np.sum(metrics["straight_speed_sum"]) / straight_steps
        )
    if turn_steps > 0 and straight_steps > 0:
        summary["turn_straight_speed_ratio"] = float(
            summary["turn_speed_mean_mps"] / max(summary["straight_speed_mean_mps"], 1e-12)
        )
    goal_mask = metrics["goal_terminations"] > 0
    if bool(goal_mask.any()):
        summary["time_to_goal_s_mean"] = float(
            np.mean(metrics["active_steps"][goal_mask]) * control_dt_s
        )
    summary["episode_time_s_mean"] = float(np.mean(steps) * control_dt_s)
    if bool(metrics["final_goal_distance_set"].any()):
        done_set = metrics["final_goal_distance_set"]
        summary["eta_done_distance_m_mean"] = float(
            np.mean(metrics["final_goal_distance_m"][done_set])
        )
        summary["p_nav_done_distance_m_mean"] = float(
            np.mean(metrics["pnav_final_goal_distance_m"][done_set])
        )
        summary["success_margin_m_mean"] = float(
            np.mean(goal_radius_m - metrics["final_goal_distance_m"][done_set])
        )
    summary["path_overshoot_m_mean"] = float(np.mean(metrics["path_overshoot_m"]))
    success_rows = [
        score_success_contract(
            p_nav_tight_success=bool(metrics["goal_terminations"][index] > 0),
            eta_tight_ever=bool(metrics["eta_goal_capture_seen"][index]),
            eta_tight_at_done=bool(metrics["eta_goal_capture_at_done"][index]),
            success_frame=success_frame,
            eta_done_distance_m=(
                float(metrics["final_goal_distance_m"][index])
                if metrics["final_goal_distance_set"][index]
                else None
            ),
            p_nav_done_distance_m=(
                float(metrics["pnav_final_goal_distance_m"][index])
                if metrics["final_goal_distance_set"][index]
                else None
            ),
            goal_radius_m=float(goal_radius_m),
        )
        for index in range(len(metrics["goal_terminations"]))
    ]
    summary.update(
        summarize_success_contract(success_rows, success_frame=success_frame)
    )
    return summary


def _episode_records(
    metrics: dict[str, Any],
    seeds: Sequence[int],
    *,
    path_enabled: bool,
    path_length_m: float,
    action_full_scale: Sequence[float],
    control_dt_s: float = 0.02,
    endpoint_stop_enabled: bool = False,
    goal_radius_m: float = 2.0,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
    plant_scale_factors: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return per-seed closed-loop metrics for paired ABC analysis."""

    active_steps = np.maximum(metrics["active_steps"].astype(np.int64), 1)
    path_steps = np.maximum(metrics["path_steps"].astype(np.int64), 1)
    scale = np.asarray(action_full_scale, dtype=np.float64)
    if scale.shape != (3,):
        raise ValueError("action_full_scale must contain exactly three channels")
    count = len(seeds)
    arrays = (
        "goal_terminations",
        "timeout_terminations",
        "safety_terminations",
        "material_saturation_steps",
    )
    if any(len(metrics[name]) != count for name in arrays):
        raise ValueError("episode metrics do not align with the requested seeds")
    if len(metrics["active_steps"]) != count:
        raise ValueError("active step metrics do not align with the requested seeds")
    if plant_scale_factors is not None and plant_scale_factors.shape != (count, 5):
        raise ValueError("plant_scale_factors must have shape (episodes, 5)")

    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        steps = int(active_steps[index])
        row: dict[str, Any] = {
            "seed": int(seed),
            "plant_scale_factors": (
                None
                if plant_scale_factors is None
                else {
                    "drag_scale": float(plant_scale_factors[index, 0]),
                    "thrust_scale": float(plant_scale_factors[index, 1]),
                    "fin_scale": float(plant_scale_factors[index, 2]),
                    "added_mass_scale": float(plant_scale_factors[index, 3]),
                    "actuator_time_scale": float(plant_scale_factors[index, 4]),
                }
            ),
            "steps": steps,
            "success": bool(metrics["goal_terminations"][index] > 0),
            "goal_plane_miss": bool(
                metrics["goal_plane_miss_terminations"][index] > 0
            ),
            "timeout": bool(metrics["timeout_terminations"][index] > 0),
            "safety": bool(metrics["safety_terminations"][index] > 0),
            "goal_capture_pnav": bool(metrics["pnav_goal_capture_seen"][index]),
            "goal_capture_eta": bool(metrics["eta_goal_capture_seen"][index]),
            "goal_capture_eta_at_done": bool(
                metrics["eta_goal_capture_at_done"][index]
            ),
            "endpoint_entry_reached": bool(
                metrics["endpoint_entry_reached"][index]
            ),
            "mean_reward": float(metrics["reward_sum"][index] / steps),
            "material_saturation_rate": float(
                metrics["material_saturation_steps"][index] / steps
            ),
            "command_residual_abs_mean": (
                None
                if metrics["command_residual_steps"][index] <= 0
                else float(
                    np.mean(
                        np.abs(
                            _episode_series(
                                metrics,
                                "command_residual_applied_values",
                                "command_residual_applied_indices",
                                index,
                            )
                        )
                    )
                )
            ),
            "command_residual_scale_clip_rate": float(
                metrics["command_residual_scale_clip_steps"][index]
                / max(int(metrics["command_residual_steps"][index]), 1)
            ),
            "command_residual_rate_clip_rate": float(
                metrics["command_residual_rate_clip_steps"][index]
                / max(int(metrics["command_residual_steps"][index]), 1)
            ),
            "actuator_residual_abs_mean": (
                None
                if metrics["actuator_residual_steps"][index] <= 0
                else float(
                    np.mean(
                        np.abs(
                            _episode_series(
                                metrics,
                                "actuator_residual_applied_values",
                                "actuator_residual_applied_indices",
                                index,
                            )
                        )
                    )
                )
            ),
            "actuator_residual_scale_clip_rate": float(
                metrics["actuator_residual_scale_clip_steps"][index]
                / max(int(metrics["actuator_residual_steps"][index]), 1)
            ),
            "actuator_residual_magnitude_clip_rate": float(
                metrics["actuator_residual_magnitude_clip_steps"][index]
                / max(int(metrics["actuator_residual_steps"][index]), 1)
            ),
            "actuator_residual_rate_clip_rate": float(
                metrics["actuator_residual_rate_clip_steps"][index]
                / max(int(metrics["actuator_residual_steps"][index]), 1)
            ),
            "actuator_residual_gate_mean": (
                None
                if not metrics["actuator_residual_gate_indices"]
                or not any(
                    index in values
                    for values in metrics["actuator_residual_gate_indices"]
                )
                else float(
                    np.mean(
                        _episode_series(
                            metrics,
                            "actuator_residual_gate_values",
                            "actuator_residual_gate_indices",
                            index,
                        )
                    )
                )
            ),
            "mean_abs_yaw_error_rad": float(
                metrics["abs_yaw_error_sum"][index] / steps
            ),
            "mean_abs_pitch_error_rad": float(
                metrics["abs_pitch_error_sum"][index] / steps
            ),
            "mean_abs_speed_error_mps": float(
                metrics["abs_speed_error_sum"][index] / steps
            ),
            "yaw_rmse_rad": float(
                np.sqrt(metrics["yaw_error_sq_sum"][index] / steps)
            ),
            "pitch_rmse_rad": float(
                np.sqrt(metrics["pitch_error_sq_sum"][index] / steps)
            ),
            "speed_rmse_mps": float(
                np.sqrt(metrics["speed_error_sq_sum"][index] / steps)
            ),
            "roll_abs_max_rad": float(metrics["abs_roll_max"][index]),
            "final_hold_fraction": float(metrics["final_hold_fraction"][index]),
        }
        row.update(
            score_success_contract(
                p_nav_tight_success=row["success"],
                eta_tight_ever=row["goal_capture_eta"],
                eta_tight_at_done=row["goal_capture_eta_at_done"],
                success_frame=success_frame,
                eta_done_distance_m=row.get("final_goal_distance_m"),
                p_nav_done_distance_m=row.get("p_nav_done_distance_m"),
                goal_radius_m=float(goal_radius_m),
            )
        )
        if path_enabled:
            path_count = int(path_steps[index])
            pre_goal_count = int(metrics["pre_goal_path_steps"][index])
            miss_distance = float(metrics["miss_distance_at_plane_m"][index])
            overshoot = float(metrics["post_goal_overshoot_m"][index])
            terminal_steps = int(metrics["terminal_longitudinal_steps"][index])
            row.update(
                {
                    "path_horizontal_rmse_m": float(
                        np.sqrt(metrics["horizontal_error_sq_sum"][index] / path_count)
                    ),
                    "path_vertical_rmse_m": float(
                        np.sqrt(metrics["vertical_error_sq_sum"][index] / path_count)
                    ),
                    "path_three_d_rmse_m": float(
                        np.sqrt(metrics["three_d_error_sq_sum"][index] / path_count)
                    ),
                    "pre_goal_3d_rmse_m": (
                        None
                        if pre_goal_count <= 0
                        else float(
                            np.sqrt(
                                metrics["pre_goal_three_d_error_sq_sum"][index]
                                / pre_goal_count
                            )
                        )
                    ),
                    "final_path_progress_m": float(
                        metrics["final_progress_m"][index]
                    ),
                    "path_completion_ratio": float(
                        np.clip(
                            metrics["final_progress_m"][index]
                            / max(path_length_m, 1e-12),
                            0.0,
                            1.0,
                        )
                    ),
                    "miss_distance_at_plane_m": (
                        None if not np.isfinite(miss_distance) else miss_distance
                    ),
                    "post_goal_overshoot_m": (
                        None
                        if endpoint_stop_enabled or not np.isfinite(overshoot)
                        else overshoot
                    ),
                    "terminal_longitudinal_rmse_m": (
                        None
                        if endpoint_stop_enabled or terminal_steps <= 0
                        else float(
                            np.sqrt(
                                metrics["terminal_longitudinal_sq_sum"][index]
                                / terminal_steps
                            )
                        )
                    ),
                }
            )
        path_action_count = int(metrics["path_action_steps"][index])
        if path_action_count > 0:
            path_action_abs = metrics["path_action_delta_abs_sum"][index]
            row.update(
                {
                    "path_action_delta_l2_mean": float(
                        metrics["path_action_delta_l2_sum"][index]
                        / path_action_count
                    ),
                    "path_action_delta_l2_max": float(
                        metrics["path_action_delta_l2_max"][index]
                    ),
                    "path_action_delta_abs_mean": float(
                        np.sum(path_action_abs)
                        / (path_action_count * 3)
                    ),
                    "path_action_delta_rudder_abs_mean_rad": float(
                        path_action_abs[0] / path_action_count * scale[0]
                    ),
                    "path_action_delta_stern_abs_mean_rad": float(
                        path_action_abs[1] / path_action_count * scale[1]
                    ),
                    "path_action_delta_rpm_abs_mean_rpm": float(
                        path_action_abs[2] / path_action_count * scale[2]
                    ),
                }
            )
        history_action_count = int(metrics["history_action_steps"][index])
        if history_action_count > 0:
            history_action_abs = metrics["history_action_delta_abs_sum"][index]
            row.update(
                {
                    "history_action_delta_l2_mean": float(
                        metrics["history_action_delta_l2_sum"][index]
                        / history_action_count
                    ),
                    "history_action_delta_l2_max": float(
                        metrics["history_action_delta_l2_max"][index]
                    ),
                    "history_action_delta_abs_mean": float(
                        np.sum(history_action_abs)
                        / (history_action_count * 3)
                    ),
                    "history_action_delta_rudder_abs_mean_rad": float(
                        history_action_abs[0] / history_action_count * scale[0]
                    ),
                    "history_action_delta_stern_abs_mean_rad": float(
                        history_action_abs[1] / history_action_count * scale[1]
                    ),
                    "history_action_delta_rpm_abs_mean_rpm": float(
                        history_action_abs[2] / history_action_count * scale[2]
                    ),
                }
            )
        residual_ratio_values = _episode_series(
            metrics,
            "residual_to_base_ratio_values",
            "residual_to_base_ratio_indices",
            index,
        )
        if residual_ratio_values.size:
            row["residual_to_base_l2_ratio_mean"] = float(
                np.mean(residual_ratio_values)
            )
            row["residual_to_base_l2_ratio_p95"] = float(
                np.percentile(residual_ratio_values, 95)
            )
        surge_values = _episode_series(
            metrics, "ground_surge_values", "ground_surge_indices", index
        )
        if surge_values.size:
            row["ground_surge_mean_mps"] = float(np.mean(surge_values))
            row["ground_surge_std_mps"] = float(np.std(surge_values))
        surge_delta_values = _episode_series(
            metrics,
            "ground_surge_delta_values",
            "ground_surge_delta_indices",
            index,
        )
        if surge_delta_values.size:
            row["ground_surge_delta_abs_p95_mps"] = float(
                np.percentile(np.abs(surge_delta_values), 95)
            )
        speed_values = _episode_series(
            metrics, "ground_speed_values", "ground_speed_indices", index
        )
        if speed_values.size:
            row["ground_speed_mean_mps"] = float(np.mean(speed_values))
            row["ground_speed_std_mps"] = float(np.std(speed_values))
            row["ground_speed_min_mps"] = float(np.min(speed_values))
            row["ground_speed_max_mps"] = float(np.max(speed_values))
        envelope_steps = int(metrics["ground_speed_envelope_steps"][index])
        if envelope_steps > 0:
            row["ground_speed_below_min_rate"] = float(
                metrics["ground_speed_below_min_steps"][index] / envelope_steps
            )
            row["ground_speed_above_max_rate"] = float(
                metrics["ground_speed_above_max_steps"][index] / envelope_steps
            )
        row.update(
            {
                "initial_speed_mps": float(metrics["initial_speed_mps"][index]),
                "initial_propeller_rpm": float(
                    metrics["initial_propeller_rpm"][index]
                ),
                "initial_speed_ratio": float(metrics["initial_speed_ratio"][index]),
                "initial_propeller_rpm_ratio": float(
                    metrics["initial_propeller_rpm_ratio"][index]
                ),
            }
        )
        rpm_values = _episode_series(metrics, "rpm_values", "rpm_indices", index)
        if rpm_values.size:
            row["rpm_sq_mean_rpm2"] = float(np.mean(np.square(rpm_values)))
            row["rpm_sq_integral_proxy_rpm2_s"] = float(
                np.sum(np.square(rpm_values)) * control_dt_s
            )
        curvature_values = _episode_series(
            metrics,
            "path_curvature_values",
            "path_curvature_indices",
            index,
        )
        if curvature_values.size:
            row["abs_curvature_mean_1_per_m"] = float(
                np.mean(np.abs(curvature_values))
            )
            row["turn_step_rate"] = float(
                np.mean(np.abs(curvature_values) >= TURN_CURVATURE_THRESHOLD)
            )
        action_count = int(metrics["action_delta_steps"][index])
        if action_count > 0:
            action_abs = metrics["action_delta_abs_sum"][index]
            row.update(
                {
                    "action_delta_abs_mean": float(
                        np.sum(action_abs) / (action_count * scale.size)
                    ),
                    "action_delta_l2_mean": float(
                        metrics["action_delta_l2_sum"][index] / action_count
                    ),
                    "action_delta_rudder_abs_mean_rad": float(
                        action_abs[0] / action_count * scale[0]
                    ),
                    "action_delta_stern_abs_mean_rad": float(
                        action_abs[1] / action_count * scale[1]
                    ),
                    "action_delta_rpm_abs_mean_rpm": float(
                        action_abs[2] / action_count * scale[2]
                    ),
                }
            )
        turn_steps = int(metrics["turn_steps"][index])
        straight_steps = int(metrics["straight_steps"][index])
        if turn_steps > 0:
            row["turn_speed_mean_mps"] = float(
                metrics["turn_speed_sum"][index] / turn_steps
            )
        if straight_steps > 0:
            row["straight_speed_mean_mps"] = float(
                metrics["straight_speed_sum"][index] / straight_steps
            )
        if turn_steps > 0 and straight_steps > 0:
            row["turn_straight_speed_ratio"] = float(
                row["turn_speed_mean_mps"]
                / max(row["straight_speed_mean_mps"], 1e-12)
            )
        row["episode_time_s"] = float(steps * control_dt_s)
        row["time_to_goal_s"] = (
            float(steps * control_dt_s) if row["success"] else None
        )
        row["path_overshoot_m"] = float(metrics["path_overshoot_m"][index])
        if bool(metrics["final_goal_distance_set"][index]):
            row["final_goal_distance_m"] = float(
                metrics["final_goal_distance_m"][index]
            )
            row["p_nav_done_distance_m"] = float(
                metrics["pnav_final_goal_distance_m"][index]
            )
        rows.append(row)
    return rows


def _restore_residual_norm_projection(
    policy: PCActorCritic,
    checkpoint: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Restore the optional AUV-SM residual norm projection from a checkpoint."""

    auv_sm_metadata = checkpoint.get("auv_sm_metadata")
    if isinstance(auv_sm_metadata, dict):
        projection_kind = auv_sm_metadata.get("residual_projection")
        if projection_kind not in (None, "relative_l2_base_ratio"):
            raise ValueError(
                f"unsupported residual projection in AUV-SM metadata: {projection_kind!r}"
            )
        policy.set_residual_norm_projection(
            auv_sm_metadata.get("residual_norm_limit_ratio"),
            eps=float(auv_sm_metadata.get("residual_norm_eps", 1e-8)),
        )
        return
    pc_metadata = metadata.get("pc")
    if isinstance(pc_metadata, dict):
        policy.set_residual_norm_projection(
            pc_metadata.get("frozen_bc_residual_norm_limit_ratio"),
            eps=float(pc_metadata.get("frozen_bc_residual_norm_eps", 1e-8)),
        )


def _build_policy(
    training_config_path: Path,
    checkpoint_path: Path,
    *,
    num_actor_obs: int,
    num_critic_obs: int,
    device: torch.device,
) -> tuple[PCActorCritic, dict[str, Any]]:
    training = load_training_config(training_config_path)
    if training.pc is None:
        raise ValueError("evaluation requires a PC training config")
    path_blocks = _final_path_blocks(training)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = validate_pc_checkpoint_metadata(
        checkpoint,
        expected_actor_observation_dim=num_actor_obs,
        expected_critic_observation_dim=num_critic_obs,
        expected_schema_hash=pc_schema_hash(
            path_dim=training.pc.path_dim,
            path_blocks=path_blocks,
            dade_history_mode=training.pc.dade_history_mode,
            actor_command_dim=training.pc.actor_command_dim,
            critic_command_dim=training.pc.critic_command_dim,
            history_path_dim=training.pc.history_path_dim,
            actor_current_encoder_enabled=training.pc.actor_current_encoder_enabled,
        ),
        expected_path_dim=training.pc.path_dim,
        expected_path_blocks=path_blocks,
    )
    assert metadata is not None
    expected_hash = sha256_file(training_config_path)
    if metadata["training_config_sha256"] != expected_hash:
        raise ValueError(
            "checkpoint training-config hash does not match --training-config"
        )
    policy_config = training.rsl_train_cfg()["policy"]
    policy = PCActorCritic(
        num_actor_obs,
        num_critic_obs,
        3,
        **{key: value for key, value in policy_config.items() if key != "class_name"},
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain model_state_dict")
    policy.load_state_dict(state_dict, strict=True)
    _restore_residual_norm_projection(policy, checkpoint, metadata)
    if policy.frozen_bc_path_encoder is not None:
        if (
            training.base_mode != "frozen_pc_bc_residual"
            or training.base_checkpoint is None
            or training.base_checkpoint_sha256 is None
        ):
            raise ValueError(
                "frozen BC path encoder requires a configured frozen BC checkpoint"
            )
        base_checkpoint_path = _resolve(Path(training.base_checkpoint))
        if sha256_file(base_checkpoint_path) != training.base_checkpoint_sha256:
            raise ValueError("base BC checkpoint SHA-256 does not match the training config")
        restored_alpha = restore_pc_bc_frozen_path_encoder_alpha(
            policy, base_checkpoint_path
        )
        saved_frozen = metadata.get("pc", {}).get("frozen_bc_path_encoder")
        if isinstance(saved_frozen, dict) and saved_frozen.get("alpha") is not None:
            if not np.isclose(
                float(saved_frozen["alpha"]),
                restored_alpha,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "checkpoint and BC base disagree on frozen path encoder alpha"
                )
    elif metadata.get("pc", {}).get("frozen_bc_path_encoder") is not None:
        raise ValueError("checkpoint metadata declares a missing frozen BC path encoder")
    if policy.frozen_bc_current_encoder is not None:
        if (
            training.base_mode != "frozen_pc_bc_residual"
            or training.base_checkpoint is None
            or training.base_checkpoint_sha256 is None
        ):
            raise ValueError(
                "frozen BC current encoder requires a configured frozen BC checkpoint"
            )
        base_checkpoint_path = _resolve(Path(training.base_checkpoint))
        if sha256_file(base_checkpoint_path) != training.base_checkpoint_sha256:
            raise ValueError("base BC checkpoint SHA-256 does not match the training config")
        restored_alpha = restore_pc_bc_frozen_current_encoder_alpha(
            policy, base_checkpoint_path
        )
        saved_frozen = metadata.get("pc", {}).get("frozen_bc_current_encoder")
        if isinstance(saved_frozen, dict) and saved_frozen.get("alpha") is not None:
            if not np.isclose(
                float(saved_frozen["alpha"]),
                restored_alpha,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "checkpoint and BC base disagree on frozen current encoder alpha"
                )
    elif metadata.get("pc", {}).get("frozen_bc_current_encoder") is not None:
        raise ValueError("checkpoint metadata declares a missing frozen BC current encoder")
    policy.eval()
    return policy, metadata


def _build_curriculum_sampler(
    curriculum_stage: str | None,
    *,
    current_speed_range_mps: tuple[float, float] | None = None,
    shear_scale: float | None = None,
    current_variation_speed_amplitude_mps: float = 0.0,
) -> CurriculumStageSampler | None:
    """Build the fixed-stage sampler with optional eval-only stress overrides."""

    if curriculum_stage is None:
        if current_speed_range_mps is not None or shear_scale is not None:
            raise ValueError("curriculum stress overrides require a curriculum stage")
        return None
    stages = default_curriculum_stages()
    if curriculum_stage not in stages:
        raise ValueError(f"unknown curriculum stage: {curriculum_stage!r}")
    if current_speed_range_mps is not None or shear_scale is not None:
        stage = stages[curriculum_stage]
        if current_speed_range_mps is not None:
            stage = replace(stage, current_speed_range_mps=current_speed_range_mps)
        if shear_scale is not None:
            stage = replace(stage, shear_scale=shear_scale)
        stages[curriculum_stage] = stage
    return CurriculumStageSampler(
        stages=stages,
        stage_mix=((curriculum_stage, 1.0),),
        current_variation_speed_amplitude_mps=current_variation_speed_amplitude_mps,
    )


def _run(
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    environment_config_path: Path | None,
    seeds: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    history_ablation: str = "none",
    path_ablation: str = "none",
    action_mode: str = "policy",
    injected_residual_profile: tuple[str, tuple[float, float, float] | None] = (
        "zero",
        None,
    ),
    allow_stage_override: bool = False,
    allow_sensor_override: bool = False,
    allow_actor_observation_override: bool = False,
    allow_guidance_rate_limit_override: bool = False,
    allow_diversity_environment_override: bool = False,
    allow_start_progress_training_override: bool = False,
    curriculum_stage: str | None = None,
    disable_curriculum_direct_command: bool = False,
    curriculum_current_speed_range_mps: tuple[float, float] | None = None,
    curriculum_shear_scale: float | None = None,
    guidance_config_path: Path | None = None,
    controller_config_path: Path | None = None,
    trace_output: Path | None = None,
    trace_sensor_ages: bool = False,
    trace_estimator_state: bool = False,
    trace_reward_breakdown: bool = False,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
    actuator_residual_scale_override: float | None = None,
    actuator_residual_channel_gains: tuple[float, float, float] | None = None,
    actuator_residual_scale_near: float | None = None,
    actuator_residual_scale_far: float | None = None,
    actuator_residual_scale_switch_distance_m: float | None = None,
    actuator_residual_near_channel_gains: tuple[float, float, float] | None = None,
    actuator_residual_channel_gain_near_distance_m: float | None = None,
    actuator_residual_channel_gain_far_distance_m: float | None = None,
    terminal_gate_enter_m: float = 20.0,
    terminal_gate_full_m: float = 5.0,
    ppo_residual_attenuation_full_m: float = 1.25,
    ppo_residual_attenuation_enter_m: float = 3.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    success_frame = validate_success_frame(success_frame)
    _validate_actuator_residual_scale_schedule(
        near=actuator_residual_scale_near,
        far=actuator_residual_scale_far,
        switch_distance_m=actuator_residual_scale_switch_distance_m,
        constant_override=actuator_residual_scale_override,
    )
    _validate_actuator_channel_gain_schedule(
        near_gains=actuator_residual_near_channel_gains,
        near_distance_m=actuator_residual_channel_gain_near_distance_m,
        far_distance_m=actuator_residual_channel_gain_far_distance_m,
    )
    if action_mode not in {
        "policy",
        "frozen-base",
        "zero-command-residual",
        "zero-actuator-residual",
        "zero-residual",
        "gated-residual",
        "attenuated-residual",
        "injected-residual",
        "alos",
    }:
        raise ValueError(f"unsupported action mode: {action_mode}")
    if action_mode == "attenuated-residual":
        if (
            not np.isfinite(ppo_residual_attenuation_full_m)
            or ppo_residual_attenuation_full_m <= 0.0
            or not np.isfinite(ppo_residual_attenuation_enter_m)
            or ppo_residual_attenuation_enter_m <= ppo_residual_attenuation_full_m
        ):
            raise ValueError(
                "PPO attenuation requires finite positive full_m < enter_m"
            )
    if (
        not np.isfinite(terminal_gate_enter_m)
        or terminal_gate_enter_m <= 0.0
        or not np.isfinite(terminal_gate_full_m)
        or terminal_gate_full_m <= 0.0
        or terminal_gate_full_m >= terminal_gate_enter_m
    ):
        raise ValueError(
            "terminal gate requires finite positive enter_m/full_m with full_m < enter_m"
        )
    training = load_training_config(training_config_path)
    if training.pc is None or training.history is None:
        raise ValueError("evaluation requires a PC training config with history")
    resolved_environment_config = (
        environment_config_path
        if environment_config_path is not None
        else _resolve(Path(training.environment_config))
    )
    config = load_config(resolved_environment_config)
    base_config = None
    sensor_override = False
    if environment_config_path is not None:
        base_config = load_config(_resolve(Path(training.environment_config)))
        sensor_override = bool(
            allow_sensor_override and config.sensors != base_config.sensors
        )
        _validate_environment_override(
            base_config,
            config,
            allow_sensor_override=allow_sensor_override,
            allow_actor_observation_override=allow_actor_observation_override,
            allow_guidance_rate_limit_override=allow_guidance_rate_limit_override,
            allow_diversity_environment_override=(
                allow_diversity_environment_override
            ),
            allow_start_progress_training_override=(
                allow_start_progress_training_override
            ),
        )
    if config.pc_control_mode.startswith("path_") and training.pc.path_dim != PC_PATH_DIM:
        raise ValueError(
            "ALOS-free path_* PC control requires pc.path_dim=PC_PATH_DIM (8)"
        )
    if guidance_config_path is not None:
        config = replace(config, guidance=load_config(guidance_config_path).guidance)
    if controller_config_path is not None:
        controller_override = load_config(controller_config_path).controller
        _validate_controller_pitch_override(config.controller, controller_override)
        config = replace(config, controller=controller_override)
    if config.arm != "PC" or config.policy_mode != "pure_controller":
        raise ValueError("environment config is not a PC pure-controller config")
    effective_current_override = curriculum_current_speed_range_mps
    if effective_current_override is None:
        effective_current_override = config.curriculum.current_speed_range_override_mps
    effective_shear_override = curriculum_shear_scale
    if effective_shear_override is None:
        effective_shear_override = config.curriculum.shear_scale_override
    curriculum_sampler = _build_curriculum_sampler(
        curriculum_stage,
        current_speed_range_mps=effective_current_override,
        shear_scale=effective_shear_override,
        current_variation_speed_amplitude_mps=(
            0.0
            if config.current.time_variation is None
            else config.current.time_variation.speed_amplitude_mps
        ),
    )
    inner = BatchedPCLowLevelEnv(
        config,
        num_envs=len(seeds),
        device=device.type,
        dtype=dtype,
        validate_finite=True,
        path_dim=training.pc.path_dim,
        curriculum_sampler=curriculum_sampler,
        curriculum_direct_commands_enabled=not disable_curriculum_direct_command,
    )
    resolved_action_interface = resolve_pc_action_interface(
        training.pc.action_interface,
        command_residual_enabled=config.command_residual.enabled,
    )
    if resolved_action_interface == "command":
        inner = BatchedPCCommandResidualEnv(
            inner,
            scale_heading_rad=config.command_residual.scale_heading_rad,
            scale_pitch_rad=config.command_residual.scale_pitch_rad,
            scale_speed_mps=config.command_residual.scale_speed_mps,
            rate_heading_rad_s=config.command_residual.rate_heading_rad_s,
            rate_pitch_rad_s=config.command_residual.rate_pitch_rad_s,
            rate_speed_mps2=config.command_residual.rate_speed_mps2,
            controller_state=config.command_residual.controller_state,
            terminal_gate_enabled=config.command_residual.terminal_gate_enabled,
            terminal_gate_enter_m=config.command_residual.terminal_gate_enter_m,
            terminal_gate_full_m=config.command_residual.terminal_gate_full_m,
        )
    elif resolved_action_interface == "actuator":
        if (
            actuator_residual_scale_near is not None
            and action_mode not in {"policy", "gated-residual"}
        ):
            raise ValueError(
                "actuator residual scale schedule requires policy or gated-residual action mode"
            )
        if action_mode == "gated-residual":
            if training.pc.actuator_terminal_gate_enabled:
                raise ValueError(
                    "gated-residual eval requires training.pc.actuator_terminal_gate_enabled=false"
                )
        residual_scale = training.pc.actuator_residual_scale
        channel_gains = (
            tuple(training.pc.actuator_residual_channel_gains)
            if actuator_residual_channel_gains is None
            else tuple(float(value) for value in actuator_residual_channel_gains)
        )
        if len(channel_gains) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in channel_gains
        ):
            raise ValueError(
                "actuator residual channel gains must contain three finite positive values"
            )
        if actuator_residual_scale_override is not None:
            if (
                not np.isfinite(actuator_residual_scale_override)
                or actuator_residual_scale_override <= 0.0
            ):
                raise ValueError(
                    "actuator residual scale override must be finite and positive"
                )
            residual_scale = float(actuator_residual_scale_override)
        inner = BatchedPCActuatorResidualEnv(
            inner,
            residual_scale=residual_scale,
            residual_channel_gains=channel_gains,
            max_action=training.pc.actuator_max_action,
            controller_state="actor",
            terminal_gate_enabled=training.pc.actuator_terminal_gate_enabled,
            terminal_gate_enter_m=training.pc.actuator_terminal_gate_enter_m,
            terminal_gate_full_m=training.pc.actuator_terminal_gate_full_m,
            residual_scale_near=actuator_residual_scale_near,
            residual_scale_far=actuator_residual_scale_far,
            residual_scale_switch_distance_m=(
                actuator_residual_scale_switch_distance_m
            ),
            residual_channel_gains_near=actuator_residual_near_channel_gains,
            residual_channel_gain_near_distance_m=(
                actuator_residual_channel_gain_near_distance_m
            ),
            residual_channel_gain_far_distance_m=(
                actuator_residual_channel_gain_far_distance_m
            ),
        )
    elif resolved_action_interface != "absolute":
        raise ValueError("unsupported resolved PC action interface")
    if (
        actuator_residual_scale_near is not None
        and resolved_action_interface != "actuator"
    ):
        raise ValueError(
            "actuator residual scale schedule requires the actuator action interface"
        )
    if (
        actuator_residual_channel_gains is not None
        and resolved_action_interface != "actuator"
    ):
        raise ValueError(
            "actuator residual channel gains require the actuator action interface"
        )
    if (
        actuator_residual_near_channel_gains is not None
        and resolved_action_interface != "actuator"
    ):
        raise ValueError(
            "actuator residual channel gain schedule requires the actuator action interface"
        )
    if action_mode == "zero-command-residual" and resolved_action_interface != "command":
        raise ValueError(
            "zero-command-residual requires the checkpoint's command residual interface"
        )
    if action_mode == "zero-actuator-residual" and resolved_action_interface != "actuator":
        raise ValueError(
            "zero-actuator-residual requires the checkpoint's actuator residual interface"
        )
    frozen_residual_output = (
        training.base_mode == "frozen_pc_bc_residual"
        and training.base_output_mode == "residual"
    )
    if action_mode == "zero-residual" and not frozen_residual_output:
        raise ValueError(
            "zero-residual requires base_policy.output_mode='residual'"
        )
    if action_mode == "attenuated-residual" and not frozen_residual_output:
        raise ValueError(
            "attenuated-residual requires frozen BC residual output"
        )
    if action_mode == "injected-residual" and not frozen_residual_output:
        raise ValueError(
            "injected-residual requires frozen BC residual output"
        )
    path_blocks = _final_path_blocks(training)
    path_enabled = str(config.pc_stage) in {"C2", "C3", "C4", "C5", "C6"}
    endpoint_stop_enabled = bool(
        config.termination.pc_goal_plane_miss_termination_enabled
    )
    env = BatchTorchPCVecEnv(
        inner,
        short_window=training.history.short_window,
        long_window=training.history.long_window,
        short_stride=training.history.short_stride,
        long_stride=training.history.long_stride,
        actor_command_dim=training.pc.actor_command_dim,
        critic_command_dim=training.pc.critic_command_dim,
        history_path_dim=training.pc.history_path_dim,
        actor_current_encoder_enabled=training.pc.actor_current_encoder_enabled,
        normalization=PCNormalization(
            position_scale_m=config.normalization.position_scale_m,
            attitude_scale_rad=config.normalization.attitude_scale_rad,
            angular_rate_scale_rad_s=config.normalization.angular_rate_scale_rad_s,
            surge_speed_scale_mps=config.normalization.surge_speed_scale_mps,
            sway_heave_speed_scale_mps=config.normalization.sway_heave_speed_scale_mps,
            current_scale_mps=config.normalization.current_scale_mps,
        ),
        current_frame_dropout_prob=0.0,
        path_blocks=path_blocks,
    )
    policy, metadata = _build_policy(
        training_config_path,
        checkpoint_path,
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        device=device,
    )
    checkpoint_action_interface = metadata.get("pc", {}).get("action_interface")
    if (
        checkpoint_action_interface is not None
        and str(checkpoint_action_interface) != resolved_action_interface
    ):
        raise ValueError(
            "checkpoint and training config disagree on the PC action interface: "
            f"{checkpoint_action_interface!r} != {resolved_action_interface!r}"
        )
    checkpoint_bias = metadata.get("pc", {}).get("base_residual_bias")
    if checkpoint_bias is not None and not np.allclose(
        np.asarray(checkpoint_bias, dtype=np.float64),
        np.asarray(training.pc.base_residual_bias, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "checkpoint and training config disagree on pc.base_residual_bias: "
            f"{checkpoint_bias!r} != {list(training.pc.base_residual_bias)!r}"
        )
    frozen_residual_env: BatchTorchPCFrozenBCResidualVecEnv | None = None
    if _uses_frozen_residual_wrapper(
        base_mode=training.base_mode,
        base_output_mode=training.base_output_mode,
        action_mode=action_mode,
    ):
        if resolved_action_interface not in {"absolute", "actuator"}:
            raise ValueError(
                "frozen BC residual output requires the resolved absolute or "
                f"actuator action interface, got {resolved_action_interface!r}"
            )
        assert training.pc is not None
        frozen_residual_env = BatchTorchPCFrozenBCResidualVecEnv(
            env,
            base_action_fn=policy.base_action,
            terminal_gate_enabled=training.pc.residual_terminal_gate_enabled,
            terminal_gate_enter_m=training.pc.residual_terminal_gate_enter_m,
            terminal_gate_full_m=training.pc.residual_terminal_gate_full_m,
            rate_aware_residual_enabled=training.pc.rate_aware_residual_enabled,
            ppo_rate_fraction=training.pc.ppo_rate_fraction,
            ppo_fin_gain=training.pc.ppo_fin_gain,
            ppo_rpm_gain=training.pc.ppo_rpm_gain,
            base_residual_bias=training.pc.base_residual_bias,
        )
        if action_mode == "gated-residual":
            frozen_residual_env.set_terminal_gate(
                enabled=True,
                enter_m=terminal_gate_enter_m,
                full_m=terminal_gate_full_m,
            )
        env = frozen_residual_env
    actor_decision_stride = training.actor_decision_stride
    if actor_decision_stride <= 0:
        raise ValueError("actor_decision_stride must be positive")
    expected_normalization = PCNormalization(
        position_scale_m=config.normalization.position_scale_m,
        attitude_scale_rad=config.normalization.attitude_scale_rad,
        angular_rate_scale_rad_s=config.normalization.angular_rate_scale_rad_s,
        surge_speed_scale_mps=config.normalization.surge_speed_scale_mps,
        sway_heave_speed_scale_mps=config.normalization.sway_heave_speed_scale_mps,
        current_scale_mps=config.normalization.current_scale_mps,
    ).as_dict()
    if metadata["normalization"] != expected_normalization:
        raise ValueError("environment override changes PC normalization scales")
    stage_override = metadata["pc_stage"] != str(config.pc_stage)
    if stage_override and not _stage_override_allowed(
        metadata["pc_stage"],
        str(config.pc_stage),
        allow_stage_override=allow_stage_override,
    ):
        raise ValueError("environment override changes the PC stage")
    if metadata["actuator_envelope"] != config.actuator.envelope:
        raise ValueError("environment override changes the actuator envelope")
    expected_action_full_scale = (
        config.actuator.policy_max_rudder_angle_rad,
        config.actuator.policy_max_stern_plane_angle_rad,
        config.actuator.policy_max_propeller_rpm,
    )
    if any(
        not np.isclose(actual, expected, rtol=0.0, atol=1e-12)
        for actual, expected in zip(
            metadata["action_full_scale"], expected_action_full_scale, strict=True
        )
    ):
        raise ValueError("environment override changes the policy action scale")
    observations, _ = env.reset(seeds)
    assert inner.eta is not None
    assert inner.nu is not None
    assert inner.actuator is not None
    initial_heading_rad = inner.eta[:, 5].detach().cpu().numpy().copy()
    true_projection_hint = inner.path.project(inner.eta[:, :3]).segment_index
    metrics = _new_episode_metrics(len(seeds))
    metrics["initial_speed_mps"] = inner.nu[:, 0].detach().cpu().numpy().copy()
    metrics["initial_propeller_rpm"] = (
        inner.actuator[:, 2].detach().cpu().numpy().copy()
    )
    metrics["initial_speed_ratio"] = (
        inner._episode_initial_speed_ratio.detach().cpu().numpy().copy()
    )
    metrics["initial_propeller_rpm_ratio"] = (
        inner._episode_initial_propeller_rpm_ratio.detach().cpu().numpy().copy()
    )
    speed_envelope_min_mps = float(inner.vehicle.min_speed_mps)
    speed_envelope_max_mps = float(inner.vehicle.max_speed_mps)
    trace: dict[str, list[np.ndarray]] | None = (
        {} if trace_output is not None else None
    )

    def record_trace(name: str, value: torch.Tensor | np.ndarray) -> None:
        if trace is None:
            return
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        trace.setdefault(name, []).append(array[active_numpy])

    held_action = torch.zeros(
        (len(seeds), 3), dtype=torch.float32, device=device
    )
    ppo_residual_attenuation = None
    ppo_residual_attenuation_distance = None
    for step_index in range(config.plant.max_episode_steps):
        active = ~metrics["done"]
        if not bool(active.any()):
            break
        active_tensor = torch.as_tensor(active, device=device, dtype=torch.bool)
        decision_step = step_index % actor_decision_stride == 0
        ppo_residual_attenuation = None
        ppo_residual_attenuation_distance = None
        base_action = None
        residual_action = None
        clean_action = None
        history_clean_action = None
        if decision_step:
            with torch.inference_mode():
                history_observations = ablate_pc_history(
                    observations, policy.actor_layout, history_ablation
                )
                policy_observations = ablate_pc_path(
                    history_observations, policy.actor_layout, path_ablation
                )
                base_action = policy.base_action(policy_observations)
                if action_mode == "alos":
                    held_action = torch.zeros_like(held_action)
                    residual_action = held_action
                    clean_action = held_action
                    history_clean_action = held_action
                elif frozen_residual_output:
                    if action_mode in {"zero-residual", "frozen-base"}:
                        held_action = torch.zeros_like(held_action)
                        residual_action = held_action
                        clean_action = held_action
                        history_clean_action = held_action
                    elif action_mode == "gated-residual":
                        residual_action = policy.residual_action(policy_observations)
                        held_action = residual_action
                        clean_action = held_action
                        history_clean_action = held_action
                    elif action_mode == "policy":
                        residual_action = policy.residual_action(policy_observations)
                        held_action = policy.act_inference(policy_observations)
                        clean_action = (
                            policy.act_inference(history_observations)
                            if path_ablation != "none"
                            else held_action
                        )
                        history_clean_action = (
                            policy.act_inference(
                                ablate_pc_path(
                                    observations, policy.actor_layout, path_ablation
                                )
                            )
                            if history_ablation != "none"
                            else held_action
                        )
                    elif action_mode == "injected-residual":
                        profile_name, profile_const = injected_residual_profile
                        if profile_name == "zero":
                            residual_action = torch.zeros_like(held_action)
                        elif profile_name == "const":
                            assert profile_const is not None
                            residual_action = torch.as_tensor(
                                profile_const,
                                device=device,
                                dtype=held_action.dtype,
                            ).expand(held_action.shape[0], 3).clone()
                        else:
                            ppo_residual = policy.residual_action(policy_observations)
                            if profile_name == "ppo":
                                residual_action = ppo_residual
                            elif profile_name == "neg-ppo":
                                residual_action = -ppo_residual
                            elif profile_name == "ppo-gain":
                                assert profile_const is not None
                                residual_action = profile_const[0] * ppo_residual
                            else:
                                raise ValueError(
                                    f"unsupported injected residual profile {profile_name!r}"
                                )
                        held_action = residual_action
                        clean_action = held_action
                        history_clean_action = held_action
                    elif action_mode == "attenuated-residual":
                        residual_action = policy.residual_action(policy_observations)
                        gate_observation = inner.observation
                        position = gate_observation.actor_state[:, :3].to(
                            device=device, dtype=residual_action.dtype
                        )
                        goal = inner.path.goal_ned_m_for(inner.num_envs).to(
                            device=device, dtype=residual_action.dtype
                        )
                        attenuation_distance = torch.linalg.norm(
                            goal - position, dim=1
                        )
                        attenuation = _ppo_residual_attenuation(
                            attenuation_distance,
                            full_m=ppo_residual_attenuation_full_m,
                            enter_m=ppo_residual_attenuation_enter_m,
                        )
                        held_action = residual_action * attenuation[:, None]
                        ppo_residual_attenuation = attenuation
                        ppo_residual_attenuation_distance = attenuation_distance
                        clean_action = held_action
                        history_clean_action = held_action
                    else:
                        raise ValueError(
                            f"action mode {action_mode!r} is not valid for "
                            "frozen BC residual output"
                        )
                elif action_mode in {
                    "zero-command-residual",
                    "zero-actuator-residual",
                }:
                    held_action = torch.zeros_like(held_action)
                    residual_action = held_action
                    clean_action = held_action
                    history_clean_action = held_action
                elif action_mode == "frozen-base":
                    if resolved_action_interface not in {"absolute", "actuator"}:
                        raise ValueError(
                            "frozen-base evaluation requires the checkpoint's "
                            "absolute or actuator interface"
                        )
                    if base_action is None:
                        raise ValueError(
                            "frozen-base evaluation requires an enabled frozen BC actor"
                        )
                    held_action = base_action
                    residual_action = torch.zeros_like(base_action)
                    clean_action = (
                        policy.base_action(history_observations)
                        if path_ablation != "none"
                        else held_action
                    )
                    history_clean_action = (
                        policy.base_action(
                            ablate_pc_path(
                                observations, policy.actor_layout, path_ablation
                            )
                        )
                        if history_ablation != "none"
                        else held_action
                    )
                elif action_mode == "gated-residual":
                    if base_action is None:
                        raise ValueError(
                            "gated-residual evaluation requires an enabled frozen BC actor"
                        )
                    residual_action = policy.residual_action(policy_observations)
                    gate_observation = inner.observation
                    position = gate_observation.actor_state[:, :3].to(
                        device=device, dtype=held_action.dtype
                    )
                    goal = inner.path.goal_ned_m_for(inner.num_envs).to(
                        device=device, dtype=held_action.dtype
                    )
                    distance = torch.linalg.norm(goal - position, dim=1)
                    gate = smooth_terminal_gate(
                        distance,
                        enter_m=terminal_gate_enter_m,
                        full_m=terminal_gate_full_m,
                    )
                    held_action = base_action + residual_action * gate[:, None]
                    clean_action = held_action
                    history_clean_action = held_action
                else:
                    residual_action = policy.residual_action(policy_observations)
                    held_action = policy.act_inference(policy_observations)
                    clean_action = (
                        policy.act_inference(history_observations)
                        if path_ablation != "none"
                        else held_action
                    )
                    history_clean_action = (
                        policy.act_inference(
                            ablate_pc_path(
                                observations, policy.actor_layout, path_ablation
                            )
                        )
                        if history_ablation != "none"
                        else held_action
                    )
        action = torch.where(
            active_tensor[:, None], held_action, torch.zeros_like(held_action)
        )
        with torch.inference_mode():
            observations, reward, done, infos = env.step(action, auto_reset=False)

        assert inner.eta is not None and inner.nu is not None
        command = inner.observation.command
        yaw_error = torch.remainder(
            command.heading_rad - inner.eta[:, 5] + torch.pi, 2.0 * torch.pi
        ) - torch.pi
        pitch_error = command.pitch_rad - inner.eta[:, 4]
        assert inner.current_ned_mps is not None
        ground_velocity = ground_velocity_body_for_contract(
            inner.eta,
            inner.nu,
            inner.current_ned_mps,
            config.plant.velocity_contract,
        )
        speed_error = command.speed_mps - ground_velocity[:, 0]
        roll = inner.eta[:, 3].abs()
        roll_rate = inner.nu[:, 3].abs()
        projection = infos["projection"]
        assert inner.sensor_output is not None
        sensor = inner.sensor_output
        active_numpy = active
        if ppo_residual_attenuation is not None:
            attenuation_indices = np.flatnonzero(active_numpy)
            attenuation_values = (
                ppo_residual_attenuation.detach().cpu().numpy()
            )
            attenuation_distance_values = (
                ppo_residual_attenuation_distance.detach().cpu().numpy()
            )
            metrics["ppo_residual_attenuation_values"].append(
                attenuation_values[active_numpy]
            )
            metrics["ppo_residual_attenuation_distance_values"].append(
                attenuation_distance_values[active_numpy]
            )
            metrics["ppo_residual_attenuation_indices"].append(
                attenuation_indices
            )
        if infos.get("command_residual_applied") is not None:
            command_delta = (
                infos["command_residual_applied"].detach().cpu().numpy()
            )
            active_indices = np.flatnonzero(active_numpy)
            metrics["command_residual_applied_values"].append(
                command_delta[active_numpy]
            )
            metrics["command_residual_applied_indices"].append(active_indices)
            metrics["command_residual_scale_clip_steps"][active_numpy] += (
                infos["command_residual_scale_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["command_residual_rate_clip_steps"][active_numpy] += (
                infos["command_residual_rate_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["command_residual_steps"][active_numpy] += 1
        if infos.get("actuator_residual_applied") is not None:
            actuator_delta = (
                infos["actuator_residual_applied"].detach().cpu().numpy()
            )
            active_indices = np.flatnonzero(active_numpy)
            metrics["actuator_residual_applied_values"].append(
                actuator_delta[active_numpy]
            )
            metrics["actuator_residual_applied_indices"].append(active_indices)
            if infos.get("actuator_residual_gate") is not None:
                actuator_gate = (
                    infos["actuator_residual_gate"].detach().cpu().numpy()
                )
                metrics["actuator_residual_gate_values"].append(
                    actuator_gate[active_numpy]
                )
                metrics["actuator_residual_gate_indices"].append(active_indices)
            metrics["actuator_residual_scale_clip_steps"][active_numpy] += (
                infos["actuator_residual_scale_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["actuator_residual_magnitude_clip_steps"][active_numpy] += (
                infos["actuator_residual_magnitude_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["actuator_residual_rate_clip_steps"][active_numpy] += (
                infos["actuator_residual_rate_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["actuator_residual_steps"][active_numpy] += 1
        if infos.get("frozen_residual_gate") is not None:
            frozen_indices = np.flatnonzero(active_numpy)
            frozen_gate = (
                infos["frozen_residual_gate"].detach().cpu().numpy()
            )
            metrics["frozen_residual_gate_values"].append(
                frozen_gate[active_numpy]
            )
            metrics["frozen_residual_gate_indices"].append(frozen_indices)
            metrics["frozen_residual_clip_steps"][active_numpy] += (
                infos["frozen_residual_clipped"]
                .detach()
                .cpu()
                .numpy()[active_numpy]
                .any(axis=1)
                .astype(np.int64)
            )
            metrics["frozen_residual_steps"][active_numpy] += 1
        goal_ned_m = inner.path.goal_ned_m_for(len(seeds))
        goal_tangent_ned = inner.path.goal_tangent_ned_for(len(seeds))
        position_source = config.termination.position_source
        if position_source == "eta":
            termination_position_ned_m = inner.eta[:, :3]
        elif position_source == "p_nav":
            termination_position_ned_m = sensor.p_nav
        else:
            termination_position_ned_m = (
                sensor.p_nav if endpoint_stop_enabled else inner.eta[:, :3]
            )
        endpoint_goal_distance_m = (
            torch.linalg.norm(termination_position_ned_m - goal_ned_m, dim=1)
            .cpu()
            .numpy()
        )
        signed_goal_tangent_offset_m = (
            ((termination_position_ned_m - goal_ned_m) * goal_tangent_ned)
            .sum(dim=1)
            .cpu()
            .numpy()
        )
        signed_goal_tangent_offset_eta_m = (
            ((inner.eta[:, :3] - goal_ned_m) * goal_tangent_ned)
            .sum(dim=1)
            .cpu()
            .numpy()
        )
        eta_goal_distance_m = (
            torch.linalg.norm(inner.eta[:, :3] - goal_ned_m, dim=1)
            .cpu()
            .numpy()
        )
        pnav_goal_distance_m = (
            torch.linalg.norm(sensor.p_nav - goal_ned_m, dim=1)
            .cpu()
            .numpy()
        )
        endpoint_now = (
            endpoint_goal_distance_m <= config.plant.goal_radius_m + 1e-12
        ) | (signed_goal_tangent_offset_m >= 0.0)
        metrics["pnav_goal_capture_seen"][
            active_numpy & (pnav_goal_distance_m <= config.plant.goal_radius_m + 1e-12)
        ] = True
        metrics["eta_goal_capture_seen"][
            active_numpy & (eta_goal_distance_m <= config.plant.goal_radius_m + 1e-12)
        ] = True
        metrics["endpoint_entry_reached"][active_numpy & endpoint_now] = True
        cruise_speed_mps = infos["episode_command_speed_mps"].cpu().numpy()
        ground_speed_mps = (
            torch.linalg.norm(ground_velocity, dim=1).cpu().numpy()
        )
        ground_speed_ratio = ground_speed_mps / np.maximum(
            cruise_speed_mps, 1e-9
        )
        previous_speed_min = metrics["ground_speed_ratio_min"][active_numpy]
        current_speed_ratio = ground_speed_ratio[active_numpy]
        speed_lower = current_speed_ratio < previous_speed_min
        active_indices = np.flatnonzero(active_numpy)
        metrics["ground_speed_ratio_min"][active_numpy] = np.minimum(
            previous_speed_min, current_speed_ratio
        )
        metrics["ground_speed_ratio_min_time_s"][active_indices[speed_lower]] = (
            step_index * config.plant.dt_s
        )
        metrics["ground_speed_ratio_sum"][active_numpy] += current_speed_ratio
        metrics["ground_speed_steps"][active_numpy] += 1
        assert inner.actuator is not None
        surge_numpy = ground_velocity[:, 0].detach().cpu().numpy()
        rpm_numpy = inner.actuator[:, 2].detach().cpu().numpy()
        curvature_numpy = projection.curvature_1_per_m.detach().cpu().numpy()
        metrics["ground_surge_sum"][active_numpy] += surge_numpy[active_numpy]
        metrics["ground_surge_sq_sum"][active_numpy] += np.square(
            surge_numpy[active_numpy]
        )
        metrics["ground_speed_sum"][active_numpy] += ground_speed_mps[active_numpy]
        metrics["ground_speed_sq_sum"][active_numpy] += np.square(
            ground_speed_mps[active_numpy]
        )
        metrics["rpm_sum"][active_numpy] += rpm_numpy[active_numpy]
        metrics["rpm_sq_sum"][active_numpy] += np.square(rpm_numpy[active_numpy])
        metrics["ground_surge_values"].append(surge_numpy[active_numpy])
        metrics["ground_surge_indices"].append(active_indices)
        metrics["ground_speed_values"].append(ground_speed_mps[active_numpy])
        metrics["ground_speed_indices"].append(active_indices)
        metrics["ground_speed_below_min_steps"][active_numpy] += (
            ground_speed_mps[active_numpy] < speed_envelope_min_mps - 1e-9
        )
        metrics["ground_speed_above_max_steps"][active_numpy] += (
            ground_speed_mps[active_numpy] > speed_envelope_max_mps + 1e-9
        )
        metrics["ground_speed_envelope_steps"][active_numpy] += 1
        metrics["rpm_values"].append(rpm_numpy[active_numpy])
        metrics["rpm_indices"].append(active_indices)
        metrics["path_curvature_values"].append(curvature_numpy[active_numpy])
        metrics["path_curvature_indices"].append(active_indices)
        previous_surge = metrics["previous_ground_surge_for_delta"]
        delta_valid = active_numpy & np.isfinite(previous_surge)
        if bool(delta_valid.any()):
            delta_indices = np.flatnonzero(delta_valid)
            metrics["ground_surge_delta_values"].append(
                np.abs(surge_numpy - previous_surge)[delta_valid]
            )
            metrics["ground_surge_delta_indices"].append(delta_indices)
        previous_surge[active_numpy] = surge_numpy[active_numpy]
        turn_active = active_numpy & (
            np.abs(curvature_numpy) >= TURN_CURVATURE_THRESHOLD
        )
        straight_active = active_numpy & ~turn_active
        metrics["turn_speed_sum"][turn_active] += ground_speed_mps[turn_active]
        metrics["turn_steps"][turn_active] += 1
        metrics["straight_speed_sum"][straight_active] += ground_speed_mps[
            straight_active
        ]
        metrics["straight_steps"][straight_active] += 1
        if decision_step:
            action_numpy = action.detach().cpu().numpy()
            previous_action = metrics["previous_action_for_delta"]
            action_valid = active_numpy & metrics["has_previous_action"]
            if bool(action_valid.any()):
                action_delta = np.abs(action_numpy - previous_action)
                metrics["action_delta_abs_sum"][action_valid] += action_delta[
                    action_valid
                ]
                metrics["action_delta_l2_sum"][action_valid] += np.linalg.norm(
                    action_delta[action_valid], axis=1
                )
                metrics["action_delta_steps"][action_valid] += 1
            previous_action[active_numpy] = action_numpy[active_numpy]
            metrics["has_previous_action"][active_numpy] = True
        if decision_step and path_ablation != "none":
            action_delta = (action - clean_action).detach().abs()
            metrics["path_action_delta_l2_sum"][active_numpy] += (
                torch.linalg.norm(action_delta, dim=1).cpu().numpy()[active_numpy]
            )
            metrics["path_action_delta_l2_max"][active_numpy] = np.maximum(
                metrics["path_action_delta_l2_max"][active_numpy],
                torch.linalg.norm(action_delta, dim=1).cpu().numpy()[active_numpy],
            )
            metrics["path_action_delta_abs_sum"][active_numpy] += (
                action_delta.cpu().numpy()[active_numpy]
            )
            metrics["path_action_steps"][active_numpy] += 1
        if decision_step and history_ablation != "none":
            action_delta = (action - history_clean_action).detach().abs()
            metrics["history_action_delta_l2_sum"][active_numpy] += (
                torch.linalg.norm(action_delta, dim=1).cpu().numpy()[active_numpy]
            )
            metrics["history_action_delta_l2_max"][active_numpy] = np.maximum(
                metrics["history_action_delta_l2_max"][active_numpy],
                torch.linalg.norm(action_delta, dim=1).cpu().numpy()[active_numpy],
            )
            metrics["history_action_delta_abs_sum"][active_numpy] += (
                action_delta.cpu().numpy()[active_numpy]
            )
            metrics["history_action_steps"][active_numpy] += 1
        if base_action is not None:
            residual_abs = residual_action.detach().abs().cpu().numpy()[active_numpy]
            base_l2 = (
                torch.linalg.norm(base_action.detach(), dim=1)
                .cpu()
                .numpy()[active_numpy]
            )
            residual_l2 = (
                torch.linalg.norm(residual_action.detach(), dim=1)
                .cpu()
                .numpy()[active_numpy]
            )
            metrics["residual_action_values"].append(
                residual_abs
            )
            metrics["base_action_l2_values"].append(
                base_l2
            )
            metrics["residual_action_l2_values"].append(
                residual_l2
            )
            metrics["residual_to_base_ratio_values"].append(
                residual_l2 / np.maximum(base_l2, 1e-12)
            )
            metrics["residual_to_base_ratio_indices"].append(
                np.flatnonzero(active_numpy)
            )
        metrics["active_steps"][active_numpy] += 1
        metrics["reward_sum"][active_numpy] += reward.cpu().numpy()[active_numpy]
        metrics["abs_yaw_error_sum"][active_numpy] += yaw_error.abs().cpu().numpy()[
            active_numpy
        ]
        metrics["abs_pitch_error_sum"][active_numpy] += pitch_error.abs().cpu().numpy()[
            active_numpy
        ]
        metrics["abs_speed_error_sum"][active_numpy] += speed_error.abs().cpu().numpy()[
            active_numpy
        ]
        metrics["yaw_error_sq_sum"][active_numpy] += yaw_error.square().cpu().numpy()[
            active_numpy
        ]
        metrics["pitch_error_sq_sum"][active_numpy] += pitch_error.square().cpu().numpy()[
            active_numpy
        ]
        metrics["speed_error_sq_sum"][active_numpy] += speed_error.square().cpu().numpy()[
            active_numpy
        ]
        metrics["abs_roll_sum"][active_numpy] += roll.cpu().numpy()[active_numpy]
        metrics["abs_roll_max"][active_numpy] = np.maximum(
            metrics["abs_roll_max"][active_numpy],
            roll.cpu().numpy()[active_numpy],
        )
        metrics["roll_abs_values"].append(roll.cpu().numpy()[active_numpy])
        metrics["roll_rate_abs_values"].append(roll_rate.cpu().numpy()[active_numpy])
        if path_enabled:
            if projection is None:
                raise RuntimeError("C2-C6 evaluation requires ALOS path projection")
            true_projection = inner.path.project(
                inner.eta[:, :3],
                segment_hint=true_projection_hint,
                search_radius=inner.reset_projection_search_radius,
            )
            true_projection_hint = true_projection.segment_index
            true_attitude_quat = quaternion_from_euler(
                inner.eta[:, 3], inner.eta[:, 4], inner.eta[:, 5]
            )
            ground_velocity_ned = body_to_world(ground_velocity, true_attitude_quat)
            route_speed_mps = torch.clamp(
                (ground_velocity_ned * true_projection.tangent_ned).sum(dim=1),
                min=0.0,
            ).cpu().numpy()
            route_speed_ratio = route_speed_mps / np.maximum(
                cruise_speed_mps, 1e-9
            )
            previous_min = metrics["route_ground_speed_ratio_min"][active_numpy]
            current_ratio = route_speed_ratio[active_numpy]
            lower = current_ratio < previous_min
            active_indices = np.flatnonzero(active_numpy)
            metrics["route_ground_speed_ratio_min"][active_numpy] = np.minimum(
                previous_min,
                current_ratio,
            )
            metrics["route_ground_speed_ratio_min_time_s"][
                active_indices[lower]
            ] = step_index * config.plant.dt_s
            metrics["route_ground_speed_ratio_sum"][active_numpy] += route_speed_ratio[
                active_numpy
            ]
            metrics["route_ground_speed_steps"][active_numpy] += 1
            metrics["path_steps"][active_numpy] += 1
            metrics["horizontal_error_sq_sum"][active_numpy] += (
                true_projection.horizontal_error_m.square().cpu().numpy()[active_numpy]
            )
            metrics["vertical_error_sq_sum"][active_numpy] += (
                true_projection.vertical_error_m.square().cpu().numpy()[active_numpy]
            )
            metrics["three_d_error_sq_sum"][active_numpy] += (
                true_projection.error_3d_m.square().cpu().numpy()[active_numpy]
            )
            pre_goal_mask = (
                active_numpy
                & ~metrics["endpoint_entry_reached"]
                & ~endpoint_now
            )
            metrics["pre_goal_path_steps"][pre_goal_mask] += 1
            metrics["pre_goal_three_d_error_sq_sum"][pre_goal_mask] += (
                true_projection.error_3d_m.square().cpu().numpy()[pre_goal_mask]
            )
            metrics["final_progress_m"][active_numpy] = (
                true_projection.progress_m.cpu().numpy()[active_numpy]
            )
            metrics["alos_horizontal_error_sq_sum"][active_numpy] += (
                projection.horizontal_error_m.square().cpu().numpy()[active_numpy]
            )
            metrics["alos_vertical_error_sq_sum"][active_numpy] += (
                projection.vertical_error_m.square().cpu().numpy()[active_numpy]
            )
            metrics["alos_three_d_error_sq_sum"][active_numpy] += (
                projection.error_3d_m.square().cpu().numpy()[active_numpy]
            )
            metrics["alos_final_progress_m"][active_numpy] = (
                projection.progress_m.cpu().numpy()[active_numpy]
            )
            if trace is not None:
                record_trace("route_speed_mps", route_speed_mps)
                record_trace("route_tangent_ned", true_projection.tangent_ned)
                record_trace("path_progress_m", true_projection.progress_m)
                # Goal170 eval-only instrumentation: retain per-step route error
                # under the same names used by the SAC evaluator so both policy
                # families share one route-health analyzer.
                record_trace("path_progress", true_projection.progress_m)
                record_trace("path_error_3d", true_projection.error_3d_m)
                record_trace("path_horizontal_error", true_projection.horizontal_error_m)
                record_trace("path_vertical_error", true_projection.vertical_error_m)
        metrics["material_saturation_steps"][active_numpy] += infos[
            "material_saturated"
        ].cpu().numpy()[active_numpy]
        hold = infos["hold_fraction"].cpu().numpy()
        metrics["max_hold_fraction"][active_numpy] = np.maximum(
            metrics["max_hold_fraction"][active_numpy], hold[active_numpy]
        )
        metrics["final_hold_fraction"][active_numpy] = hold[active_numpy]
        metrics["sensor_p_nav_error_m"].append(
            torch.linalg.norm(sensor.p_nav - inner.eta[:, :3], dim=1)
            .cpu()
            .numpy()[active_numpy]
        )
        metrics["sensor_dvl_bottom_valid"].append(
            sensor.bottom_lock_valid.cpu().numpy()[active_numpy]
        )
        metrics["sensor_current_estimate_valid"].append(
            sensor.current_estimate_valid.cpu().numpy()[active_numpy]
        )
        metrics["sensor_usbl_valid"].append(
            sensor.usbl_valid.cpu().numpy()[active_numpy]
        )
        metrics["sensor_usbl_outlier"].append(
            sensor.usbl_outlier.cpu().numpy()[active_numpy]
        )
        metrics["sensor_nav_age_s"].append(
            sensor.nav_age_s.cpu().numpy()[active_numpy]
        )
        metrics["sensor_dvl_bottom_age_s"].append(
            sensor.dvl_bottom_age_s.cpu().numpy()[active_numpy]
        )
        metrics["sensor_current_estimate_age_s"].append(
            sensor.current_estimate_age_s.cpu().numpy()[active_numpy]
        )
        guidance_current_valid = infos["guidance_current_valid"].cpu().numpy()
        guidance_current_age = infos["guidance_current_age_s"].cpu().numpy()
        guidance_crab_deg = np.degrees(
            infos["guidance_crab_angle_rad"].cpu().numpy()
        )
        guidance_speed_scale = infos["guidance_speed_scale"].cpu().numpy()
        guidance_terminal_phase = (
            infos["guidance_terminal_phase"].cpu().numpy() > 0.5
        )
        guidance_terminal_ratio = infos[
            "guidance_terminal_speed_ratio"
        ].cpu().numpy()
        guidance_saturation_ema = infos["guidance_saturation_ema"].cpu().numpy()
        guidance_saturation_emergency = (
            infos["guidance_saturation_emergency"].cpu().numpy() > 0.5
        )
        metrics["guidance_current_valid"].append(
            guidance_current_valid[active_numpy]
        )
        metrics["guidance_current_age_s"].append(
            guidance_current_age[active_numpy]
        )
        metrics["guidance_crab_angle_deg"].append(guidance_crab_deg[active_numpy])
        metrics["guidance_speed_scale"].append(
            guidance_speed_scale[active_numpy]
        )
        metrics["guidance_terminal_phase"].append(
            guidance_terminal_phase[active_numpy]
        )
        metrics["guidance_terminal_speed_ratio"].append(
            guidance_terminal_ratio[active_numpy]
        )
        terminal_active = active_numpy & guidance_terminal_phase
        if bool(terminal_active.any()):
            metrics["guidance_terminal_speed_ratio_values"].append(
                guidance_terminal_ratio[terminal_active]
            )
        metrics["guidance_saturation_ema"].append(
            guidance_saturation_ema[active_numpy]
        )
        metrics["guidance_saturation_emergency"].append(
            guidance_saturation_emergency[active_numpy]
        )
        metrics["guidance_saturation_emergency_steps"][
            active_numpy & guidance_saturation_emergency
        ] += 1
        metrics["guidance_route_floor_active"].append(
            (infos["guidance_route_speed_floor_active"].cpu().numpy() > 0.5)[
                active_numpy
            ]
        )
        metrics["guidance_route_floor_launch_active"].append(
            (
                infos["guidance_route_speed_floor_launch_active"]
                .cpu()
                .numpy()
                > 0.5
            )[active_numpy]
        )
        metrics["guidance_route_floor_valid"].append(
            (infos["guidance_route_speed_floor_valid"].cpu().numpy() > 0.5)[
                active_numpy
            ]
        )
        metrics["guidance_route_floor_boost_mps"].append(
            infos["guidance_route_speed_floor_boost_mps"].cpu().numpy()[
                active_numpy
            ]
        )
        metrics["guidance_route_floor_ratio"].append(
            infos["guidance_route_speed_floor_ratio"].cpu().numpy()[active_numpy]
        )

        if trace is not None:
            goal_distance = torch.linalg.norm(
                inner.eta[:, :3] - inner.path.goal_ned_m_for(len(seeds)), dim=1
            )
            record_trace("step", np.full(len(seeds), step_index, dtype=np.int64))
            record_trace(
                "time_s", np.full(len(seeds), step_index * config.plant.dt_s)
            )
            assert inner.episode_seed is not None
            record_trace("episode_seed", inner.episode_seed)
            record_trace("seed", inner.episode_seed)
            record_trace("eta", inner.eta)
            record_trace("nu", inner.nu)
            record_trace("current_ned_mps", inner.current_ned_mps)
            record_trace(
                "command",
                torch.stack(
                    [command.heading_rad, command.pitch_rad, command.speed_mps],
                    dim=1,
                ),
            )
            if infos.get("command_residual_applied") is not None:
                record_trace(
                    "command_residual_applied", infos["command_residual_applied"]
                )
                record_trace(
                    "command_base_physical", infos["command_base_physical"]
                )
                record_trace(
                    "command_corrected_physical", infos["command_corrected_physical"]
                )
            if infos.get("actuator_residual_applied") is not None:
                record_trace(
                    "actuator_residual_applied",
                    infos["actuator_residual_applied"],
                )
                record_trace(
                    "actuator_residual_effective",
                    infos["actuator_residual_effective"],
                )
                record_trace("actuator_base_physical", infos["actuator_base_physical"])
                record_trace(
                    "actuator_total_physical", infos["actuator_total_physical"]
                )
                if infos.get("actuator_residual_scale") is not None:
                    record_trace(
                        "actuator_residual_scale",
                        infos["actuator_residual_scale"],
                    )
                if infos.get("actuator_residual_schedule_distance_m") is not None:
                    record_trace(
                        "actuator_residual_schedule_distance_m",
                        infos["actuator_residual_schedule_distance_m"],
                    )
                if infos.get("actuator_residual_channel_gains") is not None:
                    record_trace(
                        "actuator_residual_effective_channel_gains",
                        infos["actuator_residual_channel_gains"],
                    )
                if infos.get("actuator_residual_channel_gain_near_gate") is not None:
                    record_trace(
                        "actuator_residual_channel_gain_near_gate",
                        infos["actuator_residual_channel_gain_near_gate"],
                    )
                if infos.get("actuator_residual_ppo_applied") is not None:
                    record_trace(
                        "actuator_residual_bc_applied",
                        infos["actuator_residual_bc_applied"],
                    )
                    record_trace(
                        "actuator_residual_bc_effective",
                        infos["actuator_residual_bc_effective"],
                    )
                    record_trace(
                        "actuator_residual_ppo_applied",
                        infos["actuator_residual_ppo_applied"],
                    )
                    record_trace(
                        "actuator_residual_ppo_effective",
                        infos["actuator_residual_ppo_effective"],
                    )
            if ppo_residual_attenuation is not None:
                record_trace(
                    "ppo_residual_attenuation",
                    ppo_residual_attenuation,
                )
                record_trace(
                    "ppo_residual_attenuation_distance_m",
                    ppo_residual_attenuation_distance,
                )
            if infos.get("frozen_residual_gate") is not None:
                record_trace(
                    "frozen_residual_base_action",
                    infos["frozen_residual_base_action"],
                )
                record_trace(
                    "frozen_residual_action", infos["frozen_residual_action"]
                )
                record_trace("frozen_residual_gate", infos["frozen_residual_gate"])
                record_trace(
                    "frozen_residual_executed_action",
                    infos["frozen_residual_executed_action"],
                )
            record_trace("ground_velocity_body_mps", ground_velocity)
            record_trace("ground_speed_mps", torch.linalg.norm(ground_velocity, dim=1))
            record_trace("goal_distance_m", goal_distance)
            record_trace("p_nav", sensor.p_nav)
            record_trace("goal_ned_m", goal_ned_m)
            record_trace("goal_tangent_ned", goal_tangent_ned)
            record_trace("goal_distance_pnav_m", endpoint_goal_distance_m)
            record_trace("goal_distance_eta_m", eta_goal_distance_m)
            record_trace(
                "signed_goal_tangent_offset_pnav_m", signed_goal_tangent_offset_m
            )
            record_trace(
                "signed_goal_tangent_offset_eta_m",
                signed_goal_tangent_offset_eta_m,
            )
            record_trace("endpoint_entry_reached", endpoint_now)
            record_trace("terminal_kind", infos["terminal_kind"])
            record_trace("cruise_speed_mps", infos["episode_command_speed_mps"])
            record_trace("heading_error_rad", yaw_error)
            record_trace("pitch_error_rad", pitch_error)
            record_trace("speed_error_mps", speed_error)
            record_trace("requested_action", action)
            if base_action is not None:
                record_trace("base_action", base_action)
            if residual_action is not None:
                record_trace("residual_action", residual_action)
            if inner.actuator is not None:
                record_trace("actual_actuator", inner.actuator)
            record_trace("material_saturated", infos["material_saturated"])
            record_trace(
                "guidance_speed_target_mps",
                infos["guidance_route_speed_floor_target_mps"],
            )
            record_trace(
                "guidance_route_floor_filtered_mps",
                infos["guidance_route_speed_floor_filtered_mps"],
            )
            record_trace(
                "guidance_route_floor_boost_mps",
                infos["guidance_route_speed_floor_boost_mps"],
            )
            record_trace(
                "guidance_route_floor_active",
                infos["guidance_route_speed_floor_active"],
            )
            record_trace(
                "guidance_route_floor_valid",
                infos["guidance_route_speed_floor_valid"],
            )
            record_trace("guidance_terminal_phase", infos["guidance_terminal_phase"])
            record_trace(
                "guidance_terminal_speed_ratio",
                infos["guidance_terminal_speed_ratio"],
            )
            record_trace(
                "guidance_target_heading_rad",
                infos["guidance_target_heading_rad"],
            )
            record_trace(
                "guidance_target_pitch_rad",
                infos["guidance_target_pitch_rad"],
            )
            record_trace(
                "guidance_target_speed_mps",
                infos["guidance_target_speed_mps"],
            )
            record_trace(
                "guidance_limited_heading_rad",
                infos["guidance_limited_heading_rad"],
            )
            record_trace(
                "guidance_limited_pitch_rad",
                infos["guidance_limited_pitch_rad"],
            )
            record_trace(
                "guidance_limited_speed_mps",
                infos["guidance_limited_speed_mps"],
            )
            record_trace(
                "guidance_heading_rate_clipped",
                infos["guidance_heading_rate_clipped"],
            )
            record_trace(
                "guidance_pitch_rate_clipped",
                infos["guidance_pitch_rate_clipped"],
            )
            record_trace(
                "guidance_speed_rate_clipped",
                infos["guidance_speed_rate_clipped"],
            )
            record_trace("dvl_ground_body", sensor.dvl_ground_body)
            record_trace("dvl_bottom_valid", sensor.bottom_lock_valid)
            record_trace("current_estimate_body", sensor.current_estimate_body)
            record_trace("current_estimate_valid", sensor.current_estimate_valid)
            record_trace("current_estimate_age_s", sensor.current_estimate_age_s)
            _record_trace_sensor_ages(
                record_trace, sensor, enabled=trace_sensor_ages
            )
            _record_trace_estimator_state(
                record_trace, sensor, enabled=trace_estimator_state
            )
            if trace_reward_breakdown:
                for name, value in _reward_breakdown_trace_fields(infos).items():
                    record_trace(name, value)

        terminal = infos["terminal_kind"].cpu().numpy()
        finished = active_numpy & (done.cpu().numpy() > 0.5)
        goal = finished & (terminal == int(PCTerminalKind.GOAL))
        plane_miss = finished & (terminal == int(PCTerminalKind.GOAL_PLANE_MISS))
        timeout = finished & (terminal == int(PCTerminalKind.TIME_LIMIT))
        safety = finished & ~goal & ~plane_miss & ~timeout
        metrics["goal_terminations"][goal] += 1
        metrics["goal_plane_miss_terminations"][plane_miss] += 1
        metrics["timeout_terminations"][timeout] += 1
        metrics["safety_terminations"][safety] += 1
        metrics["miss_distance_at_plane_m"][plane_miss] = endpoint_goal_distance_m[
            plane_miss
        ]
        if not endpoint_stop_enabled:
            terminal_mask = active_numpy & metrics["endpoint_entry_reached"]
            metrics["terminal_longitudinal_steps"][terminal_mask] += 1
            metrics["terminal_longitudinal_sq_sum"][terminal_mask] += np.square(
                signed_goal_tangent_offset_m[terminal_mask]
            )
            overshoot = np.maximum(
                0.0, signed_goal_tangent_offset_m[terminal_mask]
            )
            previous = metrics["post_goal_overshoot_m"][terminal_mask]
            metrics["post_goal_overshoot_m"][terminal_mask] = np.where(
                np.isfinite(previous),
                np.maximum(previous, overshoot),
                overshoot,
            )
        if bool(finished.any()):
            goal_distance = torch.linalg.norm(
                inner.eta[:, :3] - inner.path.goal_ned_m_for(len(seeds)), dim=1
            ).detach().cpu().numpy()
            path_length = inner.path.total_length_m_for(len(seeds)).detach().cpu().numpy()
            metrics["final_goal_distance_m"][finished] = goal_distance[finished]
            metrics["final_goal_distance_set"][finished] = True
            metrics["pnav_final_goal_distance_m"][finished] = pnav_goal_distance_m[
                finished
            ]
            metrics["eta_goal_capture_at_done"][finished] = (
                goal_distance[finished] <= config.plant.goal_radius_m + 1e-12
            )
            metrics["path_overshoot_m"][finished] = np.maximum(
                metrics["final_progress_m"][finished] - path_length[finished], 0.0
            )
        metrics["done"][finished] = True

    if not bool(metrics["done"].all()):
        raise RuntimeError("some PC episodes did not terminate")
    if trace is not None:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_output,
            **{
                name: np.concatenate(values, axis=0) if values else np.empty(0)
                for name, values in trace.items()
            },
        )
    path_length_m = inner.path.total_length_m
    summary = _summary(
        metrics,
        path_enabled=path_enabled,
        path_length_m=path_length_m,
        action_full_scale=metadata["action_full_scale"],
        speed_envelope_min_mps=speed_envelope_min_mps,
        speed_envelope_max_mps=speed_envelope_max_mps,
        control_dt_s=config.plant.dt_s,
        endpoint_stop_enabled=endpoint_stop_enabled,
        goal_radius_m=config.plant.goal_radius_m,
        success_frame=success_frame,
    )
    episodes = _episode_records(
        metrics,
        seeds,
        path_enabled=path_enabled,
        path_length_m=path_length_m,
        action_full_scale=metadata["action_full_scale"],
        control_dt_s=config.plant.dt_s,
        endpoint_stop_enabled=endpoint_stop_enabled,
        goal_radius_m=config.plant.goal_radius_m,
        success_frame=success_frame,
        plant_scale_factors=(
            None
            if inner.plant_scales is None
            else inner.plant_scales.detach().cpu().numpy()
        ),
    )
    if inner.curriculum_specs is not None:
        if len(inner.curriculum_specs) != len(episodes):
            raise ValueError("curriculum specs do not align with evaluation episodes")
        for index, (row, spec) in enumerate(
            zip(episodes, inner.curriculum_specs, strict=True)
        ):
            row["curriculum"] = curriculum_episode_metadata(
                spec,
                initial_heading_rad=float(initial_heading_rad[index]),
            )
    contract = {
        "endpoint_stop_enabled": endpoint_stop_enabled,
        "sensor_override": sensor_override,
        "actor_observation_source": (
            "ideal" if config.observation.ideal_actor_observations else "nominal"
        ),
        "actor_observation_override": bool(
            base_config is not None
            and config.observation.ideal_actor_observations
            != base_config.observation.ideal_actor_observations
        ),
        "base_sensor": (
            None
            if base_config is None or base_config.sensors is None
            else {
                "profile": base_config.sensors.profile,
                "seed": base_config.sensors.seed,
            }
        ),
        "override_sensor": (
            None
            if config.sensors is None
            else {
                "profile": config.sensors.profile,
                "seed": config.sensors.seed,
            }
        ),
    }
    return summary, metadata, episodes, contract


def main() -> int:
    args = _parse_args()
    if args.seed_start < 0 or args.seed_count <= 0:
        raise ValueError("seed start must be non-negative and seed count positive")
    _validate_trace_sensor_ages(args.trace_output, args.trace_sensor_ages)
    _validate_trace_estimator_state(
        args.trace_output, args.trace_estimator_state
    )
    _validate_trace_reward_breakdown(
        args.trace_output, args.trace_reward_breakdown
    )
    training_config_path = _resolve(args.training_config)
    checkpoint_path = _resolve(args.checkpoint)
    training_config = load_training_config(training_config_path)
    environment_config_path = (
        None if args.environment_config is None else _resolve(args.environment_config)
    )
    guidance_config_path = (
        None if args.guidance_config is None else _resolve(args.guidance_config)
    )
    controller_config_path = (
        None if args.controller_config is None else _resolve(args.controller_config)
    )
    selected_environment_config = (
        _resolve(Path(training_config.environment_config))
        if environment_config_path is None
        else environment_config_path
    )
    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.curriculum_current_speed_range_mps is not None:
        if args.curriculum_stage is None:
            raise ValueError(
                "--curriculum-current-speed-range-mps requires --curriculum-stage"
            )
        speed_min, speed_max = (
            float(value) for value in args.curriculum_current_speed_range_mps
        )
        if not (np.isfinite(speed_min) and np.isfinite(speed_max)):
            raise ValueError("curriculum current speed range must be finite")
        if speed_min < 0.0 or speed_max < speed_min:
            raise ValueError(
                "curriculum current speed range requires 0 <= MIN <= MAX"
            )
    if args.curriculum_shear_scale is not None:
        if args.curriculum_stage is None:
            raise ValueError("--curriculum-shear-scale requires --curriculum-stage")
        if not np.isfinite(args.curriculum_shear_scale) or args.curriculum_shear_scale < 0.0:
            raise ValueError("curriculum shear scale must be finite and non-negative")
    summary, metadata, episodes, contract = _run(
        training_config_path=training_config_path,
        checkpoint_path=checkpoint_path,
        environment_config_path=environment_config_path,
        seeds=seeds,
        device=torch.device(args.device),
        dtype=dtype,
        history_ablation=args.history_ablation,
        path_ablation=args.path_ablation,
        action_mode=args.action_mode,
        injected_residual_profile=args.injected_residual_profile,
        allow_stage_override=args.allow_stage_override,
        allow_sensor_override=args.allow_sensor_override,
        allow_actor_observation_override=args.allow_actor_observation_override,
        allow_guidance_rate_limit_override=args.allow_guidance_rate_limit_override,
        allow_diversity_environment_override=(
            args.allow_diversity_environment_override
        ),
        allow_start_progress_training_override=(
            args.allow_start_progress_training_override
        ),
        curriculum_stage=args.curriculum_stage,
        disable_curriculum_direct_command=args.disable_curriculum_direct_command,
        curriculum_current_speed_range_mps=(
            None
            if args.curriculum_current_speed_range_mps is None
            else tuple(float(value) for value in args.curriculum_current_speed_range_mps)
        ),
        curriculum_shear_scale=args.curriculum_shear_scale,
        guidance_config_path=guidance_config_path,
        controller_config_path=controller_config_path,
        trace_output=None if args.trace_output is None else _resolve(args.trace_output),
        trace_sensor_ages=args.trace_sensor_ages,
        trace_estimator_state=args.trace_estimator_state,
        trace_reward_breakdown=args.trace_reward_breakdown,
        success_frame=args.success_frame,
        actuator_residual_scale_override=args.actuator_residual_scale_override,
        actuator_residual_channel_gains=args.actuator_residual_channel_gains,
        actuator_residual_scale_near=args.actuator_residual_scale_near,
        actuator_residual_scale_far=args.actuator_residual_scale_far,
        actuator_residual_scale_switch_distance_m=(
            args.actuator_residual_scale_switch_distance_m
        ),
        actuator_residual_near_channel_gains=args.actuator_residual_near_channel_gains,
        actuator_residual_channel_gain_near_distance_m=(
            args.actuator_residual_channel_gain_near_distance_m
        ),
        actuator_residual_channel_gain_far_distance_m=(
            args.actuator_residual_channel_gain_far_distance_m
        ),
        terminal_gate_enter_m=args.terminal_gate_enter_m,
        terminal_gate_full_m=args.terminal_gate_full_m,
        ppo_residual_attenuation_full_m=args.ppo_residual_attenuation_full_m,
        ppo_residual_attenuation_enter_m=args.ppo_residual_attenuation_enter_m,
    )
    selected_config = load_config(selected_environment_config)
    selected_stage = str(selected_config.pc_stage)
    effective_guidance = selected_config.guidance
    if guidance_config_path is not None:
        effective_guidance = load_config(guidance_config_path).guidance
    effective_controller = selected_config.controller
    if controller_config_path is not None:
        effective_controller = load_config(controller_config_path).controller
    report = {
        "schema_version": "auv_pc_ppo_evaluation.v1",
        "training_config": str(training_config_path),
        "environment_config": str(selected_environment_config),
        "curriculum_stage": args.curriculum_stage,
        "curriculum_current_speed_range_override_mps": (
            None
            if args.curriculum_current_speed_range_mps is None
            else [
                float(value)
                for value in args.curriculum_current_speed_range_mps
            ]
        ),
        "curriculum_shear_scale_override": args.curriculum_shear_scale,
        "curriculum_direct_commands_enabled": (
            not args.disable_curriculum_direct_command
        ),
        "environment_config_override": (
            None if environment_config_path is None else str(environment_config_path)
        ),
        "guidance_config": (
            None if guidance_config_path is None else str(guidance_config_path)
        ),
        "controller_config": (
            None if controller_config_path is None else str(controller_config_path)
        ),
        "guidance_rate_limit_override": args.allow_guidance_rate_limit_override,
        "diversity_environment_override": (
            args.allow_diversity_environment_override
        ),
        "start_progress_training_override": (
            args.allow_start_progress_training_override
        ),
        "guidance_command_rate_limit_enabled": (
            effective_guidance.command_rate_limit_enabled
        ),
        "guidance_pitch_rate_limit_rad_s": (
            effective_guidance.pitch_rate_limit_rad_s
        ),
        "guidance_heading_rate_limit_rad_s": (
            effective_guidance.heading_rate_limit_rad_s
        ),
        "guidance_speed_rate_limit_mps2": (
            effective_guidance.speed_rate_limit_mps2
        ),
        "guidance_terminal_braking_enabled": (
            effective_guidance.terminal_braking_enabled
        ),
        "guidance_terminal_speed_ratio": effective_guidance.terminal_speed_ratio,
        "guidance_vertical_gain": effective_guidance.vertical_gain,
        "guidance_vertical_lookahead_m": effective_guidance.vertical_lookahead_m,
        "controller_pitch_gain": effective_controller.pitch_gain,
        "controller_pitch_rate_gain": effective_controller.pitch_rate_gain,
        "trace_output": (
            None if args.trace_output is None else str(_resolve(args.trace_output))
        ),
        "trace_reward_breakdown": args.trace_reward_breakdown,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": metadata,
        "plant_randomization": selected_config.plant_randomization.as_dict(),
        "seeds": list(seeds),
        "actor_decision_stride": training_config.actor_decision_stride,
        "history_ablation": args.history_ablation,
        "path_ablation": args.path_ablation,
        "action_mode": args.action_mode,
        "injected_residual_profile": {
            "name": args.injected_residual_profile[0],
            "const": (
                list(args.injected_residual_profile[1])
                if args.injected_residual_profile[1] is not None
                else None
            ),
        },
        "actuator_residual_scale_override": args.actuator_residual_scale_override,
        "actuator_residual_channel_gains": (
            list(args.actuator_residual_channel_gains)
            if args.actuator_residual_channel_gains is not None
            else [1.0, 1.0, 1.0]
        ),
        "actuator_residual_scale_near": args.actuator_residual_scale_near,
        "actuator_residual_scale_far": args.actuator_residual_scale_far,
        "actuator_residual_scale_switch_distance_m": (
            args.actuator_residual_scale_switch_distance_m
        ),
        "actuator_residual_scale_schedule_enabled": (
            args.actuator_residual_scale_near is not None
        ),
        "actuator_residual_channel_gains_near": (
            list(args.actuator_residual_near_channel_gains)
            if args.actuator_residual_near_channel_gains is not None
            else None
        ),
        "actuator_residual_channel_gain_near_distance_m": (
            args.actuator_residual_channel_gain_near_distance_m
        ),
        "actuator_residual_channel_gain_far_distance_m": (
            args.actuator_residual_channel_gain_far_distance_m
        ),
        "actuator_residual_channel_gain_schedule_enabled": (
            args.actuator_residual_near_channel_gains is not None
        ),
        "terminal_gate_enter_m": (
            args.terminal_gate_enter_m
            if args.action_mode == "gated-residual"
            else None
        ),
        "terminal_gate_full_m": (
            args.terminal_gate_full_m
            if args.action_mode == "gated-residual"
            else None
        ),
        "ppo_residual_attenuation_enabled": (
            args.action_mode == "attenuated-residual"
        ),
        "ppo_residual_attenuation_full_m": (
            args.ppo_residual_attenuation_full_m
            if args.action_mode == "attenuated-residual"
            else None
        ),
        "ppo_residual_attenuation_enter_m": (
            args.ppo_residual_attenuation_enter_m
            if args.action_mode == "attenuated-residual"
            else None
        ),
        "pc_action_interface": metadata.get("pc", {}).get("action_interface"),
        "success_frame": args.success_frame,
        "stage_override": metadata["pc_stage"] != selected_stage,
        "sensor_override": contract["sensor_override"],
        "actor_observation_source": contract["actor_observation_source"],
        "actor_observation_override": contract["actor_observation_override"],
        "endpoint_stop_enabled": contract["endpoint_stop_enabled"],
        "base_sensor": contract["base_sensor"],
        "override_sensor": contract["override_sensor"],
        "metric_definitions": {
            "path_*": "projection of true eta position onto the reference path",
            "alos_projection_*": "ALOS projection of p_nav; diagnostic only",
            "nav_position_error_*": "p_nav minus true NED position",
            "dvl_bottom_outage_rate": "fraction of control steps with bottom-lock invalid",
            "current_estimate_valid_rate": "fraction of control steps with a valid current estimate",
            "guidance_current_valid_rate": "fraction of steps whose deployment current estimate passed the crab gate",
            "terminal_speed_ratio_mean": "mean command speed divided by episode cruise speed on terminal-capture steps",
            "guidance_route_speed_ratio_min": "minimum command speed divided by episode cruise speed over all active control steps",
            "ground_speed_ratio_min": "minimum measured ground-speed magnitude divided by episode cruise speed over all active control steps",
            "ground_speed_ratio_mean": "mean measured ground-speed magnitude divided by episode cruise speed over all active control steps",
            "ground_route_speed_ratio_min": "minimum measured route-progress speed divided by episode cruise speed over all active control steps",
            "ground_route_speed_ratio_episode_min_mean": "mean over episodes of each episode's minimum measured route-progress speed divided by its cruise speed",
            "ground_route_speed_ratio_mean": "mean measured route-progress speed divided by episode cruise speed over all active control steps",
            "guidance_crab_deg_mean": "signed crab heading offset requested by ALOS-R",
            "guidance_saturation_ema_mean": "EMA of material actuator saturation used by the speed governor",
            "guidance_saturation_emergency_rate": "fraction of steps with sustained saturation protection active",
            "usbl_outlier_rate": "fraction of control steps flagged as a raw USBL outlier",
            "pre_goal_3d_rmse_m": "true-position 3D path RMSE before the goal radius or terminal tangent plane is reached",
            "goal_plane_miss_rate": "fraction of episodes terminated by missing the terminal goal plane",
            "miss_distance_at_plane_m": "true or estimated goal distance at a goal-plane-miss termination under the selected terminal protocol",
            "post_goal_overshoot_m": "mean positive terminal tangent offset after endpoint entry; legacy termination only",
            "terminal_longitudinal_rmse_m": "RMSE of signed terminal tangent offset after endpoint entry; legacy termination only",
            "success_pnav_tight": "deployment success: p_nav tight-radius GOAL termination",
            "success_eta_tight_ever": "research primary: true eta ever entered tight radius",
            "success_eta_tight_at_done": "secondary: true eta at done remained inside tight radius",
        },
        "metrics": summary,
        "episodes": episodes,
    }
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PC20 PPO closed-loop evaluation",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- environment config: `{report['environment_config']}`",
        f"- seeds: `{len(seeds)}`",
        "- `path_*`: true `eta` position; `alos_projection_*`: navigation-position diagnostic",
    ]
    lines.extend(
        f"- `{name}`: `{_format_summary_value(value)}`"
        for name, value in summary.items()
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _format_summary_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return format(value, ".6g")


if __name__ == "__main__":
    raise SystemExit(main())
