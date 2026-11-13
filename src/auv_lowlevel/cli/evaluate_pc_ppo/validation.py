"""Validation helpers for evaluator overrides and trace controls."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.pc_command_residual import smooth_terminal_gate
from auv_lowlevel.batch.pc_path import PC_PATH_DEFAULT_BLOCKS

from .common import DIVERSITY_REFERENCE_OVERRIDE_FIELDS


def _validate_trace_sensor_ages(trace_output: Path | None, trace_sensor_ages: bool) -> None:
    if trace_sensor_ages and trace_output is None:
        raise ValueError("--trace-sensor-ages requires --trace-output")


def _validate_trace_estimator_state(trace_output: Path | None, trace_estimator_state: bool) -> None:
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
        raise ValueError("actuator residual schedule requires near, far, and switch distance")
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
            "actuator channel gain schedule requires near gains, near distance, and far distance"
        )
    if near_gains is None:
        return
    assert near_distance_m is not None
    assert far_distance_m is not None
    if not np.isfinite(near_distance_m) or not np.isfinite(far_distance_m):
        raise ValueError("actuator channel gain distances must be finite")
    if near_distance_m <= 0.0 or far_distance_m <= near_distance_m:
        raise ValueError("actuator channel gain distances require 0 < near_distance < far_distance")


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

    if not np.isfinite(full_m) or full_m <= 0.0 or not np.isfinite(enter_m) or enter_m <= full_m:
        raise ValueError("PPO attenuation requires finite positive full_m < enter_m")
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


def _record_trace_sensor_ages(record_trace: Any, sensor: Any, *, enabled: bool) -> None:
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


def _record_trace_estimator_state(record_trace: Any, sensor: Any, *, enabled: bool) -> None:
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
        raise ValueError("environment override may change only plant.goal_radius_m")
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
    if not allow_start_progress_training_override and (
        base.reference.start_progress_m != override.reference.start_progress_m
        or base.reference.start_progress_full_fraction
        != override.reference.start_progress_full_fraction
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
        raise ValueError("environment override may change only reference.command_speed_range_mps")


def _validate_controller_pitch_override(base: Any, override: Any) -> None:
    """Allow only pitch-loop gains to differ for an eval-only authority probe."""

    allowed = replace(
        base,
        pitch_gain=override.pitch_gain,
        pitch_rate_gain=override.pitch_rate_gain,
    )
    if _dataclass_differs(override, allowed):
        raise ValueError("controller override may change only pitch_gain and pitch_rate_gain")


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
    return tuple(block for block in training.pc.path_curriculum[-1][1] if block != "state_base")


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
    return base_mode == "frozen_pc_bc_residual" and base_output_mode == "residual"
