"""Stable exports for the refactored PC PPO evaluator."""

from .args import (
    _parse_args,
    _parse_injected_residual_profile,
    _parse_residual_channel_gains,
)
from .common import (
    DIVERSITY_REFERENCE_OVERRIDE_FIELDS,
    REPO_ROOT,
    TURN_CURVATURE_THRESHOLD,
    _resolve,
)
from .episodes import _episode_records
from .main import main
from .metrics import (
    _actuator_residual_summary,
    _command_residual_summary,
    _concat_series,
    _episode_series,
    _frozen_residual_summary,
    _new_episode_metrics,
    _ppo_residual_attenuation_summary,
    _residual_action_summary,
)
from .output import _format_summary_value
from .policy import (
    _build_curriculum_sampler,
    _build_policy,
    _restore_residual_norm_projection,
)
from .runner import _run
from .summary import _summary
from .validation import (
    _dataclass_differs,
    _final_path_blocks,
    _ppo_residual_attenuation,
    _record_trace_estimator_state,
    _record_trace_sensor_ages,
    _reward_breakdown_trace_fields,
    _stage_override_allowed,
    _trace_estimator_state_fields,
    _trace_sensor_age_fields,
    _uses_frozen_residual_wrapper,
    _validate_actuator_channel_gain_schedule,
    _validate_actuator_residual_scale_schedule,
    _validate_controller_pitch_override,
    _validate_environment_override,
    _validate_trace_estimator_state,
    _validate_trace_reward_breakdown,
    _validate_trace_sensor_ages,
)

__all__ = [
    "DIVERSITY_REFERENCE_OVERRIDE_FIELDS",
    "REPO_ROOT",
    "TURN_CURVATURE_THRESHOLD",
    "_actuator_residual_summary",
    "_build_curriculum_sampler",
    "_build_policy",
    "_command_residual_summary",
    "_concat_series",
    "_dataclass_differs",
    "_episode_records",
    "_episode_series",
    "_final_path_blocks",
    "_format_summary_value",
    "_frozen_residual_summary",
    "_new_episode_metrics",
    "_parse_args",
    "_parse_injected_residual_profile",
    "_parse_residual_channel_gains",
    "_ppo_residual_attenuation",
    "_ppo_residual_attenuation_summary",
    "_record_trace_estimator_state",
    "_record_trace_sensor_ages",
    "_residual_action_summary",
    "_resolve",
    "_restore_residual_norm_projection",
    "_reward_breakdown_trace_fields",
    "_run",
    "_stage_override_allowed",
    "_summary",
    "_trace_estimator_state_fields",
    "_trace_sensor_age_fields",
    "_uses_frozen_residual_wrapper",
    "_validate_actuator_channel_gain_schedule",
    "_validate_actuator_residual_scale_schedule",
    "_validate_controller_pitch_override",
    "_validate_environment_override",
    "_validate_trace_estimator_state",
    "_validate_trace_reward_breakdown",
    "_validate_trace_sensor_ages",
    "main",
]
