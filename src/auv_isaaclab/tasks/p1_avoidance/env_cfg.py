"""Configuration entrypoint for the external P1 IsaacLab task.

The implementation of the DirectRLEnv lives in ``env.py`` (G3).  Keeping the
config in a separate module lets G2 validate JSON and derived dimensions without
starting Omniverse or importing IsaacLab.
"""

from __future__ import annotations

from auv_isaaclab.config import (
    ConfigBundle,
    ConfigError,
    CurriculumSection,
    EnvSection,
    EvaluationSection,
    ExperimentSection,
    ObservationSection,
    P1AvoidanceEnvCfg,
    PpoSection,
    RewardSection,
    load_config,
)

__all__ = [
    "ConfigBundle",
    "ConfigError",
    "CurriculumSection",
    "EnvSection",
    "EvaluationSection",
    "ExperimentSection",
    "ObservationSection",
    "P1AvoidanceEnvCfg",
    "PpoSection",
    "RewardSection",
    "load_config",
]
