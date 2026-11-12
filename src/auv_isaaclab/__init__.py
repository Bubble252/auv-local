"""External IsaacLab task package for the P1 PPO route.

This package lives outside the IsaacLab source tree.  The configuration layer
is deliberately importable without IsaacLab so that JSON contracts can be
validated on a machine without a GPU or a running Omniverse app.
"""

from .config import (
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
