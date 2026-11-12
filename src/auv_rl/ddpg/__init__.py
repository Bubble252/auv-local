"""DDPG components used by the P1 training runner.

The package is intentionally imported only by training entry points so the
planning and perception contracts remain usable without PyTorch installed.
"""

from .agent import DDPGAgent, DDPGConfig
from .networks import Critic, NoisyLinear, PolicyActor
from .replay import PrioritizedSequenceReplayBuffer, SumTree
from .trainer import evaluate_ddpg, train_ddpg
from .vector_trainer import (
    VectorEvaluationMetrics,
    VectorTrainingMetrics,
    evaluate_parallel_ddpg,
    train_parallel_ddpg,
)

__all__ = [
    "Critic",
    "DDPGAgent",
    "DDPGConfig",
    "NoisyLinear",
    "PolicyActor",
    "PrioritizedSequenceReplayBuffer",
    "SumTree",
    "VectorTrainingMetrics",
    "VectorEvaluationMetrics",
    "evaluate_ddpg",
    "evaluate_parallel_ddpg",
    "train_parallel_ddpg",
    "train_ddpg",
]
