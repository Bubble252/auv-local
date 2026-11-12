from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .train_bc import build_sequence_dataset

if TYPE_CHECKING:
    from ..ddpg.agent import DDPGAgent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BCInitializationReport:
    actor_checkpoint: str | None
    actor_checkpoint_sha256: str | None
    demos: str | None
    demo_sha256: str | None
    replay_transitions: int
    auxiliary_samples: int

    @property
    def teacher_used(self) -> bool:
        return self.actor_checkpoint is not None or self.demos is not None

    @property
    def bc_pretrained(self) -> bool:
        return self.actor_checkpoint is not None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "teacher_used": self.teacher_used,
            "bc_pretrained": self.bc_pretrained,
            "actor_checkpoint": self.actor_checkpoint,
            "actor_checkpoint_sha256": self.actor_checkpoint_sha256,
            "demo_dataset": self.demos,
            "demo_sha256": self.demo_sha256,
            "replay_transitions": self.replay_transitions,
            "auxiliary_samples": self.auxiliary_samples,
            "oracle_in_actor": False,
        }


def initialize_agent_from_bc(
    agent: DDPGAgent,
    *,
    actor_checkpoint: Path | str | None,
    demos: Path | str | None,
    history_length: int,
    expected_observation_version: str,
) -> BCInitializationReport:
    """Load a BC actor and optional demo replay data into a DDPG agent."""

    if history_length <= 0:
        raise ValueError("history_length must be positive")
    checkpoint_path = None if actor_checkpoint is None else Path(actor_checkpoint)
    demo_path = None if demos is None else Path(demos)
    replay_transitions = 0
    auxiliary_samples = 0

    if checkpoint_path is not None:
        agent.load_actor_checkpoint(
            checkpoint_path,
            expected_observation_version=expected_observation_version,
        )
        # BC training is deterministic; its NoisyNet sigma parameters are not
        # meaningful for exploration, so restore the configured sigma values.
        agent.reset_actor_noise_sigmas()
    if demo_path is not None:
        payload = np.load(demo_path, allow_pickle=False)
        required = {"observations", "actions", "rewards", "next_observations", "dones"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"demo dataset is missing fields: {sorted(missing)}")
        use_privileged = agent.config.privileged_dim > 0 and "privileged" in payload.files
        agent.replay.add_batch(
            np.asarray(payload["observations"], dtype=np.float32),
            np.asarray(payload["actions"], dtype=np.float32),
            np.asarray(payload["rewards"], dtype=np.float32),
            np.asarray(payload["next_observations"], dtype=np.float32),
            np.asarray(payload["dones"], dtype=np.bool_),
            episode_ids=np.asarray(payload["episode_ids"], dtype=np.int64),
            episode_steps=np.asarray(payload["episode_steps"], dtype=np.int64),
            privileged=(
                np.asarray(payload["privileged"], dtype=np.float32) if use_privileged else None
            ),
            next_privileged=(
                np.asarray(payload["next_privileged"], dtype=np.float32) if use_privileged else None
            ),
        )
        replay_transitions = int(len(payload["observations"]))
        dataset = build_sequence_dataset(demo_path, history_length=history_length)
        auxiliary_samples = int(len(dataset.train_observations))
        agent.configure_behavior_cloning(
            dataset.train_observations,
            dataset.train_actions,
        )

    return BCInitializationReport(
        actor_checkpoint=None if checkpoint_path is None else str(checkpoint_path),
        actor_checkpoint_sha256=None if checkpoint_path is None else _sha256(checkpoint_path),
        demos=None if demo_path is None else str(demo_path),
        demo_sha256=None if demo_path is None else _sha256(demo_path),
        replay_transitions=replay_transitions,
        auxiliary_samples=auxiliary_samples,
    )


__all__ = ["BCInitializationReport", "initialize_agent_from_bc"]
