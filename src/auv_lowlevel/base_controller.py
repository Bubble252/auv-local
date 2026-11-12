from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .bc import BCPolicy
from .config import LowLevelConfig
from .observation_schema import observation_schema_for_dim

BaseMode = Literal["online_pd_pi", "frozen_bc"]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BaseControllerMetadata:
    mode: BaseMode
    checkpoint: str | None
    checkpoint_sha256: str | None
    frozen: bool


class BaseControllerAdapter:
    """Resolve an online PD/PI or a frozen BC policy without changing LowLevelEnv."""

    def __init__(
        self,
        config: LowLevelConfig,
        mode: BaseMode,
        *,
        checkpoint: Path | str | None = None,
        checkpoint_sha256: str | None = None,
        device: str = "cpu",
    ) -> None:
        if mode not in {"online_pd_pi", "frozen_bc"}:
            raise ValueError(f"unsupported base controller mode: {mode}")
        if mode == "frozen_bc" and checkpoint is None:
            raise ValueError("frozen_bc requires a checkpoint")
        if mode == "online_pd_pi" and checkpoint is not None:
            raise ValueError("online_pd_pi must not receive a checkpoint")
        self.config = config
        self.mode = mode
        self.device = device
        self.policy: BCPolicy | None = None
        self.checkpoint: Path | None = None if checkpoint is None else Path(checkpoint)
        self.checkpoint_sha256: str | None = None
        if self.checkpoint is not None:
            self.checkpoint_sha256 = sha256_file(self.checkpoint)
            if (
                checkpoint_sha256 is not None
                and self.checkpoint_sha256 != checkpoint_sha256
            ):
                raise ValueError(
                    "base BC checkpoint hash mismatch: "
                    f"{self.checkpoint_sha256} != {checkpoint_sha256}"
                )
            schema = observation_schema_for_dim(config.observation.dim)
            self.policy = BCPolicy(
                self.checkpoint,
                device=device,
                expected_observation_dim=schema.dim,
                expected_observation_schema=schema.version,
                expected_speed_feedback=config.observation.speed_feedback,
                expected_current_contract=config.observation.current_contract,
            )

    @property
    def metadata(self) -> BaseControllerMetadata:
        return BaseControllerMetadata(
            mode=self.mode,
            checkpoint=None if self.checkpoint is None else str(self.checkpoint),
            checkpoint_sha256=self.checkpoint_sha256,
            frozen=self.mode == "frozen_bc",
        )

    def env_kwargs(self) -> dict[str, Any]:
        if self.mode == "online_pd_pi":
            return {"base_mode": "teacher", "base_command_fn": None}
        if self.policy is None:
            raise RuntimeError("frozen BC policy was not loaded")
        return {"base_mode": "frozen", "base_command_fn": self.policy}


__all__ = [
    "BaseControllerAdapter",
    "BaseControllerMetadata",
    "BaseMode",
    "sha256_file",
]
