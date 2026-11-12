"""PC20 behavior cloning, checkpoints, and PPO actor warm-start helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoded_policy import _activation, _mlp
from .pc_normalization import PCNormalization
from .pc_path import (
    PC_PATH_DIMS,
    normalize_path_features,
    pc_path_feature_names_for_dim,
)
from .pc_policy import (
    PC_ACTOR_STATE_NAMES,
    PC_COMMAND_DIM,
    PC_COMMAND_NAMES,
    PC_COMMAND_NAMES_BY_DIM,
    PC_DADE_FEATURE_DIM,
    PC_SCHEMA_VERSION,
    PC_SIDE_CHANNEL_DIM,
    PCActorCritic,
    PCReservedSlotEncoder,
    pc_schema_hash,
)

PC_BC_CHECKPOINT_SCHEMA = "auv_pc_bc_checkpoint.v1"
PC_BC_DATASET_SCHEMA = "auv_pc_bc_dataset.v1"
PC_ACTION_ENVELOPE = "pc20"
PC_ACTION_HEAD_CONTINUOUS = "continuous"
PC_ACTION_HEAD_SIGN_MAGNITUDE = "sign_magnitude"
PC_SIGN_MAGNITUDE_OUTPUT_DIM = 9


def _normalize_action_head(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {PC_ACTION_HEAD_CONTINUOUS, PC_ACTION_HEAD_SIGN_MAGNITUDE}:
        raise ValueError(
            "action_head must be 'continuous' or 'sign_magnitude'"
        )
    return normalized


def pc_action_full_scale() -> tuple[float, float, float]:
    return (math.radians(20.0), math.radians(20.0), 1525.0)


def pc_action_from_actuator(actuator: Tensor) -> Tensor:
    """Map physical ``[rudder, stern, rpm]`` to normalized PC actions."""

    value = torch.as_tensor(actuator)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("actuator must have shape (N, 3)")
    full_scale = value.new_tensor(pc_action_full_scale())
    action = torch.empty_like(value)
    action[:, :2] = value[:, :2] / full_scale[:2]
    action[:, 2] = 2.0 * value[:, 2] / full_scale[2] - 1.0
    return action


def pc_actuator_from_action(action: Tensor) -> Tensor:
    """Map normalized PC actions to physical ``[rudder, stern, rpm]``."""

    value = torch.as_tensor(action)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("action must have shape (N, 3)")
    if not torch.isfinite(value).all():
        raise ValueError("action must be finite")
    full_scale = value.new_tensor(pc_action_full_scale())
    clipped = torch.clamp(value, -1.0, 1.0)
    actuator = torch.empty_like(clipped)
    actuator[:, :2] = clipped[:, :2] * full_scale[:2]
    actuator[:, 2] = 0.5 * full_scale[2] * (clipped[:, 2] + 1.0)
    return actuator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_tensor(command: Tensor, command_dim: int = PC_COMMAND_DIM) -> Tensor:
    value = torch.as_tensor(command)
    if command_dim == 0:
        if value.ndim != 2 or value.shape[1] != 0:
            raise ValueError("command must have shape (N, 0)")
        return value
    if command_dim != PC_COMMAND_DIM:
        raise ValueError("command_dim must be 0 or 3")
    if value.ndim != 2 or value.shape[1] != PC_COMMAND_DIM:
        raise ValueError("command must have shape (N, 3)")
    wrapped_heading = (value[:, 0] + math.pi) % (2.0 * math.pi) - math.pi
    return torch.stack(
        [
            wrapped_heading / math.pi,
            value[:, 1] / (math.pi / 2.0),
            value[:, 2] / 2.5,
        ],
        dim=1,
    )


@dataclass(frozen=True)
class PCDemonstrationDataset:
    actor_state: np.ndarray
    command: np.ndarray
    action: np.ndarray
    episode_id: np.ndarray
    seed: np.ndarray
    step: np.ndarray
    actual_actuator: np.ndarray
    reward: np.ndarray
    observation_schema: str = PC_SCHEMA_VERSION
    action_envelope: str = PC_ACTION_ENVELOPE
    current_estimate_body: np.ndarray | None = None
    actor_path: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.actor_state.ndim != 2 or self.actor_state.shape[1] != len(PC_ACTOR_STATE_NAMES):
            raise ValueError("actor_state must have shape (N, 12)")
        count = int(self.actor_state.shape[0])
        if count <= 0:
            raise ValueError("demonstration dataset must be non-empty")
        if self.command.shape[0] != count or self.command.shape[1] not in (0, PC_COMMAND_DIM):
            raise ValueError("command must have shape (N, 0) or (N, 3)")
        for name, value, width in (
            ("action", self.action, 3),
            ("actual_actuator", self.actual_actuator, 3),
        ):
            if value.shape != (count, width):
                raise ValueError(f"{name} must have shape (N, {width})")
        for name, value in (
            ("episode_id", self.episode_id),
            ("seed", self.seed),
            ("step", self.step),
            ("reward", self.reward),
        ):
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
        for name, value in (
            ("actor_state", self.actor_state),
            ("command", self.command),
            ("action", self.action),
            ("actual_actuator", self.actual_actuator),
            ("reward", self.reward),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf")
        if self.current_estimate_body is not None:
            if self.current_estimate_body.shape != (count, 3):
                raise ValueError("current_estimate_body must have shape (N, 3)")
            if not np.all(np.isfinite(self.current_estimate_body)):
                raise ValueError("current_estimate_body contains NaN or Inf")
        if self.actor_path is not None:
            if (
                self.actor_path.ndim != 2
                or self.actor_path.shape[0] != count
                or self.actor_path.shape[1] not in PC_PATH_DIMS
            ):
                raise ValueError(
                    f"actor_path must have shape (N, path_dim) with path_dim in {PC_PATH_DIMS}"
                )
            if not np.all(np.isfinite(self.actor_path)):
                raise ValueError("actor_path contains NaN or Inf")
        if self.observation_schema != PC_SCHEMA_VERSION:
            raise ValueError("PC demonstrations require observation_schema='pc_v1'")
        if self.action_envelope != PC_ACTION_ENVELOPE:
            raise ValueError("PC demonstrations require action_envelope='pc20'")

    @property
    def sample_count(self) -> int:
        return int(self.actor_state.shape[0])

    @property
    def episode_count(self) -> int:
        return int(np.unique(self.episode_id).size)

    @property
    def actor_command_dim(self) -> int:
        return int(self.command.shape[1])

    @property
    def path_dim(self) -> int:
        return 0 if self.actor_path is None else int(self.actor_path.shape[1])


@dataclass(frozen=True)
class PCBCTrainingConfig:
    state_encoder_hidden_dims: tuple[int, ...] = (128, 128)
    actor_hidden_dims: tuple[int, ...] = (256, 128)
    d_z_total: int = 32
    n_in_reserved: int = 8
    n_z_reserved: int = 4
    activation: str = "elu"
    batch_size: int = 4096
    max_epochs: int = 200
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    seed: int = 0
    device: str = "cuda"
    actor_command_dim: int = PC_COMMAND_DIM
    path_dim: int = 0
    path_encoder_hidden_dims: tuple[int, ...] = (64,)
    path_encoder_alpha: float = 1.0
    actor_current_encoder_enabled: bool = False
    actor_current_encoder_hidden_dims: tuple[int, ...] = (64,)
    actor_current_encoder_alpha: float = 1.0
    saturation_weight: float = 0.0
    saturation_threshold: float = 0.9
    action_head: str = PC_ACTION_HEAD_CONTINUOUS
    sign_deadband: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_head", _normalize_action_head(self.action_head))
        if not self.state_encoder_hidden_dims or any(
            value <= 0 for value in self.state_encoder_hidden_dims
        ):
            raise ValueError("state_encoder_hidden_dims must be positive")
        if not self.actor_hidden_dims or any(value <= 0 for value in self.actor_hidden_dims):
            raise ValueError("actor_hidden_dims must be positive")
        if self.d_z_total <= self.n_z_reserved:
            raise ValueError("d_z_total must exceed n_z_reserved")
        if self.n_in_reserved < 0 or self.n_z_reserved < 0:
            raise ValueError("reserved slot counts must be non-negative")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, max_epochs, and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.actor_command_dim not in (0, PC_COMMAND_DIM):
            raise ValueError("actor_command_dim must be 0 or 3")
        if self.path_dim not in (0, *PC_PATH_DIMS):
            raise ValueError(f"path_dim must be 0 or one of {PC_PATH_DIMS}")
        if self.path_dim and (
            not self.path_encoder_hidden_dims
            or any(int(value) <= 0 for value in self.path_encoder_hidden_dims)
        ):
            raise ValueError("path_encoder_hidden_dims must be positive")
        if self.path_dim and self.path_encoder_alpha <= 0.0:
            raise ValueError("path_encoder_alpha must be positive")
        if not self.actor_current_encoder_hidden_dims or any(
            int(value) <= 0 for value in self.actor_current_encoder_hidden_dims
        ):
            raise ValueError("actor_current_encoder_hidden_dims must be positive")
        if not math.isfinite(self.actor_current_encoder_alpha) or (
            self.actor_current_encoder_alpha < 0.0
        ):
            raise ValueError("actor_current_encoder_alpha must be finite and non-negative")
        if not math.isfinite(self.saturation_weight) or self.saturation_weight < 0.0:
            raise ValueError("saturation_weight must be finite and non-negative")
        if not math.isfinite(self.saturation_threshold) or not (
            0.0 < self.saturation_threshold <= 1.0
        ):
            raise ValueError("saturation_threshold must lie in (0, 1]")
        if not math.isfinite(self.sign_deadband) or not (0.0 < self.sign_deadband < 1.0):
            raise ValueError("sign_deadband must lie in (0, 1)")
        if self.action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE and self.saturation_weight != 0.0:
            raise ValueError(
                "sign_magnitude BC does not support saturation_weight; keep it at 0"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "state_encoder_hidden_dims": list(self.state_encoder_hidden_dims),
            "actor_hidden_dims": list(self.actor_hidden_dims),
            "path_encoder_hidden_dims": list(self.path_encoder_hidden_dims),
            "actor_current_encoder_hidden_dims": list(
                self.actor_current_encoder_hidden_dims
            ),
        }


class SignMagnitudeActionHead(nn.Module):
    """Decode two fin channels as sign class times sigmoid magnitude.

    Raw actor output layout:

    ``[rudder_neg, rudder_zero, rudder_pos,
       stern_neg, stern_zero, stern_pos,
       rudder_magnitude_logit, stern_magnitude_logit,
       rpm_continuous]``

    The head is stateless. The actor's final linear layer is widened from 3 to
    9 outputs only when this head is explicitly selected.
    """

    surface_channels = 2
    sign_classes = 3
    output_dim = PC_SIGN_MAGNITUDE_OUTPUT_DIM

    def __init__(self, deadband: float = 0.05) -> None:
        super().__init__()
        deadband = float(deadband)
        if not math.isfinite(deadband) or not (0.0 < deadband < 1.0):
            raise ValueError("sign_deadband must lie in (0, 1)")
        self.deadband = deadband

    def _split(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if raw.ndim != 2 or raw.shape[1] != self.output_dim:
            raise ValueError(
                f"sign/magnitude actor output must have shape (N, {self.output_dim})"
            )
        logits = raw[:, : self.surface_channels * self.sign_classes].reshape(
            raw.shape[0], self.surface_channels, self.sign_classes
        )
        magnitude_logits = raw[
            :, self.surface_channels * self.sign_classes : self.surface_channels * self.sign_classes + self.surface_channels
        ]
        rpm = raw[:, -1]
        return logits, magnitude_logits, rpm

    def decode(self, raw: Tensor) -> Tensor:
        logits, magnitude_logits, rpm = self._split(raw)
        sign = logits.argmax(dim=-1).to(dtype=raw.dtype) - 1.0
        magnitude = torch.sigmoid(magnitude_logits)
        return torch.cat(
            [sign * magnitude, torch.clamp(rpm, -1.0, 1.0).unsqueeze(1)],
            dim=1,
        )

    def targets(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if action.ndim != 2 or action.shape[1] != 3:
            raise ValueError("action must have shape (N, 3)")
        surface = action[:, :2]
        sign_target = torch.ones(
            surface.shape, dtype=torch.long, device=action.device
        )
        sign_target = torch.where(surface > self.deadband, 2, sign_target)
        sign_target = torch.where(surface < -self.deadband, 0, sign_target)
        magnitude_target = surface.abs()
        magnitude_mask = magnitude_target > self.deadband
        return sign_target, magnitude_target, magnitude_mask

    def loss(self, raw: Tensor, action: Tensor) -> dict[str, Tensor]:
        logits, magnitude_logits, rpm = self._split(raw)
        sign_target, magnitude_target, magnitude_mask = self.targets(action)
        sign_loss = F.cross_entropy(
            logits.reshape(-1, self.sign_classes),
            sign_target.reshape(-1),
        )
        predicted_magnitude = torch.sigmoid(magnitude_logits)
        magnitude_denominator = magnitude_mask.sum().clamp_min(1)
        magnitude_loss = (
            F.smooth_l1_loss(
                predicted_magnitude,
                magnitude_target,
                reduction="none",
            )
            * magnitude_mask.to(dtype=raw.dtype)
        ).sum() / magnitude_denominator.to(dtype=raw.dtype)
        rpm_loss = F.mse_loss(rpm, action[:, 2])
        total = sign_loss + magnitude_loss + rpm_loss
        return {
            "loss": total,
            "sign_loss": sign_loss,
            "magnitude_loss": magnitude_loss,
            "rpm_loss": rpm_loss,
        }

    def diagnostics(self, raw: Tensor, action: Tensor) -> dict[str, Any]:
        predicted = self.decode(raw)
        sign_target, magnitude_target, magnitude_mask = self.targets(action)
        predicted_sign = predicted[:, :2].sign()
        accuracy_mask = action[:, :2].abs() >= 0.5
        sign_accuracy: list[float | None] = []
        for channel in range(self.surface_channels):
            mask = accuracy_mask[:, channel]
            if not bool(mask.any()):
                sign_accuracy.append(None)
                continue
            sign_accuracy.append(
                float(
                    (
                        predicted_sign[:, channel][mask]
                        == sign_target[:, channel][mask].to(dtype=raw.dtype) - 1.0
                    )
                    .to(dtype=raw.dtype)
                    .mean()
                    .item()
                )
            )
        magnitude_mae: list[float | None] = []
        saturation_recall: list[float | None] = []
        teacher_saturation: list[float] = []
        predicted_saturation: list[float] = []
        for channel in range(self.surface_channels):
            mask = magnitude_mask[:, channel]
            if bool(mask.any()):
                magnitude_mae.append(
                    float(
                        (
                            predicted[:, channel][mask].abs()
                            - magnitude_target[:, channel][mask]
                        )
                        .abs()
                        .mean()
                        .item()
                    )
                )
            else:
                magnitude_mae.append(None)
            teacher_mask = action[:, channel].abs() >= 0.99
            teacher_saturation.append(float(teacher_mask.to(dtype=raw.dtype).mean().item()))
            predicted_saturation.append(
                float((predicted[:, channel].abs() >= 0.99).to(dtype=raw.dtype).mean().item())
            )
            if bool(teacher_mask.any()):
                saturation_recall.append(
                    float(
                        (predicted[:, channel][teacher_mask].abs() >= 0.99)
                        .to(dtype=raw.dtype)
                        .mean()
                        .item()
                    )
                )
            else:
                saturation_recall.append(None)
        zero_fraction = [
            float((predicted[:, channel].abs() < self.deadband).to(dtype=raw.dtype).mean().item())
            for channel in range(2)
        ]
        return {
            "sign_accuracy_high_action": sign_accuracy,
            "magnitude_mae_nonzero": magnitude_mae,
            "saturation_recall": saturation_recall,
            "teacher_saturation_rate": teacher_saturation,
            "predicted_saturation_rate": predicted_saturation,
            "saturation_ratio": [
                predicted_saturation[index] / teacher_saturation[index]
                if teacher_saturation[index] > 0.0
                else None
                for index in range(self.surface_channels)
            ],
            "predicted_zero_fraction": zero_fraction,
        }


class PCBCTrainableActor(nn.Module):
    """Actor path identical to ``PCActorCritic`` with zero DADE input."""

    def __init__(
        self,
        *,
        d_z_total: int,
        n_in_reserved: int,
        n_z_reserved: int,
        state_encoder_hidden_dims: Sequence[int],
        actor_hidden_dims: Sequence[int],
        activation: str,
        actor_command_dim: int = PC_COMMAND_DIM,
        path_dim: int = 0,
        path_encoder_hidden_dims: Sequence[int] = (64,),
        path_encoder_alpha: float = 1.0,
        actor_current_encoder_enabled: bool = False,
        actor_current_encoder_hidden_dims: Sequence[int] = (64,),
        actor_current_encoder_alpha: float = 1.0,
        action_head: str = PC_ACTION_HEAD_CONTINUOUS,
        sign_deadband: float = 0.05,
    ) -> None:
        super().__init__()
        if actor_command_dim not in (0, PC_COMMAND_DIM):
            raise ValueError("actor_command_dim must be 0 or 3")
        if path_dim not in (0, *PC_PATH_DIMS):
            raise ValueError(f"path_dim must be 0 or one of {PC_PATH_DIMS}")
        self.actor_command_dim = int(actor_command_dim)
        self.path_dim = int(path_dim)
        self.path_encoder_alpha = float(path_encoder_alpha)
        self.actor_current_encoder_enabled = bool(actor_current_encoder_enabled)
        self.actor_current_encoder_hidden_dims = tuple(
            int(value) for value in actor_current_encoder_hidden_dims
        )
        self.actor_current_encoder_alpha = float(actor_current_encoder_alpha)
        self.action_head = _normalize_action_head(action_head)
        self.sign_deadband = float(sign_deadband)
        if not math.isfinite(self.sign_deadband) or not (0.0 < self.sign_deadband < 1.0):
            raise ValueError("sign_deadband must lie in (0, 1)")
        if self.path_dim and self.path_encoder_alpha <= 0.0:
            raise ValueError("path_encoder_alpha must be positive")
        if self.actor_current_encoder_enabled and (
            not self.actor_current_encoder_hidden_dims
            or any(value <= 0 for value in self.actor_current_encoder_hidden_dims)
        ):
            raise ValueError("actor_current_encoder_hidden_dims must be positive")
        if not math.isfinite(self.actor_current_encoder_alpha) or (
            self.actor_current_encoder_alpha < 0.0
        ):
            raise ValueError("actor_current_encoder_alpha must be finite and non-negative")
        self.actor_encoder = PCReservedSlotEncoder(
            active_input_dim=len(PC_ACTOR_STATE_NAMES),
            d_z_total=d_z_total,
            n_in_reserved=n_in_reserved,
            n_z_reserved=n_z_reserved,
            hidden_dims=state_encoder_hidden_dims,
            activation=activation,
        )
        self.path_encoder: nn.Sequential | None = None
        if self.path_dim:
            self.path_encoder = _mlp(
                self.path_dim,
                tuple(int(value) for value in path_encoder_hidden_dims),
                int(d_z_total),
                _activation(activation),
            )
        self.actor_current_encoder: nn.Sequential | None = None
        if self.actor_current_encoder_enabled:
            self.actor_current_encoder = _mlp(
                PC_SIDE_CHANNEL_DIM,
                self.actor_current_encoder_hidden_dims,
                int(d_z_total),
                _activation(activation),
            )
            final = [
                layer
                for layer in self.actor_current_encoder.modules()
                if isinstance(layer, nn.Linear)
            ][-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.actor = _mlp(
            d_z_total + self.actor_command_dim + PC_DADE_FEATURE_DIM,
            actor_hidden_dims,
            3 if self.action_head == PC_ACTION_HEAD_CONTINUOUS else PC_SIGN_MAGNITUDE_OUTPUT_DIM,
            _activation(activation),
        )
        self.actor_output_dim = int(self.actor[-1].out_features)
        self.sign_magnitude_head = (
            SignMagnitudeActionHead(self.sign_deadband)
            if self.action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE
            else None
        )

    def set_actor_current_encoder_alpha(self, alpha: float) -> None:
        value = float(alpha)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("actor_current_encoder_alpha must be finite and non-negative")
        self.actor_current_encoder_alpha = value

    def forward(
        self,
        actor_state: Tensor,
        command: Tensor | None = None,
        actor_path: Tensor | None = None,
        actor_current: Tensor | None = None,
    ) -> Tensor:
        return self.decode_action(
            self.forward_raw(actor_state, command, actor_path, actor_current)
        )

    def forward_raw(
        self,
        actor_state: Tensor,
        command: Tensor | None = None,
        actor_path: Tensor | None = None,
        actor_current: Tensor | None = None,
    ) -> Tensor:
        z_t = self.actor_encoder(actor_state)
        if self.path_encoder is not None:
            if actor_path is None or actor_path.shape != (actor_state.shape[0], self.path_dim):
                raise ValueError(f"actor_path must have shape (N, {self.path_dim})")
            z_t = z_t + self.path_encoder_alpha * self.path_encoder(actor_path)
        elif actor_path is not None:
            raise ValueError("actor_path must be None when path_dim=0")
        if self.actor_current_encoder is not None:
            if actor_current is None or actor_current.shape != (
                actor_state.shape[0],
                PC_SIDE_CHANNEL_DIM,
            ):
                raise ValueError(
                    f"actor_current must have shape (N, {PC_SIDE_CHANNEL_DIM})"
                )
            z_t = z_t + self.actor_current_encoder_alpha * self.actor_current_encoder(
                actor_current
            )
        elif actor_current is not None:
            raise ValueError("actor_current must be None when the current encoder is disabled")
        if self.actor_command_dim == 0:
            if command is None:
                command = actor_state.new_zeros((actor_state.shape[0], 0))
            elif command.shape != (actor_state.shape[0], 0):
                raise ValueError("command must be None or have shape (N, 0)")
        else:
            if command is None or command.shape != (actor_state.shape[0], self.actor_command_dim):
                raise ValueError(f"command must have shape (N, {self.actor_command_dim})")
        c_hat = actor_state.new_zeros((actor_state.shape[0], PC_DADE_FEATURE_DIM))
        return self.actor(torch.cat([z_t, command, c_hat], dim=1))

    def decode_action(self, raw: Tensor) -> Tensor:
        if self.action_head == PC_ACTION_HEAD_CONTINUOUS:
            if raw.ndim != 2 or raw.shape[1] != 3:
                raise ValueError("continuous actor output must have shape (N, 3)")
            return raw
        if self.sign_magnitude_head is None:
            raise RuntimeError("sign/magnitude action head is not initialized")
        return self.sign_magnitude_head.decode(raw)


def load_pc_demonstration_dataset(
    path: Path | str,
) -> tuple[PCDemonstrationDataset, dict[str, Any]]:
    dataset_path = Path(path)
    with np.load(dataset_path) as payload:
        required = {
            "actor_state",
            "command",
            "action",
            "episode_id",
            "seed",
            "step",
            "actual_actuator",
            "reward",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"PC demonstration dataset is missing arrays: {missing}")
        dataset = PCDemonstrationDataset(
            actor_state=np.asarray(payload["actor_state"], dtype=np.float32),
            command=np.asarray(payload["command"], dtype=np.float32),
            action=np.asarray(payload["action"], dtype=np.float32),
            episode_id=np.asarray(payload["episode_id"], dtype=np.int64),
            seed=np.asarray(payload["seed"], dtype=np.int64),
            step=np.asarray(payload["step"], dtype=np.int64),
            actual_actuator=np.asarray(payload["actual_actuator"], dtype=np.float32),
            reward=np.asarray(payload["reward"], dtype=np.float32),
            current_estimate_body=(
                np.asarray(payload["current_estimate_body"], dtype=np.float32)
                if "current_estimate_body" in payload.files
                else None
            ),
            actor_path=(
                np.asarray(payload["actor_path"], dtype=np.float32)
                if "actor_path" in payload.files
                else None
            ),
            observation_schema=str(
                payload["observation_schema"]
                if "observation_schema" in payload.files
                else PC_SCHEMA_VERSION
            ),
            action_envelope=str(
                payload["action_envelope"]
                if "action_envelope" in payload.files
                else PC_ACTION_ENVELOPE
            ),
        )
    metadata_path = dataset_path.with_suffix(".manifest.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    )
    metadata.update(
        {
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": _sha256(dataset_path),
            "sample_count": dataset.sample_count,
            "episode_count": dataset.episode_count,
            "observation_schema": dataset.observation_schema,
            "action_envelope": dataset.action_envelope,
            "actor_command_dim": dataset.actor_command_dim,
            "path_dim": dataset.path_dim,
        }
    )
    return dataset, metadata


def _episode_split(
    episode_id: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_id)
    if episodes.size < 2:
        raise ValueError("at least two episodes are required for a train/validation split")
    shuffled = episodes.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    validation_count = int(round(shuffled.size * validation_fraction))
    validation_count = min(max(validation_count, 1), shuffled.size - 1)
    validation_episodes = set(int(value) for value in shuffled[:validation_count])
    validation_mask = np.isin(episode_id, list(validation_episodes))
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def _per_channel_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean((prediction - target).square(), dim=0)


def _pc_surface_sample_weights(
    action: Tensor,
    *,
    saturation_weight: float,
    saturation_threshold: float,
) -> Tensor:
    if action.ndim != 2 or action.shape[1] != 3:
        raise ValueError("action must have shape (N, 3)")
    if saturation_weight < 0.0:
        raise ValueError("saturation_weight must be non-negative")
    if not 0.0 < saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must lie in (0, 1]")
    saturated = (action[:, :2].abs() >= saturation_threshold).any(dim=1)
    return 1.0 + saturation_weight * saturated.to(dtype=action.dtype)


def _weighted_mse(prediction: Tensor, target: Tensor, sample_weight: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have the same rank-2 shape")
    if sample_weight.shape != (prediction.shape[0],):
        raise ValueError("sample_weight must have shape (N,)")
    per_sample = torch.mean((prediction - target).square(), dim=1)
    denominator = sample_weight.sum()
    if float(denominator) <= 0.0:
        raise ValueError("sample_weight must have a positive sum")
    return torch.sum(per_sample * sample_weight) / denominator


def _restore_init_checkpoint(
    model: PCBCTrainableActor,
    config: PCBCTrainingConfig,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Load model weights from an existing BC checkpoint without optimizer state."""

    metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
    expected_schema_hash = pc_schema_hash(
        path_dim=config.path_dim,
        actor_command_dim=config.actor_command_dim,
        critic_command_dim=0,
        actor_current_encoder_enabled=config.actor_current_encoder_enabled,
    )
    mismatches: list[str] = []

    def require_equal(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            mismatches.append(f"{name}: checkpoint={actual!r}, requested={expected!r}")

    require_equal("observation_schema", metadata["observation_schema"], PC_SCHEMA_VERSION)
    require_equal("action_envelope", metadata["action_envelope"], PC_ACTION_ENVELOPE)
    require_equal("pc_schema_hash", metadata["pc_schema_hash"], expected_schema_hash)
    require_equal("actor_command_dim", metadata["actor_command_dim"], config.actor_command_dim)
    require_equal("path_dim", metadata["path_dim"], config.path_dim)
    require_equal("d_z_total", int(metadata["d_z_total"]), config.d_z_total)
    require_equal("n_in_reserved", int(metadata["n_in_reserved"]), config.n_in_reserved)
    require_equal("n_z_reserved", int(metadata["n_z_reserved"]), config.n_z_reserved)
    require_equal(
        "state_encoder_hidden_dims",
        list(metadata["state_encoder_hidden_dims"]),
        list(config.state_encoder_hidden_dims),
    )
    require_equal(
        "actor_hidden_dims",
        list(metadata["actor_hidden_dims"]),
        list(config.actor_hidden_dims),
    )
    require_equal("activation", metadata["activation"], config.activation)
    require_equal("action_head", metadata["action_head"], config.action_head)
    require_equal("actor_output_dim", int(metadata["actor_output_dim"]), model.actor_output_dim)
    if config.path_dim:
        require_equal(
            "path_encoder_hidden_dims",
            list(metadata["path_encoder_hidden_dims"]),
            list(config.path_encoder_hidden_dims),
        )
        require_equal(
            "path_encoder_alpha",
            float(metadata["path_encoder_alpha"]),
            float(config.path_encoder_alpha),
        )
    expected_current_encoder = (
        "encoded" if config.actor_current_encoder_enabled else "none"
    )
    require_equal(
        "actor_current_encoder",
        metadata["actor_current_encoder"],
        expected_current_encoder,
    )
    if config.actor_current_encoder_enabled:
        require_equal(
            "actor_current_encoder_hidden_dims",
            list(metadata["actor_current_encoder_hidden_dims"]),
            list(config.actor_current_encoder_hidden_dims),
        )
        require_equal(
            "actor_current_encoder_alpha",
            float(metadata["actor_current_encoder_alpha"]),
            float(config.actor_current_encoder_alpha),
        )
    if config.action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE:
        require_equal(
            "sign_deadband",
            float(metadata["sign_deadband"]),
            float(config.sign_deadband),
        )
    if mismatches:
        raise ValueError(
            "init checkpoint architecture does not match training config: "
            + "; ".join(mismatches)
        )

    try:
        model.load_state_dict(metadata["state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"init checkpoint state_dict is incompatible: {checkpoint_path}"
        ) from error
    return {
        "init_checkpoint_path": str(Path(checkpoint_path).resolve()),
        "init_checkpoint_sha256": metadata["checkpoint_sha256"],
        "fine_tune_lr": float(config.learning_rate),
        "init_checkpoint_architecture": {
            "actor_command_dim": int(metadata["actor_command_dim"]),
            "path_dim": int(metadata["path_dim"]),
            "path_encoder_alpha": metadata["path_encoder_alpha"],
            "actor_current_encoder": metadata["actor_current_encoder"],
            "actor_current_encoder_alpha": metadata["actor_current_encoder_alpha"],
            "action_head": metadata["action_head"],
            "actor_output_dim": int(metadata["actor_output_dim"]),
        },
    }


def train_pc_bc(
    dataset: PCDemonstrationDataset,
    config: PCBCTrainingConfig,
    *,
    output_dir: Path | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    normalization: PCNormalization | None = None,
    init_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    normalization = PCNormalization() if normalization is None else normalization
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_indices, validation_indices = _episode_split(
        dataset.episode_id,
        config.validation_fraction,
        config.seed,
    )
    if dataset.actor_command_dim != config.actor_command_dim:
        raise ValueError(
            "demonstration command width does not match actor_command_dim: "
            f"dataset={dataset.actor_command_dim}, config={config.actor_command_dim}"
        )
    if dataset.path_dim != config.path_dim:
        raise ValueError(
            "demonstration path width does not match path_dim: "
            f"dataset={dataset.path_dim}, config={config.path_dim}"
        )
    if config.actor_current_encoder_enabled and dataset.current_estimate_body is None:
        raise ValueError(
            "demonstrations are missing current_estimate_body required by the current encoder"
        )
    state = torch.as_tensor(dataset.actor_state, dtype=torch.float32, device=device)
    command = torch.as_tensor(dataset.command, dtype=torch.float32, device=device)
    actor_path = (
        torch.as_tensor(dataset.actor_path, dtype=torch.float32, device=device)
        if dataset.actor_path is not None
        else None
    )
    actor_current = (
        torch.as_tensor(dataset.current_estimate_body, dtype=torch.float32, device=device)
        if config.actor_current_encoder_enabled
        else None
    )
    action = torch.as_tensor(dataset.action, dtype=torch.float32, device=device)
    state = normalization.normalize_actor_state(state)
    command = _command_tensor(command, config.actor_command_dim)
    if actor_path is not None:
        actor_path = normalize_path_features(actor_path)
    if actor_current is not None:
        actor_current = normalization.normalize_current(actor_current)
    train_index = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    validation_index = torch.as_tensor(validation_indices, dtype=torch.long, device=device)

    if not torch.all(torch.abs(action) <= 1.0 + 1e-6):
        raise ValueError("PC teacher actions must lie in [-1, 1]")
    action = torch.clamp(action, -1.0, 1.0)
    sample_weight = _pc_surface_sample_weights(
        action,
        saturation_weight=config.saturation_weight,
        saturation_threshold=config.saturation_threshold,
    )

    model = PCBCTrainableActor(
        d_z_total=config.d_z_total,
        n_in_reserved=config.n_in_reserved,
        n_z_reserved=config.n_z_reserved,
        state_encoder_hidden_dims=config.state_encoder_hidden_dims,
        actor_hidden_dims=config.actor_hidden_dims,
        activation=config.activation,
        actor_command_dim=config.actor_command_dim,
        path_dim=config.path_dim,
        path_encoder_hidden_dims=config.path_encoder_hidden_dims,
        path_encoder_alpha=config.path_encoder_alpha,
        actor_current_encoder_enabled=config.actor_current_encoder_enabled,
        actor_current_encoder_hidden_dims=config.actor_current_encoder_hidden_dims,
        actor_current_encoder_alpha=config.actor_current_encoder_alpha,
        action_head=config.action_head,
        sign_deadband=config.sign_deadband,
    ).to(device)
    init_metadata = (
        None
        if init_checkpoint is None
        else _restore_init_checkpoint(model, config, Path(init_checkpoint))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []
    validation_losses_unweighted: list[float] = []
    best_validation_loss = math.inf
    best_validation_loss_unweighted = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    best_mse_normalized = np.zeros(3, dtype=np.float64)
    best_action_diagnostics: dict[str, Any] | None = None
    stale_epochs = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    for epoch in range(config.max_epochs):
        model.train()
        shuffled = train_index[torch.randperm(train_index.numel(), generator=generator)]
        train_loss_numerator = 0.0
        train_weight_sum = 0.0
        for start in range(0, shuffled.numel(), config.batch_size):
            batch_index = shuffled[start : start + config.batch_size].to(device)
            raw_prediction = model.forward_raw(
                state[batch_index],
                command[batch_index],
                None if actor_path is None else actor_path[batch_index],
                None if actor_current is None else actor_current[batch_index],
            )
            prediction = model.decode_action(raw_prediction)
            target = action[batch_index]
            if config.action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE:
                assert model.sign_magnitude_head is not None
                loss = model.sign_magnitude_head.loss(raw_prediction, target)["loss"]
                numerator = float(loss.detach()) * int(batch_index.numel())
                denominator = float(batch_index.numel())
            elif config.saturation_weight == 0.0:
                loss = torch.mean((prediction - target).square())
                numerator = float(loss.detach()) * int(batch_index.numel())
                denominator = float(batch_index.numel())
            else:
                weights = sample_weight[batch_index]
                loss = _weighted_mse(prediction, target, weights)
                numerator = float(
                    torch.sum(
                        torch.mean((prediction - target).square(), dim=1)
                        * weights
                    ).detach()
                )
                denominator = float(weights.sum().detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_numerator += numerator
            train_weight_sum += denominator
        train_loss = train_loss_numerator / train_weight_sum

        model.eval()
        with torch.inference_mode():
            validation_raw_prediction = model.forward_raw(
                state[validation_index],
                command[validation_index],
                None if actor_path is None else actor_path[validation_index],
                None if actor_current is None else actor_current[validation_index],
            )
            validation_prediction = model.decode_action(validation_raw_prediction)
            validation_target = action[validation_index]
            validation_loss_unweighted = float(
                torch.mean((validation_prediction - validation_target).square()).item()
            )
            if config.action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE:
                assert model.sign_magnitude_head is not None
                validation_loss = float(
                    model.sign_magnitude_head.loss(
                        validation_raw_prediction,
                        validation_target,
                    )["loss"].item()
                )
                action_diagnostics = model.sign_magnitude_head.diagnostics(
                    validation_raw_prediction,
                    validation_target,
                )
            elif config.saturation_weight == 0.0:
                validation_loss = validation_loss_unweighted
                action_diagnostics = None
            else:
                validation_loss = float(
                    _weighted_mse(
                        validation_prediction,
                        validation_target,
                        sample_weight[validation_index],
                    ).item()
                )
                action_diagnostics = None
            validation_mse = (
                _per_channel_mse(validation_prediction, validation_target).cpu().numpy()
            )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_losses_unweighted.append(validation_loss_unweighted)
        if validation_loss < best_validation_loss - 1e-12:
            best_validation_loss = validation_loss
            best_validation_loss_unweighted = validation_loss_unweighted
            best_epoch = epoch
            best_mse_normalized = validation_mse
            best_action_diagnostics = action_diagnostics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("PC BC training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    full_scale = torch.tensor(pc_action_full_scale(), dtype=torch.float64)
    validation_rmse_normalized = np.sqrt(best_mse_normalized)
    validation_rmse_physical = validation_rmse_normalized * np.array(
        [full_scale[0].item(), full_scale[1].item(), 0.5 * full_scale[2].item()]
    )
    result = {
        "train_samples": int(train_index.numel()),
        "validation_samples": int(validation_index.numel()),
        "train_episodes": int(np.unique(dataset.episode_id[train_indices]).size),
        "validation_episodes": int(np.unique(dataset.episode_id[validation_indices]).size),
        "best_epoch": best_epoch,
        "epochs_completed": len(train_losses),
        "best_validation_loss": best_validation_loss,
        "best_validation_loss_unweighted": best_validation_loss_unweighted,
        "saturation_weight": float(config.saturation_weight),
        "saturation_threshold": float(config.saturation_threshold),
        "saturation_sample_fraction": float(sample_weight.gt(1.0).float().mean().item()),
        "action_head": config.action_head,
        "sign_deadband": float(config.sign_deadband),
        "actor_output_dim": model.actor_output_dim,
        "actor_current_encoder": (
            "encoded" if config.actor_current_encoder_enabled else "none"
        ),
        "actor_current_encoder_hidden_dims": list(
            config.actor_current_encoder_hidden_dims
        ),
        "actor_current_encoder_alpha": float(config.actor_current_encoder_alpha),
        "action_diagnostics": best_action_diagnostics,
        "validation_rmse_normalized": validation_rmse_normalized.tolist(),
        "validation_rmse_physical": validation_rmse_physical.tolist(),
        "train_loss": train_losses,
        "validation_loss": validation_losses,
        "validation_loss_unweighted": validation_losses_unweighted,
    }
    if init_metadata is not None:
        result.update(init_metadata)

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output / "pc_bc_actor.pt"
        checkpoint = {
            "schema_version": PC_BC_CHECKPOINT_SCHEMA,
            "policy_kind": "pc_bc_actor",
            "observation_schema": PC_SCHEMA_VERSION,
            "action_envelope": PC_ACTION_ENVELOPE,
            "actor_state_names": list(PC_ACTOR_STATE_NAMES),
            "command_names": list(PC_COMMAND_NAMES_BY_DIM[config.actor_command_dim]),
            "actor_command_dim": config.actor_command_dim,
            "path_dim": config.path_dim,
            "path_feature_names": (
                list(pc_path_feature_names_for_dim(config.path_dim))
                if config.path_dim
                else []
            ),
            "path_encoder_hidden_dims": (
                list(config.path_encoder_hidden_dims) if config.path_dim else []
            ),
            "path_encoder_alpha": (
                config.path_encoder_alpha if config.path_dim else None
            ),
            "actor_current_encoder": (
                "encoded" if config.actor_current_encoder_enabled else "none"
            ),
            "actor_current_encoder_hidden_dims": list(
                config.actor_current_encoder_hidden_dims
            ),
            "actor_current_encoder_alpha": float(
                config.actor_current_encoder_alpha
            ),
            "actor_current_dim": 3 if config.actor_current_encoder_enabled else 0,
            "z_t_dim": config.d_z_total,
            "actor_top_level_input_dim": (
                config.d_z_total + config.actor_command_dim + PC_DADE_FEATURE_DIM
            ),
            "pc_schema_hash": pc_schema_hash(
                path_dim=config.path_dim,
                actor_command_dim=config.actor_command_dim,
                critic_command_dim=0,
                actor_current_encoder_enabled=config.actor_current_encoder_enabled,
            ),
            "d_z_total": config.d_z_total,
            "n_in_reserved": config.n_in_reserved,
            "n_z_reserved": config.n_z_reserved,
            "state_encoder_hidden_dims": list(config.state_encoder_hidden_dims),
            "actor_hidden_dims": list(config.actor_hidden_dims),
            "activation": config.activation,
            "action_full_scale": list(pc_action_full_scale()),
            "action_head": config.action_head,
            "sign_deadband": (
                float(config.sign_deadband)
                if config.action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE
                else None
            ),
            "actor_output_dim": model.actor_output_dim,
            "action_decode_rule": (
                "surface_sign_argmax_times_sigmoid_magnitude_rpm_clamp"
                if config.action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE
                else "direct_continuous_3d"
            ),
            "normalization": normalization.as_dict(),
            "state_dict": model.state_dict(),
            "training_config": config.as_dict(),
            "training_result": result,
            "dataset": dict(metadata or {}),
            **({} if init_metadata is None else init_metadata),
        }
        torch.save(checkpoint, checkpoint_path)
        result["checkpoint_path"] = str(checkpoint_path.resolve())
        result["checkpoint_sha256"] = _sha256(checkpoint_path)
        (output / "training_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "auv_pc_bc_training_manifest.v1",
                    **checkpoint,
                    "state_dict": "<stored in pc_bc_actor.pt>",
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": result["checkpoint_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return result


def _actor_final_linear_output_dim(state_dict: Mapping[str, Tensor]) -> int:
    layers: list[tuple[int, int]] = []
    for name, value in state_dict.items():
        if not name.startswith("actor.") or not name.endswith(".weight"):
            continue
        fields = name.split(".")
        if len(fields) != 3:
            continue
        try:
            index = int(fields[1])
        except ValueError:
            continue
        if value.ndim != 2:
            continue
        layers.append((index, int(value.shape[0])))
    if not layers:
        raise ValueError("PC BC checkpoint has no actor linear layers")
    return max(layers, key=lambda item: item[0])[1]


def read_pc_bc_checkpoint_metadata(path: Path | str) -> dict[str, Any]:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != PC_BC_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a PC BC actor checkpoint")
    required = (
        "observation_schema",
        "action_envelope",
        "d_z_total",
        "n_in_reserved",
        "n_z_reserved",
        "state_encoder_hidden_dims",
        "actor_hidden_dims",
        "action_full_scale",
        "normalization",
        "state_dict",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"PC BC checkpoint is missing fields: {missing}")
    if checkpoint["observation_schema"] != PC_SCHEMA_VERSION:
        raise ValueError("PC BC checkpoint observation schema mismatch")
    if checkpoint["action_envelope"] != PC_ACTION_ENVELOPE:
        raise ValueError("PC BC checkpoint action envelope mismatch")
    if not np.allclose(
        np.asarray(checkpoint["action_full_scale"], dtype=np.float64),
        np.asarray(pc_action_full_scale(), dtype=np.float64),
    ):
        raise ValueError("PC BC checkpoint action scale mismatch")
    command_names = tuple(
        str(name) for name in checkpoint.get("command_names", PC_COMMAND_NAMES)
    )
    actor_command_dim = int(checkpoint.get("actor_command_dim", len(command_names)))
    if actor_command_dim not in PC_COMMAND_NAMES_BY_DIM:
        raise ValueError("PC BC checkpoint actor_command_dim must be 0, 1, or 3")
    if len(command_names) != actor_command_dim:
        raise ValueError("PC BC checkpoint command_names do not match actor_command_dim")
    if tuple(command_names) != tuple(PC_COMMAND_NAMES_BY_DIM[actor_command_dim]):
        raise ValueError("PC BC checkpoint command_names are not in canonical order")
    path_dim = int(checkpoint.get("path_dim", 0))
    if path_dim not in (0, *PC_PATH_DIMS):
        raise ValueError(
            f"PC BC checkpoint path_dim must be 0 or one of {PC_PATH_DIMS}"
        )
    expected_path_feature_names = (
        pc_path_feature_names_for_dim(path_dim) if path_dim else ()
    )
    path_feature_names = tuple(
        str(name)
        for name in checkpoint.get("path_feature_names", expected_path_feature_names)
    )
    if path_dim and tuple(path_feature_names) != tuple(expected_path_feature_names):
        raise ValueError("PC BC checkpoint path feature names are not canonical")
    path_encoder_hidden_dims = tuple(
        int(value) for value in checkpoint.get("path_encoder_hidden_dims", (64,) if path_dim else ())
    )
    if path_dim and not path_encoder_hidden_dims:
        raise ValueError("PC BC checkpoint is missing path encoder dimensions")
    path_encoder_alpha = checkpoint.get("path_encoder_alpha")
    if path_dim:
        path_encoder_alpha = float(1.0 if path_encoder_alpha is None else path_encoder_alpha)
        if path_encoder_alpha <= 0.0:
            raise ValueError("PC BC checkpoint path_encoder_alpha must be positive")
        if not any(name.startswith("path_encoder.") for name in checkpoint["state_dict"]):
            raise ValueError("PC BC checkpoint path_dim is set but path encoder weights are missing")
    else:
        path_encoder_alpha = None
    actor_current_encoder = str(checkpoint.get("actor_current_encoder", "none"))
    if actor_current_encoder not in {"none", "encoded"}:
        raise ValueError("PC BC checkpoint actor_current_encoder must be 'none' or 'encoded'")
    actor_current_encoder_hidden_dims = tuple(
        int(value)
        for value in checkpoint.get(
            "actor_current_encoder_hidden_dims", (64,) if actor_current_encoder == "encoded" else ()
        )
    )
    actor_current_encoder_alpha = checkpoint.get("actor_current_encoder_alpha")
    if actor_current_encoder == "encoded":
        if not actor_current_encoder_hidden_dims or any(
            value <= 0 for value in actor_current_encoder_hidden_dims
        ):
            raise ValueError("PC BC checkpoint is missing current encoder dimensions")
        actor_current_encoder_alpha = float(
            1.0 if actor_current_encoder_alpha is None else actor_current_encoder_alpha
        )
        if not math.isfinite(actor_current_encoder_alpha) or actor_current_encoder_alpha < 0.0:
            raise ValueError("PC BC checkpoint current encoder alpha must be positive")
        if not any(
            name.startswith("actor_current_encoder.") for name in checkpoint["state_dict"]
        ):
            raise ValueError(
                "PC BC checkpoint current encoder is enabled but weights are missing"
            )
    else:
        if any(
            name.startswith("actor_current_encoder.") for name in checkpoint["state_dict"]
        ):
            raise ValueError(
                "PC BC checkpoint current encoder is disabled but weights are present"
            )
        actor_current_encoder_hidden_dims = ()
        actor_current_encoder_alpha = None
    action_head = _normalize_action_head(
        str(checkpoint.get("action_head", PC_ACTION_HEAD_CONTINUOUS))
    )
    sign_deadband_value = checkpoint.get("sign_deadband")
    if action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE:
        if sign_deadband_value is None:
            raise ValueError("sign_magnitude checkpoint is missing sign_deadband")
        expected_output_dim = PC_SIGN_MAGNITUDE_OUTPUT_DIM
    else:
        expected_output_dim = 3
    sign_deadband = float(0.05 if sign_deadband_value is None else sign_deadband_value)
    if not math.isfinite(sign_deadband) or not (0.0 < sign_deadband < 1.0):
        raise ValueError("PC BC checkpoint sign_deadband must lie in (0, 1)")
    declared_output_dim = checkpoint.get("actor_output_dim")
    if declared_output_dim is None:
        if action_head == PC_ACTION_HEAD_SIGN_MAGNITUDE:
            raise ValueError("sign_magnitude checkpoint is missing actor_output_dim")
        declared_output_dim = expected_output_dim
    declared_output_dim = int(declared_output_dim)
    if declared_output_dim != expected_output_dim:
        raise ValueError(
            "PC BC checkpoint actor_output_dim does not match action_head: "
            f"declared={declared_output_dim}, expected={expected_output_dim}"
        )
    state_output_dim = _actor_final_linear_output_dim(checkpoint["state_dict"])
    if state_output_dim != expected_output_dim:
        raise ValueError(
            "PC BC checkpoint actor weights do not match action_head: "
            f"weights={state_output_dim}, expected={expected_output_dim}"
        )
    z_t_dim = int(checkpoint["d_z_total"])
    expected_top_level_input_dim = z_t_dim + actor_command_dim + PC_DADE_FEATURE_DIM
    declared_z_t_dim = checkpoint.get("z_t_dim")
    if declared_z_t_dim is not None and int(declared_z_t_dim) != z_t_dim:
        raise ValueError(
            "PC BC checkpoint z_t_dim is inconsistent with d_z_total: "
            f"declared={declared_z_t_dim}, expected={z_t_dim}"
        )
    declared_top_level_input_dim = checkpoint.get("actor_top_level_input_dim")
    if (
        declared_top_level_input_dim is not None
        and int(declared_top_level_input_dim) != expected_top_level_input_dim
    ):
        raise ValueError(
            "PC BC checkpoint actor_top_level_input_dim is inconsistent with "
            "its z/command/DADE layout: "
            f"declared={declared_top_level_input_dim}, "
            f"expected={expected_top_level_input_dim}"
        )
    return {
        **checkpoint,
        "command_names": list(command_names),
        "actor_command_dim": actor_command_dim,
        "path_dim": path_dim,
        "path_feature_names": list(path_feature_names) if path_dim else [],
        "path_encoder_hidden_dims": list(path_encoder_hidden_dims) if path_dim else [],
        "path_encoder_alpha": path_encoder_alpha,
        "actor_current_encoder": actor_current_encoder,
        "actor_current_encoder_hidden_dims": (
            list(actor_current_encoder_hidden_dims)
            if actor_current_encoder == "encoded"
            else []
        ),
        "actor_current_encoder_alpha": actor_current_encoder_alpha,
        "z_t_dim": z_t_dim,
        "actor_top_level_input_dim": expected_top_level_input_dim,
        "action_head": action_head,
        "sign_deadband": sign_deadband,
        "actor_output_dim": declared_output_dim,
        "action_decode_rule": str(
            checkpoint.get(
                "action_decode_rule",
                "direct_continuous_3d"
                if action_head == PC_ACTION_HEAD_CONTINUOUS
                else "surface_sign_argmax_times_sigmoid_magnitude_rpm_clamp",
            )
        ),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }


def restore_pc_bc_frozen_path_encoder_alpha(
    policy: PCActorCritic,
    checkpoint_path: Path | str,
) -> float:
    """Restore the frozen BC path-encoder scalar missing from model state.

    The path-encoder alpha is a runtime scalar rather than an ``nn.Module``
    parameter, so it is not present in ``model_state_dict``. Evaluation must
    restore it from the exact BC checkpoint referenced by the training config.
    """

    if policy.frozen_bc_path_encoder is None:
        raise ValueError("policy has no frozen BC path encoder")
    metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
    source_path_dim = int(metadata["path_dim"])
    if source_path_dim != policy.actor_layout.path_dim:
        raise ValueError(
            "PC BC path dimension does not match the PPO policy: "
            f"BC={source_path_dim}, policy={policy.actor_layout.path_dim}"
        )
    if tuple(metadata["path_encoder_hidden_dims"]) != tuple(
        policy.frozen_bc_path_encoder_hidden_dims
    ):
        raise ValueError(
            "PC BC path encoder dimensions do not match the PPO policy: "
            f"BC={metadata['path_encoder_hidden_dims']}, "
            f"policy={policy.frozen_bc_path_encoder_hidden_dims}"
        )
    alpha = metadata["path_encoder_alpha"]
    if alpha is None:
        raise ValueError("PC BC checkpoint has no frozen path encoder alpha")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("PC BC checkpoint path encoder alpha must be finite and positive")
    policy.frozen_bc_path_encoder_alpha = alpha
    return alpha


def restore_pc_bc_frozen_current_encoder_alpha(
    policy: PCActorCritic,
    checkpoint_path: Path | str,
) -> float:
    """Restore the frozen BC current-encoder scalar missing from model state."""

    if policy.frozen_bc_current_encoder is None:
        raise ValueError("policy has no frozen BC current encoder")
    metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
    if metadata["actor_current_encoder"] != "encoded":
        raise ValueError("PC BC checkpoint has no frozen current encoder")
    if tuple(metadata["actor_current_encoder_hidden_dims"]) != tuple(
        policy.frozen_bc_current_encoder_hidden_dims
    ):
        raise ValueError(
            "PC BC current encoder dimensions do not match the PPO policy: "
            f"BC={metadata['actor_current_encoder_hidden_dims']}, "
            f"policy={policy.frozen_bc_current_encoder_hidden_dims}"
        )
    alpha = metadata["actor_current_encoder_alpha"]
    if alpha is None:
        raise ValueError("PC BC checkpoint has no frozen current encoder alpha")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(
            "PC BC checkpoint current encoder alpha must be finite and positive"
        )
    policy.frozen_bc_current_encoder_alpha = alpha
    return alpha


class PCBCPolicy:
    """Deterministic PC BC actor used for BC-only closed-loop evaluation."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
        self.device = torch.device(device)
        self.metadata = metadata
        self.model = PCBCTrainableActor(
            d_z_total=int(metadata["d_z_total"]),
            n_in_reserved=int(metadata["n_in_reserved"]),
            n_z_reserved=int(metadata["n_z_reserved"]),
            state_encoder_hidden_dims=tuple(metadata["state_encoder_hidden_dims"]),
            actor_hidden_dims=tuple(metadata["actor_hidden_dims"]),
            activation=str(metadata.get("activation", "elu")),
            actor_command_dim=int(metadata["actor_command_dim"]),
            path_dim=int(metadata["path_dim"]),
            path_encoder_hidden_dims=tuple(metadata["path_encoder_hidden_dims"] or (64,)),
            path_encoder_alpha=(
                float(metadata["path_encoder_alpha"])
                if metadata["path_dim"]
                else 1.0
            ),
            actor_current_encoder_enabled=(
                metadata.get("actor_current_encoder", "none") == "encoded"
            ),
            actor_current_encoder_hidden_dims=tuple(
                metadata.get("actor_current_encoder_hidden_dims", ())
            ),
            actor_current_encoder_alpha=float(
                (
                    1.0
                    if metadata.get("actor_current_encoder_alpha") is None
                    else float(metadata["actor_current_encoder_alpha"])
                )
            ),
            action_head=str(metadata["action_head"]),
            sign_deadband=float(metadata["sign_deadband"]),
        ).to(self.device)
        self.model.load_state_dict(metadata["state_dict"])
        self.model.eval()
        normalization = metadata["normalization"]
        self.normalization = PCNormalization(
            position_scale_m=float(normalization["position_scale_m"]),
            attitude_scale_rad=float(normalization["attitude_scale_rad"]),
            angular_rate_scale_rad_s=float(normalization["angular_rate_scale_rad_s"]),
            surge_speed_scale_mps=float(normalization["surge_speed_scale_mps"]),
            sway_heave_speed_scale_mps=float(normalization["sway_heave_speed_scale_mps"]),
            current_scale_mps=float(normalization["current_scale_mps"]),
        )

    def act_inference(
        self,
        actor_state: Tensor,
        command: Tensor | None = None,
        actor_path: Tensor | None = None,
        actor_current: Tensor | None = None,
    ) -> Tensor:
        state = torch.as_tensor(actor_state, dtype=torch.float32, device=self.device)
        if self.model.actor_command_dim == 0:
            if command is None:
                command_value = state.new_zeros((state.shape[0], 0))
            else:
                command_value = torch.as_tensor(
                    command, dtype=torch.float32, device=self.device
                )
                if command_value.shape != (state.shape[0], 0):
                    raise ValueError("command must be None or have shape (N, 0)")
        else:
            if command is None:
                raise ValueError("command is required for a 3-D PC BC actor")
            command_value = _command_tensor(
                torch.as_tensor(command, dtype=torch.float32, device=self.device),
                self.model.actor_command_dim,
            )
        if self.model.path_dim:
            if actor_path is None:
                raise ValueError("actor_path is required for a path-conditioned PC BC actor")
            path_value = normalize_path_features(
                torch.as_tensor(actor_path, dtype=torch.float32, device=self.device)
            )
        else:
            if actor_path is not None:
                raise ValueError("actor_path must be None when path_dim=0")
            path_value = None
        if self.model.actor_current_encoder is not None:
            if actor_current is None:
                raise ValueError(
                    "actor_current is required for a current-conditioned PC BC actor"
                )
            current_value = self.normalization.normalize_current(
                torch.as_tensor(actor_current, dtype=torch.float32, device=self.device)
            )
        else:
            if actor_current is not None:
                raise ValueError(
                    "actor_current must be None when the BC actor has no current encoder"
                )
            current_value = None
        with torch.inference_mode():
            return self.model(
                self.normalization.normalize_actor_state(state),
                command_value,
                path_value,
                current_value,
            )


def _zero_last_linear(module: nn.Module | None) -> bool:
    if module is None:
        return False
    linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linears:
        return False
    last = linears[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
    return True


def _initialize_last_linear(module: nn.Module | None, *, std: float) -> bool:
    if module is None:
        return False
    std = float(std)
    if not math.isfinite(std) or std < 0.0:
        raise ValueError("residual output initialization std must be finite and non-negative")
    linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linears:
        return False
    last = linears[-1]
    if std == 0.0:
        nn.init.zeros_(last.weight)
    else:
        nn.init.normal_(last.weight, mean=0.0, std=std)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
    return True


PC_DADE_WARMSTART_INPUT_SCALE = 0.01


def _scale_dade_input_columns(
    module: nn.Module | None,
    scale: float = PC_DADE_WARMSTART_INPUT_SCALE,
) -> bool:
    """Scale the actor-input block that was always zero during BC training.

    A hard zero preserves BC behavior exactly, but by itself leaves both the
    actor columns and DADE output without a policy-gradient path. Zero is
    therefore only useful when the auxiliary current-prediction loss can break
    the symmetry first. A small non-zero scale keeps the initial BC behavior
    close while providing a direct, albeit weak, gradient path.
    """

    if module is None:
        return False
    if scale < 0.0:
        raise ValueError("DADE warm-start input scale must be non-negative")
    linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linears or linears[0].in_features < PC_DADE_FEATURE_DIM:
        return False
    first = linears[0]
    with torch.no_grad():
        first.weight[:, -PC_DADE_FEATURE_DIM:].mul_(scale)
    return True


def load_pc_bc_actor_prior(
    policy: PCActorCritic,
    checkpoint_path: Path | str,
    *,
    expected_d_z_total: int,
    expected_n_in_reserved: int,
    expected_n_z_reserved: int,
    expected_actor_hidden_dims: Sequence[int],
    expected_state_encoder_hidden_dims: Sequence[int],
    expected_normalization: PCNormalization,
    dade_input_scale: float = PC_DADE_WARMSTART_INPUT_SCALE,
    load_bc_path_encoder: bool = False,
) -> dict[str, Any]:
    """Load BC actor/encoder and keep the initial DADE feature exactly zero."""

    if getattr(policy, "frozen_bc_residual_enabled", False):
        raise ValueError(
            "frozen BC residual mode must use load_pc_bc_frozen_actor_prior, "
            "not actor warm-start"
        )
    metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
    if metadata["action_head"] != PC_ACTION_HEAD_CONTINUOUS:
        raise ValueError(
            "PPO actor warm-start currently requires a continuous-action BC checkpoint"
        )
    if int(metadata["d_z_total"]) != expected_d_z_total:
        raise ValueError("PC BC d_z_total does not match the PPO policy")
    if int(metadata["n_in_reserved"]) != expected_n_in_reserved:
        raise ValueError("PC BC n_in_reserved does not match the PPO policy")
    if int(metadata["n_z_reserved"]) != expected_n_z_reserved:
        raise ValueError("PC BC n_z_reserved does not match the PPO policy")
    if tuple(metadata["actor_hidden_dims"]) != tuple(expected_actor_hidden_dims):
        raise ValueError("PC BC actor hidden dimensions do not match the PPO policy")
    if tuple(metadata["state_encoder_hidden_dims"]) != tuple(expected_state_encoder_hidden_dims):
        raise ValueError("PC BC encoder hidden dimensions do not match the PPO policy")
    if metadata["normalization"] != expected_normalization.as_dict():
        raise ValueError("PC BC normalization does not match the PPO environment")

    state_dict = metadata["state_dict"]
    encoder_state = {
        name.removeprefix("actor_encoder."): value
        for name, value in state_dict.items()
        if name.startswith("actor_encoder.")
    }
    actor_state = {
        name.removeprefix("actor."): value
        for name, value in state_dict.items()
        if name.startswith("actor.")
    }
    policy.actor_encoder.load_state_dict(encoder_state, strict=True)
    source_commands = tuple(str(name) for name in metadata.get("command_names", PC_COMMAND_NAMES))
    target_commands = tuple(PC_COMMAND_NAMES_BY_DIM[policy.actor_layout.command_dim])
    zero_initialized_commands = sorted(set(target_commands) - set(source_commands))
    source_first = actor_state.get("0.weight")
    target_state = policy.actor.state_dict()
    target_first = target_state.get("0.weight")
    if not isinstance(source_first, Tensor) or not isinstance(target_first, Tensor):
        raise ValueError("BC actor must expose its first linear layer as actor.0.weight")
    source_dade_start = int(expected_d_z_total) + len(source_commands)
    target_dade_start = int(expected_d_z_total) + len(target_commands)
    if source_first.shape != (
        target_first.shape[0],
        source_dade_start + PC_DADE_FEATURE_DIM,
    ):
        raise ValueError(
            "BC actor input width is inconsistent with its z/command/DADE layout: "
            f"expected {(target_first.shape[0], source_dade_start + PC_DADE_FEATURE_DIM)}, "
            f"got {tuple(source_first.shape)}"
        )
    if target_first.shape[1] != target_dade_start + PC_DADE_FEATURE_DIM:
        raise ValueError("target actor input width is inconsistent with its PC layout")
    merged = target_first.clone()
    merged[:, : int(expected_d_z_total)] = source_first[:, : int(expected_d_z_total)]
    for target_index, command_name in enumerate(target_commands):
        if command_name in source_commands:
            source_index = source_commands.index(command_name)
            merged[:, int(expected_d_z_total) + target_index] = source_first[
                :, int(expected_d_z_total) + source_index
            ]
    merged[:, target_dade_start:] = source_first[:, source_dade_start:]
    loaded = {name: value.clone() for name, value in target_state.items()}
    loaded["0.weight"] = merged
    for name, target_value in target_state.items():
        if name == "0.weight":
            continue
        source_value = actor_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            raise ValueError(f"BC actor parameter {name!r} is missing or has the wrong shape")
        loaded[name] = source_value
    policy.actor.load_state_dict(loaded, strict=True)
    path_encoder_loaded = False
    path_encoder_zero_output = True
    if policy.path_encoder is not None:
        if load_bc_path_encoder:
            checkpoint_path_dim = int(metadata["path_dim"])
            target_path_dim = int(policy.bc_path_dim)
            if checkpoint_path_dim != target_path_dim:
                raise ValueError(
                    "BC path width does not match the PPO encoder prefix: "
                    f"BC={checkpoint_path_dim}, policy={target_path_dim}"
                )
            expected_names = pc_path_feature_names_for_dim(policy.actor_layout.path_dim)[
                :target_path_dim
            ]
            if tuple(metadata.get("path_feature_names", ())) != tuple(expected_names):
                raise ValueError("BC path feature prefix does not match the PPO path layout")
            if tuple(metadata["path_encoder_hidden_dims"]) != tuple(
                policy.path_encoder_hidden_dims
            ):
                raise ValueError(
                    "BC path encoder hidden dimensions do not match the PPO policy"
                )
            if int(policy.path_encoder_output_dim) != int(expected_d_z_total):
                raise ValueError(
                    "BC path encoder loading requires a full-width d_z path output"
                )
            path_state = {
                name.removeprefix("path_encoder."): value
                for name, value in state_dict.items()
                if name.startswith("path_encoder.")
            }
            if not path_state:
                raise ValueError("BC checkpoint does not contain path encoder weights")
            policy.path_encoder.load_state_dict(path_state, strict=True)
            path_encoder_loaded = True
            path_encoder_zero_output = False
        else:
            final = [
                layer
                for layer in policy.path_encoder.modules()
                if isinstance(layer, nn.Linear)
            ][-1]
            path_encoder_zero_output = bool(
                torch.count_nonzero(final.weight).item() == 0
                and (final.bias is None or torch.count_nonzero(final.bias).item() == 0)
            )
            if not path_encoder_zero_output:
                raise RuntimeError(
                    "path encoder must preserve zero output when loading a BC actor prior"
                )
    endpoint_encoder_zero_initialized = True
    if getattr(policy, "path_endpoint_encoder", None) is not None:
        final = [
            layer
            for layer in policy.path_endpoint_encoder.modules()
            if isinstance(layer, nn.Linear)
        ][-1]
        endpoint_encoder_zero_initialized = bool(
            torch.count_nonzero(final.weight).item() == 0
            and (final.bias is None or torch.count_nonzero(final.bias).item() == 0)
        )
        if not endpoint_encoder_zero_initialized:
            raise RuntimeError(
                "endpoint encoder must preserve zero output when loading a BC actor prior"
            )
    bc_current_encoder = metadata.get("actor_current_encoder", "none") == "encoded"
    if bc_current_encoder:
        if policy.actor_current_encoder is None:
            raise ValueError(
                "BC checkpoint has an actor current encoder but the PPO policy does not"
            )
        if tuple(metadata["actor_current_encoder_hidden_dims"]) != tuple(
            policy.actor_current_encoder_hidden_dims
        ):
            raise ValueError(
                "PC BC current encoder dimensions do not match the PPO policy"
            )
        current_encoder_state = {
            name.removeprefix("actor_current_encoder."): value
            for name, value in state_dict.items()
            if name.startswith("actor_current_encoder.")
        }
        if not current_encoder_state:
            raise ValueError("BC checkpoint does not contain current encoder weights")
        policy.actor_current_encoder.load_state_dict(current_encoder_state, strict=True)
        policy.actor_current_encoder_alpha = float(
            metadata["actor_current_encoder_alpha"]
        )
        current_encoder_initialization = "loaded_from_bc"
    else:
        current_encoder_initialization = "zero_final_linear"
        if policy.actor_current_encoder is not None:
            final = [
                layer
                for layer in policy.actor_current_encoder.modules()
                if isinstance(layer, nn.Linear)
            ][-1]
            if torch.count_nonzero(final.weight).item() != 0 or (
                final.bias is not None and torch.count_nonzero(final.bias).item() != 0
            ):
                raise RuntimeError(
                    "new current encoder must preserve zero output when BC has no "
                    "current encoder"
                )
    zeroed = _zero_last_linear(policy.actor_dade)
    if policy.actor_dade is not None and not zeroed:
        raise RuntimeError("actor DADE has no linear output layer to zero")
    if (
        policy.dade_enabled
        and dade_input_scale == 0.0
        and policy.actor_current_head is None
    ):
        raise ValueError(
            "zero DADE warm-start input scale requires the auxiliary current "
            "prediction objective to break actor/DADE symmetry"
        )
    input_columns_scaled = (
        _scale_dade_input_columns(policy.actor, dade_input_scale) if policy.dade_enabled else False
    )
    if policy.dade_enabled and not input_columns_scaled:
        raise RuntimeError("actor has no DADE input columns to scale")
    return {
        "checkpoint_path": metadata["checkpoint_path"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "initialization": "pc20_bc_actor",
        "command_projection": {
            "source": list(source_commands),
            "target": list(target_commands),
            "zero_initialized_target_columns": zero_initialized_commands,
        },
        "actor_dade_output_zeroed": zeroed,
        "actor_dade_input_columns_scaled": input_columns_scaled,
        "actor_dade_input_scale": (float(dade_input_scale) if input_columns_scaled else 0.0),
        "path_encoder_present": policy.path_encoder is not None,
        "path_encoder_loaded": path_encoder_loaded,
        "bc_path_dim": int(metadata.get("path_dim", 0)),
        "observation_path_dim": int(policy.actor_layout.path_dim),
        "path_encoder_zero_output": path_encoder_zero_output,
        "path_encoder_alpha": (
            float(policy.path_encoder_alpha) if policy.path_encoder is not None else None
        ),
        "endpoint_encoder_zero_initialized": endpoint_encoder_zero_initialized,
        "actor_current_encoder_initialization": current_encoder_initialization,
        "best_epoch": metadata.get("training_result", {}).get("best_epoch"),
        "best_validation_loss": metadata.get("training_result", {}).get("best_validation_loss"),
    }


def load_pc_bc_frozen_actor_prior(
    policy: PCActorCritic,
    checkpoint_path: Path | str,
    *,
    expected_d_z_total: int,
    expected_n_in_reserved: int,
    expected_n_z_reserved: int,
    expected_actor_hidden_dims: Sequence[int],
    expected_state_encoder_hidden_dims: Sequence[int],
    expected_normalization: PCNormalization,
    expected_activation: str,
    residual_output_init_std: float = 0.0,
) -> dict[str, Any]:
    """Load the frozen BC actor and initialize the trainable residual output."""

    if not getattr(policy, "frozen_bc_residual_enabled", False):
        raise ValueError("policy is not configured for frozen BC residual training")
    if policy.frozen_bc_actor_encoder is None or policy.frozen_bc_actor is None:
        raise RuntimeError("policy has no frozen BC actor modules")
    residual_output_init_std = float(residual_output_init_std)
    if not math.isfinite(residual_output_init_std) or residual_output_init_std < 0.0:
        raise ValueError(
            "residual_output_init_std must be finite and non-negative"
        )

    metadata = read_pc_bc_checkpoint_metadata(checkpoint_path)
    if metadata["action_head"] != PC_ACTION_HEAD_CONTINUOUS:
        raise ValueError(
            "frozen BC residual currently requires a continuous-action BC checkpoint"
        )
    source_command_dim = int(metadata["actor_command_dim"])
    source_path_dim = int(metadata["path_dim"])
    if source_command_dim != policy.actor_layout.command_dim:
        raise ValueError(
            "PC BC command dimension does not match the PPO policy: "
            f"BC={source_command_dim}, policy={policy.actor_layout.command_dim}"
        )
    if source_path_dim != policy.actor_layout.path_dim:
        raise ValueError(
            "PC BC path dimension does not match the PPO policy: "
            f"BC={source_path_dim}, policy={policy.actor_layout.path_dim}"
        )
    if int(metadata["d_z_total"]) != expected_d_z_total:
        raise ValueError("PC BC d_z_total does not match the PPO policy")
    if int(metadata["n_in_reserved"]) != expected_n_in_reserved:
        raise ValueError("PC BC n_in_reserved does not match the PPO policy")
    if int(metadata["n_z_reserved"]) != expected_n_z_reserved:
        raise ValueError("PC BC n_z_reserved does not match the PPO policy")
    if tuple(metadata["actor_hidden_dims"]) != tuple(expected_actor_hidden_dims):
        raise ValueError("PC BC actor hidden dimensions do not match the PPO policy")
    if tuple(metadata["state_encoder_hidden_dims"]) != tuple(expected_state_encoder_hidden_dims):
        raise ValueError("PC BC encoder hidden dimensions do not match the PPO policy")
    if metadata["normalization"] != expected_normalization.as_dict():
        raise ValueError("PC BC normalization does not match the PPO environment")
    if str(metadata.get("activation", "elu")) != str(expected_activation):
        raise ValueError("PC BC activation does not match the PPO policy")

    state_dict = metadata["state_dict"]
    encoder_state = {
        name.removeprefix("actor_encoder."): value
        for name, value in state_dict.items()
        if name.startswith("actor_encoder.")
    }
    actor_state = {
        name.removeprefix("actor."): value
        for name, value in state_dict.items()
        if name.startswith("actor.")
    }
    source_first = actor_state.get("0.weight")
    if not isinstance(source_first, Tensor):
        raise ValueError("BC actor must expose its first linear layer as actor.0.weight")
    expected_source_width = int(expected_d_z_total) + source_command_dim + PC_DADE_FEATURE_DIM
    if source_first.shape[1] != expected_source_width:
        raise ValueError(
            "BC actor input width is inconsistent with its z/command/DADE layout: "
            f"expected {expected_source_width}, got {source_first.shape[1]}"
        )
    policy.frozen_bc_actor_encoder.load_state_dict(encoder_state, strict=True)
    policy.frozen_bc_actor.load_state_dict(actor_state, strict=True)
    if source_path_dim:
        if policy.frozen_bc_path_encoder is None:
            raise RuntimeError("policy has no frozen BC path encoder")
        expected_path_hidden = tuple(policy.frozen_bc_path_encoder_hidden_dims)
        if tuple(metadata["path_encoder_hidden_dims"]) != expected_path_hidden:
            raise ValueError(
                "PC BC path encoder dimensions do not match the PPO policy: "
                f"BC={metadata['path_encoder_hidden_dims']}, policy={expected_path_hidden}"
            )
        path_encoder_state = {
            name.removeprefix("path_encoder."): value
            for name, value in state_dict.items()
            if name.startswith("path_encoder.")
        }
        if not path_encoder_state:
            raise ValueError("BC checkpoint does not contain path encoder weights")
        policy.frozen_bc_path_encoder.load_state_dict(path_encoder_state, strict=True)
        policy.frozen_bc_path_encoder_alpha = float(metadata["path_encoder_alpha"])
    elif policy.frozen_bc_path_encoder is not None:
        raise RuntimeError("policy has a frozen BC path encoder but the BC checkpoint has none")
    bc_current_encoder = metadata.get("actor_current_encoder", "none") == "encoded"
    if bc_current_encoder != (policy.frozen_bc_current_encoder is not None):
        raise ValueError(
            "PC BC current encoder does not match the frozen BC current encoder contract"
        )
    if bc_current_encoder:
        if policy.frozen_bc_current_encoder is None:
            raise RuntimeError("policy has no frozen BC current encoder")
        if tuple(metadata["actor_current_encoder_hidden_dims"]) != tuple(
            policy.frozen_bc_current_encoder_hidden_dims
        ):
            raise ValueError(
                "PC BC current encoder dimensions do not match the PPO policy: "
                f"BC={metadata['actor_current_encoder_hidden_dims']}, "
                f"policy={policy.frozen_bc_current_encoder_hidden_dims}"
            )
        current_encoder_state = {
            name.removeprefix("actor_current_encoder."): value
            for name, value in state_dict.items()
            if name.startswith("actor_current_encoder.")
        }
        if not current_encoder_state:
            raise ValueError("BC checkpoint does not contain current encoder weights")
        policy.frozen_bc_current_encoder.load_state_dict(
            current_encoder_state, strict=True
        )
        policy.frozen_bc_current_encoder_alpha = float(
            metadata["actor_current_encoder_alpha"]
        )
    policy.freeze_frozen_bc_actor()

    residual_initialized = _initialize_last_linear(
        policy.actor,
        std=residual_output_init_std,
    )
    if not residual_initialized:
        raise RuntimeError("residual actor has no linear output layer to initialize")
    dade_head_zeroed = _zero_last_linear(policy.actor_dade_head)
    return {
        "checkpoint_path": metadata["checkpoint_path"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "initialization": "frozen_pc_bc_actor",
        "actor_command_dim": source_command_dim,
        "path_dim": source_path_dim,
        "path_encoder_alpha": policy.frozen_bc_path_encoder_alpha,
        "actor_current_encoder": (
            "encoded" if policy.frozen_bc_current_encoder is not None else "none"
        ),
        "actor_current_encoder_alpha": policy.frozen_bc_current_encoder_alpha,
        "frozen_parameters_require_grad": any(
            parameter.requires_grad
            for module in (
                policy.frozen_bc_actor_encoder,
                policy.frozen_bc_actor,
                policy.frozen_bc_path_encoder,
                policy.frozen_bc_current_encoder,
            )
            if module is not None
            for parameter in module.parameters()
        ),
        "residual_output_initialized": residual_initialized,
        "residual_output_init_std": residual_output_init_std,
        "residual_output_zeroed": residual_output_init_std == 0.0,
        "dade_action_head_output_zeroed": dade_head_zeroed,
        "residual_scale": float(policy.frozen_bc_residual_scale),
        "best_epoch": metadata.get("training_result", {}).get("best_epoch"),
        "best_validation_loss": metadata.get("training_result", {}).get("best_validation_loss"),
    }


__all__ = [
    "PC_ACTION_ENVELOPE",
    "PC_ACTION_HEAD_CONTINUOUS",
    "PC_ACTION_HEAD_SIGN_MAGNITUDE",
    "PC_BC_CHECKPOINT_SCHEMA",
    "PC_BC_DATASET_SCHEMA",
    "PC_SIGN_MAGNITUDE_OUTPUT_DIM",
    "PCBCPolicy",
    "PCBCTrainableActor",
    "PCBCTrainingConfig",
    "PCDemonstrationDataset",
    "SignMagnitudeActionHead",
    "load_pc_bc_actor_prior",
    "load_pc_bc_frozen_actor_prior",
    "load_pc_demonstration_dataset",
    "pc_action_from_actuator",
    "pc_action_full_scale",
    "pc_actuator_from_action",
    "read_pc_bc_checkpoint_metadata",
    "restore_pc_bc_frozen_current_encoder_alpha",
    "restore_pc_bc_frozen_path_encoder_alpha",
    "train_pc_bc",
]
