from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from auv_core.contracts import FloatArray, Reference, State
from torch import nn

from .observation import PREVIOUS_RESIDUAL_OBSERVATION_INDICES
from .observation_schema import ObservationSchema, resolve_observation_schema

ACTION_SCALES = np.array(
    [np.deg2rad(15.0), np.deg2rad(15.0), 1525.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class BCTrainingConfig:
    hidden_sizes: tuple[int, ...] = (256, 128)
    batch_size: int = 4096
    max_epochs: int = 200
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must be non-empty and positive")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, max_epochs, and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")


@dataclass(frozen=True)
class DemoDataset:
    observation: np.ndarray
    action: np.ndarray
    episode_id: np.ndarray
    seed: np.ndarray
    step: np.ndarray
    time_s: np.ndarray
    actual_actuator: np.ndarray
    state_position_ned_m: np.ndarray
    state_quaternion_xyzw: np.ndarray
    state_linear_velocity_body_mps: np.ndarray
    state_angular_velocity_body_rps: np.ndarray
    reference_position_ned_m: np.ndarray
    reference_quaternion_xyzw: np.ndarray
    current_ned_mps: np.ndarray
    teacher_post_integral_rpm: np.ndarray
    observation_schema: str | None = None

    def __post_init__(self) -> None:
        count = self.observation.shape[0]
        resolve_observation_schema(self.observation_dim, self.observation_schema)
        if self.action.shape != (count, 3):
            raise ValueError("action must have shape (N, 3)")
        for name in (
            "episode_id",
            "seed",
            "step",
            "time_s",
            "teacher_post_integral_rpm",
        ):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
        for name in (
            "actual_actuator",
            "state_position_ned_m",
            "state_linear_velocity_body_mps",
            "state_angular_velocity_body_rps",
            "reference_position_ned_m",
            "current_ned_mps",
        ):
            if getattr(self, name).shape != (count, 3):
                raise ValueError(f"{name} must have shape (N, 3)")
        for name in ("state_quaternion_xyzw", "reference_quaternion_xyzw"):
            if getattr(self, name).shape != (count, 4):
                raise ValueError(f"{name} must have shape (N, 4)")
        if not np.all(np.isfinite(self.observation)):
            raise ValueError("observation contains NaN or Inf")
        if not np.all(np.isfinite(self.action)):
            raise ValueError("action contains NaN or Inf")

    @property
    def sample_count(self) -> int:
        return int(self.observation.shape[0])

    @property
    def observation_dim(self) -> int:
        if self.observation.ndim != 2:
            raise ValueError("observation must have shape (N, D)")
        return int(self.observation.shape[1])

    @property
    def schema(self) -> ObservationSchema:
        return resolve_observation_schema(self.observation_dim, self.observation_schema)


@dataclass(frozen=True)
class BCTrainingResult:
    train_samples: int
    validation_samples: int
    best_epoch: int
    best_validation_loss: float
    train_loss: list[float]
    validation_loss: list[float]
    validation_mse_normalized: FloatArray
    validation_mse_actuator_units: FloatArray
    training_config_sha256: str | None
    checkpoint_path: str | None


class BCActor(nn.Module):
    def __init__(self, observation_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = observation_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            previous = size
        layers.append(nn.Linear(previous, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class BCPolicy:
    def __init__(
        self,
        checkpoint_path: Path | str,
        device: str = "cpu",
        *,
        mask_previous_residual: bool | None = None,
        expected_observation_dim: int | None = None,
        expected_observation_schema: str | None = None,
        expected_speed_feedback: str | None = None,
        expected_current_contract: str | None = None,
    ) -> None:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=device,
            weights_only=True,
        )
        hidden_sizes = tuple(int(value) for value in checkpoint["hidden_sizes"])
        self.observation_dim = int(checkpoint["observation_dim"])
        stored_schema = checkpoint.get("observation_schema")
        self.observation_schema = resolve_observation_schema(
            self.observation_dim,
            None if stored_schema is None else str(stored_schema),
        )
        if (
            expected_observation_dim is not None
            and self.observation_dim != expected_observation_dim
        ):
            raise ValueError(
                "checkpoint observation dimension mismatch: "
                f"expected {expected_observation_dim}, got {self.observation_dim}"
            )
        if (
            expected_observation_schema is not None
            and self.observation_schema.version != expected_observation_schema
        ):
            raise ValueError(
                "checkpoint observation schema mismatch: "
                f"expected {expected_observation_schema!r}, "
                f"got {self.observation_schema.version!r}"
            )
        self.speed_feedback = str(
            checkpoint.get("observation_speed_feedback") or "water_relative"
        )
        self.current_contract = str(
            checkpoint.get("observation_current_contract") or "legacy"
        )
        if (
            expected_speed_feedback is not None
            and self.speed_feedback != expected_speed_feedback
        ):
            raise ValueError(
                "checkpoint observation speed-feedback mismatch: "
                f"expected {expected_speed_feedback!r}, got {self.speed_feedback!r}"
            )
        if (
            expected_current_contract is not None
            and self.current_contract != expected_current_contract
        ):
            raise ValueError(
                "checkpoint current-contract mismatch: "
                f"expected {expected_current_contract!r}, got {self.current_contract!r}"
            )
        self.model = BCActor(self.observation_dim, hidden_sizes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.observation_mean = np.asarray(
            checkpoint["observation_mean"], dtype=np.float32
        )
        self.observation_std = np.asarray(checkpoint["observation_std"], dtype=np.float32)
        self.action_scales = np.asarray(checkpoint["action_scales"], dtype=np.float32)
        if self.observation_mean.shape != (self.observation_dim,) or (
            self.observation_std.shape != (self.observation_dim,)
        ):
            raise ValueError("checkpoint observation normalization has the wrong shape")
        if self.action_scales.shape != (3,):
            raise ValueError("checkpoint action scales have the wrong shape")
        residual_std = self.observation_std[list(PREVIOUS_RESIDUAL_OBSERVATION_INDICES)]
        if mask_previous_residual is None:
            mask_previous_residual = bool(np.all(residual_std <= 1e-4))
        self.mask_previous_residual = bool(mask_previous_residual)
        self.metadata: dict[str, Any] = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"state_dict"}
        }
        self.metadata["observation_schema"] = self.observation_schema.version
        self.metadata["observation_speed_feedback"] = self.speed_feedback
        self.metadata["observation_current_contract"] = self.current_contract
        self.metadata["mask_previous_residual"] = self.mask_previous_residual

    def act(self, observation: FloatArray) -> FloatArray:
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.ndim not in {1, 2} or (
            observation_array.shape[-1] != self.observation_dim
        ):
            raise ValueError(
                "observation must have shape "
                f"(D,) or (N, D) with D={self.observation_dim}, "
                f"got {observation_array.shape}"
            )
        normalized = (observation_array - self.observation_mean) / self.observation_std
        if self.mask_previous_residual:
            normalized = normalized.copy()
            normalized[..., list(PREVIOUS_RESIDUAL_OBSERVATION_INDICES)] = 0.0
        with torch.inference_mode():
            tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            prediction = self.model(tensor).cpu().numpy()
        if prediction.ndim == 1:
            prediction = prediction[np.newaxis, :]
        prediction[:, 0] = np.clip(prediction[:, 0], -1.0, 1.0)
        prediction[:, 1] = np.clip(prediction[:, 1], -1.0, 1.0)
        prediction[:, 2] = np.clip(prediction[:, 2], 0.0, 1.0)
        return prediction * self.action_scales

    def __call__(self, observation: FloatArray, state: State, reference: Reference) -> FloatArray:
        del state, reference
        return np.asarray(self.act(observation)[0], dtype=np.float64)


def load_demo_dataset(path: Path | str) -> DemoDataset:
    with np.load(Path(path)) as payload:
        required = {
            "observation",
            "action",
            "episode_id",
            "seed",
            "step",
            "time_s",
            "actual_actuator",
            "state_position_ned_m",
            "state_quaternion_xyzw",
            "state_linear_velocity_body_mps",
            "state_angular_velocity_body_rps",
            "reference_position_ned_m",
            "reference_quaternion_xyzw",
            "current_ned_mps",
            "teacher_post_integral_rpm",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"demo dataset is missing fields: {missing}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        if "observation_schema" in payload.files:
            stored_schema = np.asarray(payload["observation_schema"])
            if stored_schema.shape != ():
                raise ValueError("observation_schema must be a scalar")
            arrays["observation_schema"] = str(stored_schema.item())
        return DemoDataset(**arrays)


def split_by_episode(
    episode_id: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_id)
    if episodes.size < 2:
        raise ValueError("at least two episodes are required for a train/validation split")
    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    validation_count = int(round(episodes.size * validation_fraction))
    validation_count = min(max(validation_count, 1), episodes.size - 1)
    validation_episodes = set(shuffled[:validation_count].tolist())
    validation_mask = np.isin(episode_id, list(validation_episodes))
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def train_bc(
    dataset: DemoDataset,
    config: BCTrainingConfig,
    *,
    output_dir: Path | str | None = None,
    metadata: dict[str, Any] | None = None,
    observation_speed_feedback: str | None = None,
    observation_current_contract: str | None = None,
    observation_expose_current: bool | None = None,
) -> BCTrainingResult:
    schema = dataset.schema
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train_indices, validation_indices = split_by_episode(
        dataset.episode_id,
        config.validation_fraction,
        config.seed,
    )
    observation = np.asarray(dataset.observation, dtype=np.float32)
    action = np.asarray(dataset.action, dtype=np.float32) / ACTION_SCALES
    observation_mean = np.mean(observation[train_indices], axis=0, dtype=np.float64).astype(
        np.float32
    )
    observation_std = np.std(observation[train_indices], axis=0, dtype=np.float64).astype(
        np.float32
    )
    observation_std = np.maximum(observation_std, 1e-5)
    normalized_observation = (observation - observation_mean) / observation_std

    train_x = torch.as_tensor(normalized_observation[train_indices], device=device)
    train_y = torch.as_tensor(action[train_indices], device=device)
    validation_x = torch.as_tensor(normalized_observation[validation_indices], device=device)
    validation_y = torch.as_tensor(action[validation_indices], device=device)
    model = BCActor(observation.shape[1], config.hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_validation = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(config.max_epochs):
        model.train()
        permutation = torch.randperm(train_x.shape[0], device=device)
        batch_losses: list[float] = []
        for start in range(0, train_x.shape[0], config.batch_size):
            indices = permutation[start : start + config.batch_size]
            prediction = model(train_x[indices])
            loss = loss_function(prediction, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_x)
            validation_loss = float(
                loss_function(validation_prediction, validation_y).cpu()
            )
        train_losses.append(float(np.mean(batch_losses)))
        validation_losses.append(validation_loss)
        if validation_loss < best_validation - 1e-8:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_prediction = model(validation_x).cpu().numpy()
    validation_mse_normalized = np.mean(
        np.square(validation_prediction - action[validation_indices]), axis=0
    )
    validation_mse_actuator_units = validation_mse_normalized * np.square(ACTION_SCALES)

    checkpoint_path: Path | None = None
    training_config_sha256_value: str | None = None
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output / "pi0.pt"
        training_config = {
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "validation_fraction": config.validation_fraction,
            "seed": config.seed,
            "hidden_sizes": list(config.hidden_sizes),
        }
        training_config_sha256 = hashlib.sha256(
            json.dumps(
                training_config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        training_config_sha256_value = training_config_sha256
        checkpoint = {
            "schema_version": "auv_odarrl_bc.v1",
            "observation_dim": int(observation.shape[1]),
            "observation_schema": schema.version,
            "hidden_sizes": list(config.hidden_sizes),
            "state_dict": best_state,
            "observation_mean": torch.from_numpy(observation_mean),
            "observation_std": torch.from_numpy(observation_std),
            "action_scales": torch.from_numpy(ACTION_SCALES),
            "training_config": training_config,
            "training_config_sha256": training_config_sha256,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "train_samples": int(train_indices.size),
            "validation_samples": int(validation_indices.size),
            "metadata": {} if metadata is None else metadata,
        }
        if observation_speed_feedback is not None:
            checkpoint["observation_speed_feedback"] = observation_speed_feedback
        if observation_current_contract is not None:
            checkpoint["observation_current_contract"] = observation_current_contract
        if observation_expose_current is not None:
            checkpoint["observation_expose_current"] = bool(observation_expose_current)
        torch.save(checkpoint, checkpoint_path)
    return BCTrainingResult(
        train_samples=int(train_indices.size),
        validation_samples=int(validation_indices.size),
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
        train_loss=train_losses,
        validation_loss=validation_losses,
        validation_mse_normalized=validation_mse_normalized,
        validation_mse_actuator_units=validation_mse_actuator_units,
        training_config_sha256=training_config_sha256_value,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
    )


__all__ = [
    "ACTION_SCALES",
    "BCActor",
    "BCPolicy",
    "BCTrainingConfig",
    "BCTrainingResult",
    "DemoDataset",
    "load_demo_dataset",
    "split_by_episode",
    "train_bc",
]
