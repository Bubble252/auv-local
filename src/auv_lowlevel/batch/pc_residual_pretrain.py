"""Supervised teacher-residual pretraining for frozen PC20 BC actors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .pc_bc import (
    PC_ACTION_ENVELOPE,
    PC_BC_CHECKPOINT_SCHEMA,
    PCDemonstrationDataset,
    _command_tensor,
    _episode_split,
    pc_action_full_scale,
    read_pc_bc_checkpoint_metadata,
)
from .pc_normalization import PCNormalization
from .pc_path import normalize_path_features
from .pc_policy import PC_SCHEMA_VERSION, PCActorCritic

PC_RESIDUAL_CHECKPOINT_SCHEMA = "auv_pc_supervised_residual.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pc_history_indices(
    episode_id: np.ndarray,
    step: np.ndarray,
    *,
    short_window: int,
    long_window: int,
    short_stride: int = 1,
    long_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct deployment history windows from recorded episode steps.

    The vector environment fills an episode's initial history with its first
    token, so early steps are clamped to step zero here as well. Inputs must
    contain one row per ``(episode_id, step)`` pair.
    """

    episode_id = np.asarray(episode_id, dtype=np.int64)
    step = np.asarray(step, dtype=np.int64)
    if episode_id.ndim != 1 or step.shape != episode_id.shape:
        raise ValueError("episode_id and step must be one-dimensional and equal length")
    if episode_id.size <= 0:
        raise ValueError("history reconstruction requires at least one sample")
    if np.any(episode_id < 0) or np.any(step < 0):
        raise ValueError("episode_id and step must be non-negative")
    if short_window <= 0 or long_window <= 0 or short_window > long_window:
        raise ValueError("history windows must satisfy 0 < short <= long")
    if short_stride <= 0 or long_stride <= 0:
        raise ValueError("history strides must be positive")
    if short_window * short_stride > long_window * long_stride:
        raise ValueError("short history span must not exceed long history span")

    max_step = int(step.max())
    stride = max_step + 1
    keys = episode_id * stride + step
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if sorted_keys.size > 1 and np.any(np.diff(sorted_keys) == 0):
        raise ValueError("dataset contains duplicate (episode_id, step) rows")

    def window_indices(window: int, history_stride: int) -> np.ndarray:
        offsets = np.arange(window - 1, -1, -1, dtype=np.int64) * history_stride
        result = np.empty((episode_id.size, window), dtype=np.int64)
        for column, offset in enumerate(offsets):
            target_step = np.maximum(step - offset, 0)
            target_key = episode_id * stride + target_step
            positions = np.searchsorted(sorted_keys, target_key)
            if np.any(positions >= sorted_keys.size):
                raise ValueError("dataset is missing required history steps")
            if not np.array_equal(sorted_keys[positions], target_key):
                raise ValueError("dataset is missing required history steps")
            result[:, column] = order[positions]
        return result

    return (
        window_indices(short_window, short_stride),
        window_indices(long_window, long_stride),
    )


def build_pc_transition_indices(
    episode_id: np.ndarray,
    step: np.ndarray,
) -> np.ndarray:
    """Return the index of the next recorded step in each episode, or ``-1``.

    The returned array follows the input row order. Missing steps and the last
    sample of an episode map to ``-1`` so callers can mask them safely.
    """

    episode_id = np.asarray(episode_id, dtype=np.int64)
    step = np.asarray(step, dtype=np.int64)
    if episode_id.ndim != 1 or step.shape != episode_id.shape:
        raise ValueError("episode_id and step must be one-dimensional and equal length")
    if episode_id.size <= 0:
        raise ValueError("transition indices require at least one sample")
    if np.any(episode_id < 0) or np.any(step < 0):
        raise ValueError("episode_id and step must be non-negative")

    stride = int(step.max()) + 1
    keys = episode_id * stride + step
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if sorted_keys.size > 1 and np.any(np.diff(sorted_keys) == 0):
        raise ValueError("dataset contains duplicate (episode_id, step) rows")

    positions = np.searchsorted(sorted_keys, keys + 1)
    valid = positions < sorted_keys.size
    safe_positions = np.minimum(positions, sorted_keys.size - 1)
    valid &= sorted_keys[safe_positions] == keys + 1
    valid &= episode_id[order[safe_positions]] == episode_id
    valid &= step[order[safe_positions]] == step + 1

    result = np.full(episode_id.size, -1, dtype=np.int64)
    result[valid] = order[safe_positions[valid]]
    return result


@dataclass(frozen=True)
class PCResidualTrainingConfig:
    batch_size: int = 4096
    gradient_accumulation_steps: int = 1
    max_epochs: int = 200
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    output_l2: float = 1e-4
    smoothness_l2: float = 0.0
    validation_fraction: float = 0.2
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if (
            self.batch_size <= 0
            or self.gradient_accumulation_steps <= 0
            or self.max_epochs <= 0
            or self.patience <= 0
        ):
            raise ValueError(
                "batch_size, gradient_accumulation_steps, max_epochs, and patience "
                "must be positive"
            )
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.output_l2 < 0.0:
            raise ValueError("output_l2 must be non-negative")
        if self.smoothness_l2 < 0.0:
            raise ValueError("smoothness_l2 must be non-negative")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if not self.device:
            raise ValueError("device must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective_batch_size(self) -> int:
        return int(self.batch_size) * int(self.gradient_accumulation_steps)


def _base_actions_from_state_and_command(
    policy: PCActorCritic,
    state: Tensor,
    command: Tensor,
    actor_path: Tensor | None,
    *,
    batch_size: int,
) -> Tensor:
    if policy.frozen_bc_actor_encoder is None or policy.frozen_bc_actor is None:
        raise RuntimeError("frozen BC actor is not initialized")
    chunks: list[Tensor] = []
    for start in range(0, state.shape[0], batch_size):
        state_batch = state[start : start + batch_size]
        command_batch = command[start : start + batch_size]
        path_batch = (
            None if actor_path is None else actor_path[start : start + batch_size]
        )
        with torch.inference_mode():
            z_t = policy.frozen_bc_actor_encoder(state_batch)
            if policy.frozen_bc_path_encoder is not None:
                if path_batch is None or policy.frozen_bc_path_encoder_alpha is None:
                    raise RuntimeError("frozen BC path encoder is not initialized")
                z_t = z_t + policy.frozen_bc_path_encoder_alpha * policy.frozen_bc_path_encoder(
                    path_batch
                )
            c_hat = state_batch.new_zeros((state_batch.shape[0], 64))
            chunks.append(
                policy.frozen_bc_actor(torch.cat([z_t, command_batch, c_hat], dim=1))
            )
    return torch.cat(chunks, dim=0)


def _residual_outputs(
    policy: PCActorCritic,
    observation: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    chunks: list[Tensor] = []
    mask = observation.new_tensor(policy.frozen_bc_residual_action_mask)
    for start in range(0, observation.shape[0], batch_size):
        residual = policy._actor_residual_mean(
            observation[start : start + batch_size]
        )
        chunks.append(residual * mask)
    return torch.cat(chunks, dim=0)


def _build_observation_tensors(
    policy: PCActorCritic,
    dataset: PCDemonstrationDataset,
    normalization: PCNormalization,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    raw_state = torch.as_tensor(dataset.actor_state, dtype=torch.float32, device=device)
    raw_command = torch.as_tensor(dataset.command, dtype=torch.float32, device=device)
    state = normalization.normalize_actor_state(raw_state)
    command = _command_tensor(raw_command, policy.actor_layout.command_dim)
    actor_path = (
        normalize_path_features(
            torch.as_tensor(dataset.actor_path, dtype=torch.float32, device=device)
        )
        if policy.actor_layout.path_dim
        else None
    )
    if policy.actor_layout.path_dim and actor_path is None:
        raise ValueError("path-conditioned residual training requires actor_path in demos")
    if dataset.current_estimate_body is None:
        if bool(getattr(policy, "dade_enabled", False)):
            raise ValueError("DADE residual training requires current_estimate_body in demos")
        side = torch.zeros(
            (state.shape[0], policy.actor_layout.side_channel_dim),
            dtype=state.dtype,
            device=device,
        )
    else:
        side = normalization.normalize_current(
            torch.as_tensor(dataset.current_estimate_body, dtype=torch.float32, device=device)
        )
    if policy.actor_layout.history_path_dim:
        if actor_path is None:
            raise ValueError("path-conditioned history requires actor_path in demos")
        token = torch.cat([state, actor_path, side], dim=1)
    else:
        token = torch.cat([state, side], dim=1)
    short_indices, long_indices = build_pc_history_indices(
        dataset.episode_id,
        dataset.step,
        short_window=policy.actor_layout.short_window,
        long_window=policy.actor_layout.long_window,
    )
    short = token[torch.as_tensor(short_indices, dtype=torch.long, device=device)]
    long = token[torch.as_tensor(long_indices, dtype=torch.long, device=device)]
    observation = policy.actor_layout.flatten(
        state=state,
        path=actor_path,
        current=side if policy.actor_layout.current_dim else None,
        command=command,
        short=short,
        long=long,
    )
    return observation, state, command


def _label_statistics(target: Tensor) -> dict[str, Any]:
    values = target.detach().to(dtype=torch.float64)
    abs_values = values.abs()
    return {
        "sample_count": int(values.shape[0]),
        "mean": values.mean(dim=0).tolist(),
        "std": values.std(dim=0, unbiased=False).tolist(),
        "abs_mean": abs_values.mean(dim=0).tolist(),
        "abs_max": abs_values.amax(dim=0).tolist(),
        "abs_gt_one_fraction": (abs_values > 1.0).to(torch.float64).mean(dim=0).tolist(),
    }


def _per_channel_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean(dim=0)


def _zero_last_linear(module: nn.Module) -> None:
    linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linears:
        raise RuntimeError("residual actor has no linear output layer to zero")
    nn.init.zeros_(linears[-1].weight)
    if linears[-1].bias is not None:
        nn.init.zeros_(linears[-1].bias)


def load_pc_supervised_residual(
    policy: PCActorCritic,
    checkpoint_path: Path | str,
    *,
    expected_residual_scale: float,
    expected_bc_checkpoint_sha256: str,
    expected_d_z_total: int,
    expected_n_in_reserved: int,
    expected_n_z_reserved: int,
    expected_actor_hidden_dims: Sequence[int],
    expected_state_encoder_hidden_dims: Sequence[int],
    expected_activation: str,
    expected_normalization: PCNormalization,
) -> dict[str, Any]:
    """Load only the trainable residual actor from a supervised checkpoint."""

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != PC_RESIDUAL_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a PC supervised residual checkpoint")
    if not policy.frozen_bc_residual_enabled:
        raise ValueError("supervised residual loading requires frozen BC residual mode")
    if checkpoint.get("base_bc_checkpoint_sha256") != expected_bc_checkpoint_sha256:
        raise ValueError("supervised residual was trained against a different BC checkpoint")
    checkpoint_mask = tuple(
        float(value) for value in checkpoint.get("residual_action_mask", (1.0, 1.0, 1.0))
    )
    if checkpoint_mask != tuple(policy.frozen_bc_residual_action_mask):
        raise ValueError("supervised residual action mask does not match the policy")
    if not np.isclose(
        float(checkpoint.get("residual_scale")),
        float(expected_residual_scale),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("supervised residual scale does not match the training config")
    expected_fields = {
        "d_z_total": expected_d_z_total,
        "n_in_reserved": expected_n_in_reserved,
        "n_z_reserved": expected_n_z_reserved,
        "actor_hidden_dims": list(expected_actor_hidden_dims),
        "state_encoder_hidden_dims": list(expected_state_encoder_hidden_dims),
        "activation": str(expected_activation),
        "normalization": expected_normalization.as_dict(),
    }
    for name, expected in expected_fields.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"supervised residual {name} does not match the policy")
    encoder_state = checkpoint.get("residual_actor_encoder_state_dict")
    actor_state = checkpoint.get("residual_actor_state_dict")
    if not isinstance(encoder_state, dict) or not isinstance(actor_state, dict):
        raise ValueError("supervised residual checkpoint is missing actor state dictionaries")
    policy.actor_encoder.load_state_dict(encoder_state, strict=True)
    policy.actor.load_state_dict(actor_state, strict=True)
    loaded_optional_modules: list[str] = []
    for checkpoint_key, policy_name in (
        ("residual_path_encoder_state_dict", "path_encoder"),
        ("residual_path_endpoint_encoder_state_dict", "path_endpoint_encoder"),
        ("residual_actor_current_encoder_state_dict", "actor_current_encoder"),
    ):
        state = checkpoint.get(checkpoint_key)
        module = getattr(policy, policy_name, None)
        if state:
            if module is None:
                raise ValueError(
                    f"checkpoint contains {policy_name} state but policy does not"
                )
            module.load_state_dict(state, strict=True)
            loaded_optional_modules.append(policy_name)
        elif module is not None and policy_name == "path_encoder" and policy.actor_layout.path_dim:
            raise ValueError(
                "supervised residual checkpoint is missing path encoder state; "
                "path-conditioned residuals require a fresh checkpoint"
            )
    for checkpoint_key, policy_name in (
        ("residual_dade_state_dict", "actor_dade"),
        ("residual_dade_head_state_dict", "actor_dade_head"),
        ("residual_current_head_state_dict", "actor_current_head"),
    ):
        state = checkpoint.get(checkpoint_key)
        module = getattr(policy, policy_name, None)
        if state:
            if module is None:
                raise ValueError(f"checkpoint contains {policy_name} state but policy does not")
            module.load_state_dict(state, strict=True)
        elif module is not None:
            raise ValueError(f"checkpoint is missing {policy_name} state")
    return {
        "schema_version": checkpoint["schema_version"],
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": _sha256(path),
        "base_bc_checkpoint_sha256": checkpoint["base_bc_checkpoint_sha256"],
        "residual_scale": float(checkpoint["residual_scale"]),
        "residual_action_mask": list(checkpoint_mask),
        "dataset_sha256": checkpoint.get("dataset", {}).get("dataset_sha256"),
        "training_result": checkpoint.get("training_result", {}),
        "label_statistics": checkpoint.get("label_statistics", {}),
        "loaded_optional_modules": loaded_optional_modules,
        "dade_enabled": bool(checkpoint.get("infos", {}).get("pc_metadata", {}).get("dade_enabled", False)),
    }


def train_pc_supervised_residual(
    policy: PCActorCritic,
    dataset: PCDemonstrationDataset,
    *,
    bc_checkpoint_path: Path | str,
    config: PCResidualTrainingConfig,
    normalization: PCNormalization,
    output_dir: Path | str | None = None,
    dataset_metadata: Mapping[str, Any] | None = None,
    checkpoint_infos: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a residual actor to ``teacher - frozen_BC`` labels."""

    if not policy.frozen_bc_residual_enabled:
        raise ValueError("PCActorCritic must use frozen_bc_residual_enabled=true")
    if policy.frozen_bc_actor_encoder is None or policy.frozen_bc_actor is None:
        raise RuntimeError("frozen BC actor is not initialized")

    bc_path = Path(bc_checkpoint_path)
    bc_metadata = read_pc_bc_checkpoint_metadata(bc_path)
    if bc_metadata["schema_version"] != PC_BC_CHECKPOINT_SCHEMA:
        raise ValueError("base checkpoint is not a PC BC actor")
    if bc_metadata["normalization"] != normalization.as_dict():
        raise ValueError("BC checkpoint normalization does not match the residual trainer")

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    policy.to(device)
    policy.train()
    _zero_last_linear(policy.actor)
    initial_model_state = {
        name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
    }
    initial_frozen_state = {
        name: value.clone()
        for name, value in initial_model_state.items()
        if name.startswith("frozen_bc_")
    }

    train_indices, validation_indices = _episode_split(
        dataset.episode_id,
        config.validation_fraction,
        config.seed,
    )
    action = torch.as_tensor(dataset.action, dtype=torch.float32, device=device)
    observation, state, command = _build_observation_tensors(
        policy,
        dataset,
        normalization,
        device=device,
    )
    _state, actor_path, _command, _short, _long = policy.actor_layout.split_with_path(
        observation
    )
    base = _base_actions_from_state_and_command(
        policy,
        state,
        command,
        actor_path,
        batch_size=config.batch_size,
    )
    scale = float(policy.frozen_bc_residual_scale)
    if scale <= 0.0:
        raise ValueError("policy residual scale must be positive")
    residual_mask = observation.new_tensor(policy.frozen_bc_residual_action_mask)
    residual_target = (action - base) / scale * residual_mask
    transition_index = torch.as_tensor(
        build_pc_transition_indices(dataset.episode_id, dataset.step),
        dtype=torch.long,
        device=device,
    )
    train_index = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    validation_index = torch.as_tensor(validation_indices, dtype=torch.long, device=device)

    frozen_parameters = [
        parameter
        for module in (
            policy.frozen_bc_actor_encoder,
            policy.frozen_bc_actor,
            policy.frozen_bc_path_encoder,
        )
        if module is not None
        for parameter in module.parameters()
    ]
    if any(parameter.requires_grad for parameter in frozen_parameters):
        raise RuntimeError("frozen BC parameters still require gradients")
    trainable_modules = [policy.actor_encoder, policy.actor]
    for name in (
        "path_encoder",
        "path_endpoint_encoder",
        "actor_current_encoder",
        "actor_dade",
        "actor_dade_head",
        "actor_current_head",
    ):
        module = getattr(policy, name, None)
        if module is not None:
            trainable_modules.append(module)
    trainable_parameters = [
        parameter for module in trainable_modules for parameter in module.parameters()
    ]
    if not trainable_parameters:
        raise RuntimeError("residual actor has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def validation_metrics() -> tuple[float, np.ndarray, np.ndarray]:
        policy.eval()
        with torch.inference_mode():
            prediction = _residual_outputs(
                policy,
                observation[validation_index],
                batch_size=config.batch_size,
            )
            target = residual_target[validation_index]
            residual_mse = float(torch.mean((prediction - target).square()).item())
            residual_mse_channels = _per_channel_mse(prediction, target).cpu().numpy()
            predicted_action = base[validation_index] + scale * prediction
            action_mse_channels = _per_channel_mse(
                predicted_action, action[validation_index]
            ).cpu().numpy()
        policy.train()
        return residual_mse, residual_mse_channels, action_mse_channels

    initial_validation_mse, _, _ = validation_metrics()
    best_validation_mse = initial_validation_mse
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    best_epoch = -1
    stale_epochs = 0
    train_losses: list[float] = []
    validation_losses: list[float] = [initial_validation_mse]
    for epoch in range(config.max_epochs):
        permutation = torch.randperm(train_index.numel(), device=device)
        shuffled = train_index[permutation]
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pending_accumulation = 0
        for start in range(0, shuffled.numel(), config.batch_size):
            batch_index = shuffled[start : start + config.batch_size]
            prediction = _residual_outputs(
                policy,
                observation[batch_index],
                batch_size=config.batch_size,
            )
            target = residual_target[batch_index]
            loss = torch.mean((prediction - target).square())
            aux_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
            aux_coef = float(getattr(policy, "aux_current_coef", 0.0))
            if aux_coef > 0.0:
                aux_loss = policy.actor_auxiliary_current_loss(observation[batch_index])
                loss = loss + aux_coef * aux_loss
            if config.output_l2 > 0.0:
                loss = loss + config.output_l2 * torch.mean(prediction.square())
            if config.smoothness_l2 > 0.0:
                next_index = transition_index[batch_index]
                valid = next_index >= 0
                if torch.any(valid):
                    next_prediction = _residual_outputs(
                        policy,
                        observation[next_index[valid]],
                        batch_size=config.batch_size,
                    )
                    predicted_action = base[batch_index[valid]] + scale * prediction[valid]
                    next_action = base[next_index[valid]] + scale * next_prediction
                    loss = loss + config.smoothness_l2 * torch.mean(
                        (next_action - predicted_action).square()
                    )
            loss_value = loss.detach()
            (loss / config.gradient_accumulation_steps).backward()
            pending_accumulation += 1
            if (
                pending_accumulation >= config.gradient_accumulation_steps
                or start + config.batch_size >= shuffled.numel()
            ):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation = 0
            epoch_loss += float(loss_value.item()) * int(batch_index.numel())
        train_loss = epoch_loss / int(train_index.numel())
        train_losses.append(train_loss)
        validation_mse, _, _ = validation_metrics()
        validation_losses.append(validation_mse)
        if validation_mse < best_validation_mse - 1e-12:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    policy.load_state_dict(best_state)
    policy.eval()
    with torch.inference_mode():
        prediction = _residual_outputs(
            policy,
            observation,
            batch_size=config.batch_size,
        )
        predicted_action = base + scale * prediction
        validation_residual_mse, validation_residual_mse_channels, validation_action_mse_channels = (
            validation_metrics()
        )
        full_action_rmse_normalized = torch.sqrt(
            _per_channel_mse(predicted_action, action)
        ).cpu().numpy()
    full_scale = torch.tensor(pc_action_full_scale(), dtype=torch.float64).numpy()
    physical_scale = np.array([full_scale[0], full_scale[1], 0.5 * full_scale[2]])
    validation_action_rmse_physical = np.sqrt(validation_action_mse_channels) * physical_scale
    training_result = {
        "train_samples": int(train_index.numel()),
        "validation_samples": int(validation_index.numel()),
        "train_episodes": int(np.unique(dataset.episode_id[train_indices]).size),
        "validation_episodes": int(np.unique(dataset.episode_id[validation_indices]).size),
        "best_epoch": best_epoch,
        "epochs_completed": len(train_losses),
        "initial_validation_loss": initial_validation_mse,
        "best_validation_loss": best_validation_mse,
        "validation_residual_rmse": np.sqrt(
            validation_residual_mse_channels
        ).tolist(),
        "validation_action_rmse_normalized": np.sqrt(
            validation_action_mse_channels
        ).tolist(),
        "validation_action_rmse_physical": validation_action_rmse_physical.tolist(),
        "full_action_rmse_normalized": full_action_rmse_normalized.tolist(),
        "train_loss": train_losses,
        "validation_loss": validation_losses,
        "aux_current_coef": float(getattr(policy, "aux_current_coef", 0.0)),
        "dade_enabled": bool(getattr(policy, "dade_enabled", False)),
        "output_l2": config.output_l2,
        "smoothness_l2": config.smoothness_l2,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.effective_batch_size,
    }
    residual_abs = prediction.detach().abs()
    label_statistics = _label_statistics(residual_target)
    label_statistics["prediction_abs_mean"] = residual_abs.mean(dim=0).tolist()
    label_statistics["prediction_abs_max"] = residual_abs.amax(dim=0).tolist()
    frozen_drift = 0.0
    for name, value in initial_frozen_state.items():
        current = policy.state_dict()[name].detach().cpu()
        if not current.is_floating_point() or not value.is_floating_point():
            continue
        frozen_drift = max(frozen_drift, float(torch.max(torch.abs(current - value)).item()))

    checkpoint_base = {
        "schema_version": PC_RESIDUAL_CHECKPOINT_SCHEMA,
        "policy_kind": "pc_supervised_residual",
        "observation_schema": PC_SCHEMA_VERSION,
        "action_envelope": PC_ACTION_ENVELOPE,
        "base_bc_checkpoint_path": str(bc_path.resolve()),
        "base_bc_checkpoint_sha256": _sha256(bc_path),
        "residual_scale": scale,
        "residual_action_mask": list(policy.frozen_bc_residual_action_mask),
        "d_z_total": int(bc_metadata["d_z_total"]),
        "n_in_reserved": int(bc_metadata["n_in_reserved"]),
        "n_z_reserved": int(bc_metadata["n_z_reserved"]),
        "state_encoder_hidden_dims": list(bc_metadata["state_encoder_hidden_dims"]),
        "actor_hidden_dims": list(bc_metadata["actor_hidden_dims"]),
        "activation": str(bc_metadata.get("activation", "elu")),
        "normalization": normalization.as_dict(),
        "dataset": dict(dataset_metadata or {}),
        "training_config": config.as_dict(),
        "label_statistics": label_statistics,
        "frozen_bc_parameter_max_abs_drift": frozen_drift,
        "residual_actor_encoder_state_dict": {
            name.removeprefix("actor_encoder."): value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
            if name.startswith("actor_encoder.")
        },
        "residual_actor_state_dict": {
            name.removeprefix("actor."): value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
            if name.startswith("actor.")
        },
        "residual_path_encoder_state_dict": {
            name.removeprefix("path_encoder."): value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
            if name.startswith("path_encoder.") and policy.path_encoder is not None
        },
        "residual_path_endpoint_encoder_state_dict": {
            name.removeprefix("path_endpoint_encoder."): value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
            if name.startswith("path_endpoint_encoder.")
            and policy.path_endpoint_encoder is not None
        },
        "residual_actor_current_encoder_state_dict": {
            name.removeprefix("actor_current_encoder."): value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
            if name.startswith("actor_current_encoder.")
            and policy.actor_current_encoder is not None
        },
        "residual_dade_state_dict": {
            name.removeprefix("actor_dade."): value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
            if name.startswith("actor_dade.") and policy.actor_dade is not None
        },
        "residual_dade_head_state_dict": {
            name.removeprefix("actor_dade_head."): value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
            if name.startswith("actor_dade_head.") and policy.actor_dade_head is not None
        },
        "residual_current_head_state_dict": {
            name.removeprefix("actor_current_head."): value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
            if name.startswith("actor_current_head.") and policy.actor_current_head is not None
        },
        "model_state_dict": {
            name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
        },
        "infos": dict(checkpoint_infos or {}),
    }
    checkpoint_base["training_result"] = training_result
    result = {**training_result, "frozen_bc_parameter_max_abs_drift": frozen_drift}

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        initial_path = output / "model_initial.pt"
        final_path = output / "final.pt"
        initial_checkpoint = dict(checkpoint_base)
        initial_checkpoint["model_state_dict"] = {
            name: value.clone() for name, value in initial_model_state.items()
        }
        initial_checkpoint["residual_actor_encoder_state_dict"] = {
            name.removeprefix("actor_encoder."): value.clone()
            for name, value in initial_model_state.items()
            if name.startswith("actor_encoder.")
        }
        initial_checkpoint["residual_actor_state_dict"] = {
            name.removeprefix("actor."): value.clone()
            for name, value in initial_model_state.items()
            if name.startswith("actor.")
        }
        initial_checkpoint["residual_path_encoder_state_dict"] = {
            name.removeprefix("path_encoder."): value.clone()
            for name, value in initial_model_state.items()
            if name.startswith("path_encoder.") and policy.path_encoder is not None
        }
        initial_checkpoint["residual_path_endpoint_encoder_state_dict"] = {
            name.removeprefix("path_endpoint_encoder."): value.clone()
            for name, value in initial_model_state.items()
            if name.startswith("path_endpoint_encoder.")
            and policy.path_endpoint_encoder is not None
        }
        initial_checkpoint["residual_actor_current_encoder_state_dict"] = {
            name.removeprefix("actor_current_encoder."): value.clone()
            for name, value in initial_model_state.items()
            if name.startswith("actor_current_encoder.")
            and policy.actor_current_encoder is not None
        }
        initial_checkpoint["residual_dade_state_dict"] = {
            name.removeprefix("actor_dade."): value.clone()
            for name, value in initial_model_state.items()
            if name.startswith("actor_dade.") and policy.actor_dade is not None
        }
        initial_checkpoint["residual_dade_head_state_dict"] = {
            name.removeprefix("actor_dade_head."): value.clone()
            for name, value in initial_model_state.items()
            if name.startswith("actor_dade_head.") and policy.actor_dade_head is not None
        }
        initial_checkpoint["residual_current_head_state_dict"] = {
            name.removeprefix("actor_current_head."): value.clone()
            for name, value in initial_model_state.items()
            if name.startswith("actor_current_head.") and policy.actor_current_head is not None
        }
        initial_checkpoint["training_result"] = {}
        initial_checkpoint["label_statistics"] = {}
        torch.save(initial_checkpoint, initial_path)
        torch.save(checkpoint_base, final_path)
        result.update(
            {
                "initial_checkpoint_path": str(initial_path.resolve()),
                "initial_checkpoint_sha256": _sha256(initial_path),
                "checkpoint_path": str(final_path.resolve()),
                "checkpoint_sha256": _sha256(final_path),
            }
        )
        manifest = {
            key: value
            for key, value in checkpoint_base.items()
            if not key.endswith("_state_dict") and key != "model_state_dict"
        }
        manifest.update(
            {
                "status": "completed",
                "initial_checkpoint_path": result["initial_checkpoint_path"],
                "initial_checkpoint_sha256": result["initial_checkpoint_sha256"],
                "checkpoint_path": result["checkpoint_path"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )
        (output / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "training_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


__all__ = [
    "PC_RESIDUAL_CHECKPOINT_SCHEMA",
    "PCResidualTrainingConfig",
    "build_pc_history_indices",
    "build_pc_transition_indices",
    "load_pc_supervised_residual",
    "train_pc_supervised_residual",
]
