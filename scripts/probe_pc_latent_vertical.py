#!/usr/bin/env python3
"""Goal137: probe vertical/pitch information and residual-head sensitivity.

The script is deliberately offline. It loads the Goal86 P-A residual BC
checkpoint, reconstructs ``z_t`` with the training normalization, fits linear
probes on an episode-isolated split, measures central-difference Jacobians in
physical input units, and compares two small supervised adaptation protocols.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from auv_lowlevel.batch.pc_bc import PCBCTrainableActor, _command_tensor
from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import normalize_path_features

TARGET_NAMES = ("vertical_error_m", "pitch_error_rad", "q_rad_s", "w_mps")
FEATURE_SET_NAMES = ("state_z", "path_z", "z_t")
KEY_INPUT_NAMES = (
    "vertical_error",
    "tangent_pitch_error",
    "theta",
    "q",
    "w",
)
REFERENCE_INPUT_NAMES = (
    "cross_track",
    "heading_error",
    "phi",
    "r",
)

# (name, kind, column, physical epsilon, group)
INPUT_SPECS = (
    ("vertical_error", "path", 1, 0.05, "key"),
    ("tangent_pitch_error", "path", 3, 0.01, "key"),
    ("theta", "state", 4, 0.01, "key"),
    ("q", "state", 7, 0.01, "key"),
    ("w", "state", 11, 0.01, "key"),
    ("cross_track", "path", 0, 0.05, "reference"),
    ("heading_error", "path", 2, 0.01, "reference"),
    ("phi", "state", 3, 0.01, "reference"),
    ("r", "state", 8, 0.01, "reference"),
)


@dataclass(frozen=True)
class AdaptationConfig:
    epochs: int = 120
    patience: int = 20
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 137


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=86051)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--jac-samples", type=int, default=2048)
    parser.add_argument("--adapt-epochs", type=int, default=120)
    parser.add_argument("--adapt-patience", type=int, default=20)
    parser.add_argument("--adapt-batch-size", type=int, default=4096)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--adapt-weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_checkpoint_and_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], PCBCTrainableActor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"not a PC BC checkpoint: {checkpoint_path}")
    config = checkpoint.get("training_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing training_config")
    if str(checkpoint.get("action_head", "continuous")) != "continuous":
        raise ValueError("Goal137 currently supports continuous residual checkpoints only")
    model = PCBCTrainableActor(
        d_z_total=int(config["d_z_total"]),
        n_in_reserved=int(config["n_in_reserved"]),
        n_z_reserved=int(config["n_z_reserved"]),
        state_encoder_hidden_dims=tuple(int(v) for v in config["state_encoder_hidden_dims"]),
        actor_hidden_dims=tuple(int(v) for v in config["actor_hidden_dims"]),
        activation=str(config["activation"]),
        actor_command_dim=int(config["actor_command_dim"]),
        path_dim=int(config["path_dim"]),
        path_encoder_hidden_dims=tuple(int(v) for v in config["path_encoder_hidden_dims"]),
        path_encoder_alpha=float(config["path_encoder_alpha"]),
        actor_current_encoder_enabled=bool(config["actor_current_encoder_enabled"]),
        actor_current_encoder_hidden_dims=tuple(
            int(v) for v in config["actor_current_encoder_hidden_dims"]
        ),
        actor_current_encoder_alpha=float(config["actor_current_encoder_alpha"]),
        action_head=str(config["action_head"]),
        sign_deadband=float(config.get("sign_deadband", 0.05)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return checkpoint, model


def load_arrays(path: Path, *, max_samples: int | None = None) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        required = {
            "actor_state",
            "command",
            "actor_path",
            "residual_normalized",
            "residual_physical",
            "episode_id",
            "seed",
            "step",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"dataset is missing arrays: {missing}")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    count = int(arrays["actor_state"].shape[0])
    if max_samples is not None and count > max_samples:
        step = max(count // int(max_samples), 1)
        indices = np.arange(0, count, step, dtype=np.int64)[:max_samples]
        arrays = {name: value[indices] for name, value in arrays.items()}
    return arrays


def episode_split(
    episode_id: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the BC training split: whole episodes, no row leakage."""

    episodes = np.unique(np.asarray(episode_id, dtype=np.int64))
    if episodes.size < 2:
        raise ValueError("at least two episodes are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    shuffled = episodes.copy()
    np.random.default_rng(int(seed)).shuffle(shuffled)
    validation_count = min(
        max(int(round(shuffled.size * float(validation_fraction))), 1),
        shuffled.size - 1,
    )
    validation_episodes = set(int(value) for value in shuffled[:validation_count])
    validation_mask = np.isin(episode_id, list(validation_episodes))
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def prepare_tensors(
    arrays: dict[str, np.ndarray],
    model: PCBCTrainableActor,
    *,
    action_full_scale: np.ndarray,
    device: torch.device,
) -> dict[str, Tensor | np.ndarray]:
    normalization = PCNormalization()
    raw_state = torch.as_tensor(arrays["actor_state"], dtype=torch.float32, device=device)
    raw_path = torch.as_tensor(arrays["actor_path"], dtype=torch.float32, device=device)
    raw_command = torch.as_tensor(arrays["command"], dtype=torch.float32, device=device)
    state = normalization.normalize_actor_state(raw_state)
    path = normalize_path_features(raw_path, path_dim=int(model.path_dim))
    command = _command_tensor(raw_command, int(model.actor_command_dim))
    current = raw_state.new_zeros((raw_state.shape[0], 3))
    return {
        "normalization": normalization,
        "raw_state": raw_state,
        "raw_path": raw_path,
        "raw_command": raw_command,
        "state": state,
        "path": path,
        "command": command,
        "current": current,
        "episode_id": np.asarray(arrays["episode_id"], dtype=np.int64),
        "seed": np.asarray(arrays["seed"], dtype=np.int64),
        "step": np.asarray(arrays["step"], dtype=np.int64),
        # The BC checkpoint predicts residual / full_scale.  The npz
        # ``residual_normalized`` field uses the RL residual denominator
        # (0.25 * full_scale), so using it directly would be a 4x label error.
        "target": torch.as_tensor(
            np.asarray(arrays["residual_physical"], dtype=np.float64)
            / np.asarray(action_full_scale, dtype=np.float64),
            dtype=torch.float32,
            device=device,
        ),
    }


def encode_latents(
    model: PCBCTrainableActor,
    tensors: dict[str, Tensor | np.ndarray],
) -> dict[str, Tensor]:
    state = tensors["state"]
    path = tensors["path"]
    current = tensors["current"]
    assert isinstance(state, Tensor)
    assert isinstance(path, Tensor)
    assert isinstance(current, Tensor)
    with torch.no_grad():
        state_z = model.actor_encoder(state)
        path_z = (
            path.new_zeros(state_z.shape)
            if model.path_encoder is None
            else model.path_encoder_alpha * model.path_encoder(path)
        )
        current_z = (
            state_z.new_zeros(state_z.shape)
            if model.actor_current_encoder is None
            else model.actor_current_encoder_alpha * model.actor_current_encoder(current)
        )
        z_t = state_z + path_z + current_z
    return {"state_z": state_z, "path_z": path_z + current_z, "z_t": z_t}


def predict_residual(
    model: PCBCTrainableActor,
    state: Tensor,
    path: Tensor,
    current: Tensor,
    command: Tensor,
) -> Tensor:
    raw = model.forward_raw(state, command, path, current)
    return model.decode_action(raw)


def target_arrays(tensors: dict[str, Tensor | np.ndarray]) -> dict[str, np.ndarray]:
    raw_path = tensors["raw_path"]
    raw_state = tensors["raw_state"]
    assert isinstance(raw_path, Tensor)
    assert isinstance(raw_state, Tensor)
    return {
        "vertical_error_m": raw_path[:, 1].detach().cpu().numpy().astype(np.float64),
        "pitch_error_rad": raw_path[:, 3].detach().cpu().numpy().astype(np.float64),
        "q_rad_s": raw_state[:, 7].detach().cpu().numpy().astype(np.float64),
        "w_mps": raw_state[:, 11].detach().cpu().numpy().astype(np.float64),
    }


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2:
        return 0.0
    p = prediction - prediction.mean()
    t = target - target.mean()
    denominator = float(np.linalg.norm(p) * np.linalg.norm(t))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(p, t) / denominator)


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    residual = prediction - target
    sse = float(np.sum(residual * residual))
    sst = float(np.sum((target - target.mean()) ** 2))
    r2 = 0.0 if sst <= 1e-12 else 1.0 - sse / sst
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(residual))),
        "pearson": _pearson(prediction, target),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
    }


def ridge_probe(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    x_train = np.asarray(features[train_index], dtype=np.float64)
    x_validation = np.asarray(features[validation_index], dtype=np.float64)
    if x_train.ndim != 2 or x_validation.ndim != 2:
        raise ValueError("features must be rank-2")
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_train = (x_train - mean) / std
    x_validation = (x_validation - mean) / std
    gram = x_train.T @ x_train
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    result: dict[str, Any] = {}
    for name, target in targets.items():
        y_train = np.asarray(target[train_index], dtype=np.float64)
        y_validation = np.asarray(target[validation_index], dtype=np.float64)
        y_mean = float(y_train.mean())
        weights = np.linalg.solve(gram, x_train.T @ (y_train - y_mean))
        prediction = x_validation @ weights + y_mean
        result[name] = _regression_metrics(prediction, y_validation)
    return result


def probe_all_targets(
    latents: dict[str, Tensor],
    targets: dict[str, np.ndarray],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for feature_name in FEATURE_SET_NAMES:
        features = latents[feature_name].detach().cpu().numpy()
        result[feature_name] = ridge_probe(
            features,
            targets,
            train_index,
            validation_index,
            alpha=alpha,
        )
    return result


def _state_dict_drift(
    initial: dict[str, Tensor],
    adapted: dict[str, Tensor],
    prefixes: Sequence[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for prefix in prefixes:
        keys = [
            key
            for key in initial
            if key.startswith(prefix) and initial[key].is_floating_point()
        ]
        if not keys:
            continue
        values[prefix] = max(
            float((adapted[key].detach().cpu() - initial[key].detach().cpu()).abs().max())
            for key in keys
        )
    return values


def _all_parameter_update_norm(initial: dict[str, Tensor], adapted: dict[str, Tensor]) -> float:
    squared = 0.0
    for key, value in initial.items():
        if key not in adapted or not value.is_floating_point():
            continue
        delta = adapted[key].detach().cpu().double() - value.detach().cpu().double()
        squared += float(delta.square().sum())
    return math.sqrt(squared)


def _full_scale(checkpoint: dict[str, Any]) -> np.ndarray:
    scale = np.asarray(checkpoint.get("action_full_scale"), dtype=np.float64)
    if scale.shape != (3,):
        raise ValueError("checkpoint action_full_scale must have shape (3,)")
    return scale


def jacobian_summary(
    model: PCBCTrainableActor,
    tensors: dict[str, Tensor | np.ndarray],
    indices: np.ndarray,
    *,
    action_full_scale: np.ndarray,
    max_samples: int,
    input_specs: Iterable[tuple[str, str, int, float, str]] = INPUT_SPECS,
) -> dict[str, Any]:
    if indices.size == 0:
        raise ValueError("Jacobian evaluation requires non-empty indices")
    take = min(int(max_samples), int(indices.size))
    chosen = indices[np.linspace(0, indices.size - 1, take, dtype=np.int64)]
    raw_state = tensors["raw_state"]
    raw_path = tensors["raw_path"]
    command = tensors["command"]
    current = tensors["current"]
    assert isinstance(raw_state, Tensor)
    assert isinstance(raw_path, Tensor)
    assert isinstance(command, Tensor)
    assert isinstance(current, Tensor)
    normalization = PCNormalization()
    state_base = raw_state[chosen].clone()
    path_base = raw_path[chosen].clone()
    command_base = command[chosen]
    current_base = current[chosen]

    def forward(raw_state_value: Tensor, raw_path_value: Tensor) -> Tensor:
        return predict_residual(
            model,
            normalization.normalize_actor_state(raw_state_value),
            normalize_path_features(raw_path_value, path_dim=int(model.path_dim)),
            current_base,
            command_base,
        )

    with torch.no_grad():
        base_action = forward(state_base, path_base)
    base_norm = torch.linalg.vector_norm(base_action, dim=1)
    base_norm_mean = float(base_norm.mean().detach().cpu())
    result: dict[str, Any] = {
        "sample_count": int(take),
        "residual_norm_mean": base_norm_mean,
        "residual_abs_mean": base_action.abs().mean(dim=0).detach().cpu().tolist(),
        "inputs": {},
    }
    specs = tuple(input_specs)
    for name, kind, column, eps, group in specs:
        state_plus = state_base.clone()
        state_minus = state_base.clone()
        path_plus = path_base.clone()
        path_minus = path_base.clone()
        if kind == "state":
            state_plus[:, column] += float(eps)
            state_minus[:, column] -= float(eps)
            raw_values = state_base[:, column]
        elif kind == "path":
            path_plus[:, column] += float(eps)
            path_minus[:, column] -= float(eps)
            raw_values = path_base[:, column]
        else:
            raise ValueError(f"unsupported input kind: {kind}")
        with torch.no_grad():
            action_plus = forward(state_plus, path_plus)
            action_minus = forward(state_minus, path_minus)
        jacobian = (action_plus - action_minus) / (2.0 * float(eps))
        jacobian_abs_mean = jacobian.abs().mean(dim=0)
        jacobian_l2 = torch.linalg.vector_norm(jacobian, dim=1).mean()
        input_std = float(raw_values.std(unbiased=False).detach().cpu())
        response_per_std = (
            0.0
            if base_norm_mean <= 1e-12
            else float(jacobian_l2.detach().cpu()) * input_std / base_norm_mean
        )
        result["inputs"][name] = {
            "kind": kind,
            "column": int(column),
            "eps": float(eps),
            "group": group,
            "input_mean": float(raw_values.mean().detach().cpu()),
            "input_std": input_std,
            "jacobian_abs_mean_normalized": jacobian_abs_mean.detach().cpu().tolist(),
            "jacobian_abs_mean_physical": (
                jacobian_abs_mean.detach().cpu().numpy() * action_full_scale
            ).tolist(),
            "jacobian_l2_mean_normalized": float(jacobian_l2.detach().cpu()),
            "response_per_std": response_per_std,
            "perturbation_action_delta_mean": float(
                torch.linalg.vector_norm(action_plus - action_minus, dim=1).mean().detach().cpu()
            ),
        }
    key_values = [result["inputs"][name]["response_per_std"] for name in KEY_INPUT_NAMES]
    reference_values = [
        result["inputs"][name]["response_per_std"] for name in REFERENCE_INPUT_NAMES
    ]
    result["key_response_median"] = float(np.median(key_values))
    result["reference_response_median"] = float(np.median(reference_values))
    result["key_to_reference_ratio"] = (
        None
        if result["reference_response_median"] <= 1e-12
        else result["key_response_median"] / result["reference_response_median"]
    )
    return result


def regression_metrics_for_residual(
    prediction: Tensor,
    target: Tensor,
    *,
    action_full_scale: np.ndarray,
) -> dict[str, Any]:
    pred = prediction.detach().cpu().numpy().astype(np.float64)
    true = target.detach().cpu().numpy().astype(np.float64)
    per_channel: list[dict[str, Any]] = []
    for index in range(3):
        per_channel.append(_regression_metrics(pred[:, index], true[:, index]))
    residual = pred - true
    return {
        "sample_count": int(pred.shape[0]),
        "mse_normalized": float(np.mean(residual * residual)),
        "rmse_normalized": float(np.sqrt(np.mean(residual * residual))),
        "rmse_physical": (
            np.sqrt(np.mean(residual * residual, axis=0)) * action_full_scale
        ).tolist(),
        "per_channel": per_channel,
        "mean_prediction": pred.mean(axis=0).tolist(),
        "mean_target": true.mean(axis=0).tolist(),
    }


def set_trainable(model: PCBCTrainableActor, mode: str) -> dict[str, Any]:
    if mode not in {"head_only", "last_layer"}:
        raise ValueError(f"unsupported adaptation mode: {mode}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    if mode == "last_layer":
        last_state_layer = model.actor_encoder.network[-1]
        if not isinstance(last_state_layer, nn.Linear):
            raise RuntimeError("actor_encoder final layer is not Linear")
        for parameter in last_state_layer.parameters():
            parameter.requires_grad_(True)
        if model.path_encoder is not None:
            path_linears = [
                layer for layer in model.path_encoder.modules() if isinstance(layer, nn.Linear)
            ]
            for parameter in path_linears[-1].parameters():
                parameter.requires_grad_(True)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    return {
        "mode": mode,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "trainable_parameter_names": names,
    }


def _evaluate_model(
    model: PCBCTrainableActor,
    tensors: dict[str, Tensor | np.ndarray],
    validation_index: np.ndarray,
    *,
    action_full_scale: np.ndarray,
) -> dict[str, Any]:
    state = tensors["state"]
    path = tensors["path"]
    current = tensors["current"]
    command = tensors["command"]
    target = tensors["target"]
    assert isinstance(state, Tensor)
    assert isinstance(path, Tensor)
    assert isinstance(current, Tensor)
    assert isinstance(command, Tensor)
    assert isinstance(target, Tensor)
    index = torch.as_tensor(validation_index, dtype=torch.long, device=state.device)
    with torch.no_grad():
        prediction = predict_residual(
            model,
            state[index],
            path[index],
            current[index],
            command[index],
        )
    return regression_metrics_for_residual(
        prediction,
        target[index],
        action_full_scale=action_full_scale,
    )


def reinitialize_actor_head(model: PCBCTrainableActor, *, seed: int) -> None:
    """Reset only the actor MLP so both adaptation arms share one new head."""

    torch.manual_seed(int(seed))
    for module in model.actor.modules():
        reset = getattr(module, "reset_parameters", None)
        if reset is not None and module is not model.actor:
            reset()


def train_adaptation(
    checkpoint_path: Path,
    tensors: dict[str, Tensor | np.ndarray],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    mode: str,
    config: AdaptationConfig,
    action_full_scale: np.ndarray,
    device: torch.device,
) -> tuple[PCBCTrainableActor, dict[str, Any]]:
    _checkpoint, model = load_checkpoint_and_model(checkpoint_path, device=device)
    checkpoint_metrics = _evaluate_model(
        model,
        tensors,
        validation_index,
        action_full_scale=action_full_scale,
    )
    checkpoint_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    reinitialize_actor_head(model, seed=int(config.seed))
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    trainable = set_trainable(model, mode)
    state = tensors["state"]
    path = tensors["path"]
    current = tensors["current"]
    command = tensors["command"]
    target = tensors["target"]
    assert isinstance(state, Tensor)
    assert isinstance(path, Tensor)
    assert isinstance(current, Tensor)
    assert isinstance(command, Tensor)
    assert isinstance(target, Tensor)
    train_tensor = torch.as_tensor(train_index, dtype=torch.long, device=device)
    validation_tensor = torch.as_tensor(validation_index, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    reinitialized_metrics = _evaluate_model(
        model,
        tensors,
        validation_index,
        action_full_scale=action_full_scale,
    )
    best_loss = float(reinitialized_metrics["mse_normalized"])
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for epoch in range(int(config.epochs)):
        model.train()
        shuffled = train_tensor[torch.randperm(train_tensor.numel(), generator=generator)]
        loss_sum = 0.0
        count = 0
        for start in range(0, shuffled.numel(), int(config.batch_size)):
            batch = shuffled[start : start + int(config.batch_size)]
            prediction = predict_residual(
                model,
                state[batch],
                path[batch],
                current[batch],
                command[batch],
            )
            loss = torch.mean((prediction - target[batch]).square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config.grad_clip),
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(batch.numel())
            count += int(batch.numel())
        train_loss = loss_sum / max(count, 1)
        model.eval()
        with torch.no_grad():
            validation_prediction = predict_residual(
                model,
                state[validation_tensor],
                path[validation_tensor],
                current[validation_tensor],
                command[validation_tensor],
            )
            validation_loss = float(
                torch.mean((validation_prediction - target[validation_tensor]).square()).cpu()
            )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break
    model.load_state_dict(best_state, strict=True)
    model.eval()
    adapted_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    metrics = _evaluate_model(
        model,
        tensors,
        validation_index,
        action_full_scale=action_full_scale,
    )
    return model, {
        "trainable": trainable,
        "checkpoint_metrics": checkpoint_metrics,
        "reinitialized_head_metrics": reinitialized_metrics,
        "adapted_metrics": metrics,
        "best_epoch": int(best_epoch),
        "epochs_completed": len(train_losses),
        "best_validation_mse": float(best_loss),
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "encoder_drift_from_reinitialized_head": _state_dict_drift(
            initial_state,
            adapted_state,
            ("actor_encoder.", "path_encoder.", "actor_current_encoder."),
        ),
        "encoder_drift_from_checkpoint": _state_dict_drift(
            checkpoint_state,
            adapted_state,
            ("actor_encoder.", "path_encoder.", "actor_current_encoder."),
        ),
        "all_parameter_update_norm": _all_parameter_update_norm(
            initial_state,
            adapted_state,
        ),
    }


def decide(summary: dict[str, Any]) -> dict[str, str]:
    z_t_probe = summary["probe"]["z_t"]
    retained = sum(
        int(z_t_probe[name]["r2"] >= 0.50 and z_t_probe[name]["pearson"] >= 0.70)
        for name in TARGET_NAMES
    )
    clearly_lost = sum(
        int(z_t_probe[name]["r2"] < 0.25) for name in TARGET_NAMES
    )
    if retained >= 3:
        probe_status = "RETAINED"
    elif clearly_lost >= 3:
        probe_status = "NOT_RETAINED"
    else:
        probe_status = "AMBIGUOUS"

    initial_jac = summary["jacobian"]["initial"]
    key_values = [
        initial_jac["inputs"][name]["response_per_std"] for name in KEY_INPUT_NAMES
    ]
    low_count = sum(int(value < 0.05) for value in key_values)
    ratio = initial_jac["key_to_reference_ratio"]
    if low_count >= 4 or (ratio is not None and ratio < 0.50):
        response_status = "LOW_VERTICAL_RESPONSE"
    elif sum(int(value >= 0.10) for value in key_values) >= 4:
        response_status = "RESPONSIVE"
    else:
        response_status = "PARTIAL"

    head_only = summary["adaptation"]["head_only"]["adapted_metrics"]
    last_layer = summary["adaptation"]["last_layer"]["adapted_metrics"]
    improvement = (
        (head_only["rmse_normalized"] - last_layer["rmse_normalized"])
        / max(head_only["rmse_normalized"], 1e-12)
    )
    head_jac = summary["jacobian"]["adaptation_head_only"]["key_response_median"]
    last_jac = summary["jacobian"]["adaptation_last_layer"]["key_response_median"]
    sensitivity_change = (
        (last_jac - head_jac) / max(abs(head_jac), 1e-12)
        if head_jac is not None and last_jac is not None
        else 0.0
    )
    if improvement < 0.02 and abs(sensitivity_change) < 0.20:
        last_layer_status = "NOT_A_PRIMARY_BOTTLENECK"
    elif improvement >= 0.05 or sensitivity_change >= 0.50:
        last_layer_status = "LAST_LAYER_HELPS"
    else:
        last_layer_status = "WEAK_LAST_LAYER_EFFECT"

    if probe_status == "RETAINED" and response_status == "LOW_VERTICAL_RESPONSE":
        chain = "LATENT_PRESENT_HEAD_UNDERUSES_IT"
    elif probe_status == "NOT_RETAINED":
        chain = "LATENT_INFORMATION_LOST_OR_NOT_LINEARLY_ACCESSIBLE"
    elif response_status == "RESPONSIVE":
        chain = "NOT_PRIMARILY_ENCODER_OR_HEAD_SENSITIVITY"
    else:
        chain = "MIXED_OR_AMBIGUOUS"
    return {
        "probe_status": probe_status,
        "response_status": response_status,
        "last_layer_status": last_layer_status,
        "chain": chain,
        "retained_target_count": str(retained),
        "clearly_lost_target_count": str(clearly_lost),
        "low_key_response_count": str(low_count),
        "last_layer_relative_rmse_improvement": f"{improvement:.6f}",
        "last_layer_sensitivity_change": f"{sensitivity_change:.6f}",
    }


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    verdict = summary["verdict"]
    lines = [
        "# Goal137 结果摘要",
        "",
        "**性质**：离线诊断；验证集为 Goal86 训练 episode 的内部 26/6 拆分，不是正式 held-out。",
        "",
        "## 裁决",
        "",
        f"- probe：`{verdict['probe_status']}`（retained={verdict['retained_target_count']}/4）",
        f"- action response：`{verdict['response_status']}`（low={verdict['low_key_response_count']}/5）",
        f"- last-layer：`{verdict['last_layer_status']}`",
        f"- chain：`{verdict['chain']}`",
        "",
        "## Latent probe（验证 episode）",
        "",
        "| target | state_z R² / r | path_z R² / r | z_t R² / r | z_t RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in TARGET_NAMES:
        cells = []
        for feature in FEATURE_SET_NAMES:
            item = summary["probe"][feature][target]
            cells.append(f"{_fmt(item['r2'], 3)} / {_fmt(item['pearson'], 3)}")
        rmse = summary["probe"]["z_t"][target]["rmse"]
        lines.append(f"| `{target}` | {cells[0]} | {cells[1]} | {cells[2]} | {_fmt(rmse, 5)} |")

    lines += [
        "",
        "## Residual Jacobian（初始 BC 残差头）",
        "",
        "| input | group | response/std | J L2 (norm/physical) | physical J per action |",
        "|---|---|---:|---:|---:|",
    ]
    for name, item in summary["jacobian"]["initial"]["inputs"].items():
        physical = ", ".join(f"{value:.5g}" for value in item["jacobian_abs_mean_physical"])
        lines.append(
            f"| `{name}` | {item['group']} | {_fmt(item['response_per_std'], 5)} | "
            f"{_fmt(item['jacobian_l2_mean_normalized'], 5)} | {physical} |"
        )
    lines += [
        "",
        f"- key median response/std：`{_fmt(summary['jacobian']['initial']['key_response_median'], 5)}`",
        f"- reference median response/std：`{_fmt(summary['jacobian']['initial']['reference_response_median'], 5)}`",
        f"- key/reference ratio：`{_fmt(summary['jacobian']['initial']['key_to_reference_ratio'], 5)}`",
        "",
        "## 冻结编码器 vs 微调最后一层",
        "",
        "| arm | best epoch | val RMSE norm | val RMSE physical | status |",
        "|---|---:|---:|---|---:|",
    ]
    for arm in ("head_only", "last_layer"):
        item = summary["adaptation"][arm]
        metrics = item["adapted_metrics"]
        physical = ", ".join(f"{value:.4g}" for value in metrics["rmse_physical"])
        lines.append(
            f"| `{arm}` | {item['best_epoch']} | {_fmt(metrics['rmse_normalized'], 6)} | "
            f"{physical} | {item['trainable']['mode']} |"
        )
    lines += [
        "",
        f"- 相对 RMSE 改善（last-layer vs head-only）：`{verdict['last_layer_relative_rmse_improvement']}`",
        f"- key sensitivity 变化：`{verdict['last_layer_sensitivity_change']}`",
        "",
        "## 限制",
        "",
        "- 这是离线线性可解码性与局部 Jacobian，不是闭环 success。",
        "- 本机没有 `P_D3` PPO final checkpoint；结论严格对应 Goal86 residual BC head。",
        "- 6 个验证 episode 曾用于 BC early stopping；本结果不能写成正式泛化估计。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.jac_samples = min(int(args.jac_samples), 32)
        args.adapt_epochs = min(int(args.adapt_epochs), 2)
        args.adapt_patience = 1
    if args.adapt_epochs <= 0 or args.jac_samples <= 0:
        raise ValueError("adapt_epochs and jac_samples must be positive")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, model = load_checkpoint_and_model(args.bc_checkpoint, device=device)
    action_full_scale = _full_scale(checkpoint)
    arrays = load_arrays(args.dataset, max_samples=2048 if args.smoke else None)
    tensors = prepare_tensors(
        arrays,
        model,
        action_full_scale=action_full_scale,
        device=device,
    )
    episode_id = tensors["episode_id"]
    assert isinstance(episode_id, np.ndarray)
    train_index, validation_index = episode_split(
        episode_id,
        float(args.validation_fraction),
        int(args.split_seed),
    )
    if np.intersect1d(
        np.unique(episode_id[train_index]), np.unique(episode_id[validation_index])
    ).size:
        raise RuntimeError("episode leakage between train and validation")
    targets = target_arrays(tensors)
    latents = encode_latents(model, tensors)
    probe = probe_all_targets(
        latents,
        targets,
        train_index,
        validation_index,
        alpha=float(args.ridge_alpha),
    )
    initial_jacobian = jacobian_summary(
        model,
        tensors,
        validation_index,
        action_full_scale=action_full_scale,
        max_samples=int(args.jac_samples),
    )
    adapt_config = AdaptationConfig(
        epochs=int(args.adapt_epochs),
        patience=int(args.adapt_patience),
        batch_size=int(args.adapt_batch_size),
        learning_rate=float(args.adapt_lr),
        weight_decay=float(args.adapt_weight_decay),
        seed=int(args.seed),
    )
    adaptation: dict[str, Any] = {}
    adapted_models: dict[str, PCBCTrainableActor] = {}
    for mode in ("head_only", "last_layer"):
        adapted_model, result = train_adaptation(
            args.bc_checkpoint,
            tensors,
            train_index,
            validation_index,
            mode=mode,
            config=adapt_config,
            action_full_scale=action_full_scale,
            device=device,
        )
        adapted_models[mode] = adapted_model
        adaptation[mode] = result
        checkpoint_path = output_dir / f"{mode}.pt"
        torch.save(
            {
                "schema_version": "goal137_adaptation.v1",
                "base_checkpoint": str(args.bc_checkpoint.resolve()),
                "base_checkpoint_sha256": _sha256(args.bc_checkpoint),
                "mode": mode,
                "trainable": result["trainable"],
                "best_epoch": result["best_epoch"],
                "best_validation_mse": result["best_validation_mse"],
                "state_dict": adapted_model.state_dict(),
            },
            checkpoint_path,
        )
        result["checkpoint_path"] = str(checkpoint_path)

    adapted_jacobians = {
        "adaptation_head_only": jacobian_summary(
            adapted_models["head_only"],
            tensors,
            validation_index,
            action_full_scale=action_full_scale,
            max_samples=int(args.jac_samples),
        ),
        "adaptation_last_layer": jacobian_summary(
            adapted_models["last_layer"],
            tensors,
            validation_index,
            action_full_scale=action_full_scale,
            max_samples=int(args.jac_samples),
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": "goal137_latent_vertical_probe.v1",
        "command": [sys.executable, *sys.argv],
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "bc_checkpoint": str(args.bc_checkpoint.resolve()),
        "bc_checkpoint_sha256": _sha256(args.bc_checkpoint),
        "device": str(device),
        "split": {
            "validation_fraction": float(args.validation_fraction),
            "split_seed": int(args.split_seed),
            "train_samples": int(train_index.size),
            "validation_samples": int(validation_index.size),
            "train_episodes": int(np.unique(episode_id[train_index]).size),
            "validation_episodes": int(np.unique(episode_id[validation_index]).size),
            "train_episode_ids": np.unique(episode_id[train_index]).tolist(),
            "validation_episode_ids": np.unique(episode_id[validation_index]).tolist(),
        },
        "probe": probe,
        "jacobian": {"initial": initial_jacobian, **adapted_jacobians},
        "adaptation": adaptation,
        "action_full_scale": action_full_scale.tolist(),
        "config": {
            "ridge_alpha": float(args.ridge_alpha),
            "jac_samples": int(args.jac_samples),
            "adaptation": {
                "epochs": int(args.adapt_epochs),
                "patience": int(args.adapt_patience),
                "batch_size": int(args.adapt_batch_size),
                "learning_rate": float(args.adapt_lr),
                "weight_decay": float(args.adapt_weight_decay),
                "seed": int(args.seed),
            },
            "smoke": bool(args.smoke),
        },
        "limitations": [
            "P_D3 PPO final checkpoint is unavailable locally; this audit uses the Goal86 residual BC checkpoint.",
            "The six validation episodes were used for BC early stopping, so probe metrics are diagnostic rather than a formal held-out estimate.",
            "Jacobians are local and offline; they are not closed-loop success or authority measurements.",
        ],
    }
    summary["verdict"] = decide(summary)
    summary = _jsonable(summary)
    (output_dir / "probe.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "probe.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "verdict": summary["verdict"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
