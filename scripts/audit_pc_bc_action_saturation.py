#!/usr/bin/env python3
"""Audit per-channel saturation of a PC BC policy on recorded demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.actions import ActuatorLimits
from auv_lowlevel.batch.actions import BatchedPureActionPipeline
from auv_lowlevel.batch.pc_bc import PCBCPolicy, load_pc_demonstration_dataset
from auv_lowlevel.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--path-zero", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        help="optional environment config used for rate-limit diagnostics",
    )
    parser.add_argument("--dt-s", type=float, default=0.02)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_actions(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("prediction and target must be rank-2 arrays")
    if prediction.shape != target.shape or prediction.shape[1] != 3:
        raise ValueError("prediction and target must both have shape (N, 3)")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")

    error = prediction.astype(np.float64) - target.astype(np.float64)
    predicted_abs = np.abs(prediction[:, :2]).astype(np.float64)
    target_abs = np.abs(target[:, :2]).astype(np.float64)
    high_action = target_abs >= 0.5
    nonzero_magnitude = target_abs > 0.05
    teacher_saturated = target_abs >= threshold
    predicted_saturated = predicted_abs >= threshold
    sign_accuracy: list[float | None] = []
    magnitude_mae: list[float | None] = []
    saturation_recall: list[float | None] = []
    for channel in range(2):
        accuracy_mask = high_action[:, channel]
        sign_accuracy.append(
            float(
                np.mean(
                    np.sign(prediction[:, channel][accuracy_mask])
                    == np.sign(target[:, channel][accuracy_mask])
                )
            )
            if accuracy_mask.any()
            else None
        )
        magnitude_mask = nonzero_magnitude[:, channel]
        magnitude_mae.append(
            float(
                np.mean(
                    np.abs(
                        predicted_abs[:, channel][magnitude_mask]
                        - target_abs[:, channel][magnitude_mask]
                    )
                )
            )
            if magnitude_mask.any()
            else None
        )
        channel_saturated = teacher_saturated[:, channel]
        saturation_recall.append(
            float(np.mean(predicted_saturated[:, channel][channel_saturated]))
            if channel_saturated.any()
            else None
        )
    correlation: list[float] = []
    for channel in range(3):
        predicted = prediction[:, channel].astype(np.float64)
        expected = target[:, channel].astype(np.float64)
        if np.std(predicted) <= 1e-12 or np.std(expected) <= 1e-12:
            correlation.append(float("nan"))
        else:
            correlation.append(float(np.corrcoef(predicted, expected)[0, 1]))
    return {
        "samples": int(prediction.shape[0]),
        "threshold": float(threshold),
        "teacher_channel_saturation_rate": (
            np.abs(target) >= threshold
        ).mean(axis=0).astype(float).tolist(),
        "predicted_channel_saturation_rate": (
            np.abs(prediction) >= threshold
        ).mean(axis=0).astype(float).tolist(),
        "teacher_any_surface_saturation_rate": float(
            np.any(np.abs(target[:, :2]) >= threshold, axis=1).mean()
        ),
        "predicted_any_surface_saturation_rate": float(
            np.any(np.abs(prediction[:, :2]) >= threshold, axis=1).mean()
        ),
        "predicted_mean_abs": np.abs(prediction).mean(axis=0).astype(float).tolist(),
        "teacher_mean_abs": np.abs(target).mean(axis=0).astype(float).tolist(),
        "sign_accuracy_high_action": sign_accuracy,
        "magnitude_mae_nonzero": magnitude_mae,
        "teacher_saturation_recall": saturation_recall,
        "teacher_zero_fraction_surface": (target_abs < 0.05).mean(axis=0).astype(float).tolist(),
        "predicted_zero_fraction_surface": (predicted_abs < 0.05)
        .mean(axis=0)
        .astype(float)
        .tolist(),
        "predicted_teacher_saturation_ratio": [
            float(predicted_saturated[:, channel].mean() / teacher_saturated[:, channel].mean())
            if teacher_saturated[:, channel].mean() > 0.0
            else None
            for channel in range(2)
        ],
        "normalized_mae": np.abs(error).mean(axis=0).astype(float).tolist(),
        "normalized_rmse": np.sqrt(np.square(error).mean(axis=0)).astype(float).tolist(),
        "channel_correlation": correlation,
    }


def _ordered_pairs(
    episode_id: np.ndarray,
    step: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((step, episode_id))
    ordered_episode = episode_id[order]
    same_episode = ordered_episode[1:] == ordered_episode[:-1]
    return order, same_episode


def _reversal_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    episode_id: np.ndarray,
    step: np.ndarray,
    *,
    dt_s: float,
) -> dict[str, Any]:
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    order, same_episode = _ordered_pairs(episode_id, step)
    predicted = prediction[order]
    expected = target[order]

    def _rates(values: np.ndarray) -> dict[str, float]:
        previous = values[:-1, :2]
        current = values[1:, :2]
        reversal = (
            same_episode[:, None]
            & (np.abs(previous) > 0.5)
            & (np.abs(current) > 0.5)
            & (previous * current < 0.0)
        )
        denominator = max(float(same_episode.sum()) * dt_s, 1e-12)
        channel_rates = (reversal.sum(axis=0) / denominator).astype(float).tolist()
        any_rate = float(reversal.any(axis=1).sum() / denominator)
        return {
            "channel_reversal_rate_per_s": channel_rates,
            "any_channel_reversal_rate_per_s": any_rate,
        }

    return {
        "predicted": _rates(predicted),
        "teacher": _rates(expected),
    }


def _rate_limit_metrics(
    prediction: np.ndarray,
    teacher_action: np.ndarray,
    actual_actuator: np.ndarray,
    episode_id: np.ndarray,
    step: np.ndarray,
    *,
    config_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    limits = ActuatorLimits(
        max_rudder_angle_rad=config.actuator.policy_max_rudder_angle_rad,
        max_stern_plane_angle_rad=config.actuator.policy_max_stern_plane_angle_rad,
        max_rudder_rate_rad_s=config.actuator.policy_max_rudder_rate_rad_s,
        max_stern_plane_rate_rad_s=config.actuator.policy_max_stern_plane_rate_rad_s,
        max_propeller_rpm=config.actuator.policy_max_propeller_rpm,
        material_tolerance_fraction=config.actuator.material_tolerance_fraction,
    )
    pipeline = BatchedPureActionPipeline(
        limits,
        dtype=torch.float64,
        validate_finite=False,
    )
    order, _ = _ordered_pairs(episode_id, step)
    ordered_episode = episode_id[order]
    action = torch.as_tensor(prediction[order], dtype=torch.float64)
    teacher = torch.as_tensor(teacher_action[order], dtype=torch.float64)
    actuator = torch.as_tensor(actual_actuator[order], dtype=torch.float64)
    previous = torch.cat([torch.zeros_like(actuator[:1]), actuator[:-1]], dim=0)
    reset = torch.as_tensor(ordered_episode[1:] != ordered_episode[:-1])
    previous[1:][reset] = 0.0
    with torch.inference_mode():
        applied = pipeline.apply(action, previous, config.plant.dt_s)
        teacher_applied = pipeline.apply(teacher, previous, config.plant.dt_s)
        rate_hit = applied.rate_clipped.bool()
        material_rate_hit = applied.material_rate_clipped.bool()
        teacher_rate_hit = teacher_applied.rate_clipped.bool()
    return {
        "config_path": str(config_path),
        "dt_s": float(config.plant.dt_s),
        "predicted_rate_limit_channel_fraction": rate_hit.double().mean(dim=0).tolist(),
        "predicted_rate_limit_any_channel_fraction": float(
            rate_hit.any(dim=1).double().mean().item()
        ),
        "predicted_material_rate_limit_any_channel_fraction": float(
            material_rate_hit.any(dim=1).double().mean().item()
        ),
        "teacher_rate_limit_channel_fraction": teacher_rate_hit.double().mean(dim=0).tolist(),
        "teacher_rate_limit_any_channel_fraction": float(
            teacher_rate_hit.any(dim=1).double().mean().item()
        ),
        "predicted_minus_teacher_rate_limit_any_channel_fraction": float(
            rate_hit.any(dim=1).double().mean().item()
            - teacher_rate_hit.any(dim=1).double().mean().item()
        ),
    }


def _audit(
    checkpoint_path: Path,
    demos_path: Path,
    *,
    threshold: float,
    batch_size: int,
    device: str,
    path_zero: bool,
    config_path: Path | None,
    dt_s: float,
) -> dict[str, Any]:
    dataset, metadata = load_pc_demonstration_dataset(demos_path)
    policy = PCBCPolicy(checkpoint_path, device=device)
    if policy.model.actor_command_dim != dataset.actor_command_dim:
        raise ValueError("checkpoint and demonstrations disagree on actor_command_dim")
    if policy.model.path_dim != dataset.path_dim:
        raise ValueError("checkpoint and demonstrations disagree on path_dim")
    if dataset.actor_path is None and policy.model.path_dim:
        raise ValueError("checkpoint requires actor_path but demonstrations do not contain it")

    predictions: list[np.ndarray] = []
    command = torch.as_tensor(dataset.command, dtype=torch.float32)
    actor_path = (
        torch.as_tensor(dataset.actor_path, dtype=torch.float32)
        if dataset.actor_path is not None
        else None
    )
    for start in range(0, dataset.sample_count, batch_size):
        stop = min(start + batch_size, dataset.sample_count)
        batch_path = None
        if actor_path is not None:
            batch_path = torch.zeros_like(actor_path[start:stop]) if path_zero else actor_path[start:stop]
        prediction = policy.act_inference(
            torch.as_tensor(dataset.actor_state[start:stop], dtype=torch.float32),
            command=command[start:stop],
            actor_path=batch_path,
        )
        predictions.append(prediction.detach().cpu().numpy())
    prediction_array = np.concatenate(predictions, axis=0)
    result = _summarize_actions(prediction_array, dataset.action, threshold=threshold)
    result["reversal"] = _reversal_metrics(
        prediction_array,
        dataset.action,
        dataset.episode_id,
        dataset.step,
        dt_s=dt_s,
    )
    if config_path is not None:
        result["rate_limit"] = _rate_limit_metrics(
            prediction_array,
            dataset.action,
            dataset.actual_actuator,
            dataset.episode_id,
            dataset.step,
            config_path=config_path,
        )
    else:
        result["rate_limit"] = None
    full_scale = np.asarray(policy.metadata["action_full_scale"], dtype=np.float64)
    result.update(
        {
            "schema_version": "auv_pc_bc_action_saturation_audit.v1",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "demos_path": str(demos_path.resolve()),
            "dataset_sha256": metadata.get("dataset_sha256", _sha256(demos_path)),
            "actor_command_dim": dataset.actor_command_dim,
            "path_dim": dataset.path_dim,
            "path_zero": bool(path_zero),
            "device": device,
            "action_head": policy.metadata["action_head"],
            "sign_deadband": policy.metadata["sign_deadband"],
            "episode_count": dataset.episode_count,
            "dt_s": float(dt_s),
            "physical_mae": (np.asarray(result["normalized_mae"]) * full_scale).tolist(),
            "physical_rmse": (np.asarray(result["normalized_rmse"]) * full_scale).tolist(),
        }
    )
    return result


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.dt_s <= 0.0:
        raise ValueError("--dt-s must be positive")
    if not args.checkpoint.is_file() or not args.demos.is_file():
        raise FileNotFoundError("checkpoint and demos must both exist")
    if args.config is not None and not args.config.is_file():
        raise FileNotFoundError(f"config does not exist: {args.config}")
    result = _audit(
        args.checkpoint,
        args.demos,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
        path_zero=args.path_zero,
        config_path=args.config,
        dt_s=args.dt_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
