from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel._paths import resolve_from_root
from auv_lowlevel.bc import BCPolicy, split_by_episode
from auv_lowlevel.config import load_config
from auv_lowlevel.observation_schema import (
    ObservationSchema,
    observation_schema_for_dim,
)
from auv_lowlevel.ppo_config import load_training_config
from auv_lowlevel.residual import (
    ResidualLabelSet,
    build_residual_labels,
    label_statistics,
    mask_residual_actor_inputs,
)
from auv_lowlevel.warmstart import ACTOR_WARMSTART_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised residual-actor warm start for the ODARRL I arm"
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--teacher-rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--anchor-demos", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-fraction", type=float, default=0.25)
    parser.add_argument("--teacher-mix", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--output-l2", type=float, default=1e-4)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_from_root(REPO_ROOT, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_observation(
    observation: np.ndarray,
    expected_schema: ObservationSchema,
) -> np.ndarray:
    return mask_residual_actor_inputs(
        observation,
        expected_schema=expected_schema,
    )


def _base_actions(policy: BCPolicy, observation: np.ndarray) -> np.ndarray:
    chunks = [
        policy.act(observation[start : start + 8192])
        for start in range(0, observation.shape[0], 8192)
    ]
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _load_teacher_rollouts(
    paths: list[Path],
    expected_schema: ObservationSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    next_episode = 0
    for path in paths:
        with np.load(path) as payload:
            action_fields = [
                field
                for field in ("teacher_action", "base_command", "action")
                if field in payload.files
            ]
            if "observation" not in payload.files or not action_fields:
                raise ValueError(
                    f"{path} must contain observation and one of "
                    "teacher_action/base_command/action"
                )
            observation = np.asarray(payload["observation"], dtype=np.float32)
            action = np.asarray(payload[action_fields[0]], dtype=np.float32)
            stored_schema = (
                None
                if "observation_schema" not in payload.files
                else str(np.asarray(payload["observation_schema"]).item())
            )
            if "episode_id" in payload.files:
                raw_episode = np.asarray(payload["episode_id"], dtype=np.int64)
            else:
                raw_episode = np.zeros(observation.shape[0], dtype=np.int64)
        if observation.ndim != 2 or observation.shape[1] != expected_schema.dim:
            raise ValueError(
                f"{path} has observation shape {observation.shape}; "
                f"expected (N, {expected_schema.dim})"
            )
        if action.shape != (observation.shape[0], 3):
            raise ValueError(f"{path} has inconsistent observation/action shapes")
        resolve_observation_schema_from_source(stored_schema, expected_schema, path)
        if raw_episode.shape != (observation.shape[0],):
            raise ValueError(f"{path} has an inconsistent episode_id shape")
        unique_episode = np.unique(raw_episode)
        episode_id = np.searchsorted(unique_episode, raw_episode) + next_episode
        next_episode += int(unique_episode.size)
        observations.append(observation)
        actions.append(action)
        episodes.append(episode_id.astype(np.int64, copy=False))
        sources.append(np.full(observation.shape[0], 0, dtype=np.int8))
        metadata.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "samples": int(observation.shape[0]),
                "episode_count": int(unique_episode.size),
                "observation_dim": int(observation.shape[1]),
                "observation_schema": expected_schema.version,
                "role": "teacher_rollout",
            }
        )
    return (
        np.concatenate(observations),
        np.concatenate(actions),
        np.concatenate(episodes),
        np.concatenate(sources),
        metadata,
    )


def resolve_observation_schema_from_source(
    stored_schema: str | None,
    expected_schema: ObservationSchema,
    path: Path,
) -> None:
    if stored_schema is not None and stored_schema != expected_schema.version:
        raise ValueError(
            f"{path} stores observation_schema={stored_schema!r}, "
            f"expected {expected_schema.version!r}"
        )


def _load_anchor_demos(
    paths: list[Path],
    *,
    expected_schema: ObservationSchema,
    sample_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not paths or sample_limit <= 0:
        return (
            np.empty((0, expected_schema.dim), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            [],
        )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    episode_offset = 0
    for path in paths:
        with np.load(path) as payload:
            required = {"observation", "action", "episode_id"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"{path} is missing fields: {missing}")
            observation = np.asarray(payload["observation"], dtype=np.float32)
            action = np.asarray(payload["action"], dtype=np.float32)
            episode_id = np.asarray(payload["episode_id"], dtype=np.int64) + episode_offset
            stored_schema = (
                None
                if "observation_schema" not in payload.files
                else str(np.asarray(payload["observation_schema"]).item())
            )
        if (
            observation.shape != (action.shape[0], expected_schema.dim)
            or action.shape[1:] != (3,)
        ):
            raise ValueError(f"{path} has inconsistent observation/action shapes")
        resolve_observation_schema_from_source(stored_schema, expected_schema, path)
        observations.append(observation)
        actions.append(action)
        episodes.append(episode_id)
        episode_offset = int(episode_id.max()) + 1
        metadata.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "samples": int(observation.shape[0]),
                "role": "anchor_demo",
                "episode_offset": episode_offset,
            }
        )
    observation = np.concatenate(observations)
    action = np.concatenate(actions)
    episode_id = np.concatenate(episodes)
    unique = np.unique(episode_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    selected: list[int] = []
    selected_count = 0
    for value in unique:
        selected.append(int(value))
        selected_count += int(np.count_nonzero(episode_id == value))
        if selected_count >= sample_limit:
            break
    mask = np.isin(episode_id, selected)
    return observation[mask], action[mask], episode_id[mask], metadata


def _label_set(
    observation: np.ndarray,
    teacher_action: np.ndarray,
    policy: BCPolicy,
    residual_scales: np.ndarray,
    episode_id: np.ndarray,
    source: np.ndarray,
    *,
    expected_schema: ObservationSchema,
    teacher_mix: float,
    max_action: float,
) -> ResidualLabelSet:
    prepared = _prepare_observation(observation, expected_schema)
    base_action = _base_actions(policy, prepared)
    mixed_teacher = base_action + teacher_mix * (teacher_action - base_action)
    return build_residual_labels(
        prepared,
        mixed_teacher,
        base_action,
        residual_scales,
        episode_id,
        source,
        max_action=max_action,
        observation_schema=expected_schema.version,
    )


def _subset(labels: ResidualLabelSet, indices: np.ndarray) -> ResidualLabelSet:
    return ResidualLabelSet(
        observation=labels.observation[indices],
        teacher_action=labels.teacher_action[indices],
        base_action=labels.base_action[indices],
        target_normalized=labels.target_normalized[indices],
        episode_id=labels.episode_id[indices],
        source=labels.source[indices],
        observation_schema=labels.observation_schema,
    )


def main() -> int:
    args = _parse_args()
    import torch
    from rsl_rl.modules import ActorCritic
    from torch import nn

    if args.batch_size <= 0 or args.max_epochs <= 0 or args.patience <= 0:
        raise ValueError("batch-size, max-epochs, and patience must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or args.output_l2 < 0.0:
        raise ValueError("optimizer and regularization parameters are invalid")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must lie in (0, 1)")
    if not 0.0 <= args.anchor_fraction < 1.0:
        raise ValueError("anchor-fraction must lie in [0, 1)")
    if not 0.1 <= args.teacher_mix <= 1.0:
        raise ValueError("teacher-mix must lie in [0.1, 1.0]")
    if not 0.0 < args.output_scale <= 1.0:
        raise ValueError("output-scale must lie in (0, 1]")

    training_path = _resolve(args.training_config)
    training = load_training_config(training_path)
    environment_path = _resolve(Path(training.environment_config))
    environment = load_config(environment_path)
    schema = observation_schema_for_dim(environment.observation.dim)
    if training.observation_schema != schema.version:
        raise ValueError(
            "training/environment observation schema mismatch: "
            f"training={training.observation_schema!r}, "
            f"environment dim={environment.observation.dim} implies {schema.version!r}"
        )
    if training.base_mode != "frozen_bc" or training.base_checkpoint is None:
        raise ValueError("residual warm start requires a frozen BC base policy")
    base_path = _resolve(Path(training.base_checkpoint))
    policy = BCPolicy(
        base_path,
        device="cpu",
        expected_observation_dim=schema.dim,
        expected_observation_schema=schema.version,
        expected_speed_feedback=environment.observation.speed_feedback,
        expected_current_contract=environment.observation.current_contract,
    )
    residual_scales = np.array(
        [
            environment.residual.scale_rudder,
            environment.residual.scale_stern,
            environment.residual.scale_rpm,
        ],
        dtype=np.float32,
    )

    teacher_paths = [_resolve(path) for path in args.teacher_rollouts]
    teacher_observation, teacher_action, teacher_episode, teacher_source, sources = (
        _load_teacher_rollouts(teacher_paths, schema)
    )
    anchor_limit = int(
        teacher_observation.shape[0]
        * args.anchor_fraction
        / max(1.0 - args.anchor_fraction, 1e-9)
    )
    anchor_paths = [_resolve(path) for path in args.anchor_demos]
    anchor_observation, anchor_action, anchor_episode, anchor_sources = (
        _load_anchor_demos(
            anchor_paths,
            expected_schema=schema,
            sample_limit=anchor_limit,
            seed=args.seed,
        )
    )
    sources.extend(anchor_sources)
    observation = np.concatenate([teacher_observation, anchor_observation], axis=0)
    teacher = np.concatenate([teacher_action, anchor_action], axis=0)
    episode_id = np.concatenate([teacher_episode, anchor_episode], axis=0)
    source = np.concatenate(
        [
            teacher_source,
            np.full(anchor_observation.shape[0], 1, dtype=np.int8),
        ]
    )
    labels = _label_set(
        observation,
        teacher,
        policy,
        residual_scales,
        episode_id,
        source,
        expected_schema=schema,
        teacher_mix=args.teacher_mix,
        max_action=environment.residual.max_action,
    )
    train_indices, validation_indices = split_by_episode(
        labels.episode_id,
        args.validation_fraction,
        args.seed,
    )
    train_labels = _subset(labels, train_indices)
    validation_labels = _subset(labels, validation_indices)

    output = _resolve(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} already contains files; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ActorCritic(
        labels.observation.shape[1],
        labels.observation.shape[1],
        3,
        **training.policy,
    ).to(device)
    actor = model.actor
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.HuberLoss(delta=1.0)
    train_x = torch.as_tensor(train_labels.observation, device=device)
    train_y = torch.as_tensor(train_labels.target_normalized, device=device)
    validation_x = torch.as_tensor(validation_labels.observation, device=device)
    validation_y = torch.as_tensor(validation_labels.target_normalized, device=device)
    best_state = {
        key: value.detach().cpu().clone() for key, value in actor.state_dict().items()
    }
    best_validation = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(args.max_epochs):
        actor.train()
        permutation = torch.randperm(train_x.shape[0], device=device)
        batch_losses: list[float] = []
        for start in range(0, train_x.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            prediction = actor(train_x[indices])
            loss = loss_function(prediction, train_y[indices])
            loss = loss + args.output_l2 * torch.mean(torch.square(prediction))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        actor.eval()
        with torch.inference_mode():
            validation_prediction = actor(validation_x)
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
                for key, value in actor.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    actor.load_state_dict(best_state)
    final_linear = [module for module in actor.modules() if isinstance(module, nn.Linear)][-1]
    final_linear.weight.data.mul_(args.output_scale)
    final_linear.bias.data.mul_(args.output_scale)
    actor.eval()
    with torch.inference_mode():
        prediction = actor(validation_x).cpu().numpy()
    validation_absolute = np.abs(prediction - validation_labels.target_normalized)
    validation_metrics = {
        "mae_normalized": np.mean(validation_absolute, axis=0).tolist(),
        "p95_absolute_normalized": np.percentile(
            validation_absolute, 95.0, axis=0
        ).tolist(),
        "prediction_abs_mean": np.mean(np.abs(prediction), axis=0).tolist(),
        "prediction_abs_p95": np.percentile(np.abs(prediction), 95.0, axis=0).tolist(),
        "prediction_clip_fraction": np.mean(np.abs(prediction) >= 1.0, axis=0).tolist(),
    }
    checkpoint = {
        "schema_version": ACTOR_WARMSTART_SCHEMA_VERSION,
        "actor_state_dict": actor.state_dict(),
        "observation_dim": int(labels.observation.shape[1]),
        "observation_schema": schema.version,
        "action_dim": 3,
        "policy": dict(training.policy),
        "training_config_path": str(training_path.resolve()),
        "training_config_sha256": training.config_sha256,
        "environment_config_path": str(environment_path.resolve()),
        "environment_config_sha256": environment.config_sha256,
        "base_checkpoint_path": str(base_path.resolve()),
        "base_checkpoint_sha256": _sha256(base_path),
        "residual_scales": residual_scales.tolist(),
        "teacher_mix": args.teacher_mix,
        "output_scale": args.output_scale,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
    }
    checkpoint_path = output / "actor_warmstart.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        **checkpoint,
        "actor_state_dict": None,
        "label_statistics": label_statistics(labels),
        "train_label_statistics": label_statistics(train_labels),
        "validation_label_statistics": label_statistics(validation_labels),
        "train_samples": int(train_labels.observation.shape[0]),
        "validation_samples": int(validation_labels.observation.shape[0]),
        "train_loss": train_losses,
        "validation_loss": validation_losses,
        "validation_metrics": validation_metrics,
        "dataset_sources": sources,
        "anchor_sample_limit": anchor_limit,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "output_l2": args.output_l2,
        "output_scale": args.output_scale,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "device": args.device,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    (output / "training_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
                "train_samples": report["train_samples"],
                "validation_samples": report["validation_samples"],
                "validation_metrics": validation_metrics,
                "checkpoint_path": report["checkpoint_path"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
