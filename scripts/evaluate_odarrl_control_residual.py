from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import subprocess
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from auv_core.contracts import FloatArray

from auv_lowlevel.base_controller import BaseControllerAdapter, sha256_file
from auv_lowlevel.batch.checkpoint import validate_checkpoint_metadata
from auv_lowlevel.batch.encoded_policy import ENCODED_POLICY_KINDS, EncodedActorCritic
from auv_lowlevel.batch.history import HistoryLayout
from auv_lowlevel.batch.observation import observation_schema_for_dim
from auv_lowlevel.config import load_config
from auv_lowlevel.evaluate import Rollout, run_rollout
from auv_lowlevel.ppo_config import load_training_config
from auv_lowlevel.residual import TerminalResidualGovernor
from auv_lowlevel.scripts_support import aggregate_episode_summaries, paired_episode_analysis

REPO_ROOT = Path(__file__).resolve().parents[1]

HISTORY_ABLATION_MODES = (
    "none",
    "current-only",
    "zero",
    "short-only",
    "long-only",
)

CURRENT_ABLATION_MODES = ("none", "zero")


def _ablate_history_windows(
    current: np.ndarray,
    short: np.ndarray,
    long: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace history blocks without mutating the live history buffers.

    ``current-only`` repeats the current frame in both windows and therefore
    removes all temporal information. ``short-only``/``long-only`` retain one
    horizon while replacing the other with the current frame, keeping the input
    shape in-distribution. ``zero`` is a stricter out-of-distribution control.
    """

    if mode not in HISTORY_ABLATION_MODES:
        raise ValueError(f"unsupported history ablation mode: {mode!r}")
    current = np.asarray(current, dtype=np.float32)
    short = np.asarray(short, dtype=np.float32)
    long = np.asarray(long, dtype=np.float32)
    if current.ndim != 1:
        raise ValueError("current observation must be one-dimensional")
    if short.ndim != 2 or long.ndim != 2:
        raise ValueError("history windows must be two-dimensional")
    if short.shape[1] != current.shape[0] or long.shape[1] != current.shape[0]:
        raise ValueError("history window width does not match the current observation")
    if mode == "none":
        return short, long
    current_window_short = np.repeat(current[None, :], short.shape[0], axis=0)
    current_window_long = np.repeat(current[None, :], long.shape[0], axis=0)
    if mode == "current-only":
        return current_window_short, current_window_long
    if mode == "short-only":
        return short, current_window_long
    if mode == "long-only":
        return current_window_short, long
    return np.zeros_like(short), np.zeros_like(long)


def _ablate_current_frame(
    current: np.ndarray,
    short: np.ndarray,
    long: np.ndarray,
    *,
    mode: str,
    reset: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perturb only the policy's view of the current frame.

    The live history buffers are updated before this function is called and are
    deliberately left untouched. The current frame appears both directly and
    as the last row of each history window, so both copies must be hidden.
    ``reset`` hides the whole bootstrap window because every row is then a copy
    of the reset observation. This mirrors training-time environment dropout
    and makes ``zero`` a direct test of history-only action selection.
    """

    if mode not in CURRENT_ABLATION_MODES:
        raise ValueError(f"unsupported current ablation mode: {mode!r}")
    current = np.asarray(current, dtype=np.float32)
    short = np.asarray(short, dtype=np.float32)
    long = np.asarray(long, dtype=np.float32)
    if current.ndim != 1:
        raise ValueError("current observation must be one-dimensional")
    if short.ndim != 2 or long.ndim != 2:
        raise ValueError("history windows must be two-dimensional")
    if short.shape[1] != current.shape[0] or long.shape[1] != current.shape[0]:
        raise ValueError("history window width does not match the current observation")
    if mode == "none":
        return current, short, long
    if len(short) == 0 or len(long) == 0:
        raise ValueError("history windows must not be empty")
    masked_short = short.copy()
    masked_long = long.copy()
    if reset:
        masked_short[...] = 0.0
        masked_long[...] = 0.0
    else:
        masked_short[-1] = 0.0
        masked_long[-1] = 0.0
    return np.zeros_like(current), masked_short, masked_long


class _DeterministicActorPolicy:
    def __init__(
        self,
        checkpoint_path: Path | str,
        training_config_path: Path | str,
        *,
        observation_dim: int,
        device: str = "cpu",
        action_scale: float = 1.0,
        action_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
        actor_warmstart: bool = False,
        history_ablation: str = "none",
        current_ablation: str = "none",
        terminal_residual_radius_m: float | None = None,
        terminal_residual_zero_radius_m: float = 0.0,
    ) -> None:
        import torch
        from rsl_rl.modules import ActorCritic

        training = load_training_config(training_config_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.checkpoint_metadata = _validate_policy_checkpoint(
            checkpoint,
            observation_dim=observation_dim,
            actor_warmstart=actor_warmstart,
        )
        self.history_layout = _history_layout_from_metadata(
            self.checkpoint_metadata,
            observation_dim=observation_dim,
        )
        if history_ablation not in HISTORY_ABLATION_MODES:
            raise ValueError(
                f"unsupported history ablation mode: {history_ablation!r}"
            )
        if history_ablation != "none" and self.history_layout is None:
            raise ValueError("history ablation requires a history checkpoint")
        self.history_ablation = history_ablation
        if current_ablation not in CURRENT_ABLATION_MODES:
            raise ValueError(
                f"unsupported current ablation mode: {current_ablation!r}"
            )
        if current_ablation != "none" and self.history_layout is None:
            raise ValueError("current ablation requires a history checkpoint")
        self.current_ablation = current_ablation
        policy_kind = (
            "mlp"
            if self.checkpoint_metadata is None
            else str(self.checkpoint_metadata.get("policy_kind", "mlp"))
        )
        if actor_warmstart:
            policy_kind = "actor_warmstart"
        if policy_kind in ENCODED_POLICY_KINDS:
            if self.history_layout is None:
                raise ValueError("encoded policy checkpoint is missing history windows")
            encoder_config = (self.checkpoint_metadata or {}).get("policy_config")
            if not isinstance(encoder_config, Mapping):
                raise ValueError("encoded policy checkpoint is missing policy_config")
            if training.encoder is None:
                raise ValueError("training config has no encoder for an encoded checkpoint")
            if training.encoder.as_dict() != dict(encoder_config):
                raise ValueError("training encoder config does not match checkpoint policy_config")
            if training.policy_kind != policy_kind:
                raise ValueError("training policy kind does not match checkpoint policy_kind")
        elif training.encoder is not None:
            raise ValueError("training config requests an encoder but checkpoint is not encoded")
        policy_input_dim = (
            self.history_layout.policy_input_dim
            if self.history_layout is not None
            else int(
                (self.checkpoint_metadata or {}).get("policy_input_dim", observation_dim)
            )
        )
        self.policy_input_dim = policy_input_dim
        if policy_kind in ENCODED_POLICY_KINDS:
            assert self.history_layout is not None
            critic_mode = str(
                (self.checkpoint_metadata or {}).get(
                    "critic_mode", "shared_actor_features"
                )
            )
            critic_observation_dim = int(
                (self.checkpoint_metadata or {}).get(
                    "critic_observation_dim", policy_input_dim
                )
            )
            policy_kwargs = dict(training.policy)
            policy_kwargs["independent_critic"] = (
                critic_mode == "privileged_current_independent"
            )
            self.model = EncodedActorCritic(
                policy_input_dim,
                critic_observation_dim,
                3,
                encoder=dict((self.checkpoint_metadata or {})["policy_config"]),
                history_short_window=self.history_layout.short_window,
                history_long_window=self.history_layout.long_window,
                **policy_kwargs,
            ).to(device)
        else:
            critic_mode = str(
                (self.checkpoint_metadata or {}).get(
                    "critic_mode", "shared_actor_features"
                )
            )
            critic_observation_dim = int(
                (self.checkpoint_metadata or {}).get(
                    "critic_observation_dim", policy_input_dim
                )
            )
            if (
                critic_mode == "privileged_current_independent"
                and critic_observation_dim != policy_input_dim
                and training.history is None
            ):
                raise ValueError(
                    "privileged independent critic requires history observations"
                )
            self.model = ActorCritic(
                policy_input_dim,
                critic_observation_dim,
                3,
                **training.policy,
            ).to(device)
        if actor_warmstart:
            self.model.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.device = torch.device(device)
        self.action_scale = float(action_scale)
        self.action_gains = np.asarray(action_gains, dtype=np.float64)
        self.terminal_residual_governor = (
            None
            if terminal_residual_radius_m is None
            else TerminalResidualGovernor(
                zero_radius_m=terminal_residual_zero_radius_m,
                full_scale_radius_m=terminal_residual_radius_m,
            )
        )
        self._environment: Any = None
        self._history_short: np.ndarray | None = None
        self._history_long: np.ndarray | None = None
        self._fresh_history = True

    def bind_environment(self, environment: Any) -> None:
        """Bind the rollout environment for terminal residual gating."""

        self._environment = environment

    def reset(self) -> None:
        self._history_short = None
        self._history_long = None
        self._fresh_history = True

    def _augment_observation(self, observation: FloatArray) -> np.ndarray:
        if self.history_layout is None:
            return np.asarray(observation, dtype=np.float32)
        current = np.asarray(observation, dtype=np.float32)
        if current.shape != (self.history_layout.observation_dim,):
            raise ValueError("history policy received the wrong base observation shape")
        if self._history_short is None or self._history_long is None:
            self._history_short = np.repeat(
                current[None, :], self.history_layout.short_span_steps, axis=0
            )
            self._history_long = np.repeat(
                current[None, :], self.history_layout.long_span_steps, axis=0
            )
        else:
            self._history_short = np.concatenate(
                [self._history_short[1:], current[None, :]], axis=0
            )
            self._history_long = np.concatenate(
                [self._history_long[1:], current[None, :]], axis=0
            )
        short = self._sample_history(
            self._history_short,
            size=self.history_layout.short_window,
            stride=self.history_layout.short_stride,
        )
        long = self._sample_history(
            self._history_long,
            size=self.history_layout.long_window,
            stride=self.history_layout.long_stride,
        )
        short, long = _ablate_history_windows(
            current,
            short,
            long,
            mode=self.history_ablation,
        )
        policy_current, policy_short, policy_long = _ablate_current_frame(
            current,
            short,
            long,
            mode=self.current_ablation,
            reset=getattr(self, "_fresh_history", False),
        )
        self._fresh_history = False
        return np.concatenate(
            [policy_current, policy_short.reshape(-1), policy_long.reshape(-1)]
        )

    @staticmethod
    def _sample_history(window: np.ndarray, *, size: int, stride: int) -> np.ndarray:
        if stride == 1:
            return window
        sampled = window[stride - 1 :: stride]
        if sampled.shape[0] != size:
            raise ValueError("history stride produced an inconsistent window size")
        return sampled

    def __call__(self, observation: FloatArray, step: int) -> FloatArray:
        del step
        import torch

        with torch.inference_mode():
            augmented = self._augment_observation(observation)
            tensor = torch.as_tensor(
                augmented,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action = self.model.act_inference(tensor).squeeze(0).cpu().numpy()
        residual = (
            np.asarray(action, dtype=np.float64) * self.action_scale * self.action_gains
        )
        if self.terminal_residual_governor is not None:
            if self._environment is None:
                raise RuntimeError(
                    "terminal residual governor requires bind_environment()"
                )
            position = self._environment.last_context.state.position_ned_m
            goal = self._environment.path.goal_ned_m
            distance = float(np.linalg.norm(position - goal))
            residual = residual * self.terminal_residual_governor.scale(distance)
        return residual


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained residual PPO checkpoint")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--actor-warmstart", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--goal-radius", type=float, default=None)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--history-ablation",
        choices=HISTORY_ABLATION_MODES,
        default="none",
        help=(
            "evaluation-only history perturbation: current-only replaces both "
            "windows with the current frame; short-only/long-only retain one "
            "horizon; zero removes both windows"
        ),
    )
    parser.add_argument(
        "--current-ablation",
        choices=CURRENT_ABLATION_MODES,
        default="none",
        help=(
            "evaluation-only current-frame perturbation: zero removes the "
            "current 26D frame while preserving the genuine short/long history"
        ),
    )
    parser.add_argument(
        "--terminal-residual-radius",
        type=float,
        default=None,
        help=(
            "linearly/smoothstep suppress residual authority from this distance "
            "to the configured goal radius"
        ),
    )
    parser.add_argument(
        "--terminal-residual-zero-radius",
        type=float,
        default=None,
        help=(
            "goal distance at which the residual is fully disabled; defaults to "
            "the configured goal radius and requires --terminal-residual-radius"
        ),
    )
    parser.add_argument(
        "--residual-gains",
        default="1,1,1",
        help="Comma-separated per-actuator gains for rudder, stern plane, and rpm",
    )
    return parser.parse_args()


def _resolve(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _parse_seeds(text: str | None, start: int, count: int) -> tuple[int, ...]:
    if text is None:
        return tuple(range(start, start + count))
    seeds: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if "-" in token:
            first, last = token.split("-", maxsplit=1)
            seeds.extend(range(int(first), int(last) + 1))
        elif token:
            seeds.append(int(token))
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be non-empty, unique, and non-negative")
    return tuple(seeds)


def _parse_residual_gains(text: str) -> tuple[float, float, float]:
    tokens = [token.strip() for token in text.split(",")]
    if len(tokens) != 3:
        raise ValueError("--residual-gains must contain exactly three comma-separated values")
    try:
        gains = tuple(float(token) for token in tokens)
    except ValueError as error:
        raise ValueError("--residual-gains values must be numeric") from error
    if not all(np.isfinite(gain) and gain >= 0.0 for gain in gains):
        raise ValueError("--residual-gains values must be finite and non-negative")
    return gains  # type: ignore[return-value]


def _validate_training_environment_schema(training: Any, environment: Any) -> Any:
    schema = observation_schema_for_dim(environment.observation.dim)
    if training.observation_schema != schema.version:
        raise ValueError(
            "training/environment observation schema mismatch: "
            f"training={training.observation_schema!r}, "
            f"environment dim={environment.observation.dim} implies {schema.version!r}"
        )
    return schema


def _validate_policy_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    observation_dim: int,
    actor_warmstart: bool,
) -> dict[str, Any] | None:
    schema = observation_schema_for_dim(int(observation_dim))
    if actor_warmstart:
        if checkpoint.get("schema_version") != "auv_odarrl_actor_warmstart.v1":
            raise ValueError("unsupported actor warm-start checkpoint schema")
        checkpoint_dim = checkpoint.get("observation_dim")
        if checkpoint_dim != observation_dim:
            raise ValueError(
                "actor warm-start observation dimension mismatch: "
                f"expected {observation_dim}, got {checkpoint_dim!r}"
            )
        checkpoint_schema = checkpoint.get("observation_schema")
        if checkpoint_schema != schema.version:
            raise ValueError(
                "actor warm-start observation schema mismatch: "
                f"expected {schema.version!r}, got {checkpoint_schema!r}"
            )
        return {
            "kind": "actor_warmstart",
            "schema_version": checkpoint["schema_version"],
            "observation_schema": schema.version,
            "observation_dim": observation_dim,
        }
    if "model_state_dict" not in checkpoint:
        raise ValueError("rsl_rl checkpoint is missing model_state_dict")
    metadata = validate_checkpoint_metadata(
        checkpoint,
        expected_observation_schema=schema.version,
        expected_observation_dim=schema.dim,
        required=observation_dim == 26,
    )
    if metadata is not None:
        _history_layout_from_metadata(metadata, observation_dim=observation_dim)
        policy_kind = metadata.get("policy_kind", "mlp")
        if policy_kind not in {"mlp", "history_mlp", *ENCODED_POLICY_KINDS}:
            raise ValueError(f"unsupported checkpoint policy_kind: {policy_kind!r}")
        policy_config = metadata.get("policy_config")
        if policy_kind in ENCODED_POLICY_KINDS:
            if not isinstance(policy_config, Mapping):
                raise ValueError("encoded checkpoint must contain policy_config")
            if "history_short_window" not in metadata:
                raise ValueError("encoded checkpoint must contain history windows")
        elif policy_config is not None:
            raise ValueError("policy_config is only valid for encoded checkpoints")
    return metadata


def _history_layout_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    observation_dim: int,
) -> HistoryLayout | None:
    if metadata is None:
        return None
    short_window = metadata.get("history_short_window")
    long_window = metadata.get("history_long_window")
    if short_window is None and long_window is None:
        if "policy_input_dim" in metadata and int(metadata["policy_input_dim"]) != observation_dim:
            raise ValueError("checkpoint has policy_input_dim without history metadata")
        return None
    if short_window is None or long_window is None:
        raise ValueError("checkpoint history windows must be provided together")
    layout = HistoryLayout(
        observation_dim=observation_dim,
        short_window=int(short_window),
        long_window=int(long_window),
        short_stride=int(metadata.get("history_short_stride", 1)),
        long_stride=int(metadata.get("history_long_stride", 1)),
    )
    policy_input_dim = int(metadata.get("policy_input_dim", layout.policy_input_dim))
    if policy_input_dim != layout.policy_input_dim:
        raise ValueError("checkpoint policy_input_dim is inconsistent with history windows")
    return layout


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _arrays(rollout: Rollout) -> dict[str, np.ndarray]:
    return {
        "t_s": rollout.t_s,
        "position_ned_m": rollout.position_ned_m,
        "reference_point_ned_m": rollout.reference_point_ned_m,
        "horizontal_error_m": rollout.horizontal_error_m,
        "vertical_error_m": rollout.vertical_error_m,
        "error_3d_m": rollout.error_3d_m,
        "progress_m": rollout.progress_m,
        "yaw_error_rad": rollout.yaw_error_rad,
        "water_surge_mps": rollout.water_surge_mps,
        "current_ned_mps": rollout.current_ned_mps,
        "reference_speed_mps": rollout.reference_speed_mps,
        "observation": rollout.observation,
        "base_command": rollout.base_command,
        "residual_normalized": rollout.residual_normalized,
        "delta_command": rollout.delta_command,
        "requested_total": rollout.requested_total,
        "limited_total": rollout.limited_total,
        "actual_actuator": rollout.actual_actuator,
        "reward_total": rollout.reward_total,
        "saturated": rollout.saturated,
        "material_saturated": rollout.material_saturated,
        "residual_clipped": rollout.residual_clipped,
        "constraint_violation_count": rollout.constraint_violation_count,
    }


def _run_seed(
    payload: tuple[
        str,
        str,
        str,
        bool,
        int,
        int | None,
        float | None,
        float,
        tuple[float, float, float],
        float | None,
        float | None,
        str,
        str,
        str,
    ],
) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        training_config_path,
        environment_config_path,
        residual_checkpoint,
        actor_warmstart,
        seed,
        max_steps,
        goal_radius,
        residual_scale,
        residual_gains,
        terminal_residual_radius,
        terminal_residual_zero_radius,
        history_ablation,
        current_ablation,
        output_text,
    ) = payload
    training = load_training_config(training_config_path)
    config = load_config(environment_config_path)
    _validate_training_environment_schema(training, config)
    if max_steps is not None or goal_radius is not None:
        config = replace(
            config,
            plant=replace(
                config.plant,
                max_episode_steps=config.plant.max_episode_steps
                if max_steps is None
                else max_steps,
                goal_radius_m=config.plant.goal_radius_m if goal_radius is None else goal_radius,
            ),
        )
    if terminal_residual_radius is not None:
        zero_radius = (
            config.plant.goal_radius_m
            if terminal_residual_zero_radius is None
            else terminal_residual_zero_radius
        )
        if (
            not np.isfinite(terminal_residual_radius)
            or not np.isfinite(zero_radius)
            or zero_radius < 0.0
            or terminal_residual_radius <= zero_radius
        ):
            raise ValueError(
                "--terminal-residual-radius must be finite and exceed the "
                "non-negative terminal residual zero radius"
            )
    elif terminal_residual_zero_radius is not None:
        raise ValueError(
            "--terminal-residual-zero-radius requires --terminal-residual-radius"
        )
    else:
        zero_radius = config.plant.goal_radius_m
    output = Path(output_text)
    output.mkdir(parents=True, exist_ok=True)

    if training.base_mode == "frozen_bc":
        checkpoint = _resolve(training.base_checkpoint or "")
        base_adapter = BaseControllerAdapter(
            config,
            "frozen_bc",
            checkpoint=checkpoint,
            checkpoint_sha256=training.base_checkpoint_sha256,
            device="cpu",
        )
        base = run_rollout(config, seed, **base_adapter.env_kwargs())
    else:
        base_adapter = BaseControllerAdapter(config, "online_pd_pi")
        base = run_rollout(config, seed, **base_adapter.env_kwargs())

    residual_policy = _DeterministicActorPolicy(
        residual_checkpoint,
        training_config_path,
        observation_dim=config.observation.dim,
        device="cpu",
        action_scale=residual_scale,
        action_gains=residual_gains,
        actor_warmstart=actor_warmstart,
        history_ablation=history_ablation,
        current_ablation=current_ablation,
        terminal_residual_radius_m=terminal_residual_radius,
        terminal_residual_zero_radius_m=zero_radius,
    )
    residual = run_rollout(
        config,
        seed,
        residual_policy=residual_policy,
        **base_adapter.env_kwargs(),
    )
    base_dir = output / "per_seed" / "base"
    residual_dir = output / "per_seed" / "residual"
    base_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base_dir / f"seed_{seed:05d}.npz", **_arrays(base))
    np.savez_compressed(residual_dir / f"seed_{seed:05d}.npz", **_arrays(residual))
    pair = {
        "seed": seed,
        "base": base.summary(),
        "residual": residual.summary(),
    }
    (output / "per_seed" / f"seed_{seed:05d}.json").write_text(
        json.dumps(pair, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return pair["base"], pair["residual"]


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    base = report["aggregate"]["base"]
    residual = report["aggregate"]["residual"]
    paired = report["paired_analysis"]
    lines = [
        "# Residual PPO evaluation",
        "",
        f"- training_config_sha256: `{report['training_config_sha256']}`",
        f"- environment_config_sha256: `{report['environment_config_sha256']}`",
        f"- residual_checkpoint: `{report['residual_checkpoint']}`",
        f"- observation_schema: `{report['observation_schema']}`",
        f"- observation_dim: `{report['observation_dim']}`",
        f"- residual_checkpoint_metadata: `{report['residual_checkpoint_metadata']}`",
        f"- seeds: `{report['seeds']}`",
        f"- history_ablation: `{report['history_ablation']}`",
        "",
        "| metric | base | residual | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "success_rate",
        "three_d_rmse_m_mean",
        "horizontal_rmse_m_mean",
        "vertical_rmse_m_mean",
        "minimum_goal_distance_m_mean",
        "material_saturation_rate_mean",
        "residual_clip_rate_mean",
        "residual_abs_mean_mean",
        "constraint_violations_mean",
    ):
        lines.append(
            f"| `{key}` | {base[key]:.6g} | {residual[key]:.6g} | {residual[key] - base[key]:.6g} |"
        )
    common = paired["common_success"]
    lines.extend(
        [
            "",
            "## Paired success subset",
            "",
            f"- success flips: `{paired['success_flips']}`",
            f"- common-success seeds: `{common['seed_count']}`",
            f"- base 3D RMSE: `{common['base_three_d_rmse_m_mean']}`",
            f"- residual 3D RMSE: `{common['residual_three_d_rmse_m_mean']}`",
            f"- improvement: `{common['three_d_rmse_improvement_m']}`",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    training_config_path = _resolve(args.training_config)
    training = load_training_config(training_config_path)
    environment_path = _resolve(args.environment_config or training.environment_config)
    environment = load_config(environment_path)
    schema = _validate_training_environment_schema(training, environment)
    seeds = _parse_seeds(
        args.seeds,
        environment.evaluation.seed_start,
        environment.evaluation.seed_count,
    )
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if (args.checkpoint is None) == (args.actor_warmstart is None):
        raise ValueError("exactly one of --checkpoint and --actor-warmstart is required")
    if not np.isfinite(args.residual_scale) or args.residual_scale <= 0.0:
        raise ValueError("--residual-scale must be finite and positive")
    if args.terminal_residual_radius is not None and (
        not np.isfinite(args.terminal_residual_radius)
        or args.terminal_residual_radius <= 0.0
    ):
        raise ValueError("--terminal-residual-radius must be finite and positive")
    if args.terminal_residual_zero_radius is not None:
        if args.terminal_residual_radius is None:
            raise ValueError(
                "--terminal-residual-zero-radius requires --terminal-residual-radius"
            )
        if (
            not np.isfinite(args.terminal_residual_zero_radius)
            or args.terminal_residual_zero_radius < 0.0
        ):
            raise ValueError(
                "--terminal-residual-zero-radius must be finite and non-negative"
            )
    residual_gains = _parse_residual_gains(args.residual_gains)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    actor_warmstart = args.actor_warmstart is not None
    checkpoint = _resolve(args.checkpoint or args.actor_warmstart)
    checkpoint_hash = sha256_file(checkpoint)
    import torch

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_metadata = _validate_policy_checkpoint(
        checkpoint_payload,
        observation_dim=environment.observation.dim,
        actor_warmstart=actor_warmstart,
    )
    payloads = [
        (
            str(training_config_path),
            str(environment_path),
            str(checkpoint),
            actor_warmstart,
            seed,
            args.max_steps,
            args.goal_radius,
            args.residual_scale,
            residual_gains,
            args.terminal_residual_radius,
            args.terminal_residual_zero_radius,
            args.history_ablation,
            args.current_ablation,
            str(output),
        )
        for seed in seeds
    ]
    base_episodes: list[dict[str, Any]] = []
    residual_episodes: list[dict[str, Any]] = []
    if args.workers == 1:
        for payload in payloads:
            base, residual = _run_seed(payload)
            base_episodes.append(base)
            residual_episodes.append(residual)
    else:
        results: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_seed, payload): payload[4] for payload in payloads}
            for future in as_completed(futures):
                seed = futures[future]
                results[seed] = future.result()
                print(f"finished seed {seed}", flush=True)
        for seed in seeds:
            base_episodes.append(results[seed][0])
            residual_episodes.append(results[seed][1])

    base_aggregate = aggregate_episode_summaries(base_episodes)
    residual_aggregate = aggregate_episode_summaries(residual_episodes)
    paired_analysis = paired_episode_analysis(base_episodes, residual_episodes)
    report = {
        "schema_version": "auv_odarrl_residual_eval.v1",
        "training_config_path": str(training_config_path),
        "training_config_sha256": training.config_sha256,
        "environment_config_path": str(environment_path),
        "environment_config_sha256": environment.config_sha256,
        "environment_config_override": args.environment_config is not None,
        "residual_checkpoint": str(checkpoint),
        "residual_checkpoint_sha256": checkpoint_hash,
        "residual_checkpoint_kind": "actor_warmstart" if actor_warmstart else "rsl_rl",
        "residual_checkpoint_metadata": checkpoint_metadata,
        "observation_schema": schema.version,
        "observation_dim": schema.dim,
        "policy_input_dim": int(
            (checkpoint_metadata or {}).get("policy_input_dim", schema.dim)
        ),
        "history": (
            None
            if checkpoint_metadata is None
            or "history_short_window" not in checkpoint_metadata
            else {
                "short_window": int(checkpoint_metadata["history_short_window"]),
                "long_window": int(checkpoint_metadata["history_long_window"]),
                "short_stride": int(checkpoint_metadata.get("history_short_stride", 1)),
                "long_stride": int(checkpoint_metadata.get("history_long_stride", 1)),
            }
        ),
        "git_commit": _git_commit(),
        "seeds": list(seeds),
        "max_steps_override": args.max_steps,
        "goal_radius_override_m": args.goal_radius,
        "residual_scale": args.residual_scale,
        "residual_gains": list(residual_gains),
        "terminal_residual_radius_m": args.terminal_residual_radius,
        "terminal_residual_zero_radius_m": args.terminal_residual_zero_radius,
        "history_ablation": args.history_ablation,
        "current_ablation": args.current_ablation,
        "aggregate": {"base": base_aggregate, "residual": residual_aggregate},
        "paired_analysis": paired_analysis,
        "episodes": [
            {"seed": seed, "base": base, "residual": residual}
            for seed, base, residual in zip(seeds, base_episodes, residual_episodes, strict=True)
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(report, output)
    print(
        json.dumps(
            {
                "three_d_rmse_gap_m": residual_aggregate["three_d_rmse_m_mean"]
                - base_aggregate["three_d_rmse_m_mean"],
                "success_rate_delta": residual_aggregate["success_rate"]
                - base_aggregate["success_rate"],
                "material_saturation_rate": residual_aggregate["material_saturation_rate_mean"],
                "residual_abs_mean": residual_aggregate["residual_abs_mean_mean"],
                "history_ablation": args.history_ablation,
                "current_ablation": args.current_ablation,
                "common_success_seed_count": paired_analysis["common_success"]["seed_count"],
                "common_success_three_d_rmse_improvement_m": paired_analysis["common_success"][
                    "three_d_rmse_improvement_m"
                ],
                "success_flips": paired_analysis["success_flips"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
