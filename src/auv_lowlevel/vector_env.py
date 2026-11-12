from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

import numpy as np
import torch
from auv_core.contracts import FloatArray

from .base_controller import BaseControllerAdapter, BaseMode
from .config import LowLevelConfig, load_config
from .env import LowLevelEnv


@dataclass(frozen=True)
class WorkerSpec:
    config_path: str
    base_mode: BaseMode
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    device: str
    max_episode_steps: int | None


@dataclass
class _WorkerSlot:
    index: int
    process: mp.Process
    connection: Connection
    restarts: int = 0


class _EpisodeTracker:
    def __init__(self, seed: int, limits: Any) -> None:
        self.seed = seed
        self.limits = limits
        self.steps = 0
        self.return_sum = 0.0
        self.error_3d_sq_sum = 0.0
        self.horizontal_sq_sum = 0.0
        self.vertical_sq_sum = 0.0
        self.residual_abs_sum = 0.0
        self.material_saturation_count = 0
        self.residual_clip_count = 0
        self.base_at_limit_count = 0
        self.constraint_violations = 0
        self.minimum_goal_distance_m = float("inf")
        self.final_goal_distance_m = 0.0

    def update(self, result: Any, goal_ned_m: FloatArray) -> None:
        info = result.info
        projection = info["projection_after"]
        position = info["state_after"].position_ned_m
        self.steps += 1
        self.return_sum += float(result.reward.total)
        self.error_3d_sq_sum += float(projection.error_3d_m) ** 2
        self.horizontal_sq_sum += float(projection.horizontal_error_m) ** 2
        self.vertical_sq_sum += float(projection.vertical_error_m) ** 2
        residual = np.asarray(info["residual_normalized"], dtype=np.float64)
        self.residual_abs_sum += float(np.mean(np.abs(residual)))
        self.material_saturation_count += int(bool(info["material_saturated"]))
        self.residual_clip_count += int(bool(info["residual_clipped"]))
        self.constraint_violations += int(info["constraint_violation_count"])
        base = np.asarray(info["base_command"], dtype=np.float64)
        at_limit = bool(
            abs(base[0]) >= self.limits.max_rudder_angle_rad - 1e-6
            or abs(base[1]) >= self.limits.max_stern_plane_angle_rad - 1e-6
            or base[2] <= 1e-3
            or base[2] >= self.limits.max_propeller_rpm - 1e-3
        )
        self.base_at_limit_count += int(at_limit)
        goal_distance = float(np.linalg.norm(position - goal_ned_m))
        self.minimum_goal_distance_m = min(self.minimum_goal_distance_m, goal_distance)
        self.final_goal_distance_m = goal_distance

    def log(self, terminal_kind: str | None, duration_s: float) -> dict[str, float]:
        count = max(self.steps, 1)
        success = terminal_kind == "goal"
        return {
            "/episode/success_rate": float(success),
            "/episode/return_sum": self.return_sum,
            "/episode/steps": float(self.steps),
            "/episode/duration_s": float(duration_s),
            "/episode/three_d_rmse_m": float(np.sqrt(self.error_3d_sq_sum / count)),
            "/episode/horizontal_rmse_m": float(
                np.sqrt(self.horizontal_sq_sum / count)
            ),
            "/episode/vertical_rmse_m": float(np.sqrt(self.vertical_sq_sum / count)),
            "/episode/residual_abs_mean": float(self.residual_abs_sum / count),
            "/episode/material_saturation_rate": float(
                self.material_saturation_count / count
            ),
            "/episode/residual_clip_rate": float(self.residual_clip_count / count),
            "/episode/base_at_limit_rate": float(self.base_at_limit_count / count),
            "/episode/constraint_violations": float(self.constraint_violations),
            "/episode/final_goal_distance_m": self.final_goal_distance_m,
            "/episode/minimum_goal_distance_m": self.minimum_goal_distance_m,
            "/episode/arrival_time_s": float(duration_s) if success else -1.0,
        }


def _worker_main(
    connection: Connection,
    slot_index: int,
    spec: WorkerSpec,
    initial_seed: int,
) -> None:
    torch.set_num_threads(1)
    try:
        config = load_config(spec.config_path)
        if spec.max_episode_steps is not None:
            from dataclasses import replace

            config = replace(
                config,
                plant=replace(
                    config.plant,
                    max_episode_steps=spec.max_episode_steps,
                ),
            )
        adapter = BaseControllerAdapter(
            config,
            spec.base_mode,
            checkpoint=spec.checkpoint_path,
            checkpoint_sha256=spec.checkpoint_sha256,
            device=spec.device,
        )
        env = LowLevelEnv(
            config,
            **adapter.env_kwargs(),
            expose_current=config.observation.expose_current,
        )
        connection.send({"type": "ready", "slot": slot_index})
        tracker: _EpisodeTracker | None = None
        while True:
            message = connection.recv()
            command = message.get("type")
            if command == "close":
                return
            if command == "reset":
                seed = int(message["seed"])
                observation = env.reset(seed)
                tracker = _EpisodeTracker(seed, env.limits)
                connection.send(
                    {
                        "type": "reset",
                        "observation": observation,
                        "seed": seed,
                    }
                )
                continue
            if command != "step":
                raise ValueError(f"unknown worker command: {command}")
            if tracker is None:
                raise RuntimeError("worker received step before reset")
            action = np.asarray(message["action"], dtype=np.float64)
            result = env.step(action)
            tracker.update(result, env.path.goal_ned_m)
            done = result.terminated or result.truncated
            terminal_kind = str(result.info["terminal_kind"])
            log = tracker.log(terminal_kind, result.info["state_after"].t_s) if done else None
            connection.send(
                {
                    "type": "step",
                    "observation": result.observation,
                    "reward": float(result.reward.ppo_scaled),
                    "terminated": bool(result.terminated),
                    "truncated": bool(result.truncated),
                    "log": log,
                }
            )
    except EOFError:
        return
    except BaseException:
        try:
            connection.send(
                {
                    "type": "error",
                    "slot": slot_index,
                    "message": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class PvsVecEnv:
    """Synchronous rsl_rl-compatible VecEnv backed by persistent PVS workers."""

    def __init__(
        self,
        config: LowLevelConfig,
        config_path: Path | str,
        *,
        num_envs: int,
        base_mode: BaseMode,
        checkpoint_path: Path | str | None = None,
        checkpoint_sha256: str | None = None,
        max_episode_steps: int | None = None,
        reset_timeout_s: float = 120.0,
        step_timeout_s: float = 10.0,
        max_restarts: int = 3,
        context: str = "spawn",
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if reset_timeout_s <= 0.0 or step_timeout_s <= 0.0:
            raise ValueError("worker timeouts must be positive")
        if max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        self.config = config
        self.config_path = str(Path(config_path).resolve())
        self.num_envs = int(num_envs)
        self.num_actions = 3
        self.num_obs = config.observation.dim
        self.num_privileged_obs = None
        self.device = torch.device("cpu")
        self.max_episode_length = int(
            config.plant.max_episode_steps
            if max_episode_steps is None
            else max_episode_steps
        )
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.cfg = {
            "config_path": self.config_path,
            "config_sha256": config.config_sha256,
            "num_envs": self.num_envs,
            "base_mode": base_mode,
        }
        self.unwrapped = self
        self.step_dt = config.plant.dt_s
        self._reset_timeout_s = float(reset_timeout_s)
        self._step_timeout_s = float(step_timeout_s)
        self._max_restarts = int(max_restarts)
        self._context: Any = mp.get_context(context)
        self._spec = WorkerSpec(
            config_path=self.config_path,
            base_mode=base_mode,
            checkpoint_path=(
                None if checkpoint_path is None else str(Path(checkpoint_path).resolve())
            ),
            checkpoint_sha256=checkpoint_sha256,
            device="cpu",
            max_episode_steps=max_episode_steps,
        )
        self._evaluation_seeds = set(
            range(config.evaluation.seed_start, config.evaluation.seed_start + config.evaluation.seed_count)
        )
        self._seed_cursor = config.evaluation.training_seed_start
        self._slots: list[_WorkerSlot] = []
        self._step_count = 0
        self._step_elapsed_s = 0.0
        observations: list[Any] = []
        try:
            for index in range(self.num_envs):
                slot, observation = self._start_slot(index)
                self._slots.append(slot)
                observations.append(observation)
        except BaseException:
            self.close()
            raise
        self._observations = np.stack(observations).astype(np.float32, copy=False)

    @property
    def worker_pids(self) -> tuple[int | None, ...]:
        return tuple(slot.process.pid for slot in self._slots)

    @property
    def worker_restart_counts(self) -> tuple[int, ...]:
        return tuple(slot.restarts for slot in self._slots)

    @property
    def total_env_steps(self) -> int:
        return self._step_count

    @property
    def env_step_rate_hz(self) -> float:
        if self._step_elapsed_s <= 0.0:
            return 0.0
        return float(self._step_count / self._step_elapsed_s)

    def _next_seed(self) -> int:
        while True:
            seed = self._seed_cursor
            self._seed_cursor += 1
            if seed not in self._evaluation_seeds:
                return seed

    def _start_slot(self, index: int) -> tuple[_WorkerSlot, Any]:
        parent, child = self._context.Pipe(duplex=True)
        seed = self._next_seed()
        process = self._context.Process(
            target=_worker_main,
            args=(child, index, self._spec, seed),
            name=f"pvs-worker-{index}",
            daemon=True,
        )
        process.start()
        child.close()
        slot = _WorkerSlot(index=index, process=process, connection=parent)
        try:
            ready = self._recv(slot, self._reset_timeout_s)
            if ready.get("type") != "ready":
                raise RuntimeError(f"worker {index} did not become ready: {ready}")
            observation = self._reset_slot(slot, seed)
        except BaseException:
            self._stop_slot(slot)
            raise
        return slot, observation

    def _restart_slot(self, slot: _WorkerSlot) -> Any:
        slot.restarts += 1
        if slot.restarts > self._max_restarts:
            raise RuntimeError(
                f"PVS worker {slot.index} exceeded {self._max_restarts} restarts"
            )
        self._stop_slot(slot)
        seed = self._next_seed()
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child, slot.index, self._spec, seed),
            name=f"pvs-worker-{slot.index}",
            daemon=True,
        )
        process.start()
        child.close()
        slot.process = process
        slot.connection = parent
        ready = self._recv(slot, self._reset_timeout_s)
        if ready.get("type") != "ready":
            raise RuntimeError(
                f"restarted worker {slot.index} did not become ready: {ready}"
            )
        return self._reset_slot(slot, seed)

    def _reset_slot(self, slot: _WorkerSlot, seed: int) -> Any:
        slot.connection.send({"type": "reset", "seed": int(seed)})
        payload = self._recv(slot, self._reset_timeout_s)
        if payload.get("type") != "reset":
            raise RuntimeError(
                f"worker {slot.index} reset failed: {payload.get('message', payload)}"
            )
        return np.asarray(payload["observation"], dtype=np.float32)

    def _recv(self, slot: _WorkerSlot, timeout_s: float) -> dict[str, Any]:
        deadline = monotonic() + timeout_s
        if not slot.connection.poll(max(0.0, deadline - monotonic())):
            raise TimeoutError(f"PVS worker {slot.index} timed out after {timeout_s:.3f}s")
        payload = slot.connection.recv()
        if not isinstance(payload, dict):
            raise RuntimeError(f"PVS worker {slot.index} sent an invalid payload")
        if payload.get("type") == "error":
            raise RuntimeError(
                f"PVS worker {slot.index} failed:\n{payload.get('message', '')}"
            )
        return payload

    def _stop_slot(self, slot: _WorkerSlot) -> None:
        try:
            if slot.connection.poll():
                slot.connection.recv()
        except (EOFError, OSError):
            pass
        try:
            slot.connection.send({"type": "close"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        try:
            slot.connection.close()
        except OSError:
            pass
        slot.process.join(timeout=2.0)
        if slot.process.is_alive():
            slot.process.terminate()
            slot.process.join(timeout=2.0)
        if slot.process.is_alive():
            slot.process.kill()
            slot.process.join(timeout=2.0)

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._torch_observation(), {"observations": {}}

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        observations = np.stack(
            [self._reset_slot(slot, self._next_seed()) for slot in self._slots]
        ).astype(np.float32, copy=False)
        self._observations = observations
        self.episode_length_buf.zero_()
        return self.get_observations()

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        step_started = perf_counter()
        action_array = np.asarray(actions.detach().cpu(), dtype=np.float64)
        expected_shape = (self.num_envs, self.num_actions)
        if action_array.shape != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("actions contain NaN or Inf")
        send_errors: dict[int, BaseException] = {}
        for index, (slot, action) in enumerate(zip(self._slots, action_array, strict=True)):
            try:
                slot.connection.send({"type": "step", "action": action})
            except (BrokenPipeError, EOFError, OSError) as error:
                send_errors[index] = error

        observations: list[Any] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=np.float32)
        truncated = np.zeros(self.num_envs, dtype=np.float32)
        logs: list[dict[str, float]] = []
        restarted = np.zeros(self.num_envs, dtype=np.float32)
        for index, slot in enumerate(self._slots):
            recovered = False
            try:
                if index in send_errors:
                    raise send_errors[index]
                payload = self._recv(slot, self._step_timeout_s)
                if payload.get("type") != "step":
                    raise RuntimeError(
                        f"worker {index} returned unexpected payload: {payload}"
                    )
                observation = np.asarray(payload["observation"], dtype=np.float32)
                reward = float(payload["reward"])
                is_terminated = bool(payload["terminated"])
                is_truncated = bool(payload["truncated"])
                log = payload.get("log")
            except (TimeoutError, EOFError, OSError, RuntimeError) as error:
                print(f"recycling PVS worker {index}: {error}", flush=True)
                observation = self._restart_slot(slot)
                reward = 0.0
                is_terminated = False
                is_truncated = True
                log = {"/worker/restart_rate": 1.0}
                restarted[index] = 1.0
                recovered = True
            observations.append(observation)
            rewards[index] = reward
            terminated[index] = float(is_terminated)
            truncated[index] = float(is_truncated)
            if is_terminated or is_truncated or restarted[index] > 0.0:
                if not recovered:
                    observation = self._reset_slot(slot, self._next_seed())
                    observations[-1] = observation
                if log:
                    logs.append({str(key): float(value) for key, value in log.items()})

        self._observations = np.stack(observations).astype(np.float32, copy=False)
        self.episode_length_buf.add_(1)
        done = np.maximum(terminated, truncated)
        self.episode_length_buf[torch.from_numpy(done.astype(bool))] = 0
        infos: dict[str, Any] = {
            "observations": {},
            "time_outs": torch.from_numpy(truncated.copy()),
        }
        if logs:
            keys = set.intersection(*(set(item) for item in logs))
            infos["log"] = {
                key: float(np.mean([item[key] for item in logs]))
                for key in sorted(keys)
            }
        if restarted.any():
            infos["worker_restarts"] = restarted
        self._step_count += self.num_envs
        self._step_elapsed_s += perf_counter() - step_started
        return (
            self._torch_observation(),
            torch.from_numpy(rewards),
            torch.from_numpy(done),
            infos,
        )

    def _torch_observation(self) -> torch.Tensor:
        return torch.from_numpy(self._observations.copy())

    def close(self) -> None:
        if not hasattr(self, "_slots"):
            return
        for slot in self._slots:
            self._stop_slot(slot)
        self._slots.clear()

    def __enter__(self) -> PvsVecEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["PvsVecEnv", "WorkerSpec"]
