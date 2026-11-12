#!/usr/bin/env python3
"""Read-only audit of PC terminal success and safety co-occurrence.

This script does not train and does not change reward, termination, success or
the goal radius.  It runs one closed-loop arm, records every terminal state, and
reports whether a nominal GOAL/plane-miss occurred together with a raw safety
condition or an out-of-envelope terminal vertical/pitch state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.pc import PC_SAFETY_TERMINAL_KINDS, PCTerminalKind
from auv_lowlevel.batch.pc_bc import PCBCPolicy
from auv_lowlevel.config import load_config
from auv_rl.ddpg.pc_command_residual_env import PCCommandResidualOffPolicyEnv
from auv_rl.training import load_training_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = Path("configs/lowlevel/goal86/pc_goal86_p-a_ppo_formal_20261110.json")
DEFAULT_ENVIRONMENT_CONFIG = Path(
    "configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json"
)
DEFAULT_CHECKPOINT = Path(
    "artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--action-mode", choices=("policy", "zero-residual"), default="policy"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_seeds(seed_start: int, seed_count: int) -> np.ndarray:
    if seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    seeds = np.arange(seed_start, seed_start + seed_count, dtype=np.int64)
    if ((seeds >= 100) & (seeds <= 149)).any():
        raise ValueError("held-out seeds 100-149 are closed")
    return seeds


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _raw_safety_flags(
    *,
    info: dict[str, Any],
    inner: Any,
    config: Any,
    index: int,
) -> dict[str, bool]:
    eta = np.asarray(_tensor_to_numpy(inner.eta)[index], dtype=np.float64)
    nu = np.asarray(_tensor_to_numpy(inner.nu)[index], dtype=np.float64)
    position = eta[:3]
    finite = bool(np.isfinite(eta).all() and np.isfinite(nu).all())
    roll_time = float(_tensor_to_numpy(info["roll_exceed_time_s"])[index])
    rate_time = float(_tensor_to_numpy(info["roll_rate_exceed_time_s"])[index])
    roll_safety = finite and roll_time >= float(config.termination.t_phi_term_s)
    rate_safety = finite and rate_time >= float(config.termination.t_p_term_s)
    bounds = np.asarray(config.plant.bounds_ned_m, dtype=np.float64)
    outside = bool(np.any(position < bounds[:, 0] - 1e-9) or np.any(position > bounds[:, 1] + 1e-9))
    depth_limit = False
    if config.termination.min_depth_m is not None and config.termination.max_depth_m is not None:
        depth_limit = bool(
            position[2] < float(config.termination.min_depth_m)
            or position[2] > float(config.termination.max_depth_m)
        )
    invalid_state = finite and bool(abs(float(eta[4])) >= math.pi / 2.0 - 1e-3)
    return {
        "roll_angle": roll_safety,
        "roll_rate": rate_safety,
        "depth_limit": depth_limit,
        "out_of_bounds": outside,
        "invalid_state": invalid_state,
        "nan_or_inf": not finite,
    }


def _policy_actions(
    *,
    env: PCCommandResidualOffPolicyEnv,
    policy: PCBCPolicy,
    active: np.ndarray,
    residual_scale: float,
) -> np.ndarray:
    actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
    if not bool(active.any()):
        return actions
    with torch.inference_mode():
        observation = env.inner.observation
        actor_path = observation.actor_path
        if actor_path is not None:
            actor_path = actor_path[:, : policy.model.path_dim]
        command = torch.stack(
            (
                observation.command.heading_rad,
                observation.command.pitch_rad,
                observation.command.speed_mps,
            ),
            dim=1,
        )
        actor_current = getattr(observation, "actor_current_body", None)
        policy_actions = policy.act_inference(
            observation.actor_state[active],
            command[active],
            actor_path[active] if actor_path is not None else None,
            actor_current[active] if actor_current is not None else None,
        )
    actions[active] = policy_actions.detach().cpu().numpy()
    actions *= float(residual_scale)
    return actions


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _classify_h141_a(goal_hits: int, goal_count: int) -> str:
    if goal_count == 0:
        return "INCONCLUSIVE_NO_SUCCESS"
    if goal_hits >= 3:
        return "SUPPORTED"
    if goal_hits <= 1:
        return "FALSIFIED"
    return "WEAK_CONTRACT_GAP"


def _classify_h141_c(failures: list[dict[str, Any]]) -> str:
    if not failures:
        return "INCONCLUSIVE_NO_FAILURE"
    kinds = [str(item["terminal_kind"]) for item in failures]
    plane_miss = sum(kind == "goal_plane_miss" for kind in kinds)
    hit_fraction = float(np.mean([bool(item["candidate_envelope_hit"]) for item in failures]))
    if plane_miss > len(failures) / 2 and hit_fraction < 0.5:
        return "MISSING_TERMINAL_NOT_PRIMARY"
    if hit_fraction >= 0.5:
        return "MISSING_TERMINAL_COULD_RELABEL_FAILURES"
    return "WEAK_MISSING_TERMINAL_EVIDENCE"


def main() -> int:
    args = _parse_args()
    if args.action_mode == "policy" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for --action-mode policy")
    if not np.isfinite(args.residual_scale) or args.residual_scale <= 0.0:
        raise ValueError("--residual-scale must be finite and positive")
    training_path = _resolve(args.training_config)
    environment_path = _resolve(args.environment_config)
    output = _resolve(args.output)
    seeds = _validate_seeds(args.seed_start, args.seed_count)
    payload = load_training_payload(training_path)
    observation_payload = dict(payload.get("observation", {}))
    environment = load_config(environment_path)
    action_interface = str(payload.get("action_interface", "command"))
    actuator_residual_scale = float(payload.get("actuator_residual_scale", 0.25))
    actuator_max_action = float(payload.get("actuator_max_action", 1.0))
    pc_payload = dict(payload.get("pc", {}))
    path_dim = int(observation_payload.get("path_dim", pc_payload.get("path_dim", 8)))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    checkpoint_path = None if args.checkpoint is None else _resolve(args.checkpoint)
    policy = None
    if checkpoint_path is not None:
        policy = PCBCPolicy(checkpoint_path, device=args.device)
        if policy.model.path_dim not in (0, 8, 12):
            raise ValueError(
                "terminal-contract audit requires a path-free, path8 or path12 BC checkpoint; "
                f"got path_dim={policy.model.path_dim}"
            )
    if (
        args.action_mode == "policy"
        and policy is not None
        and policy.model.actor_command_dim not in (0, 3)
    ):
        raise ValueError("terminal-contract audit requires actor_command_dim=0 or 3")
    env = PCCommandResidualOffPolicyEnv(
        environment,
        num_envs=len(seeds),
        device=args.device,
        dtype=dtype,
        path_dim=path_dim,
        action_interface=action_interface,
        actuator_residual_scale=actuator_residual_scale,
        actuator_max_action=actuator_max_action,
        validate_finite=False,
    )
    env.set_action_mode("policy" if args.action_mode == "policy" else "zero-residual")
    env.reset(seeds.tolist())
    active = np.ones(env.num_envs, dtype=bool)
    steps = np.zeros(env.num_envs, dtype=np.int64)
    terminal: list[dict[str, Any] | None] = [None] * env.num_envs
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_episode_steps)
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    for _ in range(max_steps):
        active_before = active.copy()
        if not bool(active_before.any()):
            break
        if args.action_mode == "policy":
            assert policy is not None
            actions = _policy_actions(
                env=env,
                policy=policy,
                active=active_before,
                residual_scale=float(args.residual_scale),
            )
        else:
            actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float64)
        with torch.inference_mode():
            _, _, terminated, truncated, info = env.step(
                actions, auto_reset=False, advance_mask=active_before
            )
        steps[active_before] += 1
        done_now = active_before & (terminated | truncated)
        if bool(done_now.any()):
            kind_values = _tensor_to_numpy(info["terminal_kind"]).astype(np.int64)
            eta_all = _tensor_to_numpy(env.inner.eta)
            nu_all = _tensor_to_numpy(env.inner.nu)
            projection = env.inner.path.project(env.inner.eta[:, :3])
            eta_distance = _tensor_to_numpy(projection.goal_distance_m) if hasattr(projection, "goal_distance_m") else None
            if eta_distance is None:
                goal = env.inner.path.goal_ned_m_for(env.num_envs)
                eta_distance = _tensor_to_numpy(torch.linalg.norm(env.inner.eta[:, :3] - goal, dim=1))
            p_nav_distance = _tensor_to_numpy(
                torch.linalg.norm(
                    env.inner.sensor_output.p_nav - env.inner.path.goal_ned_m_for(env.num_envs),
                    dim=1,
                )
            )
            vertical_error = _tensor_to_numpy(projection.vertical_error_m)
            horizontal_error = _tensor_to_numpy(projection.horizontal_error_m)
            path_error = _tensor_to_numpy(projection.error_3d_m)
            signed_offset = _tensor_to_numpy(info["signed_goal_tangent_offset_m"])
            for index in np.flatnonzero(done_now).tolist():
                kind = int(kind_values[index])
                safety_flags = _raw_safety_flags(
                    info=info, inner=env.inner, config=environment, index=index
                )
                raw_safety = any(safety_flags.values())
                pitch = float(eta_all[index, 4])
                q = float(nu_all[index, 4])
                w = float(nu_all[index, 2])
                vertical = float(vertical_error[index])
                candidate_flags = {
                    "pitch_angle": bool(abs(pitch) >= math.radians(45.0)),
                    "pitch_rate": bool(abs(q) >= environment.guidance.pitch_rate_limit_rad_s),
                    "heave": bool(abs(w) >= 0.5),
                    "vertical_error": bool(abs(vertical) >= 1.0),
                }
                terminal[index] = {
                    "seed": int(seeds[index]),
                    "steps": int(steps[index]),
                    "terminal_kind": PCTerminalKind(kind).name.lower(),
                    "terminal_kind_value": kind,
                    "success": kind == int(PCTerminalKind.GOAL),
                    "goal_plane_miss": kind == int(PCTerminalKind.GOAL_PLANE_MISS),
                    "timeout": kind == int(PCTerminalKind.TIME_LIMIT),
                    "safety": PCTerminalKind(kind) in PC_SAFETY_TERMINAL_KINDS,
                    "terminal_eta_distance_m": float(eta_distance[index]),
                    "terminal_p_nav_distance_m": float(p_nav_distance[index]),
                    "terminal_vertical_error_m": vertical,
                    "terminal_horizontal_error_m": float(horizontal_error[index]),
                    "terminal_path_error_3d_m": float(path_error[index]),
                    "terminal_pitch_rad": pitch,
                    "terminal_pitch_rate_rad_s": q,
                    "terminal_heave_mps": w,
                    "terminal_surge_mps": float(nu_all[index, 0]),
                    "terminal_roll_rad": float(eta_all[index, 3]),
                    "terminal_roll_rate_rad_s": float(nu_all[index, 3]),
                    "terminal_signed_goal_offset_m": float(signed_offset[index]),
                    "raw_safety_flags": safety_flags,
                    "raw_safety_any": bool(raw_safety),
                    "goal_with_raw_safety": bool(kind == int(PCTerminalKind.GOAL) and raw_safety),
                    "plane_miss_with_raw_safety": bool(
                        kind == int(PCTerminalKind.GOAL_PLANE_MISS) and raw_safety
                    ),
                    "candidate_envelope_flags": candidate_flags,
                    "candidate_envelope_hit": bool(any(candidate_flags.values())),
                }
        active[done_now] = False

    if bool(active.any()):
        raise RuntimeError(
            f"audit ended with {int(active.sum())} active episodes at max_steps={max_steps}"
        )
    rows = [item for item in terminal if item is not None]
    if len(rows) != len(seeds):
        raise RuntimeError(f"expected {len(seeds)} terminal rows, got {len(rows)}")
    successes = [row for row in rows if row["success"]]
    failures = [row for row in rows if not row["success"]]
    goal_hits = sum(row["candidate_envelope_hit"] for row in successes)
    safety_overlap = [row for row in rows if row["goal_with_raw_safety"] or row["plane_miss_with_raw_safety"]]
    failure_hits = sum(row["candidate_envelope_hit"] for row in failures)
    kind_counts: dict[str, int] = {}
    for row in rows:
        kind_counts[row["terminal_kind"]] = kind_counts.get(row["terminal_kind"], 0) + 1
    safety_flag_counts = {
        name: int(sum(bool(row["raw_safety_flags"][name]) for row in rows))
        for name in rows[0]["raw_safety_flags"]
    }
    status = {
        "h141_a": _classify_h141_a(goal_hits, len(successes)),
        "h141_b": "SUPPORTED" if safety_overlap else "FALSIFIED",
        "h141_c": _classify_h141_c(failures),
    }
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "auv_pc_goal141_terminal_contract_audit.v1",
        "action_mode": args.action_mode,
        "training_config": str(training_path),
        "training_config_sha256": _sha256(training_path),
        "environment_config": str(environment_path),
        "environment_config_sha256": environment.config_sha256,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": None if checkpoint_path is None else _sha256(checkpoint_path),
        "seeds": seeds.tolist(),
        "episode_count": len(rows),
        "counts": kind_counts,
        "success_count": len(successes),
        "failure_count": len(failures),
        "goal_candidate_envelope_hits": int(goal_hits),
        "goal_candidate_envelope_hit_fraction": float(goal_hits / max(len(successes), 1)),
        "failure_candidate_envelope_hits": int(failure_hits),
        "failure_candidate_envelope_hit_fraction": float(failure_hits / max(len(failures), 1)),
        "priority_overlap_count": len(safety_overlap),
        "priority_overlap_seeds": [row["seed"] for row in safety_overlap],
        "raw_safety_flag_counts": safety_flag_counts,
        "candidate_thresholds": {
            "pitch_angle_rad": math.radians(45.0),
            "pitch_rate_rad_s": float(environment.guidance.pitch_rate_limit_rad_s),
            "heave_mps": 0.5,
            "vertical_error_m": 1.0,
        },
        "terminal_stats": {
            "success_abs_pitch_rad": _summary(np.abs([row["terminal_pitch_rad"] for row in successes])),
            "failure_abs_pitch_rad": _summary(np.abs([row["terminal_pitch_rad"] for row in failures])),
            "success_abs_pitch_rate_rad_s": _summary(np.abs([row["terminal_pitch_rate_rad_s"] for row in successes])),
            "failure_abs_pitch_rate_rad_s": _summary(np.abs([row["terminal_pitch_rate_rad_s"] for row in failures])),
            "success_abs_heave_mps": _summary(np.abs([row["terminal_heave_mps"] for row in successes])),
            "failure_abs_heave_mps": _summary(np.abs([row["terminal_heave_mps"] for row in failures])),
            "success_abs_vertical_error_m": _summary(np.abs([row["terminal_vertical_error_m"] for row in successes])),
            "failure_abs_vertical_error_m": _summary(np.abs([row["terminal_vertical_error_m"] for row in failures])),
        },
        "status": status,
        "episodes": rows,
    }
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Goal141 terminal contract audit",
        "",
        f"- action mode: `{args.action_mode}`",
        f"- seeds: `{args.seed_start}-{args.seed_start + args.seed_count - 1}`",
        f"- terminal kinds: `{kind_counts}`",
        f"- success / failure: `{len(successes)}` / `{len(failures)}`",
        f"- GOAL candidate-envelope hits: `{goal_hits}/{len(successes)}`",
        f"- failure candidate-envelope hits: `{failure_hits}/{len(failures)}`",
        f"- GOAL/plane-miss + raw safety: `{len(safety_overlap)}`",
        f"- raw safety flag counts: `{safety_flag_counts}`",
        f"- H141-A: `{status['h141_a']}`",
        f"- H141-B: `{status['h141_b']}`",
        f"- H141-C: `{status['h141_c']}`",
    ]
    (output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "counts": kind_counts, "overlaps": len(safety_overlap)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

