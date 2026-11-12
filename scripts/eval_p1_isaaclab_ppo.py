from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a P1 PPO checkpoint analytically")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(checkpoint: Path, cfg: Any, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from rsl_rl.modules import ActorCritic

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain model_state_dict")
    policy = ActorCritic(
        cfg.policy_input_dim,
        cfg.policy_input_dim,
        2,
        actor_hidden_dims=list(cfg.ppo.actor_hidden_dims),
        critic_hidden_dims=list(cfg.ppo.critic_hidden_dims),
        activation="elu",
        init_noise_std=1.0,
    )
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    missing, unexpected = policy.actor.load_state_dict(actor_state, strict=True)
    if missing or unexpected:
        raise ValueError(
            "checkpoint is incompatible with the configured actor: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    if "std" not in state:
        raise ValueError("checkpoint does not contain the policy action std")
    with torch.no_grad():
        policy.std.copy_(state["std"].to(device=policy.std.device, dtype=policy.std.dtype))
    policy.to(device)
    policy.eval()
    return policy, payload


def _write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    columns = (
        "scenario",
        "seed",
        "success",
        "collision",
        "out_of_bounds",
        "timeout",
        "min_clearance_m",
        "time_to_goal_s",
        "path_length_m",
        "final_goal_distance_m",
        "episode_steps",
        "action_saturation",
        "all_zero_actions",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for episode in episodes:
            writer.writerow({key: episode.get(key) for key in columns})


def main() -> int:
    args = _parse_args()
    payload: dict[str, Any] = {
        "schema": "auv_p1_ppo_eval_run.v1",
        "status": "FAIL_EXCEPTION",
        "config_path": str(args.config.resolve()),
    }
    output: Path | None = None
    try:
        import torch

        from auv_isaaclab.config import config_sha256, load_config
        from auv_isaaclab.evaluation import (
            build_eval_cfg,
            eval_spec_sha256,
            evaluate_policy,
            load_eval_spec,
        )

        spec = load_eval_spec(args.config)
        checkpoint = args.checkpoint if args.checkpoint is not None else spec.checkpoint
        if checkpoint is None:
            raise ValueError("checkpoint must be set in the eval config or passed on the CLI")
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = args.output if args.output is not None else Path(spec.output)
        output.mkdir(parents=True, exist_ok=True)
        device = args.device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        bundle = load_config(spec.base_config, {"num_envs": spec.episode_count})
        eval_cfg = build_eval_cfg(bundle.config, spec)
        policy, checkpoint_payload = _load_policy(checkpoint, eval_cfg, device)
        metadata = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_iteration": int(checkpoint_payload.get("iter", -1)),
            "config_sha256": config_sha256(eval_cfg.as_dict()),
            "eval_spec_sha256": eval_spec_sha256(spec),
            "eval_spec_path": str(spec.source_path),
        }
        summary = evaluate_policy(
            policy.act_inference,
            eval_cfg,
            spec,
            device=device,
            dtype=torch.float32,
            metadata=metadata,
        )
        payload.update(summary)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _write_csv(output / "episodes.csv", summary["episodes"])
    except Exception as exc:  # pragma: no cover - requires IsaacLab/runtime
        payload.update(
            {
                "status": "FAIL_EXCEPTION",
                "error": repr(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "summary.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
