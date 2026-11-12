from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P1 with IsaacLab and rsl_rl PPO")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "ppo"
        / "isaaclab"
        / "eval_analytic.json",
    )
    parser.add_argument("--no-eval", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    payload: dict[str, Any] = {
        "schema": "auv_p1_ppo_training.v1",
        "status": "FAIL_EXCEPTION",
        "config_path": str(args.config.resolve()),
        "device": args.device,
    }
    wrapper = None
    output: Path | None = None
    try:
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        from auv_isaaclab.config import load_config
        from auv_isaaclab.evaluation import (
            build_eval_cfg,
            eval_spec_sha256,
            evaluate_policy,
            evaluation_score,
            load_eval_spec,
        )
        from auv_isaaclab.tasks.p1_avoidance.agents.rsl_rl_ppo_cfg import (
            build_rsl_rl_runner_cfg,
        )
        from auv_isaaclab.tasks.p1_avoidance.env import P1IsaacLabEnv, build_p1_env_cfg

        overrides: dict[str, Any] = {}
        if args.num_envs is not None:
            overrides["num_envs"] = args.num_envs
        if args.max_iterations is not None:
            overrides["max_iterations"] = args.max_iterations
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.output is not None:
            overrides["output"] = str(args.output)
        bundle = load_config(args.config, overrides)
        experiment_cfg = bundle.config
        output = Path(experiment_cfg.experiment.output)
        output.mkdir(parents=True, exist_ok=True)
        bundle.write(output / "config")

        seed = experiment_cfg.experiment.seed
        eval_spec = None
        eval_cfg = None
        if not args.no_eval:
            eval_spec = load_eval_spec(args.eval_config)
            eval_cfg = build_eval_cfg(experiment_cfg, eval_spec)
        torch.manual_seed(seed)
        np.random.seed(seed)
        payload.update(
            {
                "config_sha256": bundle.config_sha256,
                "output_dir": str(output.resolve()),
                "num_envs": experiment_cfg.env.num_envs,
                "num_steps_per_env": experiment_cfg.ppo.num_steps_per_env,
                "max_iterations": experiment_cfg.ppo.max_iterations,
                "seed": seed,
            }
        )

        env_cfg = build_p1_env_cfg(experiment_cfg, device=args.device)
        env = P1IsaacLabEnv(env_cfg, experiment_cfg, render_mode=None)
        runner_cfg = build_rsl_rl_runner_cfg(experiment_cfg.ppo)
        runner_cfg.update(
            {
                "seed": seed,
                "device": args.device,
                "experiment_name": experiment_cfg.experiment.name,
                "run_name": args.run_name,
                "logger": "tensorboard",
            }
        )

        class RecordingRunner(OnPolicyRunner):
            def __init__(self, *runner_args: Any, **runner_kwargs: Any) -> None:
                super().__init__(*runner_args, **runner_kwargs)
                self.records: list[dict[str, Any]] = []
                self.evaluations: list[dict[str, Any]] = []
                self.eval_errors: list[dict[str, Any]] = []
                self.best_score: tuple[float, float, float, float] | None = None
                self.best_iteration: int | None = None
                self.best_checkpoint: Path | None = None
                self.best_checkpoint_sha256: str | None = None
                self.best_gates: dict[str, Any] | None = None
                self.latest_checkpoint_sha256: str | None = None

            def _maybe_evaluate(self, completed_iteration: int) -> None:
                if eval_spec is None or eval_cfg is None:
                    return
                if output is None:
                    raise RuntimeError("training output directory is not initialized")
                interval = experiment_cfg.evaluation.eval_every_iterations
                should_evaluate = completed_iteration % interval == 0 or (
                    completed_iteration == experiment_cfg.ppo.max_iterations
                    and experiment_cfg.ppo.max_iterations >= interval
                )
                if not should_evaluate:
                    return
                try:
                    result = evaluate_policy(
                        self.alg.policy.act_inference,
                        eval_cfg,
                        eval_spec,
                        device=self.device,
                        dtype=torch.float32,
                        metadata={
                            "iteration": completed_iteration,
                            "total_timesteps": int(self.tot_timesteps),
                        },
                    )
                except Exception as exc:
                    self.eval_errors.append(
                        {
                            "iteration": completed_iteration,
                            "error": repr(exc),
                            "exception_type": type(exc).__name__,
                            "traceback": traceback.format_exc(),
                        }
                    )
                    return
                self.evaluations.append(result)
                score = evaluation_score(result)
                if self.best_score is None or score > self.best_score:
                    best_path = output / "checkpoint_best.pt"
                    self.save(str(best_path))
                    self.best_score = score
                    self.best_iteration = completed_iteration
                    self.best_checkpoint = best_path
                    self.best_checkpoint_sha256 = _sha256(best_path)
                    self.best_gates = result["gates"]
                latest_path = output / "checkpoint_latest.pt"
                self.save(str(latest_path))
                self.latest_checkpoint_sha256 = _sha256(latest_path)

            def log(self, locs: dict[str, Any], width: int = 80, pad: int = 35) -> None:
                super().log(locs, width=width, pad=pad)
                self.records.append(
                    {
                        "iteration": int(locs["it"]),
                        "losses": {key: float(value) for key, value in locs["loss_dict"].items()},
                        "learning_rate": float(self.alg.learning_rate),
                        "mean_noise_std": float(self.alg.policy.action_std.mean().item()),
                        "collection_time_s": float(locs["collection_time"]),
                        "learn_time_s": float(locs["learn_time"]),
                        "total_timesteps": int(self.tot_timesteps),
                        "fps": int(
                            self.num_steps_per_env
                            * self.env.num_envs
                            / max(locs["collection_time"] + locs["learn_time"], 1e-12)
                        ),
                        "mean_reward": float(np.mean(locs["rewbuffer"]))
                        if locs["rewbuffer"]
                        else None,
                        "mean_episode_length": float(np.mean(locs["lenbuffer"]))
                        if locs["lenbuffer"]
                        else None,
                    }
                )
                self._maybe_evaluate(int(locs["it"]) + 1)

        wrapper = RslRlVecEnvWrapper(env, clip_actions=runner_cfg["clip_actions"])
        runner = RecordingRunner(
            wrapper,
            runner_cfg,
            log_dir=str(output / "tensorboard"),
            device=args.device,
        )
        if args.resume is not None:
            runner.load(str(args.resume))

        start = time.perf_counter()
        runner.learn(
            num_learning_iterations=experiment_cfg.ppo.max_iterations,
            init_at_random_ep_len=False,
        )
        training_s = time.perf_counter() - start

        latest_checkpoint = output / "checkpoint_latest.pt"
        final_checkpoint = output / "checkpoint_final.pt"
        runner.save(str(latest_checkpoint))
        runner.save(str(final_checkpoint))
        checkpoint_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
        model_state = checkpoint_payload["model_state_dict"]
        model_finite = all(bool(torch.isfinite(value).all()) for value in model_state.values())
        observation, _ = wrapper.get_observations()
        with torch.inference_mode():
            deterministic_action = runner.alg.policy.act_inference(observation.clone())
        records = runner.records
        losses_finite = all(
            math.isfinite(value) for record in records for value in record["losses"].values()
        )
        entropy_values = [float(record["losses"].get("entropy", math.inf)) for record in records]
        entropy_finite = all(math.isfinite(value) for value in entropy_values)
        # A Gaussian policy's differential entropy can legitimately be negative
        # after the action standard deviation shrinks.  Require a finite value,
        # but keep the old positive-entropy flag as a diagnostic only.
        entropy_positive = all(value > 0.0 for value in entropy_values)
        payload.update(
            {
                "status": "PASS",
                "training_s": training_s,
                "iterations_completed": len(records),
                "records": records,
                "checkpoint": str(final_checkpoint.resolve()),
                "checkpoint_sha256": _sha256(final_checkpoint),
                "checkpoint_iteration": int(checkpoint_payload["iter"]),
                "checkpoint_latest": str(latest_checkpoint.resolve()),
                "checkpoint_latest_sha256": _sha256(latest_checkpoint),
                "checkpoint_final": str(final_checkpoint.resolve()),
                "checkpoint_final_sha256": _sha256(final_checkpoint),
                "evaluations": runner.evaluations,
                "eval_errors": runner.eval_errors,
                "evaluation_count": len(runner.evaluations),
                "evaluation_gate": (
                    runner.evaluations[-1]["status"] if runner.evaluations else "NOT_RUN"
                ),
                "best_iteration": runner.best_iteration,
                "best_score": list(runner.best_score) if runner.best_score is not None else None,
                "best_checkpoint": (
                    str(runner.best_checkpoint.resolve())
                    if runner.best_checkpoint is not None
                    else None
                ),
                "best_checkpoint_sha256": runner.best_checkpoint_sha256,
                "best_gates": runner.best_gates,
                "eval_spec": (str(eval_spec.source_path) if eval_spec is not None else None),
                "eval_spec_sha256": (
                    eval_spec_sha256(eval_spec) if eval_spec is not None else None
                ),
                "model_parameters_finite": model_finite,
                "losses_finite": losses_finite,
                "entropy_positive": entropy_positive,
                "entropy_finite": entropy_finite,
                "entropy_min": min(entropy_values) if entropy_values else None,
                "deterministic_action_finite": bool(torch.isfinite(deterministic_action).all()),
                "deterministic_action_shape": list(deterministic_action.shape),
                "total_timesteps": int(runner.tot_timesteps),
            }
        )
        if len(records) != experiment_cfg.ppo.max_iterations:
            payload["status"] = "FAIL_ITERATION_COUNT"
        elif not model_finite or not losses_finite:
            payload["status"] = "FAIL_NONFINITE"
        elif not entropy_finite:
            payload["status"] = "FAIL_ENTROPY"
        elif not payload["deterministic_action_finite"]:
            payload["status"] = "FAIL_ACTION"
        elif runner.eval_errors:
            payload["status"] = "FAIL_EVAL"
    except Exception as exc:  # pragma: no cover - requires IsaacLab runtime
        is_oom = "out of memory" in repr(exc).lower() or type(exc).__name__ in {
            "OutOfMemoryError",
            "CudaOutOfMemoryError",
        }
        payload.update(
            {
                "status": "FAIL_OOM" if is_oom else "FAIL_EXCEPTION",
                "error": repr(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if wrapper is not None:
            try:
                wrapper.close()
            except Exception:
                pass
        summary_path = (
            output / "training_summary.json"
            if output is not None
            else args.config.resolve().parent / "training_summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        try:
            simulation_app.close()
        except SystemExit:
            pass

    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
