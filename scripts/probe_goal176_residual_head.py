#!/usr/bin/env python3
"""Probe ``z_t`` and residual-head sensitivity on the Goal175 evaluator.

The script runs the existing Goal175 policy in closed loop on dev seeds 20-27,
captures the actor observations that the evaluator actually used, and then
performs two read-only diagnostics:

* ridge linear probes from ``z_t`` to vertical error, pitch error, q, and w;
* central-difference Jacobians of the raw/applied/total residual action.

No training, reward change, termination change, success change, radius change,
or held-out access is performed.
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
from torch import Tensor

from auv_lowlevel.batch.pc_normalization import PCNormalization
from auv_lowlevel.batch.pc_path import pc_path_feature_scales_for_dim
from auv_lowlevel.cli.evaluate_pc_ppo import runner as eval_runner

TARGET_NAMES = ("vertical_error_m", "pitch_error_rad", "q_rad_s", "w_mps")
STATE_Q_INDEX = 7
STATE_W_INDEX = 11
PATH_VERTICAL_INDEX = 1
PATH_PITCH_INDEX = 3
PATH_CROSS_INDEX = 0
PATH_HEADING_INDEX = 2
RIDGE_ALPHA = 1.0
TRAIN_SEEDS = (20, 21, 22, 23, 24, 25)
TEST_SEEDS = (26, 27)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "configs" / "a2_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest must be an object")
    return payload


def _run_record(manifest: dict[str, Any], arm: str) -> dict[str, Any]:
    for record in manifest.get("train_runs", []):
        if isinstance(record, dict) and record.get("arm") == arm:
            return record
    raise ValueError(f"{arm}: not found in a2_manifest.train_runs")


def _scenario_record(manifest: dict[str, Any], scenario: str) -> dict[str, Any]:
    for record in manifest.get("eval_scenarios", []):
        if isinstance(record, dict) and record.get("scenario") == scenario:
            return record
    raise ValueError(f"{scenario}: not found in a2_manifest.eval_scenarios")


def _environment_config(manifest: dict[str, Any], scenario: str, arm: str) -> Path:
    record = _scenario_record(manifest, scenario)
    mode = "ALOS-COMP" if arm.endswith("_COMP") else "ALOS-NONE"
    return Path(record["configs"][mode])


def _latent(policy: Any, observation: Tensor) -> Tensor:
    state, path, current, _command, _short, _long = policy.actor_layout.split_with_optional_current(
        observation
    )
    trainable_state = policy._trainable_actor_state(state)
    z = policy._fuse_actor_path(path, policy.actor_encoder(trainable_state))
    return policy._fuse_actor_current(current, z)


def _targets(policy: Any, observation: Tensor, normalization: PCNormalization) -> Tensor:
    state, path, _current, _command, _short, _long = (
        policy.actor_layout.split_with_optional_current(observation)
    )
    if path is None:
        raise ValueError("probe requires an enabled path block")
    path_scales = pc_path_feature_scales_for_dim(path.shape[1])
    state_scales = normalization.actor_state_scales
    return torch.stack(
        [
            path[:, PATH_VERTICAL_INDEX] * path_scales[PATH_VERTICAL_INDEX],
            path[:, PATH_PITCH_INDEX] * path_scales[PATH_PITCH_INDEX],
            state[:, STATE_Q_INDEX] * state_scales[STATE_Q_INDEX],
            state[:, STATE_W_INDEX] * state_scales[STATE_W_INDEX],
        ],
        dim=1,
    )


def _normalization(metadata: dict[str, Any]) -> PCNormalization:
    raw = metadata.get("normalization", {})
    if not isinstance(raw, dict):
        raise ValueError("checkpoint metadata has no normalization object")
    return PCNormalization(
        position_scale_m=float(raw["position_scale_m"]),
        attitude_scale_rad=float(raw["attitude_scale_rad"]),
        angular_rate_scale_rad_s=float(raw["angular_rate_scale_rad_s"]),
        surge_speed_scale_mps=float(raw["surge_speed_scale_mps"]),
        sway_heave_speed_scale_mps=float(raw["sway_heave_speed_scale_mps"]),
        current_scale_mps=float(raw["current_scale_mps"]),
    )


def _capture_eval(
    *,
    training_config: Path,
    checkpoint: Path,
    environment_config: Path,
    seeds: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    dict[str, Any], Any, dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray
]:
    state: dict[str, Any] = {"z": [], "targets": [], "observations": [], "policy": None}
    original_builder = eval_runner._build_policy

    def wrapped_builder(*args: Any, **kwargs: Any) -> Any:
        policy, metadata = original_builder(*args, **kwargs)
        normalization = _normalization(metadata)
        original_inference = policy.act_inference

        def hooked_inference(observation: Tensor) -> Tensor:
            with torch.inference_mode():
                latent = _latent(policy, observation)
                targets = _targets(policy, observation, normalization)
                state["z"].append(latent.detach().cpu().numpy())
                state["targets"].append(targets.detach().cpu().numpy())
                state["observations"].append(observation.detach().cpu().numpy())
            return original_inference(observation)

        policy.act_inference = hooked_inference
        state["policy"] = policy
        state["original_inference"] = original_inference
        return policy, metadata

    eval_runner._build_policy = wrapped_builder
    try:
        summary, metadata, episodes, contract = eval_runner._run(
            training_config_path=training_config,
            checkpoint_path=checkpoint,
            environment_config_path=environment_config,
            seeds=seeds,
            device=device,
            dtype=dtype,
            action_mode="policy",
            allow_alos_current_mode_override=True,
            disable_curriculum_direct_command=True,
            success_frame="both",
            trace_output=None,
        )
    finally:
        eval_runner._build_policy = original_builder
    if state["policy"] is None or not state["z"]:
        raise RuntimeError("evaluator never built or called the policy")
    state["policy"].act_inference = state["original_inference"]
    z = np.stack(state["z"], axis=0)
    targets = np.stack(state["targets"], axis=0)
    observations = np.stack(state["observations"], axis=0)
    return summary, state["policy"], metadata, episodes, z, targets, observations


def _trim_by_seed(
    z: np.ndarray,
    targets: np.ndarray,
    observations: np.ndarray,
    episodes: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if z.shape[0] < 1:
        raise ValueError("empty probe capture")
    if z.shape[0] != targets.shape[0] or z.shape[0] != observations.shape[0]:
        raise ValueError("capture arrays are not aligned")
    if len(episodes) != len(seeds):
        raise ValueError("episodes and seeds are not aligned")
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    o_rows: list[np.ndarray] = []
    seed_rows: list[np.ndarray] = []
    for index, seed in enumerate(seeds):
        steps = int(episodes[index]["steps"])
        take = min(steps, z.shape[0])
        if take <= 0:
            continue
        x_rows.append(z[:take, index, :])
        y_rows.append(targets[:take, index, :])
        o_rows.append(observations[:take, index, :])
        seed_rows.append(np.full(take, seed, dtype=np.int64))
    return (
        np.concatenate(x_rows, axis=0),
        np.concatenate(y_rows, axis=0),
        np.concatenate(o_rows, axis=0),
        np.concatenate(seed_rows, axis=0),
    )


def _ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean = y_train.mean(axis=0, keepdims=True)
    xz = (x_train - x_mean) / x_std
    xt = (x_test - x_mean) / x_std
    yz = y_train - y_mean
    gram = xz.T @ xz + alpha * np.eye(xz.shape[1], dtype=xz.dtype)
    weights = np.linalg.solve(gram, xz.T @ yz)
    return xt @ weights + y_mean


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    residual = float(np.sum((y_true - y_pred) ** 2, dtype=np.float64))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2, dtype=np.float64))
    if total <= 1e-15:
        return None
    return 1.0 - residual / total


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2 or float(np.std(y_true)) <= 1e-15 or float(np.std(y_pred)) <= 1e-15:
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _probe(
    z: np.ndarray,
    targets: np.ndarray,
    seeds: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    train_mask = np.isin(seeds, TRAIN_SEEDS)
    test_mask = np.isin(seeds, TEST_SEEDS)
    if not bool(train_mask.any()) or not bool(test_mask.any()):
        raise ValueError("probe split produced an empty train or test set")
    prediction = _ridge_fit_predict(z[train_mask], targets[train_mask], z[test_mask], alpha=alpha)
    test_targets = targets[test_mask]
    result: dict[str, Any] = {
        "ridge_alpha": alpha,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "targets": {},
    }
    for index, name in enumerate(TARGET_NAMES):
        y_true = test_targets[:, index]
        y_pred = prediction[:, index]
        result["targets"][name] = {
            "r2_test": _r2(y_true, y_pred),
            "pearson_test": _corr(y_true, y_pred),
            "test_mean": float(np.mean(y_true)),
            "test_std": float(np.std(y_true)),
        }
    return result


def _sample_observations(
    observations: np.ndarray,
    seeds: np.ndarray,
    *,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    sample_seeds: list[int] = []
    unique_seeds = [int(seed) for seed in np.unique(seeds)]
    per_seed = max(1, max_samples // max(len(unique_seeds), 1))
    for seed in unique_seeds:
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            continue
        take = min(per_seed, indices.size)
        selected = np.linspace(0, indices.size - 1, take, dtype=np.int64)
        rows.append(observations[indices[selected]])
        sample_seeds.extend([seed] * take)
    return np.concatenate(rows, axis=0), np.asarray(sample_seeds, dtype=np.int64)


def _policy_outputs(policy: Any, observation: Tensor) -> dict[str, Tensor]:
    return {
        "raw_residual": policy._actor_residual_mean(observation),
        "applied_residual": policy.residual_action(observation),
        "total_action": policy.act_inference(observation),
    }


def _jacobian(
    policy: Any,
    observations: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    normalization: PCNormalization,
    action_full_scale: tuple[float, float, float],
    alpha: float,
) -> dict[str, Any]:
    del alpha
    policy_dtype = next(policy.parameters()).dtype
    obs = torch.as_tensor(observations, device=device, dtype=policy_dtype)
    state_end = policy.actor_layout.state_dim
    path_dim = policy.actor_layout.path_dim
    if path_dim < 4:
        raise ValueError("Jacobian requires the 8/12-D path block")
    path_scales = pc_path_feature_scales_for_dim(path_dim)
    perturbations = {
        "path_cross": (state_end + PATH_CROSS_INDEX, 0.10 / path_scales[PATH_CROSS_INDEX]),
        "path_vertical": (
            state_end + PATH_VERTICAL_INDEX,
            0.10 / path_scales[PATH_VERTICAL_INDEX],
        ),
        "path_heading": (state_end + PATH_HEADING_INDEX, 0.05 / path_scales[PATH_HEADING_INDEX]),
        "path_pitch": (state_end + PATH_PITCH_INDEX, 0.05 / path_scales[PATH_PITCH_INDEX]),
        "state_q": (STATE_Q_INDEX, 0.05 / normalization.angular_rate_scale_rad_s),
        "state_w": (STATE_W_INDEX, 0.05 / normalization.sway_heave_speed_scale_mps),
    }
    outputs: dict[str, Any] = {}
    with torch.inference_mode():
        zero = _policy_outputs(policy, obs)
        for name, (column, normalized_delta) in perturbations.items():
            plus = obs.clone()
            minus = obs.clone()
            plus[:, column] += normalized_delta
            minus[:, column] -= normalized_delta
            plus_outputs = _policy_outputs(policy, plus)
            minus_outputs = _policy_outputs(policy, minus)
            physical_delta = 0.10 if "cross" in name or "vertical" in name else 0.05
            entry: dict[str, Any] = {
                "column": int(column),
                "normalized_delta": float(normalized_delta),
                "physical_delta": float(physical_delta),
                "channels": {},
            }
            for output_name in ("raw_residual", "applied_residual", "total_action"):
                derivative = (plus_outputs[output_name] - minus_outputs[output_name]) / (
                    2.0 * physical_delta
                )
                derivative_np = derivative.detach().cpu().numpy()
                zero_np = zero[output_name].detach().cpu().numpy()
                channel_rows: dict[str, Any] = {}
                for channel, channel_name in enumerate(("rudder", "stern", "rpm")):
                    values = derivative_np[:, channel]
                    positive = float(np.mean(values > 0.0))
                    negative = float(np.mean(values < 0.0))
                    sign_consistency = max(positive, negative)
                    effect_physical = (
                        float(np.median(np.abs(values)))
                        * physical_delta
                        * action_full_scale[channel]
                    )
                    channel_rows[channel_name] = {
                        "median_abs_derivative": float(np.median(np.abs(values))),
                        "median_signed_derivative": float(np.median(values)),
                        "sign_consistency": float(sign_consistency),
                        "effect_at_perturbation_normalized": float(
                            np.median(np.abs(values)) * physical_delta
                        ),
                        "effect_at_perturbation_physical": effect_physical,
                        "mean_abs_zero_action": float(np.mean(np.abs(zero_np[:, channel]))),
                    }
                entry["channels"][output_name] = channel_rows
            for output_name in ("raw_residual", "applied_residual"):
                for channel_name in ("rudder", "stern", "rpm"):
                    numerator = entry["channels"][output_name][channel_name][
                        "median_abs_derivative"
                    ]
                    denominator = entry["channels"]["total_action"][channel_name][
                        "median_abs_derivative"
                    ]
                    entry["channels"][output_name][channel_name]["share_of_total_median_abs"] = (
                        None if denominator <= 1e-15 else numerator / denominator
                    )
            outputs[name] = entry

    ratios: dict[str, Any] = {}
    for output_name in ("raw_residual", "applied_residual", "total_action"):
        vertical = outputs["path_vertical"]["channels"][output_name]["rudder"][
            "median_abs_derivative"
        ]
        cross = outputs["path_cross"]["channels"][output_name]["rudder"]["median_abs_derivative"]
        pitch = outputs["path_pitch"]["channels"][output_name]["rudder"]["median_abs_derivative"]
        heading = outputs["path_heading"]["channels"][output_name]["rudder"][
            "median_abs_derivative"
        ]
        ratios[output_name] = {
            "rudder_vertical_over_cross": None if cross <= 1e-15 else vertical / cross,
            "rudder_pitch_over_heading": None if heading <= 1e-15 else pitch / heading,
        }
    residual_share = {
        name: outputs[name]["channels"]["applied_residual"]["rudder"]["share_of_total_median_abs"]
        for name in (
            "path_cross",
            "path_vertical",
            "path_heading",
            "path_pitch",
            "state_q",
            "state_w",
        )
    }
    return {
        "sample_count": int(obs.shape[0]),
        "inputs": outputs,
        "ratios": ratios,
        "applied_rudder_share_of_total": residual_share,
    }


def _fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal176 residual-head signal audit",
        "",
        f"- root: `{payload['root']}`",
        f"- seeds: train `{TRAIN_SEEDS}`, test `{TEST_SEEDS}`",
        f"- ridge alpha: `{RIDGE_ALPHA}`",
        "- Jacobians are normalized action derivatives per physical unit; "
        "`physical` columns convert only the rudder channel with the checkpoint action scale.",
        "",
    ]
    for arm, arm_payload in payload["arms"].items():
        lines.extend(
            [
                f"## {arm}",
                "",
                "| target | test R² | Pearson r | test mean | test std |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for target, row in arm_payload["probe"]["targets"].items():
            lines.append(
                f"| {target} | {_fmt(row['r2_test'])} | {_fmt(row['pearson_test'])} | "
                f"{_fmt(row['test_mean'])} | {_fmt(row['test_std'])} |"
            )
        lines.extend(
            [
                "",
                "| input | raw Δrudder | applied Δrudder | total Δrudder | applied/total | applied sign | applied physical Δrudder |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for input_name, input_row in arm_payload["jacobian"]["inputs"].items():
            applied = input_row["channels"]["applied_residual"]["rudder"]
            total = input_row["channels"]["total_action"]["rudder"]
            lines.append(
                f"| {input_name} | {_fmt(input_row['channels']['raw_residual']['rudder']['effect_at_perturbation_normalized'], digits=6)} | "
                f"{_fmt(applied['effect_at_perturbation_normalized'], digits=6)} | "
                f"{_fmt(total['effect_at_perturbation_normalized'], digits=6)} | "
                f"{_fmt(applied['share_of_total_median_abs'], digits=6)} | "
                f"{_fmt(applied['sign_consistency'], digits=3)} | "
                f"{_fmt(applied['effect_at_perturbation_physical'], digits=6)} |"
            )
        ratios = arm_payload["jacobian"]["ratios"]
        lines.extend(
            [
                "",
                "| ratio | raw | applied | total |",
                "|---|---:|---:|---:|",
                f"| vertical / cross | {_fmt(ratios['raw_residual']['rudder_vertical_over_cross'])} | "
                f"{_fmt(ratios['applied_residual']['rudder_vertical_over_cross'])} | "
                f"{_fmt(ratios['total_action']['rudder_vertical_over_cross'])} |",
                f"| pitch / heading | {_fmt(ratios['raw_residual']['rudder_pitch_over_heading'])} | "
                f"{_fmt(ratios['applied_residual']['rudder_pitch_over_heading'])} | "
                f"{_fmt(ratios['total_action']['rudder_pitch_over_heading'])} |",
                "",
            ]
        )
    lines.extend(
        [
            "## Reading",
            "",
            "- A high probe R² with a near-zero vertical/pitch Jacobian supports "
            "the hypothesis that the representation exists but the residual head "
            "does not expose it into the action channel.",
            "- A low probe R² supports the alternative that the vertical information "
            "is not available in ``z_t`` and would justify representation/pretraining "
            "changes rather than another head-only pilot.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        help="policy arm to audit; repeatable (default: all *_COMP arms)",
    )
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--max-jacobian-samples", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    manifest = _manifest(root)
    arms = args.arm or sorted(
        str(record["arm"])
        for record in manifest.get("train_runs", [])
        if isinstance(record, dict) and str(record.get("arm", "")).endswith("_COMP")
    )
    if not arms:
        raise ValueError("no COMP policy arms selected")
    seeds = tuple(args.seed_start + index for index in range(args.seed_count))
    if not set(seeds).issubset(set(TRAIN_SEEDS + TEST_SEEDS)):
        raise ValueError(f"probe seeds must stay within {TRAIN_SEEDS + TEST_SEEDS}; got {seeds}")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    scenario = "p-current10__r-legacy_sine__l35__v1p2__g-default"
    payload: dict[str, Any] = {
        "schema": "goal176_residual_probe.v1",
        "root": str(root),
        "scenario": scenario,
        "seeds": list(seeds),
        "train_seeds": list(TRAIN_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "ridge_alpha": RIDGE_ALPHA,
        "arms": {},
    }
    for arm in arms:
        run = _run_record(manifest, arm)
        checkpoint = Path(str(run["output"])) / "final.pt"
        training_config = Path(str(run["training_config"]))
        environment_config = _environment_config(manifest, scenario, arm)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        summary, policy, metadata, episodes, z, targets, observations = _capture_eval(
            training_config=training_config,
            checkpoint=checkpoint,
            environment_config=environment_config,
            seeds=seeds,
            device=device,
            dtype=dtype,
        )
        z_rows, target_rows, observation_rows, seed_rows = _trim_by_seed(
            z, targets, observations, episodes, seeds
        )
        probe = _probe(z_rows, target_rows, seed_rows, alpha=RIDGE_ALPHA)
        sampled_observations, _sample_seeds = _sample_observations(
            observation_rows,
            seed_rows,
            max_samples=args.max_jacobian_samples,
        )
        jacobian = _jacobian(
            policy,
            sampled_observations,
            device=device,
            dtype=dtype,
            normalization=_normalization(metadata),
            action_full_scale=tuple(float(value) for value in metadata["action_full_scale"]),
            alpha=RIDGE_ALPHA,
        )
        payload["arms"][arm] = {
            "checkpoint": str(checkpoint),
            "training_config": str(training_config),
            "environment_config": str(environment_config),
            "checkpoint_sha256": _sha256(checkpoint),
            "capture": {
                "decision_rows": int(z.shape[0]),
                "retained_rows": int(z_rows.shape[0]),
                "episode_steps": [int(episode["steps"]) for episode in episodes],
                "episode_success": [bool(episode.get("success", False)) for episode in episodes],
                "eval_summary_keys": sorted(summary)[:8],
            },
            "probe": probe,
            "jacobian": jacobian,
        }
        print(
            f"{arm}: captured={z_rows.shape[0]} probe_r2="
            + json.dumps(
                {name: result["r2_test"] for name, result in probe["targets"].items()},
                sort_keys=True,
            ),
            flush=True,
        )
    output = root / "goal176_residual_probe.json" if args.output is None else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
