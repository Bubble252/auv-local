from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found, unused-ignore]
from numpy.typing import NDArray

from ..ddpg.networks import PolicyActor


@dataclass(frozen=True)
class SensorUseDiagnosticConfig:
    sector_count: int = 13
    sector_range_scale_m: float = 120.0
    center_shorten_m: float = 20.0
    target_bearing_shift: float = 0.1
    sensitivity_threshold: float = 0.01
    batch_size: int = 4096

    def __post_init__(self) -> None:
        if self.sector_count <= 0:
            raise ValueError("sector_count must be positive")
        if self.sector_range_scale_m <= 0.0:
            raise ValueError("sector_range_scale_m must be positive")
        if self.center_shorten_m < 0.0 or self.center_shorten_m > self.sector_range_scale_m:
            raise ValueError("center_shorten_m must be within the sector range")
        if self.target_bearing_shift <= 0.0:
            raise ValueError("target_bearing_shift must be positive")
        if self.sensitivity_threshold < 0.0:
            raise ValueError("sensitivity_threshold must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def predict_actions(
    actor: PolicyActor,
    observations: NDArray[np.float32],
    *,
    batch_size: int = 4096,
) -> NDArray[np.float64]:
    """Run one deterministic action per sequence and return the last-step action."""

    sequences = np.asarray(observations, dtype=np.float32)
    if sequences.ndim != 3 or sequences.shape[0] == 0 or sequences.shape[1] == 0:
        raise ValueError("observations must have shape (N, T, D) with N,T > 0")
    if not np.all(np.isfinite(sequences)):
        raise ValueError("observations must contain only finite values")
    device = next(actor.parameters()).device
    outputs: list[NDArray[np.float64]] = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, sequences.shape[0], batch_size):
            batch = torch.as_tensor(sequences[start : start + batch_size], device=device)
            action, _ = actor(batch, sample_noise=False)
            outputs.append(action[:, -1].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs, axis=0)


def perturb_sensor_observation(
    observations: NDArray[np.float32],
    *,
    kind: str,
    config: SensorUseDiagnosticConfig,
    sector_index: int | None = None,
) -> NDArray[np.float32]:
    """Perturb only the final observation frame, leaving history unchanged."""

    result = np.asarray(observations, dtype=np.float32).copy()
    if result.ndim != 3 or result.shape[2] < config.sector_count + 4:
        raise ValueError("observations must contain the 17-D P1 observation")
    final = result[:, -1]
    target_bearing_index = config.sector_count + 1
    if kind == "all_sectors_saturated":
        final[:, : config.sector_count] = 1.0
    elif kind == "all_sectors_zero":
        final[:, : config.sector_count] = 0.0
    elif kind == "center_shortened":
        center = config.sector_count // 2
        final[:, center] = np.maximum(
            0.0,
            final[:, center] - config.center_shorten_m / config.sector_range_scale_m,
        )
    elif kind == "single_sector_zero":
        if sector_index is None or not 0 <= sector_index < config.sector_count:
            raise ValueError("single_sector_zero requires a valid sector_index")
        final[:, sector_index] = 0.0
    elif kind == "target_bearing_zero":
        final[:, target_bearing_index] = 0.0
    elif kind == "target_bearing_shifted":
        final[:, target_bearing_index] = np.clip(
            final[:, target_bearing_index] + config.target_bearing_shift,
            -1.0,
            1.0,
        )
    else:
        raise ValueError(f"unknown perturbation kind: {kind!r}")
    return result


def _variant_metrics(
    baseline: NDArray[np.float64],
    perturbed: NDArray[np.float64],
    target: NDArray[np.float32],
    *,
    threshold: float,
) -> dict[str, Any]:
    delta = np.linalg.norm(perturbed - baseline, axis=1)
    mse = np.mean((perturbed - target) ** 2, axis=1)
    return {
        "mean_action_delta_l2": float(np.mean(delta)),
        "p95_action_delta_l2": float(np.percentile(delta, 95)),
        "fraction_delta_at_least_threshold": float(np.mean(delta >= threshold)),
        "mean_action_mse_to_teacher": float(np.mean(mse)),
        "p95_action_mse_to_teacher": float(np.percentile(mse, 95)),
    }


def diagnose_sensor_use(
    actor: PolicyActor,
    observations: NDArray[np.float32],
    teacher_actions: NDArray[np.float32],
    scenario_keys: NDArray[np.str_],
    *,
    config: SensorUseDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Measure action reliance on sonar and target features on held-out samples."""

    cfg = SensorUseDiagnosticConfig() if config is None else config
    obs = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(teacher_actions, dtype=np.float32)
    keys = np.asarray(scenario_keys).astype(str)
    if obs.ndim != 3 or actions.shape != (obs.shape[0], 2):
        raise ValueError("observations and teacher actions have incompatible shapes")
    if keys.shape != (obs.shape[0],):
        raise ValueError("scenario_keys must have one entry per sample")

    baseline = predict_actions(actor, obs, batch_size=cfg.batch_size)
    variants: dict[str, NDArray[np.float64]] = {}
    for kind in (
        "all_sectors_saturated",
        "all_sectors_zero",
        "center_shortened",
        "target_bearing_zero",
        "target_bearing_shifted",
    ):
        perturbed = perturb_sensor_observation(obs, kind=kind, config=cfg)
        variants[kind] = predict_actions(actor, perturbed, batch_size=cfg.batch_size)

    scenarios = sorted(set(keys.tolist()))
    by_scenario: dict[str, Any] = {}
    for scenario_key in scenarios:
        mask = keys == scenario_key
        variant_metrics: dict[str, dict[str, Any]] = {
            kind: _variant_metrics(
                baseline[mask],
                values[mask],
                actions[mask],
                threshold=cfg.sensitivity_threshold,
            )
            for kind, values in variants.items()
        }
        metrics: dict[str, Any] = {
            "samples": int(np.count_nonzero(mask)),
            "baseline_mse_to_teacher": float(
                np.mean((baseline[mask] - actions[mask].astype(np.float64)) ** 2)
            ),
            "variants": variant_metrics,
        }
        sector_ratio = float(variant_metrics["all_sectors_saturated"]["mean_action_delta_l2"])
        target_ratio = float(variant_metrics["target_bearing_shifted"]["mean_action_delta_l2"])
        metrics["sonar_to_target_reliance_ratio"] = float(sector_ratio / max(target_ratio, 1e-12))
        per_sector = []
        for sector_index in range(cfg.sector_count):
            perturbed = perturb_sensor_observation(
                obs,
                kind="single_sector_zero",
                config=cfg,
                sector_index=sector_index,
            )
            sector_actions = predict_actions(actor, perturbed, batch_size=cfg.batch_size)
            delta = np.linalg.norm(sector_actions[mask] - baseline[mask], axis=1)
            per_sector.append(
                {
                    "sector_index": sector_index,
                    "mean_action_delta_l2": float(np.mean(delta)),
                    "fraction_delta_at_least_threshold": float(
                        np.mean(delta >= cfg.sensitivity_threshold)
                    ),
                }
            )
        metrics["per_sector_zero_echo"] = per_sector
        by_scenario[scenario_key] = metrics

    return {
        "schema": "auv_p1_bc_sensor_use_diagnostic.v1",
        "config": {
            "sector_count": cfg.sector_count,
            "sector_range_scale_m": cfg.sector_range_scale_m,
            "center_shorten_m": cfg.center_shorten_m,
            "target_bearing_shift": cfg.target_bearing_shift,
            "sensitivity_threshold": cfg.sensitivity_threshold,
        },
        "samples": int(obs.shape[0]),
        "by_scenario": by_scenario,
        "all_sectors_saturated_delta_below_threshold": all(
            by_scenario[key]["variants"]["all_sectors_saturated"]["mean_action_delta_l2"]
            < cfg.sensitivity_threshold
            for key in scenarios
        ),
    }


__all__ = [
    "SensorUseDiagnosticConfig",
    "diagnose_sensor_use",
    "perturb_sensor_observation",
    "predict_actions",
]
