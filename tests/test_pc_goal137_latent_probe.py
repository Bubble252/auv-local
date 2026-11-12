from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from auv_lowlevel.batch.pc_policy import PCReservedSlotEncoder

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "artifacts"
    / "lowlevel"
    / "goal86"
    / "pa_residual_bc_path12_cmd3_current_20261110"
    / "pc_bc_actor.pt"
)
DATASET = (
    ROOT
    / "artifacts"
    / "goal86"
    / "pa_residual_labels_20261110"
    / "pa_residual_labels.npz"
)


def _load_module() -> ModuleType:
    path = ROOT / "scripts" / "probe_pc_latent_vertical.py"
    spec = importlib.util.spec_from_file_location("probe_pc_latent_vertical_goal137", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_episode_split_keeps_episodes_disjoint_and_matches_checkpoint_contract() -> None:
    module = _load_module()
    episode_id = np.repeat(np.arange(32, dtype=np.int64), 3)
    train, validation = module.episode_split(episode_id, 0.2, 86051)
    assert np.intersect1d(episode_id[train], episode_id[validation]).size == 0
    assert np.unique(episode_id[train]).size == 26
    assert np.unique(episode_id[validation]).size == 6


def test_ridge_probe_recovers_exact_linear_signal() -> None:
    module = _load_module()
    rng = np.random.default_rng(137)
    x = rng.normal(size=(500, 4))
    features = np.column_stack([x, x[:, 0] ** 2])
    targets = {
        "a": 2.5 * x[:, 0] - 0.75 * x[:, 1],
        "b": -x[:, 2] + 0.25,
    }
    train = np.arange(0, 400)
    validation = np.arange(400, 500)
    result = module.ridge_probe(features, targets, train, validation, alpha=1e-8)
    assert result["a"]["r2"] > 0.999
    assert result["b"]["r2"] > 0.999
    assert abs(result["a"]["pearson"] - 1.0) < 1e-6


def test_checkpoint_reconstruction_matches_model_forward() -> None:
    module = _load_module()
    device = torch.device("cpu")
    checkpoint, model = module.load_checkpoint_and_model(CHECKPOINT, device=device)
    arrays = module.load_arrays(DATASET, max_samples=16)
    tensors = module.prepare_tensors(
        arrays,
        model,
        action_full_scale=module._full_scale(checkpoint),
        device=device,
    )
    raw_state = tensors["raw_state"][:8]
    raw_path = tensors["raw_path"][:8]
    current = tensors["current"][:8]
    state = tensors["state"][:8]
    path = tensors["path"][:8]
    command = tensors["command"][:8]
    assert isinstance(raw_state, torch.Tensor)
    assert isinstance(raw_path, torch.Tensor)
    assert isinstance(current, torch.Tensor)
    assert isinstance(state, torch.Tensor)
    assert isinstance(path, torch.Tensor)
    assert isinstance(command, torch.Tensor)

    latents = module.encode_latents(model, {**tensors, "state": state, "path": path, "current": current})
    expected_z = model.actor_encoder(state)
    if model.path_encoder is not None:
        expected_z = expected_z + model.path_encoder_alpha * model.path_encoder(path)
    if model.actor_current_encoder is not None:
        expected_z = (
            expected_z
            + model.actor_current_encoder_alpha * model.actor_current_encoder(current)
        )
    assert torch.allclose(latents["z_t"], expected_z, atol=1e-6, rtol=1e-6)

    ours = module.predict_residual(model, state, path, current, command)
    expected = model.decode_action(model.forward_raw(state, command, path, current))
    assert torch.allclose(ours, expected, atol=1e-7, rtol=1e-6)
    assert checkpoint["d_z_total"] == 64
    assert isinstance(model.actor_encoder, PCReservedSlotEncoder)


def test_jacobian_summary_is_finite_and_has_expected_shape() -> None:
    module = _load_module()
    device = torch.device("cpu")
    checkpoint, model = module.load_checkpoint_and_model(CHECKPOINT, device=device)
    arrays = module.load_arrays(DATASET, max_samples=64)
    tensors = module.prepare_tensors(
        arrays,
        model,
        action_full_scale=module._full_scale(checkpoint),
        device=device,
    )
    indices = np.arange(32, dtype=np.int64)
    result = module.jacobian_summary(
        model,
        tensors,
        indices,
        action_full_scale=module._full_scale(checkpoint),
        max_samples=16,
    )
    assert result["sample_count"] == 16
    assert set(result["inputs"]) == {
        "vertical_error",
        "tangent_pitch_error",
        "theta",
        "q",
        "w",
        "cross_track",
        "heading_error",
        "phi",
        "r",
    }
    for item in result["inputs"].values():
        assert len(item["jacobian_abs_mean_normalized"]) == 3
        assert len(item["jacobian_abs_mean_physical"]) == 3
        assert np.isfinite(item["response_per_std"])
        assert item["response_per_std"] >= 0.0
