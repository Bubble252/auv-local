from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = WORKSPACE_ROOT / "third_party" / "stable-baselines3"
if SB3_ROOT.is_dir() and str(SB3_ROOT) not in sys.path:
    sys.path.insert(0, str(SB3_ROOT))

pytest.importorskip("stable_baselines3")

from stable_baselines3 import SAC  # noqa: E402

from auv_lowlevel.batch.pc_bc import read_pc_bc_checkpoint_metadata  # noqa: E402
from auv_rl.sac.pc_features import (  # noqa: E402
    PC_DADE_FEATURE_DIM,
    PCStatePathFeatureExtractor,
    initialize_sac_head_from_bc,
    zero_initialize_sac_residual,
)
from auv_rl.sac.pc_vec_env import PCHistoryVecEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
D3_CHECKPOINT = REPO_ROOT / "checkpoints" / "goal26" / "d3" / "pc_bc_actor.pt"


class _FakePCBatchEnv:
    num_envs = 2
    observation_dim = 30
    action_dim = 3
    render_mode = None

    def reset(self):
        observations = np.linspace(-0.5, 0.5, self.num_envs * self.observation_dim)
        observations = observations.reshape(self.num_envs, self.observation_dim).astype(np.float32)
        return observations, {"episode_id": np.array([0, 1])}

    def step(self, actions):
        observations, _ = self.reset()
        return (
            observations,
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=bool),
            np.zeros(self.num_envs, dtype=bool),
            {
                "final_observations": observations.copy(),
                "episode_id": np.array([2, 1]),
            },
        )

    def drain_completed_infos(self):
        return []

    def set_action_mode(self, mode: str) -> None:
        self.mode = mode

    def set_path_zero(self, enabled: bool) -> None:
        self.path_zero = enabled


def _extractor(*, path_dim: int = 12, bc_path_dim: int = 8, history_length: int = 3):
    flat_dim = 12 + path_dim + 3 + 3
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(history_length * flat_dim,),
        dtype=np.float32,
    )
    return PCStatePathFeatureExtractor(
        observation_space,
        flat_observation_dim=flat_dim,
        state_dim=12,
        path_dim=path_dim,
        bc_path_dim=bc_path_dim,
        command_dim=3,
        current_dim=3,
        history_length=history_length,
        d_z_total=64,
        n_in_reserved=8,
        n_z_reserved=4,
        state_encoder_hidden_dims=(128, 128),
        path_encoder_hidden_dims=(64,),
        endpoint_encoder_hidden_dims=(32,),
        current_encoder_hidden_dims=(64,),
        activation="elu",
    )


def test_extractor_encodes_last_frame_and_uses_bc_dade_command_layout() -> None:
    extractor = _extractor()
    flat_dim = 30
    frames = np.zeros((2, 3, flat_dim), dtype=np.float32)
    frames[:, 0, :] = 99.0
    frames[:, 1, :] = -99.0
    last = frames[:, -1]
    last[:, :12] = np.linspace(-1.0, 1.0, 24).reshape(2, 12)
    last[:, 12:24] = np.linspace(-2.0, 2.0, 24).reshape(2, 12)
    last[:, 24:27] = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float32)
    last[:, 27:30] = np.array([[0.4, 0.5, 0.6], [-0.4, -0.5, -0.6]], dtype=np.float32)

    observations = torch.from_numpy(frames.reshape(2, -1))
    features = extractor(observations)

    assert features.shape == (2, 64 + PC_DADE_FEATURE_DIM + 3)
    assert torch.count_nonzero(features[:, 64:128]) == 0
    assert torch.equal(features[:, 128:], observations[:, -6:-3])

    state = observations[:, -30:-18]
    path = observations[:, -18:-6]
    current = observations[:, -3:]
    expected_z = (
        extractor.actor_encoder(state)
        + extractor.path_encoder(path[:, :8])
        + extractor.endpoint_encoder(path[:, 8:])
        + extractor.current_encoder(current)
    )
    assert torch.allclose(features[:, :64], expected_z)


def test_extractor_rejects_mismatched_flat_width() -> None:
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(90,), dtype=np.float32)
    with pytest.raises(ValueError, match="flat observation width"):
        PCStatePathFeatureExtractor(
            observation_space,
            flat_observation_dim=29,
            state_dim=12,
            path_dim=12,
            bc_path_dim=8,
            command_dim=3,
            current_dim=3,
            history_length=3,
        )


@pytest.mark.skipif(not D3_CHECKPOINT.is_file(), reason="D3 BC checkpoint is unavailable")
def test_d3_encoder_loads_only_the_canonical_path_prefix() -> None:
    checkpoint = read_pc_bc_checkpoint_metadata(D3_CHECKPOINT)
    extractor = _extractor()
    report = extractor.load_bc_encoder(D3_CHECKPOINT)

    assert report["encoder_loaded"]
    assert report["path_encoder_loaded"]
    assert report["bc_path_dim"] == 8
    assert report["observation_path_dim"] == 12
    assert report["endpoint_dim"] == 4
    assert report["endpoint_encoder_zero_initialized"]
    assert torch.equal(
        extractor.actor_encoder.network[0].weight,
        checkpoint["state_dict"]["actor_encoder.network.0.weight"],
    )
    assert torch.equal(
        extractor.path_encoder[0].weight,
        checkpoint["state_dict"]["path_encoder.0.weight"],
    )
    endpoint_linear = [
        layer
        for layer in extractor.endpoint_encoder.modules()
        if isinstance(layer, torch.nn.Linear)
    ][-1]
    assert torch.count_nonzero(endpoint_linear.weight) == 0
    assert torch.count_nonzero(endpoint_linear.bias) == 0


@pytest.mark.skipif(not D3_CHECKPOINT.is_file(), reason="D3 BC checkpoint is unavailable")
def test_d3_encoder_rejects_path_width_mismatch() -> None:
    extractor = _extractor(path_dim=12, bc_path_dim=12)
    with pytest.raises(ValueError, match="BC path width does not match bc_path_dim"):
        extractor.load_bc_encoder(D3_CHECKPOINT)


@pytest.mark.skipif(not D3_CHECKPOINT.is_file(), reason="D3 BC checkpoint is unavailable")
def test_sac_bc_initialisation_is_exact_zero_residual() -> None:
    vec_env = PCHistoryVecEnv(_FakePCBatchEnv(), history_length=3)  # type: ignore[arg-type]
    extractor_kwargs = {
        "flat_observation_dim": 30,
        "state_dim": 12,
        "path_dim": 12,
        "bc_path_dim": 8,
        "command_dim": 3,
        "current_dim": 3,
        "history_length": 3,
        "d_z_total": 64,
        "n_in_reserved": 8,
        "n_z_reserved": 4,
        "state_encoder_hidden_dims": (128, 128),
        "path_encoder_hidden_dims": (64,),
        "endpoint_encoder_hidden_dims": (32,),
        "current_encoder_hidden_dims": (64,),
        "activation": "elu",
    }
    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs={
            "net_arch": {"pi": [256, 128], "qf": [256, 128]},
            "activation_fn": torch.nn.ELU,
            "features_extractor_class": PCStatePathFeatureExtractor,
            "features_extractor_kwargs": extractor_kwargs,
            "share_features_extractor": False,
            "log_std_init": -3.0,
        },
        device="cpu",
        seed=0,
    )
    model.actor.features_extractor.load_bc_encoder(D3_CHECKPOINT)
    report = initialize_sac_head_from_bc(model, D3_CHECKPOINT, zero_residual=True)

    assert report["head_loaded"]
    assert report["command_columns_zero_initialized"] == 3
    assert torch.count_nonzero(model.actor.mu.weight) == 0
    assert torch.count_nonzero(model.actor.mu.bias) == 0
    first_weight = model.actor.latent_pi[0].weight.detach()
    assert torch.count_nonzero(first_weight[:, 128:]) == 0
    checkpoint = read_pc_bc_checkpoint_metadata(D3_CHECKPOINT)
    assert torch.equal(first_weight[:, :128], checkpoint["state_dict"]["actor.0.weight"])

    observation = vec_env.reset()
    action, _ = model.predict(observation, deterministic=True)
    assert np.allclose(action, 0.0, atol=1e-7)


def test_zero_initialisation_starts_from_zero_residual() -> None:
    vec_env = PCHistoryVecEnv(_FakePCBatchEnv(), history_length=3)  # type: ignore[arg-type]
    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs={
            "net_arch": {"pi": [64, 32], "qf": [64, 32]},
            "features_extractor_class": PCStatePathFeatureExtractor,
            "features_extractor_kwargs": {
                "flat_observation_dim": 30,
                "state_dim": 12,
                "path_dim": 12,
                "bc_path_dim": 12,
                "command_dim": 3,
                "current_dim": 3,
                "history_length": 3,
            },
        },
        device="cpu",
        seed=0,
    )
    report = zero_initialize_sac_residual(model, log_std_init=-4.0)
    assert report["mu_zeroed"]
    observation = vec_env.reset()
    action, _ = model.predict(observation, deterministic=True)
    assert np.allclose(action, 0.0, atol=1e-7)
