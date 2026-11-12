from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from auv_lowlevel.batch.checkpoint import CHECKPOINT_METADATA_KEY, build_checkpoint_infos
from auv_lowlevel.batch.encoded_policy import DualHorizonEncoder, EncodedActorCritic
from auv_lowlevel.ppo_config import EncoderConfig, load_training_config, parse_training_config
from auv_lowlevel.warmstart import load_encoded_actor_prior

ROOT = Path(__file__).resolve().parents[1]
MLP_CONFIG_PATH = ROOT / "configs/lowlevel/odarrl_control_ppo_i2_26d_encoder_mlp.json"
TRANSFORMER_CONFIG_PATH = (
    ROOT / "configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer.json"
)
TEN_HZ_CONFIG_PATH = (
    ROOT / "configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_10hz.json"
)
PRIVILEGED_CRITIC_CONFIG_PATH = (
    ROOT
    / "configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_hidden_timevarying_privcritic.json"
)
A2_DADE_CONFIG_PATH = (
    ROOT
    / "configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_dade_pose_"
    "lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)


def _encoder_config() -> dict:
    return {
        "kind": "dual_horizon_mlp",
        "feature_dim": 16,
        "fusion_hidden": [32, 16],
        "output_dim": 16,
        "dropout": 0.0,
        "num_layers": 2,
        "num_heads": 4,
        "feedforward_dim": 32,
    }


def _policy(*, kind: str = "dual_horizon_mlp") -> EncodedActorCritic:
    encoder = _encoder_config()
    encoder["kind"] = kind
    if kind == "dual_horizon_transformer":
        encoder["feature_dim"] = 16
        encoder["dropout"] = 0.0
    return EncodedActorCritic(
        26 * 61,
        26 * 61,
        3,
        encoder=encoder,
        history_short_window=10,
        history_long_window=50,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        activation="elu",
        init_noise_std=0.03,
        noise_std_type="scalar",
    )


def test_encoder_configs_parse_and_select_encoded_policy() -> None:
    for path, kind in (
        (MLP_CONFIG_PATH, "dual_horizon_mlp"),
        (TRANSFORMER_CONFIG_PATH, "dual_horizon_transformer"),
    ):
        config = load_training_config(path)
        assert config.encoder is not None
        assert config.encoder.kind == kind
        assert config.policy_kind == kind
        assert config.vector_env_backend == "batch_torch"
        assert config.history is not None
        train_cfg = config.rsl_train_cfg()
        assert train_cfg["policy"]["class_name"] == "EncodedActorCritic"
        assert train_cfg["policy"]["history_short_window"] == 10
        assert train_cfg["policy"]["history_long_window"] == 50
        assert train_cfg["policy"]["encoder"]["kind"] == kind


@pytest.mark.parametrize(
    "kind",
    [
        "dual_horizon_transformer_multiscale",
        "dual_horizon_transformer_uniform40",
    ],
)
def test_multiscale_sampling_kinds_parse_in_training_config(kind: str) -> None:
    raw = json.loads(A2_DADE_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["history"]["long_window"] = 400
    raw["history"]["long_stride"] = 1
    raw["encoder"]["kind"] = kind
    config = parse_training_config(raw)
    assert config.history is not None
    assert config.encoder is not None
    assert config.history.long_window == 400
    assert config.encoder.kind == kind
    assert config.rsl_train_cfg()["policy"]["dade"]["kind"] == kind


def test_history_current_frame_dropout_is_parsed_but_kept_out_of_policy_config() -> None:
    raw = json.loads(TRANSFORMER_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["history"]["current_frame_dropout_prob"] = 0.5
    raw["history"]["current_frame_dropout_mode"] = "side_channel"
    config = parse_training_config(raw)
    assert config.history is not None
    assert config.history.current_frame_dropout_prob == pytest.approx(0.5)
    assert config.history.current_frame_dropout_mode == "side_channel"
    train_cfg = config.rsl_train_cfg()
    assert "current_frame_dropout_prob" not in train_cfg["policy"]

    for invalid in (-0.1, 1.1):
        raw["history"]["current_frame_dropout_prob"] = invalid
        with pytest.raises(ValueError, match="current_frame_dropout_prob"):
            parse_training_config(raw)

    raw["history"]["current_frame_dropout_prob"] = 0.5
    raw["history"]["current_frame_dropout_mode"] = "bad"
    with pytest.raises(ValueError, match="current_frame_dropout_mode"):
        parse_training_config(raw)


def test_privileged_critic_config_selects_independent_encoded_critic() -> None:
    config = load_training_config(PRIVILEGED_CRITIC_CONFIG_PATH)
    assert config.privileged_current_critic is True
    assert config.history is not None
    assert config.encoder is not None
    train_cfg = config.rsl_train_cfg()
    assert train_cfg["policy"]["independent_critic"] is True


def test_encoder_requires_history_and_rejects_bad_head_geometry() -> None:
    raw = json.loads(MLP_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["history"]
    with pytest.raises(ValueError, match="encoder requires a history"):
        parse_training_config(raw)

    raw = json.loads(TRANSFORMER_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["encoder"]["num_heads"] = 5
    with pytest.raises(ValueError, match="divisible"):
        parse_training_config(raw)

    raw = json.loads(TRANSFORMER_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["encoder"]["dropout"] = 1.0
    with pytest.raises(ValueError, match="dropout"):
        parse_training_config(raw)

    raw = json.loads(TRANSFORMER_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["history"]["short_stride"] = 5
    raw["history"]["long_stride"] = 5
    config = parse_training_config(raw)
    assert config.history is not None
    assert config.history.short_span_steps == 50
    assert config.history.long_span_steps == 250

    raw["history"]["short_stride"] = 1
    raw["history"]["long_stride"] = 0
    with pytest.raises(ValueError, match="long_stride"):
        parse_training_config(raw)


def test_ten_hz_history_config_keeps_policy_width_and_extends_spans() -> None:
    baseline = load_training_config(TRANSFORMER_CONFIG_PATH)
    subsampled = load_training_config(TEN_HZ_CONFIG_PATH)
    assert baseline.history is not None
    assert subsampled.history is not None
    assert subsampled.history.short_span_steps == 50
    assert subsampled.history.long_span_steps == 250
    assert subsampled.history.short_window == baseline.history.short_window
    assert subsampled.history.long_window == baseline.history.long_window
    assert (
        subsampled.num_envs
        * subsampled.runner["num_steps_per_env"]
        * subsampled.iterations
        == baseline.num_envs * baseline.runner["num_steps_per_env"] * baseline.iterations
    )


@pytest.mark.parametrize("kind", ["dual_horizon_mlp", "dual_horizon_transformer"])
def test_encoded_policy_exposes_rsl_rl_contract(kind: str) -> None:
    policy = _policy(kind=kind)
    policy.eval()
    observations = torch.randn(3, 26 * 61)

    actions = policy.act(observations)
    assert actions.shape == (3, 3)
    assert torch.isfinite(actions).all()
    assert policy.get_actions_log_prob(actions).shape == (3,)
    assert policy.evaluate(observations).shape == (3, 1)
    assert policy.act_inference(observations).shape == (3, 3)
    assert policy.entropy.shape == (3,)
    assert policy.action_mean.shape == (3, 3)
    assert policy.action_std.shape == (3, 3)

    loaded = _policy(kind=kind)
    assert loaded.load_state_dict(policy.state_dict(), strict=True)
    assert torch.equal(loaded.act_inference(observations), policy.act_inference(observations))


def test_encoded_policy_uses_both_history_windows() -> None:
    policy = _policy()
    policy.eval()
    base = torch.randn(1, 26 * 61)
    short_changed = base.clone()
    long_changed = base.clone()
    short_changed[:, 26 : 26 * 11] += 1.0
    long_changed[:, 26 * 11 :] += 1.0

    base_action = policy.act_inference(base)
    short_action = policy.act_inference(short_changed)
    long_action = policy.act_inference(long_changed)
    assert not torch.allclose(base_action, short_action)
    assert not torch.allclose(base_action, long_action)


@pytest.mark.parametrize(
    "kind",
    [
        "dual_horizon_transformer_attention",
        "dual_horizon_transformer_param_mean",
        "dual_horizon_transformer_attention_normquery",
        "dual_horizon_transformer_param_mean_normquery",
    ],
)
def test_transformer_pooling_variants_accept_current_query(kind: str) -> None:
    config = EncoderConfig(
        kind=kind,
        feature_dim=8,
        fusion_hidden=(16,),
        output_dim=8,
        dropout=0.0,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
    )
    encoder = DualHorizonEncoder(
        observation_dim=6,
        short_window=3,
        long_window=5,
        config=config,
    )
    with torch.no_grad():
        if "attention" in kind:
            encoder.short_encoder.query.weight.normal_()
            encoder.long_encoder.query.weight.normal_()
        else:
            encoder.short_encoder.query.weight.normal_()
            encoder.long_encoder.query.weight.normal_()
    short = torch.randn(2, 3, 6)
    long = torch.randn(2, 5, 6)
    current_a = torch.randn(2, 6)
    current_b = torch.randn(2, 6)
    output_a = encoder(current_a, short, long)
    output_b = encoder(current_b, short, long)
    assert output_a.shape == (2, 8)
    assert torch.isfinite(output_a).all()
    assert not torch.allclose(output_a, output_b)
    if "attention" in kind:
        weights = encoder.short_encoder.last_attention_weights
        assert weights is not None
        assert weights.shape == (2, 3)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


@pytest.mark.parametrize(
    "kind",
    [
        "dual_horizon_transformer_attention_normquery",
        "dual_horizon_transformer_param_mean_normquery",
    ],
)
def test_normquery_variants_have_matched_parameters_and_nonuniform_attention(
    kind: str,
) -> None:
    def make(encoder_kind: str) -> DualHorizonEncoder:
        return DualHorizonEncoder(
            observation_dim=6,
            short_window=3,
            long_window=5,
            config=EncoderConfig(
                kind=encoder_kind,
                feature_dim=8,
                fusion_hidden=(16,),
                output_dim=8,
                dropout=0.0,
                num_layers=1,
                num_heads=2,
                feedforward_dim=16,
            ),
        )

    torch.manual_seed(0)
    encoder = make(kind)
    control_kind = kind.replace("_attention_", "_param_mean_")
    control = make(control_kind)
    assert sum(p.numel() for p in encoder.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    if "_attention_" in kind:
        with torch.no_grad():
            encoder(
                torch.randn(32, 6),
                torch.randn(32, 3, 6),
                torch.randn(32, 5, 6),
            )
        weights = encoder.short_encoder.last_attention_weights
        assert weights is not None
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
        normalized_entropy = entropy / torch.log(torch.tensor(3.0))
        assert normalized_entropy.mean() < 0.99


def test_attention_and_param_mean_have_matched_parameter_counts() -> None:
    def make(kind: str) -> DualHorizonEncoder:
        return DualHorizonEncoder(
            observation_dim=6,
            short_window=3,
            long_window=5,
            config=EncoderConfig(
                kind=kind,
                feature_dim=8,
                fusion_hidden=(16,),
                output_dim=8,
                dropout=0.0,
                num_layers=1,
                num_heads=2,
                feedforward_dim=16,
            ),
        )

    attention = make("dual_horizon_transformer_attention")
    param_mean = make("dual_horizon_transformer_param_mean")
    assert sum(p.numel() for p in attention.parameters()) == sum(
        p.numel() for p in param_mean.parameters()
    )


@pytest.mark.parametrize(
    "kind",
    [
        "dual_horizon_transformer_attention",
        "dual_horizon_transformer_param_mean",
        "dual_horizon_transformer_attention_normquery",
        "dual_horizon_transformer_param_mean_normquery",
    ],
)
def test_encoder_config_parser_accepts_new_pooling_kinds(kind: str) -> None:
    raw = json.loads(TRANSFORMER_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["encoder"]["kind"] = kind
    config = parse_training_config(raw)
    assert config.encoder is not None
    assert config.encoder.kind == kind


def test_encoded_actor_prior_keeps_i1_behavior_and_encoder_gradients(tmp_path) -> None:
    actor = nn.Sequential(
        nn.Linear(26, 16),
        nn.ELU(),
        nn.Linear(16, 8),
        nn.ELU(),
        nn.Linear(8, 3),
    )
    critic = nn.Sequential(
        nn.Linear(26, 16),
        nn.ELU(),
        nn.Linear(16, 8),
        nn.ELU(),
        nn.Linear(8, 1),
    )
    state = {
        **{f"actor.{key}": value for key, value in actor.state_dict().items()},
        **{f"critic.{key}": value for key, value in critic.state_dict().items()},
        "std": torch.full((3,), 0.03),
    }
    checkpoint = {
        "model_state_dict": state,
        "infos": build_checkpoint_infos(
            observation_schema="v2",
            observation_dim=26,
            vector_env_backend="batch_torch",
            physics_dtype="float64",
            training_config_sha256="a" * 64,
        ),
    }
    path = tmp_path / "i1.pt"
    torch.save(checkpoint, path)

    policy = _policy()
    metadata = load_encoded_actor_prior(
        policy,
        path,
        base_observation_dim=26,
        expected_observation_schema="v2",
    )
    current = torch.randn(2, 26)
    flat = torch.cat(
        [current, torch.randn(2, 10 * 26), torch.randn(2, 50 * 26)],
        dim=1,
    )
    policy.eval()
    with torch.inference_mode():
        expected = actor(current)
        actual = policy.act_inference(flat)
    assert torch.allclose(actual, expected, atol=0.05, rtol=0.0)
    assert metadata["critic_warmstarted"] is True
    assert metadata["disturbance_columns"] == 16
    assert metadata["disturbance_columns_init"] == "normal"
    assert metadata["disturbance_columns_std"] == pytest.approx(1e-3)
    assert torch.count_nonzero(policy.actor[0].weight[:, 26:]) > 0
    assert torch.count_nonzero(policy.critic[0].weight[:, 26:]) > 0
    assert torch.equal(policy.actor[0].weight[:, :26], actor[0].weight)

    policy.zero_grad(set_to_none=True)
    policy.act_inference(flat).sum().backward()
    encoder_gradients = [
        parameter.grad
        for parameter in policy.encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_gradients
    assert any(torch.count_nonzero(gradient) > 0 for gradient in encoder_gradients)


def test_encoded_actor_prior_supports_same_layout_checkpoint(tmp_path) -> None:
    source = _policy()
    path = tmp_path / "encoded_actor.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "infos": build_checkpoint_infos(
                observation_schema="v2",
                observation_dim=26,
                vector_env_backend="batch_torch",
                physics_dtype="float64",
                training_config_sha256="a" * 64,
            ),
        },
        path,
    )
    target = _policy()
    metadata = load_encoded_actor_prior(
        target,
        path,
        base_observation_dim=26,
        expected_observation_schema="v2",
    )

    assert metadata["warmstart_mode"] == "same_layout"
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_checkpoint_metadata_round_trips_encoder_config() -> None:
    config = _encoder_config()
    infos = build_checkpoint_infos(
        observation_schema="v2",
        observation_dim=26,
        vector_env_backend="batch_torch",
        physics_dtype="float64",
        training_config_sha256="a" * 64,
        policy_input_dim=26 * 61,
        history_short_window=10,
        history_long_window=50,
        history_short_stride=5,
        history_long_stride=5,
        policy_kind="dual_horizon_mlp",
        policy_config=config,
        critic_mode="privileged_current_independent",
        critic_observation_dim=29,
    )
    metadata = infos[CHECKPOINT_METADATA_KEY]
    assert metadata["policy_kind"] == "dual_horizon_mlp"
    assert metadata["policy_config"] == config
    assert metadata["history_short_stride"] == 5
    assert metadata["history_long_stride"] == 5
    assert metadata["critic_mode"] == "privileged_current_independent"
    assert metadata["critic_observation_dim"] == 29


def test_privileged_current_critic_is_independent_from_actor_features() -> None:
    policy = EncodedActorCritic(
        26 * 61,
        29,
        3,
        encoder=_encoder_config(),
        history_short_window=10,
        history_long_window=50,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        activation="elu",
        init_noise_std=0.03,
        noise_std_type="scalar",
        independent_critic=True,
    )
    policy.eval()
    actor_observations = torch.randn(2, 26 * 61)
    current_a = torch.tensor([[0.3, -0.1, 0.0], [-0.2, 0.25, 0.0]])
    current_b = torch.tensor([[-0.35, 0.2, 0.0], [0.4, -0.15, 0.0]])
    critic_a = torch.cat([actor_observations[:, :26], current_a], dim=1)
    critic_b = torch.cat([actor_observations[:, :26], current_b], dim=1)

    assert policy.critic[0].in_features == 29
    assert not torch.allclose(policy.evaluate(critic_a), policy.evaluate(critic_b))
    actor_action = policy.act_inference(actor_observations)

    def _forbidden_features(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("independent critic must not reuse actor _features()")

    policy._features = _forbidden_features  # type: ignore[method-assign]
    critic_value = policy.evaluate(critic_a)
    assert critic_value.shape == (2, 1)
    assert actor_action.shape == (2, 3)


@pytest.mark.parametrize(
    "kind",
    [
        "dual_horizon_transformer_multiscale",
        "dual_horizon_transformer_uniform40",
    ],
)
def test_long_window_sampling_variants_preserve_raw_history_contract(kind: str) -> None:
    config = EncoderConfig(
        kind=kind,
        feature_dim=8,
        fusion_hidden=(16,),
        output_dim=8,
        dropout=0.0,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
    )
    encoder = DualHorizonEncoder(
        observation_dim=6,
        short_window=10,
        long_window=400,
        config=config,
    )
    assert encoder.short_encoder.sample_indices is None
    indices = encoder.long_encoder.sample_indices
    assert indices is not None
    assert indices.shape == (40,)
    assert torch.unique(indices).numel() == 40
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 389
    assert encoder.long_encoder.position.shape[1] == 40

    short = torch.randn(2, 10, 6)
    long = torch.randn(2, 400, 6)
    current = torch.randn(2, 6)
    output = encoder(current, short, long)
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()


def test_multiscale_and_uniform40_have_matched_parameters() -> None:
    def make(kind: str) -> DualHorizonEncoder:
        return DualHorizonEncoder(
            observation_dim=6,
            short_window=10,
            long_window=400,
            config=EncoderConfig(
                kind=kind,
                feature_dim=8,
                fusion_hidden=(16,),
                output_dim=8,
                dropout=0.0,
                num_layers=1,
                num_heads=2,
                feedforward_dim=16,
            ),
        )

    multiscale = make("dual_horizon_transformer_multiscale")
    uniform = make("dual_horizon_transformer_uniform40")
    multiscale_params = sum(parameter.numel() for parameter in multiscale.parameters())
    uniform_params = sum(parameter.numel() for parameter in uniform.parameters())
    assert multiscale_params == uniform_params

    multiscale_indices = multiscale.long_encoder.sample_indices
    uniform_indices = uniform.long_encoder.sample_indices
    assert multiscale_indices is not None and uniform_indices is not None
    assert not torch.equal(multiscale_indices, uniform_indices)
    assert torch.all(multiscale_indices[1:] > multiscale_indices[:-1])
    assert torch.all(uniform_indices[1:] > uniform_indices[:-1])
