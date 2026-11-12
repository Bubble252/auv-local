"""Dual-horizon encoded actor-critic policies for DADE experiments.

The policy consumes the same flattened ``[current, short, long]`` observation
layout as :class:`BatchTorchHistoryVecEnv`. Unlike the explicit history-stack
baseline, the two history windows are encoded into a learned disturbance
feature before reaching the residual actor. This module deliberately does not
modify rsl_rl; its policy implements the small API surface PPO needs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from auv_lowlevel.ppo_config import EncoderConfig

from .history import HistoryLayout

ENCODED_POLICY_KINDS = frozenset(
    {
        "dual_horizon_mlp",
        "dual_horizon_transformer",
        "dual_horizon_transformer_attention",
        "dual_horizon_transformer_param_mean",
        "dual_horizon_transformer_attention_normquery",
        "dual_horizon_transformer_param_mean_normquery",
        "dual_horizon_transformer_multiscale",
        "dual_horizon_transformer_uniform40",
    }
)
TRANSFORMER_KINDS = frozenset(
    {
        "dual_horizon_transformer",
        "dual_horizon_transformer_attention",
        "dual_horizon_transformer_param_mean",
        "dual_horizon_transformer_attention_normquery",
        "dual_horizon_transformer_param_mean_normquery",
        "dual_horizon_transformer_multiscale",
        "dual_horizon_transformer_uniform40",
    }
)
ATTENTION_KINDS = frozenset(
    {
        "dual_horizon_transformer_attention",
        "dual_horizon_transformer_attention_normquery",
    }
)
PARAM_MEAN_KINDS = frozenset(
    {
        "dual_horizon_transformer_param_mean",
        "dual_horizon_transformer_param_mean_normquery",
    }
)
NORM_QUERY_KINDS = frozenset(
    {
        "dual_horizon_transformer_attention_normquery",
        "dual_horizon_transformer_param_mean_normquery",
    }
)
MULTISCALE_KINDS = frozenset(
    {
        "dual_horizon_transformer_multiscale",
        "dual_horizon_transformer_uniform40",
    }
)


def _long_window_sample_indices(
    kind: str,
    window: int,
    *,
    enabled: bool,
) -> Tensor | None:
    """Return deterministic raw-history indices for a 50 Hz 8 s long window.

    The policy still receives the full 400-step long history.  These variants
    only change which 40 long-window tokens reach the Transformer:

    - multiscale: 20 mid tokens over 0.2--2 s plus 20 long tokens over 2--8 s;
    - uniform40: 40 uniformly spaced tokens over 0.2--8 s.

    The short window remains the 10 most recent tokens.  Keeping the raw input
    width fixed avoids changing the environment observation contract.
    """

    if not enabled or kind not in MULTISCALE_KINDS:
        return None
    if window != 400:
        raise ValueError(f"{kind} requires a 400-step long window")
    if kind == "dual_horizon_transformer_multiscale":
        long_indices = torch.linspace(0, 299, 20).round().long()
        mid_indices = torch.linspace(300, 389, 20).round().long()
        return torch.cat([long_indices, mid_indices], dim=0)
    return torch.linspace(0, 389, 40).round().long()


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation: {name!r}")


def _coerce_encoder_config(value: EncoderConfig | Mapping[str, Any]) -> EncoderConfig:
    if isinstance(value, EncoderConfig):
        config = value
    elif isinstance(value, Mapping):
        try:
            config = EncoderConfig(
                kind=str(value["kind"]),
                feature_dim=int(value["feature_dim"]),
                fusion_hidden=tuple(int(item) for item in value["fusion_hidden"]),
                output_dim=int(value["output_dim"]),
                dropout=float(value["dropout"]),
                num_layers=int(value["num_layers"]),
                num_heads=int(value["num_heads"]),
                feedforward_dim=int(value["feedforward_dim"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid encoder configuration") from error
    else:
        raise ValueError("encoder configuration must be an EncoderConfig or mapping")
    if config.kind not in ENCODED_POLICY_KINDS:
        raise ValueError(f"unsupported encoder kind: {config.kind!r}")
    if config.feature_dim <= 0 or config.output_dim <= 0:
        raise ValueError("encoder dimensions must be positive")
    if not config.fusion_hidden or any(item <= 0 for item in config.fusion_hidden):
        raise ValueError("encoder.fusion_hidden must contain positive integers")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("encoder.dropout must lie in [0, 1)")
    if config.num_layers <= 0 or config.num_heads <= 0 or config.feedforward_dim <= 0:
        raise ValueError("encoder layer/head/feedforward dimensions must be positive")
    if config.kind in TRANSFORMER_KINDS and config.feature_dim % config.num_heads:
        raise ValueError("encoder.feature_dim must be divisible by encoder.num_heads")
    return config


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for index, hidden_dim in enumerate(hidden_dims[1:], start=1):
        layers.append(nn.Linear(hidden_dims[index - 1], hidden_dim))
        layers.append(activation)
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)


class _WindowEncoder(nn.Module):
    """Encode one fixed-length history window into a pooled feature."""

    def __init__(
        self,
        *,
        kind: str,
        observation_dim: int,
        window: int,
        config: EncoderConfig,
        sample_long_window: bool = False,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.window = window
        sample_indices = _long_window_sample_indices(
            kind,
            window,
            enabled=sample_long_window,
        )
        self.token_window = (
            int(sample_indices.numel())
            if sample_indices is not None
            else window
        )
        self.register_buffer("sample_indices", sample_indices, persistent=False)
        self.observation_dim = observation_dim
        self.pooling = "mean"
        if kind == "dual_horizon_mlp":
            self.encoder: nn.Module = nn.Sequential(
                nn.Linear(observation_dim * window, config.feature_dim),
                nn.ELU(),
                nn.Linear(config.feature_dim, config.feature_dim),
                nn.ELU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.input = nn.Linear(observation_dim, config.feature_dim)
            self.position = nn.Parameter(
                torch.zeros(1, self.token_window, config.feature_dim)
            )
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=config.feature_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.norm = nn.LayerNorm(config.feature_dim)
            if kind in ATTENTION_KINDS:
                self.pooling = "attention"
            elif kind in PARAM_MEAN_KINDS:
                self.pooling = "param_mean"
            if self.pooling != "mean":
                self.query = nn.Linear(observation_dim, config.feature_dim)
                self.norm_query = kind in NORM_QUERY_KINDS
                if not self.norm_query:
                    # Starting from mean pooling is intentional for the legacy
                    # zero-query arms. The norm-query variants instead keep the
                    # default random initialization so attention logits are not
                    # born exactly uniform.
                    nn.init.zeros_(self.query.weight)
                    nn.init.zeros_(self.query.bias)
            else:
                self.norm_query = False
        self.last_attention_weights: Tensor | None = None

    def forward(self, history: Tensor, current: Tensor | None = None) -> Tensor:
        if history.ndim != 3 or history.shape[1] != self.window:
            raise ValueError("history window has the wrong shape")
        if self.sample_indices is not None:
            history = history.index_select(1, self.sample_indices)
        if history.shape[1] != self.token_window:
            raise ValueError("sampled history has the wrong token count")
        if self.kind == "dual_horizon_mlp":
            return self.encoder(history.flatten(start_dim=1))
        else:
            encoded = self.encoder(self.input(history) + self.position)
            encoded = self.norm(encoded)
        if self.pooling == "mean":
            return encoded.mean(dim=1)
        if current is None:
            raise ValueError("attention pooling requires a current query")
        if current.ndim != 2 or current.shape[1] != self.observation_dim:
            raise ValueError("current query has the wrong shape")
        query = self.query(current)
        if self.norm_query:
            query = self.norm(query)
        if self.pooling == "attention":
            scale = float(encoded.shape[-1]) ** 0.5
            weights = torch.softmax(
                torch.einsum("bwd,bd->bw", encoded, query) / scale,
                dim=1,
            )
            self.last_attention_weights = weights.detach()
            pooled = torch.einsum("bw,bwd->bd", weights, encoded)
            return pooled + query
        pooled = encoded.mean(dim=1)
        return pooled + query


class DualHorizonEncoder(nn.Module):
    """Two separate window encoders followed by a fusion MLP."""

    def __init__(
        self,
        *,
        observation_dim: int,
        short_window: int,
        long_window: int,
        config: EncoderConfig,
    ) -> None:
        super().__init__()
        self.layout = HistoryLayout(
            observation_dim=observation_dim,
            short_window=short_window,
            long_window=long_window,
        )
        self.config = config
        self.short_encoder = _WindowEncoder(
            kind=config.kind,
            observation_dim=observation_dim,
            window=short_window,
            config=config,
        )
        self.long_encoder = _WindowEncoder(
            kind=config.kind,
            observation_dim=observation_dim,
            window=long_window,
            config=config,
            sample_long_window=True,
        )
        fusion_layers: list[nn.Module] = []
        input_dim = 2 * config.feature_dim
        for hidden_dim in config.fusion_hidden:
            fusion_layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(config.dropout)]
            )
            input_dim = hidden_dim
        fusion_layers.append(nn.Linear(input_dim, config.output_dim))
        fusion_layers.append(nn.LayerNorm(config.output_dim))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, current: Tensor, short: Tensor, long: Tensor) -> Tensor:
        short_feature = self.short_encoder(short, current)
        long_feature = self.long_encoder(long, current)
        fused = torch.cat(
            [short_feature, long_feature],
            dim=-1,
        )
        return self.fusion(fused)


class EncodedActorCritic(nn.Module):
    """rsl_rl-compatible actor-critic with a DADE-style history encoder."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        encoder: EncoderConfig | Mapping[str, Any],
        history_short_window: int,
        history_long_window: int,
        actor_hidden_dims: Sequence[int] = (256, 128),
        critic_hidden_dims: Sequence[int] = (256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        independent_critic: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"unexpected EncodedActorCritic arguments: {sorted(kwargs)}")
        self.independent_critic = bool(independent_critic)
        if num_actor_obs <= 0 or num_critic_obs <= 0:
            raise ValueError("actor and critic observation widths must be positive")
        if not self.independent_critic and num_actor_obs != num_critic_obs:
            raise ValueError(
                "shared-feature encoded actor and critic observations must have the same width"
            )
        if history_short_window <= 0 or history_long_window <= 0:
            raise ValueError("history windows must be positive")
        if history_short_window > history_long_window:
            raise ValueError("history_short_window must not exceed history_long_window")
        window_count = 1 + history_short_window + history_long_window
        if num_actor_obs % window_count != 0:
            raise ValueError("observation width is inconsistent with history windows")
        self.observation_dim = num_actor_obs // window_count
        self.layout = HistoryLayout(
            observation_dim=self.observation_dim,
            short_window=history_short_window,
            long_window=history_long_window,
        )
        self.encoder_config = _coerce_encoder_config(encoder)
        self.policy_kind = self.encoder_config.kind
        activation_module = _activation(activation)
        self.encoder = DualHorizonEncoder(
            observation_dim=self.observation_dim,
            short_window=history_short_window,
            long_window=history_long_window,
            config=self.encoder_config,
        )
        actor_input_dim = self.observation_dim + self.encoder_config.output_dim
        self.actor = _mlp(
            actor_input_dim,
            tuple(int(item) for item in actor_hidden_dims),
            num_actions,
            activation_module,
        )
        self.critic_input_dim = (
            num_critic_obs if self.independent_critic else actor_input_dim
        )
        self.critic = _mlp(
            self.critic_input_dim,
            tuple(int(item) for item in critic_hidden_dims),
            1,
            activation_module,
        )
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: Tensor | None = None) -> None:
        del dones

    def _features(self, observations: Tensor) -> Tensor:
        current, short, long = self.layout.split(observations)
        disturbance = self.encoder(current, short, long)
        return torch.cat([current, disturbance], dim=-1)

    @property
    def action_mean(self) -> Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: Tensor) -> None:
        mean = self.actor(self._features(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: Tensor, **kwargs: Any) -> Tensor:
        del kwargs
        self.update_distribution(observations)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: Tensor) -> Tensor:
        return self.actor(self._features(observations))

    def evaluate(self, critic_observations: Tensor, **kwargs: Any) -> Tensor:
        del kwargs
        if self.independent_critic:
            if critic_observations.ndim != 2 or critic_observations.shape[1] != self.critic_input_dim:
                raise ValueError(
                    "critic observations have the wrong width: "
                    f"expected {self.critic_input_dim}, got {tuple(critic_observations.shape)}"
                )
            return self.critic(critic_observations)
        return self.critic(self._features(critic_observations))

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


__all__ = [
    "ENCODED_POLICY_KINDS",
    "DualHorizonEncoder",
    "EncodedActorCritic",
]
