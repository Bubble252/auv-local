from __future__ import annotations

import math
from collections.abc import Sequence

import torch  # type: ignore[import-not-found, unused-ignore]
from torch import nn  # type: ignore[import-not-found, unused-ignore]


class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian parameter noise.

    ``sample_noise=False`` gives the mean-weight network used for evaluation.
    The layer still owns the learnable ``rho`` parameters when noise is enabled,
    so the checkpoint round-trips the complete NoisyNet state.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_noisy: bool = True,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not math.isfinite(sigma_init) or sigma_init <= 0.0:
            raise ValueError("sigma_init must be finite and positive")
        self.in_features = in_features
        self.out_features = out_features
        self.use_noisy = bool(use_noisy)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters(sigma_init)

    def reset_parameters(self, sigma_init: float = 0.5) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        rho = math.log(math.expm1(sigma_init))
        nn.init.constant_(self.weight_rho, rho)
        nn.init.constant_(self.bias_rho, rho)

    @staticmethod
    def _scaled_gaussian_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def _sample_noise(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample factorized Gaussian noise for a weight or bias tensor.

        NoisyNet factorizes the matrix noise as an outer product of independent
        output/input noise vectors.  Sampling every matrix element independently
        is not the same distribution and breaks the intended exploration
        structure.
        """

        if len(shape) == 1:
            return self._scaled_gaussian_noise(shape[0], device)
        if len(shape) == 2:
            output_noise = self._scaled_gaussian_noise(shape[0], device)
            input_noise = self._scaled_gaussian_noise(shape[1], device)
            return output_noise.unsqueeze(1) * input_noise.unsqueeze(0)
        raise ValueError("noise shape must be one- or two-dimensional")

    def forward(self, input_tensor: torch.Tensor, *, sample_noise: bool = True) -> torch.Tensor:
        if self.use_noisy and sample_noise:
            weight_sigma = torch.nn.functional.softplus(self.weight_rho)
            bias_sigma = torch.nn.functional.softplus(self.bias_rho)
            weight_epsilon = self._sample_noise(self.weight_mu.shape, self.weight_mu.device)
            bias_epsilon = self._sample_noise(self.bias_mu.shape, self.bias_mu.device)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(input_tensor, weight, bias)


def _make_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    *,
    use_noisy: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        layers.append(NoisyLinear(previous, width, use_noisy=use_noisy))
        layers.append(nn.ReLU())
        previous = width
    layers.append(NoisyLinear(previous, output_dim, use_noisy=use_noisy))
    return nn.Sequential(*layers)


class _SequenceHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        *,
        use_gru: bool,
        gru_hidden_size: int,
        use_noisy: bool,
    ) -> None:
        super().__init__()
        self.use_gru = bool(use_gru)
        if self.use_gru:
            if gru_hidden_size <= 0:
                raise ValueError("gru_hidden_size must be positive")
            self.gru: nn.GRU | None = nn.GRU(
                input_dim,
                gru_hidden_size,
                batch_first=True,
            )
            head_input = gru_hidden_size
        else:
            self.gru = None
            head_input = input_dim
        self.head = _make_mlp(head_input, hidden_sizes, output_dim, use_noisy=use_noisy)
        self.gru_hidden_size = gru_hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        sample_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if inputs.ndim not in (2, 3):
            raise ValueError("inputs must have shape (B,D) or (B,T,D)")
        unbatched_time = inputs.ndim == 2
        if unbatched_time:
            inputs = inputs.unsqueeze(1)
        if self.gru is None:
            outputs = self.head(inputs)
            next_hidden = None
        else:
            if hidden is None:
                hidden = torch.zeros(
                    1,
                    inputs.shape[0],
                    self.gru_hidden_size,
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
            outputs, next_hidden = self.gru(inputs, hidden)
            outputs = self.head(outputs)
        if unbatched_time:
            outputs = outputs.squeeze(1)
        return outputs, next_hidden


class PolicyActor(nn.Module):
    """Actor with optional GRU and NoisyNet exploration."""

    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(
        self,
        observation_dim: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        *,
        hidden_sizes: Sequence[int] = (160, 154),
        use_gru: bool = True,
        gru_hidden_size: int = 64,
        use_noisy: bool = True,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        low = torch.as_tensor(tuple(action_low), dtype=torch.float32)
        high = torch.as_tensor(tuple(action_high), dtype=torch.float32)
        if low.ndim != 1 or low.shape != high.shape or low.numel() == 0:
            raise ValueError("action bounds must be non-empty equal-length vectors")
        if bool(torch.any(low > high)):
            raise ValueError("action_low must not exceed action_high")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.action_dim = int(low.numel())
        self.body = _SequenceHead(
            observation_dim,
            hidden_sizes,
            self.action_dim,
            use_gru=use_gru,
            gru_hidden_size=gru_hidden_size,
            use_noisy=use_noisy,
        )
        self.use_gru = bool(use_gru)
        self.gru_hidden_size = gru_hidden_size

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        sample_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw, next_hidden = self.body(
            observation,
            hidden,
            sample_noise=sample_noise,
        )
        half_range = 0.5 * (self.action_high - self.action_low)
        center = 0.5 * (self.action_high + self.action_low)
        return torch.tanh(raw) * half_range + center, next_hidden


class Critic(nn.Module):
    """Q critic with optional GRU over observation/action sequences."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_sizes: Sequence[int] = (160, 154),
        use_gru: bool = True,
        gru_hidden_size: int = 64,
        use_noisy: bool = True,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.body = _SequenceHead(
            observation_dim + action_dim,
            hidden_sizes,
            1,
            use_gru=use_gru,
            gru_hidden_size=gru_hidden_size,
            use_noisy=use_noisy,
        )
        self.use_gru = bool(use_gru)
        self.gru_hidden_size = gru_hidden_size

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        sample_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError("observation and action leading dimensions must match")
        inputs = torch.cat((observation, action), dim=-1)
        return self.body(inputs, hidden, sample_noise=sample_noise)


__all__ = ["Critic", "NoisyLinear", "PolicyActor"]
