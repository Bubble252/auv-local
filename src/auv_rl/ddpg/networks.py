from __future__ import annotations

import math
from collections.abc import Sequence

import torch  # type: ignore[import-not-found, unused-ignore]
from torch import nn  # type: ignore[import-not-found, unused-ignore]


def _activation_module(name: str) -> nn.Module:
    """Build one of the activation functions used in the P1 ablations."""

    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(negative_slope=0.01)
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name!r}")


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

    def _sample_per_sample_noise(
        self,
        batch_size: int,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(shape) == 1:
            noise = torch.randn(batch_size, shape[0], device=device, dtype=dtype)
            return noise.sign() * noise.abs().sqrt()
        if len(shape) == 2:
            output_noise = torch.randn(batch_size, shape[0], device=device, dtype=dtype)
            input_noise = torch.randn(batch_size, shape[1], device=device, dtype=dtype)
            output_noise = output_noise.sign() * output_noise.abs().sqrt()
            input_noise = input_noise.sign() * input_noise.abs().sqrt()
            return output_noise.unsqueeze(2) * input_noise.unsqueeze(1)
        raise ValueError("noise shape must be one- or two-dimensional")

    def forward(
        self,
        input_tensor: torch.Tensor,
        *,
        sample_noise: bool = True,
        per_sample_noise: bool = False,
    ) -> torch.Tensor:
        if self.use_noisy and sample_noise:
            weight_sigma = torch.nn.functional.softplus(self.weight_rho)
            bias_sigma = torch.nn.functional.softplus(self.bias_rho)
            if per_sample_noise:
                if input_tensor.ndim < 2:
                    raise ValueError("per-sample noise requires a batched input")
                leading_shape = input_tensor.shape[:-1]
                flattened = input_tensor.reshape(-1, self.in_features)
                batch_size = flattened.shape[0]
                weight_epsilon = self._sample_per_sample_noise(
                    batch_size,
                    self.weight_mu.shape,
                    self.weight_mu.device,
                    self.weight_mu.dtype,
                )
                bias_epsilon = self._sample_per_sample_noise(
                    batch_size,
                    self.bias_mu.shape,
                    self.bias_mu.device,
                    self.bias_mu.dtype,
                )
                weight = self.weight_mu.unsqueeze(0) + weight_sigma.unsqueeze(0) * weight_epsilon
                bias = self.bias_mu.unsqueeze(0) + bias_sigma.unsqueeze(0) * bias_epsilon
                output = torch.bmm(weight, flattened.unsqueeze(-1)).squeeze(-1) + bias
                return output.reshape(*leading_shape, self.out_features)
            weight_epsilon = self._sample_noise(self.weight_mu.shape, self.weight_mu.device)
            bias_epsilon = self._sample_noise(self.bias_mu.shape, self.bias_mu.device)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(input_tensor, weight, bias)


class _NoisyMLP(nn.Module):
    """MLP that forwards the noise mode to every :class:`NoisyLinear` layer.

    ``nn.Sequential`` cannot pass the ``sample_noise`` keyword to its children,
    which silently made evaluation sample noisy weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        *,
        use_noisy: bool,
        activation: str = "relu",
        output_use_noisy: bool | None = None,
        output_sigma_init: float = 0.5,
        hidden_sigma_init: float = 0.5,
        output_weight_init_bound: float | None = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(output_sigma_init) or output_sigma_init <= 0.0:
            raise ValueError("output_sigma_init must be finite and positive")
        if not math.isfinite(hidden_sigma_init) or hidden_sigma_init <= 0.0:
            raise ValueError("hidden_sigma_init must be finite and positive")
        if output_weight_init_bound is not None and (
            not math.isfinite(output_weight_init_bound) or output_weight_init_bound <= 0.0
        ):
            raise ValueError("output_weight_init_bound must be finite and positive")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_sizes:
            layers.append(
                NoisyLinear(
                    previous,
                    width,
                    use_noisy=use_noisy,
                    sigma_init=hidden_sigma_init,
                )
            )
            layers.append(_activation_module(activation))
            previous = width
        output_layer = NoisyLinear(
            previous,
            output_dim,
            use_noisy=use_noisy if output_use_noisy is None else output_use_noisy,
            sigma_init=output_sigma_init,
        )
        if output_weight_init_bound is not None:
            nn.init.uniform_(
                output_layer.weight_mu,
                -output_weight_init_bound,
                output_weight_init_bound,
            )
            nn.init.uniform_(
                output_layer.bias_mu,
                -output_weight_init_bound,
                output_weight_init_bound,
            )
        layers.append(output_layer)
        self.layers = nn.ModuleList(layers)

    def first_activation(self) -> nn.Module:
        """Return the activation that follows the first hidden linear layer."""

        if len(self.layers) < 2 or isinstance(self.layers[0], NoisyLinear) is False:
            raise RuntimeError("NoisyMLP does not contain a hidden activation")
        return self.layers[1]

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        sample_noise: bool = True,
        per_sample_noise: bool = False,
    ) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                outputs = layer(
                    outputs,
                    sample_noise=sample_noise,
                    per_sample_noise=per_sample_noise,
                )
            else:
                outputs = layer(outputs)
        return outputs


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
        activation: str = "relu",
        layer_norm: bool = False,
        output_use_noisy: bool | None = None,
        output_sigma_init: float = 0.5,
        hidden_sigma_init: float = 0.5,
        output_weight_init_bound: float | None = None,
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
        self.layer_norm: nn.LayerNorm | None = nn.LayerNorm(head_input) if layer_norm else None
        self.head = _NoisyMLP(
            head_input,
            hidden_sizes,
            output_dim,
            use_noisy=use_noisy,
            activation=activation,
            output_use_noisy=output_use_noisy,
            output_sigma_init=output_sigma_init,
            hidden_sigma_init=hidden_sigma_init,
            output_weight_init_bound=output_weight_init_bound,
        )
        self.gru_hidden_size = gru_hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        sample_noise: bool = True,
        per_sample_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if inputs.ndim not in (2, 3):
            raise ValueError("inputs must have shape (B,D) or (B,T,D)")
        unbatched_time = inputs.ndim == 2
        if unbatched_time:
            inputs = inputs.unsqueeze(1)
        if self.gru is None:
            normalized = inputs
            if self.layer_norm is not None:
                normalized = self.layer_norm(normalized)
            outputs = self.head(
                normalized,
                sample_noise=sample_noise,
                per_sample_noise=per_sample_noise,
            )
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
            if self.layer_norm is not None:
                outputs = self.layer_norm(outputs)
            outputs = self.head(
                outputs,
                sample_noise=sample_noise,
                per_sample_noise=per_sample_noise,
            )
        if unbatched_time:
            outputs = outputs.squeeze(1)
        return outputs, next_hidden

    def first_activation(self) -> nn.Module:
        return self.head.first_activation()


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
        activation: str = "relu",
        layer_norm: bool = False,
        actor_output_noise: bool = True,
        actor_output_sigma_init: float = 0.5,
        actor_hidden_sigma_init: float = 0.5,
        actor_output_weight_init_bound: float | None = None,
        actor_output_scale: float = 1.0,
        actor_output_squash: str = "tanh",
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
        if not math.isfinite(actor_output_scale) or not 0.0 < actor_output_scale <= 1.0:
            raise ValueError("actor_output_scale must be finite and in (0, 1]")
        squash = str(actor_output_squash).strip().lower()
        if squash not in {"tanh", "softsign"}:
            raise ValueError("actor_output_squash must be 'tanh' or 'softsign'")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.action_dim = int(low.numel())
        self.actor_output_scale = float(actor_output_scale)
        self.actor_output_squash = squash
        self.body = _SequenceHead(
            observation_dim,
            hidden_sizes,
            self.action_dim,
            use_gru=use_gru,
            gru_hidden_size=gru_hidden_size,
            use_noisy=use_noisy,
            activation=activation,
            layer_norm=layer_norm,
            output_use_noisy=use_noisy and actor_output_noise,
            output_sigma_init=actor_output_sigma_init,
            hidden_sigma_init=actor_hidden_sigma_init,
            output_weight_init_bound=actor_output_weight_init_bound,
        )
        self.use_gru = bool(use_gru)
        self.gru_hidden_size = gru_hidden_size
        self._hidden_sigma_init = float(actor_hidden_sigma_init)
        self._output_sigma_init = float(actor_output_sigma_init)

    def reset_noise_sigmas(self) -> None:
        """Reset NoisyNet sigma parameters while keeping the learned means."""

        layers = self.body.head.layers
        linear_layers = [module for module in layers if isinstance(module, NoisyLinear)]
        for index, layer in enumerate(linear_layers):
            sigma = (
                self._output_sigma_init
                if index == len(linear_layers) - 1
                else self._hidden_sigma_init
            )
            if not layer.use_noisy:
                continue
            rho = math.log(math.expm1(sigma))
            with torch.no_grad():
                layer.weight_rho.fill_(rho)
                layer.bias_rho.fill_(rho)

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        sample_noise: bool = True,
        per_sample_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw, next_hidden = self.forward_raw(
            observation,
            hidden,
            sample_noise=sample_noise,
            per_sample_noise=per_sample_noise,
        )
        half_range = 0.5 * (self.action_high - self.action_low)
        center = 0.5 * (self.action_high + self.action_low)
        return self.action_from_raw(raw) * half_range + center, next_hidden

    def squash(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw policy outputs to the normalized [-1, 1] interval."""

        if self.actor_output_squash == "softsign":
            return torch.nn.functional.softsign(raw)
        return torch.tanh(raw)

    def squash_derivative(self, raw: torch.Tensor) -> torch.Tensor:
        """Return the derivative of :meth:`squash` with respect to raw output."""

        if self.actor_output_squash == "softsign":
            return 1.0 / torch.square(1.0 + raw.abs())
        tanh = torch.tanh(raw)
        return 1.0 - torch.square(tanh)

    def action_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """Return the bounded normalized action before action-space scaling."""

        return self.actor_output_scale * self.squash(raw)

    def forward_raw(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        sample_noise: bool = True,
        per_sample_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the pre-tanh policy output used by saturation diagnostics."""

        raw, next_hidden = self.body(
            observation,
            hidden,
            sample_noise=sample_noise,
            per_sample_noise=per_sample_noise,
        )
        return raw, next_hidden


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
        activation: str = "relu",
        layer_norm: bool = False,
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
            activation=activation,
            layer_norm=layer_norm,
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
        per_sample_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError("observation and action leading dimensions must match")
        inputs = torch.cat((observation, action), dim=-1)
        return self.body(
            inputs,
            hidden,
            sample_noise=sample_noise,
            per_sample_noise=per_sample_noise,
        )

    def first_activation(self) -> nn.Module:
        return self.body.first_activation()


__all__ = ["Critic", "NoisyLinear", "PolicyActor"]
