"""Report side-effect-free diagnostics for PC actor and path parameter updates.

Non-responsibility: does not train policies, mutate parameters, or write artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


def _parameter_names(module: nn.Module, prefix: str) -> list[str]:
    return [name for name, _parameter in module.named_parameters() if name.startswith(prefix)]


def _norm(value: Tensor | None) -> float | None:
    if value is None:
        return None
    return float(torch.linalg.vector_norm(value.detach()).item())


def _linear_weights(module: nn.Module | None) -> list[tuple[str, nn.Linear]]:
    if module is None:
        return []
    return [(name, layer) for name, layer in module.named_modules() if isinstance(layer, nn.Linear)]


def policy_diagnostics(
    policy: nn.Module,
    *,
    previous_parameters: Mapping[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Return path/actor norms and optional update norms since a snapshot.

    Gradient norms are taken after the last optimizer update in an iteration.
    They are useful for detecting a disconnected branch, not for replacing the
    PPO losses or the scientific evaluation gates.
    """

    result: dict[str, Any] = {
        "path_encoder_present": False,
        "path_encoder_alpha": None,
        "path_encoder_final_weight_norm": None,
        "path_encoder_final_bias_norm": None,
        "path_encoder_final_weight_grad_norm": None,
        "path_encoder_effective_output_scale": None,
        "path_encoder_layers": {},
        "actor_layers": {},
    }

    path_encoder = getattr(policy, "path_encoder", None)
    if path_encoder is not None:
        result["path_encoder_present"] = True
        alpha = float(getattr(policy, "path_encoder_alpha", 0.0))
        result["path_encoder_alpha"] = alpha
        linear_layers = _linear_weights(path_encoder)
        if linear_layers:
            final_name, final_layer = linear_layers[-1]
            final_weight_norm = _norm(final_layer.weight)
            result["path_encoder_final_weight_norm"] = final_weight_norm
            result["path_encoder_final_bias_norm"] = _norm(final_layer.bias)
            result["path_encoder_final_weight_grad_norm"] = _norm(final_layer.weight.grad)
            if final_weight_norm is not None:
                result["path_encoder_effective_output_scale"] = alpha * final_weight_norm
            for layer_name, layer in linear_layers:
                result["path_encoder_layers"][layer_name] = {
                    "weight_norm": _norm(layer.weight),
                    "weight_grad_norm": _norm(layer.weight.grad),
                }

    actor = getattr(policy, "actor", None)
    for name, layer in _linear_weights(actor):
        result["actor_layers"][name] = {
            "weight_norm": _norm(layer.weight),
            "weight_grad_norm": _norm(layer.weight.grad),
        }

    if previous_parameters is not None:
        current = dict(policy.named_parameters())
        update_norms: dict[str, float | None] = {}
        for name, parameter in current.items():
            if name not in previous_parameters:
                continue
            previous = previous_parameters[name].to(device=parameter.device, dtype=parameter.dtype)
            update_norms[name] = _norm(parameter.detach() - previous)
        result["parameter_update_norms"] = update_norms

    return result


def parameter_snapshot(policy: nn.Module) -> dict[str, Tensor]:
    """Clone policy parameters so the next diagnostic can compute update norms."""

    return {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}


__all__ = ["parameter_snapshot", "policy_diagnostics"]
