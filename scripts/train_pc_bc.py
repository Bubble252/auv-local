from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_lowlevel.batch.pc_bc import (
    PCBCTrainingConfig,
    load_pc_demonstration_dataset,
    train_pc_bc,
)
from auv_lowlevel.batch.pc_normalization import PCNormalization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PC20 pure-controller BC actor")
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-encoder-hidden-sizes", default="128,128")
    parser.add_argument("--actor-hidden-sizes", default="256,128")
    parser.add_argument("--d-z-total", type=int, default=32)
    parser.add_argument("--n-in-reserved", type=int, default=8)
    parser.add_argument("--n-z-reserved", type=int, default=4)
    parser.add_argument("--activation", default="elu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actor-command-dim", type=int, choices=(0, 3), default=None)
    parser.add_argument("--path-dim", type=int, choices=(0, 8, 12), default=None)
    parser.add_argument("--path-encoder-hidden-sizes", default="64")
    parser.add_argument("--path-encoder-alpha", type=float, default=1.0)
    parser.add_argument("--actor-current-encoder-enabled", action="store_true")
    parser.add_argument("--actor-current-encoder-hidden-sizes", default="64")
    parser.add_argument("--actor-current-encoder-alpha", type=float, default=1.0)
    parser.add_argument("--saturation-weight", type=float, default=0.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.9)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional BC checkpoint whose model weights initialize this training run.",
    )
    parser.add_argument(
        "--action-head",
        default="continuous",
        help="continuous or sign_magnitude (sign-magnitude alias accepted)",
    )
    parser.add_argument("--sign-deadband", type=float, default=0.05)
    return parser.parse_args()


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"invalid hidden sizes: {value!r}") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("hidden sizes must be positive")
    return sizes


def main() -> int:
    args = _parse_args()
    dataset, demo_metadata = load_pc_demonstration_dataset(args.demos)
    configs = demo_metadata.get("configs")
    normalization_payload = None
    if isinstance(configs, list) and configs:
        normalization_payload = configs[0].get("normalization")
    normalization = (
        PCNormalization(**normalization_payload)
        if isinstance(normalization_payload, dict)
        else PCNormalization()
    )
    training_config = PCBCTrainingConfig(
        state_encoder_hidden_dims=_parse_sizes(args.state_encoder_hidden_sizes),
        actor_hidden_dims=_parse_sizes(args.actor_hidden_sizes),
        d_z_total=args.d_z_total,
        n_in_reserved=args.n_in_reserved,
        n_z_reserved=args.n_z_reserved,
        activation=args.activation,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
        actor_command_dim=(
            dataset.actor_command_dim
            if args.actor_command_dim is None
            else args.actor_command_dim
        ),
        path_dim=dataset.path_dim if args.path_dim is None else args.path_dim,
        path_encoder_hidden_dims=_parse_sizes(args.path_encoder_hidden_sizes),
        path_encoder_alpha=args.path_encoder_alpha,
        actor_current_encoder_enabled=args.actor_current_encoder_enabled,
        actor_current_encoder_hidden_dims=_parse_sizes(
            args.actor_current_encoder_hidden_sizes
        ),
        actor_current_encoder_alpha=args.actor_current_encoder_alpha,
        saturation_weight=args.saturation_weight,
        saturation_threshold=args.saturation_threshold,
        action_head=args.action_head,
        sign_deadband=args.sign_deadband,
    )
    result = train_pc_bc(
        dataset,
        training_config,
        output_dir=args.output,
        metadata=demo_metadata,
        normalization=normalization,
        init_checkpoint=args.init_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
