from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from auv_lowlevel.bc import BCTrainingConfig, load_demo_dataset, train_bc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ODARRL I-arm pi0 actor")
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demos-manifest", type=Path, default=None)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-observation-dim", type=int, default=None)
    parser.add_argument(
        "--speed-feedback",
        choices=("water_relative", "ground"),
        default="water_relative",
        help="Speed feedback contract used to collect the demonstrations",
    )
    parser.add_argument(
        "--current-contract",
        choices=("legacy", "hidden", "oracle"),
        default=None,
    )
    parser.add_argument("--expose-current", action="store_true", default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def main() -> int:
    args = _parse_args()
    dataset = load_demo_dataset(args.demos)
    schema = dataset.schema
    if args.expected_observation_dim is not None:
        if args.expected_observation_dim not in {26, 27}:
            raise ValueError("--expected-observation-dim must be 26 or 27")
        if schema.dim != args.expected_observation_dim:
            raise ValueError(
                "demo observation dimension mismatch: "
                f"expected {args.expected_observation_dim}, got {schema.dim}"
            )
    hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(","))
    current_contract = args.current_contract
    if current_contract is None:
        current_contract = (
            "oracle"
            if args.expose_current
            else ("hidden" if args.speed_feedback == "ground" else "legacy")
        )
    if current_contract == "hidden" and args.speed_feedback != "ground":
        raise ValueError("hidden current contract requires --speed-feedback ground")
    if current_contract == "hidden" and args.expose_current:
        raise ValueError("hidden current contract cannot expose the true current")
    if current_contract == "oracle" and not args.expose_current:
        raise ValueError("oracle current contract requires --expose-current")
    config = BCTrainingConfig(
        hidden_sizes=hidden_sizes,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )
    demo_metadata = {
        "demos_path": str(args.demos.resolve()),
        "demos_sha256": _sha256(args.demos),
        "initialization": "scratch",
        "git": _git_state(),
    }
    if args.demos_manifest is not None:
        demo_manifest = json.loads(args.demos_manifest.read_text(encoding="utf-8"))
        for key, actual in (
            ("observation_dim", schema.dim),
            ("observation_schema", schema.version),
        ):
            stored = demo_manifest.get(key)
            if stored is not None and stored != actual:
                raise ValueError(
                    f"demo manifest {key} mismatch: expected {actual}, got {stored!r}"
                )
        demo_metadata["demos_manifest_path"] = str(args.demos_manifest.resolve())
        demo_metadata["demos_manifest_sha256"] = _sha256(args.demos_manifest)
        for key in (
            "schema_version",
            "expert_type",
            "config_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "inputs",
            "input_sha256",
            "observation_speed_feedback",
            "observation_current_contract",
            "observation_current_exposed",
        ):
            if key in demo_manifest:
                demo_metadata[f"teacher_{key}"] = demo_manifest[key]
    result = train_bc(
        dataset,
        config,
        output_dir=args.output,
        metadata=demo_metadata,
        observation_speed_feedback=args.speed_feedback,
        observation_current_contract=current_contract,
        observation_expose_current=args.expose_current,
    )
    report = {
        "schema_version": "auv_odarrl_bc_training.v1",
        "train_samples": result.train_samples,
        "validation_samples": result.validation_samples,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "validation_mse_normalized": result.validation_mse_normalized.tolist(),
        "validation_mse_actuator_units": result.validation_mse_actuator_units.tolist(),
        "training_config_sha256": result.training_config_sha256,
        "train_loss": result.train_loss,
        "validation_loss": result.validation_loss,
        "checkpoint_path": result.checkpoint_path,
        **demo_metadata,
        "observation_dim": schema.dim,
        "observation_schema": schema.version,
        "hidden_sizes": list(hidden_sizes),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "device": args.device,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "train_samples",
        "validation_samples",
        "best_epoch",
        "best_validation_loss",
        "validation_mse_actuator_units",
        "checkpoint_path",
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
