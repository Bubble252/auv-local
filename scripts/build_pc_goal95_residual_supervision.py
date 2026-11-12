#!/usr/bin/env python3
"""Build Goal95 supervised residual targets from Goal86 D3/ALOS labels.

The Goal95 policy is composed as

    executed = ALOS + s_env * (r_BC + s_ppo * r_PPO)

where ``r_BC`` is the frozen Goal86 P-A residual.  This script converts the
physical D3/ALOS labels into both the PPO target ``r_PPO`` and the normalized
teacher action consumed by ``scripts/pretrain_pc_residual.py``:

    r_PPO = ((D3 - ALOS) / full_scale - s_env * r_BC) / (s_env * s_ppo)
    action = r_BC + s_ppo * r_PPO

Both arms therefore reproduce D3 when the terminal gate is fully open:

* ``d3``   : s_env=1.5, s_ppo=1.0
* ``bc15`` : s_env=1.5, s_ppo=2/3 (BC branch keeps 1.5x authority)

The saved ``action`` is in normalized actuator units and is the field consumed
by the existing PC residual pretrainer.  No environment is created and no
held-out seed is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from auv_lowlevel.batch.pc_bc import PCBCPolicy, pc_action_full_scale

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("d3", "bc15"), required=True)
    parser.add_argument("--s-env", type=float, default=1.5)
    parser.add_argument("--s-ppo", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "abs_mean": absolute.mean(axis=0).tolist(),
        "abs_p95": np.percentile(absolute, 95, axis=0).tolist(),
        "abs_max": absolute.max(axis=0).tolist(),
        "clip_fraction": (absolute > 1.0 + 1e-6).mean(axis=0).tolist(),
    }


def main() -> int:
    args = _parse_args()
    labels_path = _resolve(args.labels)
    bc_path = _resolve(args.bc_checkpoint)
    output = _resolve(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    if not np.isfinite(args.s_env) or args.s_env <= 0.0:
        raise ValueError("--s-env must be finite and positive")
    s_ppo = 1.0 if args.arm == "d3" else 2.0 / 3.0
    if args.s_ppo is not None:
        s_ppo = float(args.s_ppo)
    if not np.isfinite(s_ppo) or s_ppo <= 0.0:
        raise ValueError("--s-ppo must be finite and positive")
    if args.arm == "bc15" and not np.isclose(s_ppo, 2.0 / 3.0, rtol=0.0, atol=1e-12):
        raise ValueError("bc15 requires s_ppo=2/3 so the BC branch keeps 1.5x authority")

    payload = np.load(labels_path)
    required = (
        "actor_state",
        "command",
        "actor_path",
        "base_actuator",
        "d3_actuator",
        "episode_id",
        "seed",
        "step",
    )
    missing = sorted(set(required) - set(payload.files))
    if missing:
        raise ValueError(f"label file is missing arrays: {missing}")
    count = int(payload["actor_state"].shape[0])
    if count <= 0:
        raise ValueError("label file is empty")

    bc = PCBCPolicy(bc_path, device=args.device)
    full_scale = torch.tensor(pc_action_full_scale(), dtype=torch.float64)
    r_bc_chunks: list[np.ndarray] = []
    for start in range(0, count, int(args.batch_size)):
        end = min(start + int(args.batch_size), count)
        with torch.inference_mode():
            r_bc_chunks.append(
                bc.act_inference(
                    payload["actor_state"][start:end],
                    command=payload["command"][start:end],
                    actor_path=payload["actor_path"][start:end],
                    actor_current=np.zeros((end - start, 3), dtype=np.float32),
                ).cpu().numpy()
            )
    r_bc = np.concatenate(r_bc_chunks, axis=0).astype(np.float64)

    d3_normalized = np.asarray(payload["d3_actuator"], dtype=np.float64) / full_scale.numpy()
    base_normalized = (
        np.asarray(payload["base_actuator"], dtype=np.float64) / full_scale.numpy()
    )
    target = (
        (d3_normalized - base_normalized) - float(args.s_env) * r_bc
    ) / (float(args.s_env) * s_ppo)
    action = r_bc + s_ppo * target
    if not np.isfinite(target).all() or not np.isfinite(action).all():
        raise ValueError("supervision targets contain NaN or Inf")

    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "goal95_supervision.npz"
    np.savez_compressed(
        dataset_path,
        actor_state=np.asarray(payload["actor_state"], dtype=np.float32),
        command=np.asarray(payload["command"], dtype=np.float32),
        action=action.astype(np.float32),
        episode_id=np.asarray(payload["episode_id"], dtype=np.int64),
        seed=np.asarray(payload["seed"], dtype=np.int64),
        step=np.asarray(payload["step"], dtype=np.int64),
        actual_actuator=np.asarray(payload["d3_actuator"], dtype=np.float32),
        reward=np.zeros((count,), dtype=np.float32),
        actor_path=np.asarray(payload["actor_path"], dtype=np.float32),
        current_estimate_body=np.zeros((count, 3), dtype=np.float32),
        observation_schema=np.asarray("pc_v1"),
        action_envelope=np.asarray("pc20"),
        r_bc_normalized=r_bc.astype(np.float32),
        target_residual_normalized=target.astype(np.float32),
        base_actuator=np.asarray(payload["base_actuator"], dtype=np.float64),
        d3_actuator=np.asarray(payload["d3_actuator"], dtype=np.float64),
    )
    manifest = {
        "schema": "auv_goal95_residual_supervision.v1",
        "arm": args.arm,
        "labels": str(labels_path.resolve()),
        "labels_sha256": _sha256(labels_path),
        "bc_checkpoint": str(bc_path.resolve()),
        "bc_checkpoint_sha256": _sha256(bc_path),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "sample_count": count,
        "s_env": float(args.s_env),
        "s_ppo": float(s_ppo),
        "composition": "executed = ALOS + s_env * (r_BC + s_ppo * r_PPO)",
        "target_formula": (
            "r_PPO = ((D3 - ALOS) / full_scale - s_env * r_BC) / (s_env * s_ppo)"
        ),
        "target_statistics": _stats(target),
        "r_bc_statistics": _stats(r_bc),
        "action_statistics": _stats(action),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
