from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from auv_lowlevel.observation_schema import ObservationSchema, resolve_observation_schema

FIELDS = [
    "observation",
    "action",
    "episode_id",
    "seed",
    "step",
    "time_s",
    "actual_actuator",
    "state_position_ned_m",
    "state_quaternion_xyzw",
    "state_linear_velocity_body_mps",
    "state_angular_velocity_body_rps",
    "reference_position_ned_m",
    "reference_quaternion_xyzw",
    "current_ned_mps",
    "teacher_post_integral_rpm",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge expert and DAgger datasets")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    merged: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    schema: ObservationSchema | None = None
    for path in args.inputs:
        with np.load(path) as payload:
            missing = sorted(set(FIELDS) - set(payload.files))
            if missing:
                raise ValueError(f"{path} is missing fields: {missing}")
            observation = np.asarray(payload["observation"])
            if observation.ndim != 2:
                raise ValueError(f"{path} observation must have shape (N, D)")
            stored_schema: str | None = None
            if "observation_schema" in payload.files:
                schema_value = np.asarray(payload["observation_schema"])
                if schema_value.shape != ():
                    raise ValueError(f"{path} observation_schema must be a scalar")
                stored_schema = str(schema_value.item())
            input_schema = resolve_observation_schema(
                int(observation.shape[1]), stored_schema
            )
            if schema is None:
                schema = input_schema
            elif input_schema != schema:
                raise ValueError(
                    f"{path} observation schema mismatch: {input_schema} != {schema}"
                )
            for field in FIELDS:
                merged[field].append(np.asarray(payload[field]))
    arrays = {field: np.concatenate(values, axis=0) for field, values in merged.items()}
    assert schema is not None
    arrays["observation_schema"] = np.asarray(schema.version)
    episodes = np.unique(arrays["episode_id"])
    if episodes.size != np.unique(arrays["seed"]).size:
        raise ValueError("episode ids collide across inputs; use a distinct id offset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "schema_version": "auv_odarrl_merged_demos.v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "input_sha256": [_sha256(path) for path in args.inputs],
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "episode_count": int(episodes.size),
        "sample_count": int(arrays["observation"].shape[0]),
        "observation_dim": schema.dim,
        "observation_schema": schema.version,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
