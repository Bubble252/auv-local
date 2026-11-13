"""Merge PC20 behavior-cloning datasets without losing lineage metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .._json import write_json_document
from .pc_bc import PC_ACTION_ENVELOPE
from .pc_policy import PC_SCHEMA_VERSION

PC_BC_MERGE_MANIFEST_SCHEMA = "auv_pc_bc_merged_manifest.v1"

_CORE_FIELDS = (
    "actor_state",
    "command",
    "action",
    "episode_id",
    "seed",
    "step",
    "actual_actuator",
    "reward",
)
_INTEGER_FIELDS = frozenset({"episode_id", "seed", "step"})
_OPTIONAL_2D_WIDTHS = {
    "current_estimate_body": (3,),
    "actor_path": (8, 12),
}
_OPTIONAL_2D_FIELDS = frozenset(_OPTIONAL_2D_WIDTHS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_string(payload: np.lib.npyio.NpzFile, name: str, default: str) -> str:
    if name not in payload.files:
        return default
    value = np.asarray(payload[name])
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    return str(value.item())


def _manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(".manifest.json")


def _normalization(metadata: dict[str, Any], dataset_path: Path) -> dict[str, Any] | None:
    configs = metadata.get("configs")
    if not isinstance(configs, list) or not configs:
        return None
    first = configs[0]
    if not isinstance(first, dict):
        raise ValueError(f"{dataset_path}: first config metadata must be an object")
    normalization = first.get("normalization")
    if normalization is None:
        return None
    if not isinstance(normalization, dict):
        raise ValueError(f"{dataset_path}: normalization metadata must be an object")
    for index, config in enumerate(configs[1:], start=1):
        if not isinstance(config, dict):
            raise ValueError(f"{dataset_path}: config {index} metadata must be an object")
        if config.get("normalization") != normalization:
            raise ValueError(f"{dataset_path}: configs use inconsistent normalization")
    return normalization


def merge_pc_demonstration_datasets(
    inputs: list[Path | str] | tuple[Path | str, ...],
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge PC datasets, remapping episode ids and preserving config lineage."""

    if not inputs:
        raise ValueError("at least one PC demonstration dataset is required")
    output = Path(output_dir)
    dataset_path = output / "demos.npz"
    manifest_path = output / "demos.manifest.json"
    if not overwrite and (dataset_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"output already contains PC demonstrations: {output}")

    merged: dict[str, list[np.ndarray]] = {field: [] for field in _CORE_FIELDS}
    input_metadata: list[dict[str, Any]] = []
    merged_configs: list[Any] = []
    reference_normalization: dict[str, Any] | None = None
    episode_offset = 0
    optional_fields_present: set[str] | None = None
    command_dim: int | None = None
    path_dim: int | None = None

    for input_value in inputs:
        input_path = Path(input_value)
        with np.load(input_path, allow_pickle=False) as payload:
            missing = sorted(set(_CORE_FIELDS) - set(payload.files))
            if missing:
                raise ValueError(f"{input_path}: missing fields {missing}")
            if _scalar_string(payload, "observation_schema", PC_SCHEMA_VERSION) != PC_SCHEMA_VERSION:
                raise ValueError(f"{input_path}: unsupported PC observation schema")
            if _scalar_string(payload, "action_envelope", PC_ACTION_ENVELOPE) != PC_ACTION_ENVELOPE:
                raise ValueError(f"{input_path}: unsupported PC action envelope")
            present_optional = _OPTIONAL_2D_FIELDS.intersection(payload.files)
            if optional_fields_present is None:
                optional_fields_present = set(present_optional)
                for field in optional_fields_present:
                    merged[field] = []
            elif present_optional != optional_fields_present:
                raise ValueError(
                    f"{input_path}: optional fields differ from earlier inputs; "
                    f"expected {sorted(optional_fields_present)}, got {sorted(present_optional)}"
                )
            fields = (*_CORE_FIELDS, *sorted(optional_fields_present))
            arrays = {field: np.asarray(payload[field]) for field in fields}

        count = int(arrays["actor_state"].shape[0])
        if count <= 0 or arrays["actor_state"].ndim != 2:
            raise ValueError(f"{input_path}: actor_state must be a non-empty (N, D) array")
        for field in _CORE_FIELDS:
            expected_ndim = 2 if field in {"actor_state", "command", "action", "actual_actuator"} else 1
            if arrays[field].ndim != expected_ndim or arrays[field].shape[0] != count:
                raise ValueError(f"{input_path}: invalid shape for {field}")
        current_command_dim = int(arrays["command"].shape[1])
        if current_command_dim not in (0, 3):
            raise ValueError(f"{input_path}: command must have 0 or 3 columns")
        if command_dim is None:
            command_dim = current_command_dim
        elif command_dim != current_command_dim:
            raise ValueError(
                f"{input_path}: command width differs from earlier inputs; "
                f"expected {command_dim}, got {current_command_dim}"
            )
        for field in _CORE_FIELDS:
            if field not in _INTEGER_FIELDS and not np.all(np.isfinite(arrays[field])):
                raise ValueError(f"{input_path}: {field} contains NaN or Inf")
        for field in optional_fields_present:
            allowed_widths = _OPTIONAL_2D_WIDTHS[field]
            if arrays[field].ndim != 2 or arrays[field].shape[0] != count:
                raise ValueError(f"{input_path}: {field} must have shape (N, W)")
            width = int(arrays[field].shape[1])
            if width not in allowed_widths:
                raise ValueError(
                    f"{input_path}: {field} width must be one of {allowed_widths}, got {width}"
                )
            if field == "actor_path":
                if path_dim is None:
                    path_dim = width
                elif path_dim != width:
                    raise ValueError(
                        f"{input_path}: actor_path width differs from earlier inputs; "
                        f"expected {path_dim}, got {width}"
                    )
            if not np.all(np.isfinite(arrays[field])):
                raise ValueError(f"{input_path}: {field} contains NaN or Inf")

        episode_ids = arrays["episode_id"].astype(np.int64, copy=False)
        if np.any(episode_ids < 0):
            raise ValueError(f"{input_path}: episode_id must be non-negative")
        arrays["episode_id"] = episode_ids + episode_offset
        episode_offset = int(arrays["episode_id"].max()) + 1
        for field in (*_CORE_FIELDS, *sorted(optional_fields_present)):
            merged[field].append(arrays[field])

        metadata: dict[str, Any] = {}
        sidecar = _manifest_path(input_path)
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        normalization = _normalization(metadata, input_path)
        if normalization is not None:
            if reference_normalization is None:
                reference_normalization = normalization
            elif normalization != reference_normalization:
                raise ValueError("input datasets use inconsistent normalization")
        configs = metadata.get("configs")
        if isinstance(configs, list):
            merged_configs.extend(configs)
        first_config = configs[0] if isinstance(configs, list) and configs else {}
        if not isinstance(first_config, dict):
            first_config = {}
        input_metadata.append(
            {
                "dataset_path": str(input_path.resolve()),
                "dataset_sha256": _sha256(input_path),
                "manifest_path": str(sidecar.resolve()) if sidecar.is_file() else None,
                "samples": count,
                "episodes": int(np.unique(arrays["episode_id"]).size),
                "config_name": first_config.get("config_name"),
                "pc_stage": first_config.get("pc_stage"),
                "sensor_profile": first_config.get("sensor_profile"),
                "current_estimate_body": "current_estimate_body" in optional_fields_present,
                "actor_path": "actor_path" in optional_fields_present,
                "path_dim": (
                    int(arrays["actor_path"].shape[1])
                    if "actor_path" in optional_fields_present
                    else 0
                ),
                "actor_observation_source": first_config.get("actor_observation_source"),
                "seed_start": first_config.get("seed_start"),
                "seed_end": first_config.get("seed_end"),
                "seed_list": first_config.get("seed_list"),
                "seed_role": first_config.get("seed_role"),
            }
        )

    payload = {
        field: np.concatenate(values, axis=0).astype(
            np.float32 if field not in _INTEGER_FIELDS else np.int64
        )
        for field, values in merged.items()
    }
    payload["observation_schema"] = np.asarray(PC_SCHEMA_VERSION)
    payload["action_envelope"] = np.asarray(PC_ACTION_ENVELOPE)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **payload)
    manifest = {
        "schema_version": PC_BC_MERGE_MANIFEST_SCHEMA,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "observation_schema": PC_SCHEMA_VERSION,
        "action_envelope": PC_ACTION_ENVELOPE,
        "sample_count": int(payload["actor_state"].shape[0]),
        "episode_count": int(np.unique(payload["episode_id"]).size),
        "actor_command_dim": int(payload["command"].shape[1]),
        "path_dim": 0 if "actor_path" not in payload else int(payload["actor_path"].shape[1]),
        "inputs": input_metadata,
        "configs": merged_configs,
        "normalization": reference_normalization,
        "current_estimate_body": "current_estimate_body" in (optional_fields_present or set()),
        "actor_path": "actor_path" in (optional_fields_present or set()),
    }
    write_json_document(manifest_path, manifest)
    return manifest


__all__ = ["PC_BC_MERGE_MANIFEST_SCHEMA", "merge_pc_demonstration_datasets"]
