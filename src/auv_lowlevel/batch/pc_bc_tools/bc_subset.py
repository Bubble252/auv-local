"""Create deterministic seed subsets of PC behaviour-cloning datasets.

Non-responsibility: does not train networks or modify source dataset contents.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..._json import write_json_document

PC_BC_SUBSET_MANIFEST_SCHEMA = "auv_pc_bc_subset_manifest.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_and_manifest(input_path: Path) -> tuple[Path, Path]:
    if input_path.is_dir():
        return input_path / "demos.npz", input_path / "demos.manifest.json"
    return input_path, input_path.with_suffix(".manifest.json")


def _seed_set(config: dict[str, Any]) -> set[int] | None:
    values = config.get("seed_list")
    if isinstance(values, list) and values:
        return {int(value) for value in values}
    start = config.get("seed_start")
    end = config.get("seed_end")
    if start is not None and end is not None:
        return set(range(int(start), int(end) + 1))
    return None


def subset_pc_demonstration_dataset(
    input_path: Path | str,
    output_dir: Path | str,
    seeds: Iterable[int],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select complete seed trajectories without changing individual transitions."""

    input_dataset, input_manifest = _dataset_and_manifest(Path(input_path))
    output = Path(output_dir)
    output_dataset = output / "demos.npz"
    output_manifest = output / "demos.manifest.json"
    if not overwrite and (output_dataset.exists() or output_manifest.exists()):
        raise FileExistsError(f"output already contains PC demonstrations: {output}")

    selected_seeds = tuple(sorted({int(seed) for seed in seeds}))
    if not selected_seeds:
        raise ValueError("at least one seed is required")

    with np.load(input_dataset, allow_pickle=False) as source:
        fields = tuple(source.files)
        arrays = {field: np.asarray(source[field]) for field in fields}

    if "seed" not in arrays or "episode_id" not in arrays or "actor_state" not in arrays:
        raise ValueError(f"{input_dataset}: missing seed, episode_id or actor_state")
    seed_values = arrays["seed"].astype(np.int64, copy=False)
    if seed_values.ndim != 1:
        raise ValueError(f"{input_dataset}: seed must be one-dimensional")
    sample_count = int(seed_values.shape[0])
    missing_seeds = sorted(
        set(selected_seeds) - set(int(value) for value in np.unique(seed_values))
    )
    if missing_seeds:
        raise ValueError(f"{input_dataset}: requested seeds are absent: {missing_seeds}")

    selected_mask = np.isin(seed_values, np.asarray(selected_seeds, dtype=np.int64))
    if not np.any(selected_mask):
        raise ValueError("seed selection produced no samples")
    selected_original_episode_ids = arrays["episode_id"][selected_mask].astype(np.int64, copy=False)
    episode_ids = np.unique(selected_original_episode_ids)
    remap = {int(old): int(new) for new, old in enumerate(episode_ids)}

    subset: dict[str, np.ndarray] = {}
    for field, value in arrays.items():
        if value.shape == ():
            subset[field] = value
            continue
        if value.ndim == 0 or value.shape[0] != sample_count:
            raise ValueError(f"{input_dataset}: {field} is not aligned with seed")
        subset[field] = value[selected_mask]
    subset["episode_id"] = np.asarray(
        [remap[int(value)] for value in selected_original_episode_ids],
        dtype=np.int64,
    )

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dataset, **subset)

    manifest: dict[str, Any] = {}
    if input_manifest.is_file():
        manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    configs = manifest.get("configs")
    if isinstance(configs, list):
        for config in configs:
            if not isinstance(config, dict):
                continue
            config_seeds = _seed_set(config)
            if config_seeds is None:
                continue
            config_mask = selected_mask & np.isin(seed_values, np.asarray(sorted(config_seeds)))
            config["samples"] = int(np.sum(config_mask))
            config["episodes_started"] = (
                int(np.unique(arrays["episode_id"][config_mask]).size) if np.any(config_mask) else 0
            )

    subset_manifest = {
        **manifest,
        "schema_version": PC_BC_SUBSET_MANIFEST_SCHEMA,
        "dataset_path": str(output_dataset.resolve()),
        "dataset_sha256": _sha256(output_dataset),
        "sample_count": int(subset["actor_state"].shape[0]),
        "episode_count": int(episode_ids.size),
        "subset": {
            "source_dataset_path": str(input_dataset.resolve()),
            "source_dataset_sha256": _sha256(input_dataset),
            "source_manifest_path": (
                str(input_manifest.resolve()) if input_manifest.is_file() else None
            ),
            "source_manifest_sha256": (
                _sha256(input_manifest) if input_manifest.is_file() else None
            ),
            "seed_list": list(selected_seeds),
            "selection_rule": "complete seed trajectories; episode ids remapped contiguously",
        },
    }
    write_json_document(output_manifest, subset_manifest)
    return subset_manifest


def sample_pc_demonstration_dataset(
    input_path: Path | str,
    output_dir: Path | str,
    sample_count: int,
    *,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Sample an exact number of transitions without changing transition values."""

    input_dataset, input_manifest = _dataset_and_manifest(Path(input_path))
    output = Path(output_dir)
    output_dataset = output / "demos.npz"
    output_manifest = output / "demos.manifest.json"
    if not overwrite and (output_dataset.exists() or output_manifest.exists()):
        raise FileExistsError(f"output already contains PC demonstrations: {output}")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    with np.load(input_dataset, allow_pickle=False) as source:
        fields = tuple(source.files)
        arrays = {field: np.asarray(source[field]) for field in fields}
    if "episode_id" not in arrays or "actor_state" not in arrays:
        raise ValueError(f"{input_dataset}: missing episode_id or actor_state")
    total = int(arrays["actor_state"].shape[0])
    if sample_count > total:
        raise ValueError(f"sample_count {sample_count} exceeds available samples {total}")
    indices = np.sort(np.random.default_rng(seed).choice(total, size=sample_count, replace=False))
    selected_original_episode_ids = arrays["episode_id"][indices].astype(np.int64, copy=False)
    episode_ids = np.unique(selected_original_episode_ids)
    remap = {int(old): int(new) for new, old in enumerate(episode_ids)}

    sampled: dict[str, np.ndarray] = {}
    for field, value in arrays.items():
        if value.shape == ():
            sampled[field] = value
            continue
        if value.ndim == 0 or value.shape[0] != total:
            raise ValueError(f"{input_dataset}: {field} is not aligned with actor_state")
        sampled[field] = value[indices]
    sampled["episode_id"] = np.asarray(
        [remap[int(value)] for value in selected_original_episode_ids],
        dtype=np.int64,
    )

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dataset, **sampled)

    manifest: dict[str, Any] = {}
    if input_manifest.is_file():
        manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    configs = manifest.get("configs")
    if isinstance(configs, list) and configs:
        declared = [int(config.get("samples", 0)) for config in configs if isinstance(config, dict)]
        if sum(declared) == total:
            offset = 0
            for config, count in zip(configs, declared, strict=True):
                if not isinstance(config, dict):
                    continue
                selected_in_config = indices[(indices >= offset) & (indices < offset + count)]
                config["samples"] = int(selected_in_config.size)
                config["episodes_started"] = (
                    int(np.unique(arrays["episode_id"][selected_in_config]).size)
                    if selected_in_config.size
                    else 0
                )
                offset += count

    sampled_manifest = {
        **manifest,
        "schema_version": PC_BC_SUBSET_MANIFEST_SCHEMA,
        "dataset_path": str(output_dataset.resolve()),
        "dataset_sha256": _sha256(output_dataset),
        "sample_count": int(sampled["actor_state"].shape[0]),
        "episode_count": int(episode_ids.size),
        "sample": {
            "source_dataset_path": str(input_dataset.resolve()),
            "source_dataset_sha256": _sha256(input_dataset),
            "source_manifest_path": (
                str(input_manifest.resolve()) if input_manifest.is_file() else None
            ),
            "source_manifest_sha256": (
                _sha256(input_manifest) if input_manifest.is_file() else None
            ),
            "random_seed": int(seed),
            "sample_count": int(sample_count),
            "selection_rule": "uniform transitions without replacement; episode ids remapped contiguously",
        },
    }
    write_json_document(output_manifest, sampled_manifest)
    return sampled_manifest


__all__ = [
    "PC_BC_SUBSET_MANIFEST_SCHEMA",
    "sample_pc_demonstration_dataset",
    "subset_pc_demonstration_dataset",
]
