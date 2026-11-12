from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_CORE_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "dones",
    "scenario_keys",
    "scenario_families",
    "episode_ids",
    "episode_steps",
    "teacher_success",
    "privileged",
    "next_privileged",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_demo_datasets(
    base_path: Path | str,
    extra_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Merge successful expert demos with DAgger correction transitions."""

    base = np.load(Path(base_path), allow_pickle=False)
    extra = np.load(Path(extra_path), allow_pickle=False)
    missing = [
        field for field in _CORE_FIELDS if field not in base.files or field not in extra.files
    ]
    if missing:
        raise ValueError(f"demo datasets are missing fields: {missing}")

    base_episode_ids = np.asarray(base["episode_ids"], dtype=np.int64)
    extra_episode_ids = np.asarray(extra["episode_ids"], dtype=np.int64)
    offset = int(base_episode_ids.max()) + 1 if len(base_episode_ids) else 0
    remapped_extra = extra_episode_ids + offset

    arrays: dict[str, np.ndarray] = {}
    for field in _CORE_FIELDS:
        arrays[field] = np.concatenate(
            (np.asarray(base[field]), np.asarray(extra[field])),
            axis=0,
        )
    arrays["episode_ids"] = np.concatenate((base_episode_ids, remapped_extra))
    base_source = (
        np.asarray(base["data_source"]).astype("<U16")
        if "data_source" in base.files
        else np.full(len(base_episode_ids), "expert_success", dtype="<U16")
    )
    extra_source = (
        np.asarray(extra["data_source"]).astype("<U16")
        if "data_source" in extra.files
        else np.full(len(remapped_extra), "dagger", dtype="<U16")
    )
    arrays["data_source"] = np.concatenate((base_source, extra_source))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    demo_path = output / "demos.npz"
    np.savez_compressed(demo_path, **arrays)  # type: ignore[arg-type]
    manifest = {
        "schema": "auv_p1_bc_merged_manifest.v1",
        "demo_path": str(demo_path),
        "demo_sha256": _sha256(demo_path),
        "base_path": str(base_path),
        "base_sha256": _sha256(Path(base_path)),
        "extra_path": str(extra_path),
        "extra_sha256": _sha256(Path(extra_path)),
        "base_transitions": int(len(base_episode_ids)),
        "extra_transitions": int(len(extra_episode_ids)),
        "transitions": int(len(arrays["observations"])),
        "episodes": int(len(np.unique(arrays["episode_ids"]))),
        "data_source_counts": {
            source: int(np.count_nonzero(arrays["data_source"] == source))
            for source in sorted(set(arrays["data_source"].tolist()))
        },
    }
    (output / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


__all__ = ["merge_demo_datasets"]
