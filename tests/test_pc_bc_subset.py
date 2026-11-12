from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from auv_lowlevel.batch.pc_bc_subset import (
    sample_pc_demonstration_dataset,
    subset_pc_demonstration_dataset,
)


def _write_dataset(root: Path) -> None:
    root.mkdir()
    np.savez_compressed(
        root / "demos.npz",
        actor_state=np.arange(8, dtype=np.float32).reshape(4, 2),
        command=np.zeros((4, 0), dtype=np.float32),
        action=np.ones((4, 3), dtype=np.float32),
        episode_id=np.asarray([10, 10, 20, 20], dtype=np.int64),
        seed=np.asarray([1, 1, 2, 2], dtype=np.int64),
        step=np.asarray([0, 1, 0, 1], dtype=np.int64),
        actual_actuator=np.zeros((4, 3), dtype=np.float32),
        reward=np.zeros((4,), dtype=np.float32),
        actor_path=np.zeros((4, 8), dtype=np.float32),
        observation_schema=np.asarray("pc_v1"),
        action_envelope=np.asarray("pc20"),
    )
    manifest = {
        "schema_version": "auv_pc_bc_merged_manifest.v1",
        "sample_count": 4,
        "episode_count": 2,
        "path_dim": 8,
        "configs": [
            {"source": "teacher", "seed_list": [1], "samples": 2, "episodes_started": 1},
            {"source": "teacher", "seed_list": [2], "samples": 2, "episodes_started": 1},
        ],
    }
    (root / "demos.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_subset_selects_complete_seed_and_remaps_episode_ids(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(source)

    manifest = subset_pc_demonstration_dataset(
        source,
        tmp_path / "subset",
        [1],
    )

    with np.load(tmp_path / "subset" / "demos.npz", allow_pickle=False) as payload:
        assert payload["actor_state"].tolist() == [[0.0, 1.0], [2.0, 3.0]]
        assert payload["episode_id"].tolist() == [0, 0]
        assert payload["seed"].tolist() == [1, 1]
    assert manifest["sample_count"] == 2
    assert manifest["episode_count"] == 1
    assert manifest["configs"][0]["samples"] == 2
    assert manifest["configs"][0]["episodes_started"] == 1
    assert manifest["configs"][1]["samples"] == 0
    assert manifest["subset"]["seed_list"] == [1]


def test_subset_rejects_missing_seed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(source)

    with pytest.raises(ValueError, match="requested seeds are absent"):
        subset_pc_demonstration_dataset(source, tmp_path / "subset", [99])


def test_sample_selects_exact_count_and_updates_config_counts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(source)

    manifest = sample_pc_demonstration_dataset(
        source,
        tmp_path / "sample",
        3,
        seed=7,
    )

    with np.load(tmp_path / "sample" / "demos.npz", allow_pickle=False) as payload:
        assert payload["actor_state"].shape == (3, 2)
        assert payload["episode_id"].shape == (3,)
    assert manifest["sample_count"] == 3
    assert manifest["sample"]["random_seed"] == 7
    assert sum(config["samples"] for config in manifest["configs"]) == 3


def test_sample_rejects_too_many_samples(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(source)

    with pytest.raises(ValueError, match="exceeds available samples"):
        sample_pc_demonstration_dataset(source, tmp_path / "sample", 5)
