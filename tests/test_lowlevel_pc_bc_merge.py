from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auv_lowlevel.batch.pc_bc import load_pc_demonstration_dataset
from auv_lowlevel.batch.pc_bc_merge import merge_pc_demonstration_datasets
from auv_lowlevel.batch.pc_normalization import PCNormalization


def _write_dataset(
    path: Path,
    *,
    episode_start: int,
    seed_start: int,
    include_current_estimate: bool = False,
    include_actor_path: bool = False,
    path_dim: int = 8,
) -> None:
    rng = np.random.default_rng(seed_start)
    samples = 24
    episode_id = np.repeat(np.arange(episode_start, episode_start + 2), samples // 2)
    payload = {
        "actor_state": rng.normal(size=(samples, 12)).astype(np.float32),
        "command": rng.normal(size=(samples, 3)).astype(np.float32),
        "action": rng.uniform(-1.0, 1.0, size=(samples, 3)).astype(np.float32),
        "episode_id": episode_id.astype(np.int64),
        "seed": np.repeat(np.arange(seed_start, seed_start + 2), samples // 2).astype(np.int64),
        "step": np.tile(np.arange(samples // 2), 2).astype(np.int64),
        "actual_actuator": rng.normal(size=(samples, 3)).astype(np.float32),
        "reward": rng.normal(size=samples).astype(np.float32),
        "observation_schema": np.asarray("pc_v1"),
        "action_envelope": np.asarray("pc20"),
    }
    if include_current_estimate:
        payload["current_estimate_body"] = rng.normal(size=(samples, 3)).astype(np.float32)
    if include_actor_path:
        payload["actor_path"] = rng.normal(size=(samples, path_dim)).astype(np.float32)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "demos.npz", **payload)
    (path / "demos.manifest.json").write_text(
        json.dumps(
            {
                "configs": [
                    {
                        "config_name": path.name,
                        "pc_stage": path.name,
                        "sensor_profile": "realistic",
                        "normalization": PCNormalization().as_dict(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_merge_pc_datasets_remaps_episode_ids_and_preserves_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, episode_start=0, seed_start=10)
    _write_dataset(second, episode_start=0, seed_start=20)

    output = tmp_path / "merged"
    manifest = merge_pc_demonstration_datasets(
        [first / "demos.npz", second / "demos.npz"],
        output,
    )
    dataset, metadata = load_pc_demonstration_dataset(output / "demos.npz")

    assert dataset.sample_count == 48
    assert dataset.episode_count == 4
    assert np.array_equal(np.unique(dataset.episode_id), np.arange(4))
    assert manifest["sample_count"] == 48
    assert manifest["episode_count"] == 4
    assert len(manifest["inputs"]) == 2
    assert len(manifest["configs"]) == 2
    assert metadata["normalization"] == PCNormalization().as_dict()


def test_merge_pc_datasets_rejects_inconsistent_normalization(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, episode_start=0, seed_start=10)
    _write_dataset(second, episode_start=0, seed_start=20)
    manifest_path = second / "demos.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configs"][0]["normalization"]["position_scale_m"] = 200.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        merge_pc_demonstration_datasets([first / "demos.npz", second / "demos.npz"], tmp_path / "bad")
    except ValueError as error:
        assert "inconsistent normalization" in str(error)
    else:
        raise AssertionError("inconsistent normalization must be rejected")


def test_merge_pc_datasets_preserves_current_estimate_body(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, episode_start=0, seed_start=10, include_current_estimate=True)
    _write_dataset(second, episode_start=0, seed_start=20, include_current_estimate=True)

    output = tmp_path / "merged"
    manifest = merge_pc_demonstration_datasets(
        [first / "demos.npz", second / "demos.npz"],
        output,
    )
    dataset, _ = load_pc_demonstration_dataset(output / "demos.npz")

    assert dataset.current_estimate_body is not None
    assert dataset.current_estimate_body.shape == (48, 3)
    assert manifest["current_estimate_body"] is True
    assert all(item["current_estimate_body"] for item in manifest["inputs"])


def test_merge_pc_datasets_rejects_mixed_optional_fields(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, episode_start=0, seed_start=10, include_current_estimate=True)
    _write_dataset(second, episode_start=0, seed_start=20)

    try:
        merge_pc_demonstration_datasets([first / "demos.npz", second / "demos.npz"], tmp_path / "bad")
    except ValueError as error:
        assert "optional fields differ" in str(error)
    else:
        raise AssertionError("mixed optional fields must be rejected")


def test_merge_pc_datasets_preserves_actor_path(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, episode_start=0, seed_start=10, include_actor_path=True)
    _write_dataset(second, episode_start=0, seed_start=20, include_actor_path=True)

    output = tmp_path / "merged"
    manifest = merge_pc_demonstration_datasets(
        [first / "demos.npz", second / "demos.npz"],
        output,
    )
    dataset, _ = load_pc_demonstration_dataset(output / "demos.npz")

    assert dataset.actor_path is not None
    assert dataset.actor_path.shape == (48, 8)
    assert dataset.path_dim == 8
    assert manifest["path_dim"] == 8
    assert all(item["actor_path"] for item in manifest["inputs"])


def test_merge_pc_datasets_preserves_12d_actor_path(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(
        first,
        episode_start=0,
        seed_start=10,
        include_actor_path=True,
        path_dim=12,
    )
    _write_dataset(
        second,
        episode_start=0,
        seed_start=20,
        include_actor_path=True,
        path_dim=12,
    )

    output = tmp_path / "merged"
    manifest = merge_pc_demonstration_datasets(
        [first / "demos.npz", second / "demos.npz"],
        output,
    )
    dataset, _ = load_pc_demonstration_dataset(output / "demos.npz")

    assert dataset.actor_path is not None
    assert dataset.actor_path.shape == (48, 12)
    assert dataset.path_dim == 12
    assert manifest["path_dim"] == 12
    assert all(item["path_dim"] == 12 for item in manifest["inputs"])


def test_merge_pc_datasets_rejects_mixed_path_dimensions(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, episode_start=0, seed_start=10, include_actor_path=True, path_dim=8)
    _write_dataset(second, episode_start=0, seed_start=20, include_actor_path=True, path_dim=12)

    try:
        merge_pc_demonstration_datasets(
            [first / "demos.npz", second / "demos.npz"], tmp_path / "bad"
        )
    except ValueError as error:
        assert "actor_path width differs" in str(error)
    else:
        raise AssertionError("mixed path dimensions must be rejected")
