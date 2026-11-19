from __future__ import annotations

from pathlib import Path

import pytest

from auv_lowlevel.checkpoint_retention import prune_checkpoints


def _write(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("x", encoding="utf-8")
    return path


def test_keeps_newest_n(tmp_path: Path) -> None:
    for iteration in (10, 20, 30, 40, 50):
        _write(tmp_path, f"model_{iteration}.pt")

    removed = prune_checkpoints(tmp_path, keep_last=2)

    assert {path.name for path in removed} == {"model_10.pt", "model_20.pt", "model_30.pt"}
    assert sorted(path.name for path in tmp_path.glob("model_*.pt")) == [
        "model_40.pt",
        "model_50.pt",
    ]


def test_keeps_milestones_and_recent(tmp_path: Path) -> None:
    for iteration in (10, 20, 50, 100, 110, 120, 200, 210):
        _write(tmp_path, f"model_{iteration}.pt")

    removed = prune_checkpoints(tmp_path, keep_last=3, keep_every=100)

    assert {path.name for path in removed} == {
        "model_10.pt",
        "model_20.pt",
        "model_50.pt",
        "model_110.pt",
    }
    assert sorted(path.name for path in tmp_path.glob("model_*.pt")) == [
        "model_100.pt",
        "model_120.pt",
        "model_200.pt",
        "model_210.pt",
    ]


def test_never_touches_named_artifacts(tmp_path: Path) -> None:
    _write(tmp_path, "model_10.pt")
    untouched = [
        _write(tmp_path, "model_initial.pt"),
        _write(tmp_path, "final.pt"),
    ]
    curriculum = tmp_path / "curriculum_checkpoints"
    curriculum.mkdir()
    untouched.append(_write(curriculum, "c0_000000010.pt"))

    prune_checkpoints(tmp_path, keep_last=0, keep_every=0)
    prune_checkpoints(tmp_path, keep_last=0, keep_every=100)

    assert all(path.is_file() for path in untouched)
    assert not (tmp_path / "model_10.pt").exists()


def test_protect_is_never_removed(tmp_path: Path) -> None:
    keep = _write(tmp_path, "model_10.pt")
    _write(tmp_path, "model_20.pt")

    removed = prune_checkpoints(tmp_path, keep_last=0, keep_every=100, protect=(keep,))

    assert [path.name for path in removed] == ["model_20.pt"]
    assert keep.is_file()


def test_negative_values_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path, keep_last=-1)
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path, keep_every=-1)
