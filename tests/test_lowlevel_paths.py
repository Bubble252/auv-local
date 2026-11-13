from __future__ import annotations

from pathlib import Path

from auv_lowlevel._paths import resolve_from_root


def test_relative_path_is_joined_to_explicit_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"

    assert resolve_from_root(root, "configs/run.json") == root / "configs/run.json"


def test_absolute_path_is_returned_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    absolute = tmp_path / "elsewhere" / "run.json"

    assert resolve_from_root(root, absolute) == absolute


def test_path_input_is_not_required_to_exist(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    relative = Path("missing/../run.json")

    resolved = resolve_from_root(root, relative)

    assert resolved == root / relative
    assert resolved.exists() is False


def test_relative_resolution_does_not_read_working_directory(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    monkeypatch.chdir(tmp_path)

    assert resolve_from_root(root, "artifacts/run") == root / "artifacts/run"
