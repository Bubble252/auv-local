"""Tests for the internal JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auv_lowlevel._json import load_json_value, write_json_document


def test_load_json_value_reads_object(tmp_path: Path) -> None:
    """The helper returns the decoded object without schema changes."""
    path = tmp_path / "config.json"
    path.write_text('{"seed": 7, "name": "demo"}\n', encoding="utf-8")

    assert load_json_value(path) == {"seed": 7, "name": "demo"}


def test_load_json_value_reads_array(tmp_path: Path) -> None:
    """The helper remains a generic JSON reader, not a config-only parser."""
    path = tmp_path / "values.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert load_json_value(path) == [1, 2, 3]


def test_load_json_value_preserves_decode_error(tmp_path: Path) -> None:
    """Malformed JSON keeps raising the standard decoder exception."""
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_json_value(path)


def test_write_json_document_preserves_manifest_format(tmp_path: Path) -> None:
    """Written bytes keep sorted keys, two-space indent, and a final newline."""
    path = tmp_path / "manifest.json"

    write_json_document(path, {"z": 1, "a": [2]})

    assert path.read_bytes() == b'{\n  "a": [\n    2\n  ],\n  "z": 1\n}\n'


def test_write_json_document_does_not_create_parent(tmp_path: Path) -> None:
    """The helper preserves the legacy requirement that callers create parents."""
    with pytest.raises(FileNotFoundError):
        write_json_document(tmp_path / "missing" / "manifest.json", {"ok": True})
