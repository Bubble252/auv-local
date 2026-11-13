"""Tests for the internal JSON reading helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from auv_lowlevel._json import load_json_value


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
