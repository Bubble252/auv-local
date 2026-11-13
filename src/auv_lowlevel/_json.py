"""Internal UTF-8 JSON text helpers.

The helpers centralize file decoding and stable document output only. Schema
validation remains with each caller, and filesystem errors propagate unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_value(path: Path | str) -> Any:
    """Read and decode one UTF-8 JSON document.

    Args:
        path: Filesystem path to the JSON document.

    Returns:
        The decoded JSON value without schema coercion.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the document is malformed.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_document(path: Path | str, payload: Any) -> None:
    """Write one JSON document with sorted keys and a trailing newline.

    Args:
        path: Destination path. Parent directories must already exist.
        payload: JSON-serializable value to encode.

    Raises:
        OSError: If the destination cannot be written.
        TypeError: If the payload is not JSON serializable.
    """
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8")
