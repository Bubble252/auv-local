"""Internal UTF-8 JSON text reader.

The helper centralizes file decoding only. Schema validation remains with each
configuration parser, and decode errors continue to propagate unchanged.
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
