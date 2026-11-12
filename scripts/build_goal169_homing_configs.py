#!/usr/bin/env python3
"""Build Goal169 terminal-alignment-only environment configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/lowlevel/goal165/env_h3_gain_near10_2_eval_nu_ground.json"
OUTPUT = ROOT / "configs/lowlevel/goal169"
VARIANT_KEYS = (
    "terminal_alignment_min_m",
    "terminal_alignment_max_m",
)


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for distance in (20, 30, 40, 50):
        payload = copy.deepcopy(baseline)
        payload["guidance"]["terminal_alignment_min_m"] = float(distance)
        payload["guidance"]["terminal_alignment_max_m"] = float(distance)
        path = OUTPUT / f"env_terminal_alignment_{distance}_eval_nu_ground.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
