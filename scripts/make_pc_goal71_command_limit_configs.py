#!/usr/bin/env python3
"""Generate the two Goal71 limiter-on configs from frozen Goal59 configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "lowlevel"

SCENARIOS = {
    "default": {
        "source": CONFIGS / "pc_goal59_terminal_bounded_default_20261109.json",
        "target": CONFIGS / "pc_goal71_command_rate_limit_default_20261110.json",
        "experiment": "pc_goal71_command_rate_limit_default_20261110",
        "output": "artifacts/lowlevel/pc_goal71_command_rate_limit_default",
    },
    "strict125": {
        "source": CONFIGS / "pc_goal59_terminal_bounded_strict125_20261109.json",
        "target": CONFIGS / "pc_goal71_command_rate_limit_strict125_20261110.json",
        "experiment": "pc_goal71_command_rate_limit_strict125_20261110",
        "output": "artifacts/lowlevel/pc_goal71_command_rate_limit_strict125",
    },
}


def derive(source: Path, *, experiment: str, output: str) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    guidance = payload["guidance"]
    for key in (
        "heading_rate_limit_rad_s",
        "pitch_rate_limit_rad_s",
        "speed_rate_limit_mps2",
    ):
        if key not in guidance:
            raise ValueError(f"{source}: missing guidance.{key}")
    guidance["command_rate_limit_enabled"] = True
    payload["experiment"]["name"] = experiment
    payload["experiment"]["output"] = output
    return payload


def write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    for scenario, spec in SCENARIOS.items():
        expected = derive(
            Path(spec["source"]),
            experiment=str(spec["experiment"]),
            output=str(spec["output"]),
        )
        target = Path(spec["target"])
        if args.check:
            actual = json.loads(target.read_text(encoding="utf-8"))
            if actual != expected:
                raise SystemExit(f"{scenario}: target config is stale: {target}")
            continue
        write_payload(target, expected)
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
