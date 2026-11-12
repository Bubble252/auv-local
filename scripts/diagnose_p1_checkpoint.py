from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from auv_rl.ddpg.agent import DDPGAgent
from auv_rl.ddpg.diagnostics import diagnose_critic, write_critic_health


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a P1 DDPG checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path)
    return parser.parse_args()


def _read_eval_rows(path: Path | None) -> list[dict[str, str]] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("eval CSV contains no rows")
    return rows


def main() -> int:
    args = _parse_args()
    payload = np.load(args.probes, allow_pickle=False)
    observations = [np.asarray(payload[key], dtype=np.float32) for key in payload.files]
    if not observations:
        raise ValueError("probe archive contains no observations")
    stacked = np.stack(observations, axis=0)
    agent = DDPGAgent.load(args.checkpoint, device="cpu")
    health = diagnose_critic(agent, stacked)
    health["checkpoint"] = str(args.checkpoint)
    health["probe_file"] = str(args.probes)
    health["probe_keys"] = list(payload.files)
    write_critic_health(
        args.output,
        health,
        eval_rows=_read_eval_rows(args.eval_csv),
    )
    print(json.dumps(health, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
