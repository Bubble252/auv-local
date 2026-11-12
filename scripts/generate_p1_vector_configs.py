from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "configs/rl/vector"

CANDIDATES = {
    "v0_c2": ROOT / "configs/rl/ablations/critic/c2.json",
    "v1_c3": ROOT / "configs/rl/ablations/critic/c3.json",
    "v2_sat7": ROOT / "configs/rl/ablations/saturation/s7.json",
    "v3_r5": ROOT / "configs/rl/ablations/reward_v2/r5.json",
}


def _analytic_payload(source: Path, candidate_id: str, total: int) -> dict[str, object]:
    payload = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    environment = payload["environment"]
    environment["sensor_id"] = "analytic-13sector"
    environment["perception"] = {
        "backend": "analytic_13sector",
        "mode": "analytic_13sector_geometry",
        "noise": False,
    }
    training = payload["training"]
    training["total_env_transitions"] = total
    training["update_to_data_ratio"] = 1.0
    training["eval_seeds"] = [0, 1, 2]
    payload["output"] = f"outputs/m4/p1/analytic_multienv/screening/{candidate_id}/seed_000"
    return payload


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for candidate_id, source in CANDIDATES.items():
        payload = _analytic_payload(source, candidate_id, 100_000)
        destination = OUTPUT_DIR / f"{candidate_id}.json"
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(destination)

    smoke = _analytic_payload(CANDIDATES["v1_c3"], "smoke", 20_000)
    smoke["output"] = "outputs/m4/p1/analytic_multienv/smoke/seed_000"
    smoke_path = OUTPUT_DIR / "smoke_c3.json"
    smoke_path.write_text(
        json.dumps(smoke, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(smoke_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
