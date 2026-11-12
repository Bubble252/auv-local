#!/usr/bin/env bash
# Goal162 integration smoke: 4 envs x 2048 transitions on one GPU.
set -euo pipefail

GPU=${1:-0}
ARM=${ARM:-G1_gain_near10_2}
SEED=${SEED:-86201}

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal162_controller_gain_20261112}
SMOKE_ROOT=${SMOKE_ROOT:-$ROOT/smoke}
OUTPUT="$SMOKE_ROOT/${ARM}_seed${SEED}"
LOG="$SMOKE_ROOT/${ARM}_seed${SEED}.log"

cd "$REPO"
mkdir -p "$SMOKE_ROOT"
rm -rf "$OUTPUT"

echo "START Goal162 smoke gpu=$GPU arm=$ARM seed=$SEED"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="src:$SB3" \
"$PY" scripts/train_pc_command_residual_sac.py \
  --config "configs/lowlevel/goal162/train_${ARM,,}_seed${SEED}.json" \
  --output "$OUTPUT" \
  --smoke \
  --seed "$SEED" \
  --device cuda \
  --dtype float64 \
  --sb3-root "$SB3" \
  --overwrite >"$LOG" 2>&1

"$PY" - "$OUTPUT/training_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
speed = payload.get("speed_contract") or {}
checks = {
    "algorithm_is_sac": payload.get("algorithm") == "SAC",
    "action_interface": payload.get("action_interface") == "controller_gain",
    "action_dim": payload.get("action_dim") == 2,
    "direct_commands_inactive": payload.get("direct_commands_active") is False,
    "gain_residual_enabled": bool(
        (payload.get("controller_gain_residual") or {}).get("enabled")
    ),
    "speed_contract_ok": bool(speed.get("ok")),
    "reference_speed_1p2": abs(float(speed.get("reference_speed_mps", -1.0)) - 1.2) < 1e-9,
    "max_speed_2p5": abs(float(speed.get("max_speed_mps", -1.0)) - 2.5) < 1e-9,
    "worst_case_within_2p5": float(speed.get("worst_case_ground_speed_mps", 1e9)) <= 2.5 + 1e-9,
    "actual_transitions_2048": int(payload.get("total_env_transitions_actual", -1)) == 2048,
    "num_envs_4": int(payload.get("num_envs", -1)) == 4,
    "finite": bool(payload.get("finite", True)),
}
failed = [name for name, ok in checks.items() if not ok]
print(json.dumps({"manifest": str(path), "checks": checks}, indent=2))
if failed:
    raise SystemExit("Goal162 smoke failed: " + ", ".join(failed))
PY

echo "DONE Goal162 smoke output=$OUTPUT"
