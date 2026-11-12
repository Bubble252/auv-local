#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
QUEUE_ROOT="$ROOT/logs/pc_expanded_queue_20261103"
mkdir -p "$QUEUE_ROOT"
exec >> "$QUEUE_ROOT/gapfill.log" 2>&1

exec 8> "$QUEUE_ROOT/queue.lock"
printf '\n===== GAPFILL WAIT %s =====\n' "$(date -Is)"
flock 8
printf '===== GAPFILL START %s =====\n' "$(date -Is)"

env STAGE=B ARMS='dade_head_dropout05' SEEDS='91431 91432' \
  GPU_IDS='2 6' MIN_FREE_MIB=16500 ITERATIONS=40 \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_20261103/train_retry_after_queue" \
  bash scripts/run_pc_server_scale_20261103.sh
printf 'GAPFILL B_RETRY rc=%s %s\n' "$?" "$(date -Is)"

env STAGE=B ARMS='nodade dade dade_head dade_head_dropout05' SEEDS='91430 91431 91432' \
  TAGS='default strict125' CHECKPOINTS='0 10 20 30 39' \
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4 CPU_BASE=160 CPU_STRIDE=16 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 SEED_START=0 SEED_COUNT=64 \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_eval_20261103_retry" \
  bash scripts/eval_pc_server_scale_20261103.sh
printf 'GAPFILL B_EVAL rc=%s %s\n' "$?" "$(date -Is)"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/lowlevel/pc_server_scale_frozen_bc_residual_20261103")
rows = []
for path in sorted(root.glob("*/training_manifest.json")):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "run": path.parent.name,
            "status": manifest.get("status"),
            "initial_residual_abs_max": manifest.get("initial_residual_abs_max"),
            "frozen_bc_audit": manifest.get("frozen_bc_audit"),
        }
    )
out = root / "frozen_bc_audit_summary.json"
out.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY

summarize() {
  local name="$1"
  shift
  "$PY" scripts/summarize_pc_server_evals.py "$@"
  printf 'GAPFILL SUMMARY %s rc=%s %s\n' "$name" "$?" "$(date -Is)"
}

summarize B --eval-root "$ROOT/artifacts/lowlevel/pc_server_scale_B_eval_20261103" \
  --output-dir "$ROOT/artifacts/lowlevel/pc_server_scale_B_eval_summary_20261103" \
  --baseline-arm nodade

summarize FROZEN --eval-root "$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_eval_20261103" \
  --output-dir "$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_eval_summary_20261103" \
  --baseline-arm frozen_bc_residual_nodade

summarize C --eval-root "$ROOT/artifacts/lowlevel/pc_server_scale_C_eval_20261103" \
  --output-dir "$ROOT/artifacts/lowlevel/pc_server_scale_C_eval_summary_20261103" \
  --baseline-arm nodade

summarize B_EXT --eval-root "$ROOT/artifacts/lowlevel/pc_server_scale_B_ext_eval_20261103" \
  --output-dir "$ROOT/artifacts/lowlevel/pc_server_scale_B_ext_eval_summary_20261103" \
  --baseline-arm nodade

summarize SIDECHANNEL10 --eval-root "$ROOT/artifacts/lowlevel/pc_server_scale_B_sidechannel10_eval_20261103" \
  --output-dir "$ROOT/artifacts/lowlevel/pc_server_scale_B_sidechannel10_eval_summary_20261103" \
  --baseline-arm nodade

printf '===== GAPFILL DONE %s =====\n' "$(date -Is)"
