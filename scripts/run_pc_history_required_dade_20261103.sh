#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_history_required_dade_20261103"
mkdir -p "$QUEUE_ROOT"
exec >> "$QUEUE_ROOT/queue.log" 2>&1

TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_history_required_dade_20261103"
TRAIN_LOGS="$ROOT/logs/pc_server_scale_history_required_dade_20261103/train"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_history_required_dade_eval_20261103"
EVAL_LOGS="$ROOT/logs/pc_server_scale_history_required_dade_eval_20261103"
DIAG_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_history_required_dade_diagnostics_20261103"
GATE_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_history_required_dade_gate_20261103"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_history_required_dade_eval_summary_20261103"
ARM='frozen_bc_residual_dade_history_required'
SHARED_GPU_LOCK_DIR="$ROOT/logs/pc_server_scale_C_20261103/train_gpu1_early_20261103/locks"

run_step() {
  local name="$1"
  shift
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  return 0
}

printf '\n===== HISTORY-REQUIRED DADE QUEUE START %s =====\n' "$(date -Is)"

run_step TRAIN_HISTORY_REQUIRED_DADE env \
  STAGE=history_required_dade \
  ARMS="$ARM" \
  SEEDS='91430 91431 91432' ITERATIONS=40 \
  GPU_IDS='1' MIN_FREE_MIB=16500 GPU_WAIT_SECONDS=10800 \
  LOCK_DIR="$SHARED_GPU_LOCK_DIR" \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$TRAIN_ROOT" LOG_DIR="$TRAIN_LOGS" \
  bash scripts/run_pc_server_scale_20261103.sh

for seed in 91430 91431 91432; do
  run_step "DIAG_${ARM}_seed${seed}" \
    env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PY" scripts/diagnose_pc_dade_history.py \
    --training-config "$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_required_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json" \
    --checkpoint "$TRAIN_ROOT/${ARM}_seed${seed}/model_39.pt" \
    --output "$DIAG_ROOT/${ARM}_seed${seed}" \
    --seed-start 0 --seed-count 32 --max-steps 512 \
    --device cpu --dtype float64
done

run_step HISTORY_REQUIRED_GATE \
  "$PY" scripts/check_pc_dade_history_gate.py \
  --diagnostic-root "$DIAG_ROOT" \
  --output "$GATE_ROOT/gate.json"
GATE_RC=$?
printf 'HISTORY_REQUIRED_GATE rc=%s\n' "$GATE_RC"

common_eval=(
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4
  CPU_BASE=160 CPU_STRIDE=16
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0 SEED_COUNT=32
)

run_step EVAL_HISTORY_REQUIRED_FIXED env \
  STAGE=history_required_dade ARMS="$ARM" SEEDS='91430 91431 91432' \
  TAGS='default strict125' CHECKPOINTS='0 39' \
  "${common_eval[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$EVAL_LOGS" \
  bash scripts/eval_pc_server_scale_20261103.sh

for ablation in current-only history-zero; do
  run_step "EVAL_HISTORY_REQUIRED_${ablation}" env \
    STAGE=history_required_dade ARMS="$ARM" SEEDS='91430 91431 91432' \
    TAGS='default' CHECKPOINTS='39' HISTORY_ABLATION="$ablation" \
    "${common_eval[@]}" \
    RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="${EVAL_LOGS}_ablation" \
    bash scripts/eval_pc_server_scale_20261103.sh
done

if (( GATE_RC == 0 )); then
  run_step EVAL_HISTORY_REQUIRED_HIGHENV env \
    STAGE=history_required_dade ARMS="$ARM" SEEDS='91430 91431 91432' \
    TAGS='h1 h2 h4 c4' CHECKPOINTS='39' \
    "${common_eval[@]}" \
    RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="${EVAL_LOGS}_highenv" \
    bash scripts/eval_pc_server_scale_20261103.sh
else
  printf 'SKIP HIGHENV: history-required mechanism gate failed rc=%s\n' "$GATE_RC"
fi

run_step SUMMARY_HISTORY_REQUIRED \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$SUMMARY_ROOT" \
  --baseline-arm "$ARM"

printf '\n===== HISTORY-REQUIRED DADE QUEUE DONE %s gate_rc=%s =====\n' "$(date -Is)" "$GATE_RC"
