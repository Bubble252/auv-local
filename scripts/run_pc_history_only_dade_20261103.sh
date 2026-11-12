#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_history_only_dade_20261103"
mkdir -p "$QUEUE_ROOT"
exec >> "$QUEUE_ROOT/queue.log" 2>&1

printf '\n===== HISTORY-ONLY DADE QUEUE WAIT %s =====\n' "$(date -Is)"
exec 9> "$ROOT/logs/pc_expanded_queue_20261103/queue.lock"
flock 9
printf '===== HISTORY-ONLY DADE QUEUE START %s =====\n' "$(date -Is)"

TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_history_only_dade_20261103"
TRAIN_LOGS="$ROOT/logs/pc_server_scale_history_only_dade_20261103/train"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_history_only_dade_eval_20261103"
EVAL_LOGS="$ROOT/logs/pc_server_scale_history_only_dade_eval_20261103"
DIAG_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_history_only_dade_diagnostics_20261103"

run_step() {
  local name="$1"
  shift
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  return 0
}

run_training() {
  run_step TRAIN_HISTORY_ONLY_DADE env \
    STAGE=history_only_dade \
    ARMS='dade_history_only frozen_bc_residual_dade_history_only' \
    SEEDS='91430 91431 91432' ITERATIONS=40 \
    GPU_IDS='2 6' MIN_FREE_MIB=16500 \
    NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
    OUTPUT_ROOT="$TRAIN_ROOT" LOG_DIR="$TRAIN_LOGS" \
    bash scripts/run_pc_server_scale_20261103.sh
}

run_diagnostics() {
  local arm seed output
  for arm in dade_history_only frozen_bc_residual_dade_history_only; do
    for seed in 91430 91431 91432; do
      output="$DIAG_ROOT/${arm}_seed${seed}"
      run_step "DIAG_${arm}_seed${seed}" \
        "$PY" scripts/diagnose_pc_dade_history.py \
        --training-config "$ROOT/configs/lowlevel/$(case "$arm" in
          dade_history_only) printf '%s' 'odarrl_control_ppo_pc_c3_random_realistic_dade_history_only_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' ;;
          frozen_bc_residual_dade_history_only) printf '%s' 'odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_only_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' ;;
        esac)" \
        --checkpoint "$TRAIN_ROOT/${arm}_seed${seed}/model_39.pt" \
        --output "$output" \
        --seed-start 0 --seed-count 32 --max-steps 512 \
        --device cpu --dtype float64
    done
  done
}

run_training
run_diagnostics
printf '\n--- HISTORY_GATE START %s ---\n' "$(date -Is)"
"$PY" scripts/check_pc_dade_history_gate.py \
  --diagnostic-root "$DIAG_ROOT" \
  --output "$ROOT/artifacts/lowlevel/pc_server_scale_history_only_dade_gate_20261103/gate.json"
GATE_RC=$?
printf '%s\n' "--- HISTORY_GATE END rc=$GATE_RC $(date -Is) ---"

common_eval=(
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4
  CPU_BASE=160 CPU_STRIDE=16
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0 SEED_COUNT=32
)

run_step EVAL_HISTORY_ONLY_FIXED env \
  STAGE=history_only_dade \
  ARMS='dade_history_only frozen_bc_residual_dade_history_only' \
  SEEDS='91430 91431 91432' TAGS='default strict125' CHECKPOINTS='0 39' \
  "${common_eval[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$EVAL_LOGS" \
  bash scripts/eval_pc_server_scale_20261103.sh

if (( GATE_RC == 0 )); then
  run_step EVAL_HISTORY_ONLY_HIGHENV env \
    STAGE=history_only_dade \
    ARMS='dade_history_only frozen_bc_residual_dade_history_only' \
    SEEDS='91430 91431 91432' TAGS='h1 h2 h4 c4' CHECKPOINTS='39' \
    "${common_eval[@]}" \
    RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="${EVAL_LOGS}_highenv" \
    bash scripts/eval_pc_server_scale_20261103.sh
else
  printf 'SKIP HIGHENV: history-mechanism gate failed rc=%s\n' "$GATE_RC"
fi

run_step SUMMARY_HISTORY_ONLY_FROZEN_BASELINE \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$ROOT/artifacts/lowlevel/pc_server_scale_history_only_dade_eval_summary_20261103" \
  --baseline-arm frozen_bc_residual_dade_history_only

run_step SUMMARY_HISTORY_ONLY_DADE_BASELINE \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$ROOT/artifacts/lowlevel/pc_server_scale_history_only_dade_vs_scratch_summary_20261103" \
  --baseline-arm dade_history_only

printf '\n===== HISTORY-ONLY DADE QUEUE DONE %s gate_rc=%s =====\n' "$(date -Is)" "$GATE_RC"
