#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="${QUEUE_ROOT:-$ROOT/logs/pc_expanded_queue_20261103}"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'expanded queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

printf '\n===== EXPANDED QUEUE START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s root=%s\n' "$(hostname)" "$$" "$ROOT"

run_training() {
  local name="$1"
  shift
  printf '\n--- TRAIN %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- TRAIN $name END rc=$rc $(date -Is) ---"
  return 0
}

run_eval() {
  local name="$1"
  shift
  printf '\n--- EVAL %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- EVAL $name END rc=$rc $(date -Is) ---"
  return 0
}

COMMON_TRAIN_ENV=(
  GPU_IDS='2 6'
  MIN_FREE_MIB=16500
  NUM_ENVS=4096
  NUM_STEPS=256
  NUM_MINI_BATCHES=128
)

# 1. Close the B-level dropout05 gap without touching completed runs.
run_training B_retry env \
  STAGE=B ARMS='dade_head_dropout05' SEEDS='91431 91432' \
  ITERATIONS=40 "${COMMON_TRAIN_ENV[@]}" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_20261103/train_retry_20261103" \
  bash scripts/run_pc_server_scale_20261103.sh

# 2. Frozen PC BC prior plus trainable residual policy, four arms x three seeds.
run_training FROZEN_BC_RESIDUAL env \
  STAGE=frozen_bc_residual \
  ARMS='frozen_bc_residual_nodade frozen_bc_residual_dade frozen_bc_residual_dade_head frozen_bc_residual_dade_sidechannel10' \
  SEEDS='91430 91431 91432' ITERATIONS=100 "${COMMON_TRAIN_ENV[@]}" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_20261103/train" \
  bash scripts/run_pc_server_scale_20261103.sh

# 3. C-level no-DADE / DADE scale comparison.
run_training C_SCALE env \
  STAGE=C ARMS='nodade dade' SEEDS='91430 91431 91432' \
  ITERATIONS=100 "${COMMON_TRAIN_ENV[@]}" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_C_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_C_20261103/train" \
  bash scripts/run_pc_server_scale_20261103.sh

# 4. Extend the ordinary B-level arms to five training seeds.
run_training B_EXT env \
  STAGE=B_ext ARMS='nodade dade dade_head dade_head_dropout05' SEEDS='91433 91434' \
  ITERATIONS=40 "${COMMON_TRAIN_ENV[@]}" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_ext_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_ext_20261103/train" \
  bash scripts/run_pc_server_scale_20261103.sh

# 5. Side-channel-only dropout, kept separate from the old full-dropout arm.
run_training B_SIDECHANNEL10 env \
  STAGE=B_sidechannel10 ARMS='dade_head_sidechannel10' SEEDS='91430 91431 91432' \
  ITERATIONS=40 "${COMMON_TRAIN_ENV[@]}" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_sidechannel10_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_sidechannel10_20261103/train" \
  bash scripts/run_pc_server_scale_20261103.sh

COMMON_EVAL_ENV=(
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4
  CPU_BASE=160 CPU_STRIDE=16
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0 SEED_COUNT=64
)

# 6. Fixed-seed default and strict-radius evaluation.
run_eval B_FIXED env \
  STAGE=B ARMS='nodade dade dade_head dade_head_dropout05' SEEDS='91430 91431 91432' \
  TAGS='default strict125' CHECKPOINTS='0 10 20 30 39' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_eval_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_eval FROZEN_FIXED env \
  STAGE=frozen_bc_residual \
  ARMS='frozen_bc_residual_nodade frozen_bc_residual_dade frozen_bc_residual_dade_head frozen_bc_residual_dade_sidechannel10' \
  SEEDS='91430 91431 91432' TAGS='default strict125' \
  CHECKPOINTS='initial 0 20 40 60 80 99' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_eval_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_eval C_FIXED env \
  STAGE=C ARMS='nodade dade' SEEDS='91430 91431 91432' \
  TAGS='default strict125' CHECKPOINTS='0 20 40 60 80 99' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_C_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_C_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_C_eval_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_eval B_EXT_FIXED env \
  STAGE=B_ext ARMS='nodade dade dade_head dade_head_dropout05' SEEDS='91433 91434' \
  TAGS='default strict125' CHECKPOINTS='0 10 20 30 39' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_ext_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_ext_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_ext_eval_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_eval SIDECHANNEL10_FIXED env \
  STAGE=B_sidechannel10 ARMS='dade_head_sidechannel10' SEEDS='91430 91431 91432' \
  TAGS='default strict125' CHECKPOINTS='0 10 20 30 39' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_sidechannel10_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_sidechannel10_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_sidechannel10_eval_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

# 7. Mechanism gate: history ablations on the final frozen/C checkpoints.
for ablation in current-only history-zero; do
  run_eval "FROZEN_${ablation}" env \
    STAGE=frozen_bc_residual \
    ARMS='frozen_bc_residual_nodade frozen_bc_residual_dade frozen_bc_residual_dade_head frozen_bc_residual_dade_sidechannel10' \
    SEEDS='91430 91431 91432' TAGS='default' CHECKPOINTS='99' \
    HISTORY_ABLATION="$ablation" "${COMMON_EVAL_ENV[@]}" \
    RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_20261103" \
    OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_eval_20261103" \
    LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_eval_20261103" \
    bash scripts/eval_pc_server_scale_20261103.sh

  run_eval "C_${ablation}" env \
    STAGE=C ARMS='nodade dade' SEEDS='91430 91431 91432' \
    TAGS='default' CHECKPOINTS='99' HISTORY_ABLATION="$ablation" \
    "${COMMON_EVAL_ENV[@]}" \
    RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_C_20261103" \
    OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_C_eval_20261103" \
    LOG_DIR="$ROOT/logs/pc_server_scale_C_eval_20261103" \
    bash scripts/eval_pc_server_scale_20261103.sh
done

# 8. High-speed / C4 stress probes at the final checkpoint only.
run_eval FROZEN_HIGHENV env \
  STAGE=frozen_bc_residual \
  ARMS='frozen_bc_residual_nodade frozen_bc_residual_dade frozen_bc_residual_dade_head frozen_bc_residual_dade_sidechannel10' \
  SEEDS='91430 91431 91432' TAGS='h0 h1 h2 h3 h4 c4' CHECKPOINTS='99' \
  "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_highenv_eval_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_eval C_HIGHENV env \
  STAGE=C ARMS='nodade dade' SEEDS='91430 91431 91432' \
  TAGS='h0 h1 h2 h3 h4 c4' CHECKPOINTS='99' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_C_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_C_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_C_highenv_eval_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_eval SIDECHANNEL10_HIGHENV env \
  STAGE=B_sidechannel10 ARMS='dade_head_sidechannel10' SEEDS='91430 91431 91432' \
  TAGS='h0 h1 h2 h3 h4 c4' CHECKPOINTS='39' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_sidechannel10_20261103" \
  OUTPUT_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_B_sidechannel10_eval_20261103" \
  LOG_DIR="$ROOT/logs/pc_server_scale_B_sidechannel10_highenv_eval_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

printf '\n===== EXPANDED QUEUE DONE %s =====\n' "$(date -Is)"
