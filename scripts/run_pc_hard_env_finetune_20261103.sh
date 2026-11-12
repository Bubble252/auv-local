#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_hard_env_finetune_20261103"
mkdir -p "$QUEUE_ROOT"
exec >> "$QUEUE_ROOT/queue.log" 2>&1

printf '\n===== HARD-ENV FINETUNE QUEUE WAIT %s =====\n' "$(date -Is)"
while ps -eo args | grep -Eq '^bash scripts/run_pc_history_only_dade_20261103.sh$'; do
  printf 'WAIT_HISTORY_ONLY %s\n' "$(date -Is)"
  sleep 60
done
exec 9> "$ROOT/logs/pc_expanded_queue_20261103/queue.lock"
flock 9
printf '===== HARD-ENV FINETUNE QUEUE START %s =====\n' "$(date -Is)"

TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_hard_env_finetune_20261103"
TRAIN_LOGS="$ROOT/logs/pc_server_scale_hard_env_finetune_20261103/train"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_hard_env_finetune_eval_20261103"
EVAL_LOGS="$ROOT/logs/pc_server_scale_hard_env_finetune_eval_20261103"

run_step() {
  local name="$1"
  shift
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  return 0
}

run_step TRAIN_HARD_ENV_FINETUNE env \
  STAGE=hard_env_finetune \
  ARMS='frozen_bc_residual_nodade_h1 frozen_bc_residual_nodade_h2 frozen_bc_residual_nodade_c4' \
  SEEDS='91430 91431 91432' ITERATIONS=100 \
  GPU_IDS='2 6' MIN_FREE_MIB=16500 \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$TRAIN_ROOT" LOG_DIR="$TRAIN_LOGS" \
  bash scripts/run_pc_server_scale_20261103.sh

COMMON_EVAL=(
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4
  CPU_BASE=160 CPU_STRIDE=16
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0 SEED_COUNT=32
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT"
)

run_step EVAL_H1_IN_DISTRIBUTION env \
  STAGE=hard_env_finetune ARMS='frozen_bc_residual_nodade_h1' SEEDS='91430 91431 91432' \
  TAGS='h1' CHECKPOINTS='initial 0 50 99' "${COMMON_EVAL[@]}" \
  LOG_DIR="${EVAL_LOGS}_h1" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_step EVAL_H2_IN_DISTRIBUTION env \
  STAGE=hard_env_finetune ARMS='frozen_bc_residual_nodade_h2' SEEDS='91430 91431 91432' \
  TAGS='h2' CHECKPOINTS='initial 0 50 99' "${COMMON_EVAL[@]}" \
  LOG_DIR="${EVAL_LOGS}_h2" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_step EVAL_C4_IN_DISTRIBUTION env \
  STAGE=hard_env_finetune ARMS='frozen_bc_residual_nodade_c4' SEEDS='91430 91431 91432' \
  TAGS='c4' CHECKPOINTS='initial 0 50 99' "${COMMON_EVAL[@]}" \
  LOG_DIR="${EVAL_LOGS}_c4" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_step EVAL_HARD_ENV_DEFAULT_STRICT env \
  STAGE=hard_env_finetune \
  ARMS='frozen_bc_residual_nodade_h1 frozen_bc_residual_nodade_h2 frozen_bc_residual_nodade_c4' \
  SEEDS='91430 91431 91432' TAGS='default strict125' CHECKPOINTS='initial 99' "${COMMON_EVAL[@]}" \
  LOG_DIR="${EVAL_LOGS}_default_strict" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_step SUMMARY_HARD_ENV_FINETUNE \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_ROOT" \
  --eval-root "$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_eval_20261103" \
  --output-dir "$ROOT/artifacts/lowlevel/pc_server_scale_hard_env_finetune_eval_summary_20261103" \
  --baseline-arm frozen_bc_residual_nodade \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 50=initial \
  --checkpoint-baseline 99=initial

printf '\n===== HARD-ENV FINETUNE QUEUE DONE %s =====\n' "$(date -Is)"
