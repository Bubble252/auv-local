#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${STAGE:-pc_s14_capability_20261104}"
ARMS="${ARMS:-frozen_bc_residual_nodade_s14_route_reward_lr3e-4_scale1p0 frozen_bc_residual_dade_pose_s14_route_reward_lr3e-4_scale1p0}"
SEEDS="${SEEDS:-94030 94031 94032 94033 94034}"
TAGS="${TAGS:-grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4}"
ITERATIONS="${ITERATIONS:-100}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_STEPS="${NUM_STEPS:-256}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-128}"
GPU_IDS="${GPU_IDS:-2 4 5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-1800}"
SEED_START="${SEED_START:-0}"
SEED_COUNT="${SEED_COUNT:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
CPU_BASE="${CPU_BASE:-160}"
CPU_STRIDE="${CPU_STRIDE:-16}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float64}"

TRAIN_ROOT="${TRAIN_ROOT:-$ROOT/artifacts/lowlevel/pc_s14_capability_ppo_20261104}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/artifacts/lowlevel/pc_s14_capability_eval_20261104}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/artifacts/lowlevel/pc_s14_capability_summary_20261104}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT/logs/pc_s14_capability_train_20261104}"
EVAL_LOG="${EVAL_LOG:-$ROOT/logs/pc_s14_capability_eval_20261104}"
QUEUE_LOG="${QUEUE_LOG:-$ROOT/logs/pc_s14_capability_queue_20261104}"

SOURCE_ROOT="${SOURCE_ROOT:-$ROOT}"
if [[ ! -d "$SOURCE_ROOT/artifacts" && -d "$ROOT/../settled_局部规划器_反应式/artifacts" ]]; then
  SOURCE_ROOT="$ROOT/../settled_局部规划器_反应式"
fi

S13_NODADE_ROOT="${S13_NODADE_ROOT:-$SOURCE_ROOT/artifacts/lowlevel/pc_dade_grid_nodade_supervised_residual_20261104}"
S13_DADE_ROOT="${S13_DADE_ROOT:-$SOURCE_ROOT/artifacts/lowlevel/pc_dade_grid_dade_supervised_residual_20261104}"
S13_NODADE_ARM="${S13_NODADE_ARM:-frozen_bc_residual_nodade_grid_gpu_lr3e-4_scale1p0}"
S13_DADE_ARM="${S13_DADE_ARM:-frozen_bc_residual_dade_grid_gpu_pose_lr3e-4_scale1p0}"
S13_NODADE_SUMMARY_ARM="${S13_NODADE_SUMMARY_ARM:-supervised_frozen_bc_residual_nodade_grid_gpu_lr3e-4_scale1p0}"
S13_DADE_SUMMARY_ARM="${S13_DADE_SUMMARY_ARM:-supervised_frozen_bc_residual_dade_grid_gpu_pose_lr3e-4_scale1p0}"
S13_NODADE_EVAL_ROOT="${S13_NODADE_EVAL_ROOT:-$SOURCE_ROOT/artifacts/lowlevel/pc_dade_grid_nodade_eval_20261104}"
S13_DADE_EVAL_ROOT="${S13_DADE_EVAL_ROOT:-$SOURCE_ROOT/artifacts/lowlevel/pc_dade_grid_dade_eval_20261104}"

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT" "$REPORT_ROOT" "$TRAIN_LOG" "$EVAL_LOG" "$QUEUE_LOG"
exec 9> "$QUEUE_LOG/queue.lock"
if ! flock -n 9; then
  printf 'S14 queue already running: %s\n' "$QUEUE_LOG/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_LOG/queue.log" 2>&1

read -r -a arm_list <<< "$ARMS"
read -r -a seed_list <<< "$SEEDS"
read -r -a tag_list <<< "$TAGS"

printf '\n===== PC S14 START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s stage=%s arms=%s seeds=%s tags=%s eval_seeds=%s\n' \
  "$(hostname)" "$$" "$STAGE" "$ARMS" "$SEEDS" "$TAGS" "$SEED_COUNT"

missing=0
for seed in "${seed_list[@]}"; do
  for arm in "${arm_list[@]}"; do
    if [[ "$arm" == *nodade* ]]; then
      checkpoint="$S13_NODADE_ROOT/${S13_NODADE_ARM}_seed${seed}/final.pt"
    else
      checkpoint="$S13_DADE_ROOT/${S13_DADE_ARM}_seed${seed}/final.pt"
    fi
    if [[ ! -f "$checkpoint" ]]; then
      printf 'MISSING_S13_WARMSTART arm=%s seed=%s checkpoint=%s\n' \
        "$arm" "$seed" "$checkpoint" >&2
      missing=$((missing + 1))
    fi
  done
done
if (( missing > 0 )); then
  printf 'S14 preflight failed: %s missing warm-start checkpoints\n' "$missing" >&2
  exit 2
fi

printf 'TRAIN start %s\n' "$(date -Is)"
STAGE="$STAGE" \
ARMS="$ARMS" \
SEEDS="$SEEDS" \
ITERATIONS="$ITERATIONS" \
NUM_ENVS="$NUM_ENVS" \
NUM_STEPS="$NUM_STEPS" \
NUM_MINI_BATCHES="$NUM_MINI_BATCHES" \
GPU_IDS="$GPU_IDS" \
MIN_FREE_MIB="$MIN_FREE_MIB" \
GPU_WAIT_SECONDS="$GPU_WAIT_SECONDS" \
OUTPUT_ROOT="$TRAIN_ROOT" \
LOG_DIR="$TRAIN_LOG" \
RESIDUAL_WARMSTART_ROOT_NODADE="$S13_NODADE_ROOT" \
RESIDUAL_WARMSTART_ARM_NODADE="$S13_NODADE_ARM" \
RESIDUAL_WARMSTART_ROOT_DADE="$S13_DADE_ROOT" \
RESIDUAL_WARMSTART_ARM_DADE="$S13_DADE_ARM" \
  bash scripts/run_pc_server_basics_train_20261103.sh
train_rc=$?
printf 'TRAIN end rc=%s %s\n' "$train_rc" "$(date -Is)"
if (( train_rc != 0 )); then
  printf 'S14 training failed; skipping evaluation\n' >&2
  exit "$train_rc"
fi

printf 'EVAL start %s\n' "$(date -Is)"
STAGE="$STAGE" \
ARMS="$ARMS" \
SEEDS="$SEEDS" \
TAGS="$TAGS" \
CHECKPOINTS="final" \
SEED_START="$SEED_START" \
SEED_COUNT="$SEED_COUNT" \
DEVICE="$DEVICE" \
DTYPE="$DTYPE" \
EVAL_WORKERS="$EVAL_WORKERS" \
CPU_BASE="$CPU_BASE" \
CPU_STRIDE="$CPU_STRIDE" \
RUN_ROOT="$TRAIN_ROOT" \
OUTPUT_ROOT="$EVAL_ROOT" \
LOG_DIR="$EVAL_LOG" \
  bash scripts/eval_pc_server_basics_20261103.sh
eval_rc=$?
printf 'EVAL end rc=%s %s\n' "$eval_rc" "$(date -Is)"
if (( eval_rc != 0 )); then
  printf 'S14 evaluation failed; summary will include completed runs only\n' >&2
fi

printf 'SUMMARY start %s\n' "$(date -Is)"
"$PY" scripts/summarize_pc_s14_capability_20261104.py \
  --s14-eval-root "$EVAL_ROOT" \
  --s13-nodade-eval-root "$S13_NODADE_EVAL_ROOT" \
  --s13-dade-eval-root "$S13_DADE_EVAL_ROOT" \
  --output-dir "$REPORT_ROOT" \
  --tags "$TAGS" \
  --s13-nodade-arm "$S13_NODADE_SUMMARY_ARM" \
  --s13-dade-arm "$S13_DADE_SUMMARY_ARM" \
  --eval-seed-count "$SEED_COUNT"
summary_rc=$?
printf 'SUMMARY end rc=%s %s\n' "$summary_rc" "$(date -Is)"

if (( eval_rc != 0 || summary_rc != 0 )); then
  exit 1
fi
printf 'S14_DONE %s train=%s eval=%s report=%s\n' \
  "$(date -Is)" "$TRAIN_ROOT" "$EVAL_ROOT" "$REPORT_ROOT"
