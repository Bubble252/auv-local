#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_eval_full_20261104}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_summary_full_20261104}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/pc_multienv_supervised_residual_eval_full_20261104}"
STAGE="${STAGE:-all}"
ARMS="${ARMS:-frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0 frozen_bc_residual_dade_multienv_pose_lr3e-4_scale1p0}"
SEEDS="${SEEDS:-94030 94031 94032 94033 94034}"
CHECKPOINTS="${CHECKPOINTS:-initial final}"
EVAL_WORKERS="${EVAL_WORKERS:-6}"
CPU_BASE="${CPU_BASE:-200}"
CPU_STRIDE="${CPU_STRIDE:-12}"
PRIMARY_SEED_COUNT="${PRIMARY_SEED_COUNT:-64}"
HARD_SEED_COUNT="${HARD_SEED_COUNT:-32}"
BASELINE_ARM="${BASELINE_ARM:-supervised_frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0}"

mkdir -p "$OUTPUT_ROOT" "$SUMMARY_ROOT" "$LOG_ROOT"
exec 9> "$LOG_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'PC multienv eval queue already running: %s\n' "$LOG_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$LOG_ROOT/queue.log" 2>&1

printf '\n===== PC MULTIENV 5-SEED EVAL START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s stage=%s arms=%s seeds=%s checkpoints=%s\n' \
  "$(hostname)" "$$" "$STAGE" "$ARMS" "$SEEDS" "$CHECKPOINTS"

run_set() {
  local name="$1"
  local tags="$2"
  local seed_count="$3"
  printf 'SET_START name=%s tags=%s seed_count=%s %s\n' \
    "$name" "$tags" "$seed_count" "$(date -Is)"
  RUN_ROOT="$RUN_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  LOG_DIR="$LOG_ROOT/$name" \
  ARMS="$ARMS" \
  SEEDS="$SEEDS" \
  CHECKPOINTS="$CHECKPOINTS" \
  TAGS="$tags" \
  SEED_COUNT="$seed_count" \
  EVAL_WORKERS="$EVAL_WORKERS" \
  CPU_BASE="$CPU_BASE" \
  CPU_STRIDE="$CPU_STRIDE" \
    bash scripts/eval_pc_server_basics_20261103.sh
  local rc=$?
  printf 'SET_END name=%s rc=%s %s\n' "$name" "$rc" "$(date -Is)"
  return "$rc"
}

failures=0
if [[ "$STAGE" == "all" || "$STAGE" == "primary" ]]; then
  run_set primary "default strict125" "$PRIMARY_SEED_COUNT" || failures=$((failures + 1))
fi
if [[ "$STAGE" == "all" || "$STAGE" == "hard" ]]; then
  run_set hard "h1 h2 h3 c4" "$HARD_SEED_COUNT" || failures=$((failures + 1))
fi

printf 'SUMMARIZE %s\n' "$(date -Is)"
"$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$OUTPUT_ROOT" \
  --output-dir "$SUMMARY_ROOT" \
  --baseline-arm "$BASELINE_ARM" \
  --checkpoint-baseline final=initial
summary_rc=$?
if (( summary_rc != 0 )); then
  failures=$((failures + 1))
fi

printf 'PC_MULTIENV_EVAL_DONE failures=%s summary_rc=%s output=%s %s\n' \
  "$failures" "$summary_rc" "$OUTPUT_ROOT" "$(date -Is)"
exit "$(( failures > 0 ? 1 : 0 ))"
