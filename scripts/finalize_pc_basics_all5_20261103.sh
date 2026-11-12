#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_basics_finalize_all5_20261103"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'all5 finalizer already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

BASICS_PID="${BASICS_PID:-3745939}"
EXTENSION_PID="${EXTENSION_PID:-3779144}"
POLL_SECONDS="${POLL_SECONDS:-60}"

printf '\n===== PC BASICS ALL5 FINALIZER WAIT %s =====\n' "$(date -Is)"
printf 'basics_pid=%s extension_pid=%s\n' "$BASICS_PID" "$EXTENSION_PID"

wait_for_pid() {
  local pid="$1"
  local name="$2"
  while kill -0 "$pid" 2>/dev/null; do
    printf 'WAIT %s pid=%s %s\n' "$name" "$pid" "$(date -Is)"
    sleep "$POLL_SECONDS"
  done
  printf 'DONE %s pid=%s %s\n' "$name" "$pid" "$(date -Is)"
}

wait_for_pid "$BASICS_PID" basics
wait_for_pid "$EXTENSION_PID" extension

printf '\n===== PC BASICS ALL5 FINALIZER START %s =====\n' "$(date -Is)"

C2_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_eval_20261103"
C2_EXT_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_eval_ext_20261103"
C2_ALL5_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_summary_all5_20261103"

H3_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_eval_20261103"
H3_EXT_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_eval_ext_20261103"
H3_ALL5_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_summary_all5_20261103"

C3_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_eval_20261103"
C3_EXT_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_eval_ext_20261103"
C3_ALL5_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_summary_all5_20261103"

HARD_H1_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_h1_eval_20261103"
HARD_H2_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_h2_eval_20261103"
HARD_C4_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_c4_eval_20261103"
HARD_CROSS_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_cross_eval_20261103"
HARD_EXT_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_eval_ext_20261103"
HARD_3_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_summary_20261103"
HARD_ALL5_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_summary_all5_20261103"

H3_ARM='frozen_bc_residual_nodade_h3_lr3e-4_scale1p0'
C3_ARM='frozen_bc_residual_nodade_lr3e-4_scale1p0'
HARD_BASELINE_ARM='frozen_bc_residual_nodade_h1_lr3e-4_scale1p0'

failures=0
run_summary() {
  local name="$1"
  shift
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  if (( rc != 0 )); then
    failures=$((failures + 1))
  fi
}

checkpoint_args=(
  --checkpoint-baseline 0=initial
  --checkpoint-baseline 10=initial
  --checkpoint-baseline 20=initial
  --checkpoint-baseline 30=initial
  --checkpoint-baseline 40=initial
  --checkpoint-baseline 60=initial
  --checkpoint-baseline 80=initial
  --checkpoint-baseline 99=initial
  --checkpoint-baseline final=initial
)

run_summary SUMMARY_C2_ALL5 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$C2_EVAL_ROOT" \
  --eval-root "$C2_EXT_EVAL_ROOT" \
  --output-dir "$C2_ALL5_SUMMARY_ROOT" \
  --baseline-arm c2_nodade \
  "${checkpoint_args[@]}"

run_summary SUMMARY_H3_ALL5 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$H3_EVAL_ROOT" \
  --eval-root "$H3_EXT_EVAL_ROOT" \
  --output-dir "$H3_ALL5_SUMMARY_ROOT" \
  --baseline-arm "$H3_ARM" \
  "${checkpoint_args[@]}"

run_summary SUMMARY_C3_ALL5 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$C3_EVAL_ROOT" \
  --eval-root "$C3_EXT_EVAL_ROOT" \
  --output-dir "$C3_ALL5_SUMMARY_ROOT" \
  --baseline-arm "$C3_ARM" \
  "${checkpoint_args[@]}"

run_summary SUMMARY_HARDENV_3 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$HARD_H1_EVAL_ROOT" \
  --eval-root "$HARD_H2_EVAL_ROOT" \
  --eval-root "$HARD_C4_EVAL_ROOT" \
  --eval-root "$HARD_CROSS_EVAL_ROOT" \
  --output-dir "$HARD_3_SUMMARY_ROOT" \
  --baseline-arm "$HARD_BASELINE_ARM" \
  "${checkpoint_args[@]}"

run_summary SUMMARY_HARDENV_ALL5 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$HARD_H1_EVAL_ROOT" \
  --eval-root "$HARD_H2_EVAL_ROOT" \
  --eval-root "$HARD_C4_EVAL_ROOT" \
  --eval-root "$HARD_CROSS_EVAL_ROOT" \
  --eval-root "$HARD_EXT_EVAL_ROOT" \
  --output-dir "$HARD_ALL5_SUMMARY_ROOT" \
  --baseline-arm "$HARD_BASELINE_ARM" \
  "${checkpoint_args[@]}"

printf '\n===== PC BASICS ALL5 FINALIZER DONE %s failures=%s =====\n' "$(date -Is)" "$failures"
exit "$((failures > 0))"
