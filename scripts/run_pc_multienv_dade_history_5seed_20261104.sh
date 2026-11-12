#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_ROOT="${TRAIN_ROOT:-$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104}"
DIAG_ROOT="${DIAG_ROOT:-$ROOT/artifacts/lowlevel/pc_dade_history_multienv_5seed_20261104}"
GATE_OUTPUT="${GATE_OUTPUT:-$ROOT/artifacts/lowlevel/pc_dade_history_multienv_5seed_gate_20261104.json}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/pc_dade_history_multienv_5seed_20261104}"
ARM="${ARM:-frozen_bc_residual_dade_multienv_pose_lr3e-4_scale1p0}"
SEEDS="${SEEDS:-94030 94031 94032 94033 94034}"
TAGS="${TAGS:-default strict125}"
SEED_START="${SEED_START:-0}"
SEED_COUNT="${SEED_COUNT:-8}"
MAX_STEPS="${MAX_STEPS:-256}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
CPU_BASE="${CPU_BASE:-300}"
CPU_STRIDE="${CPU_STRIDE:-8}"
THRESHOLD="${THRESHOLD:-1e-3}"

POLICY_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_multienv_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
DEFAULT_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
STRICT_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json"

mkdir -p "$DIAG_ROOT" "$LOG_ROOT" "$(dirname "$GATE_OUTPUT")"
exec 9> "$LOG_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'DADE history queue already running: %s\n' "$LOG_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$LOG_ROOT/queue.log" 2>&1

printf '\n===== PC DADE HISTORY 5-SEED START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s arm=%s seeds=%s tags=%s eval_seeds=%s max_steps=%s\n' \
  "$(hostname)" "$$" "$ARM" "$SEEDS" "$TAGS" "$SEED_COUNT" "$MAX_STEPS"

read -r -a seed_list <<< "$SEEDS"
read -r -a tag_list <<< "$TAGS"
jobs=()
for tag in "${tag_list[@]}"; do
  for seed in "${seed_list[@]}"; do
    jobs+=("$tag:$seed")
  done
done

run_worker() {
  local slot="$1"
  local index
  local failed=0
  for ((index = slot; index < ${#jobs[@]}; index += EVAL_WORKERS)); do
    IFS=: read -r tag seed <<< "${jobs[$index]}"
    local checkpoint="$TRAIN_ROOT/${ARM}_seed${seed}/final.pt"
    local output="$DIAG_ROOT/${tag}_seed${seed}"
    local log="$LOG_ROOT/${tag}_seed${seed}.log"
    local env_config="$DEFAULT_ENV"
    if [[ "$tag" == "strict125" ]]; then
      env_config="$STRICT_ENV"
    fi
    if [[ ! -f "$checkpoint" ]]; then
      printf 'MISSING_CHECKPOINT %s\n' "$checkpoint" >&2
      failed=1
      continue
    fi
    if [[ -f "$output/summary.json" ]]; then
      printf 'SKIP tag=%s seed=%s output=%s\n' "$tag" "$seed" "$output"
      continue
    fi
    local cpu_start=$((CPU_BASE + slot * CPU_STRIDE))
    local cpu_end=$((cpu_start + CPU_STRIDE - 1))
    printf 'START tag=%s seed=%s cpu=%s-%s output=%s\n' \
      "$tag" "$seed" "$cpu_start" "$cpu_end" "$output"
    taskset -c "${cpu_start}-${cpu_end}" nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      "$PY" scripts/diagnose_pc_dade_history.py \
        --training-config "$POLICY_CONFIG" \
        --checkpoint "$checkpoint" \
        --environment-config "$env_config" \
        --output "$output" \
        --seed-start "$SEED_START" \
        --seed-count "$SEED_COUNT" \
        --max-steps "$MAX_STEPS" \
        --device cpu \
        --dtype float64 > "$log" 2>&1
    local rc=$?
    printf 'END rc=%s tag=%s seed=%s output=%s\n' "$rc" "$tag" "$seed" "$output"
    if (( rc != 0 )); then
      failed=1
      tail -n 40 "$log" >&2 || true
    fi
  done
  return "$failed"
}

worker_pids=()
for ((slot = 0; slot < EVAL_WORKERS; slot++)); do
  run_worker "$slot" &
  worker_pids+=("$!")
done
failures=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  printf 'DADE_HISTORY_DIAGNOSTIC_FAILURES=%s root=%s\n' "$failures" "$DIAG_ROOT" >&2
  exit 1
fi

printf 'GATE %s\n' "$(date -Is)"
"$PY" scripts/check_pc_dade_history_gate.py \
  --diagnostic-root "$DIAG_ROOT" \
  --output "$GATE_OUTPUT" \
  --threshold "$THRESHOLD" \
  --mode current-only \
  --mode history-zero
gate_rc=$?
printf 'DADE_HISTORY_DONE gate_rc=%s output=%s %s\n' \
  "$gate_rc" "$GATE_OUTPUT" "$(date -Is)"
exit "$gate_rc"
