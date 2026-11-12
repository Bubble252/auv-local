#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_ROOT="${TRAIN_ROOT:-$ROOT/artifacts/lowlevel/pc_residual_lr_diag_20261103}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/artifacts/lowlevel/pc_residual_lr_diag_eval_20261103}"
DIAG_ROOT="${DIAG_ROOT:-$ROOT/artifacts/lowlevel/pc_residual_lr_diag_history_20261103}"
GATE_ROOT="${GATE_ROOT:-$ROOT/artifacts/lowlevel/pc_residual_lr_diag_gate_20261103}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/pc_residual_lr_diag_eval_20261103}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-240}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"
SEED_START="${SEED_START:-0}"
SEED_COUNT="${SEED_COUNT:-16}"
CPU_START="${CPU_START:-340}"

mkdir -p "$EVAL_ROOT" "$DIAG_ROOT" "$GATE_ROOT" "$LOG_DIR"

arms=(
  nodade_lr3e-4_scale1p0_seed91430
  dade_head_isolated_lr3e-4_scale1p0_seed91430
  history_required_lr3e-4_scale1p0_seed91430
)

for ((attempt = 1; attempt <= WAIT_ATTEMPTS; attempt++)); do
  ready=1
  for arm in "${arms[@]}"; do
    if [[ ! -f "$TRAIN_ROOT/$arm/final.pt" ]]; then
      ready=0
      break
    fi
  done
  if (( ready == 1 )); then
    break
  fi
  sleep "$WAIT_SECONDS"
done

for arm in "${arms[@]}"; do
  if [[ ! -f "$TRAIN_ROOT/$arm/final.pt" ]]; then
    printf 'MISSING_FINAL arm=%s\n' "$arm"
    exit 1
  fi
done

declare -A CONFIGS=(
  [nodade_lr3e-4_scale1p0_seed91430]="$TRAIN_ROOT/configs/pc_residual_diag_nodade_lr3e-4_scale1p0.json"
  [dade_head_isolated_lr3e-4_scale1p0_seed91430]="$TRAIN_ROOT/configs/pc_residual_diag_dade_head_isolated_lr3e-4_scale1p0.json"
  [history_required_lr3e-4_scale1p0_seed91430]="$TRAIN_ROOT/configs/pc_residual_diag_history_required_lr3e-4_scale1p0.json"
)
declare -A ENV_CONFIGS=(
  [default]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
  [strict125]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json"
)

cpu_end=$((CPU_START + 15))
for tag in default strict125; do
  for arm in "${arms[@]}"; do
    for checkpoint in initial 99; do
      output="$EVAL_ROOT/$tag/$arm/ck${checkpoint}_seed${SEED_START}_$((SEED_START + SEED_COUNT - 1))"
      log="$LOG_DIR/${tag}_${arm}_ck${checkpoint}.log"
      if [[ -f "$output/summary.json" ]]; then
        printf 'SKIP %s\n' "$output"
        continue
      fi
      printf 'EVAL tag=%s arm=%s ck=%s output=%s\n' "$tag" "$arm" "$checkpoint" "$output"
      taskset -c "${CPU_START}-${cpu_end}" nice -n 10 env \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PY" scripts/evaluate_pc_ppo.py \
        --training-config "${CONFIGS[$arm]}" \
        --checkpoint "$TRAIN_ROOT/$arm/model_${checkpoint}.pt" \
        --environment-config "${ENV_CONFIGS[$tag]}" \
        --output "$output" \
        --seed-start "$SEED_START" \
        --seed-count "$SEED_COUNT" \
        --device cpu \
        --dtype float64 \
        --history-ablation none > "$log" 2>&1
      printf 'END rc=%s %s\n' "$?" "$output"
    done
  done
done

for arm in \
  dade_head_isolated_lr3e-4_scale1p0_seed91430 \
  history_required_lr3e-4_scale1p0_seed91430; do
  printf 'DIAG_HISTORY arm=%s\n' "$arm"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PY" scripts/diagnose_pc_dade_history.py \
    --training-config "${CONFIGS[$arm]}" \
    --checkpoint "$TRAIN_ROOT/$arm/model_99.pt" \
    --output "$DIAG_ROOT/$arm" \
    --seed-start 0 --seed-count 32 --max-steps 512 \
    --device cpu --dtype float64 > "$LOG_DIR/history_${arm}.log" 2>&1
  printf 'DIAG_HISTORY_END rc=%s arm=%s\n' "$?" "$arm"
done

"$PY" scripts/check_pc_dade_history_gate.py \
  --diagnostic-root "$DIAG_ROOT" \
  --output "$GATE_ROOT/gate.json" \
  --mode current-only \
  --mode history-zero \
  --mode history-reverse > "$LOG_DIR/history_gate.log" 2>&1
printf 'HISTORY_GATE_END rc=%s\n' "$?"

printf 'ALL_DIAG_EVALS_DONE root=%s\n' "$EVAL_ROOT"
