#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_TAG="${QUEUE_TAG:-}"
QUEUE_ROOT="$ROOT/logs/pc_multienv_nodade_extra_seeds_20261104${QUEUE_TAG:+_$QUEUE_TAG}"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'multienv no-DADE extra-seed queue already running: %s\n' \
    "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

DEMO="$ROOT/artifacts/lowlevel/pc_multienv_teacher_demos_full_20261104/merged/demos.npz"
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/artifacts/lowlevel/pc_multienv_nodade_extra_eval_20261104${QUEUE_TAG:+_$QUEUE_TAG}}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$ROOT/artifacts/lowlevel/pc_multienv_nodade_extra_summary_20261104${QUEUE_TAG:+_$QUEUE_TAG}}"
ARM='frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0'
CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_multienv_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
SEEDS="${SEEDS:-94031 94032}"
TRAIN_CPU_BASE="${TRAIN_CPU_BASE:-368}"
TRAIN_CPU_STRIDE="${TRAIN_CPU_STRIDE:-8}"
EVAL_CPU_BASE="${EVAL_CPU_BASE:-368}"
EVAL_CPU_STRIDE="${EVAL_CPU_STRIDE:-8}"

printf '\n===== MULTIENV NO-DADE EXTRA SEEDS START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s seeds=%s\n' "$(hostname)" "$$" "$SEEDS"
if [[ ! -f "$DEMO" ]]; then
  printf 'MISSING_DEMOS %s\n' "$DEMO" >&2
  exit 2
fi

read -r -a seed_list <<< "$SEEDS"
pids=()
names=()
for index in "${!seed_list[@]}"; do
  seed="${seed_list[$index]}"
  cpu_start=$((TRAIN_CPU_BASE + index * TRAIN_CPU_STRIDE))
  cpu="$cpu_start-$((cpu_start + TRAIN_CPU_STRIDE - 1))"
  output="$TRAIN_ROOT/${ARM}_seed${seed}"
  log="$QUEUE_ROOT/pretrain_${ARM}_seed${seed}.log"
  if [[ -f "$output/final.pt" && -f "$output/training_manifest.json" ]]; then
    printf 'SKIP_PRETRAIN seed=%s output=%s\n' "$seed" "$output"
    continue
  fi
  printf 'START_PRETRAIN seed=%s cpu=%s output=%s\n' "$seed" "$cpu" "$output"
  taskset -c "$cpu" env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" scripts/pretrain_pc_residual.py \
    --training-config "$CONFIG" \
    --demos "$DEMO" \
    --output "$output" \
    --batch-size 4096 \
    --max-epochs 200 \
    --patience 30 \
    --learning-rate 1e-3 \
    --weight-decay 1e-5 \
    --output-l2 1e-4 \
    --validation-fraction 0.2 \
    --seed "$seed" \
    --device cpu \
    --overwrite > "$log" 2>&1 &
  pids+=("$!")
  names+=("$seed")
done

failures=0
for index in "${!pids[@]}"; do
  seed="${names[$index]}"
  if wait "${pids[$index]}"; then
    printf 'END_PRETRAIN seed=%s rc=0 %s\n' "$seed" "$(date -Is)"
  else
    rc=$?
    printf 'END_PRETRAIN seed=%s rc=%s log=%s\n' \
      "$seed" "$rc" "$QUEUE_ROOT/pretrain_${ARM}_seed${seed}.log"
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  printf 'STOP_AFTER_TRAIN_FAILURES failures=%s\n' "$failures"
  exit 1
fi

mkdir -p "$EVAL_ROOT" "$SUMMARY_ROOT" "$QUEUE_ROOT/eval"
STAGE=multienv_nodade_extra_20261104 \
  ARMS="$ARM" SEEDS="$SEEDS" TAGS='default strict125' CHECKPOINTS='initial final' \
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=2 CPU_BASE="$EVAL_CPU_BASE" CPU_STRIDE="$EVAL_CPU_STRIDE" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 SEED_START=0 SEED_COUNT=64 \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval" \
  bash scripts/eval_pc_server_basics_20261103.sh
primary_rc=$?
STAGE=multienv_nodade_extra_20261104 \
  ARMS="$ARM" SEEDS="$SEEDS" TAGS='h1 h2 h3 c4' CHECKPOINTS='initial final' \
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=2 CPU_BASE="$EVAL_CPU_BASE" CPU_STRIDE="$EVAL_CPU_STRIDE" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 SEED_START=0 SEED_COUNT=32 \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval" \
  bash scripts/eval_pc_server_basics_20261103.sh
hard_rc=$?
if (( primary_rc != 0 || hard_rc != 0 )); then
  printf 'STOP_AFTER_EVAL_FAILURES primary_rc=%s hard_rc=%s\n' "$primary_rc" "$hard_rc"
  exit 1
fi

"$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$SUMMARY_ROOT" \
  --baseline-arm "$ARM" \
  --checkpoint initial --checkpoint final \
  --checkpoint-baseline final=initial
summary_rc=$?
printf 'MULTIENV_NO_DADE_EXTRA_DONE summary_rc=%s %s\n' "$summary_rc" "$(date -Is)"
exit "$summary_rc"
