#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SEEDS="${SEEDS:-91430 91431 91432 91433 91434}"
ITERATIONS="${ITERATIONS:-32}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-256}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-128}"
WARMSTART="${WARMSTART:-$ROOT/artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/lowlevel/pc_server_c3_5seed_20261103}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/pc_server_c3_5seed_20261103}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

declare -A CONFIGS=(
  [nodade]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [dade]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_aux_current_scale3e-2_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)

for seed in $SEEDS; do
  for arm in nodade dade; do
    output="$OUTPUT_ROOT/${arm}_seed${seed}"
    log="$LOG_DIR/${arm}_seed${seed}.log"
    if [[ -f "$output/training_manifest.json" && -f "$output/final.pt" ]]; then
      printf 'SKIP %s\n' "$output"
      continue
    fi
    printf 'START arm=%s seed=%s device=%s output=%s\n' "$arm" "$seed" "$CUDA_VISIBLE_DEVICES" "$output"
    "$PY" scripts/train_batch_lowlevel_ppo.py \
      --training-config "${CONFIGS[$arm]}" \
      --device cuda \
      --iterations "$ITERATIONS" \
      --num-envs "$NUM_ENVS" \
      --num-steps "$NUM_STEPS" \
      --num-mini-batches "$NUM_MINI_BATCHES" \
      --seed "$seed" \
      --output "$output" \
      --warmstart-actor "$WARMSTART" \
      --record-only 2>&1 | tee "$log"
    printf 'DONE arm=%s seed=%s\n' "$arm" "$seed"
  done
done

printf 'ALL_DONE output_root=%s\n' "$OUTPUT_ROOT"
