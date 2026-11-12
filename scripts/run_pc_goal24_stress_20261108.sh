#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-/dev/shm/goal22_gpu_env/bin/python}"
ROOT="${ROOT:-/dev/shm/pc_v1_stress_mix_20261108}"
SOURCE="${SOURCE:-/dev/shm/pc_v1_data_recipe_20261108}"
DEFAULT_STRESS_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_stress_teacher_fixed_default_20261108.json"
STRICT_STRESS_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_stress_teacher_fixed_strict125_20261108.json"
NOMINAL_MATRIX="$REPO/configs/lowlevel/eval/pc_v1_nominal_eval_matrix_teacher_fixed_20261108.json"
STRESS_MATRIX="$REPO/configs/lowlevel/eval/pc_v1_stress_eval_matrix_20261108.json"

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$ROOT/logs"

refuse_existing() {
  local path="$1"
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite existing output: $path" >&2
    exit 1
  fi
}

collect_stress() {
  local scenario="$1"
  local config="$2"
  local seed_start="$3"
  local output="$ROOT/stress_raw/$scenario"
  local log="$ROOT/logs/collect_stress_${scenario}.log"
  refuse_existing "$output/demos.npz"
  echo "=== $(date -Is) collect stress $scenario ===" | tee "$log"
  "$PY" "$REPO/scripts/collect_pc_bc_demos.py" \
    --config "$config" \
    --output "$output" \
    --num-envs 10 \
    --steps-per-env 4096 \
    --seed-start "$seed_start" \
    --device cuda \
    --dtype float64 \
    --path-dim 8 \
    --actor-command-dim 0 \
    --seed-role stress \
    2>&1 | tee -a "$log"
}

train_bc() {
  local arm="$1"
  local demos="$ROOT/${arm}_merged/demos.npz"
  local output="$ROOT/bc_$arm"
  local log="$ROOT/logs/train_bc_$arm.log"
  refuse_existing "$output/pc_bc_actor.pt"
  echo "=== $(date -Is) train $arm ===" | tee "$log"
  "$PY" "$REPO/scripts/train_pc_bc.py" \
    --demos "$demos" \
    --output "$output" \
    --state-encoder-hidden-sizes 128,128 \
    --actor-hidden-sizes 256,128 \
    --d-z-total 64 \
    --n-in-reserved 8 \
    --n-z-reserved 4 \
    --activation elu \
    --batch-size 4096 \
    --max-epochs 300 \
    --patience 30 \
    --learning-rate 1e-3 \
    --weight-decay 1e-5 \
    --validation-fraction 0.15 \
    --seed 0 \
    --device cuda \
    --actor-command-dim 0 \
    --path-dim 8 \
    --path-encoder-hidden-sizes 64 \
    --path-encoder-alpha 1.0 \
    --saturation-weight 2.0 \
    --saturation-threshold 0.9 \
    --action-head continuous \
    2>&1 | tee -a "$log"
}

eval_matrix() {
  local matrix="$1"
  local checkpoint="$2"
  local output="$3"
  local label="$4"
  local log="$ROOT/logs/eval_${label}.log"
  refuse_existing "$output/comparison.json"
  echo "=== $(date -Is) eval $label ===" | tee "$log"
  "$PY" "$REPO/scripts/run_pc_eval_matrix.py" \
    --matrix "$matrix" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    2>&1 | tee -a "$log"
}

collect_stress default "$DEFAULT_STRESS_CONFIG" 92000
collect_stress strict125 "$STRICT_STRESS_CONFIG" 92100

"$PY" "$REPO/scripts/prepare_pc_goal24_stress_mixes.py" \
  --source-root "$SOURCE" \
  --output-root "$ROOT" \
  2>&1 | tee "$ROOT/logs/prepare_stress_mixes.log"

train_bc s10
train_bc s20

eval_matrix "$NOMINAL_MATRIX" "$ROOT/bc_s10/pc_bc_actor.pt" "$ROOT/eval_nominal_s10" nominal_s10
eval_matrix "$NOMINAL_MATRIX" "$ROOT/bc_s20/pc_bc_actor.pt" "$ROOT/eval_nominal_s20" nominal_s20
eval_matrix "$STRESS_MATRIX" "$ROOT/bc_s10/pc_bc_actor.pt" "$ROOT/eval_stress_s10" stress_s10
eval_matrix "$STRESS_MATRIX" "$ROOT/bc_s20/pc_bc_actor.pt" "$ROOT/eval_stress_s20" stress_s20
eval_matrix "$STRESS_MATRIX" "$SOURCE/bc_d2/pc_bc_actor.pt" "$ROOT/eval_stress_d2" stress_d2

echo "goal24_stress_complete $(date -Is)" | tee "$ROOT/logs/stress_complete.log"
