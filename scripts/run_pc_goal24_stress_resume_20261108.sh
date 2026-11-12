#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-/dev/shm/goal22_gpu_env/bin/python}"
ROOT="${ROOT:-/dev/shm/pc_v1_stress_mix_20261108}"
SOURCE="${SOURCE:-/dev/shm/pc_v1_data_recipe_20261108}"
GLOBAL_PLANNER_SRC="${GLOBAL_PLANNER_SRC:-}"
NOMINAL_MATRIX="$REPO/configs/lowlevel/eval/pc_v1_nominal_eval_matrix_teacher_fixed_20261108.json"
STRESS_MATRIX="$REPO/configs/lowlevel/eval/pc_v1_stress_eval_matrix_20261108.json"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"

export PYTHONPATH="$REPO/src:$GLOBAL_PLANNER_SRC"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$ROOT/logs"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required input: $path" >&2
    exit 1
  fi
}

refuse_incomplete_output() {
  local output="$1"
  local expected_file="$2"
  if [[ -e "$output" && ! -f "$output/$expected_file" ]]; then
    echo "refusing partial output without $expected_file: $output" >&2
    exit 1
  fi
}

wait_for_gpu() {
  [[ "$WAIT_FOR_GPU" == "1" ]] || return 0
  while true; do
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
    if [[ -z "$apps" ]]; then
      echo "=== $(date -Is) GPU is free; starting resume ===" | tee -a "$ROOT/logs/stress_resume.log"
      return 0
    fi
    echo "$(date -Is) waiting for GPU; active PIDs: $(echo "$apps" | tr '\n' ' ')" | tee -a "$ROOT/logs/stress_resume.log"
    sleep "$POLL_SECONDS"
  done
}

train_bc() {
  local arm="$1"
  local demos="$ROOT/${arm}_merged/demos.npz"
  local output="$ROOT/bc_$arm"
  local log="$ROOT/logs/train_bc_$arm.log"
  require_file "$demos"
  refuse_incomplete_output "$output" "pc_bc_actor.pt"
  if [[ -f "$output/pc_bc_actor.pt" ]]; then
    echo "=== $(date -Is) skip existing $arm checkpoint ===" | tee -a "$log"
    return 0
  fi
  echo "=== $(date -Is) train $arm ===" | tee -a "$log"
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
  require_file "$checkpoint"
  refuse_incomplete_output "$output" "comparison.json"
  if [[ -f "$output/comparison.json" ]]; then
    echo "=== $(date -Is) skip existing $label evaluation ===" | tee -a "$log"
    return 0
  fi
  echo "=== $(date -Is) eval $label ===" | tee -a "$log"
  "$PY" "$REPO/scripts/run_pc_eval_matrix.py" \
    --matrix "$matrix" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    2>&1 | tee -a "$log"
}

require_file "$NOMINAL_MATRIX"
require_file "$STRESS_MATRIX"
require_file "$SOURCE/bc_d2/pc_bc_actor.pt"
require_file "$SOURCE/eval_stress_d2/comparison.json"

wait_for_gpu

train_bc s10
train_bc s20

eval_matrix "$NOMINAL_MATRIX" "$ROOT/bc_s10/pc_bc_actor.pt" "$ROOT/eval_nominal_s10" nominal_s10
eval_matrix "$NOMINAL_MATRIX" "$ROOT/bc_s20/pc_bc_actor.pt" "$ROOT/eval_nominal_s20" nominal_s20
eval_matrix "$STRESS_MATRIX" "$ROOT/bc_s10/pc_bc_actor.pt" "$ROOT/eval_stress_s10" stress_s10
eval_matrix "$STRESS_MATRIX" "$ROOT/bc_s20/pc_bc_actor.pt" "$ROOT/eval_stress_s20" stress_s20

set +e
"$PY" "$REPO/scripts/check_pc_goal24_stress_gates.py" \
  --root "$ROOT" \
  --nominal-reference "$SOURCE/eval_matrix_d2" \
  --stress-reference "$SOURCE/eval_stress_d2" \
  2>&1 | tee -a "$ROOT/logs/stress_gate.log"
gate_exit=${PIPESTATUS[0]}
set -e

echo "goal24_stress_gate_exit=$gate_exit" | tee -a "$ROOT/logs/stress_resume.log"
echo "goal24_stress_resume_complete $(date -Is)" | tee -a "$ROOT/logs/stress_resume.log"
