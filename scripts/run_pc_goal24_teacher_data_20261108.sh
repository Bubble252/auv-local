#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-/dev/shm/goal22_gpu_env/bin/python}"
ROOT="${ROOT:-/dev/shm/pc_v1_data_recipe_20261108}"
DEFAULT_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_default_20261108.json"
STRICT_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_strict125_20261108.json"

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$ROOT/logs"

collect() {
  local name="$1"
  local config="$2"
  local num_envs="$3"
  local seed_start="$4"
  local output="$ROOT/$name"
  local log="$ROOT/logs/${name//\//_}.log"
  if [[ -e "$output/demos.npz" ]]; then
    echo "refusing to overwrite existing dataset: $output" >&2
    exit 1
  fi
  echo "=== $(date -Is) collect $name ===" | tee "$log"
  "$PY" "$REPO/scripts/collect_pc_bc_demos.py" \
    --config "$config" \
    --output "$output" \
    --num-envs "$num_envs" \
    --steps-per-env 4096 \
    --seed-start "$seed_start" \
    --device cuda \
    --dtype float64 \
    --path-dim 8 \
    --actor-command-dim 0 2>&1 | tee -a "$log"
}

merge() {
  local name="$1"
  shift
  local output="$ROOT/$name"
  local log="$ROOT/logs/${name//\//_}.log"
  if [[ -e "$output/demos.npz" ]]; then
    echo "refusing to overwrite existing merged dataset: $output" >&2
    exit 1
  fi
  echo "=== $(date -Is) merge $name ===" | tee -a "$log"
  "$PY" "$REPO/scripts/merge_pc_bc_demos.py" --inputs "$@" --output "$output" 2>&1 | tee -a "$log"
}

collect teacher_d0/default "$DEFAULT_CONFIG" 64 90000
collect teacher_d0/strict125 "$STRICT_CONFIG" 32 90100
merge teacher_d0 "$ROOT/teacher_d0/default/demos.npz" "$ROOT/teacher_d0/strict125/demos.npz"

collect teacher_d1/default "$DEFAULT_CONFIG" 54 90000
collect teacher_d1/strict125 "$STRICT_CONFIG" 32 90100
merge teacher_d1 "$ROOT/teacher_d1/default/demos.npz" "$ROOT/teacher_d1/strict125/demos.npz"

collect teacher_d2/default "$DEFAULT_CONFIG" 51 90000
collect teacher_d2/strict125 "$STRICT_CONFIG" 25 90100
merge teacher_d2 "$ROOT/teacher_d2/default/demos.npz" "$ROOT/teacher_d2/strict125/demos.npz"

echo "teacher_data_complete $(date -Is)" | tee "$ROOT/logs/teacher_data_complete.log"
