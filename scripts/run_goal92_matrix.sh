#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
CFG_DIR="$ROOT/configs/lowlevel/goal92_20261111"
OUT_ROOT="$ROOT/artifacts/lowlevel/goal92_update_schedule_20261111"
LOG_DIR="$OUT_ROOT/logs"; mkdir -p "$OUT_ROOT" "$LOG_DIR"
run_one(){ local arm=$1 seed=$2 gpu=$3; echo "[$(date -Is)] start $arm $seed gpu=$gpu"; CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "$PY" "$ROOT/scripts/train_batch_lowlevel_ppo.py" --training-config "$CFG_DIR/${arm}.json" --seed "$seed" --device cuda:0 --output "$OUT_ROOT/${arm}_s${seed}" --overwrite >"$LOG_DIR/${arm}_s${seed}.log" 2>&1; echo "[$(date -Is)] done $arm $seed"; }
tasks=("e4 86201 0" "c10 86201 1" "c20 86201 2" "an 86201 3" "w0 86201 4" "e4_c10_an 86201 5")
for s in "${tasks[@]}"; do read -r a b c <<<"$s"; run_one "$a" "$b" "$c" & done
wait; echo GOAL92_MATRIX_COMPLETE
