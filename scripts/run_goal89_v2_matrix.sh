#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
CFG_DIR="$ROOT/configs/lowlevel/goal89_v2_20261111"
OUT_ROOT="$ROOT/artifacts/lowlevel/goal89_v2_20261111"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

run_one() {
  local arm="$1"
  local seed="$2"
  local gpu="$3"
  local config="$CFG_DIR/${arm}.json"
  local output="$OUT_ROOT/${arm}_s${seed}"
  local log="$LOG_DIR/${arm}_s${seed}.log"
  echo "[$(date -Is)] start arm=$arm seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  "$PY" "$ROOT/scripts/train_batch_lowlevel_ppo.py" \
    --training-config "$config" \
    --seed "$seed" \
    --device cuda:0 \
    --output "$output" \
    --overwrite >"$log" 2>&1
  echo "[$(date -Is)] done arm=$arm seed=$seed"
}

wave_one=(
  "A1p0_nogate 86201 0"
  "A1p0_nogate 86202 1"
  "A1p0_nogate 86203 2"
  "A1p25_nogate 86201 3"
  "A1p25_nogate 86202 4"
  "A1p25_nogate 86203 5"
  "AG20_5 86201 6"
  "AG20_5 86202 7"
)

wave_two=(
  "AG20_5 86203 0"
  "AG10_3 86201 1"
  "AG10_3 86202 2"
  "AG10_3 86203 3"
)

for spec in "${wave_one[@]}"; do
  read -r arm seed gpu <<<"$spec"
  run_one "$arm" "$seed" "$gpu" &
done
wait

for spec in "${wave_two[@]}"; do
  read -r arm seed gpu <<<"$spec"
  run_one "$arm" "$seed" "$gpu" &
done
wait

echo "[$(date -Is)] GOAL89_V2_MATRIX_COMPLETE"
