#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
CFG_DIR="$ROOT/configs/lowlevel/goal89_v2_20261111"
OUT_ROOT="$ROOT/artifacts/lowlevel/goal89_v2_20261111"
EVAL_ROOT="$OUT_ROOT/eval_dev40"
LOG_DIR="$EVAL_ROOT/logs"
DEFAULT_ENV="$ROOT/configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"
STRICT_ENV="$ROOT/configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json"
mkdir -p "$EVAL_ROOT" "$LOG_DIR"

eval_one() {
  local arm="$1"
  local seed="$2"
  local scenario="$3"
  local env_config="$4"
  local action_mode="$5"
  local checkpoint="$OUT_ROOT/${arm}_s${seed}/final.pt"
  local output="$EVAL_ROOT/${arm}_s${seed}_${scenario}"
  local log="$LOG_DIR/${arm}_s${seed}_${scenario}.log"
  echo "[$(date -Is)] eval arm=$arm seed=$seed scenario=$scenario mode=$action_mode"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" "$ROOT/scripts/evaluate_pc_ppo.py" \
    --training-config "$CFG_DIR/${arm}.json" \
    --checkpoint "$checkpoint" \
    --environment-config "$env_config" \
    --output "$output" \
    --seed-start 20 --seed-count 40 \
    --device cpu --dtype float64 \
    --action-mode "$action_mode" \
    --success-frame both >"$log" 2>&1
}

tasks=()
for arm in A1p0_nogate A1p25_nogate AG20_5 AG10_3; do
  for seed in 86201 86202 86203; do
    tasks+=("$arm $seed default $DEFAULT_ENV policy")
    tasks+=("$arm $seed strict125 $STRICT_ENV policy")
  done
done
tasks+=("A1p0_nogate 86201 default $DEFAULT_ENV zero-residual")
tasks+=("A1p0_nogate 86201 strict125 $STRICT_ENV zero-residual")

index=0
for spec in "${tasks[@]}"; do
  read -r arm seed scenario env_config action_mode <<<"$spec"
  eval_one "$arm" "$seed" "$scenario" "$env_config" "$action_mode" &
  index=$((index + 1))
  if (( index % 12 == 0 )); then
    wait
  fi
done
wait

echo "[$(date -Is)] GOAL89_V2_EVAL_COMPLETE"
