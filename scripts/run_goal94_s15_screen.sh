#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
CFG="$ROOT/configs/lowlevel/goal94_20261111/s15.json"
OUT_ROOT="$ROOT/artifacts/lowlevel/goal94_rate_aware_20261111"
EVAL_ROOT="$OUT_ROOT/eval_dev40"
LOG_DIR="$OUT_ROOT/logs"
DEFAULT_ENV="$ROOT/configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"
STRICT_ENV="$ROOT/configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json"
mkdir -p "$OUT_ROOT" "$EVAL_ROOT" "$LOG_DIR"

train_one() {
  local seed="$1" gpu="$2"
  echo "[$(date -Is)] train S15 seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  "$PY" "$ROOT/scripts/train_batch_lowlevel_ppo.py" \
    --training-config "$CFG" \
    --seed "$seed" \
    --device cuda:0 \
    --output "$OUT_ROOT/S15_s${seed}" \
    --overwrite >"$LOG_DIR/S15_s${seed}_train.log" 2>&1
  echo "[$(date -Is)] done S15 seed=$seed"
}

eval_one() {
  local seed="$1" scenario="$2" mode="$3"
  local env_config="$DEFAULT_ENV"
  if [[ "$scenario" == "strict125" ]]; then
    env_config="$STRICT_ENV"
  fi
  local output="$EVAL_ROOT/S15_s${seed}_${scenario}_${mode}"
  local log="$LOG_DIR/S15_s${seed}_${scenario}_${mode}_eval.log"
  echo "[$(date -Is)] eval S15 seed=$seed scenario=$scenario mode=$mode"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" "$ROOT/scripts/evaluate_pc_ppo.py" \
    --training-config "$CFG" \
    --checkpoint "$OUT_ROOT/S15_s${seed}/final.pt" \
    --environment-config "$env_config" \
    --output "$output" \
    --seed-start 20 --seed-count 40 \
    --device cpu --dtype float64 \
    --action-mode "$mode" \
    --success-frame both >"$log" 2>&1
}

if [[ "${SMOKE:-0}" == "1" ]]; then
  train_one 86201 0
  eval_one 86201 default policy
  eval_one 86201 strict125 policy
  echo "[$(date -Is)] GOAL94_S15_SMOKE_COMPLETE"
  exit 0
fi

for spec in "86201 0" "86202 1" "86203 2"; do
  read -r seed gpu <<<"$spec"
  train_one "$seed" "$gpu" &
done
wait

for seed in 86201 86202 86203; do
  for scenario in default strict125; do
    eval_one "$seed" "$scenario" policy &
    eval_one "$seed" "$scenario" frozen-base &
  done
done
wait

echo "[$(date -Is)] GOAL94_S15_SCREEN_COMPLETE"
