#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
CFG_DIR="$ROOT/configs/lowlevel/goal92_20261111"
OUT_ROOT="$ROOT/artifacts/lowlevel/goal92_update_schedule_20261111"
EVAL_ROOT="$OUT_ROOT/eval_dev40"; LOG_DIR="$EVAL_ROOT/logs"; mkdir -p "$EVAL_ROOT" "$LOG_DIR"
DEFAULT_ENV="$ROOT/configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"
STRICT_ENV="$ROOT/configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json"
eval_one(){ local arm=$1 seed=$2 scen=$3 env=$4; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$ROOT/scripts/evaluate_pc_ppo.py" --training-config "$CFG_DIR/${arm}.json" --checkpoint "$OUT_ROOT/${arm}_s${seed}/final.pt" --environment-config "$env" --output "$EVAL_ROOT/${arm}_s${seed}_${scen}_policy" --seed-start 20 --seed-count 40 --device cpu --dtype float64 --action-mode policy --success-frame both >"$LOG_DIR/${arm}_s${seed}_${scen}.log" 2>&1; }
for arm in e4 c10 c20 an w0 e4_c10_an; do for seed in 86201; do eval_one $arm $seed default $DEFAULT_ENV & eval_one $arm $seed strict125 $STRICT_ENV & done; done
wait; echo GOAL92_EVAL_COMPLETE
