#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=artifacts/lowlevel/goal120_ppo_optimization_20261111
GOAL119=artifacts/lowlevel/goal119_ppo_credit_approach_20261111
LOG_DIR="$ROOT/logs"
EVAL_ROOT="$ROOT/eval_policy"
mkdir -p "$LOG_DIR" "$EVAL_ROOT"

for seed in 86201 86202; do
  link="$EVAL_ROOT/P0_s$seed"
  if [[ ! -e "$link" ]]; then
    ln -s "$(pwd)/$GOAL119/eval_policy/C3_s$seed" "$link"
  fi
done

JOBS=(
  "P1_s86201"
  "P1_s86202"
  "P2_s86201"
  "P2_s86202"
  "P3_s86201"
  "P3_s86202"
)

for job in "${JOBS[@]}"; do
  arm="${job%%_*}"
  checkpoint="$ROOT/$job/final.pt"
  output="$EVAL_ROOT/$job"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    exit 1
  fi
  echo "evaluating $job"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src nohup "$PY" \
    scripts/evaluate_pc_ppo.py \
    --training-config "$ROOT/training_configs/$job.json" \
    --checkpoint "$checkpoint" \
    --environment-config "$ROOT/environment_configs/$arm.json" \
    --output "$output" \
    --seed-start 20 \
    --seed-count 40 \
    --success-frame both \
    --device cpu \
    --action-mode policy \
    --actuator-residual-scale-override 2.00 \
    --actuator-residual-channel-gains 1.000000,1.000000,1.000000 \
    --allow-guidance-rate-limit-override \
    --trace-output "$output/trace.npz" \
    --trace-reward-breakdown \
    >"$LOG_DIR/eval_policy_$job.log" 2>&1 &
  echo $! >"$LOG_DIR/eval_policy_$job.pid"
done

echo "launched ${#JOBS[@]} eval jobs"
