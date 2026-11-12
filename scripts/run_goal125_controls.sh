#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=artifacts/lowlevel/goal125_bias_ppo_20261111
BASE=artifacts/lowlevel/goal119_ppo_credit_approach_20261111
STRICT="$BASE/environment_configs/C3.json"
DEFAULT="$ROOT/environment_configs/default_Z15.json"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

JOBS=(
  "86201"
  "86202"
)

for seed in "${JOBS[@]}"; do
  config="$BASE/training_configs/C3_s${seed}.json"
  checkpoint="$BASE/C3_s${seed}/final.pt"
  for scenario in strict default; do
    if [[ "$scenario" == "strict" ]]; then
      env="$STRICT"
    else
      env="$DEFAULT"
    fi
    output="$ROOT/controls/$seed/$scenario"
    if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
      echo "refusing to overwrite non-empty output: $output" >&2
      exit 1
    fi
    echo "control $seed / $scenario"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src nohup "$PY" \
      scripts/evaluate_pc_ppo.py \
      --training-config "$config" \
      --checkpoint "$checkpoint" \
      --environment-config "$env" \
      --output "$output" \
      --seed-start 20 \
      --seed-count 40 \
      --success-frame both \
      --device cpu \
      --action-mode injected-residual \
      --injected-residual-profile const:0,-0.25,0 \
      --actuator-residual-scale-override 2.00 \
      --actuator-residual-channel-gains 1.000000,1.000000,1.000000 \
      --allow-guidance-rate-limit-override \
      --trace-output "$output/trace.npz" \
      --trace-reward-breakdown \
      >"$LOG_DIR/control_${seed}_${scenario}.log" 2>&1 &
    echo $! >"$LOG_DIR/control_${seed}_${scenario}.pid"
  done
done

echo "launched 4 fixed-bias control evals"
