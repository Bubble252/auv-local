#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=artifacts/lowlevel/goal125_bias_ppo_20261111
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

JOBS=(
  "Z15_s86201"
  "Z15_s86202"
  "B15_s86201"
  "B15_s86202"
  "B20_s86201"
  "B20_s86202"
  "BP3_s86201"
  "BP3_s86202"
)

for job in "${JOBS[@]}"; do
  arm="${job%%_*}"
  checkpoint="$ROOT/train/$job/final.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  for scenario in strict default; do
    if [[ "$scenario" == "strict" ]]; then
      env="$ROOT/environment_configs/$arm.json"
    else
      env="$ROOT/environment_configs/default_$arm.json"
    fi
    output="$ROOT/eval/$job/$scenario"
    if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
      echo "refusing to overwrite non-empty output: $output" >&2
      exit 1
    fi
    echo "evaluating $job / $scenario"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src nohup "$PY" \
      scripts/evaluate_pc_ppo.py \
      --training-config "$ROOT/training_configs/$job.json" \
      --checkpoint "$checkpoint" \
      --environment-config "$env" \
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
      >"$LOG_DIR/eval_${job}_${scenario}.log" 2>&1 &
    echo $! >"$LOG_DIR/eval_${job}_${scenario}.pid"
  done
done

echo "launched $(( ${#JOBS[@]} * 2 )) eval jobs"
