#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=artifacts/lowlevel/goal120_ppo_optimization_20261111
CONFIG_DIR="$ROOT/training_configs"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

JOBS=(
  "0 P1_s86201"
  "1 P1_s86202"
  "2 P2_s86201"
  "3 P2_s86202"
  "4 P3_s86201"
  "5 P3_s86202"
)

for entry in "${JOBS[@]}"; do
  read -r gpu job <<<"$entry"
  config="$CONFIG_DIR/$job.json"
  output="$ROOT/$job"
  if [[ ! -f "$config" ]]; then
    echo "missing config: $config" >&2
    exit 1
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    exit 1
  fi
  echo "starting $job on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src nohup "$PY" \
    scripts/train_batch_lowlevel_ppo.py \
    --training-config "$config" \
    --device cuda \
    >"$LOG_DIR/$job.log" 2>&1 &
  echo $! >"$LOG_DIR/$job.pid"
done

echo "launched ${#JOBS[@]} jobs"
