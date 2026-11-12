#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=${ROOT:-artifacts/lowlevel/goal132_geometry_terminal_correction_20261111}
SOURCE_ROOT=${SOURCE_ROOT:-artifacts/lowlevel/goal126_sac_pa_bias_20261111}
STRICT_ENV=${STRICT_ENV:-artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/B15.json}
DEFAULT_ENV=${DEFAULT_ENV:-artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/default_B15.json}
CONFIG=${CONFIG:-$SOURCE_ROOT/training_configs/S_s86201.json}
WAIT=${WAIT:-1}

mkdir -p "$ROOT/logs"

if [[ ! -f "$CONFIG" ]]; then
  echo "missing evaluation config: $CONFIG" >&2
  exit 1
fi

run_eval() {
  local arm="$1"
  local scenario="$2"
  local env_config="$3"
  local output="$4"
  shift 4
  local extra=("$@")
  local tag="${arm}_${scenario}"
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src nohup "$PY" \
    scripts/evaluate_pc_sac.py \
    --config "$CONFIG" \
    --environment-config "$env_config" \
    --output "$output" \
    --trace-output "$output" \
    --trace-reward-breakdown \
    --rich-trace \
    --seed-start 20 \
    --seed-count 40 \
    --success-frame both \
    --device cpu \
    --dtype float64 \
    --action-mode zero-residual \
    --residual-scale 1.0 \
    "${extra[@]}" \
    >"$ROOT/logs/${tag}.log" 2>&1 &
  echo $! >"$ROOT/logs/${tag}.pid"
  echo "launched $tag pid=$!"
}

run_arm() {
  local arm="$1"
  local vertical_gain="$2"
  local rpm_gain="$3"
  local extra=(--geometry-gate-enter-m 10.0 --geometry-gate-full-m 2.0)
  if [[ "$vertical_gain" != "0" ]]; then
    extra+=(--geometry-vertical-gain "$vertical_gain")
  fi
  if [[ "$rpm_gain" != "0" ]]; then
    extra+=(--geometry-rpm-gain "$rpm_gain")
  fi
  run_eval "$arm" strict "$STRICT_ENV" "$ROOT/$arm/strict" "${extra[@]}"
  run_eval "$arm" default "$DEFAULT_ENV" "$ROOT/$arm/default" "${extra[@]}"
}

run_arm Z 0 0
run_arm Vp002 0.02 0
run_arm Vp005 0.05 0
run_arm Vm002 -0.02 0
run_arm Sp005 0 0.05
run_arm Sp010 0 0.10
run_arm VSp 0.02 0.05

# Explicit zero-gain parity run exercises the added gate arguments without bias.
run_eval parity_zero strict "$STRICT_ENV" "$ROOT/parity_zero/strict" \
  --geometry-gate-enter-m 7.0 --geometry-gate-full-m 1.0
run_eval parity_zero default "$DEFAULT_ENV" "$ROOT/parity_zero/default" \
  --geometry-gate-enter-m 7.0 --geometry-gate-full-m 1.0

echo "launched 16 Goal132 evaluations"

if [[ "$WAIT" == "1" ]]; then
  wait
  echo "all Goal132 evaluations finished"
fi
