#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=${ROOT:-artifacts/lowlevel/goal128_reward_terminal_credit_20261111}
SOURCE_ROOT=${SOURCE_ROOT:-artifacts/lowlevel/goal126_sac_pa_bias_20261111}
STRICT_ENV=${STRICT_ENV:-artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/B15.json}
DEFAULT_ENV=${DEFAULT_ENV:-artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/default_B15.json}
MULTIPLIERS=(0.125 1.0)
SEEDS=(86201 86202 86203 86204 86205 86206 86207 86208)

mkdir -p "$ROOT/logs"

run_eval() {
  local tag="$1"
  local config="$2"
  local checkpoint="$3"
  local env="$4"
  local output="$5"
  local scale="$6"
  local mode="$7"
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi
  local args=(
    scripts/evaluate_pc_sac.py
    --config "$config"
    --environment-config "$env"
    --output "$output"
    --trace-output "$output"
    --trace-reward-breakdown
    --rich-trace
    --seed-start 20
    --seed-count 40
    --success-frame both
    --device cpu
    --dtype float64
    --action-mode "$mode"
    --residual-scale "$scale"
  )
  if [[ "$checkpoint" != "none" ]]; then
    args+=(--checkpoint "$checkpoint")
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src nohup "$PY" "${args[@]}" \
    >"$ROOT/logs/${tag}.log" 2>&1 &
  echo $! >"$ROOT/logs/${tag}.pid"
  echo "launched $tag pid=$!"
}

for scenario in strict default; do
  if [[ "$scenario" == "strict" ]]; then
    env="$STRICT_ENV"
  else
    env="$DEFAULT_ENV"
  fi
  run_eval \
    "zero_${scenario}" \
    "$SOURCE_ROOT/training_configs/S_s86201.json" \
    none \
    "$env" \
    "$ROOT/zero/$scenario" \
    1.0 \
    zero-residual
done

for multiplier in "${MULTIPLIERS[@]}"; do
  tag=${multiplier//./p}
  for seed in "${SEEDS[@]}"; do
    config="$SOURCE_ROOT/training_configs/S_s${seed}.json"
    checkpoint="$SOURCE_ROOT/screen/S_s${seed}/final.zip"
    if [[ ! -f "$config" ]]; then
      echo "missing config: $config" >&2
      exit 1
    fi
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      exit 1
    fi
    for scenario in strict default; do
      if [[ "$scenario" == "strict" ]]; then
        env="$STRICT_ENV"
      else
        env="$DEFAULT_ENV"
      fi
      run_eval \
        "m${tag}_S_s${seed}_${scenario}" \
        "$config" \
        "$checkpoint" \
        "$env" \
        "$ROOT/dose_m${tag}/S_s${seed}/$scenario" \
        "$multiplier" \
        policy
    done
  done
done

echo "launched 34 Goal128 trace evaluations"
