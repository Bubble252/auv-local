#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=${ROOT:-artifacts/lowlevel/goal127_sac_pa_scale_dose_20261111}
SOURCE_ROOT=${SOURCE_ROOT:-artifacts/lowlevel/goal126_sac_pa_bias_20261111}
STRICT_ENV=${STRICT_ENV:-artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/B15.json}
DEFAULT_ENV=${DEFAULT_ENV:-artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/default_B15.json}
MULTIPLIERS=(0.125 0.25 0.5 0.75)

mkdir -p "$ROOT/logs"

for multiplier in "${MULTIPLIERS[@]}"; do
  tag=${multiplier//./p}
  for seed in 86201 86202 86203 86204 86205 86206 86207 86208; do
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
      output="$ROOT/dose_m${tag}/S_s${seed}/$scenario"
      if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
        echo "refusing to overwrite non-empty output: $output" >&2
        exit 1
      fi
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src nohup "$PY" \
        scripts/evaluate_pc_sac.py \
        --config "$config" \
        --environment-config "$env" \
        --checkpoint "$checkpoint" \
        --output "$output" \
        --seed-start 20 \
        --seed-count 40 \
        --success-frame both \
        --device cpu \
        --dtype float64 \
        --action-mode policy \
        --residual-scale "$multiplier" \
        >"$ROOT/logs/eval_m${tag}_S_s${seed}_${scenario}.log" 2>&1 &
      echo $! >"$ROOT/logs/eval_m${tag}_S_s${seed}_${scenario}.pid"
      echo "launched multiplier=$multiplier seed=$seed scenario=$scenario pid=$!"
    done
  done
done

echo "launched $(( ${#MULTIPLIERS[@]} * 8 * 2 )) deterministic dev40 evaluations"
