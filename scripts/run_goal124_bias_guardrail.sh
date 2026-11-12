#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=artifacts/lowlevel/goal124_bias_guardrail_20261111
TRAIN=artifacts/lowlevel/goal119_ppo_credit_approach_20261111/training_configs/C3_s86201.json
CKPT=artifacts/lowlevel/goal119_ppo_credit_approach_20261111/C3_s86201/final.pt
STRICT=artifacts/lowlevel/goal119_ppo_credit_approach_20261111/environment_configs/C3.json
DEFAULT=$ROOT/environment_configs/default.json
mkdir -p "$ROOT/logs"

JOBS=(
  "default_zero:$DEFAULT:zero"
  "default_stern-025:$DEFAULT:const:0,-0.25,0"
  "default_stern-030:$DEFAULT:const:0,-0.30,0"
  "strict_zero:$STRICT:zero"
  "strict_stern-025:$STRICT:const:0,-0.25,0"
  "strict_stern-030:$STRICT:const:0,-0.30,0"
)

for entry in "${JOBS[@]}"; do
  name="${entry%%:*}"
  rest="${entry#*:}"
  env="${rest%%:*}"
  profile="${rest#*:}"
  output="$ROOT/$name"
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    exit 1
  fi
  echo "launching $name ($profile)"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src nohup "$PY" \
    scripts/evaluate_pc_ppo.py \
    --training-config "$TRAIN" \
    --checkpoint "$CKPT" \
    --environment-config "$env" \
    --output "$output" \
    --seed-start 20 \
    --seed-count 40 \
    --success-frame both \
    --device cpu \
    --action-mode injected-residual \
    --injected-residual-profile "$profile" \
    --actuator-residual-scale-override 2.00 \
    --actuator-residual-channel-gains 1.000000,1.000000,1.000000 \
    --allow-guidance-rate-limit-override \
    --trace-output "$output/trace.npz" \
    --trace-reward-breakdown \
    >"$ROOT/logs/$name.log" 2>&1 &
  echo $! >"$ROOT/logs/$name.pid"
done

echo "launched ${#JOBS[@]} guardrail jobs"
