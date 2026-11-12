#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=artifacts/lowlevel/goal121_action_authority_20261111
TRAIN=artifacts/lowlevel/goal119_ppo_credit_approach_20261111/training_configs/C3_s86201.json
ENV=artifacts/lowlevel/goal119_ppo_credit_approach_20261111/environment_configs/C3.json
CKPT=artifacts/lowlevel/goal119_ppo_credit_approach_20261111/C3_s86201/final.pt
mkdir -p "$ROOT/logs"

JOBS=(
  "zero:zero"
  "ppo:ppo"
  "neg-ppo:neg-ppo"
  "ppo-2x:ppo-2x"
  "ppo-4x:ppo-4x"
  "rudder+:const:0.2,0,0"
  "stern+:const:0,0.2,0"
  "stern-:const:0,-0.2,0"
  "rpm+:const:0,0,0.2"
  "rpm-:const:0,0,-0.2"
)

for entry in "${JOBS[@]}"; do
  name="${entry%%:*}"
  profile="${entry#*:}"
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
    --environment-config "$ENV" \
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

echo "launched ${#JOBS[@]} authority profiles"
