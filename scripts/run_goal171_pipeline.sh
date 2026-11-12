#!/usr/bin/env bash
# Goal171 one-command pipeline: train -> audit -> full eval -> audit -> analysis.
set -euo pipefail

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112}
MODE=${1:-all}
ARMS=${ARMS:-"R0_BC_PPO R0_BC_SAC R0_ALOS_PPO R0_ALOS_SAC R1_BC_PPO R1_BC_SAC R1_ALOS_PPO R1_ALOS_SAC"}
SEEDS=${SEEDS:-"86201"}

cd "$REPO"

run_train() {
  bash scripts/run_goal171_train.sh
}

run_eval() {
  bash scripts/run_goal171_eval.sh
}

run_analysis() {
  mkdir -p "$ROOT/analysis_full"
  PYTHONPATH="src:$SB3" \
    "$PY" scripts/analyze_goal170_route_transfer.py \
    --root "$ROOT" \
    --arms $ARMS \
    --levels nominal current07 current10 current15 \
    --radii default strict125 \
    --checkpoint-seeds $SEEDS \
    --zero-arm zero_sac50 \
    --ppo-zero-arm zero_alos50 \
    --ppo-bc-zero-arm zero_bc50 \
    --output "$ROOT/analysis_full"
}

case "$MODE" in
  train) run_train ;;
  eval) run_eval ;;
  analysis) run_analysis ;;
  all)
    run_train
    run_eval
    run_analysis
    ;;
  *)
    echo "usage: $0 [train|eval|analysis|all]" >&2
    exit 2
    ;;
esac
