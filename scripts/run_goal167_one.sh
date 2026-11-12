#!/usr/bin/env bash
# Run one Goal167 action-parameterization arm on one GPU.
set -euo pipefail

GPU=${1:?gpu index}
ARM=${2:?arm name}
SEED=${3:?seed}

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal167_action_parameterization_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}

cd "$REPO"
export PYTHONPATH="src:$SB3"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$GPU"

CONFIG="configs/lowlevel/goal167/train_${ARM,,}_seed${SEED}.json"
OUTPUT="$TRAIN_ROOT/$ARM/seed$SEED"
mkdir -p "$OUTPUT"

if [[ -f "$OUTPUT/final.zip" ]]; then
  echo "SKIP $ARM seed$SEED (final.zip exists)"
  exit 0
fi

echo "START arm=$ARM seed=$SEED gpu=$GPU config=$CONFIG output=$OUTPUT"
"$PY" scripts/train_pc_command_residual_sac.py \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --device cuda \
  --dtype float64 \
  --sb3-root "$SB3" \
  --overwrite
echo "DONE arm=$ARM seed=$SEED gpu=$GPU"
