#!/usr/bin/env bash
# Goal134 post-Gate falsification: learner residual authority dose-response.
#
# Usage:
#   LEVEL=current15 ARMS="S_s86201 S_s86202 S_s86204" \
#     SCALES="0.5 0.25" PARALLEL=6 bash scripts/run_pc_goal134_authority_dose.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-artifacts/lowlevel/goal126_sac_pa_bias_20261111}
ROOT=${ROOT:-artifacts/lowlevel/goal134_frozen_sac_stress_20261111/diagnostic/authority_dose}
CONFIG=${CONFIG:-$SOURCE_ROOT/training_configs/S_s86201.json}
LEVEL=${LEVEL:-current15}
ARMS=${ARMS:-"S_s86201 S_s86202 S_s86204"}
SCALES=${SCALES:-"0.5 0.25"}
SCENARIOS=${SCENARIOS:-"strict default"}
PARALLEL=${PARALLEL:-6}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}

mkdir -p "$ROOT/logs"
JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT

for arm in $ARMS; do
  for scale in $SCALES; do
    for scenario in $SCENARIOS; do
      echo "$arm $scale $scenario" >>"$JOB_FILE"
    done
  done
done

run_one() {
  local arm="$1"
  local scale="$2"
  local scenario="$3"
  local seed="${arm#S_s}"
  local checkpoint="$SOURCE_ROOT/screen/$arm/final.zip"
  local config="$SOURCE_ROOT/training_configs/S_s$seed.json"
  local env_config="configs/lowlevel/goal134/pc_goal134_${LEVEL}_${scenario}_20261111.json"
  local output="$ROOT/$arm/scale_${scale}_${scenario}"
  local log="$ROOT/logs/${arm}_scale_${scale}_${scenario}.log"
  if [[ ! -f "$checkpoint" ]] || [[ ! -f "$config" ]] || [[ ! -f "$env_config" ]]; then
    echo "missing checkpoint/config for $arm/$scale/$scenario" >&2
    return 1
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $arm scale=$scale $scenario"
    return 0
  fi
  echo "START $arm scale=$scale $scenario"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
    "$PY" scripts/evaluate_pc_sac.py \
    --config "$config" \
    --environment-config "$env_config" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --seed-start "$SEED_START" \
    --seed-count "$SEED_COUNT" \
    --success-frame both \
    --device cpu \
    --dtype float64 \
    --action-mode policy \
    --residual-scale "$scale" \
    >"$log" 2>&1
  echo "DONE $arm scale=$scale $scenario"
}

export PY SOURCE_ROOT ROOT LEVEL SEED_START SEED_COUNT
export -f run_one

echo "launching $(wc -l <"$JOB_FILE") authority-dose evaluations"
xargs -r -n 3 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"
echo "authority-dose complete: $ROOT"
