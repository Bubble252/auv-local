#!/usr/bin/env bash
# Goal134 frozen eval-only stress matrix.
#
# Modes:
#   MODE=zero    -> zero-residual only (5 stress x 2 scenarios = 10 runs)
#   MODE=full    -> zero + 8 Goal126 SAC checkpoints (90 runs)
#
# Usage:
#   MODE=zero PARALLEL=8 bash scripts/run_goal134_stress_matrix.sh
#   MODE=full PARALLEL=8 bash scripts/run_goal134_stress_matrix.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=${ROOT:-artifacts/lowlevel/goal134_frozen_sac_stress_20261111}
SOURCE_ROOT=${SOURCE_ROOT:-artifacts/lowlevel/goal126_sac_pa_bias_20261111}
CONFIG=${CONFIG:-$SOURCE_ROOT/training_configs/S_s86201.json}
MODE=${MODE:-zero}
PARALLEL=${PARALLEL:-8}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
TRACE=${TRACE:-0}
DRY_RUN=${DRY_RUN:-0}

STRESS_LEVELS=${STRESS_LEVELS:-"nominal current15 current20 plant20 combined20"}
SCENARIOS=${SCENARIOS:-"strict default"}
CHECKPOINT_SEEDS=${CHECKPOINT_SEEDS:-"86201 86202 86203 86204 86205 86206 86207 86208"}

mkdir -p "$ROOT/logs"
JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT

if [[ "$DRY_RUN" != "1" ]] && [[ ! -f "$CONFIG" ]]; then
  echo "missing training config: $CONFIG" >&2
  exit 1
fi

for level in $STRESS_LEVELS; do
  for scenario in $SCENARIOS; do
    echo "$level $scenario zero" >>"$JOB_FILE"
    if [[ "$MODE" == "full" ]]; then
      for seed in $CHECKPOINT_SEEDS; do
        echo "$level $scenario S_s$seed" >>"$JOB_FILE"
      done
    fi
  done
done

run_one() {
  local level="$1"
  local scenario="$2"
  local arm="$3"
  local env_config="configs/lowlevel/goal134/pc_goal134_${level}_${scenario}_20261111.json"
  local output="$ROOT/$level/$scenario/$arm"
  local log="$ROOT/logs/${level}_${scenario}_${arm}.log"
  local checkpoint=""
  local action_mode="policy"

  if [[ ! -f "$env_config" ]]; then
    echo "missing environment config: $env_config" >&2
    return 1
  fi
  if [[ "$arm" == "zero" ]]; then
    action_mode="zero-residual"
  else
    checkpoint="$SOURCE_ROOT/screen/$arm/final.zip"
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      return 1
    fi
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $level/$scenario/$arm (summary exists)"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi

  local extra=()
  if [[ "$action_mode" == "policy" ]]; then
    extra+=(--checkpoint "$checkpoint")
  fi
  if [[ "$TRACE" == "1" ]]; then
    extra+=(--trace-output "$output" --rich-trace)
  fi

  echo "START $level/$scenario/$arm"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
    "$PY" scripts/evaluate_pc_sac.py \
    --config "$CONFIG" \
    --environment-config "$env_config" \
    --output "$output" \
    --seed-start "$SEED_START" \
    --seed-count "$SEED_COUNT" \
    --success-frame both \
    --device cpu \
    --dtype float64 \
    --action-mode "$action_mode" \
    --residual-scale 1.0 \
    "${extra[@]}" \
    >"$log" 2>&1
  echo "DONE $level/$scenario/$arm"
}

export PY ROOT SOURCE_ROOT CONFIG SEED_START SEED_COUNT TRACE
export -f run_one

if [[ "$DRY_RUN" == "1" ]]; then
  cat "$JOB_FILE"
  exit 0
fi

echo "launching $(wc -l <"$JOB_FILE") Goal134 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 3 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
while read -r level scenario arm; do
  if [[ ! -f "$ROOT/$level/$scenario/$arm/summary.json" ]]; then
    echo "MISSING $level/$scenario/$arm" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"

if [[ "$missing" -ne 0 ]]; then
  echo "$missing evaluations are missing; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal134 matrix complete: $ROOT"
