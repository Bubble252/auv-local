#!/usr/bin/env bash
# Goal135: nominal + current15, strict/default, 40 dev seeds for each screen checkpoint.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=${ROOT:-artifacts/lowlevel/goal135_authority_regularized_sac_20261111}
ARMS=${ARMS:-"R030 R025 R020"}
SEEDS=${SEEDS:-"86201 86202 86203 86204 86205 86206 86207 86208"}
LEVELS=${LEVELS:-"nominal current15"}
SCENARIOS=${SCENARIOS:-"strict default"}
PARALLEL=${PARALLEL:-16}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$ROOT/logs"
JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT

for arm in $ARMS; do
  for seed in $SEEDS; do
    for level in $LEVELS; do
      for scenario in $SCENARIOS; do
        echo "$arm $seed $level $scenario" >>"$JOB_FILE"
      done
    done
  done
done

run_one() {
  local arm="$1"
  local seed="$2"
  local level="$3"
  local scenario="$4"
  local config="$ROOT/training_configs/$arm/S_s${seed}.json"
  local checkpoint="$ROOT/screen/$arm/S_s${seed}/final.zip"
  local environment="configs/lowlevel/goal134/pc_goal134_${level}_${scenario}_20261111.json"
  local output="$ROOT/eval/$arm/$level/$scenario/S_s${seed}"
  local log="$ROOT/logs/eval_${arm}_${level}_${scenario}_S_s${seed}.log"

  if [[ ! -f "$config" ]]; then
    echo "missing config: $config" >&2
    return 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    return 1
  fi
  if [[ ! -f "$environment" ]]; then
    echo "missing environment config: $environment" >&2
    return 1
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $arm/$level/$scenario/S_s$seed (summary exists)"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi

  echo "START $arm/$level/$scenario/S_s$seed"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH=src \
    "$PY" scripts/evaluate_pc_sac.py \
    --config "$config" \
    --environment-config "$environment" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --seed-start "$SEED_START" \
    --seed-count "$SEED_COUNT" \
    --success-frame both \
    --device cpu \
    --dtype float64 \
    --action-mode policy \
    --residual-scale 1.0 \
    >"$log" 2>&1
  echo "DONE $arm/$level/$scenario/S_s$seed"
}

export PY ROOT SEED_START SEED_COUNT
export -f run_one

if [[ "$DRY_RUN" == "1" ]]; then
  cat "$JOB_FILE"
  exit 0
fi

echo "launching $(wc -l <"$JOB_FILE") Goal135 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 4 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
while read -r arm seed level scenario; do
  if [[ ! -f "$ROOT/eval/$arm/$level/$scenario/S_s$seed/summary.json" ]]; then
    echo "MISSING $arm/$level/$scenario/S_s$seed" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"

if [[ "$missing" -ne 0 ]]; then
  echo "$missing evaluations are missing; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal135 screen evaluation complete: $ROOT"
