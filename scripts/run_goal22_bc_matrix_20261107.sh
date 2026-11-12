#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/dev/shm/pc_v0d_nominal_bc_rebase_20261107}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_DIR="$REPO_ROOT/configs/lowlevel"
G19_CHECKPOINT=${G19_CHECKPOINT:-$REPO_ROOT/artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/bc_satw_dagger2/pc_bc_actor.pt}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-20}
FORCE=${FORCE:-0}
ARMS=${ARMS:-"N8 N12 g19_dagger"}
SCENARIOS=${SCENARIOS:-"default strict125"}
SUMMARY=${SUMMARY:-1}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

cd "$REPO_ROOT"
mkdir -p "$ROOT/eval_goal22" "$ROOT/logs"

run_eval() {
  local arm=$1
  local scenario=$2
  local suffix=$3
  local config=$4
  local checkpoint=$5
  local path_dim=$6
  shift 6

  local output="$ROOT/eval_goal22/$arm/$scenario/$suffix"
  local log="$ROOT/logs/eval_goal22_${arm}_${scenario}_${suffix}.log"
  if [[ -f "$output/summary.json" && "$FORCE" != "1" ]]; then
    printf 'skip existing %s\n' "$output"
    return
  fi
  mkdir -p "$output"
  printf 'evaluating %s/%s/%s -> %s\n' "$arm" "$scenario" "$suffix" "$output"
  uv run python scripts/evaluate_pc_bc.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --seed-start "$SEED_START" \
    --seed-count "$SEED_COUNT" \
    --device cpu \
    --dtype float64 \
    --path-dim "$path_dim" \
    --termination-mode endpoint_stop \
    "$@" >"$log" 2>&1
}

if [[ ! -f "$ROOT/bc_N8/pc_bc_actor.pt" || ! -f "$ROOT/bc_N12/pc_bc_actor.pt" ]]; then
  printf 'missing N8/N12 BC checkpoint under %s\n' "$ROOT" >&2
  exit 2
fi
if [[ ! -f "$G19_CHECKPOINT" ]]; then
  printf 'missing g19 reference checkpoint: %s\n' "$G19_CHECKPOINT" >&2
  exit 2
fi

for arm in $ARMS; do
  path_dim=${arm#N}
  if [[ "$arm" == "g19_dagger" ]]; then
    path_dim=8
  fi
  for scenario in $SCENARIOS; do
    config="$CONFIG_DIR/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_${arm,,}_${scenario}_20261107.json"
    if [[ "$arm" == "g19_dagger" ]]; then
      config="$CONFIG_DIR/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n8_${scenario}_20261107.json"
    fi
    checkpoint="$ROOT/bc_${arm}/pc_bc_actor.pt"
    if [[ "$arm" == "g19_dagger" ]]; then
      checkpoint="$G19_CHECKPOINT"
    fi
    run_eval "$arm" "$scenario" normal "$config" "$checkpoint" "$path_dim"
    if [[ "$arm" != "g19_dagger" ]]; then
      run_eval "$arm" "$scenario" path_zero "$config" "$checkpoint" "$path_dim" --path-zero
    fi
  done
done

if [[ "$SUMMARY" == "1" ]]; then
  uv run python scripts/summarize_goal22_nominal_bc_20261107.py --root "$ROOT"
fi
