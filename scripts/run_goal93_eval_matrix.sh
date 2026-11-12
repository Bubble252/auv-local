#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/lowlevel/goal93_direction_authority_20261111}"
SEED_START="${SEED_START:-20}"
SEED_COUNT="${SEED_COUNT:-40}"
SCALE_ROOT="$OUT_ROOT/scale_sweep"
TRACE_ROOT="$OUT_ROOT/traces"
LOG_ROOT="$OUT_ROOT/logs"
DEFAULT_ENV="$ROOT/configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json"
STRICT_ENV="$ROOT/configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json"
mkdir -p "$SCALE_ROOT" "$TRACE_ROOT" "$LOG_ROOT"

declare -A CFG=(
  [pg10_3]="$ROOT/configs/lowlevel/goal90_20261111/pg10_3.json"
  [c10]="$ROOT/configs/lowlevel/goal92_20261111/c10.json"
)
declare -A CKPT=(
  [pg10_3]="$ROOT/artifacts/lowlevel/goal90_gate_scope_20261111/pg10_3_s86201/final.pt"
  [c10]="$ROOT/artifacts/lowlevel/goal92_update_schedule_20261111/c10_s86201/final.pt"
)

scale_tag() {
  local value="$1"
  printf '%s' "${value/./p}"
}

env_for() {
  if [[ "$1" == "default" ]]; then
    printf '%s' "$DEFAULT_ENV"
  else
    printf '%s' "$STRICT_ENV"
  fi
}

run_eval() {
  local arm="$1" scenario="$2" scale="$3" mode="$4" trace="$5"
  local cfg="${CFG[$arm]}"
  local ckpt="${CKPT[$arm]}"
  local env_config
  env_config="$(env_for "$scenario")"
  local tag
  tag="$(scale_tag "$scale")"
  local name="${arm}_${scenario}_scale${tag}_${mode}"
  local output="$SCALE_ROOT/$name"
  local log="$LOG_ROOT/$name.log"
  local -a extra=()
  if [[ "$trace" == "1" ]]; then
    output="$TRACE_ROOT/$name"
    extra+=(--trace-output "$output/trace.npz")
  fi
  echo "[$(date -Is)] start $name"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" "$ROOT/scripts/evaluate_pc_ppo.py" \
    --training-config "$cfg" \
    --checkpoint "$ckpt" \
    --environment-config "$env_config" \
    --output "$output" \
    --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
    --device cpu --dtype float64 \
    --action-mode "$mode" \
    --success-frame both \
    --actuator-residual-scale-override "$scale" \
    "${extra[@]}" >"$log" 2>&1
  echo "[$(date -Is)] done $name"
}

if [[ "${SMOKE:-0}" == "1" ]]; then
  run_eval pg10_3 default 1.25 policy 1
  run_eval c10 strict125 1.25 policy 0
  echo "[$(date -Is)] GOAL93_SMOKE_COMPLETE"
  exit 0
fi

# A: total residual scale sweep (24 runs).
for arm in pg10_3 c10; do
  for scenario in default strict125; do
    for scale in 1.0 1.25 1.5 1.75 2.0 2.5; do
      run_eval "$arm" "$scenario" "$scale" policy 0 &
    done
  done
done
wait

# B: policy traces at scale 1.0 and 1.25 (6 runs).
for scenario in default strict125; do
  run_eval pg10_3 "$scenario" 1.0 policy 1 &
  run_eval pg10_3 "$scenario" 1.25 policy 1 &
  run_eval c10 "$scenario" 1.25 policy 1 &
done
wait

# C: frozen-base traces at scale 1.25 (2 runs).
for scenario in default strict125; do
  run_eval pg10_3 "$scenario" 1.25 frozen-base 1 &
done
wait

echo "[$(date -Is)] GOAL93_EVAL_COMPLETE"
