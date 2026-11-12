#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$REPO/.venv/bin/python}"
ROOT="${ROOT:-/dev/shm/pc_v1_teacher_audit_20261108}"
CHECKPOINT="${CHECKPOINT:-/dev/shm/pc_v0d_nominal_bc_rebase_20261107/bc_N8/pc_bc_actor.pt}"

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$ROOT/logs"

run_eval() {
  local variant="$1"
  local scenario="$2"
  local config="$3"
  local output="$ROOT/$variant/$scenario"
  local log="$ROOT/logs/${variant}_${scenario}.log"
  if [[ -e "$output/summary.json" ]]; then
    echo "refusing to overwrite existing result: $output" >&2
    exit 1
  fi
  mkdir -p "$output"
  echo "=== $(date -Is) $variant $scenario ===" | tee "$log"
  "$PY" "$REPO/scripts/evaluate_pc_bc.py" \
    --config "$config" \
    --checkpoint "$CHECKPOINT" \
    --output "$output" \
    --seed-start 20 \
    --seed-count 40 \
    --device cpu \
    --dtype float64 \
    --path-dim 8 \
    --termination-mode endpoint_stop 2>&1 | tee -a "$log"
}

for scenario in default strict125; do
  run_eval \
    fixed_teacher \
    "$scenario" \
    "$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_${scenario}_20261108.json"
done

for scenario in default strict125; do
  suffix=""
  if [[ "$scenario" == "strict125" ]]; then
    suffix="_strict125"
  fi
  run_eval \
    control \
    "$scenario" \
    "$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current${suffix}.json"
done

"$PY" "$REPO/scripts/summarize_pc_goal24_teacher_audit.py" --root "$ROOT"
