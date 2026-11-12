#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式}"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
ROOT="${ROOT:-$REPO/artifacts/lowlevel/pc_v0d_r2b_rate_teacher_20261107}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/pc_v0d_r2b_rate_teacher_20261107}"

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$ROOT/teacher_screen" "$LOG_ROOT"

for tag in rate035_018 rate020_012; do
  for mode in default strict125; do
    suffix=""
    if [[ "$mode" == "strict125" ]]; then
      suffix="_strict125"
    fi
    config="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_${tag}_20261107${suffix}.json"
    output="$ROOT/teacher_screen/${tag}_${mode}"
    log="$LOG_ROOT/teacher_screen_${tag}_${mode}.log"
    if [[ -e "$output/teacher_summary.json" ]]; then
      echo "refusing to overwrite existing result: $output" >&2
      exit 1
    fi
    mkdir -p "$output"
    echo "=== $(date -Is) ${tag} ${mode} ===" | tee "$log"
    "$PY" "$REPO/scripts/evaluate_pc_teacher.py" \
      --config "$config" \
      --output "$output" \
      --seed-start 0 \
      --seed-count 20 \
      --device cpu \
      --dtype float64 2>&1 | tee -a "$log"
  done
done

echo "rate_teacher_screen_complete $(date -Is)" | tee "$LOG_ROOT/teacher_screen_complete.log"
