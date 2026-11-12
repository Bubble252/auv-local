#!/usr/bin/env bash
# Goal85 frozen eval-only matrix: 20 runs x 40 dev episodes.
# Usage: bash scripts/run_pc_goal85_matrix_20261110.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
OUT_ROOT="${OUT_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/goal85_unbiased_nav_20261110}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json}"
CKPT_ROOT="${CKPT_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts}"
SEED_START="${SEED_START:-20}"
SEED_COUNT="${SEED_COUNT:-40}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float64}"

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/logs"

pids=()
names=()

launch() {
  local root="$1" scenario="$2" action="$3" ckpt="$4"
  local out="$OUT_ROOT/$scenario/$root/$action/$ckpt"
  local log="$OUT_ROOT/logs/${scenario}_${root}_${action}_${ckpt}.log"
  local name="${scenario}_${root}_${action}_${ckpt}"
  if [[ -f "$out/summary.json" ]]; then
    echo "SKIP $name (summary exists)"
    return
  fi
  mkdir -p "$out"
  echo "START $name"
  (
    export PYTHONPATH=src
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    "$PYTHON" scripts/evaluate_pc_command_residual_offpolicy.py \
      --config "$TRAIN_CONFIG" \
      --environment-config "configs/lowlevel/pc_goal85_unbiased_${scenario}_${root}_20261110.json" \
      --checkpoint "$CKPT_ROOT/formal_${ckpt}/final.pt" \
      --output "$out" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --device "$DEVICE" --dtype "$DTYPE" \
      --action-mode "$action" --success-frame eta \
      --trace-output "$out"
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

for root in baseline candidate; do
  for scenario in default strict125; do
    for ckpt in s240 s752 s880 s1008; do
      launch "$root" "$scenario" policy "$ckpt"
    done
    launch "$root" "$scenario" zero-residual s240
  done
done

fail=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "FAIL ${names[$index]}"
    fail=1
  else
    echo "DONE ${names[$index]}"
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "matrix finished with failures; inspect $OUT_ROOT/logs"
  exit 1
fi
echo "matrix complete: $OUT_ROOT"
