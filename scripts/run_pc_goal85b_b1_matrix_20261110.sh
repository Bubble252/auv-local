#!/usr/bin/env bash
# Goal85b B1 frozen eval-only matrix: 60 runs x 40 dev episodes = 2400 episodes.
# Usage: bash scripts/run_pc_goal85b_b1_matrix_20261110.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
OUT_ROOT="${OUT_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/goal85b_plant_mismatch_20261110/b1}"
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
  local level="$1" scenario="$2" root="$3" action="$4" ckpt="$5"
  local config="configs/lowlevel/goal85b/pc_goal85b_${level}_${scenario}_${root}_20261110.json"
  local out="$OUT_ROOT/$level/$scenario/$root/$action/$ckpt"
  local log="$OUT_ROOT/logs/${level}_${scenario}_${root}_${action}_${ckpt}.log"
  local name="${level}_${scenario}_${root}_${action}_${ckpt}"
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
      --environment-config "$config" \
      --checkpoint "$CKPT_ROOT/formal_${ckpt}/final.pt" \
      --output "$out" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --device "$DEVICE" --dtype "$DTYPE" \
      --action-mode "$action" --success-frame eta
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

for level in m0 m05 m10; do
  for scenario in default strict125; do
    for root in baseline candidate; do
      for ckpt in s240 s752 s880 s1008; do
        launch "$level" "$scenario" "$root" policy "$ckpt"
      done
      launch "$level" "$scenario" "$root" zero-residual s240
    done
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
