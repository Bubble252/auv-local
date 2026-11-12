#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
PY="${PY:-python3}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/goal63_terminal_suppression}"
SEED_START="${SEED_START:-20}"
SEED_COUNT="${SEED_COUNT:-40}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SUFFIX="_m025"
CHECKPOINT_ROOT="/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts"
TRAINING_CONFIG="$ROOT/configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json"
EVALUATOR="$ROOT/scripts/evaluate_pc_command_residual_offpolicy.py"

mkdir -p "$OUT_ROOT/logs"

run_job() {
  local scenario="$1"
  local checkpoint="$2"
  local action_mode="$3"
  local env_config="$4"
  local goal_radius_m="$5"
  local guard_radius_m="$6"
  local candidate_radius_m
  candidate_radius_m="$(awk -v r="$goal_radius_m" 'BEGIN { printf "%.2f", r - 0.25 }')"
  local name
  local output
  if [[ "$action_mode" == "policy" ]]; then
    name="${scenario}_policy_s${checkpoint}${SUFFIX}"
    output="$OUT_ROOT/$name"
    mkdir -p "$output"
    "$PY" "$EVALUATOR" \
      --config "$TRAINING_CONFIG" \
      --environment-config "$env_config" \
      --checkpoint "$CHECKPOINT_ROOT/formal_s${checkpoint}/final.pt" \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --device cpu \
      --dtype float64 \
      --action-mode policy \
      --eval-goal-radius-m "$candidate_radius_m" \
      --eval-goal-plane-miss-guard-radius-m "$guard_radius_m" \
      >"$OUT_ROOT/logs/${name}.log" 2>&1
  else
    name="${scenario}_zero${SUFFIX}"
    output="$OUT_ROOT/$name"
    mkdir -p "$output"
    "$PY" "$EVALUATOR" \
      --config "$TRAINING_CONFIG" \
      --environment-config "$env_config" \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --device cpu \
      --dtype float64 \
      --action-mode zero-residual \
      --eval-goal-radius-m "$candidate_radius_m" \
      --eval-goal-plane-miss-guard-radius-m "$guard_radius_m" \
      >"$OUT_ROOT/logs/${name}.log" 2>&1
  fi
  echo "DONE $name"
}

launch() {
  run_job "$@" &
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
}

for scenario in default strict125; do
  if [[ "$scenario" == "default" ]]; then
    env_config="$ROOT/configs/lowlevel/pc_goal59_terminal_bounded_default_20261109.json"
    radius="2.00"
  else
    env_config="$ROOT/configs/lowlevel/pc_goal59_terminal_bounded_strict125_20261109.json"
    radius="1.25"
  fi
  for checkpoint in 240 752 880 1008; do
    launch "$scenario" "$checkpoint" "policy" "$env_config" "$radius" "$radius"
  done
  launch "$scenario" "none" "zero-residual" "$env_config" "$radius" "$radius"
done
wait
echo "ALL DONE: $OUT_ROOT"
