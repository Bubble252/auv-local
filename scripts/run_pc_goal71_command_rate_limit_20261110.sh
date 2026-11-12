#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
PY="${PY:-python3}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/goal71_command_rate_limit}"
SEED_START="${SEED_START:-20}"
SEED_COUNT="${SEED_COUNT:-40}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
DWELL_STEPS="${DWELL_STEPS:-50}"
SUFFIX="${SUFFIX:-_dwell${DWELL_STEPS}_m025_cmdlimit}"
CHECKPOINTS="${CHECKPOINTS:-240 752 880 1008}"
SCENARIOS="${SCENARIOS:-default strict125}"
INCLUDE_ZERO="${INCLUDE_ZERO:-1}"
RICH_TRACE="${RICH_TRACE:-1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$ROOT/configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json}"
EVALUATOR="${EVALUATOR:-$ROOT/scripts/evaluate_pc_command_residual_offpolicy.py}"
ENV_CONFIG_DEFAULT="${ENV_CONFIG_DEFAULT:-$ROOT/configs/lowlevel/pc_goal71_command_rate_limit_default_20261110.json}"
ENV_CONFIG_STRICT125="${ENV_CONFIG_STRICT125:-$ROOT/configs/lowlevel/pc_goal71_command_rate_limit_strict125_20261110.json}"

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
  if [[ "$action_mode" == "policy" ]]; then
    name="${scenario}_policy_s${checkpoint}${SUFFIX}"
  else
    name="${scenario}_zero${SUFFIX}"
  fi

  local output="$OUT_ROOT/$name"
  mkdir -p "$output"

  local checkpoint_args=()
  if [[ "$action_mode" == "policy" ]]; then
    checkpoint_args=(
      --checkpoint "$CHECKPOINT_ROOT/formal_s${checkpoint}/final.pt"
      --action-mode policy
    )
  else
    checkpoint_args=(--action-mode zero-residual)
  fi

  local rich_args=()
  if [[ "$RICH_TRACE" == "1" ]]; then
    rich_args=(--rich-trace)
  fi

  "$PY" "$EVALUATOR" \
    --config "$TRAINING_CONFIG" \
    --environment-config "$env_config" \
    "${checkpoint_args[@]}" \
    "${rich_args[@]}" \
    --output "$output" \
    --trace-output "$output" \
    --seed-start "$SEED_START" \
    --seed-count "$SEED_COUNT" \
    --device cpu \
    --dtype float64 \
    --eval-goal-radius-m "$candidate_radius_m" \
    --eval-goal-plane-miss-guard-radius-m "$guard_radius_m" \
    --eval-goal-plane-miss-dwell-steps "$DWELL_STEPS" \
    >"$OUT_ROOT/logs/${name}.log" 2>&1
  echo "DONE $name"
}

launch() {
  run_job "$@" &
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
}

for scenario in $SCENARIOS; do
  if [[ "$scenario" == "default" ]]; then
    env_config="$ENV_CONFIG_DEFAULT"
    radius="2.00"
  elif [[ "$scenario" == "strict125" ]]; then
    env_config="$ENV_CONFIG_STRICT125"
    radius="1.25"
  else
    echo "unknown scenario: $scenario" >&2
    exit 2
  fi
  for checkpoint in $CHECKPOINTS; do
    launch "$scenario" "$checkpoint" "policy" "$env_config" "$radius" "$radius"
  done
  if [[ "$INCLUDE_ZERO" == "1" ]]; then
    launch "$scenario" "none" "zero-residual" "$env_config" "$radius" "$radius"
  fi
done
wait
echo "ALL DONE: $OUT_ROOT"
