#!/usr/bin/env bash
# Goal155: CPU/float64 formal re-evaluation of the frozen main screen.
#
# The training runner's automatic eval is a CUDA/float64 diagnostic. The
# pre-registered Goal155 protocol is CPU/float64, so this script writes to a
# separate eval_cpu root and must be used for the main Gate verdict.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal155_authority_training_screen_20261111}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/eval_cpu}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal155_authority_training_screen_20261111/eval_cpu}"
FULL_ROOT="${FULL_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/goal86/screen}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SEEDS="${SEEDS:-86201 86202 86203}"
ARMS="${ARMS:-near10_2 mid_only far_off_only full}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"

cd "$REPO_ROOT"
mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
read -r -a SEED_ARRAY <<<"$SEEDS"
read -r -a ARM_ARRAY <<<"$ARMS"
pids=()
job_index=0

wait_one() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one CPU eval job failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

queue() {
  local log="$1"
  shift
  echo "START $(basename "$log" .log)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3:$GLOBAL_SRC" \
    "$@" >"$log" 2>&1 &
  pids+=("$!")
  job_index=$((job_index + 1))
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_one
    pids=("${pids[@]:1}")
  fi
}

launch_zero() {
  local scenario="$1" env_config="$2"
  local out="$EVAL_ROOT/zero/$scenario"
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP zero/$scenario"
    return
  fi
  queue "$LOG_ROOT/zero_${scenario}.log" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config configs/lowlevel/goal155/near10_2_seed86201.json \
      --environment-config "$env_config" \
      --output "$out" \
      --trace-output "$out" \
      --trace-reward-breakdown \
      --rich-trace \
      --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
      --success-frame both \
      --device cpu --dtype float64 \
      --action-mode zero-residual \
      --residual-scale 1.0 \
      --sb3-root "$SB3" \
      --overwrite
}

launch_arm() {
  local arm="$1" seed="$2" scenario="$3" env_config="$4"
  local config checkpoint gate_mode out
  config="configs/lowlevel/goal155/${arm}_seed${seed}.json"
  out="$EVAL_ROOT/${arm}_${seed}/$scenario"
  if [[ "$arm" == "full" ]]; then
    if [[ "$seed" == "86201" ]]; then
      checkpoint="$FULL_ROOT/t2_sac/final.zip"
    else
      checkpoint="$FULL_ROOT/t2_sac_seed${seed}/final.zip"
    fi
    config="configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json"
    gate_mode="none"
  else
    checkpoint="$ROOT/train/${arm}_seed${seed}/final.zip"
    gate_mode="$("$PY" - <<PY
import json
print(json.load(open("$config"))["learner_terminal_gate_mode"])
PY
)"
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP ${arm}_${seed}/$scenario"
    return
  fi
  queue "$LOG_ROOT/${arm}_${seed}_${scenario}.log" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$config" \
      --environment-config "$env_config" \
      --output "$out" \
      --trace-output "$out" \
      --trace-reward-breakdown \
      --rich-trace \
      --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
      --success-frame both \
      --device cpu --dtype float64 \
      --action-mode policy \
      --residual-scale 1.0 \
      --checkpoint "$checkpoint" \
      --learner-terminal-gate-mode "$gate_mode" \
      --sb3-root "$SB3" \
      --overwrite
}

for scenario in default strict125; do
  if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
  launch_zero "$scenario" "$env_config"
done

for arm in "${ARM_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    for scenario in default strict125; do
      if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
      launch_arm "$arm" "$seed" "$scenario" "$env_config"
    done
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || { echo "one CPU eval job failed; inspect $LOG_ROOT" >&2; exit 1; }
done

"$PY" scripts/analyze_pc_goal155_authority_training.py \
  --root "$ROOT" --eval-root "$EVAL_ROOT"
echo "Goal155 CPU/float64 eval complete: $EVAL_ROOT"
