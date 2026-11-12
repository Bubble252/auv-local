#!/usr/bin/env bash
# Goal155: incremental CUDA/float64 diagnostic eval for checkpoints that exist.
#
# This only fills the training runner's diagnostic eval/ root. It is not the
# formal Gate; the formal Gate must use run_goal155_eval_cpu.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal155_authority_training_screen_20261111}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/eval}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal155_authority_training_screen_20261111/eval_cuda_incremental}"
FULL_ROOT="${FULL_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/goal86/screen}"
GPUS="${GPUS:-1 2 3 4 5 6 7}"
MAX_PARALLEL="${MAX_PARALLEL:-7}"
SEEDS="${SEEDS:-86201 86202 86203}"
ARMS="${ARMS:-full near10_2 mid_only far_off_only}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"

cd "$REPO_ROOT"
mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
read -r -a SEED_ARRAY <<<"$SEEDS"
read -r -a ARM_ARRAY <<<"$ARMS"
read -r -a GPU_ARRAY <<<"$GPUS"
pids=()
job_index=0

wait_one() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one CUDA diagnostic eval job failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

queue() {
  local log="$1"
  shift
  local gpu="${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
  echo "START $(basename "$log" .log) gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
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
      --device cuda --dtype float64 \
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
    if [[ ! -f "$checkpoint" ]]; then
      echo "SKIP missing ${arm}_${seed}"
      return
    fi
    gate_mode="$("$PY" - <<PY
import json
print(json.load(open("$config"))["learner_terminal_gate_mode"])
PY
)"
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "SKIP missing checkpoint: $checkpoint"
    return
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
      --device cuda --dtype float64 \
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
  wait "$pid" || { echo "one CUDA diagnostic eval job failed; inspect $LOG_ROOT" >&2; exit 1; }
done
echo "Goal155 CUDA diagnostic eval complete: $EVAL_ROOT"
