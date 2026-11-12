#!/usr/bin/env bash
# Goal155: train near10_2 / mid_only / far_off_only residual-authority arms
# (3 seeds x 1M), then evaluate against zero and the frozen Goal86 full control.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal155_authority_training_screen_20261111}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal155_authority_training_screen_20261111}"
FULL_ROOT="${FULL_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/goal86/screen}"
TRAIN_GPUS="${TRAIN_GPUS:-0 1 2 3 4 5 6 7}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2 3 4 5 6 7}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
ARMS="${ARMS:-near10_2 mid_only far_off_only}"
SEEDS="${SEEDS:-86201 86202 86203}"
TOTAL_ENV_TRANSITIONS="${TOTAL_ENV_TRANSITIONS:-1000000}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"

cd "$REPO_ROOT"
mkdir -p "$ROOT" "$LOG_ROOT"
read -r -a TRAIN_GPU_ARRAY <<<"$TRAIN_GPUS"
read -r -a EVAL_GPU_ARRAY <<<"$EVAL_GPUS"
read -r -a ARM_ARRAY <<<"$ARMS"
read -r -a SEED_ARRAY <<<"$SEEDS"

if [[ ! -f configs/lowlevel/goal155/manifest.json ]]; then
  "$PY" scripts/build_goal155_authority_training_configs.py >/dev/null
fi

train_pids=()
train_names=()
train_index=0
wait_one_train() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one training job failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

for arm in "${ARM_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    config="configs/lowlevel/goal155/${arm}_seed${seed}.json"
    out="$ROOT/train/${arm}_seed${seed}"
    log="$LOG_ROOT/train_${arm}_seed${seed}.log"
    mkdir -p "$out"
    if [[ -f "$out/final.zip" ]]; then
      echo "SKIP train ${arm}_seed${seed}"
      train_index=$((train_index + 1))
      continue
    fi
    gpu="${TRAIN_GPU_ARRAY[$((train_index % ${#TRAIN_GPU_ARRAY[@]}))]}"
    echo "START train ${arm}_seed${seed} gpu=$gpu"
    (
      export PYTHONPATH="src:$SB3:$GLOBAL_SRC"
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_pc_command_residual_sac.py \
        --config "$config" --output "$out" \
        --total-env-transitions "$TOTAL_ENV_TRANSITIONS" \
        --device cuda --engineering-smoke --overwrite
      echo $? > "$out/exitcode"
    ) >"$log" 2>&1 &
    train_pids+=("$!")
    train_names+=("${arm}_seed${seed}")
    train_index=$((train_index + 1))
    if [[ ${#train_pids[@]} -ge $MAX_PARALLEL ]]; then
      wait_one_train
      train_pids=("${train_pids[@]:1}")
      train_names=("${train_names[@]:1}")
    fi
  done
done

for pid in "${train_pids[@]}"; do
  wait "$pid" || { echo "one training job failed; inspect $LOG_ROOT" >&2; exit 1; }
done

for arm in "${ARM_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    out="$ROOT/train/${arm}_seed${seed}"
    if [[ ! -f "$out/final.zip" ]]; then
      echo "missing checkpoint: $out/final.zip" >&2
      exit 1
    fi
  done
done
echo "training complete: $ROOT/train"

eval_pids=()
eval_names=()
eval_index=0
wait_one_eval() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one eval job failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

launch_eval() {
  local arm="$1" seed_tag="$2" scenario="$3" config="$4" env_config="$5" checkpoint="$6" mode="$7" gate_mode="$8"
  local out="$ROOT/eval/${arm}_${seed_tag}/${scenario}"
  local gpu="${EVAL_GPU_ARRAY[$((eval_index % ${#EVAL_GPU_ARRAY[@]}))]}"
  local args=(
    "$PY" scripts/evaluate_pc_sac.py
    --config "$config"
    --environment-config "$env_config"
    --output "$out"
    --trace-output "$out"
    --trace-reward-breakdown
    --rich-trace
    --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT"
    --success-frame both
    --device cuda --dtype float64
    --action-mode "$mode"
    --residual-scale 1.0
    --learner-terminal-gate-mode "$gate_mode"
    --sb3-root "$SB3"
    --overwrite
  )
  if [[ "$mode" == "policy" ]]; then
    args+=(--checkpoint "$checkpoint")
  fi
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP eval ${arm}_${seed_tag}/${scenario}"
    eval_index=$((eval_index + 1))
    return
  fi
  echo "START eval ${arm}_${seed_tag}/${scenario} gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3:$GLOBAL_SRC" \
    "${args[@]}" >"$LOG_ROOT/eval_${arm}_${seed_tag}_${scenario}.log" 2>&1 &
  eval_pids+=("$!")
  eval_names+=("${arm}_${seed_tag}/${scenario}")
  eval_index=$((eval_index + 1))
  if [[ ${#eval_pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_one_eval
    eval_pids=("${eval_pids[@]:1}")
    eval_names=("${eval_names[@]:1}")
  fi
}

launch_zero_eval() {
  local scenario="$1" env_config="$2"
  local out="$ROOT/eval/zero/${scenario}"
  local gpu="${EVAL_GPU_ARRAY[$((eval_index % ${#EVAL_GPU_ARRAY[@]}))]}"
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP eval zero_${scenario}"
    eval_index=$((eval_index + 1))
    return
  fi
  echo "START eval zero_${scenario} gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3:$GLOBAL_SRC" \
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
      --overwrite >"$LOG_ROOT/eval_zero_${scenario}.log" 2>&1 &
  eval_pids+=("$!")
  eval_names+=("zero_${scenario}")
  eval_index=$((eval_index + 1))
  if [[ ${#eval_pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_one_eval
    eval_pids=("${eval_pids[@]:1}")
    eval_names=("${eval_names[@]:1}")
  fi
}

for scenario in default strict125; do
  if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
  launch_zero_eval "$scenario" "$env_config"
done

for arm in "${ARM_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    if [[ "$seed" == "86201" ]]; then seed_tag="$seed"; else seed_tag="$seed"; fi
    config="configs/lowlevel/goal155/${arm}_seed${seed}.json"
    gate_mode="$("$PY" - <<PY
import json
print(json.load(open("$config"))["learner_terminal_gate_mode"])
PY
)"
    checkpoint="$ROOT/train/${arm}_seed${seed}/final.zip"
    for scenario in default strict125; do
      if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
      launch_eval "$arm" "$seed_tag" "$scenario" "$config" "$env_config" "$checkpoint" policy "$gate_mode"
    done
  done
done

# Reuse the frozen Goal86 full checkpoints under the same evaluator and seeds.
for seed in "${SEED_ARRAY[@]}"; do
  if [[ "$seed" == "86201" ]]; then full_dir="t2_sac"; else full_dir="t2_sac_seed${seed}"; fi
  checkpoint="$FULL_ROOT/$full_dir/final.zip"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing Goal86 full checkpoint: $checkpoint" >&2
    exit 1
  fi
  for scenario in default strict125; do
    if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
    launch_eval full "$seed" "$scenario" \
      configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json \
      "$env_config" "$checkpoint" policy none
  done
done

for pid in "${eval_pids[@]}"; do
  wait "$pid" || { echo "one eval job failed; inspect $LOG_ROOT" >&2; exit 1; }
done

"$PY" scripts/analyze_pc_goal155_authority_training.py --root "$ROOT"
echo "Goal155 screen complete: $ROOT"
