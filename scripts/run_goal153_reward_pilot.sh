#!/usr/bin/env bash
# Goal153 reward pilot: R0/R1/R2, C1 curriculum, actuator full authority.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal153_reward_curriculum_20261111/pilot}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal153_reward_pilot}"
TRAIN_GPUS="${TRAIN_GPUS:-5 7 0}"
EVAL_GPUS="${EVAL_GPUS:-5 7}"
ARMS="${ARMS:-R0 R1 R2}"
SEED="${SEED:-86201}"
TRANSITIONS="${TRANSITIONS:-200000}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"

cd "$REPO_ROOT"
mkdir -p "$ROOT/train" "$ROOT/eval" "$LOG_ROOT"
read -r -a TRAIN_GPU_ARRAY <<<"$TRAIN_GPUS"
read -r -a EVAL_GPU_ARRAY <<<"$EVAL_GPUS"
read -r -a ARM_ARRAY <<<"$ARMS"

train_pids=()
train_names=()
index=0
for arm in "${ARM_ARRAY[@]}"; do
  config="configs/lowlevel/goal153/train/${arm}_C1_sac.json"
  out="$ROOT/train/${arm}_C1"
  log="$LOG_ROOT/train_${arm}.log"
  mkdir -p "$out"
  if [[ -f "$out/final.zip" ]]; then
    echo "SKIP train $arm"
    index=$((index + 1))
    continue
  fi
  gpu="${TRAIN_GPU_ARRAY[$((index % ${#TRAIN_GPU_ARRAY[@]}))]}"
  echo "START train $arm gpu=$gpu"
  (
    export PYTHONPATH="src:$SB3"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_pc_command_residual_sac.py \
      --config "$config" --output "$out" \
      --seed "$SEED" \
      --total-env-transitions "$TRANSITIONS" \
      --device cuda --engineering-smoke --overwrite \
      --curriculum-disable-direct-command
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  train_pids+=("$!")
  train_names+=("$arm")
  index=$((index + 1))
done

fail=0
for i in "${!train_pids[@]}"; do
  if ! wait "${train_pids[$i]}"; then
    echo "TRAIN FAIL ${train_names[$i]}; see $LOG_ROOT" >&2
    fail=1
  else
    echo "TRAIN DONE ${train_names[$i]}"
  fi
done
if [[ "$fail" -ne 0 ]]; then exit 1; fi

eval_pids=()
eval_names=()
job_index=0
wait_one() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one eval job failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

launch_eval() {
  local tag="$1" config="$2" env_config="$3" scenario="$4" mode="$5" checkpoint="$6"
  local out="$ROOT/eval/$tag/$scenario"
  local gpu="${EVAL_GPU_ARRAY[$((job_index % ${#EVAL_GPU_ARRAY[@]}))]}"
  local extra=()
  local curriculum_extra=()
  if [[ "$mode" == "policy" ]]; then
    extra=(--checkpoint "$checkpoint")
  fi
  if [[ "$DISABLE_DIRECT_COMMAND" == "1" ]]; then
    curriculum_extra+=(--disable-curriculum-direct-command)
  fi
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP eval $tag/$scenario"
    job_index=$((job_index + 1))
    return
  fi
  echo "START eval $tag/$scenario gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3:$GLOBAL_SRC" \
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
      --action-mode "$mode" \
      --residual-scale 1.0 \
      --sb3-root "$SB3" \
      --overwrite "${extra[@]}" "${curriculum_extra[@]}" >"$LOG_ROOT/eval_${tag}_${scenario}.log" 2>&1 &
  eval_pids+=("$!")
  eval_names+=("$tag/$scenario")
  job_index=$((job_index + 1))
  if [[ ${#eval_pids[@]} -ge 4 ]]; then
    wait_one
    eval_pids=("${eval_pids[@]:1}")
  fi
}

# One zero-residual control per scenario is enough: it does not use the reward.
for scenario in default strict125; do
  if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
  launch_eval zero "configs/lowlevel/goal153/train/R0_C1_sac.json" "$env_config" "$scenario" zero-residual ""
  for arm in "${ARM_ARRAY[@]}"; do
    launch_eval "$arm" "configs/lowlevel/goal153/train/${arm}_C1_sac.json" "$env_config" "$scenario" policy "$ROOT/train/${arm}_C1/final.zip"
  done
done

for pid in "${eval_pids[@]}"; do
  wait "$pid" || { echo "one eval job failed; inspect $LOG_ROOT" >&2; exit 1; }
done
echo "Goal153 reward pilot complete: $ROOT"
