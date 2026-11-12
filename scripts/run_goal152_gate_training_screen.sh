#!/usr/bin/env bash
# Goal152: 3-seed near10_2 gate training screen + dev40 policy/zero evaluation.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal152_gate_training_screen_20261111}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal152_gate_training_screen_20261111}"
TRAIN_GPUS="${TRAIN_GPUS:-3 4 6}"
EVAL_GPUS="${EVAL_GPUS:-3 4 6 7}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
SEEDS="${SEEDS:-86201 86202 86203}"
TOTAL_ENV_TRANSITIONS="${TOTAL_ENV_TRANSITIONS:-1000000}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"

cd "$REPO_ROOT"
mkdir -p "$ROOT" "$LOG_ROOT"
read -r -a TRAIN_GPU_ARRAY <<<"$TRAIN_GPUS"
read -r -a EVAL_GPU_ARRAY <<<"$EVAL_GPUS"
read -r -a SEED_ARRAY <<<"$SEEDS"

train_pids=()
train_names=()
index=0
for seed in "${SEED_ARRAY[@]}"; do
  config="configs/lowlevel/goal152/near10_2_seed${seed}.json"
  out="$ROOT/train/near10_2_seed${seed}"
  log="$LOG_ROOT/train_seed${seed}.log"
  mkdir -p "$out"
  if [[ -f "$out/final.zip" ]]; then
    echo "SKIP train seed=$seed"
    index=$((index + 1))
    continue
  fi
  gpu="${TRAIN_GPU_ARRAY[$((index % ${#TRAIN_GPU_ARRAY[@]}))]}"
  echo "START train seed=$seed gpu=$gpu"
  (
    export PYTHONPATH="src:$SB3"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_pc_command_residual_sac.py \
      --config "$config" --output "$out" \
      --total-env-transitions "$TOTAL_ENV_TRANSITIONS" \
      --device cuda --engineering-smoke --overwrite
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  train_pids+=("$!")
  train_names+=("seed${seed}")
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
  local mode="$1" seed_tag="$2" config="$3" env_config="$4" scenario="$5" checkpoint="$6"
  local out="$ROOT/eval/$seed_tag/$scenario"
  local gpu="${EVAL_GPU_ARRAY[$((job_index % ${#EVAL_GPU_ARRAY[@]}))]}"
  local extra=()
  if [[ "$mode" == "policy" ]]; then
    extra=(--checkpoint "$checkpoint")
  fi
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP eval $seed_tag/$scenario"
    job_index=$((job_index + 1))
    return
  fi
  echo "START eval $seed_tag/$scenario gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$config" \
      --environment-config "$env_config" \
      --output "$out" \
      --trace-output "$out" \
      --trace-reward-breakdown \
      --rich-trace \
      --seed-start 20 --seed-count 40 \
      --success-frame both \
      --device cuda --dtype float64 \
      --action-mode "$mode" \
      --residual-scale 1.0 \
      --sb3-root "$SB3" \
      --overwrite "${extra[@]}" >"$LOG_ROOT/eval_${seed_tag}_${scenario}.log" 2>&1 &
  eval_pids+=("$!")
  eval_names+=("$seed_tag/$scenario")
  job_index=$((job_index + 1))
  if [[ ${#eval_pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_one
    eval_pids=("${eval_pids[@]:1}")
  fi
}

for scenario in default strict125; do
  if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
  launch_eval zero zero "configs/lowlevel/goal152/near10_2_seed86201.json" "$env_config" "$scenario" ""
  for seed in "${SEED_ARRAY[@]}"; do
    launch_eval policy "near10_2_seed${seed}" "configs/lowlevel/goal152/near10_2_seed${seed}.json" \
      "$env_config" "$scenario" "$ROOT/train/near10_2_seed${seed}/final.zip"
  done
done

for pid in "${eval_pids[@]}"; do
  wait "$pid" || { echo "one eval job failed; inspect $LOG_ROOT" >&2; exit 1; }
done
for tag in "${eval_names[@]}"; do
  if [[ ! -f "$ROOT/eval/${tag}/summary.json" || ! -f "$ROOT/eval/${tag}/episodes.json" ]]; then
    echo "missing output for $tag" >&2
    exit 1
  fi
done

"$PY" scripts/analyze_pc_goal152_gate_training.py --root "$ROOT"
echo "Goal152 screen complete: $ROOT"
