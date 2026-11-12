#!/usr/bin/env bash
# Goal136 C-line 1M pilot: PPO+SAC x C-full/C-hard/C-no-current.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-artifacts/lowlevel/goal136_curriculum_pilots_20261111}
GPUS=${GPUS:-"0 1 2 3 6 7"}
EVAL_SEEDS=${EVAL_SEEDS:-20}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-S_s86201}
ARMS=${ARMS:-"C-full C-hard C-no-current"}
ALGS=${ALGS:-"ppo sac"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$ROOT/logs" "$ROOT/ppo" "$ROOT/sac"
read -r -a GPU_ARRAY <<<"$GPUS"
if [[ ${#GPU_ARRAY[@]} -lt 6 ]]; then
  echo "GPUS must contain at least 6 devices" >&2
  exit 1
fi

run_ppo() {
  local arm="$1" gpu="$2"
  local config="configs/lowlevel/goal136/${arm}_ppo_pilot.json"
  local output="$ROOT/ppo/$arm/$OUTPUT_SUFFIX"
  local log="$ROOT/logs/ppo_${arm}_${OUTPUT_SUFFIX}.log"
  local force_args=()
  if [[ "$arm" == "C-no-current" ]]; then
    force_args=(--curriculum-force-current-alpha-zero)
  fi
  if [[ -f "$output/final.pt" && -f "$output/training_manifest.json" ]]; then
    echo "SKIP ppo/$arm (checkpoint exists)"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY ppo arm=$arm gpu=$gpu config=$config output=$output"
    return 0
  fi
  echo "START ppo/$arm gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/train_batch_lowlevel_ppo.py \
    --training-config "$config" \
    --output "$output" \
    --device cuda \
    --curriculum-eval-interval "$EVAL_INTERVAL" \
    --curriculum-eval-seed-count "$EVAL_SEEDS" \
    --curriculum-eval-device cuda \
    --curriculum-mix-ramp-transitions 100000 \
    "${force_args[@]}" \
    >"$log" 2>&1
  echo "DONE ppo/$arm"
}

run_sac() {
  local arm="$1" gpu="$2"
  local config="configs/lowlevel/goal136/${arm}_sac_pilot.json"
  local output="$ROOT/sac/$arm/$OUTPUT_SUFFIX"
  local log="$ROOT/logs/sac_${arm}_${OUTPUT_SUFFIX}.log"
  local force_args=()
  if [[ "$arm" == "C-no-current" ]]; then
    force_args=(--curriculum-force-current-alpha-zero)
  fi
  if [[ -f "$output/final.zip" && -f "$output/training_manifest.json" ]]; then
    echo "SKIP sac/$arm (checkpoint exists)"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY sac arm=$arm gpu=$gpu config=$config output=$output"
    return 0
  fi
  echo "START sac/$arm gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/train_pc_command_residual_sac.py \
    --config "$config" \
    --output "$output" \
    --device cuda \
    --dtype float64 \
    --curriculum-eval-interval "$EVAL_INTERVAL" \
    --curriculum-eval-seed-count "$EVAL_SEEDS" \
    --curriculum-eval-device cuda \
    --curriculum-mix-ramp-transitions 100000 \
    "${force_args[@]}" \
    >"$log" 2>&1
  echo "DONE sac/$arm"
}

export PY SB3 ROOT EVAL_SEEDS EVAL_INTERVAL OUTPUT_SUFFIX ARMS ALGS DRY_RUN
export -f run_ppo run_sac

read -r -a arm_array <<<"$ARMS"
read -r -a alg_array <<<"$ALGS"
all_arms=("C-full" "C-hard" "C-no-current")

arm_index() {
  local target="$1"
  local index
  for index in 0 1 2; do
    if [[ "${all_arms[$index]}" == "$target" ]]; then
      echo "$index"
      return 0
    fi
  done
  echo "unknown arm: $target" >&2
  return 1
}

if [[ "$DRY_RUN" == "1" ]]; then
  for arm in "${arm_array[@]}"; do
    index=$(arm_index "$arm")
    for alg in "${alg_array[@]}"; do
      if [[ "$alg" == "ppo" ]]; then
        run_ppo "$arm" "${GPU_ARRAY[$index]}"
      elif [[ "$alg" == "sac" ]]; then
        run_sac "$arm" "${GPU_ARRAY[$((index + 3))]}"
      else
        echo "unknown ALGS entry: $alg" >&2
        exit 1
      fi
    done
  done
  exit 0
fi

pids=()
for arm in "${arm_array[@]}"; do
  index=$(arm_index "$arm")
  for alg in "${alg_array[@]}"; do
    if [[ "$alg" == "ppo" ]]; then
      run_ppo "$arm" "${GPU_ARRAY[$index]}" &
      pids+=("$!")
    elif [[ "$alg" == "sac" ]]; then
      run_sac "$arm" "${GPU_ARRAY[$((index + 3))]}" &
      pids+=("$!")
    else
      echo "unknown ALGS entry: $alg" >&2
      exit 1
    fi
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more pilots failed; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal136 curriculum pilots complete: $ROOT"
