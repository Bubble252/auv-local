#!/usr/bin/env bash
# Goal143 C0/C1 ALOS control-repair pilot: PPO+SAC x C-full-alos/C-no-current-alos.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-artifacts/lowlevel/goal143_curriculum_alos_20261111}
GPUS=${GPUS:-"0 1 2 3"}
EVAL_SEEDS=${EVAL_SEEDS:-40}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-S_s86201}
ARMS=${ARMS:-"C-full-alos C-no-current-alos"}
ALGS=${ALGS:-"ppo sac"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$ROOT/logs" "$ROOT/ppo" "$ROOT/sac"
read -r -a GPU_ARRAY <<<"$GPUS"
read -r -a ARM_ARRAY <<<"$ARMS"
read -r -a ALG_ARRAY <<<"$ALGS"
ARM_COUNT=${#ARM_ARRAY[@]}
if [[ ${#GPU_ARRAY[@]} -lt $((2 * ARM_COUNT)) ]]; then
  echo "GPUS must contain at least 2 entries per arm (PPO then SAC)" >&2
  exit 1
fi

force_current_args_for() {
  local arm="$1"
  if [[ "$arm" == *"no-current"* ]]; then
    printf '%s\n' "--curriculum-force-current-alpha-zero"
  fi
}

run_ppo() {
  local arm="$1" gpu="$2"
  local config="configs/lowlevel/goal143/${arm}_ppo_pilot.json"
  local output="$ROOT/ppo/$arm/$OUTPUT_SUFFIX"
  local log="$ROOT/logs/ppo_${arm}_${OUTPUT_SUFFIX}.log"
  local force_args=()
  if [[ "$arm" == *"no-current"* ]]; then
    force_args+=(--curriculum-force-current-alpha-zero)
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
    --curriculum-disable-direct-command \
    "${force_args[@]}" \
    >"$log" 2>&1
  echo "DONE ppo/$arm"
}

run_sac() {
  local arm="$1" gpu="$2"
  local config="configs/lowlevel/goal143/${arm}_sac_pilot.json"
  local output="$ROOT/sac/$arm/$OUTPUT_SUFFIX"
  local log="$ROOT/logs/sac_${arm}_${OUTPUT_SUFFIX}.log"
  local force_args=()
  if [[ "$arm" == *"no-current"* ]]; then
    force_args+=(--curriculum-force-current-alpha-zero)
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
    --curriculum-disable-direct-command \
    "${force_args[@]}" \
    >"$log" 2>&1
  echo "DONE sac/$arm"
}

if [[ "$DRY_RUN" == "1" ]]; then
  for index in "${!ARM_ARRAY[@]}"; do
    arm="${ARM_ARRAY[$index]}"
    for alg in "${ALG_ARRAY[@]}"; do
      if [[ "$alg" == "ppo" ]]; then
        run_ppo "$arm" "${GPU_ARRAY[$index]}"
      elif [[ "$alg" == "sac" ]]; then
        run_sac "$arm" "${GPU_ARRAY[$((index + ARM_COUNT))]}"
      else
        echo "unknown ALGS entry: $alg" >&2
        exit 1
      fi
    done
  done
  exit 0
fi

pids=()
for index in "${!ARM_ARRAY[@]}"; do
  arm="${ARM_ARRAY[$index]}"
  for alg in "${ALG_ARRAY[@]}"; do
    if [[ "$alg" == "ppo" ]]; then
      run_ppo "$arm" "${GPU_ARRAY[$index]}" &
      pids+=("$!")
    elif [[ "$alg" == "sac" ]]; then
      run_sac "$arm" "${GPU_ARRAY[$((index + ARM_COUNT))]}" &
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
  echo "one or more Goal143 pilots failed; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal143 curriculum ALOS pilots complete: $ROOT"

