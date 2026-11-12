#!/usr/bin/env bash
# Goal150: current07 K4-based PPO/SAC residual screen.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal150_current07_k4_residual_20261111}
GPUS_TRAIN=${GPUS_TRAIN:-"0 3 4 6"}
SEED=${SEED:-86201}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
PARALLEL_EVAL=${PARALLEL_EVAL:-20}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}
ARMS=${ARMS:-"ppo_ent1e4 ppo_ent1e3 ppo_ent1e2 sac"}
LEVELS=${LEVELS:-"current07 current05 current10"}

ENVIRONMENT_CONFIG=${ENVIRONMENT_CONFIG:-configs/lowlevel/goal150/env_current07_k4.json}
PROFILE_MANIFEST=${PROFILE_MANIFEST:-configs/lowlevel/goal150/profile_manifest.json}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/manifests" "$ROOT/train"

read -r -a GPU_ARRAY <<<"$GPUS_TRAIN"
read -r -a ARM_ARRAY <<<"$ARMS"
if [[ ${#GPU_ARRAY[@]} -lt ${#ARM_ARRAY[@]} ]]; then
  echo "GPUS_TRAIN has ${#GPU_ARRAY[@]} entries but ARMS has ${#ARM_ARRAY[@]}" >&2
  exit 2
fi

ppo_config() {
  printf 'configs/lowlevel/goal150/%s.json' "$1"
}

train_ppo() {
  local arm="$1" gpu="$2"
  local config
  config=$(ppo_config "$arm")
  local output="$ROOT/train/$arm/S_s${SEED}"
  local log="$ROOT/logs/train_${arm}_S_s${SEED}.log"
  if [[ -f "$output/final.pt" && -f "$output/training_manifest.json" ]]; then
    echo "SKIP train/$arm (final checkpoint exists)"
    return 0
  fi
  echo "START train/$arm gpu=$gpu config=$config"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/train_batch_lowlevel_ppo.py \
    --training-config "$config" \
    --output "$output" \
    --device cuda \
    --curriculum-eval-interval 2000000 \
    --curriculum-eval-seed-count "$SEED_COUNT" \
    --curriculum-eval-device cuda \
    --curriculum-disable-direct-command \
    >"$log" 2>&1
  echo "DONE train/$arm"
}

train_sac() {
  local arm="$1" gpu="$2"
  local config="configs/lowlevel/goal150/sac_current07_k4.json"
  local output="$ROOT/train/$arm/S_s${SEED}"
  local log="$ROOT/logs/train_${arm}_S_s${SEED}.log"
  if [[ -f "$output/final.zip" && -f "$output/training_manifest.json" ]]; then
    echo "SKIP train/$arm (final checkpoint exists)"
    return 0
  fi
  echo "START train/$arm gpu=$gpu config=$config"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/train_pc_command_residual_sac.py \
    --config "$config" \
    --output "$output" \
    --device cuda \
    --dtype float64 \
    --curriculum-eval-interval 2000000 \
    --curriculum-eval-seed-count "$SEED_COUNT" \
    --curriculum-eval-device cuda \
    --curriculum-disable-direct-command \
    >"$log" 2>&1
  echo "DONE train/$arm"
}

train_one() {
  local arm="$1" gpu="$2"
  case "$arm" in
    ppo_ent1e4|ppo_ent1e3|ppo_ent1e2)
      train_ppo "$arm" "$gpu"
      ;;
    sac)
      train_sac "$arm" "$gpu"
      ;;
    *)
      echo "unknown arm: $arm" >&2
      return 1
      ;;
  esac
}

eval_args_for_level() {
  local level="$1"
  case "$level" in
    current05)
      printf '%s\n' "--curriculum-current-speed-range-mps 0.25 0.45 --curriculum-shear-scale 0.5"
      ;;
    current07)
      printf '%s\n' "--curriculum-current-speed-range-mps 0.275 0.50 --curriculum-shear-scale 0.75"
      ;;
    current10)
      printf '%s\n' "--curriculum-current-speed-range-mps 0.30 0.55 --curriculum-shear-scale 1.0"
      ;;
    *)
      echo "unknown level: $level" >&2
      return 1
      ;;
  esac
}

run_eval() {
  local arm="$1" level="$2" mode="$3" index="$4"
  local output="$ROOT/$arm/$level/$mode"
  local log="$ROOT/logs/eval_${arm}_${level}_${mode}.log"
  local config checkpoint action_mode
  local level_args
  read -r -a level_args <<<"$(eval_args_for_level "$level")"
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP eval/$arm/$level/$mode (summary exists)"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"

  if [[ "$arm" == "sac" ]]; then
    config="configs/lowlevel/goal150/sac_current07_k4.json"
    checkpoint="$ROOT/train/$arm/S_s${SEED}/final.zip"
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      return 1
    fi
    if [[ "$mode" == "zero" ]]; then
      action_mode="zero-residual"
    else
      action_mode="policy"
    fi
    echo "START eval/$arm/$level/$mode"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
      PYTHONPATH="src:$SB3" \
      "$PY" scripts/evaluate_pc_sac.py \
      --config "$config" \
      --environment-config "$ENVIRONMENT_CONFIG" \
      --checkpoint "$checkpoint" \
      --output "$output" \
      --curriculum-stage C1 \
      --disable-curriculum-direct-command \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
      --device cpu \
      --dtype float64 \
      --action-mode "$action_mode" \
      --residual-scale 1.0 \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --rich-trace \
      "${level_args[@]}" \
      >"$log" 2>&1
  else
    config="configs/lowlevel/goal150/${arm}.json"
    checkpoint="$ROOT/train/$arm/S_s${SEED}/final.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      return 1
    fi
    if [[ "$mode" == "zero" ]]; then
      action_mode="zero-command-residual"
    else
      action_mode="policy"
    fi
    echo "START eval/$arm/$level/$mode"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
      PYTHONPATH="src:$SB3" \
      "$PY" scripts/evaluate_pc_ppo.py \
      --training-config "$config" \
      --checkpoint "$checkpoint" \
      --environment-config "$ENVIRONMENT_CONFIG" \
      --curriculum-stage C1 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
      --device cpu \
      --dtype float64 \
      --action-mode "$action_mode" \
      --trace-output "$output/trace.npz" \
      --trace-reward-breakdown \
      "${level_args[@]}" \
      >"$log" 2>&1
  fi
  echo "DONE eval/$arm/$level/$mode"
}

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'seed=%s\n' "$SEED"
  printf 'gpus_train=%s\n' "$GPUS_TRAIN"
  printf 'arms=%s\n' "$ARMS"
  sha256sum "$ENVIRONMENT_CONFIG" "$PROFILE_MANIFEST" \
    configs/lowlevel/goal150/ppo_ent1e4.json \
    configs/lowlevel/goal150/ppo_ent1e3.json \
    configs/lowlevel/goal150/ppo_ent1e2.json \
    configs/lowlevel/goal150/sac_current07_k4.json
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/manifests/run_manifest.txt" 2>&1
cp "$PROFILE_MANIFEST" "$ROOT/manifests/"

if [[ "$TRAIN" == "1" ]]; then
  pids=()
  for index in "${!ARM_ARRAY[@]}"; do
    arm="${ARM_ARRAY[$index]}"
    gpu="${GPU_ARRAY[$index]}"
    train_one "$arm" "$gpu" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if [[ "$status" -ne 0 ]]; then
    echo "one or more Goal150 training arms failed; inspect $ROOT/logs" >&2
    exit 1
  fi
fi

if [[ "$EVAL" == "1" ]]; then
  JOB_FILE=$(mktemp)
  trap 'rm -f "$JOB_FILE"' EXIT
  job_index=0
  for arm in "${ARM_ARRAY[@]}"; do
    for level in $LEVELS; do
      for mode in policy zero; do
        echo "$arm $level $mode $job_index" >>"$JOB_FILE"
        job_index=$((job_index + 1))
      done
    done
  done

  export PY SB3 REPO ROOT SEED SEED_START SEED_COUNT ENVIRONMENT_CONFIG
  export -f eval_args_for_level run_eval
  echo "launching $(wc -l <"$JOB_FILE") Goal150 evaluations with PARALLEL_EVAL=$PARALLEL_EVAL"
  xargs -r -n 4 -P "$PARALLEL_EVAL" bash -c 'run_eval "$@"' _ <"$JOB_FILE"

  missing=0
  while read -r arm level mode index; do
    if [[ ! -f "$ROOT/$arm/$level/$mode/summary.json" ]]; then
      echo "MISSING eval/$arm/$level/$mode" >&2
      missing=$((missing + 1))
    fi
  done <"$JOB_FILE"
  if [[ "$missing" -ne 0 ]]; then
    echo "$missing evaluations are missing; inspect $ROOT/logs" >&2
    exit 1
  fi
  PYTHONPATH="src" "$PY" \
    scripts/analyze_pc_goal150_current07_screen.py --root "$ROOT" || true
fi

printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/manifests/run_manifest.txt"
echo "Goal150 current07 residual screen complete: $ROOT"
