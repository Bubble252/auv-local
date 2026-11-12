#!/usr/bin/env bash
# Goal161 Phase B evaluation: cap-trained B/C arms, paired zero, 4 levels x 2 radii.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal161_command_authority_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/eval}
ARMS=${ARMS:-"B_command_cap025 C_command_terminal_cap025"}
SEEDS=${SEEDS:-"86201 86202 86203"}
LEVELS=${LEVELS:-"nominal current07 current10 current15"}
RADII=${RADII:-"default:2.0 strict125:1.25"}
MODES=${MODES:-"zero policy"}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PARALLEL=${PARALLEL:-8}

cd "$REPO"
mkdir -p "$EVAL_ROOT/logs" "$EVAL_ROOT/manifests"

env_config() {
  case "$1" in
    B_command_cap025) echo configs/lowlevel/goal161/env_h3_b_command_cap025_eval.json ;;
    C_command_terminal_cap025) echo configs/lowlevel/goal161/env_h3_c_command_terminal_cap025_eval.json ;;
    *) echo "unknown arm: $1" >&2; return 2 ;;
  esac
}

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_count=0
for arm in $ARMS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      for seed in $SEEDS; do
        for mode in $MODES; do
          echo "$arm ${radius%%:*} ${radius##*:} $level $seed $mode $job_count" >>"$JOB_FILE"
          job_count=$((job_count + 1))
        done
      done
    done
  done
done

run_one() {
  local arm=$1 radius_label=$2 radius_value=$3 level=$4 seed=$5 mode=$6 index=$7
  local output="$EVAL_ROOT/$arm/$radius_label/$level/${mode}_seed${seed}"
  local log="$EVAL_ROOT/logs/${arm}_${radius_label}_${level}_${mode}_seed${seed}.log"
  local current_args=() shear_args=() checkpoint_args=() gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}

  case "$level" in
    nominal) ;;
    current07) current_args=(--curriculum-current-speed-range-mps 0.275 0.50); shear_args=(--curriculum-shear-scale 0.75) ;;
    current10) current_args=(--curriculum-current-speed-range-mps 0.30 0.55); shear_args=(--curriculum-shear-scale 1.0) ;;
    current15) current_args=(--curriculum-current-speed-range-mps 0.375 0.675); shear_args=(--curriculum-shear-scale 1.5) ;;
    *) echo "unknown level: $level" >&2; return 2 ;;
  esac
  if [[ "$mode" == "policy" ]]; then
    checkpoint_args=(--checkpoint "$TRAIN_ROOT/$arm/seed$seed/final.zip")
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $arm $radius_label $level $mode seed$seed"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START arm=$arm radius=$radius_label level=$level mode=$mode seed=$seed gpu=$gpu"
  if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config configs/lowlevel/goal161/eval_command_cap025.json \
      --environment-config "$(env_config "$arm")" \
      --curriculum-stage C3 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
      --eval-goal-radius-m "$radius_value" \
      --device "$DEVICE" \
      --dtype float64 \
      --action-mode "$([ "$mode" == "zero" ] && echo zero-residual || echo policy)" \
      --residual-scale 1.0 \
      --learner-terminal-gate-mode none \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --overwrite \
      "${checkpoint_args[@]}" \
      "${current_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1; then
    echo "FAILED arm=$arm radius=$radius_label level=$level mode=$mode seed=$seed (see $log)" >&2
    return 1
  fi
  echo "DONE arm=$arm radius=$radius_label level=$level mode=$mode seed=$seed"
}

export PY REPO SB3 ROOT TRAIN_ROOT EVAL_ROOT GPUS
export SEED_START SEED_COUNT DEVICE LEVELS RADII MODES
export -f run_one env_config

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'arms=%s\nseeds=%s\nlevels=%s\nradii=%s\nmodes=%s\n' \
    "$ARMS" "$SEEDS" "$LEVELS" "$RADII" "$MODES"
  sha256sum scripts/run_goal161_eval.sh configs/lowlevel/goal161/goal161_manifest.json \
    configs/lowlevel/goal161/eval_command_cap025.json \
    configs/lowlevel/goal161/env_h3_b_command_cap025_eval.json \
    configs/lowlevel/goal161/env_h3_c_command_terminal_cap025_eval.json
} >"$EVAL_ROOT/manifests/run_manifest.txt" 2>&1

echo "launching $job_count Goal161 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 7 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
for arm in $ARMS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      for seed in $SEEDS; do
        for mode in $MODES; do
          if [[ ! -f "$EVAL_ROOT/$arm/${radius%%:*}/$level/${mode}_seed${seed}/summary.json" ]]; then
            echo "MISSING $arm ${radius%%:*} $level $mode seed$seed" >&2
            missing=$((missing + 1))
          fi
        done
      done
    done
  done
done
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$EVAL_ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal161 evaluations missing" >&2
  exit 1
fi
echo "Goal161 Phase B evaluation complete: $EVAL_ROOT"
