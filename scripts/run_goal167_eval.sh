#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal167_action_parameterization_20261112}
ZERO_EVAL_ROOT=${ZERO_EVAL_ROOT:-$REPO/artifacts/lowlevel/goal165_controller_feedback_20261112/eval}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/eval}
ARMS=${ARMS:-"A_actuator_terminal B_command_terminal"}
SEEDS=${SEEDS:-"86201 86202 86203"}
LEVELS=${LEVELS:-"nominal current07 current10 current15"}
RADII=${RADII:-"default:2.0 strict125:1.25"}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PARALLEL=${PARALLEL:-8}

cd "$REPO"
mkdir -p "$EVAL_ROOT/logs" "$EVAL_ROOT/manifests"

for arm in $ARMS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      src="$ZERO_EVAL_ROOT/G1_gain_near10_2/${radius%%:*}/$level/zero_seed86201"
      dst="$EVAL_ROOT/$arm/${radius%%:*}/$level/zero_seed86201"
      if [[ -d "$src" && ! -e "$dst" ]]; then
        mkdir -p "$(dirname "$dst")"
        ln -s "$src" "$dst"
      fi
    done
  done
done

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_count=0
for arm in $ARMS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      for seed in $SEEDS; do
        echo "$arm ${radius%%:*} ${radius##*:} $level $seed $job_count" >>"$JOB_FILE"
        job_count=$((job_count + 1))
      done
    done
  done
done

run_one() {
  local arm=$1 radius_label=$2 radius_value=$3 level=$4 seed=$5 index=$6
  local output="$EVAL_ROOT/$arm/$radius_label/$level/policy_seed$seed"
  local log="$EVAL_ROOT/logs/${arm}_${radius_label}_${level}_policy_seed${seed}.log"
  local current_args=() shear_args=() gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
  case "$level" in
    nominal) ;;
    current07) current_args=(--curriculum-current-speed-range-mps 0.275 0.50); shear_args=(--curriculum-shear-scale 0.75) ;;
    current10) current_args=(--curriculum-current-speed-range-mps 0.30 0.55); shear_args=(--curriculum-shear-scale 1.0) ;;
    current15) current_args=(--curriculum-current-speed-range-mps 0.375 0.675); shear_args=(--curriculum-shear-scale 1.5) ;;
    *) echo "unknown level: $level" >&2; return 2 ;;
  esac
  local config="configs/lowlevel/goal167/eval_${arm,,}.json"
  local env_config="configs/lowlevel/goal167/env_${arm,,}_eval_nu_ground.json"
  local checkpoint="$TRAIN_ROOT/$arm/seed$seed/final.zip"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    return 1
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $arm $radius_label $level seed$seed"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START arm=$arm radius=$radius_label level=$level seed=$seed gpu=$gpu"
  if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$config" \
      --environment-config "$env_config" \
      --curriculum-stage C3 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
      --eval-goal-radius-m "$radius_value" \
      --device "$DEVICE" \
      --dtype float64 \
      --action-mode policy \
      --checkpoint "$checkpoint" \
      --residual-scale 1.0 \
      --learner-terminal-gate-mode none \
      --rich-trace \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --overwrite \
      "${current_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1; then
    echo "FAILED arm=$arm radius=$radius_label level=$level seed=$seed (see $log)" >&2
    return 1
  fi
  echo "DONE arm=$arm radius=$radius_label level=$level seed=$seed"
}
export PY REPO SB3 ROOT TRAIN_ROOT EVAL_ROOT GPUS
export SEED_START SEED_COUNT DEVICE LEVELS RADII
export -f run_one

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'arms=%s\nseeds=%s\nlevels=%s\nradii=%s\n' "$ARMS" "$SEEDS" "$LEVELS" "$RADII"
  sha256sum scripts/run_goal167_eval.sh \
    configs/lowlevel/goal167/eval_a_actuator_terminal.json \
    configs/lowlevel/goal167/eval_b_command_terminal.json \
    configs/lowlevel/goal167/env_a_actuator_terminal_eval_nu_ground.json \
    configs/lowlevel/goal167/env_b_command_terminal_eval_nu_ground.json
} >"$EVAL_ROOT/manifests/run_manifest.txt" 2>&1

echo "launching $job_count Goal167 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 6 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
for arm in $ARMS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      for seed in $SEEDS; do
        if [[ ! -f "$EVAL_ROOT/$arm/${radius%%:*}/$level/policy_seed$seed/summary.json" ]]; then
          echo "MISSING $arm ${radius%%:*} $level policy seed$seed" >&2
          missing=$((missing + 1))
        fi
      done
    done
  done
done
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$EVAL_ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal167 evaluations missing" >&2
  exit 1
fi
echo "Goal167 evaluation complete: $EVAL_ROOT"
