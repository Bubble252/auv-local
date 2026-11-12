#!/usr/bin/env bash
# Goal161 Phase C: eval-only command-channel masks on the cap-trained C checkpoint.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal161_command_authority_20261112/phaseC_mask}
TRAIN_ROOT=${TRAIN_ROOT:-$REPO/artifacts/lowlevel/goal161_command_authority_20261112/train}
ARM=${ARM:-C_command_terminal_cap025}
SEEDS=${SEEDS:-"86201 86202 86203"}
LEVELS=${LEVELS:-"current10 current15"}
RADII=${RADII:-"default:2.0 strict125:1.25"}
MASKS=${MASKS:-"heading:1,0,0 pitch:0,1,0 speed:0,0,1 fins:1,1,0"}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PARALLEL=${PARALLEL:-8}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/manifests"

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_count=0
for mask_spec in $MASKS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      for seed in $SEEDS; do
        echo "$mask_spec ${radius%%:*} ${radius##*:} $level $seed $job_count" >>"$JOB_FILE"
        job_count=$((job_count + 1))
      done
    done
  done
done

run_one() {
  local mask_spec=$1 radius_label=$2 radius_value=$3 level=$4 seed=$5 index=$6
  local mask_name=${mask_spec%%:*}
  local mask_values=${mask_spec##*:}
  local output="$ROOT/$mask_name/$radius_label/$level/seed$seed"
  local log="$ROOT/logs/${mask_name}_${radius_label}_${level}_seed${seed}.log"
  local current_args=() shear_args=() gpu_array=() mask_check=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
  IFS=',' read -r -a mask_check <<<"$mask_values"
  if [[ ${#mask_check[@]} -ne 3 ]]; then
    echo "mask $mask_name must contain three comma-separated values" >&2
    return 2
  fi

  case "$level" in
    current10) current_args=(--curriculum-current-speed-range-mps 0.30 0.55); shear_args=(--curriculum-shear-scale 1.0) ;;
    current15) current_args=(--curriculum-current-speed-range-mps 0.375 0.675); shear_args=(--curriculum-shear-scale 1.5) ;;
    *) echo "unknown level: $level" >&2; return 2 ;;
  esac
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $mask_name $radius_label $level seed$seed"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START mask=$mask_name radius=$radius_label level=$level seed=$seed gpu=$gpu"
  if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config configs/lowlevel/goal161/eval_command_cap025.json \
      --environment-config configs/lowlevel/goal161/env_h3_c_command_terminal_cap025_eval.json \
      --checkpoint "$TRAIN_ROOT/$ARM/seed$seed/final.zip" \
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
      --residual-scale 1.0 \
      --learner-terminal-gate-mode none \
      --learner-command-channel-mask "$mask_values" \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --overwrite \
      "${current_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1; then
    echo "FAILED mask=$mask_name radius=$radius_label level=$level seed=$seed (see $log)" >&2
    return 1
  fi
  echo "DONE mask=$mask_name radius=$radius_label level=$level seed=$seed"
}

export PY REPO SB3 ROOT TRAIN_ROOT ARM GPUS DEVICE SEED_START SEED_COUNT
export -f run_one

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\narm=%s\nseeds=%s\nlevels=%s\nradii=%s\nmasks=%s\n' \
    "$REPO" "$ARM" "$SEEDS" "$LEVELS" "$RADII" "$MASKS"
  sha256sum scripts/run_goal161_phaseC_mask_eval.sh \
    configs/lowlevel/goal161/eval_command_cap025.json \
    configs/lowlevel/goal161/env_h3_c_command_terminal_cap025_eval.json
} >"$ROOT/manifests/run_manifest.txt" 2>&1

echo "launching $job_count Goal161 Phase C mask evaluations with PARALLEL=$PARALLEL"
xargs -r -n 6 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
while read -r mask_spec radius_label radius_value level seed index; do
  mask_name=${mask_spec%%:*}
  if [[ ! -f "$ROOT/$mask_name/$radius_label/$level/seed$seed/summary.json" ]]; then
    echo "MISSING $mask_name $radius_label $level seed$seed" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal161 Phase C mask evaluations missing" >&2
  exit 1
fi
echo "Goal161 Phase C mask evaluation complete: $ROOT"
