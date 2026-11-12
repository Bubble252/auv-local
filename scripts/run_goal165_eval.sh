#!/usr/bin/env bash
# Goal165 evaluation: 3 policy seeds + paired zero, 4 levels x 2 radii.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal165_controller_feedback_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/eval}
ARM=${ARM:-G1_gain_near10_2}
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

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_count=0
first_seed=${SEEDS%% *}
for radius in $RADII; do
  for level in $LEVELS; do
    for seed in $SEEDS; do
      for mode in $MODES; do
        if [[ "$mode" == "zero" && "$seed" != "$first_seed" ]]; then
          continue
        fi
        echo "${radius%%:*} ${radius##*:} $level $seed $mode $job_count" >>"$JOB_FILE"
        job_count=$((job_count + 1))
      done
    done
  done
done

run_one() {
  local radius_label=$1 radius_value=$2 level=$3 seed=$4 mode=$5 index=$6
  local output="$EVAL_ROOT/$ARM/$radius_label/$level/${mode}_seed${seed}"
  local log="$EVAL_ROOT/logs/${ARM}_${radius_label}_${level}_${mode}_seed${seed}.log"
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
    checkpoint_args=(--checkpoint "$TRAIN_ROOT/$ARM/seed$seed/final.zip")
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $ARM $radius_label $level $mode seed$seed"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START radius=$radius_label level=$level mode=$mode seed=$seed gpu=$gpu"
  if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config configs/lowlevel/goal165/eval_g1_gain_near10_2.json \
      --environment-config configs/lowlevel/goal165/env_h3_gain_near10_2_eval_nu_ground.json \
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
      --rich-trace \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --overwrite \
      "${checkpoint_args[@]}" \
      "${current_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1; then
    echo "FAILED radius=$radius_label level=$level mode=$mode seed=$seed (see $log)" >&2
    return 1
  fi
  echo "DONE radius=$radius_label level=$level mode=$mode seed=$seed"
}

export PY REPO SB3 ROOT TRAIN_ROOT EVAL_ROOT GPUS ARM
export SEED_START SEED_COUNT DEVICE LEVELS RADII MODES
export -f run_one

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'arm=%s\nseeds=%s\nlevels=%s\nradii=%s\nmodes=%s\n' \
    "$ARM" "$SEEDS" "$LEVELS" "$RADII" "$MODES"
  sha256sum scripts/run_goal165_eval.sh \
    configs/lowlevel/goal165/eval_g1_gain_near10_2.json \
    configs/lowlevel/goal165/env_h3_gain_near10_2_eval_nu_ground.json
} >"$EVAL_ROOT/manifests/run_manifest.txt" 2>&1

echo "launching $job_count Goal165 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 6 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
for radius in $RADII; do
  for level in $LEVELS; do
    for seed in $SEEDS; do
      for mode in $MODES; do
        if [[ ! -f "$EVAL_ROOT/$ARM/${radius%%:*}/$level/${mode}_seed${seed}/summary.json" ]]; then
          if [[ "$mode" == "zero" && "$seed" != "$first_seed" ]]; then
            continue
          fi
          echo "MISSING ${radius%%:*} $level $mode seed$seed" >&2
          missing=$((missing + 1))
        fi
      done
    done
  done
done
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$EVAL_ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal165 evaluations missing" >&2
  exit 1
fi
echo "Goal165 evaluation complete: $EVAL_ROOT"
