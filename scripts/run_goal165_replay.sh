#!/usr/bin/env bash
# Goal165: corrected controller-feedback zero replay for the speed-envelope gate.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal165_controller_feedback_20261112}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/replay_zero}
SEEDS=${SEEDS:-"20"}
LEVELS=${LEVELS:-"nominal current07 current10 current15"}
GPUS=${GPUS:-"0 1 2 3"}
PARALLEL=${PARALLEL:-4}

cd "$REPO"
mkdir -p "$EVAL_ROOT/logs" "$EVAL_ROOT/manifests"

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_count=0
for level in $LEVELS; do
  for seed in $SEEDS; do
    echo "$level $seed $job_count" >>"$JOB_FILE"
    job_count=$((job_count + 1))
  done
done

run_one() {
  local level=$1 seed=$2 index=$3
  local output="$EVAL_ROOT/$level/zero_seed${seed}"
  local log="$EVAL_ROOT/logs/${level}_zero_seed${seed}.log"
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
  if [[ -e "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START Goal165 corrected replay level=$level seed_block=$seed gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config configs/lowlevel/goal162/eval_g1_gain_near10_2.json \
      --environment-config configs/lowlevel/goal165/env_h3_gain_near10_2_eval_nu_ground.json \
      --curriculum-stage C3 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start 20 \
      --seed-count 40 \
      --success-frame both \
      --eval-goal-radius-m 2.0 \
      --device cuda \
      --dtype float64 \
      --action-mode zero-residual \
      --residual-scale 1.0 \
      --learner-terminal-gate-mode none \
      --rich-trace \
      --trace-output "$output" \
      --sb3-root "$SB3" \
      --overwrite \
      "${current_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1
  echo "DONE Goal165 corrected replay level=$level seed_block=$seed"
}

export PY REPO SB3 ROOT EVAL_ROOT GPUS
export -f run_one

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\nseeds=%s\nlevels=%s\n' "$REPO" "$SEEDS" "$LEVELS"
  sha256sum scripts/run_goal165_replay.sh \
    configs/lowlevel/goal162/eval_g1_gain_near10_2.json \
    configs/lowlevel/goal165/env_h3_gain_near10_2_eval_nu_ground.json
} >"$EVAL_ROOT/manifests/run_manifest.txt" 2>&1

echo "launching $job_count Goal165 corrected replays with PARALLEL=$PARALLEL"
xargs -r -n 3 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
for level in $LEVELS; do
  for seed in $SEEDS; do
    if [[ ! -f "$EVAL_ROOT/$level/zero_seed${seed}/trace.npz" ]]; then
      echo "MISSING $level seed$seed" >&2
      missing=$((missing + 1))
    fi
  done
done
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$EVAL_ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
    echo "$missing Goal165 corrected replays missing" >&2
  exit 1
fi
echo "Goal165 corrected replay complete: $EVAL_ROOT"
