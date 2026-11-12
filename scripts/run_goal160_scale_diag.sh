#!/usr/bin/env bash
# Goal160 post-Gate diagnostic: eval-only command residual scale sweep.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal160_action_parameterization_20261112/diag_scale_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$REPO/artifacts/lowlevel/goal160_action_parameterization_20261112/train}
ARMS=${ARMS:-"B_command_near10_2 C_command_terminal_train"}
SCALES=${SCALES:-"0.25 0.5"}
SEEDS=${SEEDS:-"86201 86202 86203"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PARALLEL=${PARALLEL:-8}

cd "$REPO"
mkdir -p "$ROOT/logs"

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
index=0
for arm in $ARMS; do
  for scale in $SCALES; do
    for seed in $SEEDS; do
      echo "$arm $scale $seed $index" >>"$JOB_FILE"
      index=$((index + 1))
    done
  done
done

run_one() {
  local arm="$1" scale="$2" seed="$3" index="$4"
  local output="$ROOT/$arm/scale_${scale}/seed${seed}"
  local log="$ROOT/logs/${arm}_scale${scale}_seed${seed}.log"
  local gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $arm scale=$scale seed=$seed"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START $arm scale=$scale seed=$seed gpu=$gpu"
  if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config configs/lowlevel/goal160/eval_command_near10_2.json \
      --environment-config configs/lowlevel/goal160/env_h3_command_near10_2_eval.json \
      --curriculum-stage C3 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start 20 \
      --seed-count 40 \
      --success-frame both \
      --eval-goal-radius-m 2.0 \
      --device cuda \
      --dtype float64 \
      --action-mode policy \
      --residual-scale "$scale" \
      --learner-terminal-gate-mode none \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --checkpoint "$TRAIN_ROOT/$arm/seed$seed/final.zip" \
      >"$log" 2>&1; then
    echo "FAILED $arm scale=$scale seed=$seed (see $log)" >&2
    return 1
  fi
  echo "DONE $arm scale=$scale seed=$seed"
}

export PY REPO SB3 ROOT TRAIN_ROOT GPUS
export -f run_one

echo "launching $(wc -l <"$JOB_FILE") Goal160 scale diagnostics with PARALLEL=$PARALLEL"
xargs -r -n 4 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE" || true

missing=0
while read -r arm scale seed index; do
  if [[ ! -f "$ROOT/$arm/scale_${scale}/seed${seed}/summary.json" ]]; then
    echo "MISSING $arm scale=$scale seed=$seed" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing scale diagnostics missing" >&2
  exit 1
fi
echo "Goal160 scale diagnostics complete: $ROOT"
