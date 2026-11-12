#!/usr/bin/env bash
# Goal161 Phase A: eval-only command residual authority confirmation.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal161_command_authority_20261112/phaseA}
TRAIN_ROOT=${TRAIN_ROOT:-$REPO/artifacts/lowlevel/goal160_action_parameterization_20261112/train}
ARMS=${ARMS:-"B_command_near10_2 C_command_terminal_train"}
SCALES=${SCALES:-"0.25 0.5"}
SEEDS=${SEEDS:-"86201 86202 86203"}
LEVELS=${LEVELS:-"nominal current07 current10 current15"}
RADII=${RADII:-"default:2.0 strict125:1.25"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PARALLEL=${PARALLEL:-8}

cd "$REPO"
mkdir -p "$ROOT/logs"

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
index=0
for arm in $ARMS; do
  for scale in $SCALES; do
    for radius in $RADII; do
      for level in $LEVELS; do
        for seed in $SEEDS; do
          echo "$arm $scale ${radius%%:*} ${radius##*:} $level $seed $index" >>"$JOB_FILE"
          index=$((index + 1))
        done
      done
    done
  done
done

run_one() {
  local arm="$1" scale="$2" radius_label="$3" radius_value="$4" level="$5" seed="$6" index="$7"
  local output="$ROOT/$arm/scale_${scale}/$radius_label/$level/seed${seed}"
  local log="$ROOT/logs/${arm}_scale${scale}_${radius_label}_${level}_seed${seed}.log"
  local gpu_array=() current_args=() shear_args=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
  case "$level" in
    nominal) ;;
    current07) current_args=(--curriculum-current-speed-range-mps 0.275 0.50); shear_args=(--curriculum-shear-scale 0.75) ;;
    current10) current_args=(--curriculum-current-speed-range-mps 0.30 0.55); shear_args=(--curriculum-shear-scale 1.0) ;;
    current15) current_args=(--curriculum-current-speed-range-mps 0.375 0.675); shear_args=(--curriculum-shear-scale 1.5) ;;
    *) echo "unknown level: $level" >&2; return 2 ;;
  esac
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $arm scale=$scale $radius_label/$level seed=$seed"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START $arm scale=$scale $radius_label/$level seed=$seed gpu=$gpu"
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
      --eval-goal-radius-m "$radius_value" \
      --device cuda \
      --dtype float64 \
      --action-mode policy \
      --residual-scale "$scale" \
      --learner-terminal-gate-mode none \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --checkpoint "$TRAIN_ROOT/$arm/seed$seed/final.zip" \
      "${current_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1; then
    echo "FAILED $arm scale=$scale $radius_label/$level seed=$seed (see $log)" >&2
    return 1
  fi
  echo "DONE $arm scale=$scale $radius_label/$level seed=$seed"
}

export PY REPO SB3 ROOT TRAIN_ROOT GPUS
export -f run_one

echo "launching $(wc -l <"$JOB_FILE") Goal161 Phase A evals with PARALLEL=$PARALLEL"
xargs -r -n 7 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE" || true

missing=0
while read -r arm scale radius_label radius_value level seed index; do
  if [[ ! -f "$ROOT/$arm/scale_${scale}/$radius_label/$level/seed${seed}/summary.json" ]]; then
    echo "MISSING $arm scale=$scale $radius_label/$level seed=$seed" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal161 Phase A evals missing" >&2
  exit 1
fi
echo "Goal161 Phase A complete: $ROOT"
