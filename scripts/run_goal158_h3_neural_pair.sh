#!/usr/bin/env bash
# Goal158: pair Goal86 1M near10_2 residuals with the H3 strong baseline.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
GOAL86_CKPT_ROOT=${GOAL86_CKPT_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/goal86/screen}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal158_h3_neural_pair_20261112}
PARALLEL=${PARALLEL:-6}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"3 4 6 7"}
LEVELS=${LEVELS:-"nominal current07 current10 current15"}
POLICY_CONFIG=${POLICY_CONFIG:-configs/lowlevel/goal154/authority/near10_2.json}
ENV_CONFIG=${ENV_CONFIG:-configs/lowlevel/goal157/env_h3.json}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/manifests"
for seed in 86201 86202 86203; do
  if [[ "$seed" == 86201 ]]; then
    ckpt="$GOAL86_CKPT_ROOT/t2_sac/final.zip"
  else
    ckpt="$GOAL86_CKPT_ROOT/t2_sac_seed${seed}/final.zip"
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "missing checkpoint: $ckpt" >&2
    exit 2
  fi
done

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  sha256sum "$POLICY_CONFIG" "$ENV_CONFIG" \
    "$GOAL86_CKPT_ROOT/t2_sac/final.zip" \
    "$GOAL86_CKPT_ROOT/t2_sac_seed86202/final.zip" \
    "$GOAL86_CKPT_ROOT/t2_sac_seed86203/final.zip"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/manifests/run_manifest.txt" 2>&1

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_index=0
for level in $LEVELS; do
  for seed in 86201 86202 86203; do
    for mode in zero policy; do
      echo "$level $seed $mode $job_index" >>"$JOB_FILE"
      job_index=$((job_index + 1))
    done
  done
done

run_one() {
  local level="$1"
  local seed="$2"
  local mode="$3"
  local index="$4"
  local output="$ROOT/$level/${mode}_seed${seed}"
  local log="$ROOT/logs/${level}_${mode}_seed${seed}.log"
  local current_speed_args=()
  local shear_args=()
  local checkpoint_args=()
  local gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
  local ckpt=""
  if [[ "$seed" == 86201 ]]; then
    ckpt="$GOAL86_CKPT_ROOT/t2_sac/final.zip"
  else
    ckpt="$GOAL86_CKPT_ROOT/t2_sac_seed${seed}/final.zip"
  fi

  case "$level" in
    nominal) ;;
    current07)
      current_speed_args=(--curriculum-current-speed-range-mps 0.275 0.50)
      shear_args=(--curriculum-shear-scale 0.75)
      ;;
    current10)
      current_speed_args=(--curriculum-current-speed-range-mps 0.30 0.55)
      shear_args=(--curriculum-shear-scale 1.0)
      ;;
    current15)
      current_speed_args=(--curriculum-current-speed-range-mps 0.375 0.675)
      shear_args=(--curriculum-shear-scale 1.5)
      ;;
    *)
      echo "unknown level: $level" >&2
      return 2
      ;;
  esac

  if [[ "$mode" == "policy" ]]; then
    checkpoint_args=(--checkpoint "$ckpt")
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $level/$mode/seed$seed (summary exists)"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START level=$level mode=$mode seed=$seed gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$POLICY_CONFIG" \
      --environment-config "$ENV_CONFIG" \
      --curriculum-stage C1 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
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
      "${current_speed_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1
  echo "DONE level=$level mode=$mode seed=$seed"
}

export PY REPO ROOT SB3 GOAL86_CKPT_ROOT SEED_START SEED_COUNT DEVICE GPUS
export POLICY_CONFIG ENV_CONFIG
export -f run_one

echo "launching $(wc -l <"$JOB_FILE") Goal158 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 4 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
while read -r level seed mode index; do
  if [[ ! -f "$ROOT/$level/${mode}_seed${seed}/summary.json" ]]; then
    echo "MISSING $level/$mode/seed$seed" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"

printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing evaluations are missing; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal158 H3 neural pair complete: $ROOT"
