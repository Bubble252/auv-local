#!/usr/bin/env bash
# Goal160 three-arm x three-seed x 1M training queue.
# One job per GPU; a finishing job releases its GPU token for the next job.
set -euo pipefail

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal160_action_parameterization_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}
LOGS=${LOGS:-$ROOT/logs}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
ARMS=${ARMS:-"A_actuator_near10_2 B_command_near10_2 C_command_terminal_train"}
SEEDS=${SEEDS:-"86201 86202 86203"}

cd "$REPO"
mkdir -p "$TRAIN_ROOT" "$LOGS" "$ROOT/gpu_locks"

JOBS=()
for arm in $ARMS; do
  for seed in $SEEDS; do
    if [[ -f "$TRAIN_ROOT/$arm/seed$seed/final.zip" ]]; then
      echo "SKIP $arm seed$seed (final.zip exists)"
      continue
    fi
    JOBS+=("$arm $seed")
  done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "all Goal160 jobs already complete"
  exit 0
fi

run_one() {
  local arm=$1
  local seed=$2
  local log="$LOGS/${arm}_seed${seed}.log"
  local gpu
  while true; do
    for gpu in $GPUS; do
      if flock -n "$ROOT/gpu_locks/gpu_${gpu}.lock" \
        bash scripts/run_goal160_one.sh "$gpu" "$arm" "$seed" >>"$log" 2>&1; then
        return 0
      fi
    done
    sleep 3
  done
}
export -f run_one
export REPO ROOT TRAIN_ROOT LOGS GPUS

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'gpus=%s\n' "$GPUS"
  printf 'arms=%s\n' "$ARMS"
  printf 'seeds=%s\n' "$SEEDS"
  sha256sum scripts/run_goal160_one.sh scripts/run_goal160_train.sh \
    configs/lowlevel/goal160/goal160_manifest.json
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/train_run_manifest.txt" 2>&1

echo "launching ${#JOBS[@]} Goal160 jobs on GPUs: $GPUS"
GPU_COUNT=$(wc -w <<<"$GPUS")
printf '%s\n' "${JOBS[@]}" | xargs -r -n 2 -P "$GPU_COUNT" \
  bash -c 'export GPUS="$1"; shift; run_one "$@"' _ "$GPUS"

missing=0
for arm in $ARMS; do
  for seed in $SEEDS; do
    if [[ ! -f "$TRAIN_ROOT/$arm/seed$seed/final.zip" ]]; then
      echo "MISSING $arm seed$seed" >&2
      missing=$((missing + 1))
    fi
  done
done

printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/train_run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal160 jobs are missing; inspect $LOGS" >&2
  exit 1
fi
echo "Goal160 training complete: $TRAIN_ROOT"
