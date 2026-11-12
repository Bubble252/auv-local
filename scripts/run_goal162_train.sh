#!/usr/bin/env bash
# Goal162: three seeds x 1M controller-gain residual training.
set -euo pipefail

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal162_controller_gain_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}
LOGS=${LOGS:-$ROOT/logs}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
ARM=${ARM:-G1_gain_near10_2}
SEEDS=${SEEDS:-"86201 86202 86203"}

cd "$REPO"
mkdir -p "$TRAIN_ROOT" "$LOGS" "$ROOT/gpu_locks"

JOBS=()
for seed in $SEEDS; do
  if [[ -f "$TRAIN_ROOT/$ARM/seed$seed/final.zip" ]]; then
    echo "SKIP $ARM seed$seed (final.zip exists)"
    continue
  fi
  JOBS+=("$ARM $seed")
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "all Goal162 jobs already complete"
  exit 0
fi

run_one() {
  local arm=$1 seed=$2 log gpu
  log="$LOGS/${arm}_seed${seed}.log"
  while true; do
    for gpu in $GPUS; do
      if flock -n "$ROOT/gpu_locks/gpu_${gpu}.lock" \
        bash scripts/run_goal162_one.sh "$gpu" "$arm" "$seed" >>"$log" 2>&1; then
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
  printf 'arm=%s\n' "$ARM"
  printf 'seeds=%s\n' "$SEEDS"
  sha256sum scripts/run_goal162_one.sh scripts/run_goal162_train.sh \
    configs/lowlevel/goal162/goal162_manifest.json
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/train_run_manifest.txt" 2>&1

echo "launching ${#JOBS[@]} Goal162 jobs on GPUs: $GPUS"
GPU_COUNT=$(wc -w <<<"$GPUS")
printf '%s\n' "${JOBS[@]}" | xargs -r -n 2 -P "$GPU_COUNT" \
  bash -c 'export GPUS="$1"; shift; run_one "$@"' _ "$GPUS"

missing=0
for seed in $SEEDS; do
  if [[ ! -f "$TRAIN_ROOT/$ARM/seed$seed/final.zip" ]]; then
    echo "MISSING $ARM seed$seed" >&2
    missing=$((missing + 1))
  fi
done

printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/train_run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal162 jobs missing; inspect $LOGS" >&2
  exit 1
fi
echo "Goal162 training complete: $TRAIN_ROOT"
