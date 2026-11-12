#!/usr/bin/env bash
# Goal173 Phase 1: 8 arms x 1M transitions, frozen C3 curriculum.
set -euo pipefail

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal173_rudder_only_residual_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}
LOGS=${LOGS:-$ROOT/logs}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
ARMS=${ARMS:-"P_ALL_025 P_ALL_075 P_RUD_025 P_RUD_075 S_ALL_025 S_ALL_075 S_RUD_025 S_RUD_075"}
SEEDS=${SEEDS:-"86201"}
MIN_CURRICULUM_EPISODES=${MIN_CURRICULUM_EPISODES:-100}

cd "$REPO"
export PYTHONPATH="src:$SB3"
"$PY" scripts/check_goal173_protocol.py --phase train
mkdir -p "$TRAIN_ROOT" "$LOGS" "$ROOT/manifests"

# A stale launcher can keep writing the same output after a restart. Fail closed
# instead of silently mixing two trainers into one provenance root.
exec 9>"$ROOT/.train.lock"
if ! flock -n 9; then
  echo "another Goal173 trainer holds $ROOT/.train.lock" >&2
  exit 3
fi
if pgrep -f "train_(batch_lowlevel_ppo|pc_command_residual_sac)\.py.*$ROOT" >/dev/null; then
  echo "stale Goal173 training process still references $ROOT" >&2
  pgrep -af "train_(batch_lowlevel_ppo|pc_command_residual_sac)\.py.*$ROOT" >&2 || true
  exit 3
fi

run_one() {
  local gpu=$1 arm=$2 seed=$3
  local config output log checkpoint
  config="configs/lowlevel/goal173/train_${arm,,}_seed${seed}.json"
  output="$TRAIN_ROOT/$arm/seed$seed"
  log="$LOGS/${arm}_seed${seed}.log"
  if [[ ! -f "$config" ]]; then
    echo "missing config: $config" >&2
    return 2
  fi
  checkpoint=""
  if [[ -f "$output/final.zip" ]]; then checkpoint="$output/final.zip"; fi
  if [[ -f "$output/final.pt" ]]; then checkpoint="$output/final.pt"; fi
  if [[ -n "$checkpoint" && -f "$output/training_manifest.json" && -f "$output/curriculum_episodes.jsonl" ]]; then
    echo "SKIP $arm seed$seed"
    return 0
  fi
  if [[ -n "$checkpoint" ]]; then
    echo "refusing to skip incomplete training output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  export PYTHONPATH="src:$SB3"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu"
  echo "START arm=$arm seed=$seed gpu=$gpu config=$config"
  if [[ "$arm" == S_* ]]; then
    "$PY" scripts/train_pc_command_residual_sac.py \
      --config "$config" --output "$output" --device cuda --dtype float64 \
      --sb3-root "$SB3" --overwrite >"$log" 2>&1
  else
    "$PY" scripts/train_batch_lowlevel_ppo.py \
      --training-config "$config" --output "$output" --device cuda \
      --overwrite >"$log" 2>&1
  fi
  echo "DONE arm=$arm seed=$seed gpu=$gpu"
}
export -f run_one
export REPO PY SB3 ROOT TRAIN_ROOT LOGS MIN_CURRICULUM_EPISODES

{
  printf 'started_at=%s\nhost=%s\nrepo=%s\ngpus=%s\narms=%s\nseeds=%s\n' \
    "$(date -Iseconds)" "$(hostname)" "$REPO" "$GPUS" "$ARMS" "$SEEDS"
  sha256sum scripts/run_goal173_train.sh scripts/make_goal173_configs.py \
    configs/lowlevel/goal173/goal173_manifest.json 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/manifests/train_manifest.txt" 2>&1

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
read -r -a gpu_array <<<"$GPUS"
index=0
for arm in $ARMS; do
  for seed in $SEEDS; do
    gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
    echo "$gpu $arm $seed" >>"$JOB_FILE"
    index=$((index + 1))
  done
done

echo "launching $index Goal173 training jobs"
xargs -r -n 3 -P "${#gpu_array[@]}" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
for arm in $ARMS; do
  for seed in $SEEDS; do
    if [[ ! -f "$TRAIN_ROOT/$arm/seed$seed/final.zip" && ! -f "$TRAIN_ROOT/$arm/seed$seed/final.pt" ]]; then
      echo "MISSING $arm seed$seed" >&2
      missing=$((missing + 1))
    fi
  done
done
printf 'finished_at=%s\nmissing=%s\n' "$(date -Iseconds)" "$missing" >>"$ROOT/manifests/train_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal173 training jobs missing; inspect $LOGS" >&2
  exit 1
fi
"$PY" scripts/audit_goal173_artifacts.py \
  --phase train --root "$ROOT" --seeds "$SEEDS" --min-episodes "$MIN_CURRICULUM_EPISODES"
echo "Goal173 training complete: $TRAIN_ROOT"

