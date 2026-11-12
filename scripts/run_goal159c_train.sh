#!/usr/bin/env bash
# Goal159c: corrected 1M SAC screen on H3 + fixed C3/ALOS path following.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
GPUS=${GPUS:-"3 4 6"}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal159c_h3_current_sac_screen_20261112}
TRANSITIONS=${TRANSITIONS:-1000000}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/train"
read -r -a gpu_array <<<"$GPUS"
if (( ${#gpu_array[@]} < 3 )); then
  echo "three GPUs are required" >&2
  exit 2
fi

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  sha256sum configs/lowlevel/goal159/env_h3_current07_10_train_c3.json \
    configs/lowlevel/goal159/train_c3_seed86201.json \
    configs/lowlevel/goal159/train_c3_seed86202.json \
    configs/lowlevel/goal159/train_c3_seed86203.json
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
} >"$ROOT/manifests/train_manifest.txt" 2>&1

for seed in 86201 86202 86203; do
  idx=$((seed - 86201))
  gpu="${gpu_array[$idx]}"
  output="$ROOT/train/seed$seed"
  log="$ROOT/logs/train_seed$seed.log"
  if [[ -f "$output/final.zip" ]]; then
    echo "SKIP seed$seed (final.zip exists)"
    continue
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    exit 1
  fi
  echo "START seed$seed gpu$gpu transitions=$TRANSITIONS"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/train_pc_command_residual_sac.py \
      --config "configs/lowlevel/goal159/train_c3_seed$seed.json" \
      --output "$output" \
      --total-env-transitions "$TRANSITIONS" \
      --device cuda \
      --engineering-smoke \
      --overwrite \
      >"$log" 2>&1 &
done

wait
missing=0
for seed in 86201 86202 86203; do
  if [[ ! -f "$ROOT/train/seed$seed/final.zip" ]]; then
    echo "MISSING seed$seed/final.zip" >&2
    missing=$((missing + 1))
  fi
done
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/manifests/train_manifest.txt"
if (( missing != 0 )); then
  echo "$missing training runs missing; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal159c training complete: $ROOT/train"
