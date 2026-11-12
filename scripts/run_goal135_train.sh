#!/usr/bin/env bash
# Goal135: 3 authority regimes x 8 seeds, one 1M SAC run per seed on one GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=${ROOT:-artifacts/lowlevel/goal135_authority_regularized_sac_20261111}
ARMS=${ARMS:-"R030 R025 R020"}
SEEDS=${SEEDS:-"86201 86202 86203 86204 86205 86206 86207 86208"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$ROOT/logs"
if [[ ! -f "$ROOT/config_manifest.json" ]]; then
  "$PY" scripts/make_goal135_configs.py --root "$ROOT"
fi

read -r -a GPU_ARRAY <<<"$GPUS"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "GPUS must contain at least one device" >&2
  exit 1
fi

run_arm() {
  local arm="$1"
  local -a pids=()
  local -a labels=()
  local index=0
  local seed gpu gpu_count
  gpu_count=${#GPU_ARRAY[@]}
  for seed in $SEEDS; do
    gpu=${GPU_ARRAY[$((index % gpu_count))]}
    index=$((index + 1))
    local config="$ROOT/training_configs/$arm/S_s${seed}.json"
    local output="$ROOT/screen/$arm/S_s${seed}"
    local log="$ROOT/logs/train_${arm}_S_s${seed}.log"
    if [[ ! -f "$config" ]]; then
      echo "missing config: $config" >&2
      return 1
    fi
    if [[ -f "$output/final.zip" && -f "$output/training_manifest.json" ]]; then
      echo "SKIP $arm seed=$seed (checkpoint exists)"
      continue
    fi
    if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
      echo "refusing to overwrite non-empty output: $output" >&2
      return 1
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY arm=$arm seed=$seed gpu=$gpu config=$config output=$output"
      continue
    fi
    echo "START $arm seed=$seed gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      PYTHONPATH=src \
      "$PY" scripts/train_pc_command_residual_sac.py \
      --config "$config" \
      --output "$output" \
      --device cuda \
      --dtype float64 \
      >"$log" 2>&1 &
    pids+=("$!")
    labels+=("$arm/S_s${seed}")
  done

  local failed=0
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "DONE ${labels[$i]}"
    else
      echo "FAILED ${labels[$i]}" >&2
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi
  for seed in $SEEDS; do
    if [[ ! -f "$ROOT/screen/$arm/S_s${seed}/final.zip" ]]; then
      echo "missing checkpoint $ROOT/screen/$arm/S_s${seed}/final.zip" >&2
      return 1
    fi
  done
}

for arm in $ARMS; do
  run_arm "$arm"
done

echo "Goal135 screen training complete: $ROOT"
