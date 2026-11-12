#!/usr/bin/env bash
# Goal139 mixed-horizon SAC screen: M50/M75 x 3 seeds x 1M transitions.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
ROOT=${ROOT:-artifacts/lowlevel/goal139_mixed_horizon_credit_20261111}
ARMS=${ARMS:-"M50 M75"}
SEEDS=${SEEDS:-"86201 86202 86203"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$ROOT/logs"
if [[ ! -f "$ROOT/config_manifest.json" ]]; then
  python3 scripts/build_pc_goal139_mixed_credit_configs.py --artifact-root "$ROOT"
fi
read -r -a GPU_ARRAY <<<"$GPUS"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then echo "GPUS must not be empty" >&2; exit 1; fi

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
index=0
for arm in $ARMS; do
  for seed in $SEEDS; do
    gpu=${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}
    index=$((index + 1))
    echo "$arm $seed $gpu" >>"$JOB_FILE"
  done
done

run_one() {
  local arm="$1" seed="$2" gpu="$3"
  local config="$ROOT/training_configs/SAC/$arm/S_s${seed}.json"
  local output="$ROOT/screen/SAC/$arm/S_s${seed}"
  local log="$ROOT/logs/train_SAC_${arm}_S_s${seed}.log"
  if [[ ! -f "$config" ]]; then echo "missing config: $config" >&2; return 1; fi
  if [[ -f "$output/final.zip" && -f "$output/training_manifest.json" ]]; then
    echo "SKIP SAC/$arm/S_s${seed}"; return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2; return 1
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY SAC arm=$arm seed=$seed gpu=$gpu config=$config output=$output"; return 0
  fi
  echo "START SAC/$arm seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8     PYTHONPATH=src     "$PY" scripts/train_pc_command_residual_sac.py     --config "$config"     --output "$output"     --device cuda     --dtype float64     >"$log" 2>&1
  echo "DONE SAC/$arm/S_s${seed}"
}

export PY ROOT DRY_RUN
export -f run_one

if [[ "$DRY_RUN" == "1" ]]; then cat "$JOB_FILE"; exit 0; fi
while read -r arm seed gpu; do run_one "$arm" "$seed" "$gpu" & done <"$JOB_FILE"
wait
missing=0
while read -r arm seed gpu; do
  if [[ ! -f "$ROOT/screen/SAC/$arm/S_s${seed}/final.zip" ]]; then
    echo "MISSING SAC/$arm/S_s${seed}" >&2; missing=$((missing + 1))
  fi
done <"$JOB_FILE"
if [[ "$missing" -ne 0 ]]; then echo "$missing SAC runs missing; inspect $ROOT/logs" >&2; exit 1; fi
echo "Goal139 mixed-horizon SAC training complete: $ROOT"
