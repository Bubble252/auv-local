#!/usr/bin/env bash
# Goal148 Stage 1: frozen eval-only K4 stress matrix.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal148_k4_terminal_pitch_stress_20261111}
PARALLEL=${PARALLEL:-8}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
TRACE=${TRACE:-1}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"0 3 4 6 7"}

CHECKPOINT_300K=${CHECKPOINT_300K:-$REPO/artifacts/lowlevel/goal144_alos_rl_20261111/ppo/C-full-alos-command/S_s86201_gatefix2/curriculum_checkpoints/C1_000300000.pt}
CHECKPOINT_400K=${CHECKPOINT_400K:-$REPO/artifacts/lowlevel/goal144_alos_rl_20261111/ppo/C-full-alos-command/S_s86201_gatefix2/curriculum_checkpoints/C1_000400000.pt}
TRAINING_CONFIG=${TRAINING_CONFIG:-configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json}
GUIDANCE_CONFIG=${GUIDANCE_CONFIG:-configs/lowlevel/goal147/guidance_z.json}
CONTROLLER_K4=${CONTROLLER_K4:-configs/lowlevel/goal147/controller_k4.json}
LEVELS=${LEVELS:-"nominal current15 current20 plant20 combined20"}
GUARD_LEVELS=${GUARD_LEVELS:-"nominal combined20"}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/manifests"

for checkpoint in "$CHECKPOINT_300K" "$CHECKPOINT_400K"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 2
  fi
done

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'checkpoint_300k=%s\n' "$CHECKPOINT_300K"
  printf 'checkpoint_400k=%s\n' "$CHECKPOINT_400K"
  sha256sum "$CHECKPOINT_300K" "$CHECKPOINT_400K" "$TRAINING_CONFIG" \
    "$GUIDANCE_CONFIG" "$CONTROLLER_K4" \
    configs/lowlevel/goal148/env_*.json configs/lowlevel/goal148/profile_manifest.json
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/manifests/run_manifest.txt" 2>&1
cp configs/lowlevel/goal148/profile_manifest.json "$ROOT/manifests/"
cp configs/lowlevel/goal148/env_*.json "$ROOT/manifests/"

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT

job_index=0
for level in $LEVELS; do
  for arm in z k4; do
    for mode in policy zero; do
      echo "$level $arm 300k $mode $job_index" >>"$JOB_FILE"
      job_index=$((job_index + 1))
    done
  done
done
for level in $GUARD_LEVELS; do
  for mode in policy zero; do
    echo "$level k4_400k 400k $mode $job_index" >>"$JOB_FILE"
    job_index=$((job_index + 1))
  done
done

run_one() {
  local level="$1"
  local arm="$2"
  local checkpoint_kind="$3"
  local mode="$4"
  local index="$5"
  local environment_config="configs/lowlevel/goal148/env_${level}.json"
  local output="$ROOT/$level/$arm/$mode"
  local log="$ROOT/logs/${level}_${arm}_${mode}.log"
  local checkpoint="$CHECKPOINT_300K"
  local action_mode="policy"
  local controller=""
  local extra=()
  local current_speed_args=()
  local shear_args=()
  local gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"

  if [[ ! -f "$environment_config" ]]; then
    echo "missing environment config: $environment_config" >&2
    return 1
  fi
  if [[ "$checkpoint_kind" == "400k" ]]; then
    checkpoint="$CHECKPOINT_400K"
  fi
  if [[ "$mode" == "zero" ]]; then
    action_mode="zero-command-residual"
  fi
  if [[ "$arm" == "k4" || "$arm" == "k4_400k" ]]; then
    controller="$CONTROLLER_K4"
  fi
  case "$level" in
    current15)
      current_speed_args=(--curriculum-current-speed-range-mps 0.375 0.675)
      shear_args=(--curriculum-shear-scale 1.5)
      ;;
    current20|combined20)
      current_speed_args=(--curriculum-current-speed-range-mps 0.5 0.9)
      shear_args=(--curriculum-shear-scale 2.0)
      ;;
  esac
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $level/$arm/$mode (summary exists)"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  if [[ "$TRACE" == "1" ]]; then
    extra+=(--trace-output "$output/trace.npz" --trace-reward-breakdown)
  fi
  if [[ -n "$controller" ]]; then
    extra+=(--controller-config "$controller")
  fi

  echo "START level=$level arm=$arm checkpoint=$checkpoint_kind mode=$mode device=$DEVICE gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="src" \
    "$PY" scripts/evaluate_pc_ppo.py \
    --training-config "$TRAINING_CONFIG" \
    --checkpoint "$checkpoint" \
    --environment-config "$environment_config" \
    --curriculum-stage C1 \
    --disable-curriculum-direct-command \
    --guidance-config "$GUIDANCE_CONFIG" \
    --output "$output" \
    --seed-start "$SEED_START" \
    --seed-count "$SEED_COUNT" \
    --success-frame both \
    --device "$DEVICE" \
    --dtype float64 \
    --action-mode "$action_mode" \
    "${current_speed_args[@]}" \
    "${shear_args[@]}" \
    "${extra[@]}" \
    >"$log" 2>&1
  echo "DONE level=$level arm=$arm mode=$mode"
}

export PY REPO ROOT SEED_START SEED_COUNT TRACE DEVICE GPUS CHECKPOINT_300K CHECKPOINT_400K
export TRAINING_CONFIG GUIDANCE_CONFIG CONTROLLER_K4
export -f run_one

echo "launching $(wc -l <"$JOB_FILE") Goal148 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 5 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
while read -r level arm checkpoint_kind mode index; do
  if [[ ! -f "$ROOT/$level/$arm/$mode/summary.json" ]]; then
    echo "MISSING $level/$arm/$mode" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"

printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing evaluations are missing; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal148 K4 stress eval complete: $ROOT"
