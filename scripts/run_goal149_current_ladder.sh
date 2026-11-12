#!/usr/bin/env bash
# Goal149: frozen eval-only K4 current-boundary ladder.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal149_k4_current_ladder_20261111}
PARALLEL=${PARALLEL:-8}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
TRACE=${TRACE:-1}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"0 3 4 6 7"}
LEVELS=${LEVELS:-"nominal current05 current07 current10 current12 current15"}

CHECKPOINT=${CHECKPOINT:-$REPO/artifacts/lowlevel/goal144_alos_rl_20261111/ppo/C-full-alos-command/S_s86201_gatefix2/curriculum_checkpoints/C1_000300000.pt}
TRAINING_CONFIG=${TRAINING_CONFIG:-configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json}
ENVIRONMENT_CONFIG=${ENVIRONMENT_CONFIG:-configs/lowlevel/goal148/env_nominal.json}
GUIDANCE_CONFIG=${GUIDANCE_CONFIG:-configs/lowlevel/goal147/guidance_z.json}
CONTROLLER_K4=${CONTROLLER_K4:-configs/lowlevel/goal147/controller_k4.json}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/manifests"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'checkpoint=%s\n' "$CHECKPOINT"
  sha256sum "$CHECKPOINT" "$TRAINING_CONFIG" "$ENVIRONMENT_CONFIG" \
    "$GUIDANCE_CONFIG" "$CONTROLLER_K4"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/manifests/run_manifest.txt" 2>&1

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_index=0
for level in $LEVELS; do
  for arm in z k4; do
    for mode in policy zero; do
      echo "$level $arm $mode $job_index" >>"$JOB_FILE"
      job_index=$((job_index + 1))
    done
  done
done

run_one() {
  local level="$1"
  local arm="$2"
  local mode="$3"
  local index="$4"
  local output="$ROOT/$level/$arm/$mode"
  local log="$ROOT/logs/${level}_${arm}_${mode}.log"
  local action_mode="policy"
  local controller=""
  local current_speed_args=()
  local shear_args=()
  local extra=()
  local gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"

  if [[ "$mode" == "zero" ]]; then
    action_mode="zero-command-residual"
  fi
  if [[ "$arm" == "k4" ]]; then
    controller="$CONTROLLER_K4"
  fi
  case "$level" in
    current05)
      current_speed_args=(--curriculum-current-speed-range-mps 0.25 0.45)
      shear_args=(--curriculum-shear-scale 0.5)
      ;;
    current07)
      current_speed_args=(--curriculum-current-speed-range-mps 0.275 0.50)
      shear_args=(--curriculum-shear-scale 0.75)
      ;;
    current10)
      current_speed_args=(--curriculum-current-speed-range-mps 0.30 0.55)
      shear_args=(--curriculum-shear-scale 1.0)
      ;;
    current12)
      current_speed_args=(--curriculum-current-speed-range-mps 0.34 0.60)
      shear_args=(--curriculum-shear-scale 1.25)
      ;;
    current15)
      current_speed_args=(--curriculum-current-speed-range-mps 0.375 0.675)
      shear_args=(--curriculum-shear-scale 1.5)
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

  echo "START level=$level arm=$arm mode=$mode device=$DEVICE gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="src" \
    "$PY" scripts/evaluate_pc_ppo.py \
    --training-config "$TRAINING_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --environment-config "$ENVIRONMENT_CONFIG" \
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

export PY REPO ROOT SEED_START SEED_COUNT TRACE DEVICE GPUS
export CHECKPOINT TRAINING_CONFIG ENVIRONMENT_CONFIG GUIDANCE_CONFIG CONTROLLER_K4
export -f run_one

echo "launching $(wc -l <"$JOB_FILE") Goal149 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 4 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
while read -r level arm mode index; do
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
echo "Goal149 current ladder eval complete: $ROOT"
