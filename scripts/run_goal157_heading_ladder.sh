#!/usr/bin/env bash
# Goal157: frozen eval-only K4 + heading-gain ladder, zero residual only.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
GOAL136=${GOAL136:-/data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal157_controller_heading_gain_20261112}
PARALLEL=${PARALLEL:-6}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"3 4 6 7"}
LEVELS=${LEVELS:-"nominal current05 current07 current10 current15"}
ARMS=${ARMS:-"h2 h3 h4"}

CHECKPOINT=${CHECKPOINT:-$GOAL136/artifacts/lowlevel/goal144_alos_rl_20261111/ppo/C-full-alos-command/S_s86201_gatefix2/curriculum_checkpoints/C1_000300000.pt}
TRAINING_CONFIG=${TRAINING_CONFIG:-configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json}
GUIDANCE_CONFIG=${GUIDANCE_CONFIG:-$GOAL136/configs/lowlevel/goal147/guidance_z.json}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/manifests"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi
for arm in $ARMS; do
  if [[ ! -f "configs/lowlevel/goal157/env_${arm}.json" ]]; then
    echo "missing environment config: configs/lowlevel/goal157/env_${arm}.json" >&2
    exit 2
  fi
done

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'checkpoint=%s\n' "$CHECKPOINT"
  sha256sum "$CHECKPOINT" "$TRAINING_CONFIG" "$GUIDANCE_CONFIG" \
    configs/lowlevel/goal157/env_h2.json \
    configs/lowlevel/goal157/env_h3.json \
    configs/lowlevel/goal157/env_h4.json
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/manifests/run_manifest.txt" 2>&1

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_index=0
for level in $LEVELS; do
  for arm in $ARMS; do
    echo "$level $arm $job_index" >>"$JOB_FILE"
    job_index=$((job_index + 1))
  done
done

run_one() {
  local level="$1"
  local arm="$2"
  local index="$3"
  local output="$ROOT/$level/$arm/zero"
  local log="$ROOT/logs/${level}_${arm}_zero.log"
  local current_speed_args=()
  local shear_args=()
  local gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"

  case "$level" in
    nominal) ;;
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
    current15)
      current_speed_args=(--curriculum-current-speed-range-mps 0.375 0.675)
      shear_args=(--curriculum-shear-scale 1.5)
      ;;
    *)
      echo "unknown level: $level" >&2
      return 2
      ;;
  esac

  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $level/$arm/zero (summary exists)"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START level=$level arm=$arm device=$DEVICE gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config "$TRAINING_CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --environment-config "configs/lowlevel/goal157/env_${arm}.json" \
      --curriculum-stage C1 \
      --disable-curriculum-direct-command \
      --guidance-config "$GUIDANCE_CONFIG" \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
      --device "$DEVICE" \
      --dtype float64 \
      --action-mode zero-command-residual \
      --trace-output "$output/trace.npz" \
      --trace-reward-breakdown \
      "${current_speed_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1
  echo "DONE level=$level arm=$arm"
}

export PY REPO ROOT SEED_START SEED_COUNT DEVICE GPUS
export CHECKPOINT TRAINING_CONFIG GUIDANCE_CONFIG
export -f run_one

echo "launching $(wc -l <"$JOB_FILE") Goal157 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 3 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
while read -r level arm index; do
  if [[ ! -f "$ROOT/$level/$arm/zero/summary.json" ]]; then
    echo "MISSING $level/$arm/zero" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"

printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing evaluations are missing; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal157 heading-gain ladder complete: $ROOT"
