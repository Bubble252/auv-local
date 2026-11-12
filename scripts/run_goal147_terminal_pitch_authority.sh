#!/usr/bin/env bash
# Goal147: frozen eval-only terminal slowdown / pitch controller gain arms.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal147_c1_terminal_pitch_authority_20261111}
GPUS=${GPUS:-"0 3 6 7"}
MAX_PARALLEL=${MAX_PARALLEL:-10}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}

CHECKPOINT=${CHECKPOINT:-$REPO/artifacts/lowlevel/goal144_alos_rl_20261111/ppo/C-full-alos-command/S_s86201_gatefix2/curriculum_checkpoints/C1_000300000.pt}
TRAINING_CONFIG=${TRAINING_CONFIG:-configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json}
ENVIRONMENT_CONFIG=${ENVIRONMENT_CONFIG:-configs/lowlevel/goal136/curriculum/C0.json}

cd "$REPO"
read -r -a GPU_ARRAY <<<"$GPUS"
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
    configs/lowlevel/goal147/guidance_z.json \
    configs/lowlevel/goal147/guidance_b85.json \
    configs/lowlevel/goal147/guidance_b75.json \
    configs/lowlevel/goal147/guidance_k4.json \
    configs/lowlevel/goal147/guidance_r2.json \
    configs/lowlevel/goal147/controller_k4.json \
    configs/lowlevel/goal147/controller_r2.json
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/manifests/run_manifest.txt" 2>&1

cp configs/lowlevel/goal147/profile_manifest.json "$ROOT/manifests/"
cp configs/lowlevel/goal147/*.json "$ROOT/manifests/"

pids=()
status=0
job_index=0

gpu_for_job() {
  echo "${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
}

wait_batch() {
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  pids=()
}

launch_job() {
  local arm="$1"
  local mode="$2"
  local action_mode="$3"
  local guidance="$4"
  local controller="$5"
  local gpu
  gpu=$(gpu_for_job)
  local out="$ROOT/$arm/$mode"
  mkdir -p "$out"
  echo "START arm=$arm mode=$mode action=$action_mode gpu=$gpu guidance=$guidance controller=${controller:-none}"
  (
    cd "$REPO"
    args=(
      "$PY" scripts/evaluate_pc_ppo.py
      --training-config "$TRAINING_CONFIG"
      --checkpoint "$CHECKPOINT"
      --environment-config "$ENVIRONMENT_CONFIG"
      --curriculum-stage C1
      --disable-curriculum-direct-command
      --guidance-config "$guidance"
      --output "$out"
      --trace-output "$out/trace.npz"
      --trace-reward-breakdown
      --seed-start "$SEED_START"
      --seed-count "$SEED_COUNT"
      --success-frame both
      --device cuda
      --dtype float64
      --action-mode "$action_mode"
    )
    if [[ -n "$controller" ]]; then
      args+=(--controller-config "$controller")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH="src" \
      nohup "${args[@]}" >"$ROOT/logs/${arm}_${mode}.log" 2>&1
  ) &
  pids+=("$!")
  job_index=$((job_index + 1))
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_batch
  fi
}

for arm in z b85 b75 k4 r2; do
  guidance="configs/lowlevel/goal147/guidance_${arm}.json"
  controller=""
  if [[ "$arm" == "k4" || "$arm" == "r2" ]]; then
    controller="configs/lowlevel/goal147/controller_${arm}.json"
  fi
  launch_job "$arm" policy policy "$guidance" "$controller"
  launch_job "$arm" zero zero-command-residual "$guidance" "$controller"
done

wait_batch
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/manifests/run_manifest.txt"
if [[ "$status" -ne 0 ]]; then
  echo "one or more Goal147 evaluations failed; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal147 terminal pitch authority eval complete: $ROOT"
