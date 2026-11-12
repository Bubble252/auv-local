#!/usr/bin/env bash
# Goal146: frozen eval-only Z/P1/P2/S1 terminal guidance authority arms.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal146_c1_terminal_authority_20261111}
GPUS=${GPUS:-"3 4 6 7"}
MAX_PARALLEL=${MAX_PARALLEL:-8}
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
    configs/lowlevel/goal146/guidance_z.json \
    configs/lowlevel/goal146/guidance_p1_pitch20deg.json \
    configs/lowlevel/goal146/guidance_p2_pitch30deg.json \
    configs/lowlevel/goal146/guidance_s1_speed035.json
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$ROOT/manifests/run_manifest.txt" 2>&1

cp configs/lowlevel/goal146/guidance_manifest.json "$ROOT/manifests/"
cp configs/lowlevel/goal146/guidance_z.json "$ROOT/manifests/"
cp configs/lowlevel/goal146/guidance_p1_pitch20deg.json "$ROOT/manifests/"
cp configs/lowlevel/goal146/guidance_p2_pitch30deg.json "$ROOT/manifests/"
cp configs/lowlevel/goal146/guidance_s1_speed035.json "$ROOT/manifests/"

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
  local gpu
  gpu=$(gpu_for_job)
  local out="$ROOT/$arm/$mode"
  mkdir -p "$out"
  echo "START arm=$arm mode=$mode action=$action_mode gpu=$gpu guidance=$guidance"
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH="src" \
      nohup "$PY" scripts/evaluate_pc_ppo.py \
        --training-config "$TRAINING_CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --environment-config "$ENVIRONMENT_CONFIG" \
        --curriculum-stage C1 \
        --disable-curriculum-direct-command \
        --guidance-config "$guidance" \
        --output "$out" \
        --trace-output "$out/trace.npz" \
        --trace-reward-breakdown \
        --seed-start "$SEED_START" \
        --seed-count "$SEED_COUNT" \
        --success-frame both \
        --device cuda \
        --dtype float64 \
        --action-mode "$action_mode" \
        >"$ROOT/logs/${arm}_${mode}.log" 2>&1
  ) &
  pids+=("$!")
  job_index=$((job_index + 1))
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_batch
  fi
}

for arm in z p1_pitch20deg p2_pitch30deg s1_speed035; do
  guidance="configs/lowlevel/goal146/guidance_${arm}.json"
  # policy is the RL arm; zero is the paired traditional guidance control.
  launch_job "$arm" policy policy "$guidance"
  launch_job "$arm" zero zero-command-residual "$guidance"
done

wait_batch
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/manifests/run_manifest.txt"
if [[ "$status" -ne 0 ]]; then
  echo "one or more Goal146 evaluations failed; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal146 guidance authority eval complete: $ROOT"
