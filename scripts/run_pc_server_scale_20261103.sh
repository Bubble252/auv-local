#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${STAGE:-B}"
ARMS="${ARMS:-nodade dade dade_head dade_head_dropout05}"
SEEDS="${SEEDS:-91430 91431 91432}"
GPU_IDS="${GPU_IDS:-2}"
ITERATIONS="${ITERATIONS:-40}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_STEPS="${NUM_STEPS:-256}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-128}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-1800}"
WARMSTART="${WARMSTART:-$ROOT/artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/lowlevel/pc_server_scale_${STAGE}_20261103}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/pc_server_scale_${STAGE}_20261103}"
LOCK_DIR="${LOCK_DIR:-$LOG_DIR/locks}"
JOB_LOCK_DIR="${JOB_LOCK_DIR:-$OUTPUT_ROOT/.locks}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$LOCK_DIR" "$JOB_LOCK_DIR"
read -r -a arm_list <<< "$ARMS"
read -r -a seed_list <<< "$SEEDS"
read -r -a gpu_ids <<< "$GPU_IDS"
if (( ${#gpu_ids[@]} == 0 )); then
  printf 'GPU_IDS must not be empty\n' >&2
  exit 2
fi

declare -A CONFIGS=(
  [c2_nodade]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c2_alos_nodade_stable.json"
  [c2_dade]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c2_alos_dade_stable.json"
  [nodade]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [dade]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_aux_current_scale3e-2_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [dade_head]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_head_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [dade_head_dropout05]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_head_dropout05_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [dade_head_dropout10]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_head_dropout10_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [dade_head_sidechannel10]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_head_sidechannel10_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [dade_head_isolated_lr5e-4]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_head_isolated_lr5e-4_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [dade_history_only]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_history_only_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_dade]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_dade_head]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_head_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_dade_sidechannel10]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_sidechannel10_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_dade_history_only]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_only_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_dade_history_required]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_required_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_h1]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_h1_speed_1p8_2p0_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_h2]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_h2_speed_2p0_2p25_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_c4]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c4_stress_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_dade_head_isolated_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_head_isolated_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_dade_history_only_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_only_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_h1_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_h1_speed_1p8_2p0_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_h2_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_h2_speed_2p0_2p25_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_h3_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_h3_speed_2p25_2p4_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_c4_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c4_stress_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)

jobs=()
for seed in "${seed_list[@]}"; do
  for arm in "${arm_list[@]}"; do
    if [[ -z "${CONFIGS[$arm]:-}" ]]; then
      printf 'unknown arm: %s\n' "$arm" >&2
      exit 2
    fi
    jobs+=("$arm:$seed")
  done
done

wait_for_gpu() {
  local gpu="$1"
  local deadline=$((SECONDS + GPU_WAIT_SECONDS))
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ -z "$free_mib" ]]; then
      printf 'cannot query GPU %s\n' "$gpu" >&2
      return 1
    fi
    if (( free_mib >= MIN_FREE_MIB )); then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      printf 'GPU %s has only %s MiB free; required %s MiB\n' \
        "$gpu" "$free_mib" "$MIN_FREE_MIB" >&2
      return 1
    fi
    printf 'WAIT gpu=%s free_mib=%s required=%s\n' "$gpu" "$free_mib" "$MIN_FREE_MIB"
    sleep 30
  done
}

run_worker() {
  local slot="$1"
  local gpu="${gpu_ids[$slot]}"
  local index
  for ((index = slot; index < ${#jobs[@]}; index += ${#gpu_ids[@]})); do
    local job="${jobs[$index]}"
    local arm="${job%%:*}"
    local seed="${job##*:}"
    local output="$OUTPUT_ROOT/${arm}_seed${seed}"
    local log="$LOG_DIR/${arm}_seed${seed}.log"
    local status="$output/run_status.json"
    local allow_overwrite=0
    local job_lock_fd

    exec {job_lock_fd}> "$JOB_LOCK_DIR/${arm}_seed${seed}.lock"
    printf 'WAIT_JOB_LOCK arm=%s seed=%s output=%s\n' "$arm" "$seed" "$output"
    flock "$job_lock_fd"

    if [[ -f "$output/training_manifest.json" ]] \
      && grep -q '"status": "completed"' "$output/training_manifest.json" \
      && [[ -f "$output/final.pt" ]]; then
      printf 'SKIP arm=%s seed=%s output=%s\n' "$arm" "$seed" "$output"
      flock -u "$job_lock_fd"
      exec {job_lock_fd}>&-
      continue
    fi
    if [[ -d "$output" ]]; then
      local non_status_files
      non_status_files="$(find "$output" -maxdepth 1 -type f ! -name 'run_status*.json' | wc -l)"
      if (( non_status_files > 0 )); then
        printf 'SKIP_NONEMPTY arm=%s seed=%s output=%s non_status_files=%s\n' \
          "$arm" "$seed" "$output" "$non_status_files"
        flock -u "$job_lock_fd"
        exec {job_lock_fd}>&-
        continue
      fi
      if [[ -f "$status" ]] && ! grep -q '"status":"completed"' "$status"; then
        allow_overwrite=1
        printf 'RERUN_STATUS_ONLY arm=%s seed=%s output=%s\n' "$arm" "$seed" "$output"
      fi
    fi

    local lock_fd
    exec {lock_fd}> "$LOCK_DIR/gpu${gpu}.lock"
    printf 'WAIT_LOCK arm=%s seed=%s gpu=%s\n' "$arm" "$seed" "$gpu"
    flock "$lock_fd"
    if ! wait_for_gpu "$gpu"; then
      mkdir -p "$output"
      printf '{"status":"blocked_gpu","arm":"%s","seed":%s,"gpu":%s,"free_mib_required":%s}\n' \
        "$arm" "$seed" "$gpu" "$MIN_FREE_MIB" > "$status"
      flock -u "$lock_fd"
      exec {lock_fd}>&-
      flock -u "$job_lock_fd"
      exec {job_lock_fd}>&-
      continue
    fi

    printf 'START arm=%s seed=%s gpu=%s output=%s\n' "$arm" "$seed" "$gpu" "$output"
    local train_args=(
      --training-config "${CONFIGS[$arm]}"
      --device cuda
      --iterations "$ITERATIONS"
      --num-envs "$NUM_ENVS"
      --num-steps "$NUM_STEPS"
      --num-mini-batches "$NUM_MINI_BATCHES"
      --seed "$seed"
      --output "$output"
      --record-only
    )
    if (( allow_overwrite )); then
      train_args+=(--overwrite)
    fi
    if [[ "$arm" != frozen_bc_residual_* ]]; then
      train_args+=(--warmstart-actor "$WARMSTART")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_batch_lowlevel_ppo.py \
      "${train_args[@]}" > "$log" 2>&1
    local rc=$?
    mkdir -p "$output"
    printf '{"status":"%s","returncode":%s,"arm":"%s","seed":%s,"gpu":%s,"iterations":%s,"num_envs":%s,"num_steps":%s,"num_mini_batches":%s}\n' \
      "$([[ $rc -eq 0 ]] && printf completed || printf failed)" "$rc" "$arm" "$seed" "$gpu" \
      "$ITERATIONS" "$NUM_ENVS" "$NUM_STEPS" "$NUM_MINI_BATCHES" > "$status"
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    flock -u "$job_lock_fd"
    exec {job_lock_fd}>&-
    printf 'END arm=%s seed=%s gpu=%s rc=%s\n' "$arm" "$seed" "$gpu" "$rc"
  done
}

for ((slot = 0; slot < ${#gpu_ids[@]}; slot++)); do
  run_worker "$slot" &
done
wait

failures=0
for job in "${jobs[@]}"; do
  arm="${job%%:*}"
  seed="${job##*:}"
  status="$OUTPUT_ROOT/${arm}_seed${seed}/run_status.json"
  if [[ ! -f "$status" ]] || ! grep -q '"status":"completed"' "$status"; then
    printf 'FAILED arm=%s seed=%s status=%s\n' "$arm" "$seed" "$status"
    failures=$((failures + 1))
  fi
done
printf 'ALL_JOBS_DONE output_root=%s failures=%s\n' "$OUTPUT_ROOT" "$failures"
exit "$((failures > 0))"
