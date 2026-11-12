#!/usr/bin/env bash
# Goal145: eval-only trace audit for Goal86 T2 and Goal144 C1.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
T2_REPO=${T2_REPO:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式}
C1_REPO=${C1_REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111}
ROOT=${ROOT:-$T2_REPO/artifacts/lowlevel/goal145_terminal_capture_vertical_audit_20261111}
GPUS=${GPUS:-"0 3 6 7"}
MAX_PARALLEL=${MAX_PARALLEL:-4}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}

read -r -a GPU_ARRAY <<<"$GPUS"
mkdir -p "$ROOT/logs" "$ROOT/t2" "$ROOT/c1"

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

launch_background() {
  local tag="$1"
  local gpu="$2"
  local log="$3"
  local cwd="$4"
  shift 4
  echo "START $tag gpu=$gpu"
  (
    cd "$cwd"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH="src" \
      nohup "$@" >"$log" 2>&1
  ) &
  pids+=("$!")
  job_index=$((job_index + 1))
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_batch
  fi
}

run_t2() {
  local tag="$1"
  local mode="$2"
  local config="$3"
  local checkpoint="$4"
  local env="$5"
  local scenario="$6"
  local out="$ROOT/t2/$tag/$scenario"
  local gpu
  gpu=$(gpu_for_job)
  mkdir -p "$out"
  local args=(
    "$PY" scripts/evaluate_pc_sac.py
    --config "$config"
    --environment-config "$env"
    --output "$out"
    --trace-output "$out"
    --trace-reward-breakdown
    --rich-trace
    --seed-start "$SEED_START"
    --seed-count "$SEED_COUNT"
    --success-frame both
    --device cuda
    --dtype float64
    --action-mode "$mode"
    --residual-scale 1.0
  )
  if [[ -n "$checkpoint" ]]; then
    args+=(--checkpoint "$checkpoint")
  fi
  launch_background "t2/$tag/$scenario" "$gpu" "$ROOT/logs/t2_${tag}_${scenario}.log" "$T2_REPO" "${args[@]}"
}

run_c1() {
  local tag="$1"
  local interface="$2"
  local mode="$3"
  local checkpoint="$4"
  local training_config="$5"
  local scenario="$6"
  local out="$ROOT/c1/$tag/$scenario"
  local gpu
  gpu=$(gpu_for_job)
  mkdir -p "$out"
  local args=(
    "$PY" scripts/evaluate_pc_ppo.py
    --training-config "$training_config"
    --checkpoint "$checkpoint"
    --environment-config configs/lowlevel/goal136/curriculum/C0.json
    --curriculum-stage C1
    --disable-curriculum-direct-command
    --output "$out"
    --trace-output "$out/trace.npz"
    --trace-reward-breakdown
    --seed-start "$SEED_START"
    --seed-count "$SEED_COUNT"
    --success-frame both
    --device cuda
    --dtype float64
    --action-mode "$mode"
  )
  launch_background "c1/$tag/$scenario" "$gpu" "$ROOT/logs/c1_${tag}_${scenario}.log" "$C1_REPO" "${args[@]}"
}

T2_CONFIG=configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json
T2_ROOT=artifacts/lowlevel/goal86/screen
T2_DEFAULT_ENV=configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json
T2_STRICT_ENV=configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json

for checkpoint in \
  "$T2_ROOT/t2_sac/final.zip" \
  "$T2_ROOT/t2_sac_seed86202/final.zip" \
  "$T2_ROOT/t2_sac_seed86203/final.zip"; do
  seed=$(basename "$(dirname "$checkpoint")")
  for scenario in default strict125; do
    if [[ "$scenario" == strict125 ]]; then
      env="$T2_STRICT_ENV"
    else
      env="$T2_DEFAULT_ENV"
    fi
    run_t2 "policy_${seed}" policy "$T2_CONFIG" "$checkpoint" "$env" "$scenario"
  done
done

for scenario in default strict125; do
  if [[ "$scenario" == strict125 ]]; then
    env="$T2_STRICT_ENV"
  else
    env="$T2_DEFAULT_ENV"
  fi
  run_t2 "zero" zero-residual "$T2_CONFIG" "" "$env" "$scenario"
done

C1_CHECKPOINT_ROOT=artifacts/lowlevel/goal144_alos_rl_20261111/ppo
C1_COMMAND_CONFIG=configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json
C1_ACTUATOR_CONFIG=configs/lowlevel/goal144/C-full-alos-actuator_ppo_pilot.json

for step in 000300000 000400000; do
  for interface in command actuator; do
    if [[ "$interface" == command ]]; then
      config="$C1_COMMAND_CONFIG"
      zero_mode=zero-command-residual
    else
      config="$C1_ACTUATOR_CONFIG"
      zero_mode=zero-actuator-residual
    fi
    checkpoint="$C1_CHECKPOINT_ROOT/C-full-alos-${interface}/S_s86201_gatefix2/curriculum_checkpoints/C1_${step}.pt"
    run_c1 "${interface}_${step}_policy" "$interface" policy "$checkpoint" "$config" default
    if [[ "$step" == 000300000 ]]; then
      run_c1 "${interface}_${step}_zero" "$interface" "$zero_mode" "$checkpoint" "$config" default
    fi
  done
done

wait_batch
if [[ "$status" -ne 0 ]]; then
  echo "one or more Goal145 trace evaluations failed; inspect $ROOT/logs" >&2
  exit 1
fi
echo "Goal145 trace audit complete: $ROOT"
