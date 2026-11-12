#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${STAGE:-B}"
ARMS="${ARMS:-nodade dade dade_head dade_head_dropout05}"
SEEDS="${SEEDS:-91430 91431 91432}"
TAGS="${TAGS:-default strict125}"
CHECKPOINTS="${CHECKPOINTS:-0 10 20 30 39}"
SEED_START="${SEED_START:-0}"
SEED_COUNT="${SEED_COUNT:-64}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float64}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
CPU_BASE="${CPU_BASE:-160}"
CPU_STRIDE="${CPU_STRIDE:-16}"
HISTORY_ABLATION="${HISTORY_ABLATION:-none}"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/lowlevel/pc_server_scale_${STAGE}_20261103}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/lowlevel/pc_server_scale_${STAGE}_eval_20261103}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/pc_server_scale_${STAGE}_eval_20261103}"
DEFAULT_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
STRICT_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json"
C2_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c2_alos.json"
C2_CLEAN_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c2_alos_clean.json"
C4_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c4_stress.json"
H0_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h0_speed_1p0_1p8.json"
H1_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json"
H2_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h2_speed_2p0_2p25.json"
H3_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h3_speed_2p25_2p4.json"
H4_ENV="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h4_speed_2p5_2p5.json"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
read -r -a arm_list <<< "$ARMS"
read -r -a seed_list <<< "$SEEDS"
read -r -a tag_list <<< "$TAGS"

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
for tag in "${tag_list[@]}"; do
  for arm in "${arm_list[@]}"; do
    for seed in "${seed_list[@]}"; do
      for checkpoint in $CHECKPOINTS; do
        jobs+=("$tag:$arm:$seed:$checkpoint")
      done
    done
  done
done

run_worker() {
  local slot="$1"
  local index
  local failed=0
  for ((index = slot; index < ${#jobs[@]}; index += EVAL_WORKERS)); do
    IFS=: read -r tag arm seed checkpoint <<< "${jobs[$index]}"
    local invalid_marker="$RUN_ROOT/${arm}_seed${seed}/INVALID_FOR_DADE_CONCLUSIONS.json"
    if [[ -f "$invalid_marker" ]]; then
      printf 'SKIP_INVALID %s\n' "$invalid_marker"
      continue
    fi
    local checkpoint_path
    case "$checkpoint" in
      initial) checkpoint_path="$RUN_ROOT/${arm}_seed${seed}/model_initial.pt" ;;
      final) checkpoint_path="$RUN_ROOT/${arm}_seed${seed}/final.pt" ;;
      *) checkpoint_path="$RUN_ROOT/${arm}_seed${seed}/model_${checkpoint}.pt" ;;
    esac
    local ablation_suffix=""
    if [[ "$HISTORY_ABLATION" != "none" ]]; then
      ablation_suffix="_${HISTORY_ABLATION}"
    fi
    local output="$OUTPUT_ROOT/$tag/${arm}_seed${seed}/ck${checkpoint}_seed${SEED_START}_$((SEED_START + SEED_COUNT - 1))${ablation_suffix}"
    local log="$LOG_DIR/${tag}_${arm}_seed${seed}_ck${checkpoint}.log"
    local env_config="$DEFAULT_ENV"
    case "$tag" in
      c2) env_config="$C2_ENV" ;;
      c2_clean) env_config="$C2_CLEAN_ENV" ;;
      strict125) env_config="$STRICT_ENV" ;;
      c4) env_config="$C4_ENV" ;;
      h0) env_config="$H0_ENV" ;;
      h1) env_config="$H1_ENV" ;;
      h2) env_config="$H2_ENV" ;;
      h3) env_config="$H3_ENV" ;;
      h4) env_config="$H4_ENV" ;;
      default) env_config="$DEFAULT_ENV" ;;
      *)
        printf 'unknown evaluation tag: %s\n' "$tag" >&2
        exit 2
        ;;
    esac

    if [[ ! -f "$checkpoint_path" ]]; then
      printf 'MISSING %s\n' "$checkpoint_path"
      failed=1
      continue
    fi
    if [[ -f "$output/summary.json" ]]; then
      printf 'SKIP %s\n' "$output"
      continue
    fi

    local command=(
      "$PY" scripts/evaluate_pc_ppo.py
      --training-config "${CONFIGS[$arm]}"
      --checkpoint "$checkpoint_path"
      --environment-config "$env_config"
      --output "$output"
      --seed-start "$SEED_START"
      --seed-count "$SEED_COUNT"
      --device "$DEVICE"
      --dtype "$DTYPE"
      --history-ablation "$HISTORY_ABLATION"
    )
    if [[ "$DEVICE" == "cpu" ]]; then
      local cpu_start=$((CPU_BASE + slot * CPU_STRIDE))
      local cpu_end=$((cpu_start + CPU_STRIDE - 1))
      command=(taskset -c "${cpu_start}-${cpu_end}" nice -n 10 "${command[@]}")
    fi
    if [[ "$tag" == "c4" || "$arm" == *_c4 || "$arm" == *_c4_* ]]; then
      command+=(--allow-stage-override)
    fi
    printf 'EVAL tag=%s arm=%s seed=%s ck=%s output=%s\n' "$tag" "$arm" "$seed" "$checkpoint" "$output"
    "${command[@]}" > "$log" 2>&1
    local rc=$?
    printf 'END rc=%s %s\n' "$rc" "$output"
    if (( rc != 0 )); then
      failed=1
    fi
  done
  return "$failed"
}

worker_pids=()
for ((slot = 0; slot < EVAL_WORKERS; slot++)); do
  run_worker "$slot" &
  worker_pids+=("$!")
done
failures=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
printf 'ALL_EVALS_DONE output_root=%s\n' "$OUTPUT_ROOT"
if (( failures > 0 )); then
  printf 'EVAL_FAILURES workers=%s output_root=%s\n' "$failures" "$OUTPUT_ROOT" >&2
  exit 1
fi
