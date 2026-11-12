#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/artifacts/lowlevel/pc_alos_r_20261104}"
EVAL_ROOT="$ARTIFACT_ROOT/eval"
REPORT_ROOT="$ARTIFACT_ROOT/report"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/pc_alos_r_20261104}"
mkdir -p "$EVAL_ROOT" "$REPORT_ROOT" "$LOG_ROOT"
exec 9> "$LOG_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'ALOS-R queue already running: %s\n' "$LOG_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$LOG_ROOT/queue.log" 2>&1

POLICY_CONFIG="${POLICY_CONFIG:-$ROOT/configs/lowlevel/odarrl_control_ppo_pc_multienv_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json}"
POLICY_ARM="${POLICY_ARM:-frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104}"
SEEDS="${SEEDS:-94030}"
STAGES="${STAGES:-R1}"
TAGS="${TAGS:-default strict125 h1 h3 c4}"
SEED_START="${SEED_START:-0}"
SEED_COUNT="${SEED_COUNT:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
CPU_BASE="${CPU_BASE:-352}"
CPU_STRIDE="${CPU_STRIDE:-8}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-$ROOT/artifacts/lowlevel/pc_multienv_nodade_5seed_summary_20261104/summary.json}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
if (( CPU_BASE < 0 || CPU_STRIDE <= 0 || EVAL_WORKERS <= 0 )); then
  printf 'CPU_BASE must be non-negative, CPU_STRIDE and EVAL_WORKERS must be positive\n' >&2
  exit 2
fi
if (( CPU_BASE + EVAL_WORKERS * CPU_STRIDE > CPU_COUNT )); then
  printf 'CPU affinity window %s-%s exceeds available CPUs 0-%s\n' \
    "$CPU_BASE" "$((CPU_BASE + EVAL_WORKERS * CPU_STRIDE - 1))" "$((CPU_COUNT - 1))" >&2
  exit 2
fi

declare -A GUIDANCE_CONFIG=(
  [R0]=""
  [R1]="$ROOT/configs/lowlevel/odarrl_control_alos_r1_terminal.json"
  [R1B]="$ROOT/configs/lowlevel/odarrl_control_alos_r1b_legacy20_floor75.json"
  [R1C]="$ROOT/configs/lowlevel/odarrl_control_alos_r1c_ground_floor75.json"
  [R1D]="$ROOT/configs/lowlevel/odarrl_control_alos_r1d_ground_floor_crab75.json"
  [R1E]="$ROOT/configs/lowlevel/odarrl_control_alos_r1e_terminal20_prespin75.json"
  [R1F]="$ROOT/configs/lowlevel/odarrl_control_alos_r1f_terminal20_noprespin75.json"
  [R1G]="$ROOT/configs/lowlevel/odarrl_control_alos_r1g_ground_floor_init110_nospin.json"
  [R1H]="$ROOT/configs/lowlevel/odarrl_control_alos_r1h_ground_floor_init110_prespin75.json"
  [R1I]="$ROOT/configs/lowlevel/odarrl_control_alos_r1i_noground_init110_nospin.json"
  [R1J]="$ROOT/configs/lowlevel/odarrl_control_alos_r1j_noground_init110_prespin75.json"
  [R1K]="$ROOT/configs/lowlevel/odarrl_control_alos_r1k_ground_floor_init100_nospin.json"
  [R4A]="$ROOT/configs/lowlevel/odarrl_control_alos_r4a_launch_init125_nospin.json"
  [R4B]="$ROOT/configs/lowlevel/odarrl_control_alos_r4b_launch_init125_prespin75.json"
  [R4C]="$ROOT/configs/lowlevel/odarrl_control_alos_r4c_launch_init125_prespin100.json"
  [R2B]="$ROOT/configs/lowlevel/odarrl_control_alos_r2b_crab_floor_init100.json"
  [R3B]="$ROOT/configs/lowlevel/odarrl_control_alos_r3b_governor_floor_init100.json"
  [R2]="$ROOT/configs/lowlevel/odarrl_control_alos_r2_crab.json"
  [R3]="$ROOT/configs/lowlevel/odarrl_control_alos_r3_governor.json"
)
declare -A ENV_CONFIG=(
  [default]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
  [strict125]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json"
  [h1]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json"
  [h2]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h2_speed_2p0_2p25.json"
  [h3]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h3_speed_2p25_2p4.json"
  [h4]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h4_speed_2p5_2p5.json"
  [c4]="$ROOT/configs/lowlevel/odarrl_control_pc_c4_stress.json"
)

if [[ ! -f "$POLICY_CONFIG" ]]; then
  printf 'missing policy training config: %s\n' "$POLICY_CONFIG" >&2
  exit 2
fi
read -r -a stage_list <<< "$STAGES"
read -r -a seed_list <<< "$SEEDS"
read -r -a tag_list <<< "$TAGS"

jobs=()
for stage in "${stage_list[@]}"; do
  if [[ -z "${GUIDANCE_CONFIG[$stage]+configured}" ]]; then
    printf 'unknown ALOS-R stage: %s\n' "$stage" >&2
    exit 2
  fi
  for tag in "${tag_list[@]}"; do
    if [[ -z "${ENV_CONFIG[$tag]+configured}" ]]; then
      printf 'unknown evaluation tag: %s\n' "$tag" >&2
      exit 2
    fi
    for seed in "${seed_list[@]}"; do
      jobs+=("$stage:$tag:$seed")
    done
  done
done

run_worker() {
  local slot="$1"
  local index
  local failed=0
  for ((index = slot; index < ${#jobs[@]}; index += EVAL_WORKERS)); do
    IFS=: read -r stage tag seed <<< "${jobs[$index]}"
    local checkpoint="$TRAIN_ROOT/${POLICY_ARM}_seed${seed}/final.pt"
    local output="$EVAL_ROOT/$stage/$tag/${POLICY_ARM}_seed${seed}/ckfinal_seed${SEED_START}_$((SEED_START + SEED_COUNT - 1))"
    local log="$LOG_ROOT/${stage}_${tag}_seed${seed}.log"
    if [[ ! -f "$checkpoint" ]]; then
      printf 'MISSING_CHECKPOINT %s\n' "$checkpoint" >&2
      failed=1
      continue
    fi
    if [[ -f "$output/summary.json" ]]; then
      printf 'SKIP stage=%s tag=%s seed=%s output=%s\n' "$stage" "$tag" "$seed" "$output"
      continue
    fi
    local command=(
      "$PY" scripts/evaluate_pc_ppo.py
      --training-config "$POLICY_CONFIG"
      --checkpoint "$checkpoint"
      --environment-config "${ENV_CONFIG[$tag]}"
      --output "$output"
      --seed-start "$SEED_START"
      --seed-count "$SEED_COUNT"
      --device cpu
      --dtype float64
    )
    if [[ -n "${GUIDANCE_CONFIG[$stage]}" ]]; then
      command+=(--guidance-config "${GUIDANCE_CONFIG[$stage]}")
    fi
    if [[ "$tag" == "c4" || "$tag" == "h4" ]]; then
      command+=(--allow-stage-override)
    fi
    local cpu_start=$((CPU_BASE + slot * CPU_STRIDE))
    local cpu_end=$((cpu_start + CPU_STRIDE - 1))
    printf 'START stage=%s tag=%s seed=%s cpu=%s-%s output=%s\n' \
      "$stage" "$tag" "$seed" "$cpu_start" "$cpu_end" "$output"
    taskset -c "${cpu_start}-${cpu_end}" nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      "${command[@]}" > "$log" 2>&1
    local rc=$?
    printf 'END rc=%s stage=%s tag=%s seed=%s output=%s\n' "$rc" "$stage" "$tag" "$seed" "$output"
    if (( rc != 0 )); then
      failed=1
      tail -n 80 "$log" >&2 || true
    fi
  done
  return "$failed"
}

printf '\n===== PC ALOS-R START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s stages=%s seeds=%s tags=%s eval_seeds=%s\n' \
  "$(hostname)" "$$" "$STAGES" "$SEEDS" "$TAGS" "$SEED_COUNT"

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
if (( failures > 0 )); then
  printf 'ALOS_R_EVAL_FAILURES=%s root=%s\n' "$failures" "$EVAL_ROOT" >&2
  exit 1
fi

for stage in "${stage_list[@]}"; do
  printf 'SUMMARIZE stage=%s %s\n' "$stage" "$(date -Is)"
  summary_baseline=()
  if [[ "$stage" != "R0" && -f "$REPORT_ROOT/R0/summary.json" ]]; then
    summary_baseline=(--baseline-summary "$REPORT_ROOT/R0/summary.json")
  elif [[ "$stage" != "R0" && -f "$BASELINE_SUMMARY" ]]; then
    summary_baseline=(--baseline-summary "$BASELINE_SUMMARY")
  fi
  "$PY" scripts/summarize_pc_alos_r.py \
    --eval-root "$EVAL_ROOT/$stage" \
    --stage "$stage" \
    --output "$REPORT_ROOT/$stage" \
    "${summary_baseline[@]}"
done
printf 'ALOS_R_DONE %s root=%s\n' "$(date -Is)" "$ARTIFACT_ROOT"
