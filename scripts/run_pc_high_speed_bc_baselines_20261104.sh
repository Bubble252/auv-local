#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_high_speed_bc_baselines_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'high-speed BC queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

BASELINE_ROOT="$ROOT/artifacts/lowlevel/pc_server_remaining_baselines_20261103"
BC_CHECKPOINT="$ROOT/artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt"
SEED_START=0
SEED_COUNT=32

declare -A CONFIG=(
  [h0]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h0_speed_1p0_1p8.json"
  [h1]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json"
  [h2]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h2_speed_2p0_2p25.json"
  [h4]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h4_speed_2p5_2p5.json"
)
declare -A CPU_RANGE=(
  [h0]="368-375"
  [h1]="368-375"
  [h2]="376-383"
  [h4]="368-383"
)
PARALLEL_STAGES=(h1 h2)
FINAL_STAGES=(h4 h0)

printf '\n===== PC HIGH-SPEED BC BASELINES START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s root=%s seeds=%s..%s\n' \
  "$(hostname)" "$$" "$ROOT" "$SEED_START" "$((SEED_START + SEED_COUNT - 1))"
if [[ ! -f "$BC_CHECKPOINT" ]]; then
  printf 'MISSING_CHECKPOINT %s\n' "$BC_CHECKPOINT" >&2
  exit 2
fi

failures=0
for phase in parallel final; do
  declare -a pids=()
  declare -a names=()
  if [[ "$phase" == "parallel" ]]; then
    stages=("${PARALLEL_STAGES[@]}")
  else
    stages=("${FINAL_STAGES[@]}")
  fi
  for stage in "${stages[@]}"; do
    output="$BASELINE_ROOT/bc_${stage}_seed${SEED_START}_$((SEED_START + SEED_COUNT - 1))"
    log="$QUEUE_ROOT/bc_${stage}.log"
    if [[ -f "$output/summary.json" ]]; then
      printf 'SKIP stage=%s output=%s\n' "$stage" "$output"
      continue
    fi
    printf 'START phase=%s stage=%s cpu=%s config=%s output=%s\n' \
      "$phase" "$stage" "${CPU_RANGE[$stage]}" "${CONFIG[$stage]}" "$output"
    taskset -c "${CPU_RANGE[$stage]}" nice -n 10 \
      env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      "$PY" scripts/evaluate_pc_bc.py \
      --config "${CONFIG[$stage]}" \
      --checkpoint "$BC_CHECKPOINT" \
      --output "$output" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --device cpu --dtype float64 > "$log" 2>&1 &
    pids+=("$!")
    names+=("$stage")
  done
  for index in "${!pids[@]}"; do
    stage="${names[$index]}"
    if wait "${pids[$index]}"; then
      printf 'END stage=%s rc=0 %s\n' "$stage" "$(date -Is)"
    else
      rc=$?
      printf 'END stage=%s rc=%s log=%s %s\n' \
        "$stage" "$rc" "$QUEUE_ROOT/bc_${stage}.log" "$(date -Is)"
      failures=$((failures + 1))
    fi
  done
done

printf 'PC_HIGH_SPEED_BC_DONE failures=%s %s\n' "$failures" "$(date -Is)"
if (( failures > 0 )); then
  exit 1
fi
