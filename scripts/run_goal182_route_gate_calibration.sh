#!/usr/bin/env bash
# Run the minimal C0-C6 frozen-reference calibration sweep and freeze PE2E/ALOSB tables.
set -euo pipefail

W="${WORKTREE:-/data/yhx/underwater-robotics/worktrees/auv-local/goal182/20261117-goal182-route-gate-calibration}"
R="${RUN_ROOT:-/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-route-gate-calibration}"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
CORE="${AUV_CORE_SOURCE:-/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src}"
STAGES="${STAGES:-C0 C1 C2 C3 C4 C5 C6}"
STAGE_ITERATIONS="${STAGE_ITERATIONS:-40}"
REQUIRED_STAGES="${REQUIRED_STAGES:-$(printf '%s' "$STAGES" | tr ' ' ',')}"
MIN_FREE_MIB="${MIN_FREE_MIB:-23000}"
PE2E_GPU="${PE2E_GPU:-6}"
ALOSB_GPU="${ALOSB_GPU:-7}"
THRESHOLDS_DIR="${THRESHOLDS_DIR:-$W/configs/governance/route_gate}"
CONFIG_ROOT="configs/lowlevel/goal182/calibration"

export PYTHONPATH="$W/src:$CORE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$R/logs" "$R/thresholds" "$THRESHOLDS_DIR"

check_gpu() {
  local gpu=$1 total used free
  read -r total used <<<"$(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | tr ',' ' ')"
  free=$((total - used))
  if (( free < MIN_FREE_MIB )); then
    printf 'GPU %s has %s MiB free, below required %s MiB\n' "$gpu" "$free" "$MIN_FREE_MIB" >&2
    return 1
  fi
  printf 'GPU_CHECK\t%s\tfree=%s\trequired=%s\n' "$gpu" "$free" "$MIN_FREE_MIB"
}

check_gpu "$PE2E_GPU"
check_gpu "$ALOSB_GPU"

run_one() {
  local control=$1 gpu=$2 stage=$3 config=$4
  local name="reference_${control}_${stage}"
  local out="$R/$name"
  local log="$R/logs/$name.log"
  if [ -e "$out" ]; then
    printf 'REFUSING_EXISTING_OUTPUT\t%s\n' "$out" >&2
    return 2
  fi
  printf 'START\t%s\tgpu=%s\tconfig=%s\n' "$name" "$gpu" "$config"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PY" "$W/scripts/train_batch_lowlevel_ppo.py" \
      --training-config "$W/$config" \
      --iterations "$STAGE_ITERATIONS" \
      --curriculum-route-gate-mode shadow \
      --curriculum-eval-interval 1000000000000 \
      --output "$out" \
      --seed 86201
  ) >"$log" 2>&1
  printf 'DONE\t%s\n' "$name"
}

run_control() {
  local control=$1 gpu=$2
  local stage
  for stage in $STAGES; do
    run_one "$control" "$gpu" "$stage" "$CONFIG_ROOT/reference_${control}_${stage}.json"
  done
}

run_control pe2e "$PE2E_GPU" &
pe2e_pid=$!
run_control alosb "$ALOSB_GPU" &
alosb_pid=$!
pe2e_status=0
alosb_status=0
wait "$pe2e_pid" || pe2e_status=$?
wait "$alosb_pid" || alosb_status=$?
if (( pe2e_status != 0 || alosb_status != 0 )); then
  printf 'CALIBRATION_SWEEP_FAILED\tpe2e=%s\talosb=%s\n' "$pe2e_status" "$alosb_status" >&2
  exit 1
fi

for control in pe2e alosb; do
  for stage in $STAGES; do
    manifest="$R/reference_${control}_${stage}/training_manifest.json"
    "$PY" - "$manifest" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "completed":
    raise SystemExit(f"calibration run is not completed: {payload.get('status')}")
PY
  done
done

for control in pe2e alosb; do
  case "$control" in
    pe2e) contract=PE2E ;;
    alosb) contract=ALOSB ;;
  esac
  calibration_config="$CONFIG_ROOT/reference_${control}_C0.json"
  reference_sha=$("$PY" - "$W/$calibration_config" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
value = payload.get("base_policy", {}).get("checkpoint_sha256")
if not isinstance(value, str) or not value:
    raise SystemExit("calibration config is missing base_policy.checkpoint_sha256")
print(value)
PY
  )
  records_args=()
  for stage in $STAGES; do
    records_args+=(--records "$R/reference_${control}_${stage}/iteration_records.jsonl")
  done
  "$PY" "$W/scripts/calibrate_goal182_route_gate.py" \
    "${records_args[@]}" \
    --control-contract "$contract" \
    --calibration-source REFERENCE_BC \
    --reference-id "frozen_bc_${control}_stage_sweep" \
    --reference-sha256 "$reference_sha" \
    --require-stages "$REQUIRED_STAGES" \
    --min-stage-records 20 \
    --output "$R/thresholds/goal182_route_gate_${control}_20_3.json"
done

for control in pe2e alosb; do
  source_table="$R/thresholds/goal182_route_gate_${control}_20_3.thresholds.json"
  target_table="$THRESHOLDS_DIR/goal182_route_gate_${control}_20_3.thresholds.json"
  cp "$source_table" "$target_table"
  printf 'THRESHOLDS_INSTALLED\t%s\n' "$target_table"
done

printf 'CALIBRATION_COMPLETE\t%s\n' "$R/thresholds"
printf 'THRESHOLDS_DIR\t%s\n' "$THRESHOLDS_DIR"
