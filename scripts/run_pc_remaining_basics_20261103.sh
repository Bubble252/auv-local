#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_remaining_basics_20261103"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'remaining-basics queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

BASELINE_ROOT="$ROOT/artifacts/lowlevel/pc_server_remaining_baselines_20261103"
C2_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_20261103"
C2_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_eval_20261103"
C2_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_summary_20261103"
H3_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_20261103"
H3_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_eval_20261103"
H3_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_summary_20261103"

C2_BC="$ROOT/artifacts/lowlevel/pc_bc_c0_c1_c2_v3_20261103/pc_bc_actor.pt"
C3_BC="$ROOT/artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt"
C3_CONFIG="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
C2_CONFIG="$ROOT/configs/lowlevel/odarrl_control_pc_c2_alos.json"
C2_CLEAN_CONFIG="$ROOT/configs/lowlevel/odarrl_control_pc_c2_alos_clean.json"
H1_CONFIG="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json"
H2_CONFIG="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h2_speed_2p0_2p25.json"
H3_CONFIG="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h3_speed_2p25_2p4.json"
H4_CONFIG="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h4_speed_2p5_2p5.json"
C4_CONFIG="$ROOT/configs/lowlevel/odarrl_control_pc_c4_stress.json"

C2_SEEDS='91205 91206 91207'
H3_SEEDS='91430 91431 91432'
H3_ARM='frozen_bc_residual_nodade_h3_lr3e-4_scale1p0'

run_step() {
  local name="$1"
  shift
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  return 0
}

run_required_step() {
  local name="$1"
  shift
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  if (( rc != 0 )); then
    printf 'required step failed; stopping queue: %s rc=%s\n' "$name" "$rc"
    exit "$rc"
  fi
}

run_teacher() {
  local name="$1"
  local config="$2"
  local seed_count="$3"
  run_step "TEACHER_${name}" \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" scripts/evaluate_pc_teacher.py \
    --config "$config" \
    --output "$BASELINE_ROOT/teacher_${name}_seed0_$((seed_count - 1))" \
    --seed-start 0 --seed-count "$seed_count" \
    --device cpu --dtype float64
}

run_bc() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local seed_count="$4"
  run_step "BC_${name}" \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" scripts/evaluate_pc_bc.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output "$BASELINE_ROOT/bc_${name}_seed0_$((seed_count - 1))" \
    --seed-start 0 --seed-count "$seed_count" \
    --device cpu --dtype float64
}

printf '\n===== REMAINING PC BASICS QUEUE START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s root=%s\n' "$(hostname)" "$$" "$ROOT"
printf 'c2_seeds=%s h3_seeds=%s\n' "$C2_SEEDS" "$H3_SEEDS"

mkdir -p "$C2_TRAIN_ROOT" "$C2_EVAL_ROOT" "$C2_SUMMARY_ROOT"
mkdir -p "$H3_TRAIN_ROOT" "$H3_EVAL_ROOT" "$H3_SUMMARY_ROOT"
mkdir -p "$BASELINE_ROOT"

# C2 reproduces the stable 32-iteration local protocol from the same mixed
# C0/C1/C2 BC prior. 2048 environments fit on GPU0's current free slice.
printf '\n--- TRAIN_C2 START %s ---\n' "$(date -Is)"
env STAGE=c2_basics ARMS='c2_nodade c2_dade' SEEDS="$C2_SEEDS" ITERATIONS=32 \
  GPU_IDS='0' MIN_FREE_MIB=9000 GPU_WAIT_SECONDS=43200 \
  LOCK_DIR="$QUEUE_ROOT/c2_gpu_locks" \
  NUM_ENVS=2048 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  WARMSTART="$C2_BC" OUTPUT_ROOT="$C2_TRAIN_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_c2_basics_20261103/train" \
  bash scripts/run_pc_server_basics_train_20261103.sh &
C2_TRAIN_PID=$!

# H3 is the missing high-speed rung after H1/H2. It waits for a free 4090
# instead of sharing a borderline-memory GPU with another training process.
printf '\n--- TRAIN_H3 START %s ---\n' "$(date -Is)"
env STAGE=h3_basics ARMS="$H3_ARM" SEEDS="$H3_SEEDS" ITERATIONS=100 \
  GPU_IDS='1' MIN_FREE_MIB=16500 GPU_WAIT_SECONDS=43200 \
  LOCK_DIR="$ROOT/logs/pc_server_scale_C_20261103/train_gpu1_early_20261103/locks" \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$H3_TRAIN_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_h3_basics_20261103/train" \
  bash scripts/run_pc_server_basics_train_20261103.sh &
H3_TRAIN_PID=$!

# Baseline evaluations are CPU-only. Run them while the two GPU training
# groups are collecting rollouts.
run_teacher c2_mild "$C2_CONFIG" 64
run_teacher c2_clean "$C2_CLEAN_CONFIG" 64
run_teacher c3_realistic "$C3_CONFIG" 64
run_teacher h1 "$H1_CONFIG" 32
run_teacher h2 "$H2_CONFIG" 32
run_teacher h3 "$H3_CONFIG" 32
run_teacher h4 "$H4_CONFIG" 32
run_teacher c4 "$C4_CONFIG" 32

run_bc c2_mild "$C2_CONFIG" "$C2_BC" 64
run_bc c2_clean "$C2_CLEAN_CONFIG" "$C2_BC" 64
run_bc c3_realistic "$C3_CONFIG" "$C3_BC" 64
run_bc h3 "$H3_CONFIG" "$C3_BC" 32

wait "$C2_TRAIN_PID"
C2_TRAIN_RC=$?
printf '%s\n' "--- TRAIN_C2 END rc=$C2_TRAIN_RC $(date -Is) ---"
wait "$H3_TRAIN_PID"
H3_TRAIN_RC=$?
printf '%s\n' "--- TRAIN_H3 END rc=$H3_TRAIN_RC $(date -Is) ---"
if (( C2_TRAIN_RC != 0 || H3_TRAIN_RC != 0 )); then
  printf 'training failed; c2_rc=%s h3_rc=%s\n' "$C2_TRAIN_RC" "$H3_TRAIN_RC"
  exit 1
fi

run_required_step AUDIT_H3 "$PY" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/lowlevel/pc_server_h3_basics_20261103")
arm = "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0"
failures = []
rows = []
for seed in (91430, 91431, 91432):
    run = root / f"{arm}_seed{seed}"
    manifest_path = run / "training_manifest.json"
    if not manifest_path.is_file():
        failures.append(f"{run.name}: missing training_manifest.json")
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("frozen_bc_audit") or {}
    row = {
        "run": run.name,
        "status": manifest.get("status"),
        "initial_residual_abs_max": manifest.get("initial_residual_abs_max"),
        "frozen_bc_parameter_max_abs_drift": audit.get(
            "frozen_bc_parameter_max_abs_drift"
        ),
        "frozen_bc_parameters_require_grad": audit.get(
            "frozen_bc_parameters_require_grad"
        ),
    }
    rows.append(row)
    if row["status"] != "completed":
        failures.append(f"{run.name}: status={row['status']!r}")
    if float(row["initial_residual_abs_max"] or 0.0) > 1e-7:
        failures.append(f"{run.name}: nonzero initial residual")
    if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
        failures.append(f"{run.name}: frozen BC drift is not zero")
    if row["frozen_bc_parameters_require_grad"]:
        failures.append(f"{run.name}: frozen BC still requires grad")

output = root / "h3_basics_audit.json"
output.write_text(
    json.dumps({"runs": rows, "failures": failures}, indent=2, sort_keys=True)
    + "\n",
    encoding="utf-8",
)
print(output)
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
PY

COMMON_EVAL_ENV=(
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4
  CPU_BASE=160 CPU_STRIDE=16
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0
)

run_required_step EVAL_C2 env \
  STAGE=c2_basics ARMS='c2_nodade c2_dade' SEEDS="$C2_SEEDS" \
  TAGS='c2 c2_clean' CHECKPOINTS='initial 0 10 20 30 final' \
  "${COMMON_EVAL_ENV[@]}" SEED_COUNT=64 \
  RUN_ROOT="$C2_TRAIN_ROOT" OUTPUT_ROOT="$C2_EVAL_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_c2_basics_eval_20261103" \
  bash scripts/eval_pc_server_basics_20261103.sh

run_required_step SUMMARY_C2 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$C2_EVAL_ROOT" \
  --output-dir "$C2_SUMMARY_ROOT" \
  --baseline-arm c2_nodade \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 10=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 30=initial \
  --checkpoint-baseline final=initial

run_required_step EVAL_H3_INDIST env \
  STAGE=h3_basics ARMS="$H3_ARM" SEEDS="$H3_SEEDS" \
  TAGS='h3' CHECKPOINTS='initial 0 20 40 60 80 99' \
  "${COMMON_EVAL_ENV[@]}" SEED_COUNT=32 \
  RUN_ROOT="$H3_TRAIN_ROOT" OUTPUT_ROOT="$H3_EVAL_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_h3_basics_eval_20261103/h3" \
  bash scripts/eval_pc_server_basics_20261103.sh

run_required_step EVAL_H3_CROSS env \
  STAGE=h3_basics ARMS="$H3_ARM" SEEDS="$H3_SEEDS" \
  TAGS='default strict125 h4' CHECKPOINTS='initial 99' \
  "${COMMON_EVAL_ENV[@]}" SEED_COUNT=32 \
  RUN_ROOT="$H3_TRAIN_ROOT" OUTPUT_ROOT="$H3_EVAL_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_h3_basics_eval_20261103/cross" \
  bash scripts/eval_pc_server_basics_20261103.sh

run_required_step SUMMARY_H3 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$H3_EVAL_ROOT" \
  --output-dir "$H3_SUMMARY_ROOT" \
  --baseline-arm "$H3_ARM" \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 40=initial \
  --checkpoint-baseline 60=initial \
  --checkpoint-baseline 80=initial \
  --checkpoint-baseline 99=initial

printf '\n===== REMAINING PC BASICS QUEUE DONE %s =====\n' "$(date -Is)"
