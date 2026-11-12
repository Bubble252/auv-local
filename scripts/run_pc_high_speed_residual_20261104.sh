#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_high_speed_residual_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'high-speed residual queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

TRAIN_SEEDS="${TRAIN_SEEDS:-94030 94031 94032 94033 94034}"
STAGES="${STAGES:-h1 h2 h3}"
RESIDUAL_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104"
RESIDUAL_ARM='frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0'
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_high_speed_residual_ppo_20261104"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_high_speed_residual_eval_20261104"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_high_speed_residual_summary_20261104"
REPORT_ROOT="$ROOT/artifacts/lowlevel/pc_high_speed_residual_report_20261104"
CALIBRATION="$ROOT/artifacts/lowlevel/pc_high_speed_calibration_v2_20261104/calibration.json"
BASELINE_ROOT="$ROOT/artifacts/lowlevel/pc_server_remaining_baselines_20261103"

ITERATIONS="${ITERATIONS:-100}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_STEPS="${NUM_STEPS:-256}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-128}"
GPU_IDS="${GPU_IDS:-1 2 4 5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-14000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-28800}"
CHECKPOINT_WAIT_SECONDS="${CHECKPOINT_WAIT_SECONDS:-21600}"
SEED_COUNT="${SEED_COUNT:-32}"

declare -A CONFIG_ARM=(
  [h1]='frozen_bc_residual_nodade_h1_lr3e-4_scale1p0'
  [h2]='frozen_bc_residual_nodade_h2_lr3e-4_scale1p0'
  [h3]='frozen_bc_residual_nodade_h3_lr3e-4_scale1p0'
)

read -r -a seed_list <<< "$TRAIN_SEEDS"
read -r -a stage_list <<< "$STAGES"
arm_list=()
for stage in "${stage_list[@]}"; do
  arm="${CONFIG_ARM[$stage]:-}"
  if [[ -z "$arm" ]]; then
    printf 'unknown high-speed stage: %s\n' "$stage" >&2
    exit 2
  fi
  arm_list+=("$arm")
done

printf '\n===== PC HIGH-SPEED RESIDUAL QUEUE START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s stages=%s seeds=%s\n' \
  "$(hostname)" "$$" "$STAGES" "$TRAIN_SEEDS"

wait_for_residual_checkpoints() {
  local deadline=$((SECONDS + CHECKPOINT_WAIT_SECONDS))
  while true; do
    local missing=()
    for seed in "${seed_list[@]}"; do
      local run="$RESIDUAL_ROOT/${RESIDUAL_ARM}_seed${seed}"
      if [[ ! -f "$run/final.pt" || ! -f "$run/training_result.json" ]]; then
        missing+=("$seed")
      fi
    done
    if (( ${#missing[@]} == 0 )); then
      printf 'RESIDUAL_CHECKPOINTS_READY seeds=%s %s\n' "$TRAIN_SEEDS" "$(date -Is)"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      printf 'RESIDUAL_CHECKPOINT_TIMEOUT missing=%s root=%s\n' \
        "${missing[*]}" "$RESIDUAL_ROOT" >&2
      return 1
    fi
    printf 'WAIT_RESIDUAL_CHECKPOINTS missing=%s %s\n' "${missing[*]}" "$(date -Is)"
    sleep 30
  done
}

run_required_step() {
  local name="$1"
  local log="$2"
  shift 2
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@" > "$log" 2>&1
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  if (( rc != 0 )); then
    printf 'FAILED %s log=%s\n' "$name" "$log" >&2
    tail -n 80 "$log" >&2 || true
  fi
  return "$rc"
}

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "$REPORT_ROOT" "$QUEUE_ROOT/train" "$QUEUE_ROOT/eval"

if ! wait_for_residual_checkpoints; then
  exit 1
fi

ARMS="${arm_list[*]}"
if ! env STAGE=high_speed_residual_20261104 ARMS="$ARMS" SEEDS="$TRAIN_SEEDS" \
  ITERATIONS="$ITERATIONS" GPU_IDS="$GPU_IDS" MIN_FREE_MIB="$MIN_FREE_MIB" \
  GPU_WAIT_SECONDS="$GPU_WAIT_SECONDS" LOCK_DIR="$QUEUE_ROOT/gpu_locks" \
  NUM_ENVS="$NUM_ENVS" NUM_STEPS="$NUM_STEPS" NUM_MINI_BATCHES="$NUM_MINI_BATCHES" \
  RESIDUAL_WARMSTART_ROOT="$RESIDUAL_ROOT" RESIDUAL_WARMSTART_ARM="$RESIDUAL_ARM" \
  OUTPUT_ROOT="$TRAIN_ROOT" LOG_DIR="$QUEUE_ROOT/train" \
  bash scripts/run_pc_server_basics_train_20261103.sh; then
  printf 'HIGH_SPEED_TRAINING_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! run_required_step AUDIT_HIGH_SPEED_RESIDUAL "$QUEUE_ROOT/audit.log" \
  env TRAIN_ROOT="$TRAIN_ROOT" TRAIN_SEEDS="$TRAIN_SEEDS" STAGES="$STAGES" "$PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["TRAIN_ROOT"])
seeds = [int(value) for value in os.environ["TRAIN_SEEDS"].split()]
arms = {
    "h1": "frozen_bc_residual_nodade_h1_lr3e-4_scale1p0",
    "h2": "frozen_bc_residual_nodade_h2_lr3e-4_scale1p0",
    "h3": "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0",
}
failures = []
rows = []
for stage in os.environ["STAGES"].split():
    for seed in seeds:
        run = root / f"{arms[stage]}_seed{seed}"
        manifest_path = run / "training_manifest.json"
        checkpoint_path = run / "final.pt"
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            failures.append(f"missing run: {run}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = manifest.get("frozen_bc_audit") or {}
        warmstart = manifest.get("residual_warmstart_metadata") or {}
        row = {
            "stage": stage,
            "seed": seed,
            "status": manifest.get("status"),
            "initial_residual_abs_max": manifest.get("initial_residual_abs_max"),
            "frozen_bc_parameter_max_abs_drift": audit.get(
                "frozen_bc_parameter_max_abs_drift"
            ),
            "frozen_bc_parameters_require_grad": audit.get(
                "frozen_bc_parameters_require_grad"
            ),
            "warmstart_checkpoint": warmstart.get("checkpoint_path"),
            "warmstart_sha256": warmstart.get("checkpoint_sha256"),
            "final_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        }
        rows.append(row)
        if row["status"] != "completed":
            failures.append(f"{run.name}: status={row['status']!r}")
        if float(row["initial_residual_abs_max"] or 0.0) <= 1e-6:
            failures.append(f"{run.name}: residual warm-start is zero or missing")
        if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
            failures.append(f"{run.name}: frozen BC drift is not zero")
        if row["frozen_bc_parameters_require_grad"]:
            failures.append(f"{run.name}: frozen BC still requires grad")
        expected = f"_seed{seed}/final.pt"
        if not str(row["warmstart_checkpoint"] or "").endswith(expected):
            failures.append(f"{run.name}: unexpected residual warm-start {row['warmstart_checkpoint']!r}")

payload = {"runs": rows, "failures": failures}
output = root / "audit.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(output)
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
PY
then
  exit 1
fi

COMMON_EVAL_ENV=(
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4
  CPU_BASE=128 CPU_STRIDE=16
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0 SEED_COUNT="$SEED_COUNT"
)

if ! env STAGE=high_speed_residual_eval_20261104 ARMS="$ARMS" SEEDS="$TRAIN_SEEDS" \
  TAGS='h1 h2 h3' CHECKPOINTS='initial final' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'HIGH_SPEED_EVALUATION_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! env STAGE=high_speed_residual_h4_eval_20261104 ARMS="${CONFIG_ARM[h3]}" \
  SEEDS="$TRAIN_SEEDS" TAGS='h4' CHECKPOINTS='final' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'HIGH_SPEED_H4_PRESSURE_EVAL_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! run_required_step SUMMARY_HIGH_SPEED_RESIDUAL "$QUEUE_ROOT/summary.log" \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$EVAL_ROOT" --output-dir "$SUMMARY_ROOT" \
  --baseline-arm "${CONFIG_ARM[h1]}" \
  --checkpoint initial --checkpoint final \
  --checkpoint-baseline final=initial; then
  exit 1
fi

gate_rc=0
if ! "$PY" scripts/summarize_pc_high_speed_20261104.py \
  --calibration "$CALIBRATION" \
  --baseline-root "$BASELINE_ROOT" \
  --residual-summary "$SUMMARY_ROOT/summary.json" \
  --output-dir "$REPORT_ROOT" \
  --seed-start 0 --seed-count "$SEED_COUNT" \
  --require-pass > "$QUEUE_ROOT/high_speed_report.log" 2>&1; then
  gate_rc=1
  printf 'HIGH_SPEED_GATE_NOT_PASSED; preserving report as evidence\n' >&2
fi

env REPORT_ROOT="$REPORT_ROOT" SUMMARY_ROOT="$SUMMARY_ROOT" TRAIN_ROOT="$TRAIN_ROOT" \
  EVAL_ROOT="$EVAL_ROOT" GATE_RC="$gate_rc" "$PY" - <<'PY'
import json
import os
from pathlib import Path

report_root = Path(os.environ["REPORT_ROOT"])
payload = {
    "training_audit": str(Path(os.environ["TRAIN_ROOT"]) / "audit.json"),
    "eval_summary": str(Path(os.environ["SUMMARY_ROOT"]) / "summary.json"),
    "eval_root": os.environ["EVAL_ROOT"],
    "high_speed_report": str(report_root / "high_speed_report.json"),
    "high_speed_report_markdown": str(report_root / "high_speed_report.md"),
    "gate_returncode": int(os.environ["GATE_RC"]),
}
(report_root / "index.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(report_root / "index.json")
PY

printf '\n===== PC HIGH-SPEED RESIDUAL QUEUE DONE %s gate_rc=%s =====\n' \
  "$(date -Is)" "$gate_rc"
exit 0
