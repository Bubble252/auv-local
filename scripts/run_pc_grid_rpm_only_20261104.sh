#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="${QUEUE_ROOT:-$ROOT/logs/pc_grid_rpm_only_20261104}"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'PC RPM-only grid queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

TREATMENT_ARM='frozen_bc_residual_nodade_grid_rpm_only_lr3e-4_scale1p0'
REFERENCE_ARM='frozen_bc_residual_nodade_grid_gpu_lr3e-4_scale1p0'
CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_nodade_rpm_only_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"

SEEDS="${SEEDS:-94030 94031 94032 94033 94034}"
SEED_COUNT="${SEED_COUNT:-32}"
TAGS="${TAGS:-grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
CPU_BASE="${CPU_BASE:-288}"
CPU_STRIDE="${CPU_STRIDE:-12}"
GPU_LIST="${GPU_LIST:-2 4 5}"
PRETRAIN_DEVICE="${PRETRAIN_DEVICE:-cuda}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4096}"
PRETRAIN_MAX_EPOCHS="${PRETRAIN_MAX_EPOCHS:-200}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-30}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-1e-5}"
PRETRAIN_OUTPUT_L2="${PRETRAIN_OUTPUT_L2:-1e-4}"
PRETRAIN_VALIDATION_FRACTION="${PRETRAIN_VALIDATION_FRACTION:-0.2}"

DEMO_ROOT="${DEMO_ROOT:-$ROOT/artifacts/lowlevel/pc_speed_current_grid_teacher_demos_20261104}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT/artifacts/lowlevel/pc_grid_rpm_only_supervised_residual_20261104}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/artifacts/lowlevel/pc_grid_rpm_only_eval_20261104}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$ROOT/artifacts/lowlevel/pc_grid_rpm_only_summary_20261104}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/artifacts/lowlevel/pc_grid_rpm_only_report_20261104}"
REFERENCE_RUN_ROOT="${REFERENCE_RUN_ROOT:-$ROOT/artifacts/lowlevel/pc_dade_grid_nodade_supervised_residual_20261104}"
REFERENCE_EVAL_ROOT="${REFERENCE_EVAL_ROOT:-$ROOT/artifacts/lowlevel/pc_grid_rpm_only_reference_eval_20261104}"

read -r -a seed_list <<< "$SEEDS"
read -r -a tag_list <<< "$TAGS"
read -r -a gpu_list <<< "$GPU_LIST"
if (( ${#seed_list[@]} == 0 || ${#tag_list[@]} == 0 || ${#gpu_list[@]} == 0 )); then
  printf 'SEEDS, TAGS and GPU_LIST must be non-empty\n' >&2
  exit 2
fi

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "$REPORT_ROOT"

run_required_step() {
  local name="$1"
  local log="$2"
  shift 2
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@" > "$log" 2>&1
  local rc=$?
  printf '%s--- %s END rc=%s %s ---\n' "" "$name" "$rc" "$(date -Is)"
  if (( rc != 0 )); then
    printf 'FAILED %s log=%s\n' "$name" "$log" >&2
    tail -n 100 "$log" >&2 || true
  fi
  return "$rc"
}

if [[ ! -f "$DEMO_ROOT/merged/demos.npz" || ! -f "$DEMO_ROOT/merged/demos.manifest.json" ]]; then
  printf 'missing merged grid demos: %s/merged\n' "$DEMO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$ROOT/artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt" ]]; then
  printf 'missing frozen BC checkpoint\n' >&2
  exit 1
fi
if [[ ! -d "$REFERENCE_RUN_ROOT" ]]; then
  printf 'missing reference training root: %s\n' "$REFERENCE_RUN_ROOT" >&2
  exit 1
fi

run_pretrain_worker() {
  local slot="$1"
  local gpu="${gpu_list[$slot]}"
  local cpu_start=$((96 + slot * 8))
  local cpu_range="${cpu_start}-$((cpu_start + 7))"
  local failed=0
  local index
  for ((index = slot; index < ${#seed_list[@]}; index += ${#gpu_list[@]})); do
    local seed="${seed_list[$index]}"
    local output="$TRAIN_ROOT/${TREATMENT_ARM}_seed${seed}"
    local log="$QUEUE_ROOT/pretrain_seed${seed}.log"
    if [[ -f "$output/final.pt" && -f "$output/training_result.json" ]]; then
      printf 'SKIP_PRETRAIN seed=%s output=%s\n' "$seed" "$output"
      continue
    fi
    printf 'START_PRETRAIN seed=%s gpu=%s cpu=%s output=%s\n' \
      "$seed" "$gpu" "$cpu_range" "$output"
    taskset -c "$cpu_range" env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PY" scripts/pretrain_pc_residual.py \
        --training-config "$CONFIG" \
        --demos "$DEMO_ROOT/merged/demos.npz" \
        --output "$output" \
        --batch-size "$PRETRAIN_BATCH_SIZE" \
        --max-epochs "$PRETRAIN_MAX_EPOCHS" \
        --patience "$PRETRAIN_PATIENCE" \
        --learning-rate "$PRETRAIN_LR" \
        --weight-decay "$PRETRAIN_WEIGHT_DECAY" \
        --output-l2 "$PRETRAIN_OUTPUT_L2" \
        --validation-fraction "$PRETRAIN_VALIDATION_FRACTION" \
        --seed "$seed" \
        --device "$PRETRAIN_DEVICE" \
        --overwrite > "$log" 2>&1
    local rc=$?
    printf 'END_PRETRAIN rc=%s seed=%s gpu=%s\n' "$rc" "$seed" "$gpu"
    if (( rc != 0 )); then
      failed=1
      tail -n 80 "$log" >&2 || true
    fi
  done
  return "$failed"
}

printf '\n===== PC RPM-ONLY GRID START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s seeds=%s tags=%s eval_seeds=%s gpus=%s device=%s\n' \
  "$(hostname)" "$$" "$SEEDS" "$TAGS" "$SEED_COUNT" "$GPU_LIST" "$PRETRAIN_DEVICE"

worker_pids=()
for ((slot = 0; slot < ${#gpu_list[@]}; slot++)); do
  run_pretrain_worker "$slot" &
  worker_pids+=("$!")
done
pretrain_failures=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    pretrain_failures=$((pretrain_failures + 1))
  fi
done
if (( pretrain_failures > 0 )); then
  printf 'PRETRAIN_FAILURES=%s\n' "$pretrain_failures" >&2
  exit 1
fi

if ! run_required_step AUDIT_RPM_ONLY "$QUEUE_ROOT/audit.log" \
  env TRAIN_ROOT="$TRAIN_ROOT" TREATMENT_ARM="$TREATMENT_ARM" SEEDS="$SEEDS" \
  PRETRAIN_DEVICE="$PRETRAIN_DEVICE" "$PY" - <<'PY'
import json
import os
from pathlib import Path

seeds = [int(value) for value in os.environ["SEEDS"].split()]
expected_cells = {"pc_grid_low_weak", "pc_grid_low_strong", "pc_grid_high_weak"}
forbidden_cells = {"pc_grid_high_strong"}
root = Path(os.environ["TRAIN_ROOT"])
arm = os.environ["TREATMENT_ARM"]
failures = []
rows = []
for seed in seeds:
    run = root / f"{arm}_seed{seed}"
    manifest_path = run / "training_manifest.json"
    if not manifest_path.is_file():
        failures.append(f"missing manifest: {manifest_path}")
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest.get("dataset") or {}
    configs = dataset.get("configs") or []
    names = {str(item.get("config_name")) for item in configs}
    result = manifest.get("training_result") or {}
    training = manifest.get("training_config") or {}
    mask = manifest.get("residual_action_mask")
    prediction_abs_max = (manifest.get("label_statistics") or {}).get(
        "prediction_abs_max", []
    )
    row = {
        "arm": arm,
        "seed": seed,
        "status": manifest.get("status"),
        "device": training.get("device"),
        "samples": dataset.get("sample_count"),
        "config_names": sorted(names),
        "train_samples": result.get("train_samples"),
        "validation_samples": result.get("validation_samples"),
        "best_epoch": result.get("best_epoch"),
        "epochs_completed": result.get("epochs_completed"),
        "initial_validation_loss": result.get("initial_validation_loss"),
        "best_validation_loss": result.get("best_validation_loss"),
        "dade_enabled": result.get("dade_enabled"),
        "residual_action_mask": mask,
        "prediction_abs_max": prediction_abs_max,
        "frozen_bc_parameter_max_abs_drift": manifest.get(
            "frozen_bc_parameter_max_abs_drift"
        ),
    }
    rows.append(row)
    if row["status"] != "completed":
        failures.append(f"{run.name}: status={row['status']!r}")
    if names != expected_cells:
        failures.append(f"{run.name}: unexpected training cells {sorted(names)}")
    if names & forbidden_cells:
        failures.append(f"{run.name}: held-out high+strong cell leaked into training")
    if len(configs) != 3:
        failures.append(f"{run.name}: expected three training cells, got {len(configs)}")
    if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
        failures.append(f"{run.name}: frozen BC drift is not zero")
    if bool(row["dade_enabled"]):
        failures.append(f"{run.name}: DADE must be disabled")
    if mask != [0.0, 0.0, 1.0]:
        failures.append(f"{run.name}: residual_action_mask={mask!r}, expected [0, 0, 1]")
    if len(prediction_abs_max) != 3 or float(prediction_abs_max[0]) != 0.0 or float(
        prediction_abs_max[1]
    ) != 0.0:
        failures.append(
            f"{run.name}: rudder/stern residual predictions are not exactly zero"
        )
    if str(row["device"]) != os.environ["PRETRAIN_DEVICE"]:
        failures.append(
            f"{run.name}: device={row['device']!r}, expected {os.environ['PRETRAIN_DEVICE']!r}"
        )
    for filename in ("model_initial.pt", "final.pt", "training_result.json"):
        if not (run / filename).is_file():
            failures.append(f"{run.name}: missing {filename}")

payload = {
    "training_cells": sorted(expected_cells),
    "heldout_cells": sorted(forbidden_cells),
    "residual_action_mask": [0.0, 0.0, 1.0],
    "runs": rows,
    "failures": failures,
}
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
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS="$EVAL_WORKERS"
  CPU_BASE="$CPU_BASE" CPU_STRIDE="$CPU_STRIDE" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0 SEED_COUNT="$SEED_COUNT" CHECKPOINTS=final
)

if ! env STAGE=pc_grid_rpm_only_20261104 ARMS="$TREATMENT_ARM" SEEDS="$SEEDS" \
  TAGS="$TAGS" "${COMMON_EVAL_ENV[@]}" RUN_ROOT="$TRAIN_ROOT" \
  OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'RPM_ONLY_EVALUATION_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! env STAGE=pc_grid_rpm_only_reference_20261104 ARMS="$REFERENCE_ARM" SEEDS="$SEEDS" \
  TAGS="$TAGS" "${COMMON_EVAL_ENV[@]}" RUN_ROOT="$REFERENCE_RUN_ROOT" \
  OUTPUT_ROOT="$REFERENCE_EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval_reference" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'REFERENCE_EVALUATION_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! run_required_step SUMMARY_RPM_ONLY "$QUEUE_ROOT/summary.log" \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$EVAL_ROOT" --eval-root "$REFERENCE_EVAL_ROOT" \
  --output-dir "$SUMMARY_ROOT" --baseline-arm "$REFERENCE_ARM" \
  --checkpoint final --eval-seed-count "$SEED_COUNT"; then
  exit 1
fi

gate_rc=0
if ! "$PY" scripts/summarize_pc_grid_rpm_only_20261104.py \
  --summary "$SUMMARY_ROOT/summary.json" --output-dir "$REPORT_ROOT" \
  --treatment-arm "$TREATMENT_ARM" --reference-arm "$REFERENCE_ARM" \
  --train-seed-count "${#seed_list[@]}" --seed-count "$SEED_COUNT" \
  --tags "${tag_list[@]}" \
  --require-pass > "$QUEUE_ROOT/report.log" 2>&1; then
  gate_rc=1
  printf 'RPM_ONLY_GATE_NOT_PASSED; preserving report as evidence\n' >&2
fi

env REPORT_ROOT="$REPORT_ROOT" SUMMARY_ROOT="$SUMMARY_ROOT" \
  TRAIN_ROOT="$TRAIN_ROOT" EVAL_ROOT="$EVAL_ROOT" \
  REFERENCE_EVAL_ROOT="$REFERENCE_EVAL_ROOT" GATE_RC="$gate_rc" "$PY" - <<'PY'
import json
import os
from pathlib import Path

report_root = Path(os.environ["REPORT_ROOT"])
payload = {
    "training_audit": str(Path(os.environ["TRAIN_ROOT"]) / "audit.json"),
    "training_root": os.environ["TRAIN_ROOT"],
    "eval_root": os.environ["EVAL_ROOT"],
    "reference_eval_root": os.environ["REFERENCE_EVAL_ROOT"],
    "summary": str(Path(os.environ["SUMMARY_ROOT"]) / "summary.json"),
    "report": str(report_root / "report.json"),
    "report_markdown": str(report_root / "report.md"),
    "gate_returncode": int(os.environ["GATE_RC"]),
}
(report_root / "index.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(report_root / "index.json")
PY

printf '\n===== PC RPM-ONLY GRID DONE %s gate_rc=%s =====\n' \
  "$(date -Is)" "$gate_rc"
exit 0
