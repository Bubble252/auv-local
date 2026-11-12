#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_SUFFIX="${RUN_SUFFIX:-}"
QUEUE_ROOT="$ROOT/logs/pc_dade_grid_dagger_20261104${RUN_SUFFIX}"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'PC DADE grid DAgger queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

DADE_ARM='frozen_bc_residual_dade_grid_gpu_pose_lr3e-4_scale1p0'
DAGGER_ARM='frozen_bc_residual_dade_grid_dagger_pose_lr3e-4_scale1p0'
DADE_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
DAGGER_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_grid_dade_dagger_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"

SEEDS="${SEEDS:-94030 94031 94032 94033 94034}"
SEED_COUNT="${SEED_COUNT:-32}"
TAGS="${TAGS:-grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4}"
CHECKPOINTS="${CHECKPOINTS:-initial final}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
CPU_BASE="${CPU_BASE:-160}"
CPU_STRIDE="${CPU_STRIDE:-16}"
GPU_LIST="${GPU_LIST:-2 4 5}"
DAGGER_ENVS="${DAGGER_ENVS:-16}"
DAGGER_STEPS="${DAGGER_STEPS:-2048}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.05}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4096}"
PRETRAIN_MAX_EPOCHS="${PRETRAIN_MAX_EPOCHS:-200}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-30}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-1e-5}"
PRETRAIN_OUTPUT_L2="${PRETRAIN_OUTPUT_L2:-1e-4}"
PRETRAIN_VALIDATION_FRACTION="${PRETRAIN_VALIDATION_FRACTION:-0.2}"

ORIGINAL_DEMO_ROOT="$ROOT/artifacts/lowlevel/pc_speed_current_grid_teacher_demos_20261104"
DADE_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_dade_grid_dade_supervised_residual_20261104"
if [[ "$RUN_SUFFIX" == "_smoke" ]]; then
  DADE_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_dade_grid_dagger_baseline_eval_smoke_20261104"
else
  DADE_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_dade_grid_dade_eval_20261104"
fi
DAGGER_DEMO_ROOT="$ROOT/artifacts/lowlevel/pc_dade_grid_dagger_demos_20261104${RUN_SUFFIX}"
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_dade_grid_dagger_supervised_residual_20261104${RUN_SUFFIX}"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_dade_grid_dagger_eval_20261104${RUN_SUFFIX}"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_dade_grid_dagger_summary_20261104${RUN_SUFFIX}"
REPORT_ROOT="$ROOT/artifacts/lowlevel/pc_dade_grid_dagger_report_20261104${RUN_SUFFIX}"

declare -A CELL_CONFIG=(
  [lw]="$ROOT/configs/lowlevel/odarrl_control_pc_grid_low_weak.json"
  [ls]="$ROOT/configs/lowlevel/odarrl_control_pc_grid_low_strong.json"
  [hw]="$ROOT/configs/lowlevel/odarrl_control_pc_grid_high_weak.json"
)
CELL_ORDER=(lw ls hw)

read -r -a seed_list <<< "$SEEDS"
read -r -a tag_list <<< "$TAGS"
read -r -a checkpoint_list <<< "$CHECKPOINTS"
read -r -a gpu_list <<< "$GPU_LIST"
if (( ${#seed_list[@]} == 0 || ${#tag_list[@]} == 0 || ${#gpu_list[@]} == 0 )); then
  printf 'SEEDS, TAGS and GPU_LIST must be non-empty\n' >&2
  exit 2
fi
if (( ${#checkpoint_list[@]} == 0 )); then
  printf 'CHECKPOINTS must be non-empty\n' >&2
  exit 2
fi

mkdir -p "$DAGGER_DEMO_ROOT" "$TRAIN_ROOT" "$EVAL_ROOT" \
  "$SUMMARY_ROOT" "$REPORT_ROOT" "$DADE_EVAL_ROOT"

printf '\n===== PC DADE GRID DAGGER START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s seeds=%s tags=%s eval_seeds=%s gpus=%s noise=%s\n' \
  "$(hostname)" "$$" "$SEEDS" "$TAGS" "$SEED_COUNT" "$GPU_LIST" "$ACTION_NOISE_STD"

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

if [[ ! -f "$ORIGINAL_DEMO_ROOT/merged/demos.npz" ]]; then
  printf 'missing original grid teacher demos: %s\n' "$ORIGINAL_DEMO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$ROOT/artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt" ]]; then
  printf 'missing frozen BC checkpoint\n' >&2
  exit 1
fi

collect_pids=()
collect_seeds=()
for index in "${!seed_list[@]}"; do
  seed="${seed_list[$index]}"
  student="$DADE_TRAIN_ROOT/${DADE_ARM}_seed${seed}/final.pt"
  output="$DAGGER_DEMO_ROOT/seed${seed}/new"
  log="$QUEUE_ROOT/collect_seed${seed}.log"
  if [[ ! -f "$student" ]]; then
    printf 'MISSING_DADE_STUDENT seed=%s checkpoint=%s\n' "$seed" "$student" >&2
    exit 1
  fi
  if [[ -f "$output/demos.npz" && -f "$output/demos.manifest.json" ]]; then
    printf 'SKIP_COLLECT seed=%s output=%s\n' "$seed" "$output"
    continue
  fi
  cpu_start=$((224 + index * 8))
  printf 'START_COLLECT seed=%s cpu=%s student=%s output=%s\n' \
    "$seed" "$cpu_start-$((cpu_start + 7))" "$student" "$output"
  taskset -c "$cpu_start-$((cpu_start + 7))" env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" scripts/collect_pc_dagger_demos.py \
    --student-training-config "$DADE_CONFIG" \
    --checkpoint "$student" \
    --config "${CELL_CONFIG[lw]}" "${CELL_CONFIG[ls]}" "${CELL_CONFIG[hw]}" \
    --output "$output" \
    --num-envs "$DAGGER_ENVS" \
    --steps-per-env "$DAGGER_STEPS" \
    --seed-start "$((100000 + index * 1000))" \
    --episode-id-offset "$((1000000 + index * 10000))" \
    --action-noise-std "$ACTION_NOISE_STD" \
    --device cpu --dtype float64 --overwrite > "$log" 2>&1 &
  collect_pids+=("$!")
  collect_seeds+=("$seed")
done

collect_failures=0
for index in "${!collect_pids[@]}"; do
  seed="${collect_seeds[$index]}"
  if wait "${collect_pids[$index]}"; then
    printf 'END_COLLECT seed=%s rc=0\n' "$seed"
  else
    printf 'END_COLLECT seed=%s rc!=0 log=%s\n' \
      "$seed" "$QUEUE_ROOT/collect_seed${seed}.log"
    collect_failures=$((collect_failures + 1))
  fi
done
if (( collect_failures > 0 )); then
  exit 1
fi

for seed in "${seed_list[@]}"; do
  combined="$DAGGER_DEMO_ROOT/seed${seed}/combined"
  if [[ -f "$combined/demos.npz" && -f "$combined/demos.manifest.json" ]]; then
    printf 'SKIP_MERGE seed=%s output=%s\n' "$seed" "$combined"
    continue
  fi
  if ! run_required_step "MERGE_DAGGER_SEED${seed}" "$QUEUE_ROOT/merge_seed${seed}.log" \
    "$PY" scripts/merge_pc_bc_demos.py \
    --inputs "$ORIGINAL_DEMO_ROOT/merged/demos.npz" "$DAGGER_DEMO_ROOT/seed${seed}/new/demos.npz" \
    --output "$combined" --overwrite; then
    exit 1
  fi
done

run_pretrain_worker() {
  local slot="$1"
  local gpu="${gpu_list[$slot]}"
  local cpu_start=$((96 + slot * 8))
  local cpu_range="${cpu_start}-$((cpu_start + 7))"
  local failed=0
  local index
  for ((index = slot; index < ${#seed_list[@]}; index += ${#gpu_list[@]})); do
    local seed="${seed_list[$index]}"
    local output="$TRAIN_ROOT/${DAGGER_ARM}_seed${seed}"
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
        --training-config "$DAGGER_CONFIG" \
        --demos "$DAGGER_DEMO_ROOT/seed${seed}/combined/demos.npz" \
        --output "$output" \
        --batch-size "$PRETRAIN_BATCH_SIZE" \
        --max-epochs "$PRETRAIN_MAX_EPOCHS" \
        --patience "$PRETRAIN_PATIENCE" \
        --learning-rate "$PRETRAIN_LR" \
        --weight-decay "$PRETRAIN_WEIGHT_DECAY" \
        --output-l2 "$PRETRAIN_OUTPUT_L2" \
        --validation-fraction "$PRETRAIN_VALIDATION_FRACTION" \
        --seed "$seed" \
        --device cuda \
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

if ! run_required_step AUDIT_DADE_DAGGER "$QUEUE_ROOT/audit.log" \
  env TRAIN_ROOT="$TRAIN_ROOT" ARM="$DAGGER_ARM" SEEDS="$SEEDS" \
  "$PY" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

root = Path(os.environ["TRAIN_ROOT"])
arm = os.environ["ARM"]
seeds = [int(value) for value in os.environ["SEEDS"].split()]
allowed = {"pc_grid_low_weak", "pc_grid_low_strong", "pc_grid_high_weak"}
forbidden = {"pc_grid_high_strong"}
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
    entry_counts = Counter(str(item.get("config_name")) for item in configs)
    result = manifest.get("training_result") or {}
    row = {
        "seed": seed,
        "status": manifest.get("status"),
        "samples": dataset.get("sample_count"),
        "config_entries": len(configs),
        "unique_config_entries": len(names),
        "config_entry_counts": dict(sorted(entry_counts.items())),
        "config_names": sorted(names),
        "train_samples": result.get("train_samples"),
        "validation_samples": result.get("validation_samples"),
        "best_epoch": result.get("best_epoch"),
        "epochs_completed": result.get("epochs_completed"),
        "initial_validation_loss": result.get("initial_validation_loss"),
        "best_validation_loss": result.get("best_validation_loss"),
        "dade_enabled": result.get("dade_enabled"),
        "frozen_bc_parameter_max_abs_drift": manifest.get(
            "frozen_bc_parameter_max_abs_drift"
        ),
    }
    rows.append(row)
    if row["status"] != "completed":
        failures.append(f"{run.name}: status={row['status']!r}")
    if names != allowed:
        failures.append(f"{run.name}: unexpected training cells {sorted(names)}")
    if names & forbidden:
        failures.append(f"{run.name}: held-out high+strong cell leaked into training")
    if len(names) != 3:
        failures.append(f"{run.name}: expected three unique training cells, got {len(names)}")
    missing_sources = sorted(name for name in allowed if entry_counts[name] != 2)
    if missing_sources:
        failures.append(
            f"{run.name}: expected two demo sources per training cell, "
            f"bad counts { {name: entry_counts[name] for name in missing_sources} }"
        )
    if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
        failures.append(f"{run.name}: frozen BC drift is not zero")
    if not bool(row["dade_enabled"]):
        failures.append(f"{run.name}: DADE is not enabled")
    for filename in ("model_initial.pt", "final.pt", "training_result.json"):
        if not (run / filename).is_file():
            failures.append(f"{run.name}: missing {filename}")

payload = {
    "training_cells": sorted(allowed),
    "heldout_cells": sorted(forbidden),
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
  SEED_START=0 SEED_COUNT="$SEED_COUNT" CHECKPOINTS="$CHECKPOINTS"
)

if ! env STAGE=pc_dade_grid_baseline_dagger_20261104 ARMS="$DADE_ARM" SEEDS="$SEEDS" \
  TAGS="$TAGS" "${COMMON_EVAL_ENV[@]}" RUN_ROOT="$DADE_TRAIN_ROOT" \
  OUTPUT_ROOT="$DADE_EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval_dade_baseline" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'DADE_BASELINE_EVALUATION_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! env STAGE=pc_dade_grid_dagger_20261104 ARMS="$DAGGER_ARM" SEEDS="$SEEDS" \
  TAGS="$TAGS" "${COMMON_EVAL_ENV[@]}" RUN_ROOT="$TRAIN_ROOT" \
  OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval_dagger" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'DAGGER_EVALUATION_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! run_required_step SUMMARY_DADE_DAGGER "$QUEUE_ROOT/summary.log" \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$EVAL_ROOT" --eval-root "$DADE_EVAL_ROOT" \
  --output-dir "$SUMMARY_ROOT" --baseline-arm "$DADE_ARM" \
  --checkpoint initial --checkpoint final --checkpoint-baseline final=initial \
  --eval-seed-count "$SEED_COUNT"; then
  exit 1
fi

gate_rc=0
if ! "$PY" scripts/summarize_pc_dade_grid_20261104.py \
  --summary "$SUMMARY_ROOT/summary.json" --output-dir "$REPORT_ROOT" \
  --treatment-arm "$DAGGER_ARM" --reference-arm "$DADE_ARM" \
  --train-seed-count "${#seed_list[@]}" --seed-count "$SEED_COUNT" \
  --require-pass \
  > "$QUEUE_ROOT/report.log" 2>&1; then
  gate_rc=1
  printf 'DADE_DAGGER_GATE_NOT_PASSED; preserving report as evidence\n' >&2
fi

env REPORT_ROOT="$REPORT_ROOT" SUMMARY_ROOT="$SUMMARY_ROOT" TRAIN_ROOT="$TRAIN_ROOT" \
  DADE_TRAIN_ROOT="$DADE_TRAIN_ROOT" DADE_EVAL_ROOT="$DADE_EVAL_ROOT" \
  DAGGER_DEMO_ROOT="$DAGGER_DEMO_ROOT" EVAL_ROOT="$EVAL_ROOT" GATE_RC="$gate_rc" \
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

report_root = Path(os.environ["REPORT_ROOT"])
payload = {
    "training_audit": str(Path(os.environ["TRAIN_ROOT"]) / "audit.json"),
    "summary": str(Path(os.environ["SUMMARY_ROOT"]) / "summary.json"),
    "dagger_demos": os.environ["DAGGER_DEMO_ROOT"],
    "train_root": os.environ["TRAIN_ROOT"],
    "baseline_train_root": os.environ["DADE_TRAIN_ROOT"],
    "eval_root": os.environ["EVAL_ROOT"],
    "baseline_eval_root": os.environ["DADE_EVAL_ROOT"],
    "report": str(report_root / "report.json"),
    "report_markdown": str(report_root / "report.md"),
    "gate_returncode": int(os.environ["GATE_RC"]),
}
(report_root / "index.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(report_root / "index.json")
PY

printf '\n===== PC DADE GRID DAGGER DONE %s gate_rc=%s =====\n' \
  "$(date -Is)" "$gate_rc"
exit 0
