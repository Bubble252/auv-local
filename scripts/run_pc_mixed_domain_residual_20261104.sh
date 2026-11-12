#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_mixed_domain_residual_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'mixed-domain residual queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

TRAIN_SEEDS="${TRAIN_SEEDS:-94030 94031 94032 94033 94034}"
ARM='frozen_bc_residual_nodade_mixed_domain_lr3e-4_scale1p0'
CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_mixed_domain_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
BASELINE_ARM='frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0'

SOURCE_DEMO_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_teacher_demos_full_20261104"
DEMO_ROOT="$ROOT/artifacts/lowlevel/pc_mixed_domain_teacher_demos_20261104"
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_mixed_domain_supervised_residual_20261104"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_mixed_domain_supervised_residual_eval_20261104"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_mixed_domain_supervised_residual_summary_20261104"
COMBINED_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_mixed_domain_generalization_summary_20261104"
REPORT_ROOT="$ROOT/artifacts/lowlevel/pc_mixed_domain_generalization_report_20261104"

BASELINE_EARLY_EVAL="$ROOT/artifacts/lowlevel/pc_multienv_early_eval_nodade_seed94030_20261104"
BASELINE_EXTRA_EVAL="$ROOT/artifacts/lowlevel/pc_multienv_nodade_extra_eval_20261104"
BASELINE_EXTRA2_EVAL="$ROOT/artifacts/lowlevel/pc_multienv_nodade_extra_eval_20261104_seed33_34"
BASELINE_32SEED_EVAL="$ROOT/artifacts/lowlevel/pc_multienv_nodade_32seed_comparison_eval_20261104"
H0_BC_SUMMARY="$ROOT/artifacts/lowlevel/pc_server_remaining_baselines_20261103/bc_h0_seed0_31/summary.json"

SOURCE_CELLS=(c2 c3 h2 h3)
TRAIN_CPU_BASE="${TRAIN_CPU_BASE:-224}"
TRAIN_CPU_STRIDE="${TRAIN_CPU_STRIDE:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EVAL_CPU_BASE="${EVAL_CPU_BASE:-16}"
EVAL_CPU_STRIDE="${EVAL_CPU_STRIDE:-16}"
BASELINE_EVAL_CPU_BASE="${BASELINE_EVAL_CPU_BASE:-280}"
BASELINE_EVAL_CPU_STRIDE="${BASELINE_EVAL_CPU_STRIDE:-16}"
SEED_COUNT="${SEED_COUNT:-32}"
BASELINE_WAIT_SECONDS="${BASELINE_WAIT_SECONDS:-21600}"

read -r -a seed_list <<< "$TRAIN_SEEDS"

printf '\n===== PC MIXED-DOMAIN RESIDUAL START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s seeds=%s train_cells=%s heldout=h0,h1,c4\n' \
  "$(hostname)" "$$" "$TRAIN_SEEDS" "${SOURCE_CELLS[*]}"

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

mkdir -p "$DEMO_ROOT" "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "$COMBINED_SUMMARY_ROOT" "$REPORT_ROOT"

merge_inputs=()
for cell in "${SOURCE_CELLS[@]}"; do
  path="$SOURCE_DEMO_ROOT/$cell/demos.npz"
  manifest="$SOURCE_DEMO_ROOT/$cell/demos.manifest.json"
  if [[ ! -f "$path" || ! -f "$manifest" ]]; then
    printf 'MISSING_SOURCE_DEMOS cell=%s path=%s\n' "$cell" "$path" >&2
    exit 2
  fi
  merge_inputs+=("$path")
done

if [[ -f "$DEMO_ROOT/merged/demos.npz" && -f "$DEMO_ROOT/merged/demos.manifest.json" ]]; then
  printf 'SKIP_MERGE output=%s\n' "$DEMO_ROOT/merged"
else
  if ! run_required_step MERGE_MIXED_DOMAIN "$QUEUE_ROOT/merge.log" \
    "$PY" scripts/merge_pc_bc_demos.py \
    --inputs "${merge_inputs[@]}" --output "$DEMO_ROOT/merged" --overwrite; then
    exit 1
  fi
fi

train_pids=()
train_names=()
for index in "${!seed_list[@]}"; do
  seed="${seed_list[$index]}"
  output="$TRAIN_ROOT/${ARM}_seed${seed}"
  log="$QUEUE_ROOT/pretrain_${ARM}_seed${seed}.log"
  if [[ -f "$output/final.pt" && -f "$output/training_result.json" ]]; then
    printf 'SKIP_PRETRAIN seed=%s output=%s\n' "$seed" "$output"
    continue
  fi
  cpu_start=$((TRAIN_CPU_BASE + index * TRAIN_CPU_STRIDE))
  cpu="$cpu_start-$((cpu_start + TRAIN_CPU_STRIDE - 1))"
  printf 'START_PRETRAIN seed=%s cpu=%s output=%s\n' "$seed" "$cpu" "$output"
  taskset -c "$cpu" env OMP_NUM_THREADS="$TRAIN_CPU_STRIDE" MKL_NUM_THREADS="$TRAIN_CPU_STRIDE" \
    "$PY" scripts/pretrain_pc_residual.py \
    --training-config "$CONFIG" \
    --demos "$DEMO_ROOT/merged/demos.npz" \
    --output "$output" \
    --batch-size 4096 \
    --max-epochs 200 \
    --patience 30 \
    --learning-rate 1e-3 \
    --weight-decay 1e-5 \
    --output-l2 1e-4 \
    --validation-fraction 0.2 \
    --seed "$seed" \
    --device cpu \
    --overwrite > "$log" 2>&1 &
  train_pids+=("$!")
  train_names+=("$seed")
done

train_failures=0
for index in "${!train_pids[@]}"; do
  seed="${train_names[$index]}"
  if wait "${train_pids[$index]}"; then
    printf 'END_PRETRAIN seed=%s rc=0 %s\n' "$seed" "$(date -Is)"
  else
    rc=$?
    printf 'END_PRETRAIN seed=%s rc=%s log=%s\n' \
      "$seed" "$rc" "$QUEUE_ROOT/pretrain_${ARM}_seed${seed}.log"
    train_failures=$((train_failures + 1))
  fi
done
if (( train_failures > 0 )); then
  printf 'MIXED_DOMAIN_TRAINING_FAILURES=%s\n' "$train_failures" >&2
  exit 1
fi

if ! run_required_step AUDIT_MIXED_DOMAIN "$QUEUE_ROOT/audit.log" \
  env TRAIN_ROOT="$TRAIN_ROOT" ARM="$ARM" TRAIN_SEEDS="$TRAIN_SEEDS" "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TRAIN_ROOT"])
arm = os.environ["ARM"]
seeds = [int(value) for value in os.environ["TRAIN_SEEDS"].split()]
allowed = {
    "pc_c2_alos_migration",
    "pc_c3_random_realistic_alos_v1",
    "pc_c3_random_h2_speed_2p0_2p25",
    "pc_c3_random_h3_speed_2p25_2p4",
}
forbidden_tokens = ("h0", "h1", "c4_stress")
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
    names = {str(config.get("config_name")) for config in configs}
    config_paths = [str(config.get("config_path", "")) for config in configs]
    result = manifest.get("training_result") or {}
    row = {
        "seed": seed,
        "status": manifest.get("status"),
        "samples": dataset.get("sample_count"),
        "config_names": sorted(names),
        "train_samples": result.get("train_samples"),
        "validation_samples": result.get("validation_samples"),
        "best_epoch": result.get("best_epoch"),
        "epochs_completed": result.get("epochs_completed"),
        "initial_validation_loss": result.get("initial_validation_loss"),
        "best_validation_loss": result.get("best_validation_loss"),
        "frozen_bc_parameter_max_abs_drift": manifest.get(
            "frozen_bc_parameter_max_abs_drift"
        ),
    }
    rows.append(row)
    if row["status"] != "completed":
        failures.append(f"{run.name}: status={row['status']!r}")
    if names != allowed:
        failures.append(f"{run.name}: unexpected training cells {sorted(names)}")
    if len(configs) != 4:
        failures.append(f"{run.name}: expected four training cells, got {len(configs)}")
    for config_path in config_paths:
        if any(token in config_path for token in forbidden_tokens):
            failures.append(f"{run.name}: held-out cell leaked into training: {config_path}")
    if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
        failures.append(f"{run.name}: frozen BC drift is not zero")
    if not (run / "model_initial.pt").is_file():
        failures.append(f"{run.name}: missing zero-residual initial checkpoint")

payload = {
    "training_cells": sorted(allowed),
    "heldout_cells": ["h0", "h1", "c4"],
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
  CPU_BASE="$EVAL_CPU_BASE" CPU_STRIDE="$EVAL_CPU_STRIDE"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0 SEED_COUNT="$SEED_COUNT"
)

if ! env STAGE=mixed_domain_eval_20261104 ARMS="$ARM" SEEDS="$TRAIN_SEEDS" \
  TAGS='default strict125 h0 h1 h2 h3 h4 c4' CHECKPOINTS='initial final' \
  "${COMMON_EVAL_ENV[@]}" RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" \
  LOG_DIR="$QUEUE_ROOT/eval" bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'MIXED_DOMAIN_EVALUATION_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! run_required_step SUMMARY_MIXED_DOMAIN "$QUEUE_ROOT/summary_mixed.log" \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$EVAL_ROOT" --output-dir "$SUMMARY_ROOT" \
  --baseline-arm "$ARM" --checkpoint initial --checkpoint final \
  --checkpoint-baseline final=initial; then
  exit 1
fi

if ! env STAGE=multienv_nodade_32seed_comparison_20261104 ARMS="$BASELINE_ARM" \
  SEEDS='94031 94032 94033 94034' TAGS='default strict125' CHECKPOINTS='initial final' \
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS="$EVAL_WORKERS" \
  CPU_BASE="$BASELINE_EVAL_CPU_BASE" CPU_STRIDE="$BASELINE_EVAL_CPU_STRIDE" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 SEED_START=0 SEED_COUNT="$SEED_COUNT" \
  RUN_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104" \
  OUTPUT_ROOT="$BASELINE_32SEED_EVAL" LOG_DIR="$QUEUE_ROOT/eval_baseline_32seed" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'BASELINE_32SEED_EVALUATION_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

wait_for_baseline_evals() {
  local deadline=$((SECONDS + BASELINE_WAIT_SECONDS))
  while true; do
    local early_count extra_count extra2_count baseline32_count
    early_count=$(find "$BASELINE_EARLY_EVAL" -path '*/ckfinal_seed*/summary.json' 2>/dev/null | wc -l)
    extra_count=$(find "$BASELINE_EXTRA_EVAL" -path '*/ckfinal_seed*/summary.json' 2>/dev/null | wc -l)
    extra2_count=$(find "$BASELINE_EXTRA2_EVAL" -path '*/ckfinal_seed*/summary.json' 2>/dev/null | wc -l)
    baseline32_count=$(find "$BASELINE_32SEED_EVAL" -path '*/ckfinal_seed*/summary.json' 2>/dev/null | wc -l)
    if (( early_count >= 6 && extra_count >= 12 && extra2_count >= 12 && baseline32_count >= 8 )); then
      printf 'BASELINE_EVALS_READY early=%s extra=%s extra2=%s baseline32=%s\n' \
        "$early_count" "$extra_count" "$extra2_count" "$baseline32_count"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      printf 'BASELINE_EVAL_TIMEOUT early=%s/6 extra=%s/12 extra2=%s/12 baseline32=%s/8\n' \
        "$early_count" "$extra_count" "$extra2_count" "$baseline32_count" >&2
      return 1
    fi
    printf 'WAIT_BASELINE_EVALS early=%s/6 extra=%s/12 extra2=%s/12 baseline32=%s/8 %s\n' \
      "$early_count" "$extra_count" "$extra2_count" "$baseline32_count" "$(date -Is)"
    sleep 30
  done
}

if ! wait_for_baseline_evals; then
  exit 1
fi

if ! run_required_step SUMMARY_MIXED_DOMAIN_GENERALIZATION "$QUEUE_ROOT/summary_combined.log" \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$EVAL_ROOT" \
  --eval-root "$BASELINE_EARLY_EVAL" \
  --eval-root "$BASELINE_EXTRA_EVAL" \
  --eval-root "$BASELINE_EXTRA2_EVAL" \
  --eval-root "$BASELINE_32SEED_EVAL" \
  --output-dir "$COMBINED_SUMMARY_ROOT" \
  --baseline-arm "$BASELINE_ARM" --checkpoint final --eval-seed-count "$SEED_COUNT"; then
  exit 1
fi

gate_rc=0
if ! "$PY" scripts/summarize_pc_mixed_domain_20261104.py \
  --summary "$COMBINED_SUMMARY_ROOT/summary.json" \
  --h0-bc-summary "$H0_BC_SUMMARY" \
  --output-dir "$REPORT_ROOT" \
  --baseline-arm "$BASELINE_ARM" --treatment-arm "$ARM" \
  --seed-count "$SEED_COUNT" --require-pass > "$QUEUE_ROOT/report.log" 2>&1; then
  gate_rc=1
  printf 'MIXED_DOMAIN_GATE_NOT_PASSED; preserving report as evidence\n' >&2
fi

env REPORT_ROOT="$REPORT_ROOT" TRAIN_ROOT="$TRAIN_ROOT" SUMMARY_ROOT="$SUMMARY_ROOT" \
  COMBINED_SUMMARY_ROOT="$COMBINED_SUMMARY_ROOT" EVAL_ROOT="$EVAL_ROOT" GATE_RC="$gate_rc" \
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

report_root = Path(os.environ["REPORT_ROOT"])
payload = {
    "training_audit": str(Path(os.environ["TRAIN_ROOT"]) / "audit.json"),
    "mixed_summary": str(Path(os.environ["SUMMARY_ROOT"]) / "summary.json"),
    "combined_summary": str(Path(os.environ["COMBINED_SUMMARY_ROOT"]) / "summary.json"),
    "eval_root": os.environ["EVAL_ROOT"],
    "generalization_report": str(report_root / "report.json"),
    "generalization_report_markdown": str(report_root / "report.md"),
    "gate_returncode": int(os.environ["GATE_RC"]),
}
(report_root / "index.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(report_root / "index.json")
PY

printf '\n===== PC MIXED-DOMAIN RESIDUAL DONE %s gate_rc=%s =====\n' \
  "$(date -Is)" "$gate_rc"
exit 0
