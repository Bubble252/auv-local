#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_grid_dagger_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'PC grid DAgger queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

GRID_ARM='frozen_bc_residual_nodade_grid_lr3e-4_scale1p0'
DAGGER_ARM='frozen_bc_residual_nodade_grid_dagger_lr3e-4_scale1p0'
GRID_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_grid_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_grid_dagger_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
MULTIENV_ARM='frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0'
MIXED_ARM='frozen_bc_residual_nodade_mixed_domain_lr3e-4_scale1p0'
TRAIN_SEEDS="${TRAIN_SEEDS:-94030 94031 94032 94033 94034}"
SEED_COUNT="${SEED_COUNT:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DAGGER_ENVS="${DAGGER_ENVS:-16}"
DAGGER_STEPS="${DAGGER_STEPS:-2048}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.0}"

GRID_DEMO_ROOT="$ROOT/artifacts/lowlevel/pc_speed_current_grid_teacher_demos_20261104"
GRID_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_speed_current_grid_supervised_residual_20261104"
GRID_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_speed_current_grid_eval_20261104"
DAGGER_ROOT="$ROOT/artifacts/lowlevel/pc_grid_dagger_demos_20261104"
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_grid_dagger_supervised_residual_20261104"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_grid_dagger_eval_20261104"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_grid_dagger_summary_20261104"
REPORT_ROOT="$ROOT/artifacts/lowlevel/pc_grid_dagger_report_20261104"

declare -A CELL_CONFIG=(
  [lw]="$ROOT/configs/lowlevel/odarrl_control_pc_grid_low_weak.json"
  [ls]="$ROOT/configs/lowlevel/odarrl_control_pc_grid_low_strong.json"
  [hw]="$ROOT/configs/lowlevel/odarrl_control_pc_grid_high_weak.json"
)
CELL_ORDER=(lw ls hw)

read -r -a seed_list <<< "$TRAIN_SEEDS"
mkdir -p "$DAGGER_ROOT" "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "$REPORT_ROOT"
printf '\n===== PC GRID DAGGER START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s seeds=%s cells=%s envs=%s steps=%s noise=%s\n' \
  "$(hostname)" "$$" "$TRAIN_SEEDS" "${CELL_ORDER[*]}" \
  "$DAGGER_ENVS" "$DAGGER_STEPS" "$ACTION_NOISE_STD"

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

if [[ ! -f "$GRID_DEMO_ROOT/merged/demos.npz" ]]; then
  printf 'missing original grid teacher demos: %s\n' "$GRID_DEMO_ROOT/merged/demos.npz" >&2
  exit 1
fi

collect_pids=()
collect_seeds=()
for index in "${!seed_list[@]}"; do
  seed="${seed_list[$index]}"
  student="$GRID_TRAIN_ROOT/${GRID_ARM}_seed${seed}/final.pt"
  output="$DAGGER_ROOT/seed${seed}/new"
  log="$QUEUE_ROOT/collect_seed${seed}.log"
  if [[ ! -f "$student" ]]; then
    printf 'MISSING_STUDENT seed=%s checkpoint=%s\n' "$seed" "$student" >&2
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
    --student-training-config "$GRID_CONFIG" \
    --checkpoint "$student" \
    --config "${CELL_CONFIG[lw]}" "${CELL_CONFIG[ls]}" "${CELL_CONFIG[hw]}" \
    --output "$output" \
    --num-envs "$DAGGER_ENVS" \
    --steps-per-env "$DAGGER_STEPS" \
    --seed-start "$((100000 + (seed - ${seed_list[0]}) * 1000))" \
    --episode-id-offset "$((1000000 + (seed - ${seed_list[0]}) * 10000))" \
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
  combined="$DAGGER_ROOT/seed${seed}/combined"
  if [[ -f "$combined/demos.npz" && -f "$combined/demos.manifest.json" ]]; then
    printf 'SKIP_MERGE seed=%s output=%s\n' "$seed" "$combined"
    continue
  fi
  if ! run_required_step "MERGE_DAgger_SEED${seed}" "$QUEUE_ROOT/merge_seed${seed}.log" \
    "$PY" scripts/merge_pc_bc_demos.py \
    --inputs "$GRID_DEMO_ROOT/merged/demos.npz" "$DAGGER_ROOT/seed${seed}/new/demos.npz" \
    --output "$combined" --overwrite; then
    exit 1
  fi
done

train_pids=()
train_names=()
for index in "${!seed_list[@]}"; do
  seed="${seed_list[$index]}"
  output="$TRAIN_ROOT/${DAGGER_ARM}_seed${seed}"
  log="$QUEUE_ROOT/pretrain_seed${seed}.log"
  if [[ -f "$output/final.pt" && -f "$output/training_result.json" ]]; then
    printf 'SKIP_PRETRAIN seed=%s output=%s\n' "$seed" "$output"
    continue
  fi
  cpu_start=$((264 + index * 8))
  printf 'START_PRETRAIN seed=%s cpu=%s output=%s\n' \
    "$seed" "$cpu_start-$((cpu_start + 7))" "$output"
  taskset -c "$cpu_start-$((cpu_start + 7))" env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" scripts/pretrain_pc_residual.py \
    --training-config "$CONFIG" \
    --demos "$DAGGER_ROOT/seed${seed}/combined/demos.npz" \
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
    printf 'END_PRETRAIN seed=%s rc=0\n' "$seed"
  else
    printf 'END_PRETRAIN seed=%s rc!=0 log=%s\n' \
      "$seed" "$QUEUE_ROOT/pretrain_seed${seed}.log"
    train_failures=$((train_failures + 1))
  fi
done
if (( train_failures > 0 )); then
  exit 1
fi

if ! run_required_step AUDIT_DAGGER_RESIDUAL "$QUEUE_ROOT/audit.log" \
  env TRAIN_ROOT="$TRAIN_ROOT" ARM="$DAGGER_ARM" TRAIN_SEEDS="$TRAIN_SEEDS" "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TRAIN_ROOT"])
arm = os.environ["ARM"]
seeds = [int(value) for value in os.environ["TRAIN_SEEDS"].split()]
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
    result = manifest.get("training_result") or {}
    row = {
        "seed": seed,
        "status": manifest.get("status"),
        "samples": dataset.get("sample_count"),
        "config_entries": len(configs),
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
    if names & forbidden:
        failures.append(f"{run.name}: held-out high+strong cell leaked into training")
    if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
        failures.append(f"{run.name}: frozen BC drift is not zero")
    if not (run / "model_initial.pt").is_file():
        failures.append(f"{run.name}: missing pretrain initial checkpoint")
    if not (run / "final.pt").is_file():
        failures.append(f"{run.name}: missing final checkpoint")

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
  CPU_BASE=320 CPU_STRIDE=16 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0 SEED_COUNT="$SEED_COUNT"
)

if ! env STAGE=grid_dagger_eval_20261104 ARMS="$DAGGER_ARM" SEEDS="$TRAIN_SEEDS" \
  TAGS='grid_lw grid_ls grid_hw grid_hs default strict125 c4 h4' \
  CHECKPOINTS='initial final' "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$QUEUE_ROOT/eval" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'DAGGER_EVALUATION_FAILED %s\n' "$(date -Is)" >&2
  exit 1
fi

if ! run_required_step SUMMARY_DAGGER "$QUEUE_ROOT/summary.log" \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$EVAL_ROOT" --eval-root "$GRID_EVAL_ROOT" \
  --output-dir "$SUMMARY_ROOT" --baseline-arm "$DAGGER_ARM" \
  --checkpoint initial --checkpoint final --checkpoint-baseline final=initial \
  --eval-seed-count "$SEED_COUNT"; then
  exit 1
fi

gate_rc=0
if ! "$PY" scripts/summarize_pc_speed_current_grid_20261104.py \
  --summary "$SUMMARY_ROOT/summary.json" --output-dir "$REPORT_ROOT" \
  --grid-arm "$DAGGER_ARM" --multienv-arm "$MULTIENV_ARM" \
  --mixed-domain-arm "$MIXED_ARM" --seed-count "$SEED_COUNT" --require-pass \
  > "$QUEUE_ROOT/report.log" 2>&1; then
  gate_rc=1
  printf 'DAGGER_GATE_NOT_PASSED; preserving report as evidence\n' >&2
fi

env REPORT_ROOT="$REPORT_ROOT" SUMMARY_ROOT="$SUMMARY_ROOT" TRAIN_ROOT="$TRAIN_ROOT" \
  DAGGER_ROOT="$DAGGER_ROOT" EVAL_ROOT="$EVAL_ROOT" GRID_EVAL_ROOT="$GRID_EVAL_ROOT" \
  GATE_RC="$gate_rc" "$PY" - <<'PY'
import json
import os
from pathlib import Path

report_root = Path(os.environ["REPORT_ROOT"])
payload = {
    "training_audit": str(Path(os.environ["TRAIN_ROOT"]) / "audit.json"),
    "summary": str(Path(os.environ["SUMMARY_ROOT"]) / "summary.json"),
    "eval_root": os.environ["EVAL_ROOT"],
    "grid_eval_root": os.environ["GRID_EVAL_ROOT"],
    "dagger_demos": os.environ["DAGGER_ROOT"],
    "report": str(report_root / "report.json"),
    "report_markdown": str(report_root / "report.md"),
    "gate_returncode": int(os.environ["GATE_RC"]),
}
(report_root / "index.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(report_root / "index.json")
PY

printf '\n===== PC GRID DAGGER DONE %s gate_rc=%s =====\n' \
  "$(date -Is)" "$gate_rc"
exit 0
