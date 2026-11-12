#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_frozen_bc_residual_hardenv_v2_20261103"
mkdir -p "$QUEUE_ROOT"

# Share the S8b queue lock so this queue starts only after the corrected C3
# experiments have finished training and evaluation.
exec 9> "$ROOT/logs/pc_frozen_bc_residual_v2_20261103/queue.lock"
printf '\n===== FROZEN BC RESIDUAL HARD-ENV V2 WAIT %s =====\n' "$(date -Is)" >> "$QUEUE_ROOT/queue.log"
if ! flock 9; then
  printf 'failed to acquire S8b queue lock\n' >> "$QUEUE_ROOT/queue.log"
  exit 1
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_20261103"
TRAIN_LOGS="$ROOT/logs/pc_server_scale_frozen_bc_residual_hardenv_v2_20261103/train"
EVAL_H1_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_h1_eval_20261103"
EVAL_H2_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_h2_eval_20261103"
EVAL_C4_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_c4_eval_20261103"
EVAL_CROSS_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_cross_eval_20261103"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_summary_20261103"
GPU_LOCK_DIR="$ROOT/logs/pc_frozen_bc_residual_v2_20261103/locks"

H1_ARM='frozen_bc_residual_nodade_h1_lr3e-4_scale1p0'
H2_ARM='frozen_bc_residual_nodade_h2_lr3e-4_scale1p0'
C4_ARM='frozen_bc_residual_nodade_c4_lr3e-4_scale1p0'
ARMS="$H1_ARM $H2_ARM $C4_ARM"
SEEDS='91430 91431 91432'

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

printf '\n===== FROZEN BC RESIDUAL HARD-ENV V2 QUEUE START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s root=%s\n' "$(hostname)" "$$" "$ROOT"
printf 'arms=%s\nseeds=%s\n' "$ARMS" "$SEEDS"

run_required_step TRAIN_FROZEN_RESIDUAL_HARDENV_V2 env \
  STAGE=frozen_bc_residual_hardenv_v2 ARMS="$ARMS" SEEDS="$SEEDS" ITERATIONS=100 \
  GPU_IDS='4 5' MIN_FREE_MIB=16500 GPU_WAIT_SECONDS=43200 \
  LOCK_DIR="$GPU_LOCK_DIR" \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$TRAIN_ROOT" LOG_DIR="$TRAIN_LOGS" \
  bash scripts/run_pc_server_scale_20261103.sh

run_required_step AUDIT_FROZEN_RESIDUAL_HARDENV_V2 "$PY" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_20261103")
arms = (
    "frozen_bc_residual_nodade_h1_lr3e-4_scale1p0",
    "frozen_bc_residual_nodade_h2_lr3e-4_scale1p0",
    "frozen_bc_residual_nodade_c4_lr3e-4_scale1p0",
)
seeds = (91430, 91431, 91432)
rows = []
failures = []
for arm in arms:
    for seed in seeds:
        run = root / f"{arm}_seed{seed}"
        manifest_path = run / "training_manifest.json"
        row = {"run": run.name, "arm": arm, "seed": seed}
        if not manifest_path.is_file():
            failures.append(f"{run.name}: missing training_manifest.json")
            rows.append(row)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = manifest.get("frozen_bc_audit") or {}
        groups = manifest.get("optimizer_param_group_counts") or {}
        row.update(
            {
                "status": manifest.get("status"),
                "initial_residual_abs_max": manifest.get("initial_residual_abs_max"),
                "frozen_bc_audit": audit,
                "optimizer_param_group_counts": groups,
                "optimizer_parameter_count": manifest.get("optimizer_parameter_count"),
                "environment_config": manifest.get("environment_config"),
            }
        )
        if manifest.get("status") != "completed":
            failures.append(f"{run.name}: training status is not completed")
        if not audit.get("frozen_bc_parameters_unchanged"):
            failures.append(f"{run.name}: frozen BC changed")
        if float(audit.get("frozen_bc_parameter_max_abs_drift", 1.0)) != 0.0:
            failures.append(f"{run.name}: frozen BC drift is not zero")
        if audit.get("frozen_bc_parameters_require_grad"):
            failures.append(f"{run.name}: frozen BC still requires grad")
        if float(manifest.get("initial_residual_abs_max", 1.0)) > 1e-7:
            failures.append(f"{run.name}: initial residual is not zero")
        if int(groups.get("actor", 0)) <= 0 or int(groups.get("critic", 0)) <= 0:
            failures.append(f"{run.name}: actor/critic optimizer groups are missing")
        if int(groups.get("dade_head", 0)) != 0:
            failures.append(f"{run.name}: no-DADE run unexpectedly has a dade_head group")
        rows.append(row)

output = root / "frozen_bc_residual_hardenv_v2_audit.json"
output.write_text(
    json.dumps({"runs": rows, "failures": failures}, indent=2, sort_keys=True) + "\n",
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
)

run_step EVAL_FROZEN_RESIDUAL_HARDENV_V2_H1 env \
  STAGE=frozen_bc_residual_hardenv_v2 ARMS="$H1_ARM" SEEDS="$SEEDS" \
  TAGS='h1' CHECKPOINTS='initial 0 20 40 60 80 99' \
  SEED_START=0 SEED_COUNT=32 "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_H1_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_hardenv_v2_eval_h1_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_step EVAL_FROZEN_RESIDUAL_HARDENV_V2_H2 env \
  STAGE=frozen_bc_residual_hardenv_v2 ARMS="$H2_ARM" SEEDS="$SEEDS" \
  TAGS='h2' CHECKPOINTS='initial 0 20 40 60 80 99' \
  SEED_START=0 SEED_COUNT=32 "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_H2_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_hardenv_v2_eval_h2_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_step EVAL_FROZEN_RESIDUAL_HARDENV_V2_C4 env \
  STAGE=frozen_bc_residual_hardenv_v2 ARMS="$C4_ARM" SEEDS="$SEEDS" \
  TAGS='c4' CHECKPOINTS='initial 0 20 40 60 80 99' \
  SEED_START=0 SEED_COUNT=32 "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_C4_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_hardenv_v2_eval_c4_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_step EVAL_FROZEN_RESIDUAL_HARDENV_V2_CROSS env \
  STAGE=frozen_bc_residual_hardenv_v2 ARMS="$ARMS" SEEDS="$SEEDS" \
  TAGS='default strict125' CHECKPOINTS='initial 99' \
  SEED_START=0 SEED_COUNT=64 "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_CROSS_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_hardenv_v2_eval_cross_20261103" \
  bash scripts/eval_pc_server_scale_20261103.sh

summary_args=(
  --checkpoint-baseline 0=initial
  --checkpoint-baseline 20=initial
  --checkpoint-baseline 40=initial
  --checkpoint-baseline 60=initial
  --checkpoint-baseline 80=initial
  --checkpoint-baseline 99=initial
)

run_required_step SUMMARY_FROZEN_RESIDUAL_HARDENV_V2_H1 \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_H1_ROOT" \
  --output-dir "$SUMMARY_ROOT/h1" \
  --baseline-arm "$H1_ARM" "${summary_args[@]}"

run_required_step SUMMARY_FROZEN_RESIDUAL_HARDENV_V2_H2 \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_H2_ROOT" \
  --output-dir "$SUMMARY_ROOT/h2" \
  --baseline-arm "$H2_ARM" "${summary_args[@]}"

run_required_step SUMMARY_FROZEN_RESIDUAL_HARDENV_V2_C4 \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_C4_ROOT" \
  --output-dir "$SUMMARY_ROOT/c4" \
  --baseline-arm "$C4_ARM" "${summary_args[@]}"

run_required_step SUMMARY_FROZEN_RESIDUAL_HARDENV_V2_CROSS \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_CROSS_ROOT" \
  --output-dir "$SUMMARY_ROOT/cross" \
  --baseline-arm "$H1_ARM" \
  --checkpoint-baseline 99=initial

printf '\n===== FROZEN BC RESIDUAL HARD-ENV V2 QUEUE DONE %s =====\n' "$(date -Is)"
