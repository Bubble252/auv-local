#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_frozen_bc_residual_v2_20261103"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'frozen residual v2 queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_20261103"
TRAIN_LOGS="$ROOT/logs/pc_server_scale_frozen_bc_residual_v2_20261103/train"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_eval_20261103"
EVAL_LOGS="$ROOT/logs/pc_server_scale_frozen_bc_residual_v2_eval_20261103"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_eval_summary_20261103"

# GPU 4/5 were verified idle before launch. The queue keeps its own GPU locks so
# it serializes its own jobs without blocking on the saturated GPU1 queue.
GPU_LOCK_DIR="$QUEUE_ROOT/locks"
ARMS='frozen_bc_residual_nodade_lr3e-4_scale1p0 frozen_bc_residual_dade_head_isolated_lr3e-4_scale1p0 frozen_bc_residual_dade_history_only_lr3e-4_scale1p0'
SEEDS='91430 91431 91432'
BASELINE_ARM='frozen_bc_residual_nodade_lr3e-4_scale1p0'

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

printf '\n===== FROZEN BC RESIDUAL V2 QUEUE START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s root=%s\n' "$(hostname)" "$$" "$ROOT"
printf 'arms=%s\nseeds=%s\n' "$ARMS" "$SEEDS"

run_step TRAIN_FROZEN_RESIDUAL_V2 env \
  STAGE=frozen_bc_residual_v2 ARMS="$ARMS" SEEDS="$SEEDS" ITERATIONS=100 \
  GPU_IDS='4 5' MIN_FREE_MIB=16500 GPU_WAIT_SECONDS=43200 \
  LOCK_DIR="$GPU_LOCK_DIR" \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$TRAIN_ROOT" LOG_DIR="$TRAIN_LOGS" \
  bash scripts/run_pc_server_scale_20261103.sh

run_required_step AUDIT_FROZEN_RESIDUAL_V2 "$PY" - <<'PY'
import json
from pathlib import Path

import torch

root = Path("artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_20261103")
arms = (
    "frozen_bc_residual_nodade_lr3e-4_scale1p0",
    "frozen_bc_residual_dade_head_isolated_lr3e-4_scale1p0",
    "frozen_bc_residual_dade_history_only_lr3e-4_scale1p0",
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
        row.update(
            {
                "status": manifest.get("status"),
                "initial_residual_abs_max": manifest.get("initial_residual_abs_max"),
                "frozen_bc_audit": audit,
                "optimizer_param_group_counts": manifest.get(
                    "optimizer_param_group_counts"
                ),
                "optimizer_parameter_count": manifest.get("optimizer_parameter_count"),
            }
        )
        if not audit.get("frozen_bc_parameters_unchanged"):
            failures.append(f"{run.name}: frozen BC changed")
        if float(audit.get("frozen_bc_parameter_max_abs_drift", 1.0)) != 0.0:
            failures.append(f"{run.name}: frozen BC drift is not zero")
        if audit.get("frozen_bc_parameters_require_grad"):
            failures.append(f"{run.name}: frozen BC still requires grad")
        if float(manifest.get("initial_residual_abs_max", 1.0)) > 1e-7:
            failures.append(f"{run.name}: initial residual is not zero")
        if "dade_head" in arm:
            groups = manifest.get("optimizer_param_group_counts") or {}
            count = int(groups.get("dade_head", 0))
            row["dade_head_parameter_count"] = count
            if count <= 0:
                failures.append(f"{run.name}: dade_head optimizer group is missing")
            checkpoint = run / "model_10.pt"
            if not checkpoint.is_file():
                failures.append(f"{run.name}: missing model_10.pt")
            else:
                state = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )["model_state_dict"]
                head_abs_max = max(
                    (
                        float(value.detach().abs().max())
                        for key, value in state.items()
                        if key.startswith("actor_dade_head.") and value.numel()
                    ),
                    default=0.0,
                )
                row["dade_head_abs_max_at_model10"] = head_abs_max
                if head_abs_max <= 0.0:
                    failures.append(f"{run.name}: dade_head weights are still zero")
        rows.append(row)

output = root / "frozen_bc_residual_v2_audit.json"
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
  SEED_START=0 SEED_COUNT=64
)

run_step EVAL_FROZEN_RESIDUAL_V2_FIXED env \
  STAGE=frozen_bc_residual_v2 ARMS="$ARMS" SEEDS="$SEEDS" \
  TAGS='default strict125' CHECKPOINTS='initial 0 20 40 60 80 99' \
  "${COMMON_EVAL_ENV[@]}" \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" LOG_DIR="$EVAL_LOGS" \
  bash scripts/eval_pc_server_scale_20261103.sh

for ablation in current-only history-zero; do
  run_step "EVAL_FROZEN_RESIDUAL_V2_${ablation}" env \
    STAGE=frozen_bc_residual_v2 ARMS="$ARMS" SEEDS="$SEEDS" \
    TAGS='default' CHECKPOINTS='99' HISTORY_ABLATION="$ablation" \
    DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4 \
    CPU_BASE=160 CPU_STRIDE=16 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    SEED_START=0 SEED_COUNT=32 \
    RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" \
    LOG_DIR="${EVAL_LOGS}_ablation" \
    bash scripts/eval_pc_server_scale_20261103.sh
done

run_step EVAL_FROZEN_RESIDUAL_V2_HIGHENV env \
  STAGE=frozen_bc_residual_v2 ARMS="$ARMS" SEEDS="$SEEDS" \
  TAGS='h1 h2 h4 c4' CHECKPOINTS='99' \
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4 \
  CPU_BASE=160 CPU_STRIDE=16 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  SEED_START=0 SEED_COUNT=32 \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" \
  LOG_DIR="${EVAL_LOGS}_highenv" \
  bash scripts/eval_pc_server_scale_20261103.sh

run_step SUMMARY_FROZEN_RESIDUAL_V2 \
  "$PY" scripts/summarize_pc_server_evals.py \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$SUMMARY_ROOT" \
  --baseline-arm "$BASELINE_ARM" \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 40=initial \
  --checkpoint-baseline 60=initial \
  --checkpoint-baseline 80=initial \
  --checkpoint-baseline 99=initial

printf '\n===== FROZEN BC RESIDUAL V2 QUEUE DONE %s =====\n' "$(date -Is)"
