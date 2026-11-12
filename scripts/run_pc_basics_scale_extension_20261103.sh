#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_basics_scale_extension_20261103"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'basics scale extension queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

C2_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_20261103"
C2_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_eval_20261103"
C2_EXT_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_eval_ext_20261103"
C2_EXT_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_summary_ext_20261103"
C2_ALL5_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_c2_basics_summary_all5_20261103"

H3_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_20261103"
H3_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_eval_20261103"
H3_EXT_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_eval_ext_20261103"
H3_EXT_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_summary_ext_20261103"
H3_ALL5_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_h3_basics_summary_all5_20261103"

C3_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_20261103"
C3_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_eval_20261103"
C3_EXT_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_eval_ext_20261103"
C3_EXT_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_summary_ext_20261103"
C3_ALL5_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_summary_all5_20261103"

HARD_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_20261103"
HARD_H1_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_h1_eval_20261103"
HARD_H2_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_h2_eval_20261103"
HARD_C4_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_c4_eval_20261103"
HARD_CROSS_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_cross_eval_20261103"
HARD_EXT_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_eval_ext_20261103"
HARD_EXT_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_summary_ext_20261103"
HARD_ALL5_SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_summary_all5_20261103"

C2_BC="$ROOT/artifacts/lowlevel/pc_bc_c0_c1_c2_v3_20261103/pc_bc_actor.pt"
C2_EXT_SEEDS='91208 91209'
H3_EXT_SEEDS='91433 91434'
HARD_EXT_SEEDS='91433 91434'
H3_ARM='frozen_bc_residual_nodade_h3_lr3e-4_scale1p0'
C3_ARM='frozen_bc_residual_nodade_lr3e-4_scale1p0'
HARD_ARMS='frozen_bc_residual_nodade_h1_lr3e-4_scale1p0 frozen_bc_residual_nodade_h2_lr3e-4_scale1p0 frozen_bc_residual_nodade_c4_lr3e-4_scale1p0'

printf '\n===== PC BASICS SCALE EXTENSION START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s root=%s\n' "$(hostname)" "$$" "$ROOT"
printf 'c2_ext_seeds=%s h3_ext_seeds=%s hard_ext_seeds=%s\n' \
  "$C2_EXT_SEEDS" "$H3_EXT_SEEDS" "$HARD_EXT_SEEDS"

run_required_step() {
  local name="$1"
  shift
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@"
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  if (( rc != 0 )); then
    printf 'required step failed; stopping extension queue: %s rc=%s\n' "$name" "$rc"
    exit "$rc"
  fi
}

mkdir -p \
  "$C2_EXT_EVAL_ROOT" "$C2_EXT_SUMMARY_ROOT" "$C2_ALL5_SUMMARY_ROOT" \
  "$H3_EXT_EVAL_ROOT" "$H3_EXT_SUMMARY_ROOT" "$H3_ALL5_SUMMARY_ROOT" \
  "$C3_EXT_EVAL_ROOT" "$C3_EXT_SUMMARY_ROOT" "$C3_ALL5_SUMMARY_ROOT" \
  "$HARD_EXT_EVAL_ROOT" "$HARD_EXT_SUMMARY_ROOT" "$HARD_ALL5_SUMMARY_ROOT"

printf '\n--- TRAIN_C2_EXT START %s ---\n' "$(date -Is)"
env STAGE=c2_basics_ext ARMS='c2_nodade c2_dade' SEEDS="$C2_EXT_SEEDS" ITERATIONS=32 \
  GPU_IDS='2' MIN_FREE_MIB=12000 GPU_WAIT_SECONDS=21600 \
  LOCK_DIR="$QUEUE_ROOT/c2_gpu_locks" \
  NUM_ENVS=2048 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  WARMSTART="$C2_BC" OUTPUT_ROOT="$C2_TRAIN_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_c2_basics_ext_20261103/train" \
  bash scripts/run_pc_server_basics_train_20261103.sh &
C2_TRAIN_PID=$!

printf '\n--- TRAIN_H3_EXT START %s ---\n' "$(date -Is)"
env STAGE=h3_basics_ext ARMS="$H3_ARM" SEEDS="$H3_EXT_SEEDS" ITERATIONS=100 \
  GPU_IDS='4 5' MIN_FREE_MIB=16500 GPU_WAIT_SECONDS=21600 \
  LOCK_DIR="$QUEUE_ROOT/h3_gpu_locks" \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$H3_TRAIN_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_h3_basics_ext_20261103/train" \
  bash scripts/run_pc_server_basics_train_20261103.sh &
H3_TRAIN_PID=$!

wait "$C2_TRAIN_PID"
C2_TRAIN_RC=$?
wait "$H3_TRAIN_PID"
H3_TRAIN_RC=$?
printf '%s\n' "--- TRAIN_C2_EXT END rc=$C2_TRAIN_RC $(date -Is) ---"
printf '%s\n' "--- TRAIN_H3_EXT END rc=$H3_TRAIN_RC $(date -Is) ---"
if (( C2_TRAIN_RC != 0 || H3_TRAIN_RC != 0 )); then
  printf 'extension phase 1 failed; c2_rc=%s h3_rc=%s\n' "$C2_TRAIN_RC" "$H3_TRAIN_RC"
  exit 1
fi

printf '\n--- TRAIN_C3_EXT START %s ---\n' "$(date -Is)"
env STAGE=frozen_bc_residual_v2_ext ARMS="$C3_ARM" SEEDS="$HARD_EXT_SEEDS" ITERATIONS=100 \
  GPU_IDS='2' MIN_FREE_MIB=16500 GPU_WAIT_SECONDS=21600 \
  LOCK_DIR="$QUEUE_ROOT/c3_gpu_locks" \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$C3_TRAIN_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_v2_ext_20261103/train" \
  bash scripts/run_pc_server_basics_train_20261103.sh &
C3_TRAIN_PID=$!

printf '\n--- TRAIN_HARDENV_EXT START %s ---\n' "$(date -Is)"
env STAGE=frozen_bc_residual_hardenv_v2_ext ARMS="$HARD_ARMS" SEEDS="$HARD_EXT_SEEDS" ITERATIONS=100 \
  GPU_IDS='4 5' MIN_FREE_MIB=16500 GPU_WAIT_SECONDS=21600 \
  LOCK_DIR="$QUEUE_ROOT/hardenv_gpu_locks" \
  NUM_ENVS=4096 NUM_STEPS=256 NUM_MINI_BATCHES=128 \
  OUTPUT_ROOT="$HARD_TRAIN_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_hardenv_v2_ext_20261103/train" \
  bash scripts/run_pc_server_basics_train_20261103.sh &
HARD_TRAIN_PID=$!

wait "$C3_TRAIN_PID"
C3_TRAIN_RC=$?
wait "$HARD_TRAIN_PID"
HARD_TRAIN_RC=$?
printf '%s\n' "--- TRAIN_C3_EXT END rc=$C3_TRAIN_RC $(date -Is) ---"
printf '%s\n' "--- TRAIN_HARDENV_EXT END rc=$HARD_TRAIN_RC $(date -Is) ---"
if (( C3_TRAIN_RC != 0 || HARD_TRAIN_RC != 0 )); then
  printf 'extension phase 2 failed; c3_rc=%s hardenv_rc=%s\n' "$C3_TRAIN_RC" "$HARD_TRAIN_RC"
  exit 1
fi

printf '\n--- AUDIT_FROZEN_BC_EXT START %s ---\n' "$(date -Is)"
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/lowlevel")
checks = {
    root / "pc_server_h3_basics_20261103": (
        "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0",
        (91430, 91431, 91432, 91433, 91434),
    ),
    root / "pc_server_scale_frozen_bc_residual_v2_20261103": (
        "frozen_bc_residual_nodade_lr3e-4_scale1p0",
        (91430, 91431, 91432, 91433, 91434),
    ),
    root / "pc_server_scale_frozen_bc_residual_hardenv_v2_20261103": (
        (
            "frozen_bc_residual_nodade_h1_lr3e-4_scale1p0",
            "frozen_bc_residual_nodade_h2_lr3e-4_scale1p0",
            "frozen_bc_residual_nodade_c4_lr3e-4_scale1p0",
        ),
        (91430, 91431, 91432, 91433, 91434),
    ),
}

failures = []
rows = []
for train_root, (arms, seeds) in checks.items():
    if isinstance(arms, str):
        arms = (arms,)
    for arm in arms:
        for seed in seeds:
            run = train_root / f"{arm}_seed{seed}"
            manifest_path = run / "training_manifest.json"
            if not manifest_path.is_file():
                failures.append(f"{run.name}: missing training_manifest.json")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = manifest.get("frozen_bc_audit") or {}
            groups = manifest.get("optimizer_param_group_counts") or {}
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
                "optimizer_param_group_counts": groups,
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
            if int(groups.get("actor", 0)) <= 0 or int(groups.get("critic", 0)) <= 0:
                failures.append(f"{run.name}: optimizer actor/critic group missing")

output = root / "pc_basics_scale_extension_audit_20261103.json"
output.write_text(
    json.dumps({"runs": rows, "failures": failures}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(output)
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
PY
AUDIT_RC=$?
printf '%s\n' "--- AUDIT_FROZEN_BC_EXT END rc=$AUDIT_RC $(date -Is) ---"
if (( AUDIT_RC != 0 )); then
  exit "$AUDIT_RC"
fi

COMMON_EVAL_ENV=(
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4
  CPU_BASE=240 CPU_STRIDE=16
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  SEED_START=0
)

run_required_step EVAL_C2_EXT env \
  STAGE=c2_basics_ext ARMS='c2_nodade c2_dade' SEEDS="$C2_EXT_SEEDS" \
  TAGS='c2 c2_clean' CHECKPOINTS='initial 0 10 20 30 final' \
  "${COMMON_EVAL_ENV[@]}" SEED_COUNT=64 \
  RUN_ROOT="$C2_TRAIN_ROOT" OUTPUT_ROOT="$C2_EXT_EVAL_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_c2_basics_eval_ext_20261103" \
  bash scripts/eval_pc_server_basics_20261103.sh

run_required_step SUMMARY_C2_EXT \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$C2_EXT_EVAL_ROOT" \
  --output-dir "$C2_EXT_SUMMARY_ROOT" \
  --baseline-arm c2_nodade \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 10=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 30=initial \
  --checkpoint-baseline final=initial

run_required_step EVAL_H3_EXT_INDIST env \
  STAGE=h3_basics_ext ARMS="$H3_ARM" SEEDS="$H3_EXT_SEEDS" \
  TAGS='h3' CHECKPOINTS='initial 0 20 40 60 80 99' \
  "${COMMON_EVAL_ENV[@]}" SEED_COUNT=32 \
  RUN_ROOT="$H3_TRAIN_ROOT" OUTPUT_ROOT="$H3_EXT_EVAL_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_h3_basics_eval_ext_20261103/h3" \
  bash scripts/eval_pc_server_basics_20261103.sh

run_required_step EVAL_H3_EXT_CROSS env \
  STAGE=h3_basics_ext ARMS="$H3_ARM" SEEDS="$H3_EXT_SEEDS" \
  TAGS='default strict125 h4' CHECKPOINTS='initial 99' \
  "${COMMON_EVAL_ENV[@]}" SEED_COUNT=32 \
  RUN_ROOT="$H3_TRAIN_ROOT" OUTPUT_ROOT="$H3_EXT_EVAL_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_h3_basics_eval_ext_20261103/cross" \
  bash scripts/eval_pc_server_basics_20261103.sh

run_required_step SUMMARY_H3_EXT \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$H3_EXT_EVAL_ROOT" \
  --output-dir "$H3_EXT_SUMMARY_ROOT" \
  --baseline-arm "$H3_ARM" \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 40=initial \
  --checkpoint-baseline 60=initial \
  --checkpoint-baseline 80=initial \
  --checkpoint-baseline 99=initial

run_required_step EVAL_C3_EXT env \
  STAGE=frozen_bc_residual_v2_ext ARMS="$C3_ARM" SEEDS="$HARD_EXT_SEEDS" \
  TAGS='default strict125' CHECKPOINTS='initial 0 20 40 60 80 99' \
  "${COMMON_EVAL_ENV[@]}" SEED_COUNT=64 \
  RUN_ROOT="$C3_TRAIN_ROOT" OUTPUT_ROOT="$C3_EXT_EVAL_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_v2_eval_ext_20261103" \
  bash scripts/eval_pc_server_basics_20261103.sh

run_required_step SUMMARY_C3_EXT \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$C3_EXT_EVAL_ROOT" \
  --output-dir "$C3_EXT_SUMMARY_ROOT" \
  --baseline-arm "$C3_ARM" \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 40=initial \
  --checkpoint-baseline 60=initial \
  --checkpoint-baseline 80=initial \
  --checkpoint-baseline 99=initial

for tag in h1 h2 c4; do
  case "$tag" in
    h1) arm="frozen_bc_residual_nodade_h1_lr3e-4_scale1p0" ;;
    h2) arm="frozen_bc_residual_nodade_h2_lr3e-4_scale1p0" ;;
    c4) arm="frozen_bc_residual_nodade_c4_lr3e-4_scale1p0" ;;
  esac
  run_required_step "EVAL_HARDENV_EXT_${tag^^}" env \
    STAGE=frozen_bc_residual_hardenv_v2_ext ARMS="$arm" SEEDS="$HARD_EXT_SEEDS" \
    TAGS="$tag" CHECKPOINTS='initial 0 20 40 60 80 99' \
    "${COMMON_EVAL_ENV[@]}" SEED_COUNT=32 \
    RUN_ROOT="$HARD_TRAIN_ROOT" OUTPUT_ROOT="$HARD_EXT_EVAL_ROOT" \
    LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_hardenv_v2_eval_ext_20261103/$tag" \
    bash scripts/eval_pc_server_basics_20261103.sh
done

run_required_step EVAL_HARDENV_EXT_CROSS env \
  STAGE=frozen_bc_residual_hardenv_v2_ext ARMS="$HARD_ARMS" SEEDS="$HARD_EXT_SEEDS" \
  TAGS='default strict125' CHECKPOINTS='initial 99' \
  "${COMMON_EVAL_ENV[@]}" SEED_COUNT=64 \
  RUN_ROOT="$HARD_TRAIN_ROOT" OUTPUT_ROOT="$HARD_EXT_EVAL_ROOT" \
  LOG_DIR="$ROOT/logs/pc_server_scale_frozen_bc_residual_hardenv_v2_eval_ext_20261103/cross" \
  bash scripts/eval_pc_server_basics_20261103.sh

run_required_step SUMMARY_HARDENV_EXT \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$HARD_EXT_EVAL_ROOT" \
  --output-dir "$HARD_EXT_SUMMARY_ROOT" \
  --baseline-arm frozen_bc_residual_nodade_h1_lr3e-4_scale1p0 \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 40=initial \
  --checkpoint-baseline 60=initial \
  --checkpoint-baseline 80=initial \
  --checkpoint-baseline 99=initial

# Wait for the original three-seed queue to finish training, evaluation and
# its own summary before writing the final five-seed union summaries.
printf '\n--- WAIT_BASICS_QUEUE_FOR_ALL5 %s ---\n' "$(date -Is)"
exec 8> "$ROOT/logs/pc_remaining_basics_20261103/queue.lock"
flock 8
printf '%s\n' "--- BASICS_QUEUE_RELEASED $(date -Is) ---"

run_required_step SUMMARY_C2_ALL5 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$C2_EVAL_ROOT" \
  --eval-root "$C2_EXT_EVAL_ROOT" \
  --output-dir "$C2_ALL5_SUMMARY_ROOT" \
  --baseline-arm c2_nodade \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 10=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 30=initial \
  --checkpoint-baseline final=initial

run_required_step SUMMARY_H3_ALL5 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$H3_EVAL_ROOT" \
  --eval-root "$H3_EXT_EVAL_ROOT" \
  --output-dir "$H3_ALL5_SUMMARY_ROOT" \
  --baseline-arm "$H3_ARM" \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 40=initial \
  --checkpoint-baseline 60=initial \
  --checkpoint-baseline 80=initial \
  --checkpoint-baseline 99=initial

run_required_step SUMMARY_C3_ALL5 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$C3_EVAL_ROOT" \
  --eval-root "$C3_EXT_EVAL_ROOT" \
  --output-dir "$C3_ALL5_SUMMARY_ROOT" \
  --baseline-arm "$C3_ARM" \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 40=initial \
  --checkpoint-baseline 60=initial \
  --checkpoint-baseline 80=initial \
  --checkpoint-baseline 99=initial

run_required_step SUMMARY_HARDENV_ALL5 \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$HARD_H1_EVAL_ROOT" \
  --eval-root "$HARD_H2_EVAL_ROOT" \
  --eval-root "$HARD_C4_EVAL_ROOT" \
  --eval-root "$HARD_CROSS_EVAL_ROOT" \
  --eval-root "$HARD_EXT_EVAL_ROOT" \
  --output-dir "$HARD_ALL5_SUMMARY_ROOT" \
  --baseline-arm frozen_bc_residual_nodade_h1_lr3e-4_scale1p0 \
  --checkpoint-baseline 0=initial \
  --checkpoint-baseline 20=initial \
  --checkpoint-baseline 40=initial \
  --checkpoint-baseline 60=initial \
  --checkpoint-baseline 80=initial \
  --checkpoint-baseline 99=initial

printf '\n===== PC BASICS SCALE EXTENSION DONE %s =====\n' "$(date -Is)"
