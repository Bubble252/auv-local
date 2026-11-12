#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_multienv_dade_extension_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'multienv DADE extension supervisor already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_eval_full_20261104"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_summary_full_20261104"
SUMMARY_3SEED_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_summary_full_3seed_20261104"
DIAG_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_dade_diagnostics_full_20261104"
GPU_QUEUE_ROOT="$ROOT/logs/pc_multienv_dade_gpu_seeds_20261104"
GPU_AUDIT="$TRAIN_ROOT/audit_dade_gpu_seeds.json"

NODADE_ARM='frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0'
DADE_ARM='frozen_bc_residual_dade_multienv_pose_lr3e-4_scale1p0'
NODADE_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_multienv_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
DADE_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_multienv_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"

FIRST_BATCH_PID="${FIRST_BATCH_PID:-4124160}"
CPU_SEED_PID="${CPU_SEED_PID:-3879051}"
MAIN_QUEUE_PID="${MAIN_QUEUE_PID:-3873284}"
EXTRA_SEEDS="${EXTRA_SEEDS:-94033 94034}"
EXTRA_GPUS="${EXTRA_GPUS:-1 5}"
EXTRA_CPU_BASE="${EXTRA_CPU_BASE:-352}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-28800}"

wait_for_paths() {
  local label="$1"
  local pid="$2"
  shift 2
  local deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
  while true; do
    local missing=0
    local path
    for path in "$@"; do
      if [[ ! -f "$path" ]]; then
        missing=1
        break
      fi
    done
    if (( missing == 0 )); then
      printf 'WAIT_OK label=%s %s\n' "$label" "$(date -Is)"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'WAIT_PROCESS_GONE label=%s pid=%s %s\n' "$label" "$pid" "$(date -Is)"
      return 1
    fi
    if (( SECONDS >= deadline )); then
      printf 'WAIT_TIMEOUT label=%s pid=%s %s\n' "$label" "$pid" "$(date -Is)"
      return 124
    fi
    sleep 30
  done
}

wait_for_pid_exit() {
  local label="$1"
  local pid="$2"
  local deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
  while kill -0 "$pid" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      printf 'WAIT_PID_TIMEOUT label=%s pid=%s %s\n' "$label" "$pid" "$(date -Is)"
      return 124
    fi
    sleep 30
  done
  printf 'WAIT_PID_GONE label=%s pid=%s %s\n' "$label" "$pid" "$(date -Is)"
}

run_step() {
  local name="$1"
  local log="$2"
  shift 2
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@" > "$log" 2>&1
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  if (( rc != 0 )); then
    printf 'FAILED %s log=%s\n' "$name" "$log"
    tail -n 80 "$log" || true
  fi
  return "$rc"
}

validate_dade_run() {
  local run="$1"
  "$PY" - "$run" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
manifest = json.loads((run / "training_manifest.json").read_text(encoding="utf-8"))
result = manifest.get("training_result") or {}
failures = []
if manifest.get("status") != "completed":
    failures.append(f"status={manifest.get('status')!r}")
if float(manifest.get("frozen_bc_parameter_max_abs_drift") or 0.0) != 0.0:
    failures.append("frozen BC drift is not zero")
if result.get("dade_enabled") is not True:
    failures.append("DADE is not enabled")
if not (run / "final.pt").is_file() or not (run / "model_initial.pt").is_file():
    failures.append("missing initial/final checkpoint")
print(json.dumps({"run": str(run), "failures": failures}, indent=2, sort_keys=True))
raise SystemExit(bool(failures))
PY
}

printf '\n===== PC MULTIENV DADE SUPERVISOR START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s first_batch=%s cpu_seed=%s main_queue=%s extra_seeds=%s extra_gpus=%s\n' \
  "$(hostname)" "$$" "$FIRST_BATCH_PID" "$CPU_SEED_PID" "$MAIN_QUEUE_PID" "$EXTRA_SEEDS" "$EXTRA_GPUS"

if ! wait_for_paths FIRST_GPU_BATCH "$FIRST_BATCH_PID" \
  "$TRAIN_ROOT/${DADE_ARM}_seed94031/final.pt" \
  "$TRAIN_ROOT/${DADE_ARM}_seed94032/final.pt"; then
  printf 'First GPU batch did not complete cleanly; stopping.\n'
  exit 1
fi

if [[ -f "$GPU_AUDIT" ]]; then
  cp -f "$GPU_AUDIT" "$TRAIN_ROOT/audit_dade_gpu_seeds_94031_94032.json"
fi

printf '\n--- EXTRA_DADE_GPU_BATCH START %s ---\n' "$(date -Is)"
env SEEDS="$EXTRA_SEEDS" GPUS="$EXTRA_GPUS" CPU_BASE="$EXTRA_CPU_BASE" \
  bash scripts/run_pc_multienv_dade_gpu_seeds_20261104.sh
extra_rc=$?
printf '%s\n' "--- EXTRA_DADE_GPU_BATCH END rc=$extra_rc $(date -Is) ---"
if (( extra_rc != 0 )); then
  printf 'Extra GPU batch failed; stopping before evaluation.\n'
  exit "$extra_rc"
fi
if [[ -f "$GPU_AUDIT" ]]; then
  cp -f "$GPU_AUDIT" "$TRAIN_ROOT/audit_dade_gpu_seeds_94033_94034.json"
fi

cpu_run="$TRAIN_ROOT/${DADE_ARM}_seed94030"
if ! wait_for_paths CPU_DADE_SEED94030 "$CPU_SEED_PID" \
  "$cpu_run/final.pt" "$cpu_run/model_initial.pt" "$cpu_run/training_manifest.json"; then
  printf 'CPU DADE seed 94030 did not complete cleanly; main queue remains paused.\n'
  exit 1
fi
if ! validate_dade_run "$cpu_run"; then
  printf 'CPU DADE seed 94030 failed audit; main queue remains paused.\n'
  exit 1
fi

main_state="$(ps -o stat= -p "$MAIN_QUEUE_PID" 2>/dev/null | tr -d ' ' || true)"
if [[ -z "$main_state" ]]; then
  printf 'Main queue pid %s is gone; cannot resume.\n' "$MAIN_QUEUE_PID"
  exit 1
fi
if [[ "$main_state" == T* ]]; then
  kill -CONT "$MAIN_QUEUE_PID"
  printf 'RESUMED_MAIN_QUEUE pid=%s previous_state=%s %s\n' \
    "$MAIN_QUEUE_PID" "$main_state" "$(date -Is)"
else
  printf 'MAIN_QUEUE_ALREADY_RUNNING pid=%s state=%s\n' "$MAIN_QUEUE_PID" "$main_state"
fi

if ! wait_for_pid_exit MAIN_QUEUE "$MAIN_QUEUE_PID"; then
  printf 'Main queue did not finish; stopping before extra evaluation.\n'
  exit 1
fi

if [[ -d "$SUMMARY_ROOT" && ! -d "$SUMMARY_3SEED_ROOT" ]]; then
  cp -a "$SUMMARY_ROOT" "$SUMMARY_3SEED_ROOT"
fi
if [[ -f "$DIAG_ROOT/dade_history_gate.json" ]]; then
  cp -f "$DIAG_ROOT/dade_history_gate.json" "$DIAG_ROOT/dade_history_gate_3seed.json"
fi

if ! run_step EXTRA_EVAL "$QUEUE_ROOT/extra_eval.log" env \
  STAGE="multienv_full_20261104" \
  ARMS="$NODADE_ARM $DADE_ARM" \
  SEEDS="$EXTRA_SEEDS" \
  TAGS='default strict125' CHECKPOINTS='initial final' \
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4 \
  CPU_BASE=96 CPU_STRIDE=16 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  SEED_START=0 SEED_COUNT=64 \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" \
  LOG_DIR="$QUEUE_ROOT/eval" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'Extra primary evaluation failed; stopping before summary.\n'
  exit 1
fi

if ! run_step EXTRA_EVAL_HARD "$QUEUE_ROOT/extra_eval_hard.log" env \
  STAGE="multienv_full_20261104" \
  ARMS="$NODADE_ARM $DADE_ARM" \
  SEEDS="$EXTRA_SEEDS" \
  TAGS='h1 h2 h3 c4' CHECKPOINTS='initial final' \
  DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4 \
  CPU_BASE=96 CPU_STRIDE=16 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  SEED_START=0 SEED_COUNT=32 \
  RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" \
  LOG_DIR="$QUEUE_ROOT/eval" \
  bash scripts/eval_pc_server_basics_20261103.sh; then
  printf 'Extra hard evaluation failed; stopping before summary.\n'
  exit 1
fi

run_extra_diagnostics() {
  local arm="$1"
  local config="$2"
  local tags="$3"
  local seed
  local tag
  local env_config
  for tag in $tags; do
    case "$tag" in
      default) env_config="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json" ;;
      strict125) env_config="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json" ;;
      *) printf 'unknown diagnostic tag: %s\n' "$tag" >&2; return 2 ;;
    esac
    for seed in $EXTRA_SEEDS; do
      local checkpoint="$TRAIN_ROOT/${arm}_seed${seed}/final.pt"
      local output="$DIAG_ROOT/${arm}/${tag}_seed${seed}"
      if [[ -f "$output/summary.json" ]]; then
        printf 'SKIP_DIAGNOSTIC output=%s\n' "$output"
        continue
      fi
      run_step "DIAGNOSTIC_${arm^^}_${tag^^}_SEED${seed}" \
        "$QUEUE_ROOT/diag_${arm}_${tag}_seed${seed}.log" \
        taskset -c 320-335 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PY" scripts/diagnose_pc_dade_history.py \
        --training-config "$config" \
        --checkpoint "$checkpoint" \
        --environment-config "$env_config" \
        --output "$output" \
        --seed-start 0 --seed-count 8 \
        --max-steps 256 --device cpu --dtype float64 || return 1
    done
  done
}

if ! run_extra_diagnostics "$NODADE_ARM" "$NODADE_CONFIG" default; then
  printf 'Extra no-DADE diagnostics failed; stopping before summary.\n'
  exit 1
fi
if ! run_extra_diagnostics "$DADE_ARM" "$DADE_CONFIG" 'default strict125'; then
  printf 'Extra DADE diagnostics failed; stopping before summary.\n'
  exit 1
fi

gate_rc=0
if ! run_step DADE_HISTORY_GATE_5SEED "$QUEUE_ROOT/history_gate_5seed.log" \
  "$PY" scripts/check_pc_dade_history_gate.py \
  --diagnostic-root "$DIAG_ROOT/$DADE_ARM" \
  --output "$DIAG_ROOT/dade_history_gate.json" \
  --threshold 1e-3 --mode current-only --mode history-zero; then
  gate_rc=1
  printf 'DADE history gate failed; preserving result as negative evidence\n'
fi

summary_cmd=(
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py
  --eval-root "$EVAL_ROOT"
  --output-dir "$SUMMARY_ROOT"
  --baseline-arm "supervised_${NODADE_ARM}"
  --checkpoint-baseline final=initial
)
summary_rc=0
if ! run_step SUMMARY_MULTIENV_5SEED "$QUEUE_ROOT/summary_5seed.log" "${summary_cmd[@]}"; then
  summary_rc=1
fi

env SUMMARY_ROOT="$SUMMARY_ROOT" SUMMARY_3SEED_ROOT="$SUMMARY_3SEED_ROOT" \
  DIAG_ROOT="$DIAG_ROOT" TRAIN_ROOT="$TRAIN_ROOT" "$PY" - <<'PY'
import json
import os
from pathlib import Path

summary_root = Path(os.environ["SUMMARY_ROOT"])
diag_root = Path(os.environ["DIAG_ROOT"])
train_root = Path(os.environ["TRAIN_ROOT"])
payload = {
    "summary": str(summary_root / "summary.json"),
    "summary_markdown": str(summary_root / "summary.md"),
    "summary_3seed": str(Path(os.environ["SUMMARY_3SEED_ROOT"])),
    "training_audit_3seed": str(train_root / "audit.json"),
    "training_audit_dade_gpu_94031_94032": str(
        train_root / "audit_dade_gpu_seeds_94031_94032.json"
    ),
    "training_audit_dade_gpu_94033_94034": str(
        train_root / "audit_dade_gpu_seeds_94033_94034.json"
    ),
    "dade_history_gate": str(diag_root / "dade_history_gate.json"),
    "dade_history_gate_3seed": str(diag_root / "dade_history_gate_3seed.json"),
    "diagnostics_root": str(diag_root),
}
(summary_root / "index.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(summary_root / "index.json")
PY

printf '\n===== PC MULTIENV DADE SUPERVISOR DONE %s extra_rc=%s gate_rc=%s summary_rc=%s =====\n' \
  "$(date -Is)" "$extra_rc" "$gate_rc" "$summary_rc"
exit "$((summary_rc > 0))"
