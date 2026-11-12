#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PROFILE="${PROFILE:-full}"
if [[ "$PROFILE" != "smoke" && "$PROFILE" != "full" ]]; then
  printf 'PROFILE must be smoke or full, got %s\n' "$PROFILE" >&2
  exit 2
fi

QUEUE_ROOT="$ROOT/logs/pc_multienv_residual_${PROFILE}_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'multienv residual queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

DEMO_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_teacher_demos_${PROFILE}_20261104"
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_${PROFILE}_20261104"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_eval_${PROFILE}_20261104"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_summary_${PROFILE}_20261104"
DIAG_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_dade_diagnostics_${PROFILE}_20261104"

NODADE_ARM='frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0'
DADE_ARM='frozen_bc_residual_dade_multienv_pose_lr3e-4_scale1p0'
NODADE_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_multienv_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
DADE_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_multienv_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"

if [[ "$PROFILE" == "smoke" ]]; then
  TRAIN_SEEDS="${TRAIN_SEEDS:-94030}"
  NUM_ENVS=8
  STEPS_PER_ENV=64
  MAX_EPOCHS=2
  PATIENCE=2
  PRETRAIN_BATCH_SIZE=256
  DIAG_SEED_COUNT=4
  DIAG_MAX_STEPS=64
else
  TRAIN_SEEDS="${TRAIN_SEEDS:-94030 94031 94032}"
  NUM_ENVS=16
  STEPS_PER_ENV=4096
  MAX_EPOCHS=200
  PATIENCE=30
  PRETRAIN_BATCH_SIZE=4096
  DIAG_SEED_COUNT=8
  DIAG_MAX_STEPS=256
fi

declare -A DEMO_CONFIG=(
  [c2]="$ROOT/configs/lowlevel/odarrl_control_pc_c2_alos.json"
  [c3]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
  [h1]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json"
  [h2]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h2_speed_2p0_2p25.json"
  [h3]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h3_speed_2p25_2p4.json"
  [c4]="$ROOT/configs/lowlevel/odarrl_control_pc_c4_stress.json"
)
declare -A DEMO_SEED=(
  [c2]=95000
  [c3]=95100
  [h1]=95200
  [h2]=95300
  [h3]=95400
  [c4]=95500
)
DEMO_ORDER=(c2 c3 h1 h2 h3 c4)
ARMS=("$NODADE_ARM" "$DADE_ARM")

failures=0
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
    failures=$((failures + 1))
  fi
  return "$rc"
}

run_evidence_step() {
  local name="$1"
  local log="$2"
  shift 2
  printf '\n--- %s START %s ---\n' "$name" "$(date -Is)"
  "$@" > "$log" 2>&1
  local rc=$?
  printf '%s\n' "--- $name END rc=$rc $(date -Is) ---"
  return "$rc"
}

start_collect() {
  local name="$1"
  local cpu="$2"
  local output="$DEMO_ROOT/$name"
  local log="$QUEUE_ROOT/collect_${name}.log"
  if [[ -f "$output/demos.npz" && -f "$output/demos.manifest.json" ]]; then
    printf 'SKIP_COLLECT name=%s output=%s\n' "$name" "$output"
    return 0
  fi
  printf 'START_COLLECT name=%s cpu=%s seed=%s output=%s\n' \
    "$name" "$cpu" "${DEMO_SEED[$name]}" "$output" >&2
  taskset -c "$cpu" env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" scripts/collect_pc_bc_demos.py \
    --config "${DEMO_CONFIG[$name]}" \
    --output "$output" \
    --num-envs "$NUM_ENVS" --steps-per-env "$STEPS_PER_ENV" \
    --device cpu --dtype float64 \
    --seed-start "${DEMO_SEED[$name]}" > "$log" 2>&1 &
  COLLECT_PID=$!
}

collect_demos() {
  local pids=()
  local names=()
  local index=0
  local cpu_ranges=(320-327 328-335 336-343 344-351 352-359 360-367)
  for name in "${DEMO_ORDER[@]}"; do
    local output="$DEMO_ROOT/$name"
    if [[ -f "$output/demos.npz" && -f "$output/demos.manifest.json" ]]; then
      printf 'SKIP_COLLECT name=%s output=%s\n' "$name" "$output"
      index=$((index + 1))
      continue
    fi
    start_collect "$name" "${cpu_ranges[$index]}"
    pids+=("$COLLECT_PID")
    names+=("$name")
    index=$((index + 1))
  done
  local position
  for position in "${!pids[@]}"; do
    if wait "${pids[$position]}"; then
      printf 'END_COLLECT name=%s rc=0\n' "${names[$position]}"
    else
      printf 'END_COLLECT name=%s rc!=0 log=%s\n' \
        "${names[$position]}" "$QUEUE_ROOT/collect_${names[$position]}.log"
      failures=$((failures + 1))
    fi
  done
}

pretrain_seed() {
  local arm="$1"
  local seed="$2"
  local config="$3"
  local output="$TRAIN_ROOT/${arm}_seed${seed}"
  local log="$QUEUE_ROOT/pretrain_${arm}_seed${seed}.log"
  if [[ -f "$output/final.pt" && -f "$output/training_manifest.json" ]]; then
    printf 'SKIP_PRETRAIN arm=%s seed=%s output=%s\n' "$arm" "$seed" "$output"
    return 0
  fi
  run_step "PRETRAIN_${arm^^}_SEED${seed}" "$log" \
    taskset -c 320-351 env OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
    "$PY" scripts/pretrain_pc_residual.py \
    --training-config "$config" \
    --demos "$DEMO_ROOT/merged/demos.npz" \
    --output "$output" \
    --batch-size "$PRETRAIN_BATCH_SIZE" \
    --max-epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --learning-rate 1e-3 \
    --weight-decay 1e-5 \
    --output-l2 1e-4 \
    --validation-fraction 0.2 \
    --seed "$seed" \
    --device cpu \
    --overwrite
}

run_eval() {
  local name="$1"
  local tags="$2"
  local seed_count="$3"
  run_step "$name" "$QUEUE_ROOT/${name}.log" env \
    STAGE="multienv_${PROFILE}_20261104" \
    ARMS="$NODADE_ARM $DADE_ARM" \
    SEEDS="$TRAIN_SEEDS" \
    TAGS="$tags" CHECKPOINTS='initial final' \
    DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4 \
    CPU_BASE=96 CPU_STRIDE=16 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    SEED_START=0 SEED_COUNT="$seed_count" \
    RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" \
    LOG_DIR="$QUEUE_ROOT/eval" \
    bash scripts/eval_pc_server_basics_20261103.sh
}

run_history_diagnostics() {
  local arm="$1"
  local config="$2"
  local tags="$3"
  local seed
  local tag
  for tag in $tags; do
    local env_config
    case "$tag" in
      default) env_config="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json" ;;
      strict125) env_config="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_strict125.json" ;;
      *) printf 'unknown diagnostic tag: %s\n' "$tag" >&2; failures=$((failures + 1)); continue ;;
    esac
    for seed in $TRAIN_SEEDS; do
      local checkpoint="$TRAIN_ROOT/${arm}_seed${seed}/final.pt"
      local output="$DIAG_ROOT/${arm}/${tag}_seed${seed}"
      local log="$QUEUE_ROOT/diag_${arm}_${tag}_seed${seed}.log"
      if [[ ! -f "$checkpoint" ]]; then
        printf 'MISSING_DIAGNOSTIC_CHECKPOINT %s\n' "$checkpoint"
        failures=$((failures + 1))
        continue
      fi
      if [[ -f "$output/summary.json" ]]; then
        printf 'SKIP_DIAGNOSTIC output=%s\n' "$output"
        continue
      fi
      run_step "DIAGNOSTIC_${arm^^}_${tag^^}_SEED${seed}" "$log" \
        taskset -c 320-335 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PY" scripts/diagnose_pc_dade_history.py \
        --training-config "$config" \
        --checkpoint "$checkpoint" \
        --environment-config "$env_config" \
        --output "$output" \
        --seed-start 0 --seed-count "$DIAG_SEED_COUNT" \
        --max-steps "$DIAG_MAX_STEPS" --device cpu --dtype float64
    done
  done
}

printf '\n===== PC MULTIENV SUPERVISED RESIDUAL START profile=%s %s =====\n' \
  "$PROFILE" "$(date -Is)"
printf 'host=%s pid=%s root=%s\n' "$(hostname)" "$$" "$ROOT"
printf 'train_seeds=%s demo_envs=%s\n' "$TRAIN_SEEDS" "${DEMO_ORDER[*]}"
mkdir -p "$DEMO_ROOT" "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "$DIAG_ROOT"

collect_demos
if (( failures > 0 )); then
  printf 'teacher collection failures=%s; stopping before merge\n' "$failures"
  exit 1
fi

if [[ -f "$DEMO_ROOT/merged/demos.npz" && -f "$DEMO_ROOT/merged/demos.manifest.json" ]]; then
  printf 'SKIP_MERGE output=%s\n' "$DEMO_ROOT/merged"
else
  merge_inputs=()
  for name in "${DEMO_ORDER[@]}"; do
    merge_inputs+=("$DEMO_ROOT/$name/demos.npz")
  done
  if ! run_step MERGE_DEMOS "$QUEUE_ROOT/merge.log" \
    "$PY" scripts/merge_pc_bc_demos.py \
    --inputs "${merge_inputs[@]}" \
    --output "$DEMO_ROOT/merged" --overwrite; then
    exit 1
  fi
fi

for seed in $TRAIN_SEEDS; do
  pretrain_seed "$NODADE_ARM" "$seed" "$NODADE_CONFIG"
  pretrain_seed "$DADE_ARM" "$seed" "$DADE_CONFIG"
done
if (( failures > 0 )); then
  printf 'pretraining failures=%s; stopping before audit\n' "$failures"
  exit 1
fi

run_step AUDIT_MULTIENV_RESIDUAL "$QUEUE_ROOT/audit.log" \
  env TRAIN_ROOT="$TRAIN_ROOT" TRAIN_SEEDS="$TRAIN_SEEDS" \
  NODADE_ARM="$NODADE_ARM" DADE_ARM="$DADE_ARM" "$PY" - <<'PY'
import json
import os
from pathlib import Path

import torch

root = Path(os.environ["TRAIN_ROOT"])
seeds = tuple(int(value) for value in os.environ["TRAIN_SEEDS"].split())
nodade_arm = os.environ["NODADE_ARM"]
dade_arm = os.environ["DADE_ARM"]
failures = []
rows = []
for arm in (nodade_arm, dade_arm):
    for seed in seeds:
        run = root / f"{arm}_seed{seed}"
        manifest_path = run / "training_manifest.json"
        final_path = run / "final.pt"
        initial_path = run / "model_initial.pt"
        if not manifest_path.is_file() or not final_path.is_file() or not initial_path.is_file():
            failures.append(f"{run.name}: missing manifest/initial/final artifacts")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        keys = tuple(checkpoint.get("model_state_dict", {}))
        row = {
            "run": run.name,
            "status": manifest.get("status"),
            "frozen_bc_parameter_max_abs_drift": manifest.get(
                "frozen_bc_parameter_max_abs_drift"
            ),
            "dade_enabled": (manifest.get("training_result") or {}).get("dade_enabled"),
            "has_dade_actor": any(key.startswith("actor_dade.") for key in keys),
            "has_dade_head": any(key.startswith("actor_dade_head.") for key in keys),
            "has_current_head": any(key.startswith("actor_current_head.") for key in keys),
            "final_checkpoint": str(final_path),
        }
        rows.append(row)
        if row["status"] != "completed":
            failures.append(f"{run.name}: status={row['status']!r}")
        if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
            failures.append(f"{run.name}: frozen BC drift is not zero")
        if arm == dade_arm:
            if row["dade_enabled"] is not True:
                failures.append(f"{run.name}: DADE is not enabled")
            if not (row["has_dade_actor"] and row["has_dade_head"] and row["has_current_head"]):
                failures.append(f"{run.name}: DADE/head/current-head state is incomplete")
        elif row["dade_enabled"] is True:
            failures.append(f"{run.name}: no-DADE arm unexpectedly enabled DADE")

output = root / "audit.json"
output.write_text(
    json.dumps({"runs": rows, "failures": failures}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(output)
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
PY
if (( failures > 0 )); then
  exit 1
fi

if [[ "$PROFILE" == "smoke" ]]; then
  run_eval EVAL_SMOKE 'default strict125' 4
else
  run_eval EVAL_PRIMARY 'default strict125' 64
  run_eval EVAL_HARD 'h1 h2 h3 c4' 32
fi
if (( failures > 0 )); then
  printf 'evaluation failures=%s; stopping before DADE diagnostics\n' "$failures"
  exit 1
fi

run_history_diagnostics "$NODADE_ARM" "$NODADE_CONFIG" default
if [[ "$PROFILE" == "smoke" ]]; then
  run_history_diagnostics "$DADE_ARM" "$DADE_CONFIG" 'default strict125'
else
  run_history_diagnostics "$DADE_ARM" "$DADE_CONFIG" 'default strict125'
fi

gate_rc=0
if ! run_evidence_step DADE_HISTORY_GATE "$QUEUE_ROOT/history_gate.log" \
  "$PY" scripts/check_pc_dade_history_gate.py \
  --diagnostic-root "$DIAG_ROOT/$DADE_ARM" \
  --output "$DIAG_ROOT/dade_history_gate.json" \
  --threshold 1e-3 \
  --mode current-only --mode history-zero; then
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
run_step SUMMARY_MULTIENV_RESIDUAL "$QUEUE_ROOT/summary.log" "${summary_cmd[@]}"

env SUMMARY_ROOT="$SUMMARY_ROOT" DIAG_ROOT="$DIAG_ROOT" TRAIN_ROOT="$TRAIN_ROOT" \
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

summary_root = Path(os.environ["SUMMARY_ROOT"])
diag_root = Path(os.environ["DIAG_ROOT"])
train_root = Path(os.environ["TRAIN_ROOT"])
payload = {
    "summary": str(summary_root / "summary.json"),
    "summary_markdown": str(summary_root / "summary.md"),
    "training_audit": str(train_root / "audit.json"),
    "dade_history_gate": str(diag_root / "dade_history_gate.json"),
    "diagnostics_root": str(diag_root),
}
(summary_root / "index.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(summary_root / "index.json")
PY

printf '\n===== PC MULTIENV SUPERVISED RESIDUAL DONE profile=%s %s failures=%s gate_rc=%s =====\n' \
  "$PROFILE" "$(date -Is)" "$failures" "$gate_rc"
exit "$((failures > 0))"
