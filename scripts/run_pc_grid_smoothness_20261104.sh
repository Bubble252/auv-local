#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_grid_smoothness_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'grid smoothness queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

DEMO_ROOT="$ROOT/artifacts/lowlevel/pc_speed_current_grid_teacher_demos_20261104"
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_grid_smoothness_supervised_residual_20261104"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_grid_smoothness_eval_20261104"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_grid_smoothness_summary_20261104"
REPORT_ROOT="$ROOT/artifacts/lowlevel/pc_grid_smoothness_report_20261104"
BASELINE_TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_speed_current_grid_supervised_residual_20261104"
BASELINE_EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_speed_current_grid_eval_20261104"

BASELINE_ARM='frozen_bc_residual_nodade_grid_lr3e-4_scale1p0'
SMOOTH_ARM='frozen_bc_residual_nodade_grid_smooth1e-3_lr3e-4_scale1p0'
L2_ARM='frozen_bc_residual_nodade_grid_smooth1e-3_l2_1e-2_lr3e-4_scale1p0'
SMOOTH_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_grid_frozen_bc_residual_nodade_smooth1e-3_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
L2_CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_grid_frozen_bc_residual_nodade_smooth1e-3_l2_1e-2_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"

TRAIN_SEEDS="${TRAIN_SEEDS:-94030 94031 94032 94033 94034}"
SEED_COUNT="${SEED_COUNT:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
MAX_TRAIN_WORKERS="${MAX_TRAIN_WORKERS:-4}"
GPU_FREE_MIB="${GPU_FREE_MIB:-10000}"
CPU_BASE="${CPU_BASE:-224}"
CPU_STRIDE="${CPU_STRIDE:-8}"

DEMOS="$DEMO_ROOT/merged/demos.npz"
DEMO_MANIFEST="$DEMO_ROOT/merged/demos.manifest.json"

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

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "$REPORT_ROOT" "$QUEUE_ROOT"

if [[ ! -f "$DEMOS" || ! -f "$DEMO_MANIFEST" ]]; then
  printf 'missing merged grid teacher demos: %s\n' "$DEMOS" >&2
  exit 1
fi
if [[ ! -d "$BASELINE_EVAL_ROOT" ]]; then
  printf 'missing baseline evaluation root: %s\n' "$BASELINE_EVAL_ROOT" >&2
  exit 1
fi
if [[ ! -f "$SMOOTH_CONFIG" || ! -f "$L2_CONFIG" ]]; then
  printf 'missing smoothness training config\n' >&2
  exit 1
fi

if [[ "${GPUS:-auto}" == "auto" ]]; then
  mapfile -t GPU_LIST < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
      awk -F', ' -v minimum="$GPU_FREE_MIB" '$2 >= minimum {print $1}'
  )
else
  read -r -a GPU_LIST <<< "$GPUS"
fi
if (( ${#GPU_LIST[@]} == 0 )); then
  printf 'no free GPUs with at least %s MiB\n' "$GPU_FREE_MIB" >&2
  exit 1
fi
TRAIN_WORKERS="${TRAIN_WORKERS:-$MAX_TRAIN_WORKERS}"
if (( TRAIN_WORKERS > ${#GPU_LIST[@]} )); then
  TRAIN_WORKERS="${#GPU_LIST[@]}"
fi

read -r -a seed_list <<< "$TRAIN_SEEDS"
jobs=()
for seed in "${seed_list[@]}"; do
  jobs+=("$SMOOTH_ARM|$seed|$SMOOTH_CONFIG|0.001|0.0001")
  jobs+=("$L2_ARM|$seed|$L2_CONFIG|0.001|0.01")
done

printf '\n===== PC GRID SMOOTHNESS START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s seeds=%s gpus=%s workers=%s eval_seeds=%s\n' \
  "$(hostname)" "$$" "$TRAIN_SEEDS" "${GPU_LIST[*]:0:TRAIN_WORKERS}" \
  "$TRAIN_WORKERS" "$SEED_COUNT"

train_failures=0
for ((batch_start = 0; batch_start < ${#jobs[@]}; batch_start += TRAIN_WORKERS)); do
  batch_pids=()
  batch_labels=()
  for ((slot = 0; slot < TRAIN_WORKERS; slot++)); do
    index=$((batch_start + slot))
    if (( index >= ${#jobs[@]} )); then
      break
    fi
    IFS='|' read -r arm seed config smoothness output_l2 <<< "${jobs[$index]}"
    gpu="${GPU_LIST[$slot]}"
    output="$TRAIN_ROOT/${arm}_seed${seed}"
    log="$QUEUE_ROOT/pretrain_${arm}_seed${seed}_gpu${gpu}.log"
    if [[ -f "$output/final.pt" && -f "$output/training_manifest.json" ]]; then
      printf 'SKIP_PRETRAIN arm=%s seed=%s output=%s\n' "$arm" "$seed" "$output"
      continue
    fi
    free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ -z "$free_mib" ]] || (( free_mib < GPU_FREE_MIB )); then
      printf 'GPU_NOT_READY gpu=%s free_mib=%s need_mib=%s\n' \
        "$gpu" "${free_mib:-unknown}" "$GPU_FREE_MIB" >&2
      train_failures=$((train_failures + 1))
      continue
    fi
    cpu_start=$((CPU_BASE + slot * CPU_STRIDE))
    cpu_range="$cpu_start-$((cpu_start + CPU_STRIDE - 1))"
    printf 'START_PRETRAIN arm=%s seed=%s gpu=%s free_mib=%s cpu=%s smoothness=%s output_l2=%s\n' \
      "$arm" "$seed" "$gpu" "$free_mib" "$cpu_range" "$smoothness" "$output_l2"
    taskset -c "$cpu_range" env CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" scripts/pretrain_pc_residual.py \
      --training-config "$config" \
      --demos "$DEMOS" \
      --output "$output" \
      --batch-size 4096 \
      --max-epochs 200 \
      --patience 30 \
      --learning-rate 1e-3 \
      --weight-decay 1e-5 \
      --output-l2 "$output_l2" \
      --smoothness-l2 "$smoothness" \
      --validation-fraction 0.2 \
      --seed "$seed" \
      --device cuda \
      --overwrite > "$log" 2>&1 &
    batch_pids+=("$!")
    batch_labels+=("${arm}_seed${seed}_gpu${gpu}")
  done
  for position in "${!batch_pids[@]}"; do
    label="${batch_labels[$position]}"
    if wait "${batch_pids[$position]}"; then
      printf 'END_PRETRAIN %s rc=0\n' "$label"
    else
      printf 'END_PRETRAIN %s rc!=0 log=%s\n' "$label" "$QUEUE_ROOT/pretrain_${label}.log"
      train_failures=$((train_failures + 1))
    fi
  done
  if (( train_failures > 0 )); then
    printf 'TRAINING_FAILURES=%s; stopping before remaining batches\n' "$train_failures" >&2
    exit 1
  fi
done

if ! run_required_step AUDIT_GRID_SMOOTHNESS "$QUEUE_ROOT/audit.log" \
  env TRAIN_ROOT="$TRAIN_ROOT" TRAIN_SEEDS="$TRAIN_SEEDS" \
  SMOOTH_ARM="$SMOOTH_ARM" L2_ARM="$L2_ARM" "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TRAIN_ROOT"])
seeds = tuple(int(value) for value in os.environ["TRAIN_SEEDS"].split())
arms = {
    os.environ["SMOOTH_ARM"]: {"smoothness_l2": 0.001, "output_l2": 0.0001},
    os.environ["L2_ARM"]: {"smoothness_l2": 0.001, "output_l2": 0.01},
}
training_cells = {"pc_grid_low_weak", "pc_grid_low_strong", "pc_grid_high_weak"}
heldout_cells = {"pc_grid_high_strong"}
failures = []
rows = []
for arm, expected in arms.items():
    for seed in seeds:
        run = root / f"{arm}_seed{seed}"
        manifest_path = run / "training_manifest.json"
        final_path = run / "final.pt"
        initial_path = run / "model_initial.pt"
        if not manifest_path.is_file():
            failures.append(f"{run.name}: missing manifest")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("training_config") or {}
        result = manifest.get("training_result") or {}
        dataset = manifest.get("dataset") or {}
        labels = manifest.get("label_statistics") or {}
        names = {str(item.get("config_name")) for item in dataset.get("configs") or []}
        prediction_abs_mean = labels.get("prediction_abs_mean") or []
        row = {
            "run": run.name,
            "status": manifest.get("status"),
            "seed": config.get("seed"),
            "device": config.get("device"),
            "smoothness_l2": config.get("smoothness_l2"),
            "output_l2": config.get("output_l2"),
            "dade_enabled": result.get("dade_enabled"),
            "frozen_bc_parameter_max_abs_drift": manifest.get("frozen_bc_parameter_max_abs_drift"),
            "training_cells": sorted(names),
            "prediction_abs_mean": prediction_abs_mean,
        }
        rows.append(row)
        if row["status"] != "completed":
            failures.append(f"{run.name}: status={row['status']!r}")
        if int(row["seed"] or -1) != seed:
            failures.append(f"{run.name}: manifest seed={row['seed']!r}")
        if row["device"] != "cuda":
            failures.append(f"{run.name}: training device={row['device']!r}")
        if row["dade_enabled"] is not False:
            failures.append(f"{run.name}: DADE unexpectedly enabled")
        if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
            failures.append(f"{run.name}: frozen BC drift is not zero")
        if float(row["smoothness_l2"] or 0.0) != expected["smoothness_l2"]:
            failures.append(f"{run.name}: smoothness_l2={row['smoothness_l2']!r}")
        if float(row["output_l2"] or 0.0) != expected["output_l2"]:
            failures.append(f"{run.name}: output_l2={row['output_l2']!r}")
        if names != training_cells:
            failures.append(f"{run.name}: unexpected training cells {sorted(names)}")
        if names & heldout_cells:
            failures.append(f"{run.name}: held-out high+strong cell leaked into training")
        if not prediction_abs_mean or max(float(value) for value in prediction_abs_mean) <= 1e-8:
            failures.append(f"{run.name}: residual predictions are identically zero")
        if not final_path.is_file() or not initial_path.is_file():
            failures.append(f"{run.name}: missing initial/final checkpoint")

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

if ! run_required_step EVAL_GRID_SMOOTHNESS "$QUEUE_ROOT/eval.log" env \
  STAGE=grid_smoothness_20261104 ARMS="$SMOOTH_ARM $L2_ARM" SEEDS="$TRAIN_SEEDS" \
  TAGS='grid_lw grid_ls grid_hw grid_hs default strict125 c4 h4' \
  CHECKPOINTS='final' DEVICE=cpu DTYPE=float64 EVAL_WORKERS="$EVAL_WORKERS" \
  CPU_BASE=96 CPU_STRIDE=16 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  SEED_START=0 SEED_COUNT="$SEED_COUNT" RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$EVAL_ROOT" \
  LOG_DIR="$QUEUE_ROOT/eval" bash scripts/eval_pc_server_basics_20261103.sh; then
  exit 1
fi

if ! run_required_step SUMMARY_GRID_SMOOTHNESS "$QUEUE_ROOT/summary.log" \
  "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
  --eval-root "$BASELINE_EVAL_ROOT" --eval-root "$EVAL_ROOT" \
  --output-dir "$SUMMARY_ROOT" --baseline-arm "supervised_$BASELINE_ARM" \
  --checkpoint initial --checkpoint final --checkpoint-baseline final=initial \
  --eval-seed-count "$SEED_COUNT"; then
  exit 1
fi

gate_rc=0
if ! "$PY" scripts/summarize_pc_grid_smoothness_20261104.py \
  --summary "$SUMMARY_ROOT/summary.json" --output-dir "$REPORT_ROOT" \
  --baseline-arm "$BASELINE_ARM" --treatment-arm "$SMOOTH_ARM" \
  --treatment-arm "$L2_ARM" --require-pass > "$QUEUE_ROOT/report.log" 2>&1; then
  gate_rc=1
  printf 'GRID_SMOOTHNESS_GATE_NOT_PASSED; preserving report as evidence\n' >&2
fi

env REPORT_ROOT="$REPORT_ROOT" SUMMARY_ROOT="$SUMMARY_ROOT" TRAIN_ROOT="$TRAIN_ROOT" \
  EVAL_ROOT="$EVAL_ROOT" BASELINE_EVAL_ROOT="$BASELINE_EVAL_ROOT" \
  BASELINE_TRAIN_ROOT="$BASELINE_TRAIN_ROOT" GATE_RC="$gate_rc" "$PY" - <<'PY'
import json
import os
from pathlib import Path

report_root = Path(os.environ["REPORT_ROOT"])
payload = {
    "training_audit": str(Path(os.environ["TRAIN_ROOT"]) / "audit.json"),
    "summary": str(Path(os.environ["SUMMARY_ROOT"]) / "summary.json"),
    "eval_root": os.environ["EVAL_ROOT"],
    "baseline_eval_root": os.environ["BASELINE_EVAL_ROOT"],
    "baseline_train_root": os.environ["BASELINE_TRAIN_ROOT"],
    "report": str(report_root / "report.json"),
    "report_markdown": str(report_root / "report.md"),
    "gate_returncode": int(os.environ["GATE_RC"]),
}
(report_root / "index.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(report_root / "index.json")
PY

printf '\n===== PC GRID SMOOTHNESS DONE %s gate_rc=%s =====\n' \
  "$(date -Is)" "$gate_rc"
exit 0
