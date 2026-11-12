#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_multienv_dade_gpu_seeds_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'multienv DADE GPU seed queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

DEMO="$ROOT/artifacts/lowlevel/pc_multienv_teacher_demos_full_20261104/merged/demos.npz"
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104"
CONFIG="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_multienv_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
ARM='frozen_bc_residual_dade_multienv_pose_lr3e-4_scale1p0'
SEEDS="${SEEDS:-94031 94032}"
GPUS="${GPUS:-2 4}"
CPU_BASE="${CPU_BASE:-368}"
CPU_STRIDE="${CPU_STRIDE:-8}"

read -r -a seed_list <<< "$SEEDS"
read -r -a gpu_list <<< "$GPUS"
if (( ${#seed_list[@]} == 0 )); then
  printf 'SEEDS must not be empty\n' >&2
  exit 2
fi
if (( ${#seed_list[@]} != ${#gpu_list[@]} )); then
  printf 'SEEDS and GPUS must have the same length: %s vs %s\n' "$SEEDS" "$GPUS" >&2
  exit 2
fi
if [[ ! -f "$DEMO" ]]; then
  printf 'missing merged teacher demos: %s\n' "$DEMO" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'missing training config: %s\n' "$CONFIG" >&2
  exit 1
fi

printf '\n===== MULTIENV DADE GPU SEEDS START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s seeds=%s gpus=%s\n' "$(hostname)" "$$" "$SEEDS" "$GPUS"

mkdir -p "$TRAIN_ROOT" "$QUEUE_ROOT"
pids=()
names=()
for index in "${!seed_list[@]}"; do
  seed="${seed_list[$index]}"
  gpu="${gpu_list[$index]}"
  output="$TRAIN_ROOT/${ARM}_seed${seed}"
  log="$QUEUE_ROOT/pretrain_seed${seed}_gpu${gpu}.log"
  if [[ -f "$output/final.pt" && -f "$output/training_manifest.json" ]]; then
    printf 'SKIP_PRETRAIN seed=%s output=%s\n' "$seed" "$output"
    continue
  fi
  free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "$free_mib" ]] || (( free_mib < 10000 )); then
    printf 'GPU_NOT_READY gpu=%s free_mib=%s\n' "$gpu" "${free_mib:-unknown}" >&2
    exit 1
  fi
  cpu_start=$((CPU_BASE + index * CPU_STRIDE))
  cpu_range="$cpu_start-$((cpu_start + CPU_STRIDE - 1))"
  printf 'START_PRETRAIN seed=%s gpu=%s free_mib=%s cpu=%s output=%s\n' \
    "$seed" "$gpu" "$free_mib" "$cpu_range" "$output"
  taskset -c "$cpu_range" env CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/pretrain_pc_residual.py \
    --training-config "$CONFIG" \
    --demos "$DEMO" \
    --output "$output" \
    --batch-size 4096 \
    --max-epochs 200 \
    --patience 30 \
    --learning-rate 1e-3 \
    --weight-decay 1e-5 \
    --output-l2 1e-4 \
    --validation-fraction 0.2 \
    --seed "$seed" \
    --device cuda \
    --overwrite > "$log" 2>&1 &
  pids+=("$!")
  names+=("$seed")
done

failures=0
for index in "${!pids[@]}"; do
  seed="${names[$index]}"
  if wait "${pids[$index]}"; then
    printf 'END_PRETRAIN seed=%s rc=0 %s\n' "$seed" "$(date -Is)"
  else
    rc=$?
    printf 'END_PRETRAIN seed=%s rc=%s log=%s\n' \
      "$seed" "$rc" "$QUEUE_ROOT/pretrain_seed${seed}_gpu${gpu_list[$index]}.log"
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  printf 'STOP_AFTER_TRAIN_FAILURES failures=%s\n' "$failures"
  exit 1
fi

env TRAIN_ROOT="$TRAIN_ROOT" ARM="$ARM" SEEDS="$SEEDS" "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TRAIN_ROOT"])
arm = os.environ["ARM"]
seeds = [int(value) for value in os.environ["SEEDS"].split()]
rows = []
failures = []
for seed in seeds:
    run = root / f"{arm}_seed{seed}"
    manifest_path = run / "training_manifest.json"
    if not manifest_path.is_file():
        failures.append(f"{run.name}: missing manifest")
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = manifest.get("training_result") or {}
    row = {
        "seed": seed,
        "status": manifest.get("status"),
        "samples": (manifest.get("dataset") or {}).get("sample_count"),
        "best_epoch": result.get("best_epoch"),
        "epochs_completed": result.get("epochs_completed"),
        "initial_validation_loss": result.get("initial_validation_loss"),
        "best_validation_loss": result.get("best_validation_loss"),
        "frozen_bc_parameter_max_abs_drift": manifest.get(
            "frozen_bc_parameter_max_abs_drift"
        ),
        "training_device": (manifest.get("training_config") or {}).get("device"),
        "dade_enabled": result.get("dade_enabled"),
    }
    rows.append(row)
    if row["status"] != "completed":
        failures.append(f"{run.name}: status={row['status']!r}")
    if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
        failures.append(f"{run.name}: frozen BC drift is not zero")
    if row["dade_enabled"] is not True:
        failures.append(f"{run.name}: DADE is not enabled")

payload = {"arm": arm, "runs": rows, "failures": failures, "device": "cuda"}
output = root / "audit_dade_gpu_seeds.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(output)
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
PY
audit_rc=$?
printf 'MULTIENV_DADE_GPU_SEEDS_DONE audit_rc=%s %s\n' "$audit_rc" "$(date -Is)"
exit "$audit_rc"
