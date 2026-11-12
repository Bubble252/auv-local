#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_ROOT="$ROOT/logs/pc_supervised_residual_20261104"
mkdir -p "$QUEUE_ROOT"
exec 9> "$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'supervised residual queue already running: %s\n' "$QUEUE_ROOT/queue.lock" >&2
  exit 0
fi
exec >> "$QUEUE_ROOT/queue.log" 2>&1

DEMO_ROOT="$ROOT/artifacts/lowlevel/pc_teacher_residual_demos_20261104"
TRAIN_ROOT="$ROOT/artifacts/lowlevel/pc_supervised_residual_20261104"
EVAL_ROOT="$ROOT/artifacts/lowlevel/pc_supervised_residual_eval_20261104"
SUMMARY_ROOT="$ROOT/artifacts/lowlevel/pc_supervised_residual_summary_20261104"
TRAIN_SEEDS='91430 91431 91432'

declare -A CONFIG=(
  [frozen_bc_residual_nodade_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_h1_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_h1_speed_1p8_2p0_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_h2_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_h2_speed_2p0_2p25_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_h3_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_h3_speed_2p25_2p4_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
  [frozen_bc_residual_nodade_c4_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_ppo_pc_c4_stress_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
)
declare -A ENV_CONFIG=(
  [frozen_bc_residual_nodade_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
  [frozen_bc_residual_nodade_h1_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h1_speed_1p8_2p0.json"
  [frozen_bc_residual_nodade_h2_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h2_speed_2p0_2p25.json"
  [frozen_bc_residual_nodade_h3_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_pc_c3_random_h3_speed_2p25_2p4.json"
  [frozen_bc_residual_nodade_c4_lr3e-4_scale1p0]="$ROOT/configs/lowlevel/odarrl_control_pc_c4_stress.json"
)
declare -A DEMO_SEED=(
  [frozen_bc_residual_nodade_lr3e-4_scale1p0]=93000
  [frozen_bc_residual_nodade_h1_lr3e-4_scale1p0]=93100
  [frozen_bc_residual_nodade_h2_lr3e-4_scale1p0]=93200
  [frozen_bc_residual_nodade_h3_lr3e-4_scale1p0]=93300
  [frozen_bc_residual_nodade_c4_lr3e-4_scale1p0]=93400
)
declare -A SUMMARY_ARM=(
  [frozen_bc_residual_nodade_lr3e-4_scale1p0]="supervised_frozen_bc_residual_nodade_lr3e-4_scale1p0"
  [frozen_bc_residual_nodade_h1_lr3e-4_scale1p0]="supervised_frozen_bc_residual_nodade_h1_lr3e-4_scale1p0"
  [frozen_bc_residual_nodade_h2_lr3e-4_scale1p0]="supervised_frozen_bc_residual_nodade_h2_lr3e-4_scale1p0"
  [frozen_bc_residual_nodade_h3_lr3e-4_scale1p0]="supervised_frozen_bc_residual_nodade_h3_lr3e-4_scale1p0"
  [frozen_bc_residual_nodade_c4_lr3e-4_scale1p0]="supervised_frozen_bc_residual_nodade_c4_lr3e-4_scale1p0"
)
ARM_ORDER=(
  frozen_bc_residual_nodade_h3_lr3e-4_scale1p0
  frozen_bc_residual_nodade_lr3e-4_scale1p0
  frozen_bc_residual_nodade_h1_lr3e-4_scale1p0
  frozen_bc_residual_nodade_h2_lr3e-4_scale1p0
  frozen_bc_residual_nodade_c4_lr3e-4_scale1p0
)

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

collect_demos() {
  local arm="$1"
  local output="$DEMO_ROOT/$arm"
  local log="$QUEUE_ROOT/collect_${arm}.log"
  if [[ -f "$output/demos.npz" && -f "$output/demos.manifest.json" ]]; then
    printf 'SKIP_COLLECT arm=%s output=%s\n' "$arm" "$output"
    return 0
  fi
  run_step "COLLECT_${arm^^}" "$log" \
    taskset -c 16-31 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" scripts/collect_pc_bc_demos.py \
    --config "${ENV_CONFIG[$arm]}" \
    --output "$output" \
    --num-envs 16 --steps-per-env 4096 \
    --device cpu --dtype float64 \
    --seed-start "${DEMO_SEED[$arm]}"
}

pretrain_seed() {
  local arm="$1"
  local seed="$2"
  local demos="$DEMO_ROOT/$arm/demos.npz"
  local output="$TRAIN_ROOT/${arm}_seed${seed}"
  local log="$QUEUE_ROOT/pretrain_${arm}_seed${seed}.log"
  if [[ -f "$output/final.pt" && -f "$output/training_manifest.json" ]]; then
    printf 'SKIP_PRETRAIN arm=%s seed=%s output=%s\n' "$arm" "$seed" "$output"
    return 0
  fi
  run_step "PRETRAIN_${arm^^}_SEED${seed}" "$log" \
    taskset -c 32-47 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" scripts/pretrain_pc_residual.py \
    --training-config "${CONFIG[$arm]}" \
    --demos "$demos" \
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
    --overwrite
}

eval_arm() {
  local arm="$1"
  local tags="$2"
  local checkpoints="$3"
  local seed_count="$4"
  local output_root="$EVAL_ROOT/$arm"
  local log_dir="$QUEUE_ROOT/eval_$arm"
  local log="$QUEUE_ROOT/eval_${arm}.log"
  run_step "EVAL_${arm^^}" "$log" env \
    STAGE=supervised_residual_20261104 ARMS="$arm" SEEDS="$TRAIN_SEEDS" \
    TAGS="$tags" CHECKPOINTS="$checkpoints" \
    DEVICE=cpu DTYPE=float64 EVAL_WORKERS=4 \
    CPU_BASE=304 CPU_STRIDE=16 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    SEED_START=0 SEED_COUNT="$seed_count" \
    RUN_ROOT="$TRAIN_ROOT" OUTPUT_ROOT="$output_root" LOG_DIR="$log_dir" \
    bash scripts/eval_pc_server_basics_20261103.sh
}

summarize_arm() {
  local arm="$1"
  local output_root="$EVAL_ROOT/$arm"
  local summary_dir="$SUMMARY_ROOT/$arm"
  local log="$QUEUE_ROOT/summary_${arm}.log"
  run_step "SUMMARY_${arm^^}" "$log" \
    "$PY" scripts/summarize_pc_server_evals_basics_20261103.py \
    --eval-root "$output_root" \
    --output-dir "$summary_dir" \
    --baseline-arm "${SUMMARY_ARM[$arm]}" \
    --checkpoint-baseline final=initial
}

printf '\n===== PC SUPERVISED TEACHER-RESIDUAL START %s =====\n' "$(date -Is)"
printf 'host=%s pid=%s root=%s\n' "$(hostname)" "$$" "$ROOT"
printf 'train_seeds=%s arms=%s\n' "$TRAIN_SEEDS" "${ARM_ORDER[*]}"
mkdir -p "$DEMO_ROOT" "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT"

for arm in "${ARM_ORDER[@]}"; do
  collect_demos "$arm"
done
if (( failures > 0 )); then
  printf 'demo collection failures=%s; stopping before pretraining\n' "$failures"
  exit 1
fi

for arm in "${ARM_ORDER[@]}"; do
  for seed in $TRAIN_SEEDS; do
    pretrain_seed "$arm" "$seed"
  done
done
if (( failures > 0 )); then
  printf 'pretraining failures=%s; stopping before evaluation\n' "$failures"
  exit 1
fi

run_step AUDIT_SUPERVISED_RESIDUAL "$QUEUE_ROOT/audit.log" \
  "$PY" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/lowlevel/pc_supervised_residual_20261104")
seeds = (91430, 91431, 91432)
arms = (
    "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0",
    "frozen_bc_residual_nodade_lr3e-4_scale1p0",
    "frozen_bc_residual_nodade_h1_lr3e-4_scale1p0",
    "frozen_bc_residual_nodade_h2_lr3e-4_scale1p0",
    "frozen_bc_residual_nodade_c4_lr3e-4_scale1p0",
)
failures = []
rows = []
for arm in arms:
    for seed in seeds:
        run = root / f"{arm}_seed{seed}"
        manifest_path = run / "training_manifest.json"
        if not manifest_path.is_file():
            failures.append(f"{run.name}: missing training_manifest.json")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = {
            "run": run.name,
            "status": manifest.get("status"),
            "best_epoch": (manifest.get("training_result") or {}).get("best_epoch"),
            "best_validation_loss": (manifest.get("training_result") or {}).get(
                "best_validation_loss"
            ),
            "frozen_bc_parameter_max_abs_drift": manifest.get(
                "frozen_bc_parameter_max_abs_drift"
            ),
            "initial_checkpoint": str(run / "model_initial.pt"),
            "final_checkpoint": str(run / "final.pt"),
        }
        rows.append(row)
        if row["status"] != "completed":
            failures.append(f"{run.name}: status={row['status']!r}")
        if not (run / "model_initial.pt").is_file():
            failures.append(f"{run.name}: missing model_initial.pt")
        if not (run / "final.pt").is_file():
            failures.append(f"{run.name}: missing final.pt")
        if float(row["frozen_bc_parameter_max_abs_drift"] or 0.0) != 0.0:
            failures.append(f"{run.name}: frozen BC drift is not zero")

output = Path("artifacts/lowlevel/pc_supervised_residual_audit_20261104.json")
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

for arm in "${ARM_ORDER[@]}"; do
  case "$arm" in
    frozen_bc_residual_nodade_lr3e-4_scale1p0)
      eval_arm "$arm" 'default strict125' 'initial final' 64
      ;;
    frozen_bc_residual_nodade_h1_lr3e-4_scale1p0)
      eval_arm "$arm" 'h1' 'initial final' 32
      eval_arm "$arm" 'default strict125' 'initial final' 32
      ;;
    frozen_bc_residual_nodade_h2_lr3e-4_scale1p0)
      eval_arm "$arm" 'h2' 'initial final' 32
      eval_arm "$arm" 'default strict125' 'initial final' 32
      ;;
    frozen_bc_residual_nodade_h3_lr3e-4_scale1p0)
      eval_arm "$arm" 'h3' 'initial final' 32
      eval_arm "$arm" 'default strict125' 'initial final' 32
      ;;
    frozen_bc_residual_nodade_c4_lr3e-4_scale1p0)
      eval_arm "$arm" 'c4' 'initial final' 32
      eval_arm "$arm" 'default strict125' 'initial final' 32
      ;;
  esac
done
if (( failures > 0 )); then
  printf 'evaluation failures=%s; stopping before summaries\n' "$failures"
  exit 1
fi

for arm in "${ARM_ORDER[@]}"; do
  summarize_arm "$arm"
done

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/lowlevel/pc_supervised_residual_summary_20261104")
index = []
for summary_path in sorted(root.glob("*/summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    index.append(
        {
            "arm_root": summary_path.parent.name,
            "summary": str(summary_path),
            "evaluation_roots": summary.get("eval_roots"),
            "checkpoint_baseline": summary.get("checkpoint_baseline"),
        }
    )
(root / "index.json").write_text(
    json.dumps({"summaries": index}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(root / "index.json")
PY

printf '\n===== PC SUPERVISED TEACHER-RESIDUAL DONE %s failures=%s =====\n' \
  "$(date -Is)" "$failures"
exit "$((failures > 0))"
