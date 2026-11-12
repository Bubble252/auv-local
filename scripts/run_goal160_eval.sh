#!/usr/bin/env bash
# Goal160: evaluate three action-parameterization arms against zero.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal160_action_parameterization_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/eval}
PARALLEL=${PARALLEL:-6}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
LEVELS=${LEVELS:-"nominal current07 current10 current15"}
ARMS=${ARMS:-"A_actuator_near10_2 B_command_near10_2 C_command_terminal_train"}
SEEDS=${SEEDS:-"86201 86202 86203"}
RADII=${RADII:-"default:2.0 strict125:1.25"}

cd "$REPO"
mkdir -p "$EVAL_ROOT/logs" "$EVAL_ROOT/manifests"

policy_config() {
  case "$1" in
    A_actuator_near10_2) echo "configs/lowlevel/goal154/authority/near10_2.json" ;;
    B_command_near10_2|C_command_terminal_train) echo "configs/lowlevel/goal160/eval_command_near10_2.json" ;;
    *) return 1 ;;
  esac
}
env_config() {
  case "$1" in
    A_actuator_near10_2) echo "configs/lowlevel/goal160/env_h3_actuator_default_eval.json" ;;
    B_command_near10_2|C_command_terminal_train) echo "configs/lowlevel/goal160/env_h3_command_near10_2_eval.json" ;;
    *) return 1 ;;
  esac
}
expected_interface() {
  case "$1" in
    A_actuator_near10_2) echo "actuator" ;;
    B_command_near10_2|C_command_terminal_train) echo "command" ;;
    *) return 1 ;;
  esac
}

for arm in $ARMS; do
  for seed in $SEEDS; do
    if [[ ! -f "$TRAIN_ROOT/$arm/seed$seed/final.zip" ]]; then
      echo "missing checkpoint: $TRAIN_ROOT/$arm/seed$seed/final.zip" >&2
      exit 2
    fi
    manifest="$TRAIN_ROOT/$arm/seed$seed/training_manifest.json"
    "$PY" - "$manifest" "$(expected_interface "$arm")" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
expected = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("direct_commands_active") is not False:
    raise SystemExit(f"{path}: direct_commands_active is not false")
if payload.get("action_interface") != expected:
    raise SystemExit(f"{path}: action_interface={payload.get('action_interface')!r}, expected {expected!r}")
speed = payload.get("speed_contract") or {}
if speed.get("ok") is not True:
    raise SystemExit(f"{path}: speed_contract is not ok: {speed}")
if float(speed.get("reference_speed_mps", -1.0)) > 2.5 + 1e-12:
    raise SystemExit(f"{path}: reference speed exceeds 2.5 m/s")
print(f"preflight PASS {path}")
PY
  done
done

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  for arm in $ARMS; do
    sha256sum "$(policy_config "$arm")" "$(env_config "$arm")"
  done
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >"$EVAL_ROOT/manifests/run_manifest.txt" 2>&1

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_index=0
for arm in $ARMS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      for seed in $SEEDS; do
        for mode in zero policy; do
          echo "$arm ${radius%%:*} ${radius##*:} $level $seed $mode $job_index" >>"$JOB_FILE"
          job_index=$((job_index + 1))
        done
      done
    done
  done
done

run_one() {
  local arm="$1" radius_label="$2" radius_value="$3" level="$4" seed="$5" mode="$6" index="$7"
  local output="$EVAL_ROOT/$arm/$radius_label/$level/${mode}_seed${seed}"
  local log="$EVAL_ROOT/logs/${arm}_${radius_label}_${level}_${mode}_seed${seed}.log"
  local current_speed_args=() shear_args=() checkpoint_args=() gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"

  case "$level" in
    nominal) ;;
    current07) current_speed_args=(--curriculum-current-speed-range-mps 0.275 0.50); shear_args=(--curriculum-shear-scale 0.75) ;;
    current10) current_speed_args=(--curriculum-current-speed-range-mps 0.30 0.55); shear_args=(--curriculum-shear-scale 1.0) ;;
    current15) current_speed_args=(--curriculum-current-speed-range-mps 0.375 0.675); shear_args=(--curriculum-shear-scale 1.5) ;;
    *) echo "unknown level: $level" >&2; return 2 ;;
  esac
  if [[ "$mode" == "policy" ]]; then
    checkpoint_args=(--checkpoint "$TRAIN_ROOT/$arm/seed$seed/final.zip")
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $arm/$level/$mode/seed$seed (summary exists)"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing to overwrite non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START arm=$arm radius=$radius_label level=$level mode=$mode seed=$seed gpu=$gpu"
  if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$(policy_config "$arm")" \
      --environment-config "$(env_config "$arm")" \
      --curriculum-stage C3 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
      --eval-goal-radius-m "$radius_value" \
      --device "$DEVICE" \
      --dtype float64 \
      --action-mode "$([ "$mode" == "zero" ] && echo zero-residual || echo policy)" \
      --residual-scale 1.0 \
      --learner-terminal-gate-mode none \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --overwrite \
      "${checkpoint_args[@]}" \
      "${current_speed_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1; then
    echo "FAILED arm=$arm radius=$radius_label level=$level mode=$mode seed=$seed (see $log)" >&2
    return 1
  fi
  echo "DONE arm=$arm radius=$radius_label level=$level mode=$mode seed=$seed"
}

export PY REPO SB3 ROOT TRAIN_ROOT EVAL_ROOT SEED_START SEED_COUNT DEVICE GPUS
export -f run_one policy_config env_config

echo "launching $(wc -l <"$JOB_FILE") Goal160 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 7 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE" || true

missing=0
while read -r arm radius_label radius_value level seed mode index; do
  if [[ ! -f "$EVAL_ROOT/$arm/$radius_label/$level/${mode}_seed${seed}/summary.json" ]]; then
    echo "MISSING $arm/$radius_label/$level/$mode/seed$seed" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"

printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$EVAL_ROOT/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal160 evaluations are missing; inspect $EVAL_ROOT/logs" >&2
  exit 1
fi
echo "Goal160 evaluation complete: $EVAL_ROOT"
