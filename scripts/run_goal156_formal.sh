#!/usr/bin/env bash
# Goal156: full-residual SAC 5M training + eval-time authority gate.
#
# Refuses to start until Goal155 is closed and the caller acknowledges formal.
set -euo pipefail

if [[ "${ALLOW_FORMAL:-0}" != "1" || "${GOAL155_CLOSED:-0}" != "1" ]]; then
  echo "Refusing Goal156 formal: set ALLOW_FORMAL=1 GOAL155_CLOSED=1 after the Goal155 Gate is recorded."
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal156_full_train_eval_gate}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal156_full_train_eval_gate}"
TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"
SEEDS="${SEEDS:-86201 86202 86203}"
TOTAL_ENV_TRANSITIONS="${TOTAL_ENV_TRANSITIONS:-5000000}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"

cd "$REPO_ROOT"
mkdir -p "$ROOT/train" "$ROOT/eval" "$LOG_ROOT"
read -r -a SEED_ARRAY <<<"$SEEDS"
read -r -a GPU_ARRAY <<<"$TRAIN_GPUS"

if [[ ! -f configs/lowlevel/goal156/manifest.json ]]; then
  "$PY" scripts/build_goal156_full_train_configs.py >/dev/null
fi

train_pids=()
for index in "${!SEED_ARRAY[@]}"; do
  seed="${SEED_ARRAY[$index]}"
  config="configs/lowlevel/goal156/full_seed${seed}.json"
  out="$ROOT/train/full_seed${seed}"
  log="$LOG_ROOT/train_full_seed${seed}.log"
  mkdir -p "$out"
  if [[ -f "$out/final.zip" ]]; then
    echo "SKIP train full_seed${seed}"
    continue
  fi
  gpu="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  echo "START train full_seed${seed} gpu=$gpu"
  (
    export PYTHONPATH="src:$SB3:$GLOBAL_SRC"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_pc_command_residual_sac.py \
      --config "$config" --output "$out" \
      --total-env-transitions "$TOTAL_ENV_TRANSITIONS" \
      --device cuda --engineering-smoke --overwrite
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  train_pids+=("$!")
done
for pid in "${train_pids[@]}"; do
  wait "$pid" || { echo "one Goal156 training job failed; inspect $LOG_ROOT" >&2; exit 1; }
done

pids=()
job_index=0
wait_one() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one Goal156 eval job failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

queue_eval() {
  local out="$1" log="$2"
  shift 2
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP $(basename "$out")"
    return
  fi
  mkdir -p "$out"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3:$GLOBAL_SRC" \
    "$@" >"$log" 2>&1 &
  pids+=("$!")
  job_index=$((job_index + 1))
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_one
    pids=("${pids[@]:1}")
  fi
}

for scenario in default strict125; do
  if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
  out="$ROOT/eval/zero/$scenario"
  queue_eval "$out" "$LOG_ROOT/eval_zero_${scenario}.log" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config configs/lowlevel/goal156/full_seed86201.json \
      --environment-config "$env_config" \
      --output "$out" --trace-output "$out" \
      --trace-reward-breakdown --rich-trace \
      --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
      --success-frame both --device cpu --dtype float64 \
      --action-mode zero-residual --residual-scale 1.0 \
      --sb3-root "$SB3" --overwrite
done

for seed in "${SEED_ARRAY[@]}"; do
  checkpoint="$ROOT/train/full_seed${seed}/final.zip"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  for variant in full near10_2 mid_only far_off_only; do
    if [[ "$variant" == "full" ]]; then
      config="configs/lowlevel/goal156/full_seed${seed}.json"
      gate_mode="none"
    else
      config="configs/lowlevel/goal154/authority/${variant}.json"
      gate_mode="$("$PY" - <<PY
import json
print(json.load(open("$config"))["learner_terminal_gate_mode"])
PY
)"
    fi
    for scenario in default strict125; do
      if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
      out="$ROOT/eval/${variant}_${seed}/$scenario"
      queue_eval "$out" "$LOG_ROOT/eval_${variant}_${seed}_${scenario}.log" \
        "$PY" scripts/evaluate_pc_sac.py \
          --config "$config" \
          --environment-config "$env_config" \
          --output "$out" --trace-output "$out" \
          --trace-reward-breakdown --rich-trace \
          --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
          --success-frame both --device cpu --dtype float64 \
          --action-mode policy --residual-scale 1.0 \
          --checkpoint "$checkpoint" \
          --learner-terminal-gate-mode "$gate_mode" \
          --sb3-root "$SB3" --overwrite
    done
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || { echo "one Goal156 eval job failed; inspect $LOG_ROOT" >&2; exit 1; }
done

"$PY" scripts/analyze_pc_goal155_authority_training.py \
  --root "$ROOT" --eval-root "$ROOT/eval"
echo "Goal156 formal complete: $ROOT"
