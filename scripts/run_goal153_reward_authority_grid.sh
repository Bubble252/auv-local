#!/usr/bin/env bash
# Goal153: evaluate trained reward arms under a residual-authority grid.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal153_reward_curriculum_20261111/authority_grid}"
TRAIN_ROOT="${TRAIN_ROOT:-$REPO_ROOT/artifacts/lowlevel/goal153_reward_curriculum_20261111/pilot/train}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal153_authority_grid}"
GPUS="${GPUS:-3 4 6}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
ARMS="${ARMS:-R1 R2}"
VARIANTS="${VARIANTS:-full near10_2 far_off_only near_only mid_only full_scale_0.5}"
SEEDS="${SEEDS:-86201}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"

cd "$REPO_ROOT"
mkdir -p "$ROOT" "$LOG_ROOT"
read -r -a GPU_ARRAY <<<"$GPUS"
read -r -a ARM_ARRAY <<<"$ARMS"
read -r -a VARIANT_ARRAY <<<"$VARIANTS"
read -r -a SEED_ARRAY <<<"$SEEDS"
pids=()
job_index=0

wait_one() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one eval job failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

checkpoint_for() {
  local arm="$1" seed="$2"
  local seeded="$TRAIN_ROOT/${arm}_C1_seed${seed}/final.zip"
  if [[ -f "$seeded" ]]; then
    printf '%s' "$seeded"
  else
    printf '%s' "$TRAIN_ROOT/${arm}_C1/final.zip"
  fi
}

launch() {
  local tag="$1" config="$2" env_config="$3" scenario="$4" mode="$5" checkpoint="$6" scale="$7" gate_mode="$8"
  local out="$ROOT/$tag/$scenario"
  local gpu="${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
  local extra=()
  if [[ "$mode" == "policy" ]]; then
    extra+=(--checkpoint "$checkpoint")
  fi
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP $tag/$scenario"
    job_index=$((job_index + 1))
    return
  fi
  echo "START $tag/$scenario gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3:$GLOBAL_SRC" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$config" \
      --environment-config "$env_config" \
      --output "$out" \
      --trace-output "$out" \
      --trace-reward-breakdown \
      --rich-trace \
      --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
      --success-frame both \
      --device cuda --dtype float64 \
      --action-mode "$mode" \
      --residual-scale "$scale" \
      --learner-terminal-gate-mode "$gate_mode" \
      --sb3-root "$SB3" \
      --overwrite "${extra[@]}" >"$LOG_ROOT/${tag//\//_}_${scenario}.log" 2>&1 &
  pids+=("$!")
  job_index=$((job_index + 1))
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_one
    pids=("${pids[@]:1}")
  fi
}

for scenario in default strict125; do
  if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
  launch zero "configs/lowlevel/goal153/train/R0_C1_sac.json" "$env_config" "$scenario" zero-residual "" 1.0 none
  for arm in "${ARM_ARRAY[@]}"; do
    for variant in "${VARIANT_ARRAY[@]}"; do
      config="configs/lowlevel/goal153/authority/${arm}_${variant}.json"
      scale="$("$PY" - <<PY
import json
print(json.load(open("$config"))["evaluation"].get("residual_scale", 1.0))
PY
)"
      gate_mode="$("$PY" - <<PY
import json
print(json.load(open("$config")).get("learner_terminal_gate_mode", "none"))
PY
)"
      for seed in "${SEED_ARRAY[@]}"; do
        ckpt="$(checkpoint_for "$arm" "$seed")"
        seed_tag="seed${seed}"
        launch "$arm/$variant/$seed_tag" "$config" "$env_config" "$scenario" policy "$ckpt" "$scale" "$gate_mode"
      done
    done
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || { echo "one eval job failed; inspect $LOG_ROOT" >&2; exit 1; }
done
echo "Goal153 reward-authority grid complete: $ROOT"
