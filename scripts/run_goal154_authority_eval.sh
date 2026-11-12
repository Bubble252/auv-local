#!/usr/bin/env bash
# Goal154 eval-only authority confound audit on Goal86 T2-SAC checkpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal154_residual_authority_20261111/eval}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal154_authority}"
T2_ROOT="${T2_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/goal86/screen}"
GPUS="${GPUS:-3 4 6 7}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
SEEDS="${SEEDS:-86201 86202 86203}"
VARIANTS="${VARIANTS:-full near10_2 far_off_only near_only mid_only full_scale_0.5 full_scale_0.75}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"

cd "$REPO_ROOT"
mkdir -p "$ROOT" "$LOG_ROOT"
read -r -a GPU_ARRAY <<<"$GPUS"
read -r -a SEED_ARRAY <<<"$SEEDS"
read -r -a VARIANT_ARRAY <<<"$VARIANTS"
pids=()
names=()
job_index=0

wait_one() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one eval job failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

launch() {
  local variant="$1" seed="$2" scenario="$3" config="$4" env_config="$5" checkpoint="$6" scale="$7" gate_mode="$8"
  local seed_tag="t2_sac${seed#8620}"
  if [[ "$seed" == "86201" ]]; then seed_tag="t2_sac"; fi
  local out="$ROOT/$variant/$seed_tag/$scenario"
  local gpu="${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP $variant/$seed_tag/$scenario"
    job_index=$((job_index + 1))
    return
  fi
  local args=(
    "$PY" scripts/evaluate_pc_sac.py
    --config "$config"
    --environment-config "$env_config"
    --output "$out"
    --trace-output "$out"
    --trace-reward-breakdown
    --rich-trace
    --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT"
    --success-frame both
    --device cuda --dtype float64
    --action-mode policy
    --residual-scale "$scale"
    --learner-terminal-gate-mode "$gate_mode"
    --sb3-root "$SB3"
    --overwrite
  )
  if [[ "$variant" == "off_all" ]]; then
    # Replace the action-mode pair explicitly; do not use array string
    # substitution, which treats "--action-mode" and "policy" as separate words.
    local replaced=()
    for token in "${args[@]}"; do
      if [[ "$token" == "--action-mode" ]]; then
        replaced+=("--action-mode" "zero-residual")
        continue
      fi
      if [[ "$token" == "policy" ]]; then
        continue
      fi
      replaced+=("$token")
    done
    args=( "${replaced[@]}" )
  else
    args+=(--checkpoint "$checkpoint")
  fi
  echo "START $variant/$seed_tag/$scenario gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3:$GLOBAL_SRC" \
    "${args[@]}" >"$LOG_ROOT/${variant}_${seed_tag}_${scenario}.log" 2>&1 &
  pids+=("$!")
  names+=("$variant/$seed_tag/$scenario")
  job_index=$((job_index + 1))
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_one
    pids=("${pids[@]:1}")
  fi
}

for variant in "${VARIANT_ARRAY[@]}"; do
  config="configs/lowlevel/goal154/authority/${variant}.json"
  if [[ "$variant" == "off_all" ]]; then
    config="configs/lowlevel/goal154/authority/full.json"
  fi
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
    if [[ "$seed" == "86201" ]]; then ckpt="$T2_ROOT/t2_sac/final.zip"; else ckpt="$T2_ROOT/t2_sac_seed${seed}/final.zip"; fi
    for scenario in default strict125; do
      if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
      if [[ "$variant" == "off_all" ]]; then
        launch "$variant" "$seed" "$scenario" "$config" "$env_config" "" 1.0 none
      else
        launch "$variant" "$seed" "$scenario" "$config" "$env_config" "$ckpt" "$scale" "$gate_mode"
      fi
    done
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || { echo "one eval job failed; inspect $LOG_ROOT" >&2; exit 1; }
done
echo "Goal154 authority eval complete: $ROOT"
