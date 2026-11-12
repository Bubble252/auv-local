#!/usr/bin/env bash
# Goal156 eval-only recovery runner.
#
# The preregistered 5M training runner is left untouched while it executes.
# This script only evaluates already-finished checkpoints and fixes the
# concurrent wait bookkeeping used by the original eval queue:
#   wait -n reaps an arbitrary child, so removing the first PID from the
#   bookkeeping array can leave a reaped PID behind and later `wait` on it
#   returns 127.  Here we rebuild the live PID set with `jobs -pr` after each
#   `wait -n`, and wrap every eval in a subshell that records its exit code.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal156_full_train_eval_gate}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal156_full_train_eval_gate}"
SEEDS="${SEEDS:-86201 86202 86203}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
MAX_PARALLEL="${MAX_PARALLEL:-24}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"
ANALYZE="${ANALYZE:-1}"

cd "$REPO_ROOT"
mkdir -p "$ROOT/eval" "$LOG_ROOT"
read -r -a SEED_ARRAY <<<"$SEEDS"

for seed in "${SEED_ARRAY[@]}"; do
  checkpoint="$ROOT/train/full_seed${seed}/final.zip"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 2
  fi
done

pids=()
queued_outputs=()

queue_eval() {
  local out="$1" log="$2"
  shift 2
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP $(basename "$out")"
    return
  fi
  mkdir -p "$out"
  rm -f "$out/exitcode"
  (
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH="src:$SB3:$GLOBAL_SRC" \
      "$@" >"$log" 2>&1
    echo $? >"$out/exitcode"
  ) &
  pids+=("$!")
  queued_outputs+=("$out")
  if (( ${#pids[@]} >= MAX_PARALLEL )); then
    if ! wait -n; then
      echo "eval queue wait failed" >&2
      exit 1
    fi
    mapfile -t pids < <(jobs -pr)
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
  wait "$pid" || true
done

failed=0
for out in "${queued_outputs[@]}"; do
  code_file="$out/exitcode"
  code="$(cat "$code_file" 2>/dev/null || echo missing)"
  if [[ "$code" != "0" ]]; then
    echo "FAILED exitcode=$code $out" >&2
    failed=1
  fi
  if [[ ! -f "$out/summary.json" || ! -f "$out/episodes.json" ]]; then
    echo "FAILED missing summary/episodes: $out" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  exit 1
fi

if [[ "$ANALYZE" == "1" ]]; then
  "$PY" scripts/analyze_pc_goal155_authority_training.py \
    --root "$ROOT" --eval-root "$ROOT/eval"
  "$PY" scripts/analyze_pc_goal153_eval_grid.py \
    --root "$ROOT/eval" --zero-tag zero \
    --output-json "$ROOT/eval/grid_summary.json" \
    --output-md "$ROOT/eval/grid_summary.md"
fi
echo "Goal156 eval-only complete: $ROOT"
