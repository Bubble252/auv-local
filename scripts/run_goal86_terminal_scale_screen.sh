#!/usr/bin/env bash
# Goal156 addendum: eval-only near-terminal residual scale screen.
#
# Reuses full-authority T2-SAC checkpoints and the Goal154 near10_2 deployment
# gate.  Defaults to the three Goal86 1M checkpoints; set CHECKPOINT_LAYOUT=goal156
# and CHECKPOINT_ROOT=<goal156>/train to run the same diagnostic on 5M checkpoints.
# No training, no held-out, no algorithm change.  The only axis is the global
# residual scale.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
GOAL86_ROOT="${GOAL86_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/goal86/screen}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$GOAL86_ROOT}"
CHECKPOINT_LAYOUT="${CHECKPOINT_LAYOUT:-goal86}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal86_terminal_scale_screen_20261112}"
OUT="$ROOT/eval_cpu"
LOG_ROOT="$ROOT/logs"
SEEDS="${SEEDS:-86201 86202 86203}"
SCALES="${SCALES:-0.5 0.75 1.0 1.5}"
EVAL_SEED_START="${EVAL_SEED_START:-20}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-40}"
MAX_PARALLEL="${MAX_PARALLEL:-24}"
DEFAULT_ENV="${DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}"
STRICT_ENV="${STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}"
GATE_CONFIG="${GATE_CONFIG:-configs/lowlevel/goal154/authority/near10_2.json}"

cd "$REPO_ROOT"
mkdir -p "$OUT" "$LOG_ROOT"
read -r -a SEED_ARRAY <<<"$SEEDS"
read -r -a SCALE_ARRAY <<<"$SCALES"

checkpoint_for_seed() {
  local seed="$1"
  case "$CHECKPOINT_LAYOUT" in
    goal86)
      if [[ "$seed" == "86201" ]]; then
        echo "$CHECKPOINT_ROOT/t2_sac/final.zip"
      else
        echo "$CHECKPOINT_ROOT/t2_sac_seed${seed}/final.zip"
      fi
      ;;
    goal156)
      echo "$CHECKPOINT_ROOT/full_seed${seed}/final.zip"
      ;;
    *)
      echo "unknown CHECKPOINT_LAYOUT: $CHECKPOINT_LAYOUT" >&2
      return 1
      ;;
  esac
}

for seed in "${SEED_ARRAY[@]}"; do
  checkpoint="$(checkpoint_for_seed "$seed")"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing Goal86 checkpoint: $checkpoint" >&2
    exit 2
  fi
done

pids=()
queued_outputs=()

queue_eval() {
  local out="$1" log="$2"
  shift 2
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP $out"
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
  out="$OUT/zero/$scenario"
  queue_eval "$out" "$LOG_ROOT/zero_${scenario}.log" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$GATE_CONFIG" \
      --environment-config "$env_config" \
      --output "$out" --trace-output "$out" \
      --trace-reward-breakdown --rich-trace \
      --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
      --success-frame both --device cpu --dtype float64 \
      --action-mode zero-residual --residual-scale 1.0 \
      --learner-terminal-gate-mode none \
      --sb3-root "$SB3" --overwrite
done

for scale in "${SCALE_ARRAY[@]}"; do
  tag="near10_2_s$(printf '%03d' "$(awk -v x="$scale" 'BEGIN { printf "%d", x*100 + 0.5 }')")"
  for seed in "${SEED_ARRAY[@]}"; do
    checkpoint="$(checkpoint_for_seed "$seed")"
    for scenario in default strict125; do
      if [[ "$scenario" == strict125 ]]; then env_config="$STRICT_ENV"; else env_config="$DEFAULT_ENV"; fi
      out="$OUT/$tag/seed$seed/$scenario"
      queue_eval "$out" "$LOG_ROOT/${tag}_seed${seed}_${scenario}.log" \
        "$PY" scripts/evaluate_pc_sac.py \
          --config "$GATE_CONFIG" \
          --environment-config "$env_config" \
          --output "$out" --trace-output "$out" \
          --trace-reward-breakdown --rich-trace \
          --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
          --success-frame both --device cpu --dtype float64 \
          --action-mode policy --residual-scale "$scale" \
          --checkpoint "$checkpoint" \
          --learner-terminal-gate-mode none \
          --sb3-root "$SB3" --overwrite
    done
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

failed=0
for out in "${queued_outputs[@]}"; do
  code="$(cat "$out/exitcode" 2>/dev/null || echo missing)"
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

"$PY" scripts/analyze_pc_goal153_eval_grid.py \
  --root "$OUT" --zero-tag zero \
  --output-json "$OUT/grid_summary.json" \
  --output-md "$OUT/grid_summary.md"
echo "terminal scale screen complete: $OUT"
