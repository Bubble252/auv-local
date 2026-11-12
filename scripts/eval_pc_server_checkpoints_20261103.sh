#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
TRAINING_CONFIG="${TRAINING_CONFIG:?TRAINING_CONFIG is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
ENVIRONMENT_CONFIG="${ENVIRONMENT_CONFIG:-}"
CHECKPOINTS="${CHECKPOINTS:-0 10 20 30 31}"
SEED_START="${SEED_START:-0}"
SEED_COUNT="${SEED_COUNT:-64}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float64}"
TASKSET_CPUS="${TASKSET_CPUS:-}"

mkdir -p "$OUTPUT_ROOT"

for checkpoint_index in $CHECKPOINTS; do
  checkpoint="$RUN_DIR/model_${checkpoint_index}.pt"
  output="$OUTPUT_ROOT/ck${checkpoint_index}_seed${SEED_START}_$((SEED_START + SEED_COUNT - 1))"
  if [[ ! -f "$checkpoint" ]]; then
    printf 'MISSING %s\n' "$checkpoint"
    continue
  fi
  if [[ -f "$output/summary.json" ]]; then
    printf 'SKIP %s\n' "$output"
    continue
  fi
  command=(
    "$PY" scripts/evaluate_pc_ppo.py
    --training-config "$TRAINING_CONFIG"
    --checkpoint "$checkpoint"
    --output "$output"
    --seed-start "$SEED_START"
    --seed-count "$SEED_COUNT"
    --device "$DEVICE"
    --dtype "$DTYPE"
  )
  if [[ -n "$ENVIRONMENT_CONFIG" ]]; then
    command+=(--environment-config "$ENVIRONMENT_CONFIG")
  fi
  if [[ -n "$TASKSET_CPUS" ]]; then
    command=(taskset -c "$TASKSET_CPUS" nice -n 10 "${command[@]}")
  fi
  printf 'EVAL checkpoint=%s output=%s\n' "$checkpoint" "$output"
  "${command[@]}"
done

printf 'ALL_EVALS_DONE output_root=%s\n' "$OUTPUT_ROOT"
