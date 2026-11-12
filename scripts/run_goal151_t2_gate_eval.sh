#!/usr/bin/env bash
# Goal151 eval-only: reuse Goal86 T2-SAC checkpoints under two terminal gates.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal151_t2_terminal_gate_eval_20261111}
GPUS=${GPUS:-"0 3 4 6"}
MAX_PARALLEL=${MAX_PARALLEL:-6}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
T2_ROOT=${T2_ROOT:-artifacts/lowlevel/goal86/screen}
T2_DEFAULT_ENV=${T2_DEFAULT_ENV:-configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json}
T2_STRICT_ENV=${T2_STRICT_ENV:-configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json}

cd "$REPO_ROOT"
mkdir -p "$ROOT/logs"

read -r -a GPU_ARRAY <<<"$GPUS"
pids=()
names=()
job_index=0

wait_one() {
  local status=0
  wait -n || status=$?
  if [[ $status -ne 0 ]]; then
    echo "one eval job failed; inspect $ROOT/logs" >&2
    exit 1
  fi
}

launch() {
  local tag="$1" checkpoint="$2" config="$3" env_config="$4" scenario="$5" extra="$6"
  local out="$ROOT/$tag/$scenario"
  local gpu="${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
  mkdir -p "$out"
  mkdir -p "$ROOT/logs/$(dirname "$tag")"
  if [[ -f "$out/summary.json" && -f "$out/episodes.json" ]]; then
    echo "SKIP $tag/$scenario"
    job_index=$((job_index + 1))
    return
  fi
  echo "START $tag/$scenario gpu=$gpu"
  local args=(
    "$PY" scripts/evaluate_pc_sac.py
    --config "$config"
    --environment-config "$env_config"
    --checkpoint "$checkpoint"
    --output "$out"
    --trace-output "$out"
    --trace-reward-breakdown
    --rich-trace
    --seed-start "$SEED_START"
    --seed-count "$SEED_COUNT"
    --success-frame both
    --device cuda
    --dtype float64
    --action-mode policy
    --residual-scale 1.0
    --sb3-root "$SB3"
    --overwrite
  )
  if [[ -n "$extra" ]]; then
    # shellcheck disable=SC2206
    args+=($extra)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="src:$SB3" \
    "${args[@]}" >"$ROOT/logs/${tag}_${scenario}.log" 2>&1 &
  pids+=("$!")
  names+=("$tag/$scenario")
  job_index=$((job_index + 1))
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait_one
    pids=("${pids[@]:1}")
  fi
}

for checkpoint in \
  "$T2_ROOT/t2_sac/final.zip" \
  "$T2_ROOT/t2_sac_seed86202/final.zip" \
  "$T2_ROOT/t2_sac_seed86203/final.zip"; do
  seed_dir=$(basename "$(dirname "$checkpoint")")
  for scenario in default strict125; do
    if [[ "$scenario" == strict125 ]]; then
      env_config="$T2_STRICT_ENV"
    else
      env_config="$T2_DEFAULT_ENV"
    fi
    launch "near10_2/$seed_dir" "$checkpoint" configs/lowlevel/goal151/near10_2.json "$env_config" "$scenario" ""
    launch "shell10_3/$seed_dir" "$checkpoint" configs/lowlevel/goal151/shell10_3.json "$env_config" "$scenario" "--learner-terminal-gate-mode shell"
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || { echo "one eval job failed; inspect $ROOT/logs" >&2; exit 1; }
done

for tag in "${names[@]}"; do
  if [[ ! -f "$ROOT/${tag}/summary.json" || ! -f "$ROOT/${tag}/episodes.json" ]]; then
    echo "missing output for $tag" >&2
    exit 1
  fi
done
echo "Goal151 T2 gate evaluation complete: $ROOT"
