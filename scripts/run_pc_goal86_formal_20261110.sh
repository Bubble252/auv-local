#!/usr/bin/env bash
# Goal86 frozen 5M formal: 3 seeds per arm for PPO and SAC.
# Requires an explicit screen-Gate acknowledgement:
#   ALLOW_FORMAL=1 bash scripts/run_pc_goal86_formal_20261110.sh
set -uo pipefail

if [[ "${ALLOW_FORMAL:-0}" != "1" ]]; then
  echo "Refusing formal run before screen Gate. Set ALLOW_FORMAL=1 only after Gate PASS."
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/artifacts/lowlevel/goal86/formal}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal86_formal_20261110}"
D3="${D3:-checkpoints/goal26/d3/pc_bc_actor.pt}"

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

pids=(); names=()

launch_ppo() {
  local arm="$1" seed="$2" gpu="$3" extra=() config out log
  config="configs/lowlevel/goal86/pc_goal86_${arm}_ppo_formal_20261110.json"
  out="$OUT_ROOT/${arm}_ppo/seed_${seed}"
  log="$LOG_ROOT/${arm}_ppo_seed_${seed}.log"
  if [[ -f "$out/final.pt" ]]; then echo "SKIP ${arm}_ppo seed=$seed"; return; fi
  [[ "$arm" == "p-a" ]] && extra=(--warmstart-actor "$D3")
  echo "START ${arm}_ppo seed=$seed GPU=$gpu"
  (
    export PYTHONPATH="src:$SB3"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_batch_lowlevel_ppo.py \
      --training-config "$config" --output "$out" --seed "$seed" --device cuda "${extra[@]}"
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  pids+=("$!"); names+=("${arm}_ppo_seed_${seed}")
}

launch_sac() {
  local arm="$1" seed="$2" gpu="$3" config out log
  config="configs/lowlevel/goal86/pc_goal86_${arm}_sac_default_20261110.json"
  out="$OUT_ROOT/${arm}_sac/seed_${seed}"
  log="$LOG_ROOT/${arm}_sac_seed_${seed}.log"
  if [[ -f "$out/final.zip" ]]; then echo "SKIP ${arm}_sac seed=$seed"; return; fi
  echo "START ${arm}_sac seed=$seed GPU=$gpu"
  (
    export PYTHONPATH="src:$SB3"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_pc_command_residual_sac.py \
      --config "$config" --output "$out" --seed "$seed" \
      --total-env-transitions 5000000 --device cuda --engineering-smoke --overwrite
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  pids+=("$!"); names+=("${arm}_sac_seed_${seed}")
}

run_batch() {
  local fail=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then echo "FAIL ${names[$index]}"; fail=1; else echo "DONE ${names[$index]}"; fi
  done
  pids=(); names=()
  return "$fail"
}

# Phase 1: PPO formal is short; run it before the long SAC phase.
gpu=0
for spec in "t1 86111" "t1 86112" "t1 86113" "t2 86131" "t2 86132" "t2 86133" "p-a 86151" "p-a 86152" "p-a 86153"; do
  set -- $spec
  launch_ppo "$1" "$2" "$((gpu % 8))"
  gpu=$((gpu + 1))
done
run_batch || exit 1

# Phase 2: 9 SAC formal runs across 8 GPUs.
gpu=0
for spec in "t1 86001" "t1 86002" "t1 86003" "t2 86201" "t2 86202" "t2 86203" "pa 86101" "pa 86102" "pa 86103"; do
  set -- $spec
  launch_sac "$1" "$2" "$((gpu % 8))"
  gpu=$((gpu + 1))
done
run_batch || exit 1
echo "formal complete: $OUT_ROOT"
