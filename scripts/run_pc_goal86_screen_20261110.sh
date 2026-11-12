#!/usr/bin/env bash
# Goal86 frozen 1M screen: T1/T2/P-A x PPO/SAC, one seed per arm.
# Usage:
#   bash scripts/run_pc_goal86_screen_20261110.sh
#   ARMS="t1 t2" bash scripts/run_pc_goal86_screen_20261110.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/artifacts/lowlevel/goal86/screen}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/goal86_screen_20261110}"
D3="${D3:-checkpoints/goal26/d3/pc_bc_actor.pt}"
ARMS="${ARMS:-t1 t2}"

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

pids=()
names=()

arm_selected() {
  [[ " ${ARMS//,/ } " == *" $1 "* ]]
}

launch_ppo() {
  local arm="$1" gpu="$2" extra=()
  local config="configs/lowlevel/goal86/pc_goal86_${arm}_ppo_screen_20261110.json"
  local out="$OUT_ROOT/${arm}_ppo"
  local log="$LOG_ROOT/${arm}_ppo.log"
  if [[ -f "$out/final.pt" ]]; then echo "SKIP ${arm}_ppo"; return; fi
  [[ "$arm" == "p-a" ]] && extra=(--warmstart-actor "$D3")
  echo "START ${arm}_ppo GPU=$gpu"
  (
    export PYTHONPATH="src:$SB3"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_batch_lowlevel_ppo.py \
      --training-config "$config" --output "$out" --device cuda "${extra[@]}"
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  pids+=("$!"); names+=("${arm}_ppo")
}

launch_sac() {
  local arm="$1" gpu="$2"
  local config="configs/lowlevel/goal86/pc_goal86_${arm}_sac_default_20261110.json"
  local out="$OUT_ROOT/${arm}_sac"
  local log="$LOG_ROOT/${arm}_sac.log"
  if [[ -f "$out/final.zip" ]]; then echo "SKIP ${arm}_sac"; return; fi
  echo "START ${arm}_sac GPU=$gpu"
  (
    export PYTHONPATH="src:$SB3"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_pc_command_residual_sac.py \
      --config "$config" --output "$out" --total-env-transitions 1000000 \
      --device cuda --engineering-smoke --overwrite
    echo $? > "$out/exitcode"
  ) >"$log" 2>&1 &
  pids+=("$!"); names+=("${arm}_sac")
}

arm_selected t1 && launch_ppo t1 0
arm_selected t2 && launch_ppo t2 1
arm_selected p-a && launch_ppo p-a 2
arm_selected t1 && launch_sac t1 3
arm_selected t2 && launch_sac t2 4
arm_selected p-a && launch_sac pa 5

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "no arms selected (ARMS='$ARMS')" >&2
  exit 2
fi

fail=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then echo "FAIL ${names[$index]}"; fail=1; else echo "DONE ${names[$index]}"; fi
done
if [[ "$fail" -ne 0 ]]; then echo "screen finished with failures; inspect $LOG_ROOT"; exit 1; fi
echo "screen complete: $OUT_ROOT"
