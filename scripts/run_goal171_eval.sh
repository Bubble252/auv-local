#!/usr/bin/env bash
# Goal171 evaluation: always run the full 8 arms x 4 levels x 2 radii matrix.
set -euo pipefail

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$ROOT/train}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/eval}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs}
ENV50=configs/lowlevel/goal169/env_terminal_alignment_50_eval_nu_ground.json
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PARALLEL=${PARALLEL:-8}
ARMS=${ARMS:-"R0_BC_PPO R0_BC_SAC R0_ALOS_PPO R0_ALOS_SAC R1_BC_PPO R1_BC_SAC R1_ALOS_PPO R1_ALOS_SAC"}
SEEDS=${SEEDS:-"86201"}
LEVELS=${LEVELS:-"nominal current07 current10 current15"}
RADII=${RADII:-"default:2.0 strict125:1.25"}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
MIN_CURRICULUM_EPISODES=${MIN_CURRICULUM_EPISODES:-100}

cd "$REPO"
PYTHONPATH="src:$SB3" \
  "$PY" scripts/check_goal171_protocol.py --phase eval \
  --arms "$ARMS" --levels "$LEVELS" --radii "$RADII" \
  --seed-start "$SEED_START" --seed-count "$SEED_COUNT"
PYTHONPATH="src:$SB3" \
  "$PY" scripts/audit_goal171_artifacts.py \
  --phase train --root "$ROOT" --seeds "$SEEDS" \
  --min-episodes "$MIN_CURRICULUM_EPISODES"
mkdir -p "$EVAL_ROOT" "$LOG_ROOT" "$ROOT/manifests"

level_args() {
  case "$1" in
    nominal) ;;
    current07) echo "--curriculum-current-speed-range-mps 0.275 0.50 --curriculum-shear-scale 0.75" ;;
    current10) echo "--curriculum-current-speed-range-mps 0.30 0.55 --curriculum-shear-scale 1.0" ;;
    current15) echo "--curriculum-current-speed-range-mps 0.375 0.675 --curriculum-shear-scale 1.5" ;;
    *) echo "unsupported Goal171 phase1 level: $1" >&2; return 2 ;;
  esac
}

run_sac() {
  local arm=$1 radius_label=$2 radius_value=$3 level=$4 seed=$5 gpu=$6
  local config checkpoint output log
  config="configs/lowlevel/goal171/train_${arm,,}_seed${seed}.json"
  checkpoint="$TRAIN_ROOT/$arm/seed$seed/final.zip"
  output="$EVAL_ROOT/$arm/$radius_label/$level/policy_seed$seed"
  log="$LOG_ROOT/eval_${arm}_${radius_label}_${level}_seed${seed}.log"
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START SAC arm=$arm radius=$radius_label seed=$seed gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$config" \
      --environment-config "$ENV50" \
      --curriculum-stage C3 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --success-frame both --eval-goal-radius-m "$radius_value" \
      --device cuda --dtype float64 \
      --action-mode policy --checkpoint "$checkpoint" \
      --residual-scale 1.0 --learner-terminal-gate-mode none \
      --rich-trace --trace-output "$output" \
      --sb3-root "$SB3" --overwrite "${level_array[@]}" \
      >"$log" 2>&1
}

run_zero_sac() {
  local radius_label=$1 radius_value=$2 level=$3 gpu=$4
  local config output log
  config=configs/lowlevel/goal171/train_r0_alos_sac_seed86201.json
  output="$EVAL_ROOT/zero_sac50/$radius_label/$level/zero_seed86201"
  log="$LOG_ROOT/eval_zero_sac50_${radius_label}_${level}.log"
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START SAC-zero radius=$radius_label level=$level gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$config" \
      --environment-config "$ENV50" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --success-frame both --eval-goal-radius-m "$radius_value" \
      --device cuda --dtype float64 --action-mode zero-residual \
      --residual-scale 1.0 --learner-terminal-gate-mode none \
      --rich-trace --trace-output "$output" \
      --sb3-root "$SB3" --overwrite "${level_array[@]}" \
      >"$log" 2>&1
}

run_zero_ppo() {
  local kind=$1 radius_label=$2 radius_value=$3 level=$4 gpu=$5
  local config checkpoint output log base_env action_mode zero_arm
  base_env="configs/lowlevel/goal171/env_old_route_eval_${radius_label}_nu_ground.json"
  if [[ "$kind" == "zero_bc" ]]; then
    config=configs/lowlevel/goal171/train_r0_bc_ppo_seed86201.json
    checkpoint="$TRAIN_ROOT/R0_BC_PPO/seed86201/final.pt"
    action_mode=frozen-base
    zero_arm=zero_bc50
  else
    config=configs/lowlevel/goal171/train_r0_alos_ppo_seed86201.json
    checkpoint="$TRAIN_ROOT/R0_ALOS_PPO/seed86201/final.pt"
    action_mode=alos
    zero_arm=zero_alos50
  fi
  output="$EVAL_ROOT/$zero_arm/$radius_label/$level/zero_seed86201"
  log="$LOG_ROOT/eval_${zero_arm}_${radius_label}_${level}.log"
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START PPO-zero kind=$kind radius=$radius_label level=$level gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config "$config" \
      --checkpoint "$checkpoint" \
      --environment-config "$base_env" \
      --guidance-config "$ENV50" \
      --output "$output" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --device cuda --dtype float64 --action-mode "$action_mode" \
      --success-frame both --trace-output "$output" "${level_array[@]}" \
      >"$log" 2>&1
}

run_ppo() {
  local arm=$1 radius_label=$2 radius_value=$3 level=$4 seed=$5 gpu=$6
  local config checkpoint output log base_env
  config="configs/lowlevel/goal171/train_${arm,,}_seed${seed}.json"
  checkpoint="$TRAIN_ROOT/$arm/seed$seed/final.pt"
  output="$EVAL_ROOT/$arm/$radius_label/$level/policy_seed$seed"
  log="$LOG_ROOT/eval_${arm}_${radius_label}_${level}_seed$seed.log"
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  if [[ "$arm" == R1_* ]]; then
    base_env="configs/lowlevel/goal171/env_route_health_eval_${radius_label}_nu_ground.json"
  else
    base_env="configs/lowlevel/goal171/env_old_route_eval_${radius_label}_nu_ground.json"
  fi
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START PPO arm=$arm radius=$radius_label seed=$seed gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config "$config" \
      --checkpoint "$checkpoint" \
      --environment-config "$base_env" \
      --guidance-config "$ENV50" \
      --output "$output" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --device cuda --dtype float64 --action-mode policy \
      --success-frame both \
      --trace-output "$output" "${level_array[@]}" \
      >"$log" 2>&1
}

dispatch() {
  local kind=$1 radius_label=$2 radius_value=$3 level=$4 seed=$5 gpu=$6
  case "$kind" in
    zero_sac) run_zero_sac "$radius_label" "$radius_value" "$level" "$gpu" ;;
    zero_bc|zero_alos) run_zero_ppo "$kind" "$radius_label" "$radius_value" "$level" "$gpu" ;;
    *"_SAC") run_sac "$kind" "$radius_label" "$radius_value" "$level" "$seed" "$gpu" ;;
    *) run_ppo "$kind" "$radius_label" "$radius_value" "$level" "$seed" "$gpu" ;;
  esac
}
export -f dispatch run_sac run_ppo run_zero_sac run_zero_ppo level_args
export REPO PY SB3 ROOT TRAIN_ROOT EVAL_ROOT LOG_ROOT ENV50
export ARMS SEEDS LEVELS RADII SEED_START SEED_COUNT MIN_CURRICULUM_EPISODES

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
read -r -a gpu_array <<<"$GPUS"
index=0
for radius in $RADII; do
  for level in $LEVELS; do
    for zero_kind in zero_sac zero_bc zero_alos; do
      gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
      echo "$zero_kind ${radius%%:*} ${radius##*:} $level zero $gpu" >>"$JOB_FILE"
      index=$((index + 1))
    done
    for arm in $ARMS; do
      for seed in $SEEDS; do
        gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
        echo "$arm ${radius%%:*} ${radius##*:} $level $seed $gpu" >>"$JOB_FILE"
        index=$((index + 1))
      done
    done
  done
done

{
  printf 'started_at=%s\nhost=%s\nrepo=%s\narms=%s\nseeds=%s\nlevels=%s\nradii=%s\n' \
    "$(date -Iseconds)" "$(hostname)" "$REPO" "$ARMS" "$SEEDS" "$LEVELS" "$RADII"
  sha256sum scripts/run_goal171_eval.sh scripts/analyze_goal170_route_transfer.py \
    "$ENV50" configs/lowlevel/goal171/env_*_eval_*_nu_ground.json 2>/dev/null || true
} >"$ROOT/manifests/eval_manifest.txt"

echo "launching $index Goal171 evaluations"
xargs -r -n 6 -P "$PARALLEL" bash -c 'dispatch "$@"' _ <"$JOB_FILE"

missing=0
for radius in $RADII; do
  for level in $LEVELS; do
    for zero_kind in zero_sac zero_bc zero_alos; do
      zero_arm=${zero_kind/zero_sac/zero_sac50}
      zero_arm=${zero_arm/zero_bc/zero_bc50}
      zero_arm=${zero_arm/zero_alos/zero_alos50}
      if [[ ! -f "$EVAL_ROOT/$zero_arm/${radius%%:*}/$level/zero_seed86201/summary.json" ]]; then
        echo "MISSING $zero_arm ${radius%%:*} $level" >&2
        missing=$((missing + 1))
      fi
    done
    for arm in $ARMS; do
      for seed in $SEEDS; do
        if [[ ! -f "$EVAL_ROOT/$arm/${radius%%:*}/$level/policy_seed$seed/summary.json" ]]; then
          echo "MISSING $arm ${radius%%:*} $level seed$seed" >&2
          missing=$((missing + 1))
        fi
      done
    done
  done
done
printf 'finished_at=%s\nmissing=%s\n' "$(date -Iseconds)" "$missing" >>"$ROOT/manifests/eval_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal171 evaluations missing; inspect $LOG_ROOT" >&2
  exit 1
fi
PYTHONPATH="src:$SB3" \
  "$PY" scripts/audit_goal171_artifacts.py \
  --phase eval --root "$ROOT" --seeds "$SEEDS" \
  --seed-start "$SEED_START" --seed-count "$SEED_COUNT"
echo "Goal171 evaluation complete: $EVAL_ROOT"
