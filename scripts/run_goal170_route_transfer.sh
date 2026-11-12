#!/usr/bin/env bash
# Goal170 eval-only route-health transfer replay.  No training.
set -euo pipefail

python_bin() {
  echo "${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
}

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal170_route_health_transfer_20261112}
GOAL167_ROOT=${GOAL167_ROOT:-$REPO/artifacts/lowlevel/goal167_action_parameterization_20261112}
GOAL86_ROOT=${GOAL86_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/goal86/screen}
EVAL_ROOT=$ROOT/eval
LOG_ROOT=$ROOT/logs
G169_EVAL=${G169_EVAL:-$REPO/artifacts/lowlevel/goal169_terminal_homing_timing_20261112/eval/alignment50}
ENV50=configs/lowlevel/goal169/env_terminal_alignment_50_eval_nu_ground.json
PPO_BASE_ENV=configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json
PPO_STRICT_ENV=configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PARALLEL=${PARALLEL:-8}
MODE=${1:-smoke}
ARMS=${ARMS:-"A_actuator B_command T2_SAC T2_PPO"}

if [[ "$MODE" == "smoke" ]]; then
  LEVELS=${LEVELS:-"current10"}
  RADII=${RADII:-"default:2.0 strict125:1.25"}
  CKPT_SEEDS=${CKPT_SEEDS:-"86201"}
elif [[ "$MODE" == "full" ]]; then
  LEVELS=${LEVELS:-"nominal current07 current10 current15"}
  RADII=${RADII:-"default:2.0 strict125:1.25"}
  CKPT_SEEDS=${CKPT_SEEDS:-"86201 86202 86203"}
else
  echo "usage: $0 [smoke|full]" >&2
  exit 2
fi

cd "$REPO"
PY=$(python_bin)
mkdir -p "$EVAL_ROOT" "$LOG_ROOT" "$ROOT/manifests"

if [[ "$MODE" == "full" && "$RADII" != *:* ]]; then
  RADII="default:2.0 strict125:1.25"
fi

link_zero50() {
  for radius in $RADII; do
    local radius_label=${radius%%:*}
    for level in $LEVELS; do
      local src=$G169_EVAL/$radius_label/$level/zero_seed86201
      local dst=$EVAL_ROOT/zero50/$radius_label/$level/zero_seed86201
      if [[ ! -d "$src" ]]; then
        echo "missing Goal169 zero run: $src" >&2
        return 1
      fi
      if [[ ! -e "$dst" ]]; then
        mkdir -p "$(dirname "$dst")"
        ln -s "$src" "$dst"
      fi
    done
  done
}

t2_sac_checkpoint() {
  local seed=$1
  if [[ "$seed" == "86201" ]]; then
    echo "$GOAL86_ROOT/t2_sac/final.zip"
  else
    echo "$GOAL86_ROOT/t2_sac_seed${seed}/final.zip"
  fi
}

t2_ppo_checkpoint() {
  local seed=$1
  if [[ "$seed" == "86201" ]]; then
    echo "$GOAL86_ROOT/t2_ppo/final.pt"
  else
    echo "$GOAL86_ROOT/t2_ppo_seed${seed}/final.pt"
  fi
}

level_args() {
  local level=$1
  case "$level" in
    nominal) ;;
    current07) echo "--curriculum-current-speed-range-mps 0.275 0.50 --curriculum-shear-scale 0.75" ;;
    current10) echo "--curriculum-current-speed-range-mps 0.30 0.55 --curriculum-shear-scale 1.0" ;;
    current15) echo "--curriculum-current-speed-range-mps 0.375 0.675 --curriculum-shear-scale 1.5" ;;
    *) echo "unknown level: $level" >&2; return 2 ;;
  esac
}

run_sac() {
  local arm=$1 radius_label=$2 radius_value=$3 level=$4 ckpt_seed=$5 index=$6
  local config checkpoint output log gpu
  local -a gpu_array level_array
  read -r -a gpu_array <<<"$GPUS"
  gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
  read -r -a level_array <<<"$(level_args "$level")"
  case "$arm" in
    A_actuator) config=configs/lowlevel/goal167/eval_a_actuator_terminal.json; checkpoint=$GOAL167_ROOT/train/A_actuator_terminal/seed${ckpt_seed}/final.zip ;;
    B_command) config=configs/lowlevel/goal167/eval_b_command_terminal.json; checkpoint=$GOAL167_ROOT/train/B_command_terminal/seed${ckpt_seed}/final.zip ;;
    T2_SAC) config=configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json; checkpoint=$(t2_sac_checkpoint "$ckpt_seed") ;;
    *) echo "unknown SAC arm: $arm" >&2; return 2 ;;
  esac
  output=$EVAL_ROOT/$arm/$radius_label/$level/policy_seed$ckpt_seed
  log=$LOG_ROOT/${MODE}_${arm}_${radius_label}_${level}_policy_seed${ckpt_seed}.log
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2; return 1
  fi
  mkdir -p "$output"
  echo "START SAC arm=$arm radius=$radius_label level=$level seed=$ckpt_seed gpu=$gpu"
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
      --device "$DEVICE" --dtype float64 \
      --action-mode policy --checkpoint "$checkpoint" \
      --residual-scale 1.0 --learner-terminal-gate-mode none \
      --rich-trace --trace-output "$output" \
      --sb3-root "$SB3" --overwrite "${level_array[@]}" \
      >"$log" 2>&1
}

run_ppo_zero() {
  local radius_label=$1 radius_value=$2 level=$3 index=$4
  local output log gpu
  local -a gpu_array level_array
  local base_env checkpoint
  read -r -a gpu_array <<<"$GPUS"
  gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
  read -r -a level_array <<<"$(level_args "$level")"
  if [[ "$radius_label" == "strict125" ]]; then base_env=$PPO_STRICT_ENV; else base_env=$PPO_BASE_ENV; fi
  checkpoint=$(t2_ppo_checkpoint 86201)
  output=$EVAL_ROOT/zero_ppo50/$radius_label/$level/zero_seed86031
  log=$LOG_ROOT/${MODE}_zero_ppo50_${radius_label}_${level}.log
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2; return 1
  fi
  mkdir -p "$output"
  echo "START PPO-zero radius=$radius_label level=$level gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config configs/lowlevel/goal86/pc_goal86_t2_ppo_screen_20261110.json \
      --checkpoint "$checkpoint" \
      --environment-config "$base_env" \
      --guidance-config "$ENV50" \
      --output "$output" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --device "$DEVICE" --dtype float64 --action-mode alos \
      --success-frame both \
      --trace-output "$output" "${level_array[@]}" \
      >"$log" 2>&1
}

run_ppo() {
  local radius_label=$1 radius_value=$2 level=$3 ckpt_seed=$4 index=$5
  local checkpoint output log gpu
  local -a gpu_array level_array
  local base_env
  read -r -a gpu_array <<<"$GPUS"
  gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
  read -r -a level_array <<<"$(level_args "$level")"
  if [[ "$radius_label" == "strict125" ]]; then base_env=$PPO_STRICT_ENV; else base_env=$PPO_BASE_ENV; fi
  checkpoint=$(t2_ppo_checkpoint "$ckpt_seed")
  output=$EVAL_ROOT/T2_PPO/$radius_label/$level/policy_seed$ckpt_seed
  log=$LOG_ROOT/${MODE}_T2_PPO_${radius_label}_${level}_policy_seed${ckpt_seed}.log
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2; return 1
  fi
  mkdir -p "$output"
  echo "START PPO radius=$radius_label level=$level seed=$ckpt_seed gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config configs/lowlevel/goal86/pc_goal86_t2_ppo_screen_20261110.json \
      --checkpoint "$checkpoint" \
      --environment-config "$base_env" \
      --guidance-config "$ENV50" \
      --output "$output" \
      --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --device "$DEVICE" --dtype float64 --action-mode policy \
      --success-frame both \
      --trace-output "$output" "${level_array[@]}" \
      >"$log" 2>&1
}

link_zero50

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_count=0
for arm in $ARMS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      if [[ "$arm" == "T2_PPO" ]]; then
        echo "ppo_zero ${radius%%:*} ${radius##*:} $level 86031 $job_count" >>"$JOB_FILE"
        job_count=$((job_count + 1))
      fi
      for ckpt_seed in $CKPT_SEEDS; do
        echo "$arm ${radius%%:*} ${radius##*:} $level $ckpt_seed $job_count" >>"$JOB_FILE"
        job_count=$((job_count + 1))
      done
    done
  done
done

dispatch() {
  local kind=$1 radius_label=$2 radius_value=$3 level=$4 ckpt_seed=$5 index=$6
  if [[ "$kind" == "ppo_zero" ]]; then
    run_ppo_zero "$radius_label" "$radius_value" "$level" "$index"
  elif [[ "$kind" == "T2_PPO" ]]; then
    run_ppo "$radius_label" "$radius_value" "$level" "$ckpt_seed" "$index"
  else
    run_sac "$kind" "$radius_label" "$radius_value" "$level" "$ckpt_seed" "$index"
  fi
}
export -f dispatch run_sac run_ppo run_ppo_zero level_args t2_sac_checkpoint t2_ppo_checkpoint
export REPO PY SB3 ROOT GOAL167_ROOT GOAL86_ROOT EVAL_ROOT LOG_ROOT G169_EVAL ENV50 PPO_BASE_ENV PPO_STRICT_ENV
export SEED_START SEED_COUNT DEVICE GPUS MODE

{
  printf 'started_at=%s\nmode=%s\nhost=%s\nrepo=%s\narms=%s\nlevels=%s\nradii=%s\nckpt_seeds=%s\n' \
    "$(date -Iseconds)" "$MODE" "$(hostname)" "$REPO" "$ARMS" "$LEVELS" "$RADII" "$CKPT_SEEDS"
  sha256sum scripts/run_goal170_route_transfer.sh scripts/analyze_goal170_route_transfer.py \
    scripts/evaluate_pc_sac.py scripts/evaluate_pc_ppo.py "$ENV50" "$PPO_BASE_ENV" "$PPO_STRICT_ENV" 2>/dev/null || true
} >"$ROOT/manifests/${MODE}_run_manifest.txt"

echo "launching $job_count Goal170 evaluations with PARALLEL=$PARALLEL"
xargs -r -n 6 -P "$PARALLEL" bash -c 'dispatch "$@"' _ <"$JOB_FILE"

failed=0
for arm in $ARMS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      if [[ "$arm" == "T2_PPO" ]]; then
        if [[ ! -f "$EVAL_ROOT/zero_ppo50/${radius%%:*}/$level/zero_seed86031/summary.json" ]]; then
          echo "MISSING zero_ppo50 ${radius%%:*} $level" >&2; failed=$((failed + 1))
        fi
      fi
      for ckpt_seed in $CKPT_SEEDS; do
        if [[ ! -f "$EVAL_ROOT/$arm/${radius%%:*}/$level/policy_seed$ckpt_seed/summary.json" ]]; then
          echo "MISSING $arm ${radius%%:*} $level policy seed$ckpt_seed" >&2
          failed=$((failed + 1))
        fi
      done
    done
  done
done
printf 'finished_at=%s\nmissing=%s\n' "$(date -Iseconds)" "$failed" >>"$ROOT/manifests/${MODE}_run_manifest.txt"
if [[ "$failed" -ne 0 ]]; then
  echo "$failed Goal170 evaluations missing" >&2
  exit 1
fi
echo "Goal170 $MODE replay complete: $EVAL_ROOT"
