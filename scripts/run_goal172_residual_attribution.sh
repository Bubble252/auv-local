#!/usr/bin/env bash
# Goal172 eval-only residual attribution. The 4 levels x 2 radii matrix is
# frozen in scripts/goal172_matrix_contract.py and cannot be partial.
set -euo pipefail

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112}
TRAIN_ROOT=${TRAIN_ROOT:-$FORMAL_ROOT/train}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal172_residual_attribution_full_20261112}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs}
ENV50=configs/lowlevel/goal169/env_terminal_alignment_50_eval_nu_ground.json
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
MIN_CURRICULUM_EPISODES=${MIN_CURRICULUM_EPISODES:-100}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}

cd "$REPO"
export PYTHONPATH="src:$SB3"

radius_value() {
  case "$1" in
    default) echo 2.0 ;;
    strict125) echo 1.25 ;;
    *) echo "unsupported radius: $1" >&2; return 2 ;;
  esac
}

route_env() {
  case "$1" in
    default) echo configs/lowlevel/goal171/env_route_health_eval_default_nu_ground.json ;;
    strict125) echo configs/lowlevel/goal171/env_route_health_eval_strict125_nu_ground.json ;;
    *) echo "unsupported radius: $1" >&2; return 2 ;;
  esac
}

old_env() {
  case "$1" in
    default) echo configs/lowlevel/goal171/env_old_route_eval_default_nu_ground.json ;;
    strict125) echo configs/lowlevel/goal171/env_old_route_eval_strict125_nu_ground.json ;;
    *) echo "unsupported radius: $1" >&2; return 2 ;;
  esac
}

level_args() {
  case "$1" in
    nominal) ;;
    current07) echo "--curriculum-current-speed-range-mps 0.275 0.50 --curriculum-shear-scale 0.75" ;;
    current10) echo "--curriculum-current-speed-range-mps 0.30 0.55 --curriculum-shear-scale 1.0" ;;
    current15) echo "--curriculum-current-speed-range-mps 0.375 0.675 --curriculum-shear-scale 1.5" ;;
    *) echo "unsupported Goal172 level: $1" >&2; return 2 ;;
  esac
}

run_ppo_scale() {
  local arm=$1 radius_label=$2 tag=$3 scale=$4 level=$5 gpu=$6
  local output log config checkpoint env level_array
  output="$ROOT/screen/$arm/$radius_label/$tag/$level/policy_seed86201"
  log="$LOG_ROOT/ppo_${arm}_${radius_label}_${tag}_${level}.log"
  config="configs/lowlevel/goal171/train_${arm,,}_seed86201.json"
  checkpoint="$TRAIN_ROOT/$arm/seed86201/final.pt"
  env=$(route_env "$radius_label")
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2; return 1
  fi
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START PPO scale arm=$arm radius=$radius_label level=$level scale=$scale gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config "$config" --checkpoint "$checkpoint" \
      --environment-config "$env" --guidance-config "$ENV50" \
      --output "$output" --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --device cuda --dtype float64 --action-mode policy --success-frame both \
      --trace-output "$output" --trace-reward-breakdown \
      --actuator-residual-scale-override "$scale" "${level_array[@]}" \
      >"$log" 2>&1
}

run_sac_scale() {
  local arm=$1 radius_label=$2 tag=$3 multiplier=$4 level=$5 gpu=$6
  local output log config checkpoint radius level_array
  output="$ROOT/screen/$arm/$radius_label/$tag/$level/policy_seed86201"
  log="$LOG_ROOT/sac_${arm}_${radius_label}_${tag}_${level}.log"
  config="configs/lowlevel/goal171/train_${arm,,}_seed86201.json"
  checkpoint="$TRAIN_ROOT/$arm/seed86201/final.zip"
  radius=$(radius_value "$radius_label")
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2; return 1
  fi
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START SAC scale arm=$arm radius=$radius_label level=$level multiplier=$multiplier gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$config" --environment-config "$ENV50" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --output "$output" --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --success-frame both --eval-goal-radius-m "$radius" \
      --device cuda --dtype float64 --action-mode policy --checkpoint "$checkpoint" \
      --residual-scale "$multiplier" --learner-terminal-gate-mode none \
      --rich-trace --trace-output "$output" --sb3-root "$SB3" --overwrite \
      "${level_array[@]}" >"$log" 2>&1
}

run_ppo_channel() {
  local arm=$1 radius_label=$2 tag=$3 gains=$4 level=$5 gpu=$6
  local output log config checkpoint env level_array
  output="$ROOT/screen_channel/$arm/$radius_label/$tag/$level/policy_seed86201"
  log="$LOG_ROOT/channel_${arm}_${radius_label}_${tag}_${level}.log"
  config="configs/lowlevel/goal171/train_${arm,,}_seed86201.json"
  checkpoint="$TRAIN_ROOT/$arm/seed86201/final.pt"
  env=$(route_env "$radius_label")
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2; return 1
  fi
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START PPO channel arm=$arm radius=$radius_label level=$level channel=$tag gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config "$config" --checkpoint "$checkpoint" \
      --environment-config "$env" --guidance-config "$ENV50" \
      --output "$output" --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --device cuda --dtype float64 --action-mode policy --success-frame both \
      --trace-output "$output" --trace-reward-breakdown \
      --actuator-residual-scale-override 0.50 \
      --actuator-residual-channel-gains "$gains" "${level_array[@]}" \
      >"$log" 2>&1
}

run_ppo_rudder_scale() {
  local arm=$1 radius_label=$2 tag=$3 scale=$4 level=$5 gpu=$6
  local output log config checkpoint env level_array
  output="$ROOT/screen_rudder/$arm/$radius_label/$tag/$level/policy_seed86201"
  log="$LOG_ROOT/rudder_${arm}_${radius_label}_${tag}_${level}.log"
  config="configs/lowlevel/goal171/train_${arm,,}_seed86201.json"
  checkpoint="$TRAIN_ROOT/$arm/seed86201/final.pt"
  env=$(route_env "$radius_label")
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2; return 1
  fi
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START PPO rudder arm=$arm radius=$radius_label level=$level scale=$scale gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config "$config" --checkpoint "$checkpoint" \
      --environment-config "$env" --guidance-config "$ENV50" \
      --output "$output" --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --device cuda --dtype float64 --action-mode policy --success-frame both \
      --trace-output "$output" --trace-reward-breakdown \
      --actuator-residual-scale-override "$scale" \
      --actuator-residual-channel-gains "1.0,0.000001,0.000001" \
      "${level_array[@]}" >"$log" 2>&1
}

run_reward_zero() {
  local env_kind=$1 radius_label=$2 level=$3 gpu=$4
  local output log config checkpoint env level_array
  output="$ROOT/reward_attribution/$env_kind/$radius_label/$level/zero_seed86201"
  log="$LOG_ROOT/reward_${env_kind}_${radius_label}_${level}.log"
  if [[ "$env_kind" == "route_health" ]]; then
    config="configs/lowlevel/goal171/train_r1_bc_ppo_seed86201.json"
    checkpoint="$TRAIN_ROOT/R1_BC_PPO/seed86201/final.pt"
    env=$(route_env "$radius_label")
  else
    config="configs/lowlevel/goal171/train_r0_bc_ppo_seed86201.json"
    checkpoint="$TRAIN_ROOT/R0_BC_PPO/seed86201/final.pt"
    env=$(old_env "$radius_label")
  fi
  if [[ -f "$output/summary.json" ]]; then echo "SKIP $output"; return 0; fi
  if [[ -d "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty incomplete output: $output" >&2; return 1
  fi
  read -r -a level_array <<<"$(level_args "$level")"
  echo "START reward-zero env=$env_kind radius=$radius_label level=$level gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_ppo.py \
      --training-config "$config" --checkpoint "$checkpoint" \
      --environment-config "$env" --guidance-config "$ENV50" \
      --output "$output" --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
      --curriculum-stage C3 --disable-curriculum-direct-command \
      --device cuda --dtype float64 --action-mode frozen-base --success-frame both \
      --trace-output "$output" --trace-reward-breakdown "${level_array[@]}" \
      >"$log" 2>&1
}

dispatch() {
  local kind=$1 arm=$2 radius=$3 tag=$4 value=$5 level=$6 gpu=$7
  case "$kind" in
    ppo) run_ppo_scale "$arm" "$radius" "$tag" "$value" "$level" "$gpu" ;;
    sac) run_sac_scale "$arm" "$radius" "$tag" "$value" "$level" "$gpu" ;;
    channel) run_ppo_channel "$arm" "$radius" "$tag" "$value" "$level" "$gpu" ;;
    rudder_scale) run_ppo_rudder_scale "$arm" "$radius" "$tag" "$value" "$level" "$gpu" ;;
    reward) run_reward_zero "$arm" "$radius" "$level" "$gpu" ;;
    *) echo "unknown job kind: $kind" >&2; return 2 ;;
  esac
}

export REPO PY SB3 FORMAL_ROOT TRAIN_ROOT ROOT LOG_ROOT ENV50
export SEED_START SEED_COUNT MIN_CURRICULUM_EPISODES
export -f radius_value route_env old_env level_args
export -f run_ppo_scale run_sac_scale run_ppo_channel run_ppo_rudder_scale
export -f run_reward_zero dispatch

mkdir -p "$ROOT" "$LOG_ROOT" "$ROOT/manifests"
"$PY" scripts/audit_goal171_artifacts.py \
  --phase train --root "$FORMAL_ROOT" --seeds 86201 \
  --min-episodes "$MIN_CURRICULUM_EPISODES"
"$PY" scripts/check_goal172_protocol.py --phase input --formal-root "$FORMAL_ROOT"

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
"$PY" scripts/check_goal172_protocol.py --phase emit-jobs >"$JOB_FILE"

{
  printf 'started_at=%s\nhost=%s\nrepo=%s\nformal_root=%s\nroot=%s\n' \
    "$(date -Iseconds)" "$(hostname)" "$REPO" "$FORMAL_ROOT" "$ROOT"
  printf 'levels=nominal,current07,current10,current15\nradii=default:2.0,strict125:1.25\n'
  printf 'dev_seeds=%s-%s\n' "$SEED_START" "$((SEED_START + SEED_COUNT - 1))"
  sha256sum scripts/run_goal172_residual_attribution.sh \
    scripts/goal172_matrix_contract.py scripts/check_goal172_protocol.py \
    scripts/analyze_goal172_residual_attribution.py
} >"$ROOT/manifests/diagnostic_manifest.txt"

read -r -a GPU_ARRAY <<<"$GPUS"
index=0
job_count=$(wc -l <"$JOB_FILE")
echo "launching $job_count Goal172 evaluations on ${#GPU_ARRAY[@]} GPUs"
mapfile -t JOBS <"$JOB_FILE"
failures=0
while [[ $index -lt ${#JOBS[@]} ]]; do
  pids=()
  batch_start=$index
  while [[ $index -lt ${#JOBS[@]} && $((index - batch_start)) -lt ${#GPU_ARRAY[@]} ]]; do
    gpu=${GPU_ARRAY[$((index - batch_start))]}
    IFS=$'\t' read -r -a fields <<<"${JOBS[$index]}"
    dispatch "${fields[0]}" "${fields[1]}" "${fields[2]}" "${fields[3]}" \
      "${fields[4]}" "${fields[5]}" "$gpu" &
    pids+=("$!")
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failures=$((failures + 1)); fi
  done
done

printf 'finished_at=%s\nfailures=%s\n' "$(date -Iseconds)" "$failures" \
  >>"$ROOT/manifests/diagnostic_manifest.txt"
if [[ "$failures" -ne 0 ]]; then
  echo "Goal172 diagnostic failures: $failures" >&2
  exit 1
fi
"$PY" scripts/check_goal172_protocol.py --phase output --root "$ROOT"
echo "Goal172 complete full-matrix diagnostic: $ROOT"
