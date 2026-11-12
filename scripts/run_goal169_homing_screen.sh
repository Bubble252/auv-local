#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal169_terminal_homing_timing_20261112}
VARIANTS=${VARIANTS:-"20 30 40 50"}
LEVELS=${LEVELS:-"nominal current07 current10 current15"}
RADII=${RADII:-"default:2.0 strict125:1.25"}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
PARALLEL=${PARALLEL:-8}

cd "$REPO"
mkdir -p "$ROOT/eval/logs" "$ROOT/eval/manifests"

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_count=0
for variant in $VARIANTS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      echo "$variant ${radius%%:*} ${radius##*:} $level $job_count" >>"$JOB_FILE"
      job_count=$((job_count + 1))
    done
  done
done

run_one() {
  local variant=$1 radius_label=$2 radius_value=$3 level=$4 index=$5
  local output="$ROOT/eval/alignment$variant/$radius_label/$level/zero_seed86201"
  local log="$ROOT/eval/logs/alignment${variant}_${radius_label}_${level}.log"
  local current_args=() shear_args=() gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu=${gpu_array[$((index % ${#gpu_array[@]}))]}
  case "$level" in
    nominal) ;;
    current07) current_args=(--curriculum-current-speed-range-mps 0.275 0.50); shear_args=(--curriculum-shear-scale 0.75) ;;
    current10) current_args=(--curriculum-current-speed-range-mps 0.30 0.55); shear_args=(--curriculum-shear-scale 1.0) ;;
    current15) current_args=(--curriculum-current-speed-range-mps 0.375 0.675); shear_args=(--curriculum-shear-scale 1.5) ;;
    *) echo "unknown level: $level" >&2; return 2 ;;
  esac
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP alignment$variant $radius_label $level"
    return 0
  fi
  if [[ -d "$output" ]] && [[ -n "$(ls -A "$output" 2>/dev/null)" ]]; then
    echo "refusing non-empty output: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "START alignment=$variant radius=$radius_label level=$level gpu=$gpu"
  if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config configs/lowlevel/goal165/eval_g1_gain_near10_2.json \
      --environment-config "configs/lowlevel/goal169/env_terminal_alignment_${variant}_eval_nu_ground.json" \
      --curriculum-stage C3 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
      --eval-goal-radius-m "$radius_value" \
      --device "$DEVICE" \
      --dtype float64 \
      --action-mode zero-residual \
      --residual-scale 1.0 \
      --learner-terminal-gate-mode none \
      --rich-trace \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --overwrite \
      "${current_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1; then
    echo "FAILED alignment=$variant radius=$radius_label level=$level (see $log)" >&2
    return 1
  fi
  echo "DONE alignment=$variant radius=$radius_label level=$level"
}

export PY REPO SB3 ROOT GPUS SEED_START SEED_COUNT DEVICE
export -f run_one

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\nvariants=%s\nlevels=%s\nradii=%s\n' "$REPO" "$VARIANTS" "$LEVELS" "$RADII"
  sha256sum scripts/run_goal169_homing_screen.sh \
    configs/lowlevel/goal165/eval_g1_gain_near10_2.json \
    configs/lowlevel/goal169/env_terminal_alignment_20_eval_nu_ground.json \
    configs/lowlevel/goal169/env_terminal_alignment_30_eval_nu_ground.json \
    configs/lowlevel/goal169/env_terminal_alignment_40_eval_nu_ground.json \
    configs/lowlevel/goal169/env_terminal_alignment_50_eval_nu_ground.json
} >"$ROOT/eval/manifests/run_manifest.txt" 2>&1

echo "launching $job_count Goal169 zero evaluations with PARALLEL=$PARALLEL"
xargs -r -n 5 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
for variant in $VARIANTS; do
  for radius in $RADII; do
    for level in $LEVELS; do
      if [[ ! -f "$ROOT/eval/alignment$variant/${radius%%:*}/$level/zero_seed86201/summary.json" ]]; then
        echo "MISSING alignment$variant ${radius%%:*} $level" >&2
        missing=$((missing + 1))
      fi
    done
  done
done
printf 'finished_at=%s\n' "$(date -Iseconds)" >>"$ROOT/eval/manifests/run_manifest.txt"
if [[ "$missing" -ne 0 ]]; then
  echo "$missing Goal169 evaluations missing" >&2
  exit 1
fi
echo "Goal169 zero screen complete: $ROOT/eval"

