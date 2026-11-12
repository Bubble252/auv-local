#!/usr/bin/env bash
# Goal159 addendum: eval-only residual-scale dose on the H3 screen.
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
SB3=${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}
TRAIN_ROOT=${TRAIN_ROOT:-$REPO/artifacts/lowlevel/goal159_h3_current_sac_screen_20261112/train}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal159_scale_diag_20261112}
PARALLEL=${PARALLEL:-6}
SEED_START=${SEED_START:-20}
SEED_COUNT=${SEED_COUNT:-40}
DEVICE=${DEVICE:-cuda}
GPUS=${GPUS:-"3 4 6 7"}
LEVELS=${LEVELS:-"current07 current10 current15"}
SCALES=${SCALES:-"0.25 0.5 1.0"}
POLICY_CONFIG=${POLICY_CONFIG:-configs/lowlevel/goal154/authority/near10_2.json}
ENV_CONFIG=${ENV_CONFIG:-configs/lowlevel/goal157/env_h3.json}
LEARNER_GATE_MODE=${LEARNER_GATE_MODE:-none}

cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/manifests"
for seed in 86201 86202 86203; do
  if [[ ! -f "$TRAIN_ROOT/seed${seed}/final.zip" ]]; then
    echo "missing checkpoint: $TRAIN_ROOT/seed${seed}/final.zip" >&2
    exit 2
  fi
done

{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  sha256sum "$POLICY_CONFIG" "$ENV_CONFIG" \
    "$TRAIN_ROOT/seed86201/final.zip" \
    "$TRAIN_ROOT/seed86202/final.zip" \
    "$TRAIN_ROOT/seed86203/final.zip"
} >"$ROOT/manifests/run_manifest.txt" 2>&1

JOB_FILE=$(mktemp)
trap 'rm -f "$JOB_FILE"' EXIT
job_index=0
for level in $LEVELS; do
  for scale in $SCALES; do
    for seed in 86201 86202 86203; do
      echo "$level $scale $seed $job_index" >>"$JOB_FILE"
      job_index=$((job_index + 1))
    done
  done
done

run_one() {
  local level="$1"
  local scale="$2"
  local seed="$3"
  local index="$4"
  local output="$ROOT/$level/scale${scale}/seed${seed}"
  local log="$ROOT/logs/${level}_scale${scale}_seed${seed}.log"
  local current_speed_args=()
  local shear_args=()
  local gpu_array=()
  read -r -a gpu_array <<<"$GPUS"
  local gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"

  case "$level" in
    current07)
      current_speed_args=(--curriculum-current-speed-range-mps 0.275 0.50)
      shear_args=(--curriculum-shear-scale 0.75)
      ;;
    current10)
      current_speed_args=(--curriculum-current-speed-range-mps 0.30 0.55)
      shear_args=(--curriculum-shear-scale 1.0)
      ;;
    current15)
      current_speed_args=(--curriculum-current-speed-range-mps 0.375 0.675)
      shear_args=(--curriculum-shear-scale 1.5)
      ;;
    *)
      echo "unknown level: $level" >&2
      return 2
      ;;
  esac

  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $level/scale$scale/seed$seed"
    return 0
  fi
  mkdir -p "$output"
  echo "START level=$level scale=$scale seed=$seed gpu=$gpu"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:$SB3" \
    "$PY" scripts/evaluate_pc_sac.py \
      --config "$POLICY_CONFIG" \
      --environment-config "$ENV_CONFIG" \
      --checkpoint "$TRAIN_ROOT/seed${seed}/final.zip" \
      --curriculum-stage C1 \
      --disable-curriculum-direct-command \
      --output "$output" \
      --seed-start "$SEED_START" \
      --seed-count "$SEED_COUNT" \
      --success-frame both \
      --device "$DEVICE" \
      --dtype float64 \
      --action-mode policy \
      --residual-scale "$scale" \
      --learner-terminal-gate-mode "$LEARNER_GATE_MODE" \
      --trace-output "$output" \
      --trace-reward-breakdown \
      --sb3-root "$SB3" \
      --overwrite \
      "${current_speed_args[@]}" \
      "${shear_args[@]}" \
      >"$log" 2>&1
  echo "DONE level=$level scale=$scale seed=$seed"
}

export PY REPO ROOT TRAIN_ROOT SB3 SEED_START SEED_COUNT DEVICE GPUS
export POLICY_CONFIG ENV_CONFIG LEARNER_GATE_MODE
export -f run_one

echo "launching $(wc -l <"$JOB_FILE") Goal159 scale evaluations with PARALLEL=$PARALLEL"
xargs -r -n 4 -P "$PARALLEL" bash -c 'run_one "$@"' _ <"$JOB_FILE"

missing=0
while read -r level scale seed index; do
  if [[ ! -f "$ROOT/$level/scale${scale}/seed${seed}/summary.json" ]]; then
    echo "MISSING $level/scale$scale/seed$seed" >&2
    missing=$((missing + 1))
  fi
done <"$JOB_FILE"

if [[ "$missing" -ne 0 ]]; then
  echo "$missing evaluations are missing" >&2
  exit 1
fi
echo "Goal159 scale diagnostic complete: $ROOT"
