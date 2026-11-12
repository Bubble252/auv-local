#!/usr/bin/env bash
set -uo pipefail

ROOT="${AUV_ROOT:-/data/yhx/underwater-robotics}"
REPO="${AUV_REPO:-${ROOT}/src/plan_p1/settled_局部规划器_反应式}"
LAB_HOME="${ISAACLAB_HOME:-${ROOT}/simulator/IsaacLab}"
VENV="${ISAACLAB_VENV:-${ROOT}/envs/venvs/isaaclab}"
GPU="${ISAACLAB_GPU:-4}"
STEPS="${ISAACLAB_STEPS:-200}"
WARMUP="${ISAACLAB_WARMUP:-20}"
REPEATS="${ISAACLAB_REPEATS:-3}"
NUM_ENVS="${ISAACLAB_NUM_ENVS:-1 64 256 1024}"
TASK="${ISAACLAB_TASK:-Isaac-Cartpole-v0}"
OUT_DIR="${ISAACLAB_OUT_DIR:-${REPO}/artifacts/infra/isaaclab/probes}"
LOG_DIR="${ISAACLAB_LOG_DIR:-${ROOT}/logs/isaaclab_scaling}"
TIMEOUT_S="${ISAACLAB_TIMEOUT_S:-1800}"
MAX_START_MIB="${ISAACLAB_MAX_START_MIB:-1024}"
ALLOW_BUSY=0
FORCE=0
SUMMARIZE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --max-start-mib) MAX_START_MIB="$2"; shift 2 ;;
    --allow-busy) ALLOW_BUSY=1; shift ;;
    --force) FORCE=1; shift ;;
    --no-summarize) SUMMARIZE=0; shift ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: isolated venv not found: ${VENV}/bin/python" >&2
  exit 2
fi
if [[ ! -x "${LAB_HOME}/isaaclab.sh" ]]; then
  echo "ERROR: IsaacLab checkout not found: ${LAB_HOME}/isaaclab.sh" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
echo "[$(date -Iseconds)] scaling probe start gpu=${GPU} task=${TASK} steps=${STEPS} warmup=${WARMUP} repeats=${REPEATS} num_envs='${NUM_ENVS}'" | tee -a "${LOG_DIR}/scaling.log"

read -r start_mem start_util driver_version < <(
  nvidia-smi --id="${GPU}" --query-gpu=memory.used,utilization.gpu,driver_version \
    --format=csv,noheader,nounits | awk -F', ' '{print $1, $2, $3}'
)
if [[ -z "${start_mem:-}" ]]; then
  echo "ERROR: cannot query GPU ${GPU}" >&2
  exit 2
fi
echo "[$(date -Iseconds)] GPU ${GPU}: memory_used=${start_mem}MiB util=${start_util}% driver=${driver_version}" | tee -a "${LOG_DIR}/scaling.log"
if (( start_mem > MAX_START_MIB )) && (( ALLOW_BUSY == 0 )); then
  echo "ERROR: GPU ${GPU} already uses ${start_mem} MiB (> ${MAX_START_MIB} MiB)." >&2
  echo "Choose an idle GPU or pass --allow-busy after confirming no training job is disturbed." >&2
  exit 3
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONNOUSERSITE=1
export ISAACLAB_PHYSICAL_GPU="${GPU}"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export RESOURCE_NAME=IsaacSim

failed_runs=0
for count in ${NUM_ENVS}; do
  for ((repeat=0; repeat<REPEATS; repeat++)); do
    output="${OUT_DIR}/num_envs_${count}_repeat_${repeat}.json"
    log="${LOG_DIR}/num_envs_${count}_repeat_${repeat}.log"
    if (( FORCE == 0 )) && [[ -f "${output}" ]] && grep -q '"status": "PASS"' "${output}"; then
      echo "[$(date -Iseconds)] SKIP existing PASS num_envs=${count} repeat=${repeat}" | tee -a "${LOG_DIR}/scaling.log"
      continue
    fi
    echo "[$(date -Iseconds)] START num_envs=${count} repeat=${repeat}" | tee -a "${LOG_DIR}/scaling.log"
    timeout "${TIMEOUT_S}" "${LAB_HOME}/isaaclab.sh" -p "${REPO}/scripts/probe_isaaclab_scaling.py" \
      --headless \
      --device "cuda:${GPU}" \
      --task "${TASK}" \
      --num_envs "${count}" \
      --steps "${STEPS}" \
      --warmup_steps "${WARMUP}" \
      --seed 0 \
      --repeat-index "${repeat}" \
      --output "${output}" >"${log}" 2>&1
    status=$?
    if [[ ! -f "${output}" ]]; then
      python3 - "${output}" "${count}" "${repeat}" "${status}" "${log}" <<'PY'
import json
import sys
import time
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema": "auv_isaaclab_scaling_probe.v1",
    "status": "FAIL_NO_OUTPUT",
    "num_envs": int(sys.argv[2]),
    "repeat_index": int(sys.argv[3]),
    "returncode": int(sys.argv[4]),
    "log": sys.argv[5],
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
      status=1
    fi
    if (( status != 0 )); then
      failed_runs=$((failed_runs + 1))
      echo "[$(date -Iseconds)] FAIL num_envs=${count} repeat=${repeat} status=${status} log=${log}" | tee -a "${LOG_DIR}/scaling.log"
    else
      echo "[$(date -Iseconds)] PASS num_envs=${count} repeat=${repeat} log=${log}" | tee -a "${LOG_DIR}/scaling.log"
    fi
    sleep 2
  done
done

if ! python3 "${REPO}/scripts/summarize_isaaclab_smoke.py" \
  --input "${OUT_DIR}" \
  --output "${REPO}/artifacts/infra/isaaclab/smoke_1env.json" \
  --required-repeats "${REPEATS}" \
  --required-steps "${STEPS}"; then
  failed_runs=$((failed_runs + 1))
  echo "[$(date -Iseconds)] FAIL single-environment smoke gate" | tee -a "${LOG_DIR}/scaling.log"
fi

if (( SUMMARIZE == 1 )); then
  python3 "${REPO}/scripts/summarize_isaaclab_scaling.py" \
    --input "${OUT_DIR}" \
    --csv "${REPO}/artifacts/infra/isaaclab/scaling.csv" \
    --report "${REPO}/artifacts/infra/isaaclab/scale_report.json" \
    --required-repeats "${REPEATS}" \
    --required-steps "${STEPS}" || true
fi

echo "[$(date -Iseconds)] scaling probe finished failed_runs=${failed_runs}" | tee -a "${LOG_DIR}/scaling.log"
exit $(( failed_runs > 0 ? 1 : 0 ))
