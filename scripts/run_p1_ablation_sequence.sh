#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG [CONFIG ...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${P1_PYTHON:-python}"
CALIBRATION="${P1_CALIBRATION:-${ROOT}/outputs/m4/p1/sonar_calibration/calibration.json}"
DEVICE="${P1_DEVICE:-cuda}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for config in "$@"; do
  absolute="$(realpath "${config}")"
  name="$(basename "${absolute}" .json)"
  family="$(basename "$(dirname "${absolute}")")"
  log_dir="${ROOT}/outputs/m4/p1/ablations/${family}/${name}"
  mkdir -p "${log_dir}"
  echo "[$(date -Iseconds)] START ${name}" | tee -a "${ROOT}/outputs/m4/p1/ablations/campaign.log"
  "${PYTHON_BIN}" "${ROOT}/scripts/train_p1_oceansim.py" \
    --config "${absolute}" \
    --algorithm ddpg \
    --device "${DEVICE}" \
    --calibration "${CALIBRATION}" \
    >>"${log_dir}/launch.log" 2>&1
  echo "[$(date -Iseconds)] END ${name}" | tee -a "${ROOT}/outputs/m4/p1/ablations/campaign.log"
done
