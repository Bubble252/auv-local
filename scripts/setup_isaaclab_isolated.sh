#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUV_ROOT:-/data/yhx/underwater-robotics}"
REPO="${AUV_REPO:-${ROOT}/src/plan_p1/settled_局部规划器_反应式}"
BASE_PYTHON="${ISAACLAB_BASE_PYTHON:-${ROOT}/envs/conda_envs/oceansim/bin/python}"
VENV="${ISAACLAB_VENV:-${ROOT}/envs/venvs/isaaclab}"
LAB_HOME="${ISAACLAB_HOME:-${ROOT}/simulator/IsaacLab}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/isaaclab_install.log"
MANIFEST="${REPO}/artifacts/infra/isaaclab/install_manifest.json"
ISAACSIM_SPEC="${ISAACSIM_SPEC:-isaacsim[all,extscache]==5.0.0.0}"
ISAACSIM_REQUIRED="${ISAACSIM_REQUIRED:-5.0.0.0}"
ISAACLAB_REF="${ISAACLAB_REF:-v2.2.1}"
ISAACLAB_REPO="${ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_INSTALL_FRAMEWORK="${ISAACLAB_INSTALL_FRAMEWORK:-none}"
CONSTRAINTS_FILE="${ISAACLAB_CONSTRAINTS:-${ROOT}/simulator/isaaclab-constraints.txt}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
NVIDIA_INDEX="${NVIDIA_INDEX:-https://pypi.nvidia.com}"
MIN_FREE_GIB="${ISAACLAB_MIN_FREE_GIB:-25}"

STATUS="RUNNING"
FAILURE_STAGE="none"
BASE_FREEZE_BEFORE=""
BASE_FREEZE_AFTER=""
PIP_CHECK_STATUS=0
PIP_CHECK_OUTPUT=""
ISAACSIM_VERSION=""
TORCH_VERSION=""
PYTHON_VERSION=""
ISAACLAB_COMMIT=""
ISAACLAB_DESCRIBE=""

mkdir -p "${LOG_DIR}" "${ROOT}/simulator" "${ROOT}/envs/venvs" "$(dirname "${MANIFEST}")"
exec 9>"${LOG_DIR}/isaaclab_install.lock"
if ! flock -n 9; then
  echo "ERROR: another IsaacLab install is already running" >&2
  exit 3
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

write_manifest() {
  local manifest_python="${VENV}/bin/python"
  if [[ ! -x "${manifest_python}" ]]; then
    manifest_python="${BASE_PYTHON}"
  fi
  MANIFEST_PATH="${MANIFEST}" STATUS="${STATUS}" FAILURE_STAGE="${FAILURE_STAGE}" \
  ROOT="${ROOT}" REPO="${REPO}" VENV="${VENV}" LAB_HOME="${LAB_HOME}" \
  BASE_PYTHON="${BASE_PYTHON}" LOG_FILE="${LOG_FILE}" \
  ISAACSIM_SPEC="${ISAACSIM_SPEC}" ISAACSIM_VERSION="${ISAACSIM_VERSION}" \
  ISAACLAB_REF="${ISAACLAB_REF}" ISAACLAB_COMMIT="${ISAACLAB_COMMIT}" \
  ISAACLAB_DESCRIBE="${ISAACLAB_DESCRIBE}" TORCH_VERSION="${TORCH_VERSION}" \
  ISAACLAB_INSTALL_FRAMEWORK="${ISAACLAB_INSTALL_FRAMEWORK}" \
  CONSTRAINTS_FILE="${CONSTRAINTS_FILE}" \
  PYTHON_VERSION="${PYTHON_VERSION}" BASE_FREEZE_BEFORE="${BASE_FREEZE_BEFORE}" \
  BASE_FREEZE_AFTER="${BASE_FREEZE_AFTER}" PIP_CHECK_STATUS="${PIP_CHECK_STATUS}" \
  PIP_CHECK_OUTPUT="${PIP_CHECK_OUTPUT}" NVIDIA_INDEX="${NVIDIA_INDEX}" \
  TORCH_INDEX="${TORCH_INDEX}" ISAACLAB_REPO="${ISAACLAB_REPO}" \
  "${manifest_python}" - <<'PY' || true
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path

names = (
    "isaacsim", "isaacsim-core", "isaacsim-kernel", "isaacsim-rl",
    "isaaclab", "isaaclab-tasks", "torch", "torchvision", "torchaudio",
    "numpy", "gymnasium", "pillow", "psutil", "stable-baselines3", "warp-lang",
)
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

payload = {
    "schema": "auv_isaaclab_install_manifest.v1",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "status": os.environ["STATUS"],
    "failure_stage": os.environ["FAILURE_STAGE"],
    "root": os.environ["ROOT"],
    "repo": os.environ["REPO"],
    "venv": os.environ["VENV"],
    "isaaclab_home": os.environ["LAB_HOME"],
    "base_python": os.environ["BASE_PYTHON"],
    "log_file": os.environ["LOG_FILE"],
    "isaacsim_spec": os.environ["ISAACSIM_SPEC"],
    "isaacsim_version": os.environ["ISAACSIM_VERSION"],
    "isaaclab_ref": os.environ["ISAACLAB_REF"],
    "isaaclab_commit": os.environ["ISAACLAB_COMMIT"],
    "isaaclab_describe": os.environ["ISAACLAB_DESCRIBE"],
    "isaaclab_repo": os.environ["ISAACLAB_REPO"],
    "torch_version": os.environ["TORCH_VERSION"],
    "isaaclab_install_framework": os.environ["ISAACLAB_INSTALL_FRAMEWORK"],
    "constraints_file": os.environ["CONSTRAINTS_FILE"],
    "python_version": os.environ["PYTHON_VERSION"],
    "packages": packages,
    "pip_check": {
        "returncode": int(os.environ["PIP_CHECK_STATUS"]),
        "output": os.environ["PIP_CHECK_OUTPUT"],
    },
    "oceansim_fingerprint": {
        "pip_freeze_sha256_before": os.environ["BASE_FREEZE_BEFORE"],
        "pip_freeze_sha256_after": os.environ["BASE_FREEZE_AFTER"],
        "unchanged": os.environ["BASE_FREEZE_BEFORE"] == os.environ["BASE_FREEZE_AFTER"],
    },
    "download_sources": {
        "nvidia_python_index": os.environ["NVIDIA_INDEX"],
        "torch_index": os.environ["TORCH_INDEX"],
        "isaaclab_git": os.environ["ISAACLAB_REPO"],
    },
    "commands": {
        "sim_install": "python -m pip install 'isaacsim[all,extscache]==5.0.0.0' --extra-index-url https://pypi.nvidia.com",
        "torch_install": "python -m pip install -U torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
        "isaaclab_install": "./isaaclab.sh --install none",
    },
}
path = Path(os.environ["MANIFEST_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
temporary.replace(path)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
}

on_error() {
  local code=$?
  STATUS="FAIL"
  if [[ "${FAILURE_STAGE}" == "none" ]]; then
    FAILURE_STAGE="unknown"
  fi
  write_manifest
  echo "ERROR: install failed at stage ${FAILURE_STAGE} with exit code ${code}" >&2
  exit "${code}"
}
trap on_error ERR

echo "[$(date -Iseconds)] START IsaacLab isolated install"
echo "ROOT=${ROOT}"
echo "VENV=${VENV}"
echo "LAB_HOME=${LAB_HOME}"
echo "BASE_PYTHON=${BASE_PYTHON}"
echo "ISAACSIM_SPEC=${ISAACSIM_SPEC}"
echo "ISAACLAB_REF=${ISAACLAB_REF}"

FAILURE_STAGE="validate_paths"
if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "ERROR: base Python not found: ${BASE_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
  echo "ERROR: constraints file not found: ${CONSTRAINTS_FILE}" >&2
  exit 2
fi
free_kib=$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')
if (( free_kib < MIN_FREE_GIB * 1024 * 1024 )); then
  echo "ERROR: ${ROOT} has less than ${MIN_FREE_GIB} GiB free" >&2
  exit 2
fi

FAILURE_STAGE="oceansim_fingerprint_before"
BASE_FREEZE_BEFORE=$("${BASE_PYTHON}" -m pip freeze | sha256sum | awk '{print $1}')
echo "[$(date -Iseconds)] oceansim pip freeze sha256 before: ${BASE_FREEZE_BEFORE}"

FAILURE_STAGE="venv_create"
if [[ ! -e "${VENV}/pyvenv.cfg" ]]; then
  echo "[$(date -Iseconds)] creating isolated venv"
  "${BASE_PYTHON}" -m venv --copies "${VENV}"
else
  echo "[$(date -Iseconds)] reusing isolated venv"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PIP_CACHE_DIR="${ROOT}/cache/pip"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-5}"
export PIP_CONSTRAINT="${CONSTRAINTS_FILE}"
export TMPDIR="${ROOT}/tmp"
export OV_CACHE_DIR="${ROOT}/cache/ov"
export OV_DATA_DIR="${ROOT}/cache/nvidia-omniverse"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${OV_CACHE_DIR}" "${OV_DATA_DIR}"

FAILURE_STAGE="pip_bootstrap"
python -m pip install --prefer-binary --upgrade pip setuptools wheel==0.41.3

FAILURE_STAGE="isaaclab_build_prereqs"
python -m pip install --no-build-isolation flatdict==4.0.1

FAILURE_STAGE="isaacsim_install"
echo "[$(date -Iseconds)] installing Isaac Sim ${ISAACSIM_SPEC}"
python -m pip install --prefer-binary "${ISAACSIM_SPEC}" --extra-index-url "${NVIDIA_INDEX}"

FAILURE_STAGE="isaacsim_version_check"
ISAACSIM_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("isaacsim"))')
if [[ "${ISAACSIM_VERSION}" != "${ISAACSIM_REQUIRED}" ]]; then
  echo "ERROR: expected isaacsim ${ISAACSIM_REQUIRED}, got ${ISAACSIM_VERSION}" >&2
  exit 4
fi

FAILURE_STAGE="torch_install"
TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
if [[ "${TORCH_VERSION}" != 2.7.0* ]]; then
  echo "[$(date -Iseconds)] installing CUDA-enabled PyTorch ${TORCH_SPEC}"
  python -m pip install --prefer-binary -U ${TORCH_SPEC} --index-url "${TORCH_INDEX}"
  TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)')
fi

FAILURE_STAGE="isaaclab_checkout"
if [[ ! -d "${LAB_HOME}/.git" ]]; then
  echo "[$(date -Iseconds)] cloning IsaacLab ${ISAACLAB_REF}"
  git clone --branch "${ISAACLAB_REF}" --depth 1 "${ISAACLAB_REPO}" "${LAB_HOME}"
else
  echo "[$(date -Iseconds)] reusing IsaacLab checkout"
  if [[ -n "$(git -C "${LAB_HOME}" status --porcelain)" ]]; then
    echo "ERROR: existing IsaacLab checkout is dirty; refusing to modify it" >&2
    exit 5
  fi
  existing_tag=$(git -C "${LAB_HOME}" describe --tags --exact-match 2>/dev/null || true)
  if [[ "${existing_tag}" != "${ISAACLAB_REF}" ]]; then
    echo "ERROR: existing IsaacLab checkout is '${existing_tag}', expected '${ISAACLAB_REF}'" >&2
    exit 5
  fi
fi

FAILURE_STAGE="isaaclab_install"
cd "${LAB_HOME}"
./isaaclab.sh --install "${ISAACLAB_INSTALL_FRAMEWORK}"

FAILURE_STAGE="torch_repair"
python -m pip install --prefer-binary -U ${TORCH_SPEC} --index-url "${TORCH_INDEX}"

FAILURE_STAGE="cleanup_orphan_httpx"
python - <<'PY'
import importlib
import importlib.metadata as metadata
import subprocess
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


def requiring_distributions(target):
    target = canonicalize_name(target)
    required_by = set()
    importlib.invalidate_caches()
    for distribution in metadata.distributions():
        distributor = canonicalize_name(distribution.metadata.get("Name", ""))
        if distributor == target:
            continue
        for raw_requirement in distribution.requires or []:
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement:
                continue
            if canonicalize_name(requirement.name) == target:
                required_by.add(distributor or "unknown")
                break
    return sorted(required_by)


for package in ("httpx2", "httpcore2", "truststore"):
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        continue
    required_by = requiring_distributions(package)
    if required_by:
        print(f"keeping {package} because it is required by: {required_by}")
        continue
    print(f"removing orphan {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
PY

FAILURE_STAGE="version_capture"
ISAACSIM_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("isaacsim"))')
if [[ "${ISAACSIM_VERSION}" != "${ISAACSIM_REQUIRED}" ]]; then
  echo "ERROR: IsaacLab install changed isaacsim to ${ISAACSIM_VERSION}" >&2
  exit 4
fi
TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)')
PYTHON_VERSION=$(python -c 'import platform; print(platform.python_version())')
ISAACLAB_COMMIT=$(git -C "${LAB_HOME}" rev-parse HEAD)
ISAACLAB_DESCRIBE=$(git -C "${LAB_HOME}" describe --tags --always --dirty)

FAILURE_STAGE="import_check"
python - <<'PY'
import importlib.metadata as metadata
import isaaclab  # noqa: F401
import gymnasium
import PIL
import psutil
import torch
import torchaudio

print("isaacsim", metadata.version("isaacsim"))
print("isaaclab", metadata.version("isaaclab"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchaudio", torchaudio.__version__)
print("gymnasium", gymnasium.__version__)
print("pillow", PIL.__version__)
print("psutil", psutil.__version__)

assert metadata.version("isaacsim") == "5.0.0.0"
assert torch.__version__.startswith("2.7.0")
assert torchaudio.__version__.startswith("2.7.0")
assert gymnasium.__version__ == "1.2.0"
assert PIL.__version__ == "11.2.1"
assert psutil.__version__ == "5.9.8"
PY

FAILURE_STAGE="pip_check"
PIP_CHECK_STATUS=0
PIP_CHECK_OUTPUT=$(python -m pip check 2>&1) || PIP_CHECK_STATUS=$?
echo "${PIP_CHECK_OUTPUT}"
if (( PIP_CHECK_STATUS != 0 )); then
  echo "ERROR: pip check reported blocking conflicts" >&2
  exit 6
fi

FAILURE_STAGE="oceansim_fingerprint_after"
BASE_FREEZE_AFTER=$("${BASE_PYTHON}" -m pip freeze | sha256sum | awk '{print $1}')
if [[ "${BASE_FREEZE_BEFORE}" != "${BASE_FREEZE_AFTER}" ]]; then
  echo "ERROR: oceansim environment changed during installation" >&2
  exit 7
fi

STATUS="PASS"
FAILURE_STAGE="none"
write_manifest
trap - ERR
echo "[$(date -Iseconds)] END IsaacLab isolated install PASS"
