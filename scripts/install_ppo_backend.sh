#!/usr/bin/env bash
set -euo pipefail

# G1: install the rsl_rl PPO backend only into the isolated IsaacLab venv.
# This script is intended to run on the OceanSim server.

BASE="${BASE:-/data/yhx/underwater-robotics}"
ISAACLAB="${ISAACLAB:-$BASE/simulator/IsaacLab}"
ISAACLAB_VENV="${ISAACLAB_VENV:-$BASE/envs/venvs/isaaclab}"
OCEANSIM_PY="${OCEANSIM_PY:-$BASE/envs/conda_envs/oceansim/bin/python}"
LOG_DIR="${LOG_DIR:-$BASE/logs/ppo_isaaclab}"
LOG="${LOG:-$LOG_DIR/install_rsl_rl.log}"

PY="$ISAACLAB_VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "missing IsaacLab Python: $PY" >&2
  exit 2
fi
if [[ ! -d "$ISAACLAB" ]]; then
  echo "missing IsaacLab source: $ISAACLAB" >&2
  exit 2
fi
if [[ ! -x "$OCEANSIM_PY" ]]; then
  echo "missing OceanSim Python: $OCEANSIM_PY" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
export OMNI_KIT_ACCEPT_EULA=YES

isaac_before="$LOG_DIR/isaaclab_before.txt"
oceansim_before="$LOG_DIR/oceansim_before.txt"

"$PY" -m pip freeze > "$isaac_before"
"$OCEANSIM_PY" -m pip freeze > "$oceansim_before"

isaac_hash_before="$(sha256sum "$isaac_before" | cut -d' ' -f1)"
oceansim_hash_before="$(sha256sum "$oceansim_before" | cut -d' ' -f1)"

echo "isaaclab_freeze_before=$isaac_hash_before"
echo "oceansim_freeze_before=$oceansim_hash_before"

# The official IsaacLab entrypoint installs the extra declared by
# source/isaaclab_rl/setup.py: rsl-rl-lib==2.3.3.
set +e
(
  # shellcheck disable=SC1091
  source "$ISAACLAB_VENV/bin/activate"
  cd "$ISAACLAB"
  ./isaaclab.sh --install rsl_rl
) 2>&1 | tee "$LOG"
install_status=${PIPESTATUS[0]}
set -e

if [[ "$install_status" -ne 0 ]]; then
  echo "rsl_rl install failed: status=$install_status" >&2
  exit "$install_status"
fi

isaac_after="$LOG_DIR/isaaclab_after.txt"
oceansim_after="$LOG_DIR/oceansim_after.txt"
"$PY" -m pip freeze > "$isaac_after"
"$OCEANSIM_PY" -m pip freeze > "$oceansim_after"

isaac_hash_after="$(sha256sum "$isaac_after" | cut -d' ' -f1)"
oceansim_hash_after="$(sha256sum "$oceansim_after" | cut -d' ' -f1)"

echo "isaaclab_freeze_after=$isaac_hash_after"
echo "oceansim_freeze_after=$oceansim_hash_after"

"$PY" -m pip check
"$PY" - <<'PY'
import importlib

for name in ("rsl_rl", "isaaclab_rl"):
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'unknown')}")

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

print(f"RslRlVecEnvWrapper={RslRlVecEnvWrapper.__name__}")
PY

if [[ "$oceansim_hash_before" != "$oceansim_hash_after" ]]; then
  echo "OceanSim environment changed; refusing to continue" >&2
  exit 3
fi

echo "oceansim_freeze_unchanged=true"
echo "install_log=$LOG"
