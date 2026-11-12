#!/usr/bin/env bash
# Goal156 addendum: pressure-screen the frozen Goal86 1M checkpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
SB3="${SB3:-/data/yhx/underwater-robotics/src/stable-baselines3}"
GLOBAL_SRC="${GLOBAL_SRC:-}"
ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal86_1m_standing_early_20261112}"
CONFIG_ROOT="$ROOT/standing_configs"
MANIFEST="$CONFIG_ROOT/manifest.json"
ARMS="$ROOT/standing_arms.json"
OUT="$ROOT/standing_eval"
MAX_PARALLEL="${MAX_PARALLEL:-16}"

cd "$REPO_ROOT"
mkdir -p "$ROOT"

if [[ ! -f "$MANIFEST" ]]; then
  "$PY" scripts/build_pc_standing_eval_configs.py \
    --tier A_screen \
    --base-env configs/lowlevel/goal136/curriculum/C1.json \
    --output-dir "$CONFIG_ROOT" \
    --pressure nominal current10 noise_n2 plant20 \
    --route legacy_sine --length 35 --speed 1.2 \
    --radius default strict125
else
  echo "using existing frozen A_screen manifest: $MANIFEST"
fi

"$PY" scripts/build_goal86_1m_standing_arms.py --output "$ARMS"

"$PY" scripts/run_pc_standing_eval_suite.py \
  --manifest "$MANIFEST" \
  --arms "$ARMS" \
  --output-root "$OUT" \
  --python "$PY" \
  --sb3-root "$SB3" \
  --global-src "$GLOBAL_SRC" \
  --seed-start 20 --seed-count 40 \
  --device cpu --dtype float64 \
  --max-parallel "$MAX_PARALLEL" \
  --skip-existing --analyze --zero-tag zero

echo "Goal86 1M standing early screen complete: $OUT"
