#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal166_terminal_horizon_20261112}

cd "$REPO"
"$PY" scripts/analyze_pc_goal165_eval.py \
  --eval-root "$ROOT/eval" \
  --output "$ROOT/analysis"
"$PY" scripts/analyze_goal165_action_authority.py \
  --eval-root "$ROOT/eval" \
  --output "$ROOT/analysis/action_authority"
"$PY" scripts/audit_pc_speed_envelope.py \
  --trace-root "$ROOT/eval" \
  --output "$ROOT/analysis/speed_audit" \
  --modes zero,policy
echo "Goal166 analysis complete: $ROOT/analysis"
