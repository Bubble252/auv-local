#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}
REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal167_action_parameterization_20261112}
ARMS=${ARMS:-"A_actuator_terminal B_command_terminal"}

cd "$REPO"
for arm in $ARMS; do
  out="$ROOT/analysis/$arm"
  mkdir -p "$out"
  "$PY" scripts/analyze_pc_goal165_eval.py \
    --eval-root "$ROOT/eval" \
    --output "$out" \
    --arm "$arm" \
    --label "${arm}_summary"
  "$PY" scripts/analyze_goal165_action_authority.py \
    --eval-root "$ROOT/eval" \
    --output "$out/action_authority" \
    --arm "$arm"
  "$PY" scripts/audit_pc_speed_envelope.py \
    --trace-root "$ROOT/eval/$arm" \
    --output "$out/speed_audit" \
    --modes zero,policy
done
"$PY" scripts/compare_goal167_arms.py \
  --analysis-root "$ROOT/analysis" \
  --output "$ROOT/analysis/goal167_comparison.md"
echo "Goal167 analysis complete: $ROOT/analysis"
