#!/usr/bin/env bash
# Goal159c: evaluate the corrected C3/ALOS H3 screen against H3 zero.
set -euo pipefail

REPO=${REPO:-/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111}
export REPO
export TRAIN_ROOT=${TRAIN_ROOT:-$REPO/artifacts/lowlevel/goal159c_h3_current_sac_screen_20261112/train}
export ROOT=${ROOT:-$REPO/artifacts/lowlevel/goal159c_h3_current_sac_screen_20261112/eval}
export REQUIRE_DIRECT_COMMANDS_FALSE=1

exec bash "$REPO/scripts/run_goal159_eval.sh"
