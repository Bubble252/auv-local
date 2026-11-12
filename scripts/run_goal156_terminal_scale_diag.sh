#!/usr/bin/env bash
# Post-hoc Goal156 diagnostic: near10_2 scale sweep on the three 5M checkpoints.
#
# This does NOT change the preregistered Goal156 primary gate; the primary gate
# remains near10_2 scale=1.0 from run_goal156_formal.sh.  This runner only writes
# a separate scale_diag_cpu tree for terminal-authority diagnosis.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/artifacts/lowlevel/goal156_full_train_eval_gate/train}"
export CHECKPOINT_LAYOUT=goal156
export ROOT="${ROOT:-$REPO_ROOT/artifacts/lowlevel/goal156_terminal_scale_diag_20261112}"
export SCALES="${SCALES:-1.0 1.25 1.5}"

exec bash "$REPO_ROOT/scripts/run_goal86_terminal_scale_screen.sh" "$@"
