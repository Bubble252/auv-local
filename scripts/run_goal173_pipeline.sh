#!/usr/bin/env bash
# Goal173 formal entrypoint: training is never separated from the complete 88-cell evaluation.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python3}

cd "$REPO"
"$PY" scripts/check_goal173_protocol.py --phase train
bash scripts/run_goal173_train.sh
"$PY" scripts/check_goal173_protocol.py --phase eval
bash scripts/run_goal173_eval.sh
echo "Goal173 pipeline complete: training + 88-cell evaluation + artifact audits"
