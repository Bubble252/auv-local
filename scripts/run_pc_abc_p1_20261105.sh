#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"

exec "$PY" "$ROOT/scripts/run_pc_abc_p1_20261105.py" "$@"
