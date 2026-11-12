#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式}"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
ROOT="${ROOT:-$REPO/artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/pc_v0d_r2b_saturation_bc_20261107}"
DEMOS="$REPO/artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz"
DEFAULT_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json"
STRICT_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_r0_current_strict125.json"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:-unknown}"

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT="$ROOT/bc_satw"
TRAIN_LOG="$LOG_ROOT/bc_satw_train.log"
EVAL_LOG="$LOG_ROOT/bc_satw_eval.log"

if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite existing result: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT" "$ROOT/eval" "$LOG_ROOT"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUTPUT/started_at_utc.txt"

TRAIN_CMD=(
  "$PY" "$REPO/scripts/train_pc_bc.py"
  --demos "$DEMOS"
  --output "$OUTPUT"
  --state-encoder-hidden-sizes "128,128"
  --actor-hidden-sizes "256,128"
  --d-z-total 64
  --n-in-reserved 8
  --n-z-reserved 4
  --activation elu
  --batch-size 4096
  --max-epochs 300
  --patience 30
  --learning-rate 1e-3
  --weight-decay 1e-5
  --validation-fraction 0.15
  --seed 0
  --device cuda
  --actor-command-dim 0
  --path-dim 8
  --path-encoder-hidden-sizes 64
  --path-encoder-alpha 1.0
  --saturation-weight 2.0
  --saturation-threshold 0.9
)
printf '%q ' "${TRAIN_CMD[@]}" > "$OUTPUT/train_command.txt"
printf '\n' >> "$OUTPUT/train_command.txt"

echo "=== $(date -Is) weighted BC training ===" | tee "$TRAIN_LOG"
"${TRAIN_CMD[@]}" 2>&1 | tee -a "$TRAIN_LOG"

CHECKPOINT="$OUTPUT/pc_bc_actor.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint after training: $CHECKPOINT" >&2
  exit 1
fi

echo "=== $(date -Is) action saturation audit ===" | tee -a "$EVAL_LOG"
AUDIT_CMD=(
  "$PY" "$REPO/scripts/audit_pc_bc_action_saturation.py"
  --checkpoint "$CHECKPOINT"
  --demos "$DEMOS"
  --output "$OUTPUT/action_saturation.json"
  --threshold 0.99
  --device cpu
)
printf '%q ' "${AUDIT_CMD[@]}" > "$OUTPUT/eval_commands.txt"
printf '\n' >> "$OUTPUT/eval_commands.txt"
"${AUDIT_CMD[@]}" 2>&1 | tee -a "$EVAL_LOG"

for spec in \
  "satw_default|$DEFAULT_CONFIG|" \
  "satw_strict125|$STRICT_CONFIG|" \
  "satw_pathzero|$DEFAULT_CONFIG|--path-zero"; do
  IFS='|' read -r name config extra <<<"$spec"
  echo "=== $(date -Is) evaluate $name ===" | tee -a "$EVAL_LOG"
  EVAL_CMD=(
    "$PY" "$REPO/scripts/evaluate_pc_bc.py"
    --config "$config"
    --checkpoint "$CHECKPOINT"
    --output "$ROOT/eval/$name"
    --seed-start 0
    --seed-count 20
    --device cpu
    --dtype float64
    --path-dim 8
  )
  if [[ "$extra" == "--path-zero" ]]; then
    EVAL_CMD+=(--path-zero)
  fi
  printf '%q ' "${EVAL_CMD[@]}" >> "$OUTPUT/eval_commands.txt"
  printf '\n' >> "$OUTPUT/eval_commands.txt"
  "${EVAL_CMD[@]}" 2>&1 | tee -a "$EVAL_LOG"
done

SOURCE_GIT_HEAD="$SOURCE_GIT_HEAD" "$PY" - "$REPO" "$ROOT" "$OUTPUT" "$DEMOS" "$CHECKPOINT" "$TRAIN_LOG" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

repo, root, output, demos, checkpoint, train_log = map(pathlib.Path, sys.argv[1:7])


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = output / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
train_command = (output / "train_command.txt").read_text(encoding="utf-8").strip()
started_at = (output / "started_at_utc.txt").read_text(encoding="utf-8").strip()
manifest.update(
    {
        "schema_version": "auv_pc_v0d_r2b_weighted_bc_run_manifest.v1",
        "started_at_utc": started_at,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": os.environ.get("SOURCE_GIT_HEAD", "unknown"),
        "server_git_repository": False,
        "git_diff_stat": "server checkout has no .git; local worktree changes are listed in the goal report",
        "command_line": train_command,
        "evaluation_commands": (output / "eval_commands.txt").read_text(encoding="utf-8").splitlines(),
        "dataset_path": str(demos.resolve()),
        "dataset_sha256": sha256(demos),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "seeds": {"train": [0], "evaluation": list(range(20))},
        "device": {"train": "cuda", "evaluation": "cpu"},
        "dtype": {"train": "float32", "evaluation": "float64"},
        "num_envs": None,
        "saturation_weight": 2.0,
        "saturation_threshold": 0.9,
        "eval_outputs": [
            str((root / "eval" / name / "summary.json").resolve())
            for name in ("satw_default", "satw_strict125", "satw_pathzero")
        ],
        "training_log": str(train_log.resolve()),
    }
)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "weighted_bc_pipeline_complete $(date -Is)" | tee -a "$EVAL_LOG"
