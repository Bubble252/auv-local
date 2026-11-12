#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式}"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
ROOT="${ROOT:-$REPO/artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/pc_v0d_r2b_weighted_bc_dagger_20261107}"
ORIGINAL_DEMOS="$REPO/artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz"
STUDENT_CHECKPOINT="$REPO/artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/bc_satw/pc_bc_actor.pt"
DEFAULT_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json"
STRICT_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_r0_current_strict125.json"
TARGET_SEEDS=(37 34 24 20 39 31 35)
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:-unknown}"

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MERGED="$ROOT/demos_merged_dagger2"
OUTPUT="$ROOT/bc_satw_dagger2"
TRAIN_LOG="$LOG_ROOT/bc_satw_dagger2_train.log"
EVAL_LOG="$LOG_ROOT/bc_satw_dagger2_eval.log"

for path in "$ORIGINAL_DEMOS" "$STUDENT_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required input: $path" >&2
    exit 1
  fi
done
for seed in "${TARGET_SEEDS[@]}"; do
  if [[ ! -f "$ROOT/dagger_seed_$seed/demos.npz" ]]; then
    echo "missing targeted DAgger dataset for seed $seed" >&2
    exit 1
  fi
done
if [[ -e "$MERGED" ]] || [[ -e "$OUTPUT" ]] || [[ -e "$ROOT/eval" ]]; then
  echo "refusing to overwrite existing merge/train/eval output" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT" "$ROOT"

echo "=== $(date -Is) merge targeted DAgger datasets ===" | tee "$TRAIN_LOG"
MERGE_INPUTS=("$ORIGINAL_DEMOS")
for seed in "${TARGET_SEEDS[@]}"; do
  MERGE_INPUTS+=("$ROOT/dagger_seed_$seed/demos.npz")
done
MERGE_CMD=(
  "$PY" "$REPO/scripts/merge_pc_bc_demos.py"
  --inputs "${MERGE_INPUTS[@]}"
  --output "$MERGED"
)
printf '%q ' "${MERGE_CMD[@]}" > "$ROOT/merge_command.txt"
printf '\n' >> "$ROOT/merge_command.txt"
"${MERGE_CMD[@]}" 2>&1 | tee -a "$TRAIN_LOG"

echo "=== $(date -Is) weighted BC training with targeted DAgger ===" | tee -a "$TRAIN_LOG"
TRAIN_CMD=(
  "$PY" "$REPO/scripts/train_pc_bc.py"
  --demos "$MERGED/demos.npz"
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
printf '%q ' "${TRAIN_CMD[@]}" > "$OUTPUT.train_command.tmp"
mkdir -p "$OUTPUT"
mv "$OUTPUT.train_command.tmp" "$OUTPUT/train_command.txt"
"${TRAIN_CMD[@]}" 2>&1 | tee -a "$TRAIN_LOG"

CHECKPOINT="$OUTPUT/pc_bc_actor.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint after training: $CHECKPOINT" >&2
  exit 1
fi

echo "=== $(date -Is) action saturation audit ===" | tee "$EVAL_LOG"
AUDIT_CMD=(
  "$PY" "$REPO/scripts/audit_pc_bc_action_saturation.py"
  --checkpoint "$CHECKPOINT"
  --demos "$MERGED/demos.npz"
  --output "$OUTPUT/action_saturation.json"
  --threshold 0.99
  --device cpu
)
printf '%q ' "${AUDIT_CMD[@]}" > "$ROOT/eval_commands.txt"
printf '\n' >> "$ROOT/eval_commands.txt"
"${AUDIT_CMD[@]}" 2>&1 | tee -a "$EVAL_LOG"

EVAL_SPECS=(
  "dev_smoke|$DEFAULT_CONFIG|20|2|"
  "test_default|$DEFAULT_CONFIG|0|20|"
  "test_strict125|$STRICT_CONFIG|0|20|"
  "test_pathzero|$DEFAULT_CONFIG|0|20|--path-zero"
)
for spec in "${EVAL_SPECS[@]}"; do
  IFS='|' read -r name config seed_start seed_count extra <<<"$spec"
  echo "=== $(date -Is) evaluate $name ===" | tee -a "$EVAL_LOG"
  EVAL_CMD=(
    "$PY" "$REPO/scripts/evaluate_pc_bc.py"
    --config "$config"
    --checkpoint "$CHECKPOINT"
    --output "$ROOT/eval/$name"
    --seed-start "$seed_start"
    --seed-count "$seed_count"
    --device cpu
    --dtype float64
    --path-dim 8
  )
  if [[ "$extra" == "--path-zero" ]]; then
    EVAL_CMD+=(--path-zero)
  fi
  printf '%q ' "${EVAL_CMD[@]}" >> "$ROOT/eval_commands.txt"
  printf '\n' >> "$ROOT/eval_commands.txt"
  "${EVAL_CMD[@]}" 2>&1 | tee -a "$EVAL_LOG"
done

SOURCE_GIT_HEAD="$SOURCE_GIT_HEAD" "$PY" - "$REPO" "$ROOT" "$MERGED" "$OUTPUT" "$CHECKPOINT" "$TRAIN_LOG" "${TARGET_SEEDS[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

repo, root, merged, output, checkpoint, train_log = map(pathlib.Path, sys.argv[1:7])
target_seeds = [int(value) for value in sys.argv[7:]]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {
    "schema_version": "auv_pc_v0d_weighted_bc_dagger_run_manifest.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_git_head": os.environ.get("SOURCE_GIT_HEAD", "unknown"),
    "server_git_repository": False,
    "target_seeds": target_seeds,
    "merge_command": (root / "merge_command.txt").read_text(encoding="utf-8").strip(),
    "train_command": (output / "train_command.txt").read_text(encoding="utf-8").strip(),
    "eval_commands": (root / "eval_commands.txt").read_text(encoding="utf-8").splitlines(),
    "original_demos": str((repo / "artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz").resolve()),
    "original_demos_sha256": sha256(repo / "artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz"),
    "targeted_demos": [
        {
            "seed": seed,
            "path": str((root / f"dagger_seed_{seed}/demos.npz").resolve()),
            "sha256": sha256(root / f"dagger_seed_{seed}/demos.npz"),
        }
        for seed in target_seeds
    ],
    "merged_dataset": str((merged / "demos.npz").resolve()),
    "merged_dataset_sha256": sha256(merged / "demos.npz"),
    "merged_manifest": str((merged / "demos.manifest.json").resolve()),
    "checkpoint_path": str(checkpoint.resolve()),
    "checkpoint_sha256": sha256(checkpoint),
    "training_log": str(train_log.resolve()),
    "evaluation_seeds": {
        "dev_smoke": [20, 21],
        "test": list(range(20)),
    },
    "device": {"train": "cuda", "evaluation": "cpu"},
    "dtype": {"train": "float32", "evaluation": "float64"},
}
(root / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "weighted_bc_targeted_dagger_pipeline_complete $(date -Is)" | tee -a "$EVAL_LOG"
