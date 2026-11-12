#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式}"
PY="${PY:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
ROOT="${ROOT:-$REPO/artifacts/lowlevel/pc_v0d_sign_magnitude_bc_20261107}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/pc_v0d_sign_magnitude_bc_20261107}"
DEMOS="$REPO/artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz"
DEFAULT_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json"
STRICT_CONFIG="$REPO/configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_r0_current_strict125.json"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:-unknown}"

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -d "$ROOT" ]] && [[ -n "$(find "$ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite existing result: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT/s0_continuous" "$ROOT/s1_sign_magnitude" "$ROOT/audit" "$ROOT/eval" "$LOG_ROOT"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$ROOT/started_at_utc.txt"

train_run() {
  local tag="$1"
  local output="$2"
  local action_head="$3"
  local command_file="$output/train_command.txt"
  local log_file="$LOG_ROOT/${tag}_train.log"
  local command=(
    "$PY" "$REPO/scripts/train_pc_bc.py"
    --demos "$DEMOS"
    --output "$output"
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
    --saturation-weight 0.0
    --saturation-threshold 0.9
    --action-head "$action_head"
    --sign-deadband 0.05
  )
  printf '%q ' "${command[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  echo "=== $(date -Is) train $tag ($action_head) ===" | tee "$log_file"
  "${command[@]}" 2>&1 | tee -a "$log_file"
}

audit_run() {
  local tag="$1"
  local checkpoint="$2"
  local output="$ROOT/audit/${tag}_action_saturation.json"
  local log_file="$LOG_ROOT/${tag}_audit.log"
  local command=(
    "$PY" "$REPO/scripts/audit_pc_bc_action_saturation.py"
    --checkpoint "$checkpoint"
    --demos "$DEMOS"
    --config "$DEFAULT_CONFIG"
    --output "$output"
    --threshold 0.99
    --batch-size 8192
    --device cpu
  )
  printf '%q ' "${command[@]}" > "$ROOT/audit/${tag}_command.txt"
  printf '\n' >> "$ROOT/audit/${tag}_command.txt"
  echo "=== $(date -Is) audit $tag ===" | tee "$log_file"
  "${command[@]}" 2>&1 | tee -a "$log_file"
}

eval_run() {
  local tag="$1"
  local config="$2"
  local checkpoint="$3"
  local extra="${4:-}"
  local output="$ROOT/eval/$tag"
  local log_file="$LOG_ROOT/${tag}_eval.log"
  local command=(
    "$PY" "$REPO/scripts/evaluate_pc_bc.py"
    --config "$config"
    --checkpoint "$checkpoint"
    --output "$output"
    --seed-start 0
    --seed-count 20
    --device cpu
    --dtype float64
    --path-dim 8
  )
  if [[ "$extra" == "--path-zero" ]]; then
    command+=(--path-zero)
  fi
  printf '%q ' "${command[@]}" >> "$ROOT/eval_commands.txt"
  printf '\n' >> "$ROOT/eval_commands.txt"
  echo "=== $(date -Is) evaluate $tag ===" | tee "$log_file"
  "${command[@]}" 2>&1 | tee -a "$log_file"
}

train_run "s0_continuous" "$ROOT/s0_continuous" continuous
train_run "s1_sign_magnitude" "$ROOT/s1_sign_magnitude" sign_magnitude

S0_CHECKPOINT="$ROOT/s0_continuous/pc_bc_actor.pt"
S1_CHECKPOINT="$ROOT/s1_sign_magnitude/pc_bc_actor.pt"
[[ -f "$S0_CHECKPOINT" && -f "$S1_CHECKPOINT" ]]

audit_run "s0" "$S0_CHECKPOINT"
audit_run "s1" "$S1_CHECKPOINT"

eval_run "s0_default" "$DEFAULT_CONFIG" "$S0_CHECKPOINT"
eval_run "s0_strict125" "$STRICT_CONFIG" "$S0_CHECKPOINT"
eval_run "s0_pathzero" "$DEFAULT_CONFIG" "$S0_CHECKPOINT" --path-zero
eval_run "s1_default" "$DEFAULT_CONFIG" "$S1_CHECKPOINT"
eval_run "s1_strict125" "$STRICT_CONFIG" "$S1_CHECKPOINT"
eval_run "s1_pathzero" "$DEFAULT_CONFIG" "$S1_CHECKPOINT" --path-zero

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$ROOT/finished_at_utc.txt"
SOURCE_GIT_HEAD="$SOURCE_GIT_HEAD" "$PY" - "$REPO" "$ROOT" "$DEMOS" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

repo, root, demos = map(pathlib.Path, sys.argv[1:4])


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


checkpoints = {
    "s0_continuous": root / "s0_continuous" / "pc_bc_actor.pt",
    "s1_sign_magnitude": root / "s1_sign_magnitude" / "pc_bc_actor.pt",
}
payload = {
    "schema_version": "auv_pc_v0d_sign_magnitude_bc_run.v1",
    "source_git_head": os.environ.get("SOURCE_GIT_HEAD", "unknown"),
    "repo": str(repo),
    "root": str(root),
    "demos": str(demos),
    "dataset_sha256": sha256(demos),
    "started_at_utc": (root / "started_at_utc.txt").read_text().strip(),
    "finished_at_utc": (root / "finished_at_utc.txt").read_text().strip(),
    "checkpoints": {
        name: {
            "path": str(path),
            "sha256": sha256(path),
            "training_manifest": str(path.parent / "training_manifest.json"),
            "training_manifest_sha256": sha256(path.parent / "training_manifest.json"),
        }
        for name, path in checkpoints.items()
    },
    "eval_commands": (root / "eval_commands.txt").read_text().splitlines(),
}
(root / "run_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload["checkpoints"], indent=2, sort_keys=True))
PY
