#!/usr/bin/env bash
# Goal85b single-factor zero-residual diagnostic: 16 runs x 40 episodes.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
OUT_ROOT="${OUT_ROOT:-/data/yhx/underwater-robotics/src/plan_p1/goal85b_plant_mismatch_20261110/b1_component}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json}"
CKPT="${CKPT:-/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts/formal_s240/final.pt}"
SEED_START="${SEED_START:-20}"
SEED_COUNT="${SEED_COUNT:-40}"

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs"

"$PYTHON" - "$OUT_ROOT/configs" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
names = ("drag_scale", "thrust_scale", "fin_scale", "added_mass_scale")
for factor in names:
    for scenario in ("default", "strict125"):
        for root in ("baseline", "candidate"):
            source = Path(
                "configs/lowlevel/goal85b/"
                f"pc_goal85b_m0_{scenario}_{root}_20261110.json"
            )
            data = json.loads(source.read_text(encoding="utf-8"))
            block = {
                "enabled": True,
                "seed": 8505,
                "drag_scale": [1.0, 1.0],
                "thrust_scale": [1.0, 1.0],
                "fin_scale": [1.0, 1.0],
                "added_mass_scale": [1.0, 1.0],
                "actuator_time_scale": [1.0, 1.0],
            }
            block[factor] = [0.9, 1.1]
            data["plant_randomization"] = block
            name = f"pc_goal85b_component_{factor}_{scenario}_{root}_20261110"
            data["experiment"]["name"] = name
            data["experiment"]["output"] = f"artifacts/lowlevel/{name}"
            (output / f"{name}.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
PY

pids=()
names=()
for factor in drag_scale thrust_scale fin_scale added_mass_scale; do
  for scenario in default strict125; do
    for root in baseline candidate; do
      config="$OUT_ROOT/configs/pc_goal85b_component_${factor}_${scenario}_${root}_20261110.json"
      out="$OUT_ROOT/$factor/$scenario/$root/zero-residual/s240"
      log="$OUT_ROOT/logs/${factor}_${scenario}_${root}.log"
      if [[ -f "$out/summary.json" ]]; then
        echo "SKIP $factor $scenario $root"
        continue
      fi
      mkdir -p "$out"
      echo "START $factor $scenario $root"
      (
        export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        "$PYTHON" scripts/evaluate_pc_command_residual_offpolicy.py \
          --config "$TRAIN_CONFIG" \
          --environment-config "$config" \
          --checkpoint "$CKPT" \
          --output "$out" \
          --seed-start "$SEED_START" --seed-count "$SEED_COUNT" \
          --device cpu --dtype float64 \
          --action-mode zero-residual --success-frame eta
        echo $? > "$out/exitcode"
      ) >"$log" 2>&1 &
      pids+=("$!")
      names+=("${factor}_${scenario}_${root}")
    done
  done
done

fail=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "FAIL ${names[$index]}"
    fail=1
  else
    echo "DONE ${names[$index]}"
  fi
done
[[ "$fail" -eq 0 ]] || exit 1
echo "component diagnostic complete: $OUT_ROOT"
