#!/usr/bin/env bash
# Launch the frozen Goal182 first wave: 6 MOD4 arms x 3 seeds.
#
# The script is safe to restart. Completed runs are skipped, incomplete output
# directories are treated as conflicts, and a queue-level lock prevents two
# schedulers from assigning the same GPUs.
set -uo pipefail

W="${WORKTREE:-/data/yhx/underwater-robotics/worktrees/auv-local/goal182/20261117-goal182-mod4-fce2e25}"
G="${GOVERNANCE_ROOT:-/data/yhx/underwater-robotics/worktrees/auv-coordination/goal182/20261117-infra-4348af4}"
PY="${PYTHON:-/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python}"
R="${RUN_ROOT:-/data/yhx/underwater-robotics/runs/auv-local/goal182/20261117-goal182-mod4-8485ac2}"
CORE="${AUV_CORE_SOURCE:-/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src}"

MIN_FREE_MIB="${MIN_FREE_MIB:-23000}"
MAX_JOBS="${MAX_JOBS:-2}"
POLL_S="${POLL_S:-60}"
ITERATIONS="${ITERATIONS:-2600}"
GPU_WHITELIST="${GPU_WHITELIST:-0 1 6 7}"

STATUS="$R/queue_status.tsv"
SUMMARY="$R/queue_summary.txt"
QUEUE_LOG="$R/queue.log"
LOCK="$R/queue.lock"

mkdir -p "$R/formal" "$R/logs"
touch "$STATUS"

exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'Goal182 first-wave queue already holds %s\n' "$LOCK" >&2
  exit 0
fi
exec >>"$QUEUE_LOG" 2>&1

export PYTHONPATH="$W/src:$CORE${PYTHONPATH:+:$PYTHONPATH}"
export AUV_GOVERNANCE_ROOT="$G"

JOBS=(
  "pe2e_finetune_seed86201|configs/lowlevel/goal182/train_formal_mod4_pe2e_finetune_5d_m_seed86201.json"
  "pe2e_residual_seed86201|configs/lowlevel/goal182/train_formal_mod4_pe2e_residual_5d_m_seed86201.json"
  "alosb_finetune_seed86201|configs/lowlevel/goal182/train_formal_mod4_alosb_finetune_5d_m_seed86201.json"
  "alosb_residual_seed86201|configs/lowlevel/goal182/train_formal_mod4_alosb_residual_5d_m_seed86201.json"
  "alosb_trres_cmd_seed86201|configs/lowlevel/goal182/train_formal_mod4_alosb_trres_cmd_5d_m_seed86201.json"
  "alosb_trres_act_seed86201|configs/lowlevel/goal182/train_formal_mod4_alosb_trres_act_5d_m_seed86201.json"
  "pe2e_finetune_seed86202|configs/lowlevel/goal182/train_formal_mod4_pe2e_finetune_5d_m_seed86202.json"
  "pe2e_residual_seed86202|configs/lowlevel/goal182/train_formal_mod4_pe2e_residual_5d_m_seed86202.json"
  "alosb_finetune_seed86202|configs/lowlevel/goal182/train_formal_mod4_alosb_finetune_5d_m_seed86202.json"
  "alosb_residual_seed86202|configs/lowlevel/goal182/train_formal_mod4_alosb_residual_5d_m_seed86202.json"
  "alosb_trres_cmd_seed86202|configs/lowlevel/goal182/train_formal_mod4_alosb_trres_cmd_5d_m_seed86202.json"
  "alosb_trres_act_seed86202|configs/lowlevel/goal182/train_formal_mod4_alosb_trres_act_5d_m_seed86202.json"
  "pe2e_finetune_seed86203|configs/lowlevel/goal182/train_formal_mod4_pe2e_finetune_5d_m_seed86203.json"
  "pe2e_residual_seed86203|configs/lowlevel/goal182/train_formal_mod4_pe2e_residual_5d_m_seed86203.json"
  "alosb_finetune_seed86203|configs/lowlevel/goal182/train_formal_mod4_alosb_finetune_5d_m_seed86203.json"
  "alosb_residual_seed86203|configs/lowlevel/goal182/train_formal_mod4_alosb_residual_5d_m_seed86203.json"
  "alosb_trres_cmd_seed86203|configs/lowlevel/goal182/train_formal_mod4_alosb_trres_cmd_5d_m_seed86203.json"
  "alosb_trres_act_seed86203|configs/lowlevel/goal182/train_formal_mod4_alosb_trres_act_5d_m_seed86203.json"
)

declare -A RUN_PID=()
declare -A RUN_GPU=()
declare -A RUN_LOG=()
NEXT=0
FAILED=0

write_summary() {
  local tmp="$SUMMARY.tmp"
  {
    printf '# Goal182 first-wave queue\n\n'
    printf -- '- worktree: `%s`\n' "$W"
    printf -- '- run root: `%s`\n' "$R"
    printf -- '- jobs: `%s`\n' "${#JOBS[@]}"
    printf -- '- max concurrent jobs: `%s`\n' "$MAX_JOBS"
    printf -- '- minimum free GPU memory: `%s MiB`\n' "$MIN_FREE_MIB"
    printf -- '- GPU whitelist: `%s`\n' "$GPU_WHITELIST"
    printf -- '- updated: `%s`\n\n' "$(date -Iseconds)"
    printf '## Status counts\n\n'
    awk -F '\t' 'NF {count[$1]++} END {for (key in count) printf "- %s: %d\n", key, count[key]}' "$STATUS" | sort
    printf '\n## Last status rows\n\n```text\n'
    tail -n 20 "$STATUS" || true
    printf '```\n'
  } >"$tmp"
  mv "$tmp" "$SUMMARY"
}

trap write_summary EXIT

manifest_completed() {
  "$PY" - "$1" <<'PY'
import json
import os
import sys

path = os.path.join(sys.argv[1], "training_manifest.json")
try:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "completed" else 1)
PY
}

gpu_allowed() {
  local candidate=$1 item
  for item in $GPU_WHITELIST; do
    [ "$item" = "$candidate" ] && return 0
  done
  return 1
}

gpu_assigned() {
  local candidate=$1 pid
  for pid in "${!RUN_GPU[@]}"; do
    [ "${RUN_GPU[$pid]}" = "$candidate" ] && return 0
  done
  return 1
}

free_gpu() {
  local idx free
  while IFS=, read -r idx free; do
    idx=${idx//[[:space:]]/}
    free=${free//[[:space:]]/}
    gpu_allowed "$idx" || continue
    gpu_assigned "$idx" && continue
    if [ "$free" -ge "$MIN_FREE_MIB" ]; then
      printf '%s\n' "$idx"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

launch_job() {
  local job_index=$1 gpu=$2 spec name cfg out log pid seed
  spec=${JOBS[$job_index]}
  name=${spec%%|*}
  cfg=${spec#*|}
  out="$R/formal/$name"
  log="$R/logs/$name.log"
  seed=${name##*seed}

  if [ ! -f "$W/$cfg" ]; then
    printf 'MISSING_CONFIG\t%s\t%s\t%s\n' "$name" "$cfg" "$(date -Iseconds)" | tee -a "$STATUS"
    FAILED=1
    return 1
  fi
  if [ -e "$out" ] && ! manifest_completed "$out"; then
    printf 'CONFLICT\t%s\texisting output without completed manifest\n' "$name" | tee -a "$STATUS"
    FAILED=1
    return 1
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PY" "$W/scripts/train_batch_lowlevel_ppo.py" \
      --training-config "$W/$cfg" \
      --iterations "$ITERATIONS" \
      --curriculum-eval-stage-boundaries \
      --curriculum-route-gate-mode shadow \
      --output "$out" \
      --seed "$seed"
  ) >>"$log" 2>&1 &
  pid=$!

  RUN_PID[$pid]=$name
  RUN_GPU[$pid]=$gpu
  RUN_LOG[$pid]=$log
  printf 'LAUNCH\t%s\tgpu=%s\tpid=%s\tseed=%s\t%s\n' \
    "$name" "$gpu" "$pid" "$seed" "$(date -Iseconds)" | tee -a "$STATUS"
}

reap_jobs() {
  local pid rc name gpu log
  for pid in "${!RUN_PID[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"
    rc=$?
    name=${RUN_PID[$pid]}
    gpu=${RUN_GPU[$pid]}
    log=${RUN_LOG[$pid]}
    if [ "$rc" -eq 0 ] && manifest_completed "$R/formal/$name"; then
      printf 'DONE\t%s\tgpu=%s\trc=0\t%s\n' "$name" "$gpu" "$(date -Iseconds)" | tee -a "$STATUS"
    else
      printf 'FAIL\t%s\tgpu=%s\trc=%s\t%s\tlog=%s\n' \
        "$name" "$gpu" "$rc" "$(date -Iseconds)" "$log" | tee -a "$STATUS"
      FAILED=1
    fi
    unset 'RUN_PID[$pid]' 'RUN_GPU[$pid]' 'RUN_LOG[$pid]'
  done
}

for spec in "${JOBS[@]}"; do
  cfg=${spec#*|}
  if [ ! -f "$W/$cfg" ]; then
    printf 'MISSING_CONFIG\t%s\t%s\n' "${spec%%|*}" "$cfg" | tee -a "$STATUS"
    exit 2
  fi
done

printf 'QUEUE_START\t%s\tjobs=%s\tmax_jobs=%s\tmin_free_mib=%s\tgpu_whitelist=%s\n' \
  "$(date -Iseconds)" "${#JOBS[@]}" "$MAX_JOBS" "$MIN_FREE_MIB" "$GPU_WHITELIST" | tee -a "$STATUS"
write_summary

while true; do
  reap_jobs

  while [ "$NEXT" -lt "${#JOBS[@]}" ]; do
    name=${JOBS[$NEXT]%%|*}
    if manifest_completed "$R/formal/$name"; then
      printf 'SKIP_COMPLETED\t%s\t%s\n' "$name" "$(date -Iseconds)" | tee -a "$STATUS"
      NEXT=$((NEXT + 1))
    else
      break
    fi
  done

  if [ "$FAILED" -eq 0 ]; then
    while [ "${#RUN_PID[@]}" -lt "$MAX_JOBS" ] && [ "$NEXT" -lt "${#JOBS[@]}" ]; do
      gpu=$(free_gpu) || break
      launch_job "$NEXT" "$gpu" || break
      NEXT=$((NEXT + 1))
      write_summary
    done
  fi

  if [ "${#RUN_PID[@]}" -eq 0 ] && { [ "$NEXT" -ge "${#JOBS[@]}" ] || [ "$FAILED" -ne 0 ]; }; then
    break
  fi
  write_summary
  sleep "$POLL_S"
done

if [ "$FAILED" -eq 0 ]; then
  printf 'QUEUE_COMPLETE\t%s\tjobs=%s\n' "$(date -Iseconds)" "${#JOBS[@]}" | tee -a "$STATUS"
  exit 0
fi
printf 'QUEUE_FAILED\t%s\tnext_job=%s\n' "$(date -Iseconds)" "$NEXT" | tee -a "$STATUS"
exit 1
