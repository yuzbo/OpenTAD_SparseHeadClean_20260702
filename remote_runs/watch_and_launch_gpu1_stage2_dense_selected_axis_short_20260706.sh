#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702}"
LOG_DIR="$ROOT/logs/gpu1_stage2_dense_selected_axis_short_waiter"
RUN_TAG="${RUN_TAG:-gpu1_stage2_dense_selected_axis_short_waiter_$(date +%Y%m%d_%H%M%S)}"
WAITER_LOG="$LOG_DIR/${RUN_TAG}.log"

NODE="${NODE:-g0030}"
GPU_INDEX="${GPU_INDEX:-1}"
GPU_MEM_FREE_MAX_MIB="${GPU_MEM_FREE_MAX_MIB:-100}"
SSH_TIMEOUT_SECONDS="${SSH_TIMEOUT_SECONDS:-60}"
ALLOWED_EXISTING_STEP_REGEX="${ALLOWED_EXISTING_STEP_REGEX:-^1118197[.](660|batch|extern)$}"
RUN_SNAPSHOT="${RUN_SNAPSHOT:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"
TARGET_LAUNCHER="$ROOT/remote_runs/launch_gpu1_stage2_dense_selected_axis_short_20260706.sh"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$WAITER_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') [stage2-short-waiter] $*"; }
remote_gpu() { env -u LD_LIBRARY_PATH timeout "$SSH_TIMEOUT_SECONDS" /usr/bin/ssh "$NODE" "$@"; }

gpu1_uuid_and_mem() {
  local output status
  set +e
  output="$(remote_gpu /usr/bin/nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null)"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    return 0
  fi
  awk -F, -v idx="$GPU_INDEX" '
      {
        gsub(/^ +| +$/, "", $1);
        gsub(/^ +| +$/, "", $2);
        gsub(/^ +| +$/, "", $3);
        if ($1 == idx) {
          print $2, $3;
        }
      }' <<<"$output"
}

gpu1_compute_count() {
  local uuid="$1"
  local output status
  set +e
  output="$(remote_gpu /usr/bin/nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo unknown
    return 0
  fi
  awk -F, -v uuid="$uuid" '
      {
        gsub(/^ +| +$/, "", $1);
        if ($1 == uuid) {
          count++;
        }
      }
      END { print count + 0; }' <<<"$output"
}

other_active_steps_count() {
  local output status
  set +e
  output="$(squeue --steps -j 1118197 --noheader 2>/dev/null)"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo 999
    return 0
  fi
  awk -v allowed="$ALLOWED_EXISTING_STEP_REGEX" '
      NF > 0 && $1 !~ allowed {
        count++;
      }
      END { print count + 0; }' <<<"$output"
}

short_validation_already_active() {
  pgrep -f "run_gpu1_stage2_dense_selected_axis_short_20260706.sh" >/dev/null 2>&1 && return 0
  squeue --steps -j 1118197 --noheader 2>/dev/null | grep -q "stage2_dense" && return 0
  return 1
}

gpu1_free() {
  local info uuid mem apps other_steps
  info="$(gpu1_uuid_and_mem)"
  uuid="$(awk '{print $1}' <<<"$info")"
  mem="$(awk '{print $2}' <<<"$info")"
  if [[ -z "$uuid" || -z "$mem" ]]; then
    log_msg "gpu${GPU_INDEX}_state=unknown info='$info'"
    return 1
  fi
  apps="$(gpu1_compute_count "$uuid")"
  other_steps="$(other_active_steps_count)"
  log_msg "gpu${GPU_INDEX}_uuid=$uuid mem_mib=$mem compute_apps=$apps threshold_mib=$GPU_MEM_FREE_MAX_MIB other_steps=$other_steps allowed_steps_regex=$ALLOWED_EXISTING_STEP_REGEX"
  [[ "$mem" -le "$GPU_MEM_FREE_MAX_MIB" && "$other_steps" -eq 0 ]]
}

snapshot() {
  log_msg "squeue snapshot"
  squeue --steps -j 1118197 || true
  log_msg "gpu snapshot"
  remote_gpu /usr/bin/nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  remote_gpu /usr/bin/nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
}

log_msg "waiter start node=$NODE gpu=$GPU_INDEX target=$TARGET_LAUNCHER"
if [[ "$RUN_SNAPSHOT" == "1" ]]; then
  snapshot
fi

start_epoch="$(date +%s)"
while true; do
  if short_validation_already_active; then
    log_msg "stage2 short validation already active; exiting"
    exit 0
  fi

  if gpu1_free; then
    log_msg "gpu${GPU_INDEX} is free; launching Stage-2 dense selected-axis short validation"
    bash "$TARGET_LAUNCHER"
    sleep 15
    if [[ "$RUN_SNAPSHOT" == "1" ]]; then
      snapshot
    fi
    log_msg "waiter exit after launch request"
    exit 0
  fi

  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))
  if (( elapsed >= MAX_WAIT_SECONDS )); then
    log_msg "timeout waiting for gpu${GPU_INDEX} after ${elapsed}s; not launching"
    exit 1
  fi
  log_msg "gpu${GPU_INDEX} busy; sleeping ${POLL_SECONDS}s elapsed=${elapsed}s"
  sleep "$POLL_SECONDS"
done
