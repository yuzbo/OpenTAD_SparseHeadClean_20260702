#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702
LOG_DIR="$ROOT/logs/gpu1_bridge_absrange_expanded_waiter"
RUN_TAG="${RUN_TAG:-gpu1_bridge_absrange_expanded_waiter_$(date +%Y%m%d_%H%M%S)}"
WAITER_LOG="$LOG_DIR/${RUN_TAG}.log"

SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-1118197}"
OLD_STEP_ID="${OLD_STEP_ID:-1118197.621}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"

TARGET_WORK_DIR="$ROOT/exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4/gpu1_id1"
TARGET_LOG_DIR="$ROOT/logs/gpu1_bridge_absrange_expanded_long"
TARGET_LAUNCHER="$ROOT/remote_runs/launch_gpu1_bridge_absrange_expanded_long_20260706.sh"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$WAITER_LOG") 2>&1

log_msg() { echo "$(date '+%F %T') $*"; }

target_already_started() {
  [[ -f "$TARGET_WORK_DIR/log.json" ]] && return 0
  compgen -G "$TARGET_LOG_DIR/driver_*.out" >/dev/null && return 0
  return 1
}

old_step_active() {
  squeue --steps -j "$SLURM_JOB_ID_TARGET" --noheader 2>/dev/null \
    | awk '{print $1}' \
    | grep -Fxq "$OLD_STEP_ID"
}

snapshot() {
  log_msg "squeue snapshot"
  squeue --steps -j "$SLURM_JOB_ID_TARGET" || true
  log_msg "gpu snapshot"
  ssh g0030 /usr/bin/nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  ssh g0030 /usr/bin/nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
}

log_msg "waiter start old_step=$OLD_STEP_ID target=$TARGET_WORK_DIR"
snapshot

start_epoch=$(date +%s)
while true; do
  if target_already_started; then
    log_msg "target already has output/logs; not launching again"
    exit 0
  fi

  if old_step_active; then
    now_epoch=$(date +%s)
    elapsed=$((now_epoch - start_epoch))
    if (( elapsed >= MAX_WAIT_SECONDS )); then
      log_msg "timeout waiting for old step after ${elapsed}s; not launching"
      exit 1
    fi
    log_msg "old step $OLD_STEP_ID still active; sleeping ${POLL_SECONDS}s elapsed=${elapsed}s"
    sleep "$POLL_SECONDS"
    continue
  fi

  log_msg "old step $OLD_STEP_ID is gone; launching absrange_expanded through existing launcher"
  snapshot
  bash "$TARGET_LAUNCHER"
  sleep 15
  snapshot
  log_msg "waiter exit after launch request"
  exit 0
done
