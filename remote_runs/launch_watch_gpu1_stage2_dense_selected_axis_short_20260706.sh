#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702
LOG_DIR="$ROOT/logs/gpu1_stage2_dense_selected_axis_short_waiter"
RUN_TAG="gpu1_stage2_dense_selected_axis_short_waiter_20260706_$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="$LOG_DIR/driver_${RUN_TAG}.out"
WAITER_SCRIPT="$ROOT/remote_runs/watch_and_launch_gpu1_stage2_dense_selected_axis_short_20260706.sh"

mkdir -p "$LOG_DIR"

if pgrep -f "$WAITER_SCRIPT" >/dev/null 2>&1; then
  echo "waiter already running for $WAITER_SCRIPT"
  exit 0
fi

export RUN_TAG

nohup bash "$WAITER_SCRIPT" \
  > "$DRIVER_LOG" 2>&1 &

echo "RUN_TAG=$RUN_TAG"
echo "DRIVER_LOG=$DRIVER_LOG"
echo "LOCAL_LAUNCH_PID=$!"
