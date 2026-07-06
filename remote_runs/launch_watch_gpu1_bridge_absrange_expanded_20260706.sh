#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702
LOG_DIR="$ROOT/logs/gpu1_bridge_absrange_expanded_waiter"
RUN_TAG="gpu1_bridge_absrange_expanded_waiter_20260706_$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="$LOG_DIR/driver_${RUN_TAG}.out"

mkdir -p "$LOG_DIR"
export RUN_TAG

nohup bash "$ROOT/remote_runs/watch_and_launch_gpu1_bridge_absrange_expanded_20260706.sh" \
  > "$DRIVER_LOG" 2>&1 &

echo "RUN_TAG=$RUN_TAG"
echo "DRIVER_LOG=$DRIVER_LOG"
echo "LOCAL_LAUNCH_PID=$!"
