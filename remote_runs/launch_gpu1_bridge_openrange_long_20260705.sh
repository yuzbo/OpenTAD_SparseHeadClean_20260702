#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702
LOG_DIR="$ROOT/logs/gpu1_bridge_openrange_long"
RUN_TAG="gpu1_bridge_openrange_long_20260705_$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="$LOG_DIR/driver_${RUN_TAG}.out"

mkdir -p "$LOG_DIR"
export RUN_TAG
export EXP_ID=2
export PORT=32324
export CUDA_VISIBLE_DEVICES=1

nohup srun --jobid=1118197 --overlap -w g0030 -N1 -n1 \
  bash "$ROOT/remote_runs/run_gpu1_bridge_openrange_long_20260705.sh" \
  > "$DRIVER_LOG" 2>&1 &

echo "RUN_TAG=$RUN_TAG"
echo "DRIVER_LOG=$DRIVER_LOG"
echo "LOCAL_LAUNCH_PID=$!"
