#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702
LOG_DIR="$ROOT/logs/gpu1_stage2_dense_selected_axis_short"
RUN_TAG="gpu1_stage2_dense_selected_axis_short_20260706_$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="$LOG_DIR/driver_${RUN_TAG}.out"

mkdir -p "$LOG_DIR"
export RUN_TAG
export RUN_TRAIN=1
export PRECHECK_ONLY=0
export ALLOW_GPU1_SHORT_VALIDATION=1
export EXP_ID="${EXP_ID:-1}"
export PORT_BASE="${PORT_BASE:-32340}"
export CUDA_VISIBLE_DEVICES=1

nohup srun --jobid=1118197 --overlap -w g0030 -N1 -n1 \
  bash "$ROOT/remote_runs/run_gpu1_stage2_dense_selected_axis_short_20260706.sh" \
  > "$DRIVER_LOG" 2>&1 &

echo "RUN_TAG=$RUN_TAG"
echo "DRIVER_LOG=$DRIVER_LOG"
echo "LOCAL_LAUNCH_PID=$!"
